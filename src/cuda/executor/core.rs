//! Core CudaExecutor implementation: constructor, availability, profiling API, graph tracking
//!
//! This module contains the core functionality of the CudaExecutor including:
//! - Constructor and device initialization
//! - Availability checking
//! - BrickProfiler API for per-brick timing
//! - Execution graph tracking for ASCII tree visualization
//! - Tile-level profiling for cache hierarchy analysis
//! - Context, device, and pool management methods

#![allow(clippy::wildcard_imports)] // Internal module organization uses super::*

use super::*;

impl CudaExecutor {
    /// Create a new CUDA executor for the specified device
    ///
    /// # Arguments
    ///
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    pub fn new(device_ordinal: i32) -> Result<Self, GpuError> {
        // Ensure process-level sentinel keeps the primary context alive.
        // This prevents cuDevicePrimaryCtxRelease from destroying the context
        // when individual executors drop (sentinel holds refcount ≥ 1).
        ensure_sentinel(device_ordinal)?;

        // Check out a pooled context to avoid cuDevicePrimaryCtxRetain churn.
        // First call creates a fresh context; subsequent calls reuse the pooled one.
        let context = checkout_context(device_ordinal)?;
        let (compute_stream, transfer_stream, stream) = checkout_streams(&context)?;

        Ok(Self {
            // Initialize in struct declaration order (for clarity)
            kernels: CudaKernels::new(),
            memory_pool: GpuMemoryPool::new(),
            staging_pool: StagingBufferPool::new(), // PARITY-042: pinned memory pool
            modules: std::mem::ManuallyDrop::new(HashMap::new()),
            weight_cache: HashMap::new(),
            quantized_weight_cache: HashMap::new(), // PAR-005: quantized weight cache
            quantized_weight_types: HashMap::new(), // PAR-058: weight quant types
            rmsnorm_cache: HashMap::new(),          // PAR-023: RMSNorm gamma cache
            bias_cache: HashMap::new(),             // BIAS-FIX: QKV bias cache
            // PAR-043: Pre-indexed layer weights for O(1) access
            indexed_layer_weights: Vec::new(),
            output_norm_ptr: 0,
            output_norm_len: 0,
            lm_head_ptr: 0,
            lm_head_len: 0,
            lm_head_qtype: WeightQuantType::Q4K, // Default, updated on weight load
            lm_head_bias_ptr: 0,
            lm_head_bias_len: 0,
            logits_buffer: None,
            logits_buffer_size: 0,
            workspace: TransformerWorkspace::default(), // PAR-044: lazy init on first forward
            gemv_input_buffer: None,                    // PAR-007: lazy init on first GEMV
            gemv_output_buffer: None,
            gemv_input_size: 0,
            gemv_output_size: 0,
            kv_cache_gpu: HashMap::new(), // PAR-018 + PAR-021: GPU-resident KV cache
            kv_cache_lengths: HashMap::new(),
            kv_cache_max_len: 0,
            kv_num_heads: 0,
            kv_num_kv_heads: 0, // PAR-021 GQA
            kv_head_dim: 0,
            rope_theta: 10000.0, // PAR-060: default RoPE theta
            rope_type: 0,        // CORRECTNESS-011: default NORM style
            compute_stream: PoolableStream::new(compute_stream),
            transfer_stream: PoolableStream::new(transfer_stream),
            stream: PoolableStream::new(stream),
            // PAR-054: CUDA Graph Capture (lazy init on first decode)
            decode_graph: None,
            position_buf: None,
            seq_len_buf: None,
            // PAR-119: Batched KV caches (lazy init in init_batched_kv_cache)
            batched_kv_k_caches: HashMap::new(),
            batched_kv_v_caches: HashMap::new(),
            batched_kv_lengths: Vec::new(),
            batched_k_ptrs: None,
            batched_v_ptrs: None,
            batched_seq_lens_gpu: None,
            batched_kv_stride: 0,
            batched_kv_allocated_batch: 0,
            // PAR-121: Batched graph fields
            batched_decode_graphs: HashMap::new(),
            batched_graph_input_buf: None,
            batched_graph_positions_buf: None,
            batched_graph_seq_lens_buf: None,
            batched_graph_batch_size: 0,
            graph_input_buf: None,
            decode_token_count: 0,
            // PAR-068: Pre-allocated argmax buffers (lazy init on first use)
            argmax_block_vals: None,
            argmax_block_idxs: None,
            argmax_result: None,
            argmax_num_blocks: 0,
            // PAR-118: Flash Decoding (disabled by default, enable via init_flash_decoding)
            flash_decode_partials: None,
            flash_decode_max_seq_len: 0,
            flash_decode_enabled: false,
            // QWEN-007: Q8 KV cache (disabled by default, enable via init_kv_cache_q8_gpu)
            kv_cache_q8_enabled: false,
            kv_cache_q8_k: HashMap::new(),
            kv_cache_q8_v: HashMap::new(),
            kv_cache_q8_k_scales: HashMap::new(),
            kv_cache_q8_v_scales: HashMap::new(),
            // QWEN-010: Auto-tune tile size based on GPU
            // RTX 4090 (sm_89) has 72MB L2 cache - use 64x64 tiles
            // Default: 32x32 tiles for other GPUs
            optimal_tile_size: Self::detect_optimal_tile_size(&context),
            // PAR-073: BrickProfiler (disabled by default for zero overhead)
            // Enable with executor.enable_profiling() for per-brick timing
            profiler: trueno::BrickProfiler::new(),
            context: std::mem::ManuallyDrop::new(context), // Last field — ManuallyDrop skips cuDevicePrimaryCtxRelease - dropped last
        })
    }

    /// Check if CUDA is available on this system
    #[must_use]
    pub fn is_available() -> bool {
        cuda_available()
    }

    /// QWEN-010: Detect optimal tile size based on GPU architecture
    ///
    /// RTX 4090 (Ada Lovelace, sm_89) has 72MB L2 cache vs A100's 40MB.
    /// Larger tiles (64x64) improve L2 cache utilization on RTX 4090.
    fn detect_optimal_tile_size(context: &CudaContext) -> u32 {
        // Get device name for GPU detection
        let device_name = context.device_name().unwrap_or_default();

        // RTX 4090, RTX 4080, RTX 4070 (Ada Lovelace, sm_89) benefit from 64x64 tiles
        // These GPUs have 72MB, 64MB, 48MB L2 cache respectively
        if device_name.contains("4090")
            || device_name.contains("4080")
            || device_name.contains("4070")
        {
            64
        } else {
            // Default: 32x32 tiles for other GPUs (A100, V100, older consumer cards)
            32
        }
    }

    /// Get the optimal tile size for this GPU
    ///
    /// Returns 64 for RTX 40-series (Ada Lovelace), 32 for other GPUs.
    #[must_use]
    pub fn optimal_tile_size(&self) -> u32 {
        self.optimal_tile_size
    }

    /// Compile PTX into a CUDA module, with process-level blocklisting.
    ///
    /// If the same PTX previously failed to compile (poisoning the CUDA
    /// context), this method returns an error immediately without calling
    /// `cuModuleLoadData`, preventing repeated context poisoning.
    pub(crate) fn compile_ptx(&self, ptx: &str) -> Result<CudaModule, GpuError> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        ptx.hash(&mut hasher);
        let hash = hasher.finish();

        {
            let bl = BROKEN_PTX.lock().unwrap();
            if bl.contains(&hash) {
                return Err(GpuError::ModuleLoad(
                    "kernel blocklisted (previous compilation failure)".to_string(),
                ));
            }
        }
        match CudaModule::from_ptx(&self.context, ptx) {
            Ok(m) => Ok(m),
            Err(e) => {
                if verbose() {
                    eprintln!(
                        "[CUDA-POOL] PTX compilation failed (hash={hash:016x}), blocklisting"
                    );
                }
                BROKEN_PTX.lock().unwrap().insert(hash);
                Err(e)
            },
        }
    }
}

/// Custom Drop: synchronize, then return streams and context to pools.
///
/// Erlang-style fail-fast: synchronize to detect any async kernel crashes
/// BEFORE returning resources to the pool.  A poisoned stream/context in the
/// pool would cascade failures to every subsequent test.
///
/// After this runs, Rust auto-drops remaining fields in declaration order:
/// - GPU buffers (cuMemFree) — context thread-local pointer is still set
/// - PoolableStream wrappers — inner is None (already extracted), no-op
/// - modules (ManuallyDrop) — intentionally leaked, no cuModuleUnload
/// - context (ManuallyDrop<CudaContext>) — returned to pool or leaked
impl Drop for CudaExecutor {
    fn drop(&mut self) {
        // Fail-fast: synchronize to detect async kernel crashes immediately.
        // Without this, a crashing kernel silently poisons the pooled
        // streams/context and cascades failures to ALL subsequent tests.
        let ctx_healthy = self.context.synchronize().is_ok();

        // Extract streams from PoolableStream wrappers.
        let compute = self.compute_stream.take();
        let transfer = self.transfer_stream.take();
        let legacy = self.stream.take();

        if ctx_healthy {
            // Context is healthy — return resources to pools for reuse.
            if let (Some(s1), Some(s2), Some(s3)) = (compute, transfer, legacy) {
                checkin_streams(s1, s2, s3);
            }
            let ctx = unsafe { std::mem::ManuallyDrop::take(&mut self.context) };
            checkin_context(ctx);
        } else {
            // Context is POISONED — do NOT return resources to pools.
            // Let streams drop normally (cuStreamDestroy — will fail, that's OK).
            // Let context drop normally (cuDevicePrimaryCtxRelease).
            // The sentinel still holds one retain, keeping the primary context
            // alive.  The NEXT executor will get a fresh context from the pool
            // (empty → CudaContext::new) which retains the same primary context.
            // If the primary context itself is irrecoverable, the next
            // CudaContext::new will also fail — surfacing the error immediately.
            eprintln!(
                "[CUDA-FAILFAST] Context poisoned during executor lifetime — \
                 streams and context NOT returned to pool. \
                 Next executor will create fresh resources."
            );
            // Take context out of ManuallyDrop so it drops normally (release).
            let _ctx = unsafe { std::mem::ManuallyDrop::take(&mut self.context) };
        }
    }
}

impl CudaExecutor {
    /// Get number of CUDA devices
    ///
    /// Returns 0 if CUDA is not available.
    #[must_use]
    pub fn num_devices() -> usize {
        device_count().unwrap_or(0)
    }

    /// Set the CUDA context as current for the calling thread
    ///
    /// CUDA contexts are thread-local. When using async/multi-threaded code
    /// (like axum/tokio), you must call `make_current()` before any CUDA
    /// operation if the operation might run on a different thread than where
    /// the executor was created.
    ///
    /// # Errors
    ///
    /// Returns error if cuCtxSetCurrent fails (e.g., invalid context).
    pub fn make_current(&self) -> Result<(), GpuError> {
        self.context.make_current()
    }

    // ========================================================================
    // PAR-073: BrickProfiler API for per-brick timing
    // ========================================================================

    /// Enable per-brick profiling for real timing measurements.
    ///
    /// When enabled, each brick operation is timed individually using
    /// `std::time::Instant` with CUDA sync for accurate GPU timing.
    ///
    /// # Performance Impact
    ///
    /// Profiling adds ~1 CUDA sync per brick, which adds overhead.
    /// Use only during development/benchmarking, not production.
    pub fn enable_profiling(&mut self) {
        self.profiler.enable();
    }

    /// Disable per-brick profiling (default state).
    pub fn disable_profiling(&mut self) {
        self.profiler.disable();
    }

    /// Check if profiling is enabled.
    #[must_use]
    pub fn is_profiling_enabled(&self) -> bool {
        self.profiler.is_enabled()
    }

    /// Get the brick profiler for reading statistics.
    #[must_use]
    pub fn profiler(&self) -> &trueno::BrickProfiler {
        &self.profiler
    }

    /// Get mutable access to the brick profiler.
    pub fn profiler_mut(&mut self) -> &mut trueno::BrickProfiler {
        &mut self.profiler
    }

    /// Reset profiler statistics.
    pub fn reset_profiler(&mut self) {
        self.profiler.reset();
    }

    /// Get profiler summary report.
    #[must_use]
    pub fn profiler_summary(&self) -> String {
        self.profiler.summary()
    }

    /// Start timing a brick (internal use, legacy string API).
    ///
    /// When profiling is enabled, syncs the stream and starts a timer.
    /// Returns the timer handle for use with `stop_brick_timer()`.
    #[must_use]
    pub(crate) fn start_brick_timer(&mut self, name: &str) -> Option<trueno::BrickTimer> {
        if !self.profiler.is_enabled() {
            return None;
        }
        // Sync to ensure previous work is complete (immediate mode)
        if self.profiler.sync_mode() == trueno::SyncMode::Immediate {
            let _ = self.stream.synchronize();
        }
        Some(self.profiler.start(name))
    }

    /// Stop timing a brick and record the sample.
    ///
    /// When profiling is enabled, syncs the stream and records the elapsed time.
    pub(crate) fn stop_brick_timer(&mut self, timer: Option<trueno::BrickTimer>, elements: u64) {
        if let Some(t) = timer {
            // Sync to capture real GPU time (immediate mode)
            if self.profiler.sync_mode() == trueno::SyncMode::Immediate {
                let _ = self.stream.synchronize();
            }
            self.profiler.stop(t, elements);
        }
    }

    // ========================================================================
    // PAR-200: BrickId-based profiling (O(1) hot path)
    // ========================================================================

    /// Start timing a brick using BrickId (PAR-200 fast path).
    ///
    /// This is the preferred API for known brick types.
    #[inline]
    #[must_use]
    pub(crate) fn start_brick_id(
        &mut self,
        brick_id: trueno::BrickId,
    ) -> Option<trueno::BrickIdTimer> {
        if !self.profiler.is_enabled() {
            return None;
        }
        if self.profiler.sync_mode() == trueno::SyncMode::Immediate {
            let _ = self.stream.synchronize();
        }
        Some(self.profiler.start_brick(brick_id))
    }

    /// Stop timing a brick using BrickId (PAR-200 fast path).
    #[inline]
    pub(crate) fn stop_brick_id(&mut self, timer: Option<trueno::BrickIdTimer>, elements: u64) {
        if let Some(t) = timer {
            if self.profiler.sync_mode() == trueno::SyncMode::Immediate {
                let _ = self.stream.synchronize();
            }
            self.profiler.stop_brick(t, elements);
        }
    }

    /// Record a deferred measurement (PAR-200, ~5% overhead).
    ///
    /// Use this for production profiling. Call `finalize_profiling()` after
    /// GPU sync to apply all pending measurements.
    #[inline]
    pub(crate) fn record_brick_deferred(
        &mut self,
        brick_id: trueno::BrickId,
        start_ns: u64,
        elements: u64,
    ) {
        self.profiler.record_deferred(brick_id, start_ns, elements);
    }

    /// Finalize all pending profiling measurements.
    ///
    /// Must be called after `stream.synchronize()` when using deferred mode.
    #[inline]
    pub(crate) fn finalize_profiling(&mut self) {
        if self.profiler.has_pending() {
            let end_ns = self.profiler.elapsed_ns();
            self.profiler.finalize(end_ns);
        }
    }

    /// Reset profiler epoch for deferred timing.
    #[inline]
    pub(crate) fn reset_profiler_epoch(&mut self) {
        self.profiler.reset_epoch();
    }

    /// Get profiler category breakdown (PAR-200).
    #[must_use]
    pub fn profiler_category_stats(&self) -> [trueno::CategoryStats; trueno::BrickCategory::COUNT] {
        self.profiler.category_stats()
    }

    /// Print profiler category breakdown (PAR-200).
    pub fn print_profiler_categories(&self) {
        self.profiler.print_category_stats();
    }

    /// Set profiler sync mode (PAR-200).
    ///
    /// # Modes
    /// - `Immediate`: Sync after each kernel (accurate, ~200% overhead)
    /// - `PerLayer`: Sync once per layer (~20% overhead)
    /// - `Deferred`: Sync once per forward pass (~5% overhead)
    /// - `None`: No profiling overhead
    pub fn set_profiler_sync_mode(&mut self, mode: trueno::SyncMode) {
        self.profiler.set_sync_mode(mode);
    }

    /// Get current profiler sync mode.
    #[must_use]
    pub fn profiler_sync_mode(&self) -> trueno::SyncMode {
        self.profiler.sync_mode()
    }

    // ========================================================================
    // PAR-201: Execution Graph Tracking (ASCII tree visualization)
    // ========================================================================

    /// Enable execution graph tracking for brick→kernel→PTX relationships.
    ///
    /// When enabled, each brick operation is recorded in a hierarchical graph
    /// that can be visualized as an ASCII tree for CI/CD and debugging.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// cuda_model.enable_graph_tracking();
    /// cuda_model.forward_gpu_resident(token, &mut cache, pos)?;
    /// println!("{}", cuda_model.execution_graph_ascii());
    /// // Output:
    /// // Layer 0
    /// // ├── RmsNorm  50.0µs (4096 elem)
    /// // │   └── rmsnorm_kernel  <<<16,256,1>>> smem=1024B
    /// // └── QkvProjection  200.0µs (4096 elem)
    /// ```
    pub fn enable_graph_tracking(&mut self) {
        self.profiler.enable_graph();
    }

    /// Disable execution graph tracking.
    pub fn disable_graph_tracking(&mut self) {
        self.profiler.disable_graph();
    }

    /// Check if execution graph tracking is enabled.
    #[must_use]
    pub fn is_graph_tracking_enabled(&self) -> bool {
        self.profiler.is_graph_enabled()
    }

    /// Get the execution graph (immutable).
    #[must_use]
    pub fn execution_graph(&self) -> &trueno::ExecutionGraph {
        self.profiler.execution_graph()
    }

    /// Get the execution graph as ASCII tree (headless mode for CI/CD).
    ///
    /// PAR-201: Zero-dependency tree visualization for snapshot tests, logging,
    /// and CI pipelines.
    #[must_use]
    pub fn execution_graph_ascii(&self) -> String {
        self.profiler.execution_graph().to_ascii_tree()
    }

    // ========================================================================
    // TILING-SPEC-001: Tile-Level Profiling (Phase 15)
    // ========================================================================

    /// Enable tile-level profiling for hierarchical cache analysis.
    ///
    /// When enabled, `start_tile_timer()`/`stop_tile_timer()` record per-tile
    /// statistics (GFLOP/s, arithmetic intensity, throughput) at Macro/Midi/Micro
    /// levels for identifying memory-bound vs compute-bound bottlenecks.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// cuda_model.enable_tile_profiling();
    /// cuda_model.forward_tiled_profiled(...)?;
    /// println!("{}", cuda_model.tile_summary());
    /// // Output:
    /// // === Tile Profiling Summary (TILING-SPEC-001) ===
    /// // Level       Samples   Avg µs    GFLOP/s   AI      Elements
    /// // macro            28    5280.0      0.40  0.50    1048576
    /// ```
    pub fn enable_tile_profiling(&mut self) {
        self.profiler.enable_tile_profiling();
    }

    /// Disable tile-level profiling.
    pub fn disable_tile_profiling(&mut self) {
        self.profiler.disable_tile_profiling();
    }

    /// Check if tile profiling is enabled.
    #[must_use]
    pub fn is_tile_profiling_enabled(&self) -> bool {
        self.profiler.is_tile_profiling_enabled()
    }

    /// Start timing a tile operation.
    ///
    /// # Arguments
    /// * `level` - Tile hierarchy level (Macro/Midi/Micro)
    /// * `layer_idx` - Layer index (for Macro tiles) or head index (for Midi tiles)
    /// * `op_idx` - Operation index within the layer (0=QKV, 1=Attn, 2=FFN)
    ///
    /// # Returns
    /// Timer handle to pass to `stop_tile_timer()`.
    #[must_use]
    pub(crate) fn start_tile_timer(
        &mut self,
        level: trueno::TileLevel,
        layer_idx: u32,
        op_idx: u32,
    ) -> Option<trueno::TileTimer> {
        if !self.profiler.is_tile_profiling_enabled() {
            return None;
        }
        // Sync to ensure previous work is complete
        let _ = self.stream.synchronize();
        Some(self.profiler.start_tile(level, layer_idx, op_idx))
    }

    /// Stop timing a tile operation and record statistics.
    ///
    /// # Arguments
    /// * `timer` - Timer handle from `start_tile_timer()`
    /// * `elements` - Number of elements processed (for throughput calculation)
    /// * `flops` - Number of floating-point operations (for GFLOP/s calculation)
    pub(crate) fn stop_tile_timer(
        &mut self,
        timer: Option<trueno::TileTimer>,
        elements: u64,
        flops: u64,
    ) {
        if let Some(t) = timer {
            // Sync to capture real GPU time
            let _ = self.stream.synchronize();
            self.profiler.stop_tile(t, elements, flops);
        }
    }

    /// Get tile statistics for a given level.
    #[must_use]
    pub fn tile_stats(&self, level: trueno::TileLevel) -> &trueno::TileStats {
        self.profiler.tile_stats(level)
    }

    /// Get tile profiling summary report.
    #[must_use]
    pub fn tile_summary(&self) -> String {
        self.profiler.tile_summary()
    }

    /// Get tile statistics as JSON (PMAT integration).
    #[must_use]
    pub fn tile_stats_json(&self) -> String {
        self.profiler.tile_stats_to_json()
    }

    /// Reset tile statistics.
    pub fn reset_tile_stats(&mut self) {
        self.profiler.reset_tile_stats();
    }

    /// Clear the execution graph.
    pub fn clear_execution_graph(&mut self) {
        self.profiler.execution_graph_mut().clear();
    }

    /// Get device name
    pub fn device_name(&self) -> Result<String, GpuError> {
        self.context.device_name()
    }

    /// Get free and total GPU memory in bytes
    pub fn memory_info(&self) -> Result<(usize, usize), GpuError> {
        self.context.memory_info()
    }

    /// Get reference to CUDA context (CORRECTNESS-002: for testing Q6K kernel directly)
    #[must_use]
    pub fn context(&self) -> &CudaContext {
        &self.context
    }

    /// Get reference to compute stream (WAPR-PERF-014: reuse stream for KV scatter/attention)
    #[must_use]
    pub fn compute_stream(&self) -> &CudaStream {
        &self.compute_stream
    }

    /// Synchronize the execution stream (wait for all pending operations)
    pub fn synchronize(&self) -> Result<(), GpuError> {
        self.stream.synchronize()
    }

    /// Get memory pool statistics (IMP-900d)
    #[must_use]
    pub fn pool_stats(&self) -> PoolStats {
        self.memory_pool.stats()
    }

    /// Get staging buffer pool statistics (PARITY-042)
    #[must_use]
    pub fn staging_pool_stats(&self) -> StagingPoolStats {
        self.staging_pool.stats()
    }

    /// Get a staging buffer for pinned memory transfers (PARITY-042)
    pub fn get_staging_buffer(&mut self, size: usize) -> PinnedHostBuffer<f32> {
        self.staging_pool.get(size)
    }

    /// Return a staging buffer to the pool (PARITY-042)
    pub fn return_staging_buffer(&mut self, buf: PinnedHostBuffer<f32>) {
        self.staging_pool.put(buf);
    }

    /// Clear memory pool buffers (releases GPU memory)
    pub fn clear_pool(&mut self) {
        self.memory_pool.clear();
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // Constructor Tests
    // ========================================================================

    #[test]
    fn test_new_device_0() {
        let result = CudaExecutor::new(0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_new_invalid_device() {
        // Device 999 almost certainly doesn't exist
        let result = CudaExecutor::new(999);
        assert!(result.is_err());
    }

    // ========================================================================
    // Availability Tests
    // ========================================================================

    #[test]
    fn test_is_available() {
        // On a CUDA-enabled system, this should return true
        let available = CudaExecutor::is_available();
        // We can't assert true unconditionally, but we know this test runs with CUDA
        let _ = available; // Just verify it can be queried
    }

    #[test]
    fn test_num_devices() {
        let count = CudaExecutor::num_devices();
        // Should be at least 1 on CUDA system
        assert!(count >= 1);
    }

    // ========================================================================
    // Context and Thread Safety Tests
    // ========================================================================

    #[test]
    fn test_make_current() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(exec.make_current().is_ok());
    }

    #[test]
    fn test_context_accessor() {
        let Some(exec) = create_executor() else {
            return;
        };
        let _ctx = exec.context();
        // Just verify it returns a reference without panicking
    }

    #[test]
    fn test_compute_stream_accessor() {
        let Some(exec) = create_executor() else {
            return;
        };
        let _stream = exec.compute_stream();
    }

    // ========================================================================
    // Profiler Tests
    // ========================================================================

    #[test]
    fn test_enable_disable_profiling() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        assert!(!exec.is_profiling_enabled());
        exec.enable_profiling();
        assert!(exec.is_profiling_enabled());
        exec.disable_profiling();
        assert!(!exec.is_profiling_enabled());
    }

    #[test]
    fn test_profiler_accessor() {
        let Some(exec) = create_executor() else {
            return;
        };
        let _profiler = exec.profiler();
    }

    #[test]
    fn test_profiler_mut_accessor() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let _profiler = exec.profiler_mut();
    }

    #[test]
    fn test_reset_profiler() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.reset_profiler();
    }

    #[test]
    fn test_profiler_summary() {
        let Some(exec) = create_executor() else {
            return;
        };
        let summary = exec.profiler_summary();
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_profiler_sync_mode() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let initial_mode = exec.profiler_sync_mode();
        exec.set_profiler_sync_mode(trueno::SyncMode::Deferred);
        assert_eq!(exec.profiler_sync_mode(), trueno::SyncMode::Deferred);
        exec.set_profiler_sync_mode(initial_mode);
    }

    #[test]
    fn test_profiler_category_stats() {
        let Some(exec) = create_executor() else {
            return;
        };
        let stats = exec.profiler_category_stats();
        assert_eq!(stats.len(), trueno::BrickCategory::COUNT);
    }

    // ========================================================================
    // Execution Graph Tests
    // ========================================================================

    #[test]
    fn test_enable_disable_graph_tracking() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        assert!(!exec.is_graph_tracking_enabled());
        exec.enable_graph_tracking();
        assert!(exec.is_graph_tracking_enabled());
        exec.disable_graph_tracking();
        assert!(!exec.is_graph_tracking_enabled());
    }

    #[test]
    fn test_execution_graph_accessor() {
        let Some(exec) = create_executor() else {
            return;
        };
        let _graph = exec.execution_graph();
    }

    #[test]
    fn test_execution_graph_ascii() {
        let Some(exec) = create_executor() else {
            return;
        };
        let ascii = exec.execution_graph_ascii();
        // Empty graph, but should still return something
        let _ = ascii;
    }

    #[test]
    fn test_clear_execution_graph() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.clear_execution_graph();
    }

    // ========================================================================
    // Tile Profiling Tests
    // ========================================================================

    #[test]
    fn test_enable_disable_tile_profiling() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        assert!(!exec.is_tile_profiling_enabled());
        exec.enable_tile_profiling();
        assert!(exec.is_tile_profiling_enabled());
        exec.disable_tile_profiling();
        assert!(!exec.is_tile_profiling_enabled());
    }

    #[test]
    fn test_tile_stats() {
        let Some(exec) = create_executor() else {
            return;
        };
        let _stats = exec.tile_stats(trueno::TileLevel::Macro);
    }

    #[test]
    fn test_tile_summary() {
        let Some(exec) = create_executor() else {
            return;
        };
        let summary = exec.tile_summary();
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_tile_stats_json() {
        let Some(exec) = create_executor() else {
            return;
        };
        let json = exec.tile_stats_json();
        assert!(json.contains('{'));
    }

    #[test]
    fn test_reset_tile_stats() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.reset_tile_stats();
    }

    // ========================================================================
    // Device Info Tests
    // ========================================================================

    #[test]
    fn test_device_name() {
        let Some(exec) = create_executor() else {
            return;
        };
        let result = exec.device_name();
        assert!(result.is_ok());
        let name = result.unwrap();
        assert!(!name.is_empty());
    }

    #[test]
    fn test_memory_info() {
        let Some(exec) = create_executor() else {
            return;
        };
        let result = exec.memory_info();
        assert!(result.is_ok());
        let (free, total) = result.unwrap();
        assert!(total > 0);
        assert!(free <= total);
    }

    // ========================================================================
    // QWEN-010: Optimal Tile Size Tests
    // ========================================================================

    #[test]
    fn test_optimal_tile_size() {
        let Some(exec) = create_executor() else {
            return;
        };
        let tile_size = exec.optimal_tile_size();
        // Should be either 32 or 64 depending on GPU
        assert!(tile_size == 32 || tile_size == 64);

        // For RTX 4090 (our development GPU), should be 64
        if let Ok(name) = exec.device_name() {
            if name.contains("4090") {
                assert_eq!(tile_size, 64, "RTX 4090 should use 64x64 tiles");
            }
        }
    }

    // ========================================================================
    // Synchronization Tests
    // ========================================================================

    #[test]
    fn test_synchronize() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(exec.synchronize().is_ok());
    }

    // ========================================================================
    // Memory Pool Tests
    // ========================================================================

    #[test]
    fn test_pool_stats() {
        let Some(exec) = create_executor() else {
            return;
        };
        let stats = exec.pool_stats();
        let _ = stats.pool_hits; // Verify stats are accessible
    }

    #[test]
    fn test_staging_pool_stats() {
        let Some(exec) = create_executor() else {
            return;
        };
        let stats = exec.staging_pool_stats();
        let _ = stats.pool_hits; // Verify stats are accessible
    }

    #[test]
    fn test_get_return_staging_buffer() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let buf = exec.get_staging_buffer(1024);
        exec.return_staging_buffer(buf);
    }

    #[test]
    fn test_clear_pool() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.clear_pool();
    }
}
