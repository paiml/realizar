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
        let context = CudaContext::new(device_ordinal)?;
        // PARITY-038: Create multiple streams for overlapped execution
        let compute_stream = CudaStream::new(&context)?;
        let transfer_stream = CudaStream::new(&context)?;
        // CORRECTNESS-011: Create a separate stream for legacy operations
        // NOTE: Graph capture uses `stream.begin_capture()` and most kernels use `stream`
        // BUT attention kernels use `compute_stream` - fixed in incremental_attention_into_inner
        // to use `stream` during graph capture mode
        let stream = CudaStream::new(&context)?;

        Ok(Self {
            // Initialize in struct declaration order (for clarity)
            kernels: CudaKernels::new(),
            memory_pool: GpuMemoryPool::new(),
            staging_pool: StagingBufferPool::new(), // PARITY-042: pinned memory pool
            modules: HashMap::new(),
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
            compute_stream,
            transfer_stream,
            stream,
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
            // PAR-073: BrickProfiler (disabled by default for zero overhead)
            // Enable with executor.enable_profiling() for per-brick timing
            profiler: trueno::BrickProfiler::new(),
            context, // Last field - dropped last
        })
    }

    /// Check if CUDA is available on this system
    #[must_use]
    pub fn is_available() -> bool {
        cuda_available()
    }

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
