//! Metrics & Health Monitoring (PMAT-802)
//!
//! M28: InferenceMetrics, HealthChecker, ShutdownCoordinator, GpuCompute, HybridScheduler.

use super::MatmulOp;
use crate::error::{RealizarError, Result};
use crate::tensor::Tensor;

// =============================================================================
// M28: Metrics & Health Monitoring (Phase 19)
// =============================================================================

/// Inference metrics collector (M28 - IMP-067)
///
/// Collects and aggregates inference performance metrics including
/// latency distribution and throughput.
#[derive(Debug)]
pub struct InferenceMetrics {
    latencies: Vec<std::time::Duration>,
    total_tokens: u64,
    start_time: std::time::Instant,
}

impl InferenceMetrics {
    /// Create a new inference metrics collector
    #[must_use]
    pub fn new() -> Self {
        Self {
            latencies: Vec::new(),
            total_tokens: 0,
            start_time: std::time::Instant::now(),
        }
    }

    /// Get total number of recorded inferences
    #[must_use]
    pub fn total_inferences(&self) -> usize {
        self.latencies.len()
    }

    /// Get total number of tokens processed
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Record an inference with its latency and token count
    pub fn record_inference(&mut self, latency: std::time::Duration, tokens: usize) {
        self.latencies.push(latency);
        self.total_tokens += tokens as u64;
    }

    /// Get latency at given percentile (0-100)
    ///
    /// Returns None if no inferences recorded.
    #[must_use]
    pub fn latency_percentile(&self, percentile: u8) -> Option<std::time::Duration> {
        if self.latencies.is_empty() {
            return None;
        }

        let mut sorted = self.latencies.clone();
        sorted.sort();

        let idx = ((percentile as usize) * sorted.len() / 100).min(sorted.len() - 1);
        Some(sorted[idx])
    }

    /// Calculate throughput in tokens per second
    #[must_use]
    pub fn throughput(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_tokens as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.latencies.clear();
        self.total_tokens = 0;
        self.start_time = std::time::Instant::now();
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for health check function
pub type HealthCheckFn = Box<dyn Fn() -> bool + Send + Sync>;

/// Health checker for system components (M28 - IMP-068)
///
/// Monitors health status of system components via registered check functions.
pub struct HealthChecker {
    checks: Vec<(String, HealthCheckFn)>,
    last_results: std::collections::HashMap<String, bool>,
}

impl std::fmt::Debug for HealthChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HealthChecker")
            .field("check_count", &self.checks.len())
            .field("last_results", &self.last_results)
            .finish()
    }
}

impl HealthChecker {
    /// Create a new health checker
    #[must_use]
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
            last_results: std::collections::HashMap::new(),
        }
    }

    /// Get number of registered checks
    #[must_use]
    pub fn check_count(&self) -> usize {
        self.checks.len()
    }

    /// Register a health check function
    pub fn register_check(&mut self, name: &str, check: HealthCheckFn) {
        self.checks.push((name.to_string(), check));
    }

    /// Run all health checks and return results
    pub fn check_all(&mut self) -> std::collections::HashMap<String, bool> {
        let mut results = std::collections::HashMap::new();
        for (name, check) in &self.checks {
            let healthy = check();
            results.insert(name.clone(), healthy);
        }
        self.last_results.clone_from(&results);
        results
    }

    /// Check if system is overall healthy (all checks pass)
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        if self.checks.is_empty() {
            return true;
        }
        self.last_results.values().all(|&v| v)
    }

    /// Clear all registered checks
    pub fn clear(&mut self) {
        self.checks.clear();
        self.last_results.clear();
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for shutdown handler function
pub type ShutdownHandlerFn = Box<dyn Fn() + Send + Sync>;

/// Graceful shutdown coordinator (M28 - IMP-069)
///
/// Coordinates shutdown sequence with request draining and handler callbacks.
pub struct ShutdownCoordinator {
    shutting_down: bool,
    pending_requests: u32,
    handlers: Vec<ShutdownHandlerFn>,
}

impl std::fmt::Debug for ShutdownCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShutdownCoordinator")
            .field("shutting_down", &self.shutting_down)
            .field("pending_requests", &self.pending_requests)
            .field("handler_count", &self.handlers.len())
            .finish()
    }
}

impl ShutdownCoordinator {
    /// Create a new shutdown coordinator
    #[must_use]
    pub fn new() -> Self {
        Self {
            shutting_down: false,
            pending_requests: 0,
            handlers: Vec::new(),
        }
    }

    /// Check if shutdown has been initiated
    #[must_use]
    pub fn is_shutting_down(&self) -> bool {
        self.shutting_down
    }

    /// Get number of pending requests
    #[must_use]
    pub fn pending_requests(&self) -> u32 {
        self.pending_requests
    }

    /// Get number of registered handlers
    #[must_use]
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }

    /// Register a shutdown handler
    pub fn register_handler(&mut self, handler: ShutdownHandlerFn) {
        self.handlers.push(handler);
    }

    /// Mark that a request has started
    pub fn request_started(&mut self) {
        self.pending_requests += 1;
    }

    /// Mark that a request has completed
    pub fn request_completed(&mut self) {
        self.pending_requests = self.pending_requests.saturating_sub(1);
    }

    /// Initiate shutdown sequence
    ///
    /// Calls all registered handlers.
    pub fn initiate_shutdown(&mut self) {
        if self.shutting_down {
            return;
        }
        self.shutting_down = true;

        // Call all handlers
        for handler in &self.handlers {
            handler();
        }
    }

    /// Check if shutdown is complete (initiated + no pending requests)
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.shutting_down && self.pending_requests == 0
    }
}

impl Default for ShutdownCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeBackend {
    /// GPU compute via trueno's wgpu backend
    Gpu,
    /// CPU compute (fallback)
    Cpu,
    /// Auto-select best available backend
    #[default]
    Auto,
}

/// GPU compute context
///
/// Provides GPU-accelerated operations with automatic fallback to CPU
/// when GPU is not available.
pub struct GpuCompute {
    backend: ComputeBackend,
    gpu: Option<trueno::backends::gpu::GpuBackend>,
}

impl GpuCompute {
    /// Create GPU compute context with auto-detected backend
    ///
    /// Attempts to initialize GPU backend, falls back to CPU if unavailable.
    ///
    /// # Errors
    ///
    /// Returns error if both GPU and CPU initialization fail (should not happen).
    pub fn auto() -> Result<Self> {
        Self::new(ComputeBackend::Auto)
    }

    /// Create GPU compute context with specified backend
    ///
    /// # Arguments
    ///
    /// * `backend` - Backend selection (Gpu, Cpu, or Auto)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `Gpu` backend requested but GPU is not available
    /// - Backend initialization fails
    pub fn new(backend: ComputeBackend) -> Result<Self> {
        match backend {
            ComputeBackend::Gpu => {
                if trueno::backends::gpu::GpuBackend::is_available() {
                    Ok(Self {
                        backend: ComputeBackend::Gpu,
                        gpu: Some(trueno::backends::gpu::GpuBackend::new()),
                    })
                } else {
                    Err(RealizarError::GpuError {
                        reason: "GPU not available".to_string(),
                    })
                }
            },
            ComputeBackend::Cpu => Ok(Self {
                backend: ComputeBackend::Cpu,
                gpu: None,
            }),
            ComputeBackend::Auto => {
                if trueno::backends::gpu::GpuBackend::is_available() {
                    Ok(Self {
                        backend: ComputeBackend::Gpu,
                        gpu: Some(trueno::backends::gpu::GpuBackend::new()),
                    })
                } else {
                    Ok(Self {
                        backend: ComputeBackend::Cpu,
                        gpu: None,
                    })
                }
            },
        }
    }

    /// Check if GPU backend is active
    #[must_use]
    pub fn is_gpu(&self) -> bool {
        self.backend == ComputeBackend::Gpu && self.gpu.is_some()
    }

    /// Get active backend type
    #[must_use]
    pub fn backend(&self) -> ComputeBackend {
        self.backend
    }

    /// GPU-accelerated matrix multiplication
    ///
    /// Computes `C = A @ B` where:
    /// - A is `[m, k]`
    /// - B is `[k, n]`
    /// - C is `[m, n]`
    ///
    /// # Arguments
    ///
    /// * `a` - Left matrix as flat f32 slice, row-major `[m, k]`
    /// * `b` - Right matrix as flat f32 slice, row-major `[k, n]`
    /// * `m` - Rows in A and C
    /// * `k` - Cols in A, rows in B
    /// * `n` - Cols in B and C
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input dimensions don't match
    /// - GPU compute fails
    #[allow(clippy::many_single_char_names)]
    pub fn matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Validate dimensions
        if a.len() != m * k {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Matrix A size {} doesn't match m*k={}*{}={}",
                    a.len(),
                    m,
                    k,
                    m * k
                ),
            });
        }
        if b.len() != k * n {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Matrix B size {} doesn't match k*n={}*{}={}",
                    b.len(),
                    k,
                    n,
                    k * n
                ),
            });
        }

        if let Some(gpu) = &mut self.gpu {
            // GPU path
            #[allow(clippy::implicit_clone)]
            gpu.matmul(a, b, m, k, n)
                .map_err(|e| RealizarError::GpuError {
                    reason: e.to_string(),
                })
        } else {
            // CPU fallback: naive matmul
            Ok(cpu_matmul(a, b, m, k, n))
        }
    }

    /// GPU-accelerated matrix multiplication with Tensor input/output
    ///
    /// # Arguments
    ///
    /// * `a` - Left tensor `[m, k]`
    /// * `b` - Right tensor `[k, n]`
    ///
    /// # Errors
    ///
    /// Returns error if tensors are not 2D or dimensions don't match.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul_tensor(&mut self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(RealizarError::InvalidShape {
                reason: "matmul_tensor requires 2D tensors".to_string(),
            });
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let k2 = b_shape[0];
        let n = b_shape[1];

        if k != k2 {
            return Err(RealizarError::InvalidShape {
                reason: format!("Inner dimensions don't match: A[{m},{k}] @ B[{k2},{n}]"),
            });
        }

        let result = self.matmul(a.data(), b.data(), m, k, n)?;
        Tensor::from_vec(vec![m, n], result)
    }

    /// GPU-accelerated vector dot product
    ///
    /// # Errors
    ///
    /// Returns error if vectors have different lengths or GPU compute fails.
    pub fn dot(&mut self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!("Vector lengths don't match: {} vs {}", a.len(), b.len()),
            });
        }

        if let Some(gpu) = &mut self.gpu {
            #[allow(clippy::implicit_clone)]
            gpu.dot(a, b).map_err(|e| RealizarError::GpuError {
                reason: e.to_string(),
            })
        } else {
            // CPU fallback
            Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
        }
    }

    /// GPU-accelerated ReLU activation
    ///
    /// # Errors
    ///
    /// Returns error if GPU compute fails.
    pub fn relu(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if let Some(gpu) = &mut self.gpu {
            #[allow(clippy::implicit_clone)]
            gpu.relu(input).map_err(|e| RealizarError::GpuError {
                reason: e.to_string(),
            })
        } else {
            Ok(input.iter().map(|&x| x.max(0.0)).collect())
        }
    }

    /// GPU-accelerated sigmoid activation
    ///
    /// # Errors
    ///
    /// Returns error if GPU compute fails.
    pub fn sigmoid(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if let Some(gpu) = &mut self.gpu {
            #[allow(clippy::implicit_clone)]
            gpu.sigmoid(input).map_err(|e| RealizarError::GpuError {
                reason: e.to_string(),
            })
        } else {
            Ok(input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect())
        }
    }
}

/// CPU fallback matmul implementation
#[allow(clippy::many_single_char_names)]
pub(crate) fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // For m=1 (vector-matrix multiply), use optimized path
    if m == 1 {
        return cpu_vector_matmul(a, b, k, n);
    }

    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// IMP-098: Parallelized vector-matrix multiply: a[1,k] @ b[k,n] -> c[1,n]
///
/// Uses parallel output chunks for multi-core utilization.
/// Each thread accumulates its chunk of outputs independently.
#[allow(clippy::many_single_char_names)]
fn cpu_vector_matmul(a: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
    use rayon::prelude::*;

    // For small n, use sequential (avoids rayon overhead)
    if n < 2048 {
        return cpu_vector_matmul_seq(a, b, k, n);
    }

    // Parallel over output chunks
    const CHUNK_SIZE: usize = 1024;
    let num_chunks = n.div_ceil(CHUNK_SIZE);

    let chunks: Vec<Vec<f32>> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(n);
            let chunk_len = end - start;
            let mut chunk_c = vec![0.0f32; chunk_len];

            // Accumulate this chunk of outputs
            for (p, &a_val) in a.iter().enumerate() {
                let row_start = p * n + start;
                let row = &b[row_start..row_start + chunk_len];
                for (j, &b_val) in row.iter().enumerate() {
                    chunk_c[j] += a_val * b_val;
                }
            }
            chunk_c
        })
        .collect();

    // Flatten chunks into result
    chunks.into_iter().flatten().collect()
}

/// Sequential fallback for small outputs
#[allow(clippy::many_single_char_names)]
fn cpu_vector_matmul_seq(a: &[f32], b: &[f32], _k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n];

    // Row-major accumulation: for each row of B, scale by corresponding a[p]
    for (p, &a_val) in a.iter().enumerate() {
        let row = &b[p * n..(p + 1) * n];
        for (j, &b_val) in row.iter().enumerate() {
            c[j] += a_val * b_val;
        }
    }

    c
}

/// CPU matmul with B transposed: A @ B^T
/// a[m,k] @ b[n,k]^T -> c[m,n]
#[allow(clippy::many_single_char_names)]
pub(crate) fn cpu_matmul_transpose_b(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                // a[i,p] * b[j,p] (b is stored row-major as [n,k])
                sum += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Transpose a matrix from [rows, cols] to [cols, rows]
pub(crate) fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut result = vec![0.0; data.len()];
    for i in 0..rows {
        for j in 0..cols {
            result[j * rows + i] = data[i * cols + j];
        }
    }
    result
}

/// IMP-096: Parallel SIMD vector-matrix multiply using transposed weights
///
/// Computes a[1,k] @ weight_t[n,k]^T + bias[n] -> c[n]
/// Each output c[j] = dot(a, weight_t[j,:]) + bias[j]
///
/// Uses transposed weights for row-major access pattern (contiguous dot products).
/// Parallelized with rayon. Compiler auto-vectorizes the inner dot product.
#[allow(clippy::many_single_char_names)]
pub(crate) fn cpu_matmul_transposed_simd(
    a: &[f32],        // Input vector: [k]
    weight_t: &[f32], // Transposed weights: [n, k] (row-major)
    bias: &[f32],     // Bias: [n]
    k: usize,
    n: usize,
) -> Vec<f32> {
    use rayon::prelude::*;

    // Process in chunks for better parallelism and cache locality
    const CHUNK_SIZE: usize = 4096;

    (0..n)
        .into_par_iter()
        .step_by(CHUNK_SIZE)
        .flat_map(|chunk_start| {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(n);
            (chunk_start..chunk_end)
                .map(|j| {
                    // Row-major access: weight_t[j, :] is contiguous in memory
                    let row = &weight_t[j * k..(j + 1) * k];

                    // Compiler auto-vectorizes this dot product pattern
                    let dot: f32 = row.iter().zip(a.iter()).map(|(&w, &h)| w * h).sum();
                    dot + bias[j]
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// GPU buffer pool for memory reuse and reduced allocation overhead
pub struct GpuBufferPool {
    /// Available buffers indexed by size bucket
    available_buffers: std::collections::HashMap<usize, Vec<Vec<f32>>>,
    /// Size buckets for efficient pooling (powers of 2)
    bucket_sizes: Vec<usize>,
    /// Maximum cached buffers per bucket
    max_per_bucket: usize,
}

impl GpuBufferPool {
    /// Create new buffer pool with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            available_buffers: std::collections::HashMap::new(),
            bucket_sizes: (10..=24).map(|i| 1 << i).collect(), // 1KB to 16MB
            max_per_bucket: 4,
        }
    }

    /// Get bucket size for requested allocation
    fn get_bucket(&self, size: usize) -> usize {
        *self
            .bucket_sizes
            .iter()
            .find(|&&b| b >= size)
            .unwrap_or(&size)
    }

    /// Acquire buffer of at least `size` elements
    pub fn acquire(&mut self, size: usize) -> Vec<f32> {
        let bucket = self.get_bucket(size);
        if let Some(buffers) = self.available_buffers.get_mut(&bucket) {
            if let Some(mut buf) = buffers.pop() {
                buf.resize(size, 0.0);
                return buf;
            }
        }
        vec![0.0; size]
    }

    /// Release buffer back to pool for reuse
    pub fn release(&mut self, mut buffer: Vec<f32>) {
        let bucket = self.get_bucket(buffer.capacity());
        let buffers = self.available_buffers.entry(bucket).or_default();
        if buffers.len() < self.max_per_bucket {
            buffer.clear();
            buffers.push(buffer);
        }
        // Otherwise just drop it
    }

    /// Clear all cached buffers
    pub fn clear(&mut self) {
        self.available_buffers.clear();
    }

    /// Get configured bucket sizes
    #[must_use]
    pub fn bucket_sizes(&self) -> &[usize] {
        &self.bucket_sizes
    }

    /// Get pool statistics
    #[must_use]
    pub fn stats(&self) -> GpuPoolStats {
        let total_buffers: usize = self.available_buffers.values().map(Vec::len).sum();
        let total_bytes: usize = self
            .available_buffers
            .iter()
            .map(|(bucket, buffers)| bucket * buffers.len() * 4)
            .sum();
        GpuPoolStats {
            cached_buffers: total_buffers,
            cached_bytes: total_bytes,
        }
    }
}

impl Default for GpuBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU buffer pool statistics
#[derive(Debug, Clone, Copy)]
pub struct GpuPoolStats {
    /// Number of cached buffers
    pub cached_buffers: usize,
    /// Total cached bytes
    pub cached_bytes: usize,
}

/// Async GPU compute handle for non-blocking operations
///
/// Per spec: "Async transfer - No host blocking"
pub struct AsyncGpuResult {
    /// Result data when ready
    result: Option<Vec<f32>>,
    /// Whether computation is complete
    ready: bool,
}

impl AsyncGpuResult {
    /// Create result that's immediately ready (CPU fallback)
    pub fn ready(data: Vec<f32>) -> Self {
        Self {
            result: Some(data),
            ready: true,
        }
    }

    /// Create pending result (GPU async)
    pub fn pending() -> Self {
        Self {
            result: None,
            ready: false,
        }
    }

    /// Check if result is ready
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Mark as ready with result
    pub fn set_result(&mut self, data: Vec<f32>) {
        self.result = Some(data);
        self.ready = true;
    }

    /// Block until result is ready (for synchronization points)
    pub fn wait(self) -> Vec<f32> {
        self.result.expect("Result not ready")
    }

    /// Try to get result without blocking
    pub fn try_get(&self) -> Option<&Vec<f32>> {
        if self.ready {
            self.result.as_ref()
        } else {
            None
        }
    }
}

/// Hybrid CPU/GPU scheduler
///
/// Automatically selects optimal backend based on workload size.
pub struct HybridScheduler {
    gpu_compute: GpuCompute,
    /// Minimum matrix size (m*k*n) to use GPU
    gpu_threshold: usize,
    /// Buffer pool for memory reuse
    buffer_pool: GpuBufferPool,
}

impl HybridScheduler {
    /// Create hybrid scheduler with auto-detected GPU
    ///
    /// # Errors
    ///
    /// Returns error if compute initialization fails.
    pub fn new() -> Result<Self> {
        Ok(Self {
            gpu_compute: GpuCompute::auto()?,
            gpu_threshold: 64 * 64 * 64, // 262K elements
            buffer_pool: GpuBufferPool::new(),
        })
    }

    /// Create scheduler with custom threshold
    ///
    /// # Arguments
    ///
    /// * `gpu_threshold` - Minimum m*k*n to trigger GPU acceleration
    ///
    /// # Errors
    ///
    /// Returns error if compute initialization fails.
    pub fn with_threshold(gpu_threshold: usize) -> Result<Self> {
        Ok(Self {
            gpu_compute: GpuCompute::auto()?,
            gpu_threshold,
            buffer_pool: GpuBufferPool::new(),
        })
    }

    /// Check if GPU is available
    #[must_use]
    pub fn has_gpu(&self) -> bool {
        self.gpu_compute.is_gpu()
    }

    /// Get GPU threshold
    #[must_use]
    pub fn gpu_threshold(&self) -> usize {
        self.gpu_threshold
    }

    /// Decide whether to use GPU for given workload
    ///
    /// IMP-097: For m=1 (single-token inference), CPU is faster due to:
    /// - No GPU data transfer overhead
    /// - No kernel launch latency
    /// - CPU SIMD is sufficient for vector-matrix multiply
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn should_use_gpu(&self, m: usize, k: usize, n: usize) -> bool {
        // IMP-097: Force CPU for single-token operations (m=1)
        // GPU kernel launch overhead exceeds compute benefit for small batch sizes
        if m <= 1 {
            return false;
        }
        self.gpu_compute.is_gpu() && (m * k * n) >= self.gpu_threshold
    }

    /// Execute matmul with automatic backend selection
    ///
    /// Uses GPU for large matrices, CPU for small ones.
    ///
    /// # Errors
    ///
    /// Returns error if compute fails.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        if self.should_use_gpu(m, k, n) {
            self.gpu_compute.matmul(a, b, m, k, n)
        } else {
            Ok(cpu_matmul(a, b, m, k, n))
        }
    }

    /// Execute matmul with pooled output buffer
    ///
    /// Reduces allocation overhead by reusing buffers.
    ///
    /// # Errors
    ///
    /// Returns error if compute fails.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul_pooled(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Acquire buffer from pool
        let mut output = self.buffer_pool.acquire(m * n);

        // Compute result
        let result = if self.should_use_gpu(m, k, n) {
            self.gpu_compute.matmul(a, b, m, k, n)?
        } else {
            cpu_matmul(a, b, m, k, n)
        };

        // Copy to pooled buffer
        output.copy_from_slice(&result);
        Ok(output)
    }

    /// Release buffer back to pool
    ///
    /// Call this when done with a buffer returned by `matmul_pooled`.
    pub fn release_buffer(&mut self, buffer: Vec<f32>) {
        self.buffer_pool.release(buffer);
    }

    /// Get buffer pool statistics
    #[must_use]
    pub fn pool_stats(&self) -> GpuPoolStats {
        self.buffer_pool.stats()
    }

    /// Execute matmul asynchronously (non-blocking on CPU fallback)
    ///
    /// Per spec: "Async transfer - No host blocking"
    ///
    /// # Errors
    ///
    /// Returns error if compute setup fails.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul_async(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<AsyncGpuResult> {
        // For CPU fallback, compute immediately
        // For GPU, this would submit to command queue without blocking
        let result = if self.should_use_gpu(m, k, n) {
            self.gpu_compute.matmul(a, b, m, k, n)?
        } else {
            cpu_matmul(a, b, m, k, n)
        };

        Ok(AsyncGpuResult::ready(result))
    }

    /// Process batch of matmuls with optimal scheduling
    ///
    /// Batches small operations for CPU, pipelines large ones for GPU.
    ///
    /// # Errors
    ///
    /// Returns error if any compute fails.
    pub fn matmul_batch(&mut self, operations: &[MatmulOp]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(operations.len());

        for (a, b, m, k, n) in operations {
            let result = self.matmul(a, b, *m, *k, *n)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Execute matmul with B transposed: A @ B^T
    ///
    /// Computes C[m,n] = A[m,k] @ B[n,k]^T
    /// where B is stored row-major as [n, k].
    ///
    /// # Errors
    ///
    /// Returns error if compute fails.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul_transpose_b(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // For attention: Q[seq, head_dim] @ K[seq, head_dim]^T = scores[seq, seq]
        // B is stored as [n, k], we need B^T which is [k, n]
        if self.should_use_gpu(m, k, n) {
            // Transpose B and use GPU matmul
            let b_t = transpose(b, n, k);
            self.gpu_compute.matmul(a, &b_t, m, k, n)
        } else {
            // CPU: compute A @ B^T directly
            Ok(cpu_matmul_transpose_b(a, b, m, k, n))
        }
    }
}
