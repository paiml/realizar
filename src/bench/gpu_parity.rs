//! GPU parity benchmarks for comparing Realizar vs Ollama/llama.cpp
//!
//! Extracted from bench/mod.rs (PMAT-802) to reduce module size.
//! Contains:
//! - IMP-800: TRUE GPU Parity Benchmark (M2 Milestone)
//! - IMP-900: Closing the 18x Gap (M3/M4 Milestones)

use serde::{Deserialize, Serialize};

// ============================================================================
// IMP-800: TRUE GPU Parity Benchmark (M2 Milestone)
// ============================================================================

/// GPU parity benchmark configuration (IMP-800b)
///
/// Configures apples-to-apples throughput comparison on same GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuParityBenchmark {
    /// Model to benchmark (phi-2 Q4_K_M)
    pub model_path: String,
    /// Prompt for generation
    pub prompt: String,
    /// Number of tokens to generate
    pub max_tokens: usize,
    /// Ollama endpoint for comparison
    pub ollama_endpoint: String,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Target CV for stable measurements
    pub target_cv: f64,
}

impl Default for GpuParityBenchmark {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prompt: "The quick brown fox".to_string(),
            max_tokens: 32,
            ollama_endpoint: "http://localhost:11434".to_string(),
            warmup_iterations: 3,
            measurement_iterations: 10,
            target_cv: 0.05,
        }
    }
}

impl GpuParityBenchmark {
    /// Create a new GPU parity benchmark with model path
    #[must_use]
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            ..Default::default()
        }
    }

    /// Set the prompt for generation
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// Set the number of tokens to generate
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the Ollama endpoint
    #[must_use]
    pub fn with_ollama_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.ollama_endpoint = endpoint.into();
        self
    }

    /// Set the number of warmup iterations
    #[must_use]
    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup_iterations = warmup;
        self
    }

    /// Set the number of measurement iterations
    #[must_use]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.measurement_iterations = iterations;
        self
    }
}

/// Benchmark result with statistical analysis (IMP-800b)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuParityResult {
    /// Realizar GPU throughput (tok/s)
    pub realizar_gpu_tps: f64,
    /// Ollama throughput (tok/s)
    pub ollama_tps: f64,
    /// Performance gap ratio (Ollama / Realizar)
    pub gap_ratio: f64,
    /// Coefficient of variation (measurement stability)
    pub cv: f64,
    /// GPU device name
    pub gpu_device: String,
    /// VRAM usage (MB)
    pub vram_mb: u64,
    /// Realizar latency p50 (ms)
    pub realizar_p50_ms: f64,
    /// Ollama latency p50 (ms)
    pub ollama_p50_ms: f64,
}

impl GpuParityResult {
    /// Create a new GPU parity result
    #[must_use]
    pub fn new(
        realizar_gpu_tps: f64,
        ollama_tps: f64,
        cv: f64,
        gpu_device: impl Into<String>,
        vram_mb: u64,
    ) -> Self {
        let gap_ratio = if realizar_gpu_tps > 0.0 {
            ollama_tps / realizar_gpu_tps
        } else {
            f64::INFINITY
        };

        Self {
            realizar_gpu_tps,
            ollama_tps,
            gap_ratio,
            cv,
            gpu_device: gpu_device.into(),
            vram_mb,
            realizar_p50_ms: 0.0,
            ollama_p50_ms: 0.0,
        }
    }

    /// Returns true if within 2x of Ollama (M2 target)
    #[must_use]
    pub fn achieves_m2_parity(&self) -> bool {
        self.gap_ratio <= 2.0
    }

    /// Returns true if within 1.25x of Ollama (M4 target)
    #[must_use]
    pub fn achieves_m4_parity(&self) -> bool {
        self.gap_ratio <= 1.25
    }

    /// Returns true if GPU is faster than CPU SIMD baseline (5 tok/s)
    #[must_use]
    pub fn gpu_faster_than_cpu(&self) -> bool {
        self.realizar_gpu_tps > 5.0
    }

    /// Returns true if measurements are stable (CV < 0.05)
    #[must_use]
    pub fn measurements_stable(&self) -> bool {
        self.cv < 0.05
    }

    /// Get speedup over CPU SIMD baseline
    #[must_use]
    pub fn cpu_speedup(&self) -> f64 {
        self.realizar_gpu_tps / 5.0 // CPU baseline ~5 tok/s
    }
}

/// Gap analysis with falsifiable claims (IMP-800c)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapAnalysis {
    /// Claimed gap reduction
    pub claimed_gap: f64,
    /// Measured gap
    pub measured_gap: f64,
    /// Statistical significance (p-value)
    pub p_value: f64,
    /// Confidence interval lower bound (95%)
    pub ci_95_lower: f64,
    /// Confidence interval upper bound (95%)
    pub ci_95_upper: f64,
    /// Popper score (falsifiability, 0-100)
    pub popper_score: f64,
    /// Claim descriptions
    pub claims: Vec<FalsifiableClaim>,
}

/// A falsifiable claim for Popperian testing (IMP-800c)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsifiableClaim {
    /// Claim identifier
    pub id: String,
    /// Claim description
    pub description: String,
    /// Expected value
    pub expected: f64,
    /// Threshold for verification
    pub threshold: f64,
    /// Measured value
    pub measured: f64,
    /// Whether claim is verified
    pub verified: bool,
}

impl FalsifiableClaim {
    /// Create a new falsifiable claim
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        expected: f64,
        threshold: f64,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            expected,
            threshold,
            measured: 0.0,
            verified: false,
        }
    }

    /// Evaluate the claim against a measured value
    #[must_use]
    pub fn evaluate(mut self, measured: f64) -> Self {
        self.measured = measured;
        self.verified = measured >= self.threshold;
        self
    }
}

impl GapAnalysis {
    /// Create a new gap analysis
    #[must_use]
    pub fn new(claimed_gap: f64, measured_gap: f64) -> Self {
        Self {
            claimed_gap,
            measured_gap,
            p_value: 0.0,
            ci_95_lower: 0.0,
            ci_95_upper: 0.0,
            popper_score: 0.0,
            claims: Vec::new(),
        }
    }

    /// Add statistical bounds
    #[must_use]
    pub fn with_statistics(mut self, p_value: f64, ci_lower: f64, ci_upper: f64) -> Self {
        self.p_value = p_value;
        self.ci_95_lower = ci_lower;
        self.ci_95_upper = ci_upper;
        self
    }

    /// Calculate and set Popper score based on claims
    pub fn calculate_popper_score(&mut self) {
        if self.claims.is_empty() {
            self.popper_score = 0.0;
            return;
        }

        let verified_count = self.claims.iter().filter(|c| c.verified).count();
        self.popper_score = (verified_count as f64 / self.claims.len() as f64) * 100.0;
    }

    /// Add a falsifiable claim
    pub fn add_claim(&mut self, claim: FalsifiableClaim) {
        self.claims.push(claim);
    }

    /// Claim is verified if measured within CI
    #[must_use]
    pub fn claim_verified(&self) -> bool {
        self.measured_gap >= self.ci_95_lower && self.measured_gap <= self.ci_95_upper
    }

    /// Create default IMP-800c claims
    #[must_use]
    pub fn with_default_claims(mut self, realizar_gpu_tps: f64) -> Self {
        // IMP-800c-1: GPU faster than CPU SIMD (>5x, threshold 25 tok/s)
        self.claims.push(
            FalsifiableClaim::new("IMP-800c-1", "GPU faster than CPU SIMD (>5x)", 5.0, 25.0)
                .evaluate(realizar_gpu_tps),
        );

        // IMP-800c-2: GPU within 10x of Ollama (threshold 24 tok/s)
        self.claims.push(
            FalsifiableClaim::new("IMP-800c-2", "GPU within 10x of Ollama", 10.0, 24.0)
                .evaluate(realizar_gpu_tps),
        );

        // IMP-800c-3: GPU within 2x of Ollama - M2 (threshold 120 tok/s)
        self.claims.push(
            FalsifiableClaim::new("IMP-800c-3", "GPU within 2x of Ollama (M2)", 2.0, 120.0)
                .evaluate(realizar_gpu_tps),
        );

        // IMP-800c-4: GPU at parity with Ollama - M4 (threshold 192 tok/s)
        self.claims.push(
            FalsifiableClaim::new("IMP-800c-4", "GPU at parity with Ollama (M4)", 1.25, 192.0)
                .evaluate(realizar_gpu_tps),
        );

        self.calculate_popper_score();
        self
    }
}

// ============================================================================
// IMP-900: Closing the 18x Gap (M3/M4 Milestones)
// ============================================================================

/// Optimized GEMM configuration (IMP-900a)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedGemmConfig {
    /// Tile size for shared memory (typically 32 or 64)
    pub tile_size: u32,
    /// Register blocking factor (typically 4 or 8)
    pub reg_block: u32,
    /// Use tensor cores if available (SM 7.0+)
    pub use_tensor_cores: bool,
    /// Vectorized loads (float4 = 4)
    pub vector_width: u32,
    /// Unroll factor for K-loop
    pub k_unroll: u32,
    /// Use double buffering for tile prefetch
    pub double_buffer: bool,
}

impl Default for OptimizedGemmConfig {
    fn default() -> Self {
        Self {
            tile_size: 32,
            reg_block: 4,
            use_tensor_cores: false,
            vector_width: 4,
            k_unroll: 4,
            double_buffer: true,
        }
    }
}

impl OptimizedGemmConfig {
    /// Create configuration for small matrices (256x256)
    #[must_use]
    pub fn small() -> Self {
        Self {
            tile_size: 16,
            reg_block: 2,
            use_tensor_cores: false,
            vector_width: 4,
            k_unroll: 4,
            double_buffer: false,
        }
    }

    /// Create configuration for large matrices (1024+)
    #[must_use]
    pub fn large() -> Self {
        Self {
            tile_size: 64,
            reg_block: 8,
            use_tensor_cores: false,
            vector_width: 4,
            k_unroll: 8,
            double_buffer: true,
        }
    }

    /// Calculate shared memory requirement (bytes)
    #[must_use]
    pub fn shared_memory_bytes(&self) -> u32 {
        // Two tiles (A and B) in shared memory
        // Each tile is tile_size × tile_size × sizeof(f32)
        let tile_bytes = self.tile_size * self.tile_size * 4;
        if self.double_buffer {
            tile_bytes * 4 // 2 tiles × 2 buffers
        } else {
            tile_bytes * 2 // 2 tiles
        }
    }

    /// Calculate threads per block
    #[must_use]
    pub fn threads_per_block(&self) -> u32 {
        // Each thread computes reg_block × reg_block elements
        let threads_per_dim = self.tile_size / self.reg_block;
        threads_per_dim * threads_per_dim
    }

    /// Calculate registers per thread (for accumulators)
    #[must_use]
    pub fn registers_per_thread(&self) -> u32 {
        // reg_block × reg_block accumulator values
        self.reg_block * self.reg_block
    }
}

/// GEMM performance result (IMP-900a)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmPerformanceResult {
    /// Matrix M dimension (rows of A, rows of C)
    pub m: u32,
    /// Matrix N dimension (cols of B, cols of C)
    pub n: u32,
    /// Matrix K dimension (cols of A, rows of B)
    pub k: u32,
    /// Time in milliseconds
    pub time_ms: f64,
    /// GFLOP/s achieved
    pub gflops: f64,
    /// Memory bandwidth achieved (GB/s)
    pub bandwidth_gbs: f64,
    /// Percentage of peak performance
    pub efficiency: f64,
}

impl GemmPerformanceResult {
    /// Create a new GEMM performance result
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32, time_ms: f64) -> Self {
        // GEMM operations: 2 * M * N * K (multiply-add)
        let ops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k);
        let gflops = ops / (time_ms * 1e6);

        // Memory: read A (M*K), read B (K*N), write C (M*N)
        let bytes = (f64::from(m) * f64::from(k)
            + f64::from(k) * f64::from(n)
            + f64::from(m) * f64::from(n))
            * 4.0;
        let bandwidth_gbs = bytes / (time_ms * 1e6);

        Self {
            m,
            n,
            k,
            time_ms,
            gflops,
            bandwidth_gbs,
            efficiency: 0.0, // Set by caller based on peak
        }
    }

    /// Set efficiency based on peak GFLOP/s
    #[must_use]
    pub fn with_peak(mut self, peak_gflops: f64) -> Self {
        self.efficiency = (self.gflops / peak_gflops) * 100.0;
        self
    }

    /// Check if performance improved by at least the given factor
    #[must_use]
    pub fn improved_by(&self, baseline_gflops: f64, factor: f64) -> bool {
        self.gflops >= baseline_gflops * factor
    }
}

/// Optimized GEMM benchmark runner (IMP-900a)
#[derive(Debug)]
pub struct OptimizedGemmBenchmark {
    /// Configuration
    pub config: OptimizedGemmConfig,
    /// Warmup iterations
    pub warmup_iterations: usize,
    /// Measurement iterations
    pub measurement_iterations: usize,
    /// Target coefficient of variation
    pub target_cv: f64,
}

impl Default for OptimizedGemmBenchmark {
    fn default() -> Self {
        Self {
            config: OptimizedGemmConfig::default(),
            warmup_iterations: 5,
            measurement_iterations: 20,
            target_cv: 0.05,
        }
    }
}

impl OptimizedGemmBenchmark {
    /// Create benchmark with custom config
    #[must_use]
    pub fn with_config(config: OptimizedGemmConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }

    /// Calculate expected improvement over naive GEMM
    #[must_use]
    pub fn expected_improvement(&self) -> f64 {
        let mut improvement = 1.0;

        // Shared memory tiling: ~2x for cache efficiency
        improvement *= 2.0;

        // Register blocking: ~1.5x for reduced memory traffic
        if self.config.reg_block >= 4 {
            improvement *= 1.5;
        }

        // Vectorized loads: ~1.3x for coalesced access
        if self.config.vector_width >= 4 {
            improvement *= 1.3;
        }

        // Double buffering: ~1.2x for latency hiding
        if self.config.double_buffer {
            improvement *= 1.2;
        }

        improvement
    }
}

/// Kernel fusion configuration (IMP-900b)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FusedOpType {
    /// GEMM + bias + activation
    GemmBiasActivation,
    /// Layer normalization + linear projection
    LayerNormLinear,
    /// Fused attention (FlashAttention-style)
    FusedAttention,
    /// FFN: up projection + gate + down projection
    FusedFfn,
}

/// Fused operation specification (IMP-900b)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedOpSpec {
    /// Type of fused operation
    pub op_type: FusedOpType,
    /// Input dimensions
    pub input_dims: Vec<u32>,
    /// Output dimensions
    pub output_dims: Vec<u32>,
    /// Activation function (if applicable)
    pub activation: Option<String>,
    /// Number of kernel launches when fused
    pub fused_launches: u32,
    /// Number of kernel launches when unfused
    pub unfused_launches: u32,
}

impl FusedOpSpec {
    /// Calculate launch reduction factor
    #[must_use]
    pub fn launch_reduction(&self) -> f64 {
        f64::from(self.unfused_launches) / f64::from(self.fused_launches)
    }

    /// Check if fusion reduces launches by at least 50%
    #[must_use]
    pub fn achieves_target_reduction(&self) -> bool {
        self.launch_reduction() >= 2.0
    }
}

/// FlashAttention configuration (IMP-900c)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlashAttentionConfig {
    /// Block size for Q tiling (Br)
    pub block_size_q: u32,
    /// Block size for K/V tiling (Bc)
    pub block_size_kv: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Number of attention heads
    pub num_heads: u32,
    /// Use causal masking
    pub causal: bool,
    /// Softmax scale (default: 1/sqrt(head_dim))
    pub scale: f32,
}

impl FlashAttentionConfig {
    /// Create configuration for phi-2 model
    #[must_use]
    pub fn phi2() -> Self {
        Self {
            block_size_q: 64,
            block_size_kv: 64,
            head_dim: 80, // phi-2: 2560 / 32 heads
            num_heads: 32,
            causal: true,
            scale: 1.0 / (80.0_f32).sqrt(),
        }
    }

    /// Calculate memory required for attention (naive vs flash)
    #[must_use]
    pub fn memory_comparison(&self, seq_len: u32) -> (u64, u64) {
        // Naive: O(N²) attention matrix
        let naive_bytes = u64::from(seq_len) * u64::from(seq_len) * 4;

        // FlashAttention: O(N) working memory
        let flash_bytes = u64::from(self.block_size_q) * u64::from(self.block_size_kv) * 4 * 2; // S and P blocks

        (naive_bytes, flash_bytes)
    }

    /// Calculate memory savings factor
    #[must_use]
    pub fn memory_savings(&self, seq_len: u32) -> f64 {
        let (naive, flash) = self.memory_comparison(seq_len);
        naive as f64 / flash as f64
    }
}

/// Memory pool configuration (IMP-900d)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Initial pool size (bytes)
    pub initial_size: usize,
    /// Maximum pool size (bytes)
    pub max_size: usize,
    /// Size classes for allocation (powers of 2)
    pub size_classes: Vec<usize>,
    /// Use pinned memory for host staging
    pub use_pinned_memory: bool,
    /// Enable async transfers
    pub async_transfers: bool,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 256 * 1024 * 1024,  // 256 MB
            max_size: 2 * 1024 * 1024 * 1024, // 2 GB
            size_classes: vec![
                4096,        // 4 KB
                16384,       // 16 KB
                65536,       // 64 KB
                262_144,     // 256 KB
                1_048_576,   // 1 MB
                4_194_304,   // 4 MB
                16_777_216,  // 16 MB
                67_108_864,  // 64 MB
                268_435_456, // 256 MB
            ],
            use_pinned_memory: true,
            async_transfers: true,
        }
    }
}

impl MemoryPoolConfig {
    /// Find the smallest size class that fits the requested size
    #[must_use]
    pub fn find_size_class(&self, requested: usize) -> Option<usize> {
        self.size_classes
            .iter()
            .copied()
            .find(|&size| size >= requested)
    }

    /// Calculate expected bandwidth improvement from pinned memory
    #[must_use]
    pub fn expected_bandwidth_improvement(&self) -> f64 {
        if self.use_pinned_memory {
            2.4 // Pinned memory typically 2-3x faster
        } else {
            1.0
        }
    }
}

/// IMP-900 combined result (M3/M4 targets)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Imp900Result {
    /// Baseline throughput (13.1 tok/s from IMP-800)
    pub baseline_tps: f64,
    /// Throughput after optimizations
    pub optimized_tps: f64,
    /// GEMM optimization improvement factor
    pub gemm_improvement: f64,
    /// Kernel fusion improvement factor
    pub fusion_improvement: f64,
    /// FlashAttention improvement factor
    pub flash_attention_improvement: f64,
    /// Memory optimization improvement factor
    pub memory_improvement: f64,
    /// Gap to Ollama
    pub gap_ratio: f64,
    /// Target milestone achieved
    pub milestone: Option<String>,
}

impl Imp900Result {
    /// Create result from baseline
    #[must_use]
    pub fn from_baseline(baseline_tps: f64) -> Self {
        Self {
            baseline_tps,
            optimized_tps: baseline_tps,
            gemm_improvement: 1.0,
            fusion_improvement: 1.0,
            flash_attention_improvement: 1.0,
            memory_improvement: 1.0,
            gap_ratio: 240.0 / baseline_tps,
            milestone: None,
        }
    }

    /// Apply GEMM optimization
    #[must_use]
    pub fn with_gemm_improvement(mut self, factor: f64) -> Self {
        self.gemm_improvement = factor;
        self.recalculate();
        self
    }

    /// Apply fusion optimization
    #[must_use]
    pub fn with_fusion_improvement(mut self, factor: f64) -> Self {
        self.fusion_improvement = factor;
        self.recalculate();
        self
    }

    /// Apply FlashAttention optimization
    #[must_use]
    pub fn with_flash_attention_improvement(mut self, factor: f64) -> Self {
        self.flash_attention_improvement = factor;
        self.recalculate();
        self
    }

    /// Apply memory optimization
    #[must_use]
    pub fn with_memory_improvement(mut self, factor: f64) -> Self {
        self.memory_improvement = factor;
        self.recalculate();
        self
    }

    /// Recalculate throughput and milestone
    fn recalculate(&mut self) {
        let total_improvement = self.gemm_improvement
            * self.fusion_improvement
            * self.flash_attention_improvement
            * self.memory_improvement;

        self.optimized_tps = self.baseline_tps * total_improvement;
        self.gap_ratio = 240.0 / self.optimized_tps;

        self.milestone = if self.gap_ratio <= 1.25 {
            Some("M4".to_string()) // Full parity
        } else if self.gap_ratio <= 2.0 {
            Some("M3".to_string()) // Near parity
        } else if self.gap_ratio <= 5.0 {
            Some("M2".to_string()) // Within 5x
        } else {
            None
        };
    }

    /// Check if M3 target achieved (>48 tok/s, <5x gap)
    #[must_use]
    pub fn achieves_m3(&self) -> bool {
        self.optimized_tps >= 48.0 && self.gap_ratio <= 5.0
    }

    /// Check if M4 target achieved (>192 tok/s, <1.25x gap)
    #[must_use]
    pub fn achieves_m4(&self) -> bool {
        self.optimized_tps >= 192.0 && self.gap_ratio <= 1.25
    }

    /// Get combined improvement factor
    #[must_use]
    pub fn total_improvement(&self) -> f64 {
        self.optimized_tps / self.baseline_tps
    }
}

// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // GpuParityBenchmark Tests
    // =========================================================================

    #[test]
    fn test_gpu_parity_benchmark_default() {
        let bench = GpuParityBenchmark::default();
        assert!(bench.model_path.is_empty());
        assert_eq!(bench.prompt, "The quick brown fox");
        assert_eq!(bench.max_tokens, 32);
        assert_eq!(bench.ollama_endpoint, "http://localhost:11434");
        assert_eq!(bench.warmup_iterations, 3);
        assert_eq!(bench.measurement_iterations, 10);
    }

    #[test]
    fn test_gpu_parity_benchmark_new() {
        let bench = GpuParityBenchmark::new("/path/to/model.gguf");
        assert_eq!(bench.model_path, "/path/to/model.gguf");
    }

    #[test]
    fn test_gpu_parity_benchmark_builder() {
        let bench = GpuParityBenchmark::new("model.gguf")
            .with_prompt("Test prompt")
            .with_max_tokens(64)
            .with_ollama_endpoint("http://other:11434")
            .with_warmup(5)
            .with_iterations(20);

        assert_eq!(bench.prompt, "Test prompt");
        assert_eq!(bench.max_tokens, 64);
        assert_eq!(bench.ollama_endpoint, "http://other:11434");
        assert_eq!(bench.warmup_iterations, 5);
        assert_eq!(bench.measurement_iterations, 20);
    }

    // =========================================================================
    // GpuParityResult Tests
    // =========================================================================

    #[test]
    fn test_gpu_parity_result_new() {
        let result = GpuParityResult::new(120.0, 240.0, 0.03, "RTX 4090", 2000);
        assert!((result.realizar_gpu_tps - 120.0).abs() < 0.01);
        assert!((result.ollama_tps - 240.0).abs() < 0.01);
        assert!((result.gap_ratio - 2.0).abs() < 0.01);
        assert_eq!(result.gpu_device, "RTX 4090");
        assert_eq!(result.vram_mb, 2000);
    }

    #[test]
    fn test_gpu_parity_result_gap_ratio_zero_realizar() {
        let result = GpuParityResult::new(0.0, 240.0, 0.03, "GPU", 1000);
        assert!(result.gap_ratio.is_infinite());
    }

    #[test]
    fn test_gpu_parity_result_achieves_m2_parity() {
        let result = GpuParityResult::new(120.0, 240.0, 0.03, "GPU", 1000);
        assert!(result.achieves_m2_parity()); // 2.0x

        let result2 = GpuParityResult::new(100.0, 240.0, 0.03, "GPU", 1000);
        assert!(!result2.achieves_m2_parity()); // 2.4x
    }

    #[test]
    fn test_gpu_parity_result_achieves_m4_parity() {
        let result = GpuParityResult::new(200.0, 240.0, 0.03, "GPU", 1000);
        assert!(result.achieves_m4_parity()); // 1.2x

        let result2 = GpuParityResult::new(150.0, 240.0, 0.03, "GPU", 1000);
        assert!(!result2.achieves_m4_parity()); // 1.6x
    }

    #[test]
    fn test_gpu_parity_result_gpu_faster_than_cpu() {
        let result = GpuParityResult::new(10.0, 240.0, 0.03, "GPU", 1000);
        assert!(result.gpu_faster_than_cpu()); // 10 > 5

        let result2 = GpuParityResult::new(4.0, 240.0, 0.03, "GPU", 1000);
        assert!(!result2.gpu_faster_than_cpu()); // 4 <= 5
    }

    #[test]
    fn test_gpu_parity_result_measurements_stable() {
        let result = GpuParityResult::new(100.0, 240.0, 0.03, "GPU", 1000);
        assert!(result.measurements_stable()); // 0.03 < 0.05

        let result2 = GpuParityResult::new(100.0, 240.0, 0.06, "GPU", 1000);
        assert!(!result2.measurements_stable()); // 0.06 >= 0.05
    }

    #[test]
    fn test_gpu_parity_result_cpu_speedup() {
        let result = GpuParityResult::new(25.0, 240.0, 0.03, "GPU", 1000);
        assert!((result.cpu_speedup() - 5.0).abs() < 0.01); // 25 / 5 = 5x
    }

    // =========================================================================
    // FalsifiableClaim Tests
    // =========================================================================

    #[test]
    fn test_falsifiable_claim_new() {
        let claim = FalsifiableClaim::new("C1", "GPU > 5x CPU", 5.0, 25.0);
        assert_eq!(claim.id, "C1");
        assert_eq!(claim.description, "GPU > 5x CPU");
        assert!((claim.expected - 5.0).abs() < 0.01);
        assert!((claim.threshold - 25.0).abs() < 0.01);
        assert!(!claim.verified);
    }

    #[test]
    fn test_falsifiable_claim_evaluate_verified() {
        let claim = FalsifiableClaim::new("C1", "test", 5.0, 25.0).evaluate(30.0);
        assert!(claim.verified);
        assert!((claim.measured - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_falsifiable_claim_evaluate_not_verified() {
        let claim = FalsifiableClaim::new("C1", "test", 5.0, 25.0).evaluate(20.0);
        assert!(!claim.verified);
    }

    // =========================================================================
    // GapAnalysis Tests
    // =========================================================================

    #[test]
    fn test_gap_analysis_new() {
        let analysis = GapAnalysis::new(18.0, 10.0);
        assert!((analysis.claimed_gap - 18.0).abs() < 0.01);
        assert!((analysis.measured_gap - 10.0).abs() < 0.01);
        assert!(analysis.claims.is_empty());
    }

    #[test]
    fn test_gap_analysis_with_statistics() {
        let analysis = GapAnalysis::new(18.0, 10.0).with_statistics(0.01, 8.0, 12.0);
        assert!((analysis.p_value - 0.01).abs() < 0.001);
        assert!((analysis.ci_95_lower - 8.0).abs() < 0.01);
        assert!((analysis.ci_95_upper - 12.0).abs() < 0.01);
    }

    #[test]
    fn test_gap_analysis_calculate_popper_score() {
        let mut analysis = GapAnalysis::new(18.0, 10.0);
        analysis.add_claim(FalsifiableClaim::new("C1", "test", 5.0, 25.0).evaluate(30.0)); // verified
        analysis.add_claim(FalsifiableClaim::new("C2", "test", 5.0, 25.0).evaluate(20.0)); // not verified
        analysis.calculate_popper_score();
        assert!((analysis.popper_score - 50.0).abs() < 0.01); // 1/2 = 50%
    }

    #[test]
    fn test_gap_analysis_calculate_popper_score_empty() {
        let mut analysis = GapAnalysis::new(18.0, 10.0);
        analysis.calculate_popper_score();
        assert_eq!(analysis.popper_score, 0.0);
    }

    #[test]
    fn test_gap_analysis_claim_verified() {
        let analysis = GapAnalysis::new(18.0, 10.0).with_statistics(0.01, 8.0, 12.0);
        assert!(analysis.claim_verified()); // 10 is within [8, 12]

        let analysis2 = GapAnalysis::new(18.0, 15.0).with_statistics(0.01, 8.0, 12.0);
        assert!(!analysis2.claim_verified()); // 15 is outside [8, 12]
    }

    #[test]
    fn test_gap_analysis_with_default_claims() {
        let analysis = GapAnalysis::new(18.0, 10.0).with_default_claims(30.0);
        assert_eq!(analysis.claims.len(), 4);
        // Claim IMP-800c-1: threshold 25 tok/s, measured 30 -> verified
        assert!(analysis.claims[0].verified);
        // Claim IMP-800c-2: threshold 24 tok/s, measured 30 -> verified
        assert!(analysis.claims[1].verified);
    }

    // =========================================================================
    // OptimizedGemmConfig Tests
    // =========================================================================

    #[test]
    fn test_optimized_gemm_config_default() {
        let config = OptimizedGemmConfig::default();
        assert_eq!(config.tile_size, 32);
        assert_eq!(config.reg_block, 4);
        assert!(!config.use_tensor_cores);
        assert_eq!(config.vector_width, 4);
        assert!(config.double_buffer);
    }

    #[test]
    fn test_optimized_gemm_config_small() {
        let config = OptimizedGemmConfig::small();
        assert_eq!(config.tile_size, 16);
        assert_eq!(config.reg_block, 2);
        assert!(!config.double_buffer);
    }

    #[test]
    fn test_optimized_gemm_config_large() {
        let config = OptimizedGemmConfig::large();
        assert_eq!(config.tile_size, 64);
        assert_eq!(config.reg_block, 8);
        assert!(config.double_buffer);
    }

    #[test]
    fn test_optimized_gemm_config_shared_memory_bytes() {
        let config = OptimizedGemmConfig::default();
        // 32 * 32 * 4 = 4096 bytes per tile
        // 2 tiles * 2 buffers = 4 * 4096 = 16384
        assert_eq!(config.shared_memory_bytes(), 16384);

        let config_no_double = OptimizedGemmConfig::small();
        // 16 * 16 * 4 = 1024 bytes per tile
        // 2 tiles = 2048
        assert_eq!(config_no_double.shared_memory_bytes(), 2048);
    }

    #[test]
    fn test_optimized_gemm_config_threads_per_block() {
        let config = OptimizedGemmConfig::default();
        // 32 / 4 = 8 threads per dim
        // 8 * 8 = 64 threads
        assert_eq!(config.threads_per_block(), 64);
    }

    #[test]
    fn test_optimized_gemm_config_registers_per_thread() {
        let config = OptimizedGemmConfig::default();
        // 4 * 4 = 16 registers
        assert_eq!(config.registers_per_thread(), 16);
    }

    // =========================================================================
    // GemmPerformanceResult Tests
    // =========================================================================

    #[test]
    fn test_gemm_performance_result_new() {
        let result = GemmPerformanceResult::new(1024, 1024, 1024, 10.0);
        // ops = 2 * 1024^3 = 2147483648
        // gflops = 2147483648 / (10 * 1e6) = 214.7
        assert!(result.gflops > 200.0);
        assert!(result.time_ms > 0.0);
    }

    #[test]
    fn test_gemm_performance_result_with_peak() {
        let result = GemmPerformanceResult::new(1024, 1024, 1024, 10.0).with_peak(300.0);
        // efficiency = (gflops / 300) * 100
        assert!(result.efficiency > 0.0 && result.efficiency <= 100.0);
    }

    #[test]
    fn test_gemm_performance_result_improved_by() {
        let result = GemmPerformanceResult::new(1024, 1024, 1024, 10.0);
        assert!(result.improved_by(100.0, 2.0)); // 214+ >= 200
        assert!(!result.improved_by(100.0, 10.0)); // 214 < 1000
    }

    // =========================================================================
    // OptimizedGemmBenchmark Tests
    // =========================================================================

    #[test]
    fn test_optimized_gemm_benchmark_default() {
        let bench = OptimizedGemmBenchmark::default();
        assert_eq!(bench.warmup_iterations, 5);
        assert_eq!(bench.measurement_iterations, 20);
    }

    #[test]
    fn test_optimized_gemm_benchmark_with_config() {
        let config = OptimizedGemmConfig::large();
        let bench = OptimizedGemmBenchmark::with_config(config);
        assert_eq!(bench.config.tile_size, 64);
    }

    #[test]
    fn test_optimized_gemm_benchmark_expected_improvement() {
        let bench = OptimizedGemmBenchmark::default();
        let improvement = bench.expected_improvement();
        // With default config: 2 * 1.5 * 1.3 * 1.2 = 4.68
        assert!(improvement > 4.0 && improvement < 5.0);
    }

    // =========================================================================
    // FusedOpType and FusedOpSpec Tests
    // =========================================================================

    #[test]
    fn test_fused_op_type_eq() {
        assert_eq!(
            FusedOpType::GemmBiasActivation,
            FusedOpType::GemmBiasActivation
        );
        assert_ne!(FusedOpType::FusedFfn, FusedOpType::FusedAttention);
    }

    #[test]
    fn test_fused_op_spec_launch_reduction() {
        let spec = FusedOpSpec {
            op_type: FusedOpType::GemmBiasActivation,
            input_dims: vec![1024, 1024],
            output_dims: vec![1024, 1024],
            activation: Some("relu".to_string()),
            fused_launches: 1,
            unfused_launches: 3,
        };
        assert!((spec.launch_reduction() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_fused_op_spec_achieves_target_reduction() {
        let spec = FusedOpSpec {
            op_type: FusedOpType::FusedFfn,
            input_dims: vec![1024],
            output_dims: vec![1024],
            activation: None,
            fused_launches: 1,
            unfused_launches: 3,
        };
        assert!(spec.achieves_target_reduction()); // 3x >= 2x

        let spec2 = FusedOpSpec {
            op_type: FusedOpType::FusedFfn,
            input_dims: vec![1024],
            output_dims: vec![1024],
            activation: None,
            fused_launches: 2,
            unfused_launches: 3,
        };
        assert!(!spec2.achieves_target_reduction()); // 1.5x < 2x
    }

    // =========================================================================
    // FlashAttentionConfig Tests
    // =========================================================================

    #[test]
    fn test_flash_attention_config_phi2() {
        let config = FlashAttentionConfig::phi2();
        assert_eq!(config.block_size_q, 64);
        assert_eq!(config.head_dim, 80);
        assert_eq!(config.num_heads, 32);
        assert!(config.causal);
    }

    #[test]
    fn test_flash_attention_config_memory_comparison() {
        let config = FlashAttentionConfig::phi2();
        let (naive, flash) = config.memory_comparison(1024);
        // naive = 1024 * 1024 * 4 = 4MB
        assert_eq!(naive, 4 * 1024 * 1024);
        // flash = 64 * 64 * 4 * 2 = 32KB
        assert!(flash < naive);
    }

    #[test]
    fn test_flash_attention_config_memory_savings() {
        let config = FlashAttentionConfig::phi2();
        let savings = config.memory_savings(2048);
        // Should be significant for large sequences
        assert!(savings > 100.0);
    }

    // =========================================================================
    // MemoryPoolConfig Tests
    // =========================================================================

    #[test]
    fn test_memory_pool_config_default() {
        let config = MemoryPoolConfig::default();
        assert_eq!(config.initial_size, 256 * 1024 * 1024);
        assert!(config.use_pinned_memory);
        assert!(config.async_transfers);
    }

    #[test]
    fn test_memory_pool_config_find_size_class() {
        let config = MemoryPoolConfig::default();
        assert_eq!(config.find_size_class(1000), Some(4096));
        assert_eq!(config.find_size_class(5000), Some(16384));
        assert_eq!(config.find_size_class(100_000_000), Some(268_435_456));
        assert_eq!(config.find_size_class(500_000_000), None); // Larger than max class
    }

    #[test]
    fn test_memory_pool_config_expected_bandwidth_improvement() {
        let config = MemoryPoolConfig::default();
        assert!((config.expected_bandwidth_improvement() - 2.4).abs() < 0.01);

        let mut config_no_pinned = MemoryPoolConfig::default();
        config_no_pinned.use_pinned_memory = false;
        assert!((config_no_pinned.expected_bandwidth_improvement() - 1.0).abs() < 0.01);
    }

    // =========================================================================
    // Imp900Result Tests
    // =========================================================================

    #[test]
    fn test_imp900_result_from_baseline() {
        let result = Imp900Result::from_baseline(13.1);
        assert!((result.baseline_tps - 13.1).abs() < 0.01);
        assert!((result.optimized_tps - 13.1).abs() < 0.01);
        assert!(result.milestone.is_none());
    }

    #[test]
    fn test_imp900_result_with_improvements() {
        let result = Imp900Result::from_baseline(13.1)
            .with_gemm_improvement(2.0)
            .with_fusion_improvement(1.5)
            .with_flash_attention_improvement(1.3)
            .with_memory_improvement(1.2);

        // 13.1 * 2 * 1.5 * 1.3 * 1.2 = 61.2
        assert!(result.optimized_tps > 60.0);
        assert_eq!(result.milestone, Some("M2".to_string())); // Within 5x of 240
    }

    #[test]
    fn test_imp900_result_achieves_m3() {
        let result = Imp900Result::from_baseline(13.1).with_gemm_improvement(4.0);
        // 13.1 * 4 = 52.4 tok/s, gap = 240/52.4 = 4.6x
        assert!(result.achieves_m3());
    }

    #[test]
    fn test_imp900_result_achieves_m4() {
        let result = Imp900Result::from_baseline(13.1)
            .with_gemm_improvement(5.0)
            .with_fusion_improvement(3.0);
        // 13.1 * 5 * 3 = 196.5 tok/s, gap = 240/196.5 = 1.22x
        assert!(result.achieves_m4());
    }

    #[test]
    fn test_imp900_result_total_improvement() {
        let result = Imp900Result::from_baseline(10.0)
            .with_gemm_improvement(2.0)
            .with_fusion_improvement(2.0);
        // optimized = 10 * 2 * 2 = 40
        assert!((result.total_improvement() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_imp900_result_milestones() {
        // from_baseline doesn't calculate milestone; use with_* to trigger recalculate

        // No milestone: gap > 5x (10 tok/s -> gap = 24x)
        let result1 = Imp900Result::from_baseline(10.0).with_gemm_improvement(1.0);
        assert!(result1.milestone.is_none());

        // M2: gap <= 5x (50 tok/s -> gap = 4.8x)
        let result2 = Imp900Result::from_baseline(50.0).with_gemm_improvement(1.0);
        assert_eq!(result2.milestone, Some("M2".to_string()));

        // M3: gap <= 2x (130 tok/s -> gap = 1.85x)
        let result3 = Imp900Result::from_baseline(130.0).with_gemm_improvement(1.0);
        assert_eq!(result3.milestone, Some("M3".to_string()));

        // M4: gap <= 1.25x (200 tok/s -> gap = 1.2x)
        let result4 = Imp900Result::from_baseline(200.0).with_gemm_improvement(1.0);
        assert_eq!(result4.milestone, Some("M4".to_string()));
    }
}
