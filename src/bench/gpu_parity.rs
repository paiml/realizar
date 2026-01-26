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
