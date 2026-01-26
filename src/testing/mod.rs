//! Model Fixture Testing Infrastructure
//!
//! PyTorch-inspired test fixture pattern for cross-format, cross-device testing.
//! Implements the patterns described in `docs/specifications/model-fixture-setup-teardown.md`.
//!
//! # Design
//!
//! ```text
//! ModelConfig → ConstructorInput → ModelFixture → ForwardInput → Output
//!                                       ↓
//!                              convert_to(format)
//!                                       ↓
//!                              ModelFixture (new format)
//! ```
//!
//! # References
//!
//! - PyTorch: `torch/testing/_internal/common_modules.py`
//! - Kuhn et al., "Combinatorial Testing", NIST SP 800-142

use std::fmt;

pub mod fixtures;
pub mod generators;

#[cfg(test)]
pub mod combinatorial_tests;

#[cfg(test)]
pub mod popperian_tests;

/// Model format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    /// PyTorch checkpoint (.pt, .pth, .bin)
    PyTorch,
    /// GGML Universal Format (.gguf)
    GGUF,
    /// Anthropic Production Runtime (.apr)
    APR,
    /// HuggingFace Safetensors (.safetensors)
    Safetensors,
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelFormat::PyTorch => write!(f, "PyTorch"),
            ModelFormat::GGUF => write!(f, "GGUF"),
            ModelFormat::APR => write!(f, "APR"),
            ModelFormat::Safetensors => write!(f, "Safetensors"),
        }
    }
}

/// Execution device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU execution
    Cpu,
    /// CUDA GPU execution with device ordinal
    Cuda(u32),
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "CPU"),
            Device::Cuda(id) => write!(f, "CUDA:{}", id),
        }
    }
}

impl Device {
    /// Check if this is a CUDA device
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    /// Get CUDA device ID if applicable
    pub fn cuda_id(&self) -> Option<u32> {
        match self {
            Device::Cuda(id) => Some(*id),
            Device::Cpu => None,
        }
    }
}

/// Quantization type for weights
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantType {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point
    F16,
    /// Brain floating point 16
    BF16,
    /// 8-bit quantization (GGML Q8_0)
    Q8_0,
    /// 4-bit quantization (GGML Q4_0)
    Q4_0,
    /// 4-bit K-quant (GGML Q4_K)
    Q4_K,
    /// 5-bit K-quant (GGML Q5_K)
    Q5_K,
    /// 6-bit K-quant (GGML Q6_K)
    Q6_K,
}

impl QuantType {
    /// Bits per weight element
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            QuantType::F32 => 32.0,
            QuantType::F16 => 16.0,
            QuantType::BF16 => 16.0,
            QuantType::Q8_0 => 8.5,  // 8 bits + scale overhead
            QuantType::Q4_0 => 4.5,
            QuantType::Q4_K => 4.5,
            QuantType::Q5_K => 5.5,
            QuantType::Q6_K => 6.5,
        }
    }

    /// Check if supported by format
    pub fn supported_by(&self, format: ModelFormat) -> bool {
        match format {
            ModelFormat::PyTorch => matches!(self, QuantType::F32 | QuantType::F16 | QuantType::BF16),
            ModelFormat::GGUF => true, // GGUF supports all
            ModelFormat::APR => true,  // APR supports all
            ModelFormat::Safetensors => matches!(self, QuantType::F32 | QuantType::F16 | QuantType::BF16),
        }
    }
}

/// Model configuration for fixture generation
///
/// Inspired by PyTorch's `ModuleInfo` pattern.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Hidden dimension (embedding size)
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads (Q)
    pub num_heads: usize,
    /// Number of KV heads (for GQA, <= num_heads)
    pub num_kv_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// RoPE theta value
    pub rope_theta: f32,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::tiny()
    }
}

impl ModelConfig {
    /// Minimal config for fast unit tests (~1ms)
    ///
    /// 2 layers, 64 hidden, 4 heads with GQA (2 KV heads)
    pub fn tiny() -> Self {
        Self {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,  // GQA: 2:1 ratio
            vocab_size: 256,
            intermediate_dim: 128,
            rope_theta: 10000.0,
            max_seq_len: 32,
            rms_norm_eps: 1e-5,
        }
    }

    /// Small config for integration tests (~10ms)
    ///
    /// 4 layers, 256 hidden, 8 heads with GQA (2 KV heads)
    pub fn small() -> Self {
        Self {
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 8,
            num_kv_heads: 2,  // GQA: 4:1 ratio
            vocab_size: 1024,
            intermediate_dim: 512,
            rope_theta: 10000.0,
            max_seq_len: 128,
            rms_norm_eps: 1e-5,
        }
    }

    /// Medium config matching TinyLlama architecture
    ///
    /// 22 layers, 2048 hidden, 32 heads with GQA (4 KV heads)
    pub fn tinyllama() -> Self {
        Self {
            hidden_dim: 2048,
            num_layers: 22,
            num_heads: 32,
            num_kv_heads: 4,  // GQA: 8:1 ratio
            vocab_size: 32000,
            intermediate_dim: 5632,
            rope_theta: 10000.0,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
        }
    }

    /// Config matching Qwen 1.5B architecture
    ///
    /// 28 layers, 1536 hidden, 12 heads with GQA (2 KV heads)
    pub fn qwen_1_5b() -> Self {
        Self {
            hidden_dim: 1536,
            num_layers: 28,
            num_heads: 12,
            num_kv_heads: 2,  // GQA: 6:1 ratio
            vocab_size: 151936,
            intermediate_dim: 8960,
            rope_theta: 1000000.0,
            max_seq_len: 32768,
            rms_norm_eps: 1e-6,
        }
    }

    /// Compute head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }

    /// Compute Q projection dimension
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim()
    }

    /// Compute K projection dimension (may differ from Q for GQA)
    pub fn k_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim()
    }

    /// Compute V projection dimension (same as K)
    pub fn v_dim(&self) -> usize {
        self.k_dim()
    }

    /// GQA group size (how many Q heads share one KV head)
    pub fn gqa_group_size(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Check if using grouped query attention
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads
    }

    /// Check if using multi-query attention (single KV head)
    pub fn is_mqa(&self) -> bool {
        self.num_kv_heads == 1
    }

    /// Estimated parameter count
    pub fn param_count(&self) -> usize {
        let embed = self.vocab_size * self.hidden_dim;
        let per_layer = {
            // Attention: Q, K, V, O projections
            let q = self.hidden_dim * self.q_dim();
            let k = self.hidden_dim * self.k_dim();
            let v = self.hidden_dim * self.v_dim();
            let o = self.q_dim() * self.hidden_dim;
            // FFN: gate, up, down
            let gate = self.hidden_dim * self.intermediate_dim;
            let up = self.hidden_dim * self.intermediate_dim;
            let down = self.intermediate_dim * self.hidden_dim;
            // Norms
            let norms = 2 * self.hidden_dim;
            q + k + v + o + gate + up + down + norms
        };
        let output_norm = self.hidden_dim;
        let lm_head = self.hidden_dim * self.vocab_size;

        embed + (self.num_layers * per_layer) + output_norm + lm_head
    }
}

/// Constructor input (PyTorch `FunctionInput` equivalent)
#[derive(Debug, Clone)]
pub struct ConstructorInput {
    /// Model configuration
    pub config: ModelConfig,
    /// Quantization type (None = F32)
    pub quantization: Option<QuantType>,
    /// Seed for deterministic weight generation
    pub weights_seed: u64,
}

impl ConstructorInput {
    /// Create with default F32 weights
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            quantization: None,
            weights_seed: 42,
        }
    }

    /// Create with specific quantization
    pub fn with_quant(config: ModelConfig, quant: QuantType, seed: u64) -> Self {
        Self {
            config,
            quantization: Some(quant),
            weights_seed: seed,
        }
    }
}

/// Forward pass input
#[derive(Debug, Clone)]
pub struct ForwardInput {
    /// Input token IDs
    pub tokens: Vec<u32>,
    /// Starting position for KV cache
    pub position: usize,
}

impl ForwardInput {
    /// Create simple forward input
    pub fn new(tokens: Vec<u32>) -> Self {
        Self { tokens, position: 0 }
    }

    /// Create with specific position (for KV cache testing)
    pub fn at_position(tokens: Vec<u32>, position: usize) -> Self {
        Self { tokens, position }
    }

    /// Sequence length
    pub fn seq_len(&self) -> usize {
        self.tokens.len()
    }
}

/// Complete test case (PyTorch `ModuleInput` equivalent)
#[derive(Clone)]
pub struct ModelTestCase {
    /// Human-readable description
    pub desc: String,
    /// Constructor arguments
    pub constructor: ConstructorInput,
    /// Forward pass arguments
    pub forward: ForwardInput,
    /// Expected L2 norm of output (for quick validation)
    pub expected_output_norm: Option<f32>,
    /// Source format
    pub source_format: ModelFormat,
    /// Target format (for conversion tests)
    pub target_format: Option<ModelFormat>,
    /// Device to run on
    pub device: Device,
}

impl fmt::Debug for ModelTestCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelTestCase")
            .field("desc", &self.desc)
            .field("source_format", &self.source_format)
            .field("target_format", &self.target_format)
            .field("device", &self.device)
            .field("config", &format!("{}L/{}H",
                self.constructor.config.num_layers,
                self.constructor.config.num_heads))
            .finish()
    }
}

impl ModelTestCase {
    /// Create a simple test case
    pub fn new(
        desc: impl Into<String>,
        config: ModelConfig,
        format: ModelFormat,
        device: Device,
    ) -> Self {
        Self {
            desc: desc.into(),
            constructor: ConstructorInput::new(config),
            forward: ForwardInput::new(vec![1, 2, 3, 4]),
            expected_output_norm: None,
            source_format: format,
            target_format: None,
            device,
        }
    }

    /// Create a conversion test case
    pub fn conversion(
        desc: impl Into<String>,
        config: ModelConfig,
        source: ModelFormat,
        target: ModelFormat,
        device: Device,
    ) -> Self {
        Self {
            desc: desc.into(),
            constructor: ConstructorInput::new(config),
            forward: ForwardInput::new(vec![1, 2, 3, 4]),
            expected_output_norm: None,
            source_format: source,
            target_format: Some(target),
            device,
        }
    }

    /// Set quantization type
    pub fn with_quant(mut self, quant: QuantType) -> Self {
        self.constructor.quantization = Some(quant);
        self
    }

    /// Set forward input tokens
    pub fn with_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.forward.tokens = tokens;
        self
    }

    /// Set expected output norm for validation
    pub fn with_expected_norm(mut self, norm: f32) -> Self {
        self.expected_output_norm = Some(norm);
        self
    }

    /// Check if this is a conversion test
    pub fn is_conversion_test(&self) -> bool {
        self.target_format.is_some()
    }
}

/// Result of running a model fixture test
#[derive(Debug)]
pub struct TestResult {
    /// Test case that was run
    pub test_case: String,
    /// Whether the test passed
    pub passed: bool,
    /// Output logits (if successful)
    pub output: Option<Vec<f32>>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Execution time in microseconds
    pub duration_us: u64,
    /// Memory usage in bytes
    pub memory_bytes: Option<usize>,
}

impl TestResult {
    /// Create a successful result
    pub fn success(test_case: &str, output: Vec<f32>, duration_us: u64) -> Self {
        Self {
            test_case: test_case.to_string(),
            passed: true,
            output: Some(output),
            error: None,
            duration_us,
            memory_bytes: None,
        }
    }

    /// Create a failure result
    pub fn failure(test_case: &str, error: impl Into<String>, duration_us: u64) -> Self {
        Self {
            test_case: test_case.to_string(),
            passed: false,
            output: None,
            error: Some(error.into()),
            duration_us,
            memory_bytes: None,
        }
    }

    /// Compute L2 norm of output
    pub fn output_l2_norm(&self) -> Option<f32> {
        self.output.as_ref().map(|o| {
            o.iter().map(|x| x * x).sum::<f32>().sqrt()
        })
    }
}

/// Tolerance thresholds for numerical comparisons
#[derive(Debug, Clone, Copy)]
pub struct Tolerances {
    /// Absolute tolerance for F32 comparisons
    pub f32_abs: f32,
    /// Relative tolerance for F32 comparisons
    pub f32_rel: f32,
    /// Tolerance for quantized weight comparisons (% L2)
    pub quant_l2_pct: f32,
    /// Tolerance for CPU vs CUDA parity
    pub device_parity: f32,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            f32_abs: 1e-5,
            f32_rel: 1e-4,
            quant_l2_pct: 5.0,
            device_parity: 1e-3,
        }
    }
}

impl Tolerances {
    /// Strict tolerances for exact comparisons
    pub fn strict() -> Self {
        Self {
            f32_abs: 1e-6,
            f32_rel: 1e-5,
            quant_l2_pct: 1.0,
            device_parity: 1e-4,
        }
    }

    /// Relaxed tolerances for quantized models
    pub fn quantized() -> Self {
        Self {
            f32_abs: 1e-3,
            f32_rel: 1e-2,
            quant_l2_pct: 10.0,
            device_parity: 5e-2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_tiny() {
        let config = ModelConfig::tiny();
        assert_eq!(config.hidden_dim, 64);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.head_dim(), 16);
        assert_eq!(config.gqa_group_size(), 2);
        assert!(config.is_gqa());
        assert!(!config.is_mqa());
    }

    #[test]
    fn test_model_config_qwen() {
        let config = ModelConfig::qwen_1_5b();
        assert_eq!(config.hidden_dim, 1536);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.gqa_group_size(), 6);
        assert!(config.is_gqa());
    }

    #[test]
    fn test_model_config_dimensions() {
        let config = ModelConfig::small();
        assert_eq!(config.q_dim(), 256); // 8 heads * 32 head_dim
        assert_eq!(config.k_dim(), 64);  // 2 kv_heads * 32 head_dim
        assert_eq!(config.v_dim(), 64);
    }

    #[test]
    fn test_quant_type_bits() {
        assert_eq!(QuantType::F32.bits_per_weight(), 32.0);
        assert_eq!(QuantType::Q4_K.bits_per_weight(), 4.5);
    }

    #[test]
    fn test_quant_type_format_support() {
        assert!(QuantType::F32.supported_by(ModelFormat::Safetensors));
        assert!(!QuantType::Q4_K.supported_by(ModelFormat::Safetensors));
        assert!(QuantType::Q4_K.supported_by(ModelFormat::GGUF));
        assert!(QuantType::Q4_K.supported_by(ModelFormat::APR));
    }

    #[test]
    fn test_model_test_case_creation() {
        let tc = ModelTestCase::new(
            "tiny CPU test",
            ModelConfig::tiny(),
            ModelFormat::GGUF,
            Device::Cpu,
        );
        assert_eq!(tc.desc, "tiny CPU test");
        assert!(!tc.is_conversion_test());
    }

    #[test]
    fn test_model_test_case_conversion() {
        let tc = ModelTestCase::conversion(
            "GGUF to APR",
            ModelConfig::tiny(),
            ModelFormat::GGUF,
            ModelFormat::APR,
            Device::Cpu,
        );
        assert!(tc.is_conversion_test());
        assert_eq!(tc.source_format, ModelFormat::GGUF);
        assert_eq!(tc.target_format, Some(ModelFormat::APR));
    }

    #[test]
    fn test_device_display() {
        assert_eq!(format!("{}", Device::Cpu), "CPU");
        assert_eq!(format!("{}", Device::Cuda(0)), "CUDA:0");
    }

    #[test]
    fn test_format_display() {
        assert_eq!(format!("{}", ModelFormat::GGUF), "GGUF");
        assert_eq!(format!("{}", ModelFormat::APR), "APR");
    }

    #[test]
    fn test_param_count() {
        let config = ModelConfig::tiny();
        let params = config.param_count();
        // Rough estimate: should be in reasonable range for tiny model
        assert!(params > 10_000, "params={}", params);
        assert!(params < 1_000_000, "params={}", params);
    }

    #[test]
    fn test_test_result_l2_norm() {
        let result = TestResult::success("test", vec![3.0, 4.0], 100);
        let norm = result.output_l2_norm().unwrap();
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_device_is_cuda() {
        assert!(!Device::Cpu.is_cuda());
        assert!(Device::Cuda(0).is_cuda());
        assert_eq!(Device::Cpu.cuda_id(), None);
        assert_eq!(Device::Cuda(7).cuda_id(), Some(7));
    }

    #[test]
    fn test_model_config_tinyllama() {
        let config = ModelConfig::tinyllama();
        assert_eq!(config.hidden_dim, 2048);
        assert_eq!(config.num_layers, 22);
    }

    #[test]
    fn test_model_config_default() {
        let default_config = ModelConfig::default();
        let tiny_config = ModelConfig::tiny();
        assert_eq!(default_config.hidden_dim, tiny_config.hidden_dim);
    }

    #[test]
    fn test_quant_type_all_bits() {
        assert_eq!(QuantType::F16.bits_per_weight(), 16.0);
        assert_eq!(QuantType::BF16.bits_per_weight(), 16.0);
        assert_eq!(QuantType::Q8_0.bits_per_weight(), 8.5);
        assert_eq!(QuantType::Q4_0.bits_per_weight(), 4.5);
        assert_eq!(QuantType::Q5_K.bits_per_weight(), 5.5);
        assert_eq!(QuantType::Q6_K.bits_per_weight(), 6.5);
    }

    #[test]
    fn test_quant_type_unsupported() {
        assert!(!QuantType::Q8_0.supported_by(ModelFormat::PyTorch));
        assert!(!QuantType::Q8_0.supported_by(ModelFormat::Safetensors));
    }

    #[test]
    fn test_forward_input_seq_len() {
        let input = ForwardInput::new(vec![1, 2, 3]);
        assert_eq!(input.seq_len(), 3);
        assert_eq!(input.position, 0);

        let input_pos = ForwardInput::at_position(vec![4, 5], 10);
        assert_eq!(input_pos.seq_len(), 2);
        assert_eq!(input_pos.position, 10);
    }

    #[test]
    fn test_model_test_case_builder() {
        let tc = ModelTestCase::new("test", ModelConfig::tiny(), ModelFormat::APR, Device::Cpu)
            .with_quant(QuantType::Q4_K)
            .with_tokens(vec![1, 2])
            .with_expected_norm(10.0);

        assert_eq!(tc.constructor.quantization, Some(QuantType::Q4_K));
        assert_eq!(tc.forward.tokens, vec![1, 2]);
        assert_eq!(tc.expected_output_norm, Some(10.0));
    }

    #[test]
    fn test_test_result_failure() {
        let result = TestResult::failure("fail test", "error message", 50);
        assert!(!result.passed);
        assert_eq!(result.test_case, "fail test");
        assert_eq!(result.error.unwrap(), "error message");
        assert!(result.output.is_none());
    }

    #[test]
    fn test_tolerances() {
        let default = Tolerances::default();
        let strict = Tolerances::strict();
        let quantized = Tolerances::quantized();

        assert!(strict.f32_abs < default.f32_abs);
        assert!(quantized.f32_abs > default.f32_abs);
    }

    #[test]
    fn test_format_display_pytorch_safetensors() {
        assert_eq!(format!("{}", ModelFormat::PyTorch), "PyTorch");
        assert_eq!(format!("{}", ModelFormat::Safetensors), "Safetensors");
    }

    #[test]
    fn test_constructor_input_with_quant() {
        let config = ModelConfig::tiny();
        let ci = ConstructorInput::with_quant(config, QuantType::Q8_0, 123);
        assert_eq!(ci.quantization, Some(QuantType::Q8_0));
        assert_eq!(ci.weights_seed, 123);
    }
}
