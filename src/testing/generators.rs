//! Synthetic Weight Generators
//!
//! Deterministic weight generation for reproducible test fixtures.
//! Based on PyTorch's `make_tensor` pattern.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use half::f16;

use super::{ModelConfig, QuantType};

/// Deterministic weight generator for reproducible tests
///
/// # Example
///
/// ```rust
/// use realizar::testing::generators::SyntheticWeightGenerator;
///
/// let gen = SyntheticWeightGenerator::new(42);
/// let weights = gen.generate_f32(&[64, 64]); // 64x64 matrix
/// assert_eq!(weights.len(), 64 * 64);
/// ```
pub struct SyntheticWeightGenerator {
    seed: u64,
}

impl SyntheticWeightGenerator {
    /// Create generator with specific seed
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate F32 weights with Xavier initialization scale
    ///
    /// Scale = 1 / sqrt(fan_in) where fan_in is the last dimension
    pub fn generate_f32(&self, shape: &[usize]) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let n: usize = shape.iter().product();
        let fan_in = *shape.last().unwrap_or(&1);
        let scale = 1.0 / (fan_in as f32).sqrt();

        (0..n)
            .map(|_| rng.gen_range(-scale..scale))
            .collect()
    }

    /// Generate F32 weights with specific scale
    pub fn generate_f32_scaled(&self, shape: &[usize], scale: f32) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let n: usize = shape.iter().product();

        (0..n)
            .map(|_| rng.gen_range(-scale..scale))
            .collect()
    }

    /// Generate F16 weights
    pub fn generate_f16(&self, shape: &[usize]) -> Vec<f16> {
        self.generate_f32(shape)
            .into_iter()
            .map(f16::from_f32)
            .collect()
    }

    /// Generate Q4_0 quantized weights (GGML format)
    ///
    /// Block format: [f16 scale][16 bytes of 4-bit values]
    /// 32 values per block, 18 bytes per block
    pub fn generate_q4_0(&self, num_elements: usize) -> Vec<u8> {
        let num_blocks = num_elements.div_ceil(32);
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut data = Vec::with_capacity(num_blocks * 18);

        for _ in 0..num_blocks {
            // Scale (f16) - random in reasonable range
            let scale = f16::from_f32(rng.gen_range(0.01..0.1));
            data.extend_from_slice(&scale.to_le_bytes());

            // 16 bytes of quantized values (32 4-bit values)
            for _ in 0..16 {
                let lo = rng.gen_range(0u8..16);
                let hi = rng.gen_range(0u8..16);
                data.push((hi << 4) | lo);
            }
        }

        data
    }

    /// Generate Q8_0 quantized weights (GGML format)
    ///
    /// Block format: [f16 scale][32 int8 values]
    /// 32 values per block, 34 bytes per block
    pub fn generate_q8_0(&self, num_elements: usize) -> Vec<u8> {
        let num_blocks = num_elements.div_ceil(32);
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut data = Vec::with_capacity(num_blocks * 34);

        for _ in 0..num_blocks {
            // Scale (f16)
            let scale = f16::from_f32(rng.gen_range(0.01..0.1));
            data.extend_from_slice(&scale.to_le_bytes());

            // 32 int8 values
            for _ in 0..32 {
                let val: i8 = rng.gen_range(-127..127);
                data.push(val as u8);
            }
        }

        data
    }

    /// Generate Q4_K quantized weights (GGML K-quant format)
    ///
    /// More complex super-block structure for better accuracy
    pub fn generate_q4_k(&self, num_elements: usize) -> Vec<u8> {
        // Q4_K uses 256-element super-blocks
        let num_super_blocks = num_elements.div_ceil(256);
        let mut rng = StdRng::seed_from_u64(self.seed);

        // Q4_K block is 144 bytes for 256 elements
        let mut data = Vec::with_capacity(num_super_blocks * 144);

        for _ in 0..num_super_blocks {
            // d (f16) - super-block scale
            let d = f16::from_f32(rng.gen_range(0.01..0.1));
            data.extend_from_slice(&d.to_le_bytes());

            // dmin (f16) - minimum scale
            let dmin = f16::from_f32(rng.gen_range(0.001..0.01));
            data.extend_from_slice(&dmin.to_le_bytes());

            // scales (12 bytes) - 8 6-bit scales packed
            for _ in 0..12 {
                data.push(rng.gen_range(0u8..64));
            }

            // qs (128 bytes) - 4-bit quantized values
            for _ in 0..128 {
                let lo = rng.gen_range(0u8..16);
                let hi = rng.gen_range(0u8..16);
                data.push((hi << 4) | lo);
            }
        }

        data
    }

    /// Generate Q5_0 quantized weights
    ///
    /// Block format: [f16 scale][4 bytes high bits][16 bytes low nibbles]
    /// 32 values per block, 22 bytes per block
    pub fn generate_q5_0(&self, num_elements: usize) -> Vec<u8> {
        let num_blocks = num_elements.div_ceil(32);
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut data = Vec::with_capacity(num_blocks * 22);

        for _ in 0..num_blocks {
            // Scale (f16)
            let scale = f16::from_f32(rng.gen_range(0.01..0.1));
            data.extend_from_slice(&scale.to_le_bytes());

            // qh (4 bytes) - high bits for 32 values
            for _ in 0..4 {
                data.push(rng.gen_range(0u8..=255));
            }

            // qs (16 bytes) - low 4-bit values
            for _ in 0..16 {
                let lo = rng.gen_range(0u8..16);
                let hi = rng.gen_range(0u8..16);
                data.push((hi << 4) | lo);
            }
        }

        data
    }

    /// Generate weights for a specific quantization type
    pub fn generate_quant(&self, num_elements: usize, quant: QuantType) -> Vec<u8> {
        match quant {
            QuantType::F32 => {
                let f32_data = self.generate_f32(&[num_elements]);
                f32_data.iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect()
            }
            QuantType::F16 => {
                let f16_data = self.generate_f16(&[num_elements]);
                f16_data.iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect()
            }
            QuantType::BF16 => {
                // BF16 is just truncated F32
                let f32_data = self.generate_f32(&[num_elements]);
                f32_data.iter()
                    .flat_map(|f| {
                        let bits = f.to_bits();
                        let bf16_bits = (bits >> 16) as u16;
                        bf16_bits.to_le_bytes()
                    })
                    .collect()
            }
            QuantType::Q4_0 => self.generate_q4_0(num_elements),
            QuantType::Q8_0 => self.generate_q8_0(num_elements),
            QuantType::Q4_K => self.generate_q4_k(num_elements),
            QuantType::Q5_K => self.generate_q5_0(num_elements), // Similar structure
            QuantType::Q6_K => self.generate_q8_0(num_elements), // Use Q8 as approximation
        }
    }

    /// Generate all weights for a model config
    pub fn generate_model_weights(&self, config: &ModelConfig, quant: QuantType) -> ModelWeights {
        let head_dim = config.head_dim();

        // Create separate generators for each weight type (deterministic)
        let embed_gen = Self::new(self.seed);
        let layer_gen = Self::new(self.seed.wrapping_add(1000));
        let output_gen = Self::new(self.seed.wrapping_add(2000));

        // Embedding table
        let embed_weights = embed_gen.generate_quant(
            config.vocab_size * config.hidden_dim,
            quant
        );

        // Per-layer weights
        let mut layer_weights = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let lg = Self::new(layer_gen.seed.wrapping_add(layer_idx as u64 * 100));

            let layer = LayerWeights {
                // Attention projections
                attn_q: lg.generate_quant(config.hidden_dim * config.q_dim(), quant),
                attn_k: lg.generate_quant(config.hidden_dim * config.k_dim(), quant),
                attn_v: lg.generate_quant(config.hidden_dim * config.v_dim(), quant),
                attn_o: lg.generate_quant(config.q_dim() * config.hidden_dim, quant),

                // FFN projections
                ffn_gate: lg.generate_quant(config.hidden_dim * config.intermediate_dim, quant),
                ffn_up: lg.generate_quant(config.hidden_dim * config.intermediate_dim, quant),
                ffn_down: lg.generate_quant(config.intermediate_dim * config.hidden_dim, quant),

                // Norms (always F32)
                attn_norm: Self::new(lg.seed + 10).generate_f32(&[config.hidden_dim]),
                ffn_norm: Self::new(lg.seed + 11).generate_f32(&[config.hidden_dim]),
            };
            layer_weights.push(layer);
        }

        // Output norm and LM head
        let output_norm = output_gen.generate_f32(&[config.hidden_dim]);
        let lm_head = Self::new(output_gen.seed + 1).generate_quant(
            config.hidden_dim * config.vocab_size,
            quant
        );

        ModelWeights {
            config: config.clone(),
            quant_type: quant,
            embed_weights,
            layer_weights,
            output_norm,
            lm_head,
        }
    }
}

/// Weights for a single transformer layer
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// Q projection weights
    pub attn_q: Vec<u8>,
    /// K projection weights
    pub attn_k: Vec<u8>,
    /// V projection weights
    pub attn_v: Vec<u8>,
    /// Output projection weights
    pub attn_o: Vec<u8>,
    /// FFN gate projection
    pub ffn_gate: Vec<u8>,
    /// FFN up projection
    pub ffn_up: Vec<u8>,
    /// FFN down projection
    pub ffn_down: Vec<u8>,
    /// Attention RMSNorm gamma (always F32)
    pub attn_norm: Vec<f32>,
    /// FFN RMSNorm gamma (always F32)
    pub ffn_norm: Vec<f32>,
}

/// Complete model weights
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Model configuration
    pub config: ModelConfig,
    /// Quantization type used
    pub quant_type: QuantType,
    /// Embedding table
    pub embed_weights: Vec<u8>,
    /// Per-layer weights
    pub layer_weights: Vec<LayerWeights>,
    /// Output RMSNorm gamma
    pub output_norm: Vec<f32>,
    /// LM head weights
    pub lm_head: Vec<u8>,
}

impl ModelWeights {
    /// Total size in bytes
    pub fn total_bytes(&self) -> usize {
        let embed = self.embed_weights.len();
        let layers: usize = self.layer_weights.iter()
            .map(|l| {
                l.attn_q.len() + l.attn_k.len() + l.attn_v.len() + l.attn_o.len() +
                l.ffn_gate.len() + l.ffn_up.len() + l.ffn_down.len() +
                l.attn_norm.len() * 4 + l.ffn_norm.len() * 4
            })
            .sum();
        let output = self.output_norm.len() * 4 + self.lm_head.len();

        embed + layers + output
    }

    /// Number of parameters (approximate)
    pub fn param_count(&self) -> usize {
        self.config.param_count()
    }
}

/// Token generator for deterministic inputs
pub struct TokenGenerator {
    seed: u64,
    vocab_size: usize,
}

impl TokenGenerator {
    /// Create generator with seed and vocabulary size
    pub fn new(seed: u64, vocab_size: usize) -> Self {
        Self { seed, vocab_size }
    }

    /// Generate deterministic token sequence
    pub fn generate(&self, seq_len: usize) -> Vec<u32> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        (0..seq_len)
            .map(|_| rng.gen_range(1..self.vocab_size as u32))
            .collect()
    }

    /// Generate tokens with specific distribution (for testing edge cases)
    pub fn generate_with_distribution(&self, seq_len: usize, common_tokens: &[u32]) -> Vec<u32> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        (0..seq_len)
            .map(|_| {
                if rng.gen_bool(0.8) && !common_tokens.is_empty() {
                    // 80% common tokens
                    common_tokens[rng.gen_range(0..common_tokens.len())]
                } else {
                    // 20% random
                    rng.gen_range(1..self.vocab_size as u32)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_deterministic() {
        let gen1 = SyntheticWeightGenerator::new(42);
        let gen2 = SyntheticWeightGenerator::new(42);

        let w1 = gen1.generate_f32(&[10, 10]);
        let w2 = gen2.generate_f32(&[10, 10]);

        assert_eq!(w1, w2, "Same seed should produce same weights");
    }

    #[test]
    fn test_generator_different_seeds() {
        let gen1 = SyntheticWeightGenerator::new(42);
        let gen2 = SyntheticWeightGenerator::new(43);

        let w1 = gen1.generate_f32(&[10, 10]);
        let w2 = gen2.generate_f32(&[10, 10]);

        assert_ne!(w1, w2, "Different seeds should produce different weights");
    }

    #[test]
    fn test_q4_0_block_size() {
        let gen = SyntheticWeightGenerator::new(42);
        let data = gen.generate_q4_0(32);
        // One block: 2 bytes scale + 16 bytes quants = 18 bytes
        assert_eq!(data.len(), 18);
    }

    #[test]
    fn test_q4_0_multiple_blocks() {
        let gen = SyntheticWeightGenerator::new(42);
        let data = gen.generate_q4_0(64);
        // Two blocks: 2 * 18 = 36 bytes
        assert_eq!(data.len(), 36);
    }

    #[test]
    fn test_q8_0_block_size() {
        let gen = SyntheticWeightGenerator::new(42);
        let data = gen.generate_q8_0(32);
        // One block: 2 bytes scale + 32 bytes quants = 34 bytes
        assert_eq!(data.len(), 34);
    }

    #[test]
    fn test_model_weights_generation() {
        let gen = SyntheticWeightGenerator::new(42);
        let config = ModelConfig::tiny();
        let weights = gen.generate_model_weights(&config, QuantType::Q4_0);

        assert_eq!(weights.layer_weights.len(), config.num_layers);
        assert_eq!(weights.output_norm.len(), config.hidden_dim);
        assert!(weights.total_bytes() > 0);
    }

    #[test]
    fn test_token_generator() {
        let gen = TokenGenerator::new(42, 256);
        let tokens = gen.generate(10);

        assert_eq!(tokens.len(), 10);
        assert!(tokens.iter().all(|&t| t > 0 && t < 256));
    }

    #[test]
    fn test_token_generator_deterministic() {
        let gen1 = TokenGenerator::new(42, 256);
        let gen2 = TokenGenerator::new(42, 256);

        assert_eq!(gen1.generate(10), gen2.generate(10));
    }

    #[test]
    fn test_f16_generation() {
        let gen = SyntheticWeightGenerator::new(42);
        let weights = gen.generate_f16(&[100]);

        assert_eq!(weights.len(), 100);
        // All values should be finite
        assert!(weights.iter().all(|w| w.is_finite()));
    }

    #[test]
    fn test_quant_dispatch() {
        let gen = SyntheticWeightGenerator::new(42);

        // F32: 4 bytes per element
        let f32_data = gen.generate_quant(100, QuantType::F32);
        assert_eq!(f32_data.len(), 400);

        // F16: 2 bytes per element
        let f16_data = gen.generate_quant(100, QuantType::F16);
        assert_eq!(f16_data.len(), 200);

        // BF16: 2 bytes per element
        let bf16_data = gen.generate_quant(100, QuantType::BF16);
        assert_eq!(bf16_data.len(), 200);

        // Q4_K: 144 bytes per 256 elements
        let q4k_data = gen.generate_quant(256, QuantType::Q4_K);
        assert_eq!(q4k_data.len(), 144);

        // Q5_K: uses Q5_0 (22 bytes per 32 elements)
        let q5k_data = gen.generate_quant(32, QuantType::Q5_K);
        assert_eq!(q5k_data.len(), 22);

        // Q6_K: uses Q8_0 (34 bytes per 32 elements)
        let q6k_data = gen.generate_quant(32, QuantType::Q6_K);
        assert_eq!(q6k_data.len(), 34);
    }

    #[test]
    fn test_generate_f32_scaled() {
        let gen = SyntheticWeightGenerator::new(42);
        let scale = 10.0;
        let weights = gen.generate_f32_scaled(&[100], scale);
        assert_eq!(weights.len(), 100);
        for &w in &weights {
            assert!(w >= -scale && w <= scale);
        }
    }

    #[test]
    fn test_token_generator_distribution() {
        let vocab_size = 1000;
        let gen = TokenGenerator::new(42, vocab_size);
        let common = vec![1, 2, 3];
        let tokens = gen.generate_with_distribution(100, &common);
        assert_eq!(tokens.len(), 100);
        assert!(tokens.iter().all(|&t| t < vocab_size as u32));
    }

    #[test]
    fn test_token_generator_distribution_empty_common() {
        let gen = TokenGenerator::new(42, 100);
        let tokens = gen.generate_with_distribution(10, &[]);
        assert_eq!(tokens.len(), 10);
    }

    #[test]
    fn test_model_weights_metrics() {
        let config = ModelConfig::tiny();
        let gen = SyntheticWeightGenerator::new(42);
        let weights = gen.generate_model_weights(&config, QuantType::F32);
        
        assert!(weights.total_bytes() > 0);
        assert_eq!(weights.param_count(), config.param_count());
    }

    #[test]
    fn test_f32_empty_shape() {
        let gen = SyntheticWeightGenerator::new(42);
        let weights = gen.generate_f32(&[]);
        assert_eq!(weights.len(), 1); // Product of empty is 1
    }
}
