
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
