
#[cfg(test)]
mod tests {
    use super::*;

    // -- Validation failure tests --

    #[test]
    fn test_validated_config_rejects_zero_hidden_dim() {
        let mut cfg = valid_llama_config();
        cfg.hidden_dim = 0;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("hidden_dim"));
    }

    #[test]
    fn test_validated_config_rejects_zero_num_layers() {
        let mut cfg = valid_llama_config();
        cfg.num_layers = 0;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("num_layers"));
    }

    #[test]
    fn test_validated_config_rejects_zero_vocab_size() {
        let mut cfg = valid_llama_config();
        cfg.vocab_size = 0;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("vocab_size"));
    }

    #[test]
    fn test_validated_config_rejects_zero_num_heads() {
        let mut cfg = valid_llama_config();
        cfg.num_heads = 0;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("num_heads"));
    }

    #[test]
    fn test_validated_config_rejects_zero_num_kv_heads() {
        let mut cfg = valid_llama_config();
        cfg.num_kv_heads = 0;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("num_kv_heads"));
    }

    #[test]
    fn test_validated_config_rejects_bad_head_dim() {
        let mut cfg = valid_llama_config();
        // 4096 % 33 != 0
        cfg.num_heads = 33;
        cfg.num_kv_heads = 33;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("divisible by num_heads"));
    }

    #[test]
    fn test_validated_config_rejects_bad_gqa_ratio() {
        let mut cfg = valid_llama_config();
        // 32 % 5 != 0
        cfg.num_kv_heads = 5;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("GQA ratio"));
    }

    #[test]
    fn test_validated_config_mha() {
        // MHA: num_heads == num_kv_heads
        let mut cfg = valid_llama_config();
        cfg.num_kv_heads = 32;
        let v = ValidatedModelConfig::validate(cfg).expect("MHA should be valid");
        assert_eq!(v.num_heads(), v.num_kv_heads());
        assert_eq!(v.head_dim(), 128);
        assert_eq!(v.kv_dim(), 4096); // 32 * 128
    }

    #[test]
    fn test_validated_config_qwen_style() {
        // Qwen 1.5B: hidden=1536, heads=12, kv_heads=2
        let cfg = GGUFConfig {
            architecture: "qwen2".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("qwen2"),
            hidden_dim: 1536,
            num_layers: 28,
            num_heads: 12,
            num_kv_heads: 2,
            vocab_size: 151936,
            intermediate_dim: 8960,
            context_length: 32768,
            rope_theta: 1_000_000.0,
            eps: 1e-6,
            rope_type: 2,
            bos_token_id: Some(151_643),
        };
        let v = ValidatedModelConfig::validate(cfg).expect("Qwen config should be valid");
        assert_eq!(v.head_dim(), 128); // 1536 / 12
        assert_eq!(v.kv_dim(), 256); // 2 * 128
    }

    /// GH-39: Qwen2.5-0.5B has unusual 7:1 GQA ratio and head_dim=64
    #[test]
    fn test_validated_config_qwen2_0_5b_dimensions() {
        let cfg = GGUFConfig {
            architecture: "qwen2".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("qwen2"),
            hidden_dim: 896,
            num_layers: 24,
            num_heads: 14,
            num_kv_heads: 2,
            vocab_size: 151936,
            intermediate_dim: 4864,
            context_length: 32768,
            rope_theta: 1_000_000.0,
            eps: 1e-6,
            rope_type: 2,
            bos_token_id: Some(151_643),
        };
        let v = ValidatedModelConfig::validate(cfg)
            .expect("GH-39: Qwen2.5-0.5B config should be valid");
        assert_eq!(v.head_dim(), 64, "GH-39: 0.5B has head_dim=64");
        assert_eq!(v.kv_dim(), 128, "GH-39: kv_dim = 2 * 64");
        assert_eq!(v.num_heads() / v.num_kv_heads(), 7, "GH-39: 7:1 GQA ratio");
        assert_eq!(v.rope_type(), 2, "GH-39: NEOX RoPE required");
    }
include!("config_gguf.rs");
}
