    // F079: Error propagation
    #[test]
    fn f079_error_propagation() {
        let brick = ActivationQuantBrick::new(32);

        // Wrong input size should error
        let wrong_input = vec![1.0f32; 64]; // Expected 32
        let result = brick.quantize(&wrong_input);

        assert!(result.is_err(), "Wrong input size should error");
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("64"), "Error should mention actual size");
            assert!(msg.contains("32"), "Error should mention expected size");
        }
    }

    // F080: Graceful handling of edge cases
    #[test]
    #[allow(deprecated)] // Testing legacy execute() methods
    fn f080_edge_cases() {
        // Empty/zero dimension handling
        let brick = ActivationQuantBrick::new(0);
        assert!(brick.execute().is_err(), "Zero dim should error");

        let flash = FlashAttentionBrick::new(0, 0, 0);
        assert!(flash.execute(10).is_err(), "Zero heads/dim should error");
    }

    // ========================================================================
    // F082-F100: Performance Regression Infrastructure
    // ========================================================================

    // F082: Iteration count for statistical validity
    #[test]
    fn f082_iteration_count() {
        let config = BenchmarkConfig::default();
        assert!(config.samples >= 100, "Need >= 100 samples for valid stats");
        assert!(config.warmup >= 10, "Need >= 10 warmup iterations");
    }

    // F083: Timing precision
    #[test]
    #[ignore = "Timing test unreliable - depends on system load"]
    fn f083_timing_precision() {
        let start = std::time::Instant::now();
        std::thread::sleep(std::time::Duration::from_micros(100));
        let elapsed = start.elapsed().as_nanos();

        // Should be able to measure ~100µs accurately
        assert!(elapsed > 50_000, "Timing should measure > 50µs");
        assert!(elapsed < 500_000, "Timing should not drift too much");
    }

    // F084: Percentile calculation
    #[test]
    fn f084_percentile_calculation() {
        let mut samples: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p50 = samples[49]; // 50th percentile
        let p99 = samples[98]; // 99th percentile

        assert_eq!(p50, 50.0, "P50 should be 50");
        assert_eq!(p99, 99.0, "P99 should be 99");
    }

    // F088: Memory bandwidth tracking
    #[test]
    fn f088_bandwidth_tracking() {
        let flash = FlashAttentionBrick::new(8, 2, 64);
        let (naive, flash_mem) = flash.memory_bytes(512);

        // Flash should use less memory
        assert!(flash_mem < naive, "Flash uses less memory");

        // Bandwidth reduction calculable
        let reduction = naive as f64 / flash_mem as f64;
        assert!(reduction > 1.0, "Should show bandwidth reduction");
    }

    // F089: Arithmetic intensity tracking
    #[test]
    #[cfg(feature = "cuda")]
    fn f089_arithmetic_intensity() {
        let brick = FusedFfnBrick::new(1536, 8960);
        let ai = brick.arithmetic_intensity();

        // FFN should have reasonable AI (not memory-bound for larger dims)
        assert!(ai > 0.0, "AI should be positive");
    }

    // F091: Regression baseline
    #[test]
    fn f091_regression_baseline() {
        // Ollama baseline for comparison
        let ollama_1_5b = 488.0; // tok/s for 1.5B
        let target_2x = ollama_1_5b * 2.0;

        let target_budget = TokenBudget::from_throughput(target_2x);
        assert!(
            target_budget.us_per_token < 1050.0,
            "2x Ollama should be ~1025µs/tok"
        );
    }

    // F095: Model size scaling
    #[test]
    #[cfg(feature = "cuda")]
    fn f095_model_size_scaling() {
        // Larger models should have proportionally larger budgets
        let small = FusedFfnBrick::new(1536, 8960); // 1.5B scale
        let large = FusedFfnBrick::new(4096, 22528); // 32B scale

        // FLOPs should scale with dimensions
        assert!(
            large.flops() > small.flops() * 4,
            "32B should have ~7x FLOPs of 1.5B"
        );
    }

    // F096: PMAT score infrastructure
    #[test]
    #[cfg(feature = "cuda")]
    fn f096_pmat_score_ready() {
        // This test verifies we track metrics needed for PMAT scoring
        let brick = FusedFfnBrick::new(64, 256);

        // Has budget
        assert!(brick.budget().us_per_token > 0.0);
        // Has FLOPs
        assert!(brick.flops() > 0);
        // Has arithmetic intensity
        assert!(brick.arithmetic_intensity() > 0.0);
    }

    // F097: CI integration infrastructure
    #[test]
    fn f097_ci_integration() {
        // BenchmarkConfig supports CI settings
        let config = BenchmarkConfig::default();
        assert!(config.samples > 0, "CI needs sample count");
        assert!(config.warmup > 0, "CI needs warmup count");
    }

    // F098: Regression detection
    #[test]
    fn f098_regression_detection() {
        let budget = TokenBudget::from_latency(100.0);

        // 20% regression threshold
        let regression_threshold = 1.2;
        let actual = 115.0; // 15% over

        let gap = budget.gap_factor(actual);
        let is_regression = gap > regression_threshold;

        assert!(!is_regression, "15% over should not trigger 20% threshold");

        let bad_actual = 125.0; // 25% over
        let bad_gap = budget.gap_factor(bad_actual);
        assert!(
            bad_gap > regression_threshold,
            "25% over should trigger regression"
        );
    }

    // F099: Output format for CI
    #[test]
    fn f099_ci_output_format() {
        let report = BenchmarkReport {
            brick_name: "test".to_string(),
            mean_us: 50.0,
            std_us: 5.0,
            cv: 10.0,
            p50_us: 50.0,
            p99_us: 58.0,
            tokens_per_sec: 20000.0,
            budget_us: 100.0,
            budget_met: true,
            statistically_valid: true,
        };

        // All fields accessible for CI output
        assert!(!report.brick_name.is_empty());
        assert!(report.mean_us > 0.0);
        assert!(report.cv > 0.0);
    }

    // F100: Zero-defect gate
    #[test]
    fn f100_zero_defect_gate() {
        // All bricks must pass their assertions
        let bricks: Vec<Box<dyn ComputeBrick<Output = Vec<f32>>>> = vec![
            Box::new(RmsNormBrick::new(vec![1.0; 64], 1e-5)),
            Box::new(FfnBrick::new(64, 256)),
            Box::new(AttentionBrick::new(4, 2, 16)),
        ];

        for brick in &bricks {
            assert!(brick.can_run(), "Brick {} should be runnable", brick.name());
            assert!(
                !brick.assertions().is_empty(),
                "Brick {} should have assertions",
                brick.name()
            );
        }
    }

    // ========================================================================
    // Coverage Tests: TokenBudget
    // ========================================================================

    #[test]
    fn test_token_budget_debug() {
        let budget = TokenBudget::from_latency(100.0);
        let debug = format!("{:?}", budget);
        assert!(debug.contains("TokenBudget"));
    }

    #[test]
    fn test_token_budget_clone() {
        let budget = TokenBudget::from_latency(100.0).with_batch_size(32);
        let cloned = budget;
        assert_eq!(budget.us_per_token, cloned.us_per_token);
        assert_eq!(budget.batch_size, cloned.batch_size);
    }

    #[test]
    fn test_token_budget_default() {
        let budget = TokenBudget::default();
        assert!(budget.us_per_token > 0.0);
        assert_eq!(budget.batch_size, 1);
    }

    // ========================================================================
    // Coverage Tests: TokenResult
    // ========================================================================

    #[test]
    fn test_token_result_debug() {
        let budget = TokenBudget::from_latency(100.0);
        let result: TokenResult<Vec<f32>> = TokenResult::new(vec![1.0, 2.0], 2, 100.0, &budget);
        let debug = format!("{:?}", result);
        assert!(debug.contains("TokenResult"));
    }

    #[test]
    fn test_token_result_clone() {
        let budget = TokenBudget::from_latency(100.0);
        let result: TokenResult<Vec<f32>> = TokenResult::new(vec![1.0, 2.0], 2, 100.0, &budget);
        let cloned = result.clone();
        assert_eq!(result.tokens_processed, cloned.tokens_processed);
        assert_eq!(result.output, cloned.output);
    }

    #[test]
    fn test_token_result_zero_tokens() {
        let budget = TokenBudget::from_latency(100.0);
        let result: TokenResult<Vec<f32>> = TokenResult::new(vec![], 0, 100.0, &budget);
        assert_eq!(result.tokens_processed, 0);
        // With zero tokens, us_per_token could be NaN (0/0) or infinity, depending on implementation
        // Just ensure the result is created without panic
    }

    // ========================================================================
    // Coverage Tests: BrickError
    // ========================================================================

    #[test]
    fn test_brick_error_display() {
        let err = BrickError::InvalidInput("test error".to_string());
        assert!(err.to_string().contains("test error"));

        let err2 = BrickError::BudgetExceeded {
            limit_us: 100.0,
            actual_us: 150.0,
        };
        assert!(err2.to_string().contains("150"));

        let err3 = BrickError::ComputeError("failed".to_string());
        assert!(err3.to_string().contains("failed"));

        let err4 = BrickError::AssertionFailed {
            name: "test".to_string(),
            expected: "expected".to_string(),
            actual: "actual".to_string(),
        };
        assert!(err4.to_string().contains("test"));
    }

    #[test]
    fn test_brick_error_debug() {
        let err = BrickError::InvalidInput("debug test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("InvalidInput"));
    }

    // ========================================================================
    // Coverage Tests: BrickAssertion
    // ========================================================================

    #[test]
    fn test_assertion_check_bounds() {
        let assertion = BrickAssertion::bounds(-1.0, 1.0);
        let data = vec![0.0, 0.5, -0.5, 0.9];
        assert!(assertion.check_f32(&data, true).is_ok());

        let bad_data = vec![0.0, 1.5, -0.5];
        assert!(assertion.check_f32(&bad_data, true).is_err());
    }

    #[test]
    fn test_assertion_check_equiv_scalar() {
        let assertion = BrickAssertion::equiv_scalar(0.01);
        let data = vec![1.0, 1.005, 1.003, 0.997];
        assert!(assertion.check_f32(&data, true).is_ok());
    }

    #[test]
    fn test_assertion_kind_debug() {
        let kind = AssertionKind::NoNaN;
        let debug = format!("{:?}", kind);
        assert!(debug.contains("NoNaN"));

        let kind2 = AssertionKind::Bounds { min: 0.0, max: 1.0 };
        let debug2 = format!("{:?}", kind2);
        assert!(debug2.contains("Bounds"));
    }

    // ========================================================================
    // Coverage Tests: BrickVerification
    // ========================================================================

    #[test]
    fn test_brick_verification_pass() {
        let v = BrickVerification::pass();
        assert!(v.is_valid);
        assert!(v.results.is_empty());
    }

    #[test]
    fn test_brick_verification_fail() {
        let v = BrickVerification::fail("test", "failed reason");
        assert!(!v.is_valid);
        assert_eq!(v.results.len(), 1);
    }

    #[test]
    fn test_brick_verification_add() {
        let mut v = BrickVerification::pass();
        v.add("check1", true, "passed");
        v.add("check2", false, "failed");
        assert!(!v.is_valid); // Should be false after failed check
        assert_eq!(v.results.len(), 2);
    }

    // ========================================================================
    // Coverage Tests: FlashAttentionBrick
    // ========================================================================

    #[test]
    fn test_flash_attention_group_size() {
        let brick = FlashAttentionBrick::new(8, 2, 64);
        assert_eq!(brick.group_size(), 4); // 8 heads / 2 kv heads
    }

    #[test]
    fn test_flash_attention_flops() {
        let brick = FlashAttentionBrick::new(8, 2, 64);
        let flops = brick.flops(512);
        assert!(flops > 0);
    }

    #[test]
    fn test_flash_attention_memory_bytes() {
        let brick = FlashAttentionBrick::new(8, 2, 64);
        let (naive, flash) = brick.memory_bytes(512);
        assert!(flash < naive);
    }

    // ========================================================================
    // Coverage Tests: QkvBrick
    // ========================================================================

    #[test]
    fn test_qkv_brick_with_bias() {
        let brick = QkvBrick::new(64, 64, 64, 64).with_bias();
        assert!(brick.has_bias);
    }

    #[test]
    fn test_qkv_brick_total_out_dim() {
        let brick = QkvBrick::new(64, 64, 32, 32);
        assert_eq!(brick.total_out_dim(), 128); // 64 + 32 + 32
    }

    // ========================================================================
    // Coverage Tests: BenchmarkConfig
    // ========================================================================

    #[test]
    fn test_benchmark_config_debug() {
        let config = BenchmarkConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("BenchmarkConfig"));
    }

    #[test]
    fn test_benchmark_config_clone() {
        let config = BenchmarkConfig::default();
        let cloned = config.clone();
        assert_eq!(config.samples, cloned.samples);
    }

    // ========================================================================
    // Coverage Tests: BenchmarkReport
    // ========================================================================

    #[test]
    fn test_benchmark_report_debug() {
        let report = BenchmarkReport {
            brick_name: "test".to_string(),
            mean_us: 50.0,
            std_us: 5.0,
            cv: 10.0,
            p50_us: 50.0,
            p99_us: 58.0,
            tokens_per_sec: 20000.0,
            budget_us: 100.0,
            budget_met: true,
            statistically_valid: true,
        };
        let debug = format!("{:?}", report);
        assert!(debug.contains("BenchmarkReport"));
    }

    #[test]
    fn test_benchmark_report_clone() {
        let report = BenchmarkReport {
            brick_name: "test".to_string(),
            mean_us: 50.0,
            std_us: 5.0,
            cv: 10.0,
            p50_us: 50.0,
            p99_us: 58.0,
            tokens_per_sec: 20000.0,
            budget_us: 100.0,
            budget_met: true,
            statistically_valid: true,
        };
        let cloned = report.clone();
        assert_eq!(report.brick_name, cloned.brick_name);
        assert_eq!(report.mean_us, cloned.mean_us);
    }

    // ========================================================================
    // Coverage Tests: LayerTiming
    // ========================================================================

    #[test]
    fn test_layer_timing_default() {
        let timing = LayerTiming::default();
        assert_eq!(timing.attn_norm_us, 0.0);
        assert_eq!(timing.qkv_us, 0.0);
    }

    #[test]
    fn test_layer_timing_debug() {
        let timing = LayerTiming::default();
        let debug = format!("{:?}", timing);
        assert!(debug.contains("LayerTiming"));
    }

    #[test]
    fn test_layer_timing_clone() {
        let timing = LayerTiming {
            attn_norm_us: 1.0,
            qkv_us: 2.0,
            rope_us: 3.0,
            attention_us: 4.0,
            o_proj_us: 5.0,
            ffn_norm_us: 6.0,
            ffn_us: 7.0,
            total_us: 8.0,
        };
        let cloned = timing.clone();
        assert_eq!(timing.attn_norm_us, cloned.attn_norm_us);
        assert_eq!(timing.qkv_us, cloned.qkv_us);
    }

    #[test]
    fn test_layer_timing_bottleneck() {
        let timing = LayerTiming {
            attn_norm_us: 1.0,
            qkv_us: 10.0, // This should be the bottleneck
            rope_us: 2.0,
            attention_us: 5.0,
            o_proj_us: 3.0,
            ffn_norm_us: 1.0,
            ffn_us: 8.0,
            total_us: 30.0,
        };
        let (name, time) = timing.bottleneck();
        assert_eq!(name, "qkv");
        assert_eq!(time, 10.0);
    }

    // ========================================================================
    // Additional Coverage Tests: Fused Operations
    // ========================================================================

    // T001: CoalescedDp4aBrick dimension validation
    #[test]
    #[cfg(feature = "cuda")]
    fn t001_coalesced_dp4a_invalid_input_length() {
        let brick = CoalescedDp4aBrick::new(256, 4);

        // Wrong input length
        let result = brick.forward(
            &[1i8; 128], // Wrong: 128 instead of 256
            1.0,
            &[0x88u8; 512],
            &[0.1f32; 4],
        );
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("128"), "Error should mention actual size");
            assert!(msg.contains("256"), "Error should mention expected size");
        }
    }

    // T002: CoalescedDp4aBrick weight length validation
    #[test]
    #[cfg(feature = "cuda")]
    fn t002_coalesced_dp4a_invalid_weights() {
        let brick = CoalescedDp4aBrick::new(256, 4);

        // Wrong weights length (should be n * k / 2 = 4 * 256 / 2 = 512)
        let result = brick.forward(
            &[1i8; 256],
            1.0,
            &[0x88u8; 256], // Wrong: 256 instead of 512
            &[0.1f32; 4],
        );
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("Weights"));
        }
    }

    // T003: CoalescedDp4aBrick scale length validation
    #[test]
    #[cfg(feature = "cuda")]
    fn t003_coalesced_dp4a_invalid_scales() {
        let brick = CoalescedDp4aBrick::new(256, 4);

        // Wrong scale length (should be n = 4)
        let result = brick.forward(
            &[1i8; 256],
            1.0,
            &[0x88u8; 512],
            &[0.1f32; 2], // Wrong: 2 instead of 4
        );
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("scale"));
        }
    }

    // T004: CoalescedDp4aBrick can_run for invalid dimensions
    #[test]
    #[cfg(feature = "cuda")]
    fn t004_coalesced_dp4a_can_run_invalid() {
        // K not multiple of 256
        let brick = CoalescedDp4aBrick::new(128, 4);
        assert!(!brick.can_run(), "K=128 not multiple of 256 should not run");

        // Zero dimensions
        let brick = CoalescedDp4aBrick::new(0, 4);
        assert!(!brick.can_run(), "K=0 should not run");

        let brick = CoalescedDp4aBrick::new(256, 0);
        assert!(!brick.can_run(), "N=0 should not run");
    }

    // T005: CoalescedDp4aBrick arithmetic intensity
    #[test]
    #[cfg(feature = "cuda")]
    fn t005_coalesced_dp4a_arithmetic_intensity() {
        let brick = CoalescedDp4aBrick::new(1024, 256);
        let ai = brick.arithmetic_intensity();

        // Should be positive and reasonable (AI can be high for compute-bound kernels)
        assert!(ai > 0.0, "Arithmetic intensity should be positive");
        assert!(ai < 1000.0, "Arithmetic intensity should be reasonable");
    }

    // T006: FusedFfnBrick invalid input length
    #[test]
    #[cfg(feature = "cuda")]
    fn t006_fused_ffn_invalid_input() {
        let brick = FusedFfnBrick::new(4, 8);

        // Wrong input length
        let result = brick.forward(
            &[1.0f32; 8], // Wrong: 8 instead of 4
            &[0.1f32; 32],
            &[0.2f32; 32],
            &[0.1f32; 32],
        );
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("Input"));
        }
    }

    // T007: FusedFfnBrick invalid gate/up proj length
    #[test]
    #[cfg(feature = "cuda")]
    fn t007_fused_ffn_invalid_gate_up() {
        let brick = FusedFfnBrick::new(4, 8);

        // Wrong gate_proj length (should be 8 * 4 = 32)
        let result = brick.forward(
            &[1.0f32; 4],
            &[0.1f32; 16], // Wrong: 16 instead of 32
            &[0.2f32; 32],
            &[0.1f32; 32],
        );
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("Gate/Up"));
        }
    }

    // T008: FusedFfnBrick invalid down proj length
    #[test]
    #[cfg(feature = "cuda")]
    fn t008_fused_ffn_invalid_down() {
        let brick = FusedFfnBrick::new(4, 8);

        // Wrong down_proj length (should be 4 * 8 = 32)
        let result = brick.forward(
            &[1.0f32; 4],
            &[0.1f32; 32],
            &[0.2f32; 32],
            &[0.1f32; 16], // Wrong: 16 instead of 32
        );
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("Down"));
        }
    }

    // T009: FusedFfnBrick can_run for invalid dimensions
    #[test]
    #[cfg(feature = "cuda")]
    fn t009_fused_ffn_can_run_invalid() {
        let brick = FusedFfnBrick::new(0, 8);
        assert!(!brick.can_run(), "Zero hidden_dim should not run");

        let brick = FusedFfnBrick::new(4, 0);
        assert!(!brick.can_run(), "Zero intermediate_dim should not run");
    }

    // T010: FlashAttentionBrick invalid query length
    #[test]
    fn t010_flash_attention_invalid_query() {
        let brick = FlashAttentionBrick::new(4, 2, 8);

        // Wrong query length (should be 4 * 8 = 32)
        let result = brick.forward(
            &[1.0f32; 16], // Wrong: 16 instead of 32
            &[0.5f32; 64],
            &[0.25f32; 64],
            4,
        );
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("Query"));
        }
    }

    // T011: FlashAttentionBrick invalid keys/values length
    #[test]
    fn t011_flash_attention_invalid_kv() {
        let brick = FlashAttentionBrick::new(4, 2, 8);
        let seq_len = 4;

        // Wrong keys length (should be seq_len * num_kv_heads * head_dim = 4 * 2 * 8 = 64)
        let result = brick.forward(
            &[1.0f32; 32],
            &[0.5f32; 32], // Wrong: 32 instead of 64
            &[0.25f32; 64],
            seq_len,
        );
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("KV"));
        }
    }

    // T012: FlashAttentionBrick zero dimension
    #[test]
    fn t012_flash_attention_zero_dim() {
        let brick = FlashAttentionBrick::new(0, 0, 0);

        let result = brick.forward(&[], &[], &[], 0);
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("Zero"));
        }
    }

    // T013: FlashAttentionBrick arithmetic intensity
    #[test]
    fn t013_flash_attention_arithmetic_intensity() {
        let brick = FlashAttentionBrick::new(8, 2, 64);
        let ai = brick.arithmetic_intensity(512);

        assert!(ai > 0.0, "Arithmetic intensity should be positive");
    }

    // T014: ActivationQuantBrick zero dimension quantize
    #[test]
    fn t014_activation_quant_zero_dim_quantize() {
        let brick = ActivationQuantBrick::new(0);
        let result = brick.quantize(&[]);
        assert!(result.is_err());
    }

    // T015: ActivationQuantBrick dequantize length mismatch
    #[test]
    fn t015_activation_quant_dequantize_mismatch() {
        let brick = ActivationQuantBrick::new(32);

        // Wrong quants length
        let result = brick.dequantize(&[0i8; 16], &[0.1f32; 1]);
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("16"));
        }
    }

    // T016: ActivationQuantBrick measure_error with zero max
    #[test]
    fn t016_activation_quant_measure_error_zero() {
        let brick = ActivationQuantBrick::new(32);
        let original = vec![0.0f32; 32];

        let (quants, scales) = brick.quantize(&original).expect("quantize failed");
        let error = brick
            .measure_error(&original, &quants, &scales)
            .expect("measure failed");

        // Error should be 0 for zero input
        assert_eq!(error, 0.0, "Error should be 0 for zero input");
    }

    // T017: RmsNormBrick input/weight length mismatch
    #[test]
    fn t017_rmsnorm_length_mismatch() {
        let brick =
            RmsNormBrick::new(vec![1.0; 32], 1e-5).with_budget(TokenBudget::from_latency(1000.0));

        // Wrong input length
        let result = brick.run(&[1.0f32; 16]);
        assert!(result.is_err());
        if let Err(BrickError::InvalidInput(msg)) = result {
            assert!(msg.contains("16"));
            assert!(msg.contains("32"));
        }
    }

    // T018: CudaGraphBrick replay without capture
    #[test]
    fn t018_cuda_graph_replay_not_captured() {
        let brick = CudaGraphBrick::new(28, 1536);

        let result = brick.replay();
        assert!(result.is_err());
        if let Err(BrickError::ComputeError(msg)) = result {
            assert!(msg.contains("not captured"));
        }
    }

    // T019: CudaGraphBrick can_run with invalid config
    #[test]
    fn t019_cuda_graph_can_run_invalid() {
        let brick = CudaGraphBrick::new(0, 1536);
        assert!(!brick.can_run(), "Zero layers should not run");

        let brick = CudaGraphBrick::new(28, 0);
        assert!(!brick.can_run(), "Zero hidden_dim should not run");
    }

    // T020: TransformerLayerBrick verify with invalid component
    #[test]
    fn t020_transformer_layer_verify() {
        let layer = TransformerLayerBrick::from_config(0, 64, 4, 2, 256, 1e-5, 10000.0, 0);

        // Should pass verification with valid config
        let verification = layer.verify();
        assert!(
            verification.is_valid,
            "Valid layer should pass verification"
        );
    }

    // T021: BottleneckReport display
    #[test]
    fn t021_bottleneck_report_display() {
        let report = BottleneckReport {
            layer_idx: 5,
            brick_name: "attention",
            actual_us: 15.0,
            budget_us: 10.0,
            gap_factor: 1.5,
        };

        let display = format!("{}", report);
        assert!(display.contains("attention"));
        assert!(display.contains("layer 5"));
        assert!(display.contains("15.0"));
        assert!(display.contains("10.0"));
        assert!(display.contains("1.50"));
    }

    // T022: BrickError std::error::Error impl
    #[test]
    fn t022_brick_error_std_error() {
        let err = BrickError::InvalidInput("test".to_string());

        // Verify Error trait is implemented (compiles)
        let _: &dyn std::error::Error = &err;
    }

    // T023: TokenResult with zero us_per_token
    #[test]
    fn t023_token_result_zero_latency() {
        let budget = TokenBudget::from_latency(100.0);
        let result: TokenResult<Vec<f32>> = TokenResult::new(vec![], 1, 0.0, &budget);

        // With 0 elapsed time, tokens_per_sec should be 0 (not infinity)
        assert_eq!(result.tokens_per_sec, 0.0);
    }

    // T024: FlashAttentionBrick with custom tile size
    #[test]
    fn t024_flash_attention_custom_tile() {
        let brick = FlashAttentionBrick::with_tile_size(8, 2, 64, 64);
        assert_eq!(brick.tile_size, 64);
        assert_eq!(brick.num_tiles(512), 8); // 512 / 64 = 8
    }

    // T025: FlashAttentionBrick can_run with zero tile
    #[test]
    fn t025_flash_attention_zero_tile() {
        let mut brick = FlashAttentionBrick::new(8, 2, 64);
        brick.tile_size = 0;
        assert!(!brick.can_run(), "Zero tile_size should not run");
    }

    // T026: AttentionBrick group_size edge case
    #[test]
    fn t026_attention_group_size_edge() {
        // Zero kv_heads should use max(1) to avoid division by zero
        let brick = AttentionBrick::new(8, 0, 64);
        assert_eq!(brick.group_size(), 8); // 8 / max(0, 1) = 8
    }

    // T027: benchmark_brick with empty samples
    #[test]
    fn t027_benchmark_empty_samples() {
        let brick = RmsNormBrick::new(vec![1.0; 4], 1e-5);
        let config = BenchmarkConfig {
            warmup: 0,
            samples: 1, // Minimal samples
            max_cv: 1.0,
        };

        let report = benchmark_brick(
            &brick,
            || 10.0, // Fixed timing
            &config,
        );

        assert_eq!(report.mean_us, 10.0);
        assert_eq!(report.std_us, 0.0); // Single sample = no variance
    }

    // T028: BrickAssertion budget_met check
    #[test]
    fn t028_assertion_budget_not_met() {
        let assertion = BrickAssertion::budget_met();
        let data = &[1.0f32, 2.0, 3.0];

        // Should pass when budget is met
        assert!(assertion.check_f32(data, true).is_ok());

        // Should fail when budget is not met
        let result = assertion.check_f32(data, false);
        assert!(result.is_err());
        if let Err(BrickError::AssertionFailed {
            name,
            expected,
            actual,
        }) = result
        {
            assert_eq!(name, "budget_met");
            assert!(expected.contains("met"));
            assert!(actual.contains("exceeded"));
        }
    }

    // T029: BrickAssertion no_inf check
    #[test]
    fn t029_assertion_inf_check() {
        let assertion = BrickAssertion::no_inf();

        // Should pass without infinity
        assert!(assertion.check_f32(&[1.0, 2.0, 3.0], true).is_ok());

        // Should fail with positive infinity
        let result = assertion.check_f32(&[1.0, f32::INFINITY, 3.0], true);
        assert!(result.is_err());

        // Should fail with negative infinity
        let result = assertion.check_f32(&[1.0, f32::NEG_INFINITY, 3.0], true);
        assert!(result.is_err());
    }

    // T030: BrickAssertion bounds check - edge values
    #[test]
    fn t030_assertion_bounds_edge() {
        let assertion = BrickAssertion::bounds(-1.0, 1.0);

        // Exact boundary values should pass
        assert!(assertion.check_f32(&[-1.0, 0.0, 1.0], true).is_ok());

        // Just outside should fail (below)
        let result = assertion.check_f32(&[-1.01], true);
        assert!(result.is_err());

        // Just outside should fail (above)
        let result = assertion.check_f32(&[1.01], true);
        assert!(result.is_err());
    }

    // T031: FlashAttentionBrick with single sequence position
    #[test]
    fn t031_flash_attention_single_pos() {
        let brick = FlashAttentionBrick::new(2, 1, 4);
        let seq_len = 1;

        let query = vec![1.0f32; 2 * 4]; // 2 heads * 4 dim
        let keys = vec![1.0f32; 4]; // 1 seq * 1 kv_head * 4 dim
        let values = vec![1.0f32; 4];

        let output = brick
            .forward(&query, &keys, &values, seq_len)
            .expect("forward failed");
        assert_eq!(output.len(), 8);

        // With uniform inputs, output should be uniform
        for &v in &output {
            assert!(!v.is_nan());
            assert!(v.is_finite());
        }
    }

    // T032: CoalescedDp4aBrick FLOPS calculation
    #[test]
    #[cfg(feature = "cuda")]
    fn t032_coalesced_dp4a_flops() {
        let brick = CoalescedDp4aBrick::new(1024, 256);
        let flops = brick.flops();

        // GEMV: 2 * K * N
        assert_eq!(flops, 2 * 1024 * 256);
    }

    // T033: FusedFfnBrick with negative values
    #[test]
    #[cfg(feature = "cuda")]
    fn t033_fused_ffn_negative_values() {
        let brick = FusedFfnBrick::new(2, 4);

        let input = vec![-1.0f32; 2];
        let gate_proj = vec![1.0f32; 8];
        let up_proj = vec![1.0f32; 8];
        let down_proj = vec![1.0f32; 8];

        let output = brick
            .forward(&input, &gate_proj, &up_proj, &down_proj)
            .expect("forward failed");

        // SiLU of negative value should be small but not NaN
        for &v in &output {
            assert!(!v.is_nan());
            assert!(v.is_finite());
        }
    }

    // T034: CoalescedDp4aBrick with all-zero weights
    #[test]
    #[cfg(feature = "cuda")]
    fn t034_coalesced_dp4a_zero_weights() {
        let brick = CoalescedDp4aBrick::new(256, 4);

        let input_q8 = vec![1i8; 256];
        let input_scale = 1.0;
        // All 0x88 = nibbles (8,8) centered to (0,0) - effectively zero weights
        let weights_q4 = vec![0x88u8; 512];
        let weight_scales = vec![0.1f32; 4];

        let output = brick
            .forward(&input_q8, input_scale, &weights_q4, &weight_scales)
            .expect("forward failed");

        // With centered zero weights, output should be zero
        for &v in &output {
            assert_eq!(v, 0.0);
        }
    }

    // T035: ActivationQuantBrick per-channel mode
    #[test]
    fn t035_activation_quant_per_channel() {
        let per_tensor = ActivationQuantBrick::new(1024);
        let per_channel = ActivationQuantBrick::with_per_channel(1024);

        assert!(!per_tensor.per_channel);
        assert!(per_channel.per_channel);

        // Per-channel should have different (tighter) budget
        assert!(per_channel.budget().us_per_token > per_tensor.budget().us_per_token);
    }

    // T036: TransformerLayerBrick total_budget calculation
    #[test]
    fn t036_transformer_layer_total_budget() {
        let layer = TransformerLayerBrick::from_config(0, 64, 4, 2, 256, 1e-5, 10000.0, 0);

        let total = layer.total_budget_us();

        // Total should equal sum of components
        let sum = layer.attn_norm.budget().us_per_token
            + layer.qkv.budget().us_per_token
            + layer.rope.budget().us_per_token
            + layer.attention.budget().us_per_token
            + layer.o_proj.budget().us_per_token
            + layer.ffn_norm.budget().us_per_token
            + layer.ffn.budget().us_per_token;

        assert!((total - sum).abs() < 0.001);
    }

    // T037: BenchmarkReport display format
    #[test]
    fn t037_benchmark_report_display() {
        let report = BenchmarkReport {
            brick_name: "test_brick".to_string(),
            mean_us: 50.5,
            std_us: 5.2,
            cv: 0.103,
            p50_us: 50.0,
            p99_us: 62.0,
            tokens_per_sec: 19802.0,
            budget_us: 100.0,
            budget_met: true,
            statistically_valid: true,
        };

        let display = format!("{}", report);
        assert!(display.contains("test_brick"));
        assert!(display.contains("50.5"));
        assert!(display.contains("PASS"));
    }

    // T038: BenchmarkReport display FAIL case
    #[test]
    fn t038_benchmark_report_display_fail() {
        let report = BenchmarkReport {
            brick_name: "test".to_string(),
            mean_us: 150.0,
            std_us: 10.0,
            cv: 0.067,
            p50_us: 150.0,
            p99_us: 180.0,
            tokens_per_sec: 6667.0,
            budget_us: 100.0,
            budget_met: false,
            statistically_valid: true,
        };

        let display = format!("{}", report);
        assert!(display.contains("FAIL"));
    }

    // T039: FlashAttentionBrick GQA with different group sizes
    #[test]
    fn t039_flash_attention_gqa_groups() {
        // 8 query heads, 2 kv heads = group size 4
        let brick1 = FlashAttentionBrick::new(8, 2, 64);
        assert_eq!(brick1.group_size(), 4);

        // 8 query heads, 8 kv heads = group size 1 (MHA)
        let brick2 = FlashAttentionBrick::new(8, 8, 64);
        assert_eq!(brick2.group_size(), 1);

        // 8 query heads, 1 kv head = group size 8 (MQA)
        let brick3 = FlashAttentionBrick::new(8, 1, 64);
        assert_eq!(brick3.group_size(), 8);
    }

    // T040: ActivationQuantBrick large dimension
    #[test]
    fn t040_activation_quant_large_dim() {
        let brick = ActivationQuantBrick::new(4096);
        let input: Vec<f32> = (0..4096).map(|i| (i as f32 - 2048.0) / 100.0).collect();

        let (quants, scales) = brick.quantize(&input).expect("quantize failed");
        assert_eq!(quants.len(), 4096);
        assert_eq!(scales.len(), 128); // 4096 / 32 = 128 blocks

        let output = brick
            .dequantize(&quants, &scales)
            .expect("dequantize failed");
        assert_eq!(output.len(), 4096);

        // Verify roundtrip error is reasonable
        let error = brick
            .measure_error(&input, &quants, &scales)
            .expect("measure failed");
        assert!(error < 0.05, "Error {} should be < 5%", error);
    }

    // T041: QkvBrick with custom budget
    #[test]
    fn t041_qkv_brick_with_budget() {
        let custom_budget = TokenBudget::from_latency(20.0);
        let brick = QkvBrick::new(64, 64, 32, 32).with_budget(custom_budget);

        assert!((brick.budget().us_per_token - 20.0).abs() < 0.001);
    }

    // T042: RopeBrick with custom budget
    #[test]
    fn t042_rope_brick_with_budget() {
        let custom_budget = TokenBudget::from_latency(5.0);
        let brick = RopeBrick::new(64, 8, 10000.0, 0).with_budget(custom_budget);

        assert!((brick.budget().us_per_token - 5.0).abs() < 0.001);
    }

    // T043: AttentionBrick with custom budget
    #[test]
    fn t043_attention_brick_with_budget() {
        let custom_budget = TokenBudget::from_latency(25.0);
        let brick = AttentionBrick::new(8, 2, 64).with_budget(custom_budget);

        assert!((brick.budget().us_per_token - 25.0).abs() < 0.001);
    }

    // T044: OProjBrick with custom budget
    #[test]
    fn t044_oproj_brick_with_budget() {
        let custom_budget = TokenBudget::from_latency(8.0);
        let brick = OProjBrick::new(512, 64).with_budget(custom_budget);

        assert!((brick.budget().us_per_token - 8.0).abs() < 0.001);
    }

    // T045: FfnBrick with custom budget
    #[test]
    fn t045_ffn_brick_with_budget() {
        let custom_budget = TokenBudget::from_latency(30.0);
        let brick = FfnBrick::new(64, 256).with_budget(custom_budget);

        assert!((brick.budget().us_per_token - 30.0).abs() < 0.001);
    }

    // T047: TokenBudget edge case - very small latency
    #[test]
    fn t047_token_budget_small_latency() {
        let budget = TokenBudget::from_latency(0.001); // 1 nanosecond
        assert_eq!(budget.tokens_per_sec, 1_000_000_000.0);
        assert!(budget.is_met(0.0005));
        assert!(!budget.is_met(0.002));
    }

    // T048: TokenBudget edge case - very high throughput
    #[test]
    fn t048_token_budget_high_throughput() {
        let budget = TokenBudget::from_throughput(1_000_000.0); // 1M tokens/sec
        assert_eq!(budget.us_per_token, 1.0);
    }

    // T049: BrickVerification accumulate multiple failures
    #[test]
    fn t049_brick_verification_multiple_failures() {
        let mut v = BrickVerification::pass();
        assert!(v.is_valid);

        v.add("check1", false, "failed 1");
        assert!(!v.is_valid);

        v.add("check2", true, "passed");
        assert!(!v.is_valid); // Still invalid

        v.add("check3", false, "failed 3");
        assert!(!v.is_valid);

        assert_eq!(v.results.len(), 3);
    }

    // T050: ComputeBrick verify with zero budget
    #[test]
    fn t050_compute_brick_verify_zero_budget() {
        // Create a brick and manually set zero budget
        let mut brick = RmsNormBrick::new(vec![1.0; 4], 1e-5);
        brick.budget = TokenBudget::from_latency(0.0);

        // Verify should fail due to zero budget
        // Note: This creates a 0/0 situation in throughput
        // but the verify() method checks us_per_token <= 0.0
        let verification = brick.verify();
        assert!(!verification.is_valid);
    }

    // T051: FlashAttentionBrick with varying sequence lengths
    #[test]
    fn t051_flash_attention_varying_seq() {
        let brick = FlashAttentionBrick::new(4, 2, 8);

        for seq_len in [1, 2, 4, 8, 16, 32] {
            let query = vec![1.0f32; 4 * 8];
            let keys = vec![0.5f32; seq_len * 2 * 8];
            let values = vec![0.25f32; seq_len * 2 * 8];

            let output = brick
                .forward(&query, &keys, &values, seq_len)
                .unwrap_or_else(|_| panic!("forward failed for seq_len={}", seq_len));

            assert_eq!(output.len(), 4 * 8);
            assert!(output.iter().all(|&v| !v.is_nan()));
        }
    }
}
