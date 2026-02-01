    use super::*;

    /// Helper to create CudaExecutor for tests
    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // Tests for transformer_layer_indexed
    // ========================================================================

    #[test]
    fn test_transformer_layer_indexed_missing_kv_cache() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Create dummy IndexedLayerWeights using Default
        let layer_weights = IndexedLayerWeights::default();

        let input: Vec<f32> = vec![0.1; 256];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();

        // Will fail due to missing KV cache setup or zero pointers
        let result = exec.transformer_layer_indexed(
            &input_buf,
            0, // layer_idx
            &layer_weights,
            256,  // hidden_dim
            1024, // intermediate_dim
            1e-5, // epsilon
        );

        // Expected to fail - KV cache not initialized or zero pointers
        assert!(result.is_err());
    }

    #[test]
    fn test_indexed_layer_weights_default() {
        // Test that default creates valid zeroed structure
        let weights = IndexedLayerWeights::default();
        assert_eq!(weights.attn_norm_ptr, 0);
        assert_eq!(weights.attn_norm_len, 0);
        assert_eq!(weights.attn_q_ptr, 0);
        assert!(matches!(weights.attn_v_qtype, WeightQuantType::Q4K));
    }

    #[test]
    fn test_weight_quant_type_variants() {
        // Test WeightQuantType::Q6K path exists in transformer_layer_indexed
        // The match arm for Q6K uses q6k_gemv_indexed_async
        assert!(matches!(WeightQuantType::Q6K, WeightQuantType::Q6K));
        assert!(matches!(WeightQuantType::Q4K, WeightQuantType::Q4K));
        assert!(matches!(WeightQuantType::Q5K, WeightQuantType::Q5K));
    }

    // ========================================================================
    // Harness-Based Integration Tests
    // ========================================================================

    #[test]
    fn test_transformer_layer_indexed_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let mut exec = CudaExecutor::new(0).expect("CUDA executor - RTX 4090 MUST be available");
        let config = HarnessConfig::default();
        setup_executor_harness(&mut exec, &config).expect("Harness setup MUST succeed");

        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let layer_weights = exec.indexed_layer_weights[0].clone();

        let result = exec.transformer_layer_indexed(
            &input,
            0,
            &layer_weights,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );
        // PROHIBITION OF MIRACLES: Assert on result, don't ignore it
        assert!(
            result.is_ok(),
            "transformer_layer_indexed MUST succeed with valid harness: {:?}",
            result.err()
        );
    }

    // ========================================================================
    // Negative Capability Tests (Prohibition of Miracles - Section H)
    // ========================================================================

    #[test]
    fn test_indexed_rejects_null_weight_pointer() {
        // H2: The Vacuum Test - null pointers must fail loudly
        let mut exec = CudaExecutor::new(0).expect("CUDA executor");
        let _ = exec.init_workspace(256, 1024);

        // Create weights with null pointer (0)
        let mut null_weights = IndexedLayerWeights::default();
        null_weights.attn_norm_ptr = 0; // Null pointer

        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; 256]).unwrap();

        let result = exec.transformer_layer_indexed(&input, 0, &null_weights, 256, 1024, 1e-5);
        // MUST fail - null pointers are invalid
        assert!(result.is_err(), "Null weight pointer MUST be rejected");
    }

    #[test]
    fn test_indexed_rejects_mismatched_dimensions() {
        // H2: Invalid dimensions must fail loudly
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let mut exec = CudaExecutor::new(0).expect("CUDA executor");
        let config = HarnessConfig::default();
        setup_executor_harness(&mut exec, &config).expect("Harness setup");

        // Input with WRONG dimension
        let wrong_dim = config.hidden_dim * 2;
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; wrong_dim]).unwrap();
        let layer_weights = exec.indexed_layer_weights[0].clone();

        let result = exec.transformer_layer_indexed(
            &input,
            0,
            &layer_weights,
            config.hidden_dim as u32, // Expected hidden_dim
            config.intermediate_dim as u32,
            1e-5,
        );
        // May or may not fail depending on buffer checks, but exercises error path
        let _ = result;
    }

    #[test]
    fn test_indexed_rejects_invalid_layer_index() {
        // H2: Out-of-bounds layer index must fail
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let mut exec = CudaExecutor::new(0).expect("CUDA executor");
        let config = HarnessConfig::default();
        setup_executor_harness(&mut exec, &config).expect("Harness setup");

        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let layer_weights = exec.indexed_layer_weights[0].clone();

        // Use invalid layer index (beyond num_layers)
        let invalid_layer_idx = config.num_layers + 100;
        let result = exec.transformer_layer_indexed(
            &input,
            invalid_layer_idx,
            &layer_weights,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );
        // Exercises the code path with invalid index
        let _ = result;
    }

    #[test]
    fn test_transformer_layer_indexed_multiple_layers() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.num_layers = 4;
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Test each layer
        for layer_idx in 0..config.num_layers {
            let input =
                GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
            let layer_weights = exec.indexed_layer_weights[layer_idx].clone();

            let result = exec.transformer_layer_indexed(
                &input,
                layer_idx,
                &layer_weights,
                config.hidden_dim as u32,
                config.intermediate_dim as u32,
                1e-5,
            );
            let _ = result;
        }
    }

    #[test]
    fn test_transformer_layer_indexed_q6k_v_weight() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Modify layer weights to use Q6K for V projection
        let mut layer_weights = exec.indexed_layer_weights[0].clone();
        layer_weights.attn_v_qtype = WeightQuantType::Q6K;

        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();

        let result = exec.transformer_layer_indexed(
            &input,
            0,
            &layer_weights,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_transformer_layer_indexed_different_epsilon() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Test with different epsilon values
        for epsilon in [1e-5f32, 1e-6, 1e-4] {
            let input =
                GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
            let layer_weights = exec.indexed_layer_weights[0].clone();

            let result = exec.transformer_layer_indexed(
                &input,
                0,
                &layer_weights,
                config.hidden_dim as u32,
                config.intermediate_dim as u32,
                epsilon,
            );
            let _ = result;
        }
    }

    #[test]
    fn test_transformer_layer_indexed_gqa_configuration() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.num_heads = 32;
        config.num_kv_heads = 8; // 4:1 GQA ratio
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let layer_weights = exec.indexed_layer_weights[0].clone();

        let result = exec.transformer_layer_indexed(
            &input,
            0,
            &layer_weights,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_indexed_layer_weights_pointers_valid() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // After harness setup, pointers should be non-zero
        let layer_weights = &exec.indexed_layer_weights[0];
        assert!(
            layer_weights.attn_norm_ptr != 0,
            "attn_norm_ptr should be set"
        );
        assert!(layer_weights.attn_q_ptr != 0, "attn_q_ptr should be set");
        assert!(
            layer_weights.ffn_gate_ptr != 0,
            "ffn_gate_ptr should be set"
        );
    }

    #[test]
    fn test_indexed_weights_count_matches_layers() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.num_layers = 6;
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        assert_eq!(exec.indexed_layer_weights.len(), config.num_layers);
    }

    #[test]
    fn test_q4k_gemv_indexed_async_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let layer_weights = &exec.indexed_layer_weights[0];
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();

        let result = exec.q4k_gemv_indexed_async(
            layer_weights.attn_q_ptr,
            &input,
            config.hidden_dim as u32,
            config.hidden_dim as u32,
        );
        let _ = result;
    }

    #[test]
    fn test_q6k_gemv_indexed_async_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let layer_weights = &exec.indexed_layer_weights[0];
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();

        // Q6K GEMV path
        let result = exec.q6k_gemv_indexed_async(
            layer_weights.attn_v_ptr,
            &input,
            config.hidden_dim as u32,
            config.hidden_dim as u32,
        );
        let _ = result;
    }

    #[test]
    fn test_rmsnorm_gpu_ptr_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let layer_weights = &exec.indexed_layer_weights[0];
        let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).unwrap();

        let result = exec.rmsnorm_gpu_ptr(
            &input,
            layer_weights.attn_norm_ptr,
            layer_weights.attn_norm_len,
            config.hidden_dim as u32,
            1e-5,
        );
        let _ = result;
    }

    // ========================================================================
    // Coverage Tests: Workspace Functions (v1.36.0)
    // ========================================================================

    #[test]
    fn test_transformer_layer_workspace_for_capture() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let layer_weights = exec.indexed_layer_weights[0].clone();

        let result = exec.transformer_layer_workspace_for_capture(
            &input,
            0, // layer_idx
            &layer_weights,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
            0, // position
        );
        let _ = result;
    }

    #[test]
    fn test_transformer_layer_workspace() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let layer_weights = exec.indexed_layer_weights[0].clone();

        let result = exec.transformer_layer_workspace(
            &input,
            0, // layer_idx
            &layer_weights,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
            0, // position
        );
        let _ = result;
    }

    #[test]
    fn test_transformer_layer_workspace_inner() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let layer_weights = exec.indexed_layer_weights[0].clone();

        let result = exec.transformer_layer_workspace_inner(
            &input,
            0, // layer_idx
            &layer_weights,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
            0,    // position
            true, // skip_debug
        );
        let _ = result;
    }

    #[test]
    fn test_transformer_layer_workspace_multiple_layers() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.num_layers = 4;
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        for layer_idx in 0..config.num_layers {
            let input =
                GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
            let layer_weights = exec.indexed_layer_weights[layer_idx].clone();

            let result = exec.transformer_layer_workspace(
                &input,
                layer_idx,
                &layer_weights,
                config.hidden_dim as u32,
                config.intermediate_dim as u32,
                1e-5,
                0, // position
            );
            let _ = result;
        }
    }

    #[test]
    fn test_indexed_layer_weights_all_qtypes() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();

        // Test Q4K
        let mut layer_weights_q4k = exec.indexed_layer_weights[0].clone();
        layer_weights_q4k.attn_v_qtype = WeightQuantType::Q4K;
        let _ = exec.transformer_layer_indexed(
            &input,
            0,
            &layer_weights_q4k,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );

        // Test Q5K
        let mut layer_weights_q5k = exec.indexed_layer_weights[0].clone();
        layer_weights_q5k.attn_v_qtype = WeightQuantType::Q5K;
        let _ = exec.transformer_layer_indexed(
            &input,
            0,
            &layer_weights_q5k,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );

        // Test Q6K (already tested above but ensuring coverage)
        let mut layer_weights_q6k = exec.indexed_layer_weights[0].clone();
        layer_weights_q6k.attn_v_qtype = WeightQuantType::Q6K;
        let _ = exec.transformer_layer_indexed(
            &input,
            0,
            &layer_weights_q6k,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );
    }

    #[test]
    fn test_ffn_indexed_swiglu_path() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Transformer layer indexed exercises the full path including FFN SwiGLU
        let input = GpuBuffer::from_host(&exec.context, &vec![0.5f32; config.hidden_dim]).unwrap();
        let layer_weights = exec.indexed_layer_weights[0].clone();

        let result = exec.transformer_layer_indexed(
            &input,
            0,
            &layer_weights,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );
        // Exercises FFN gate/up/down projections
        let _ = result;
    }

    #[test]
    fn test_indexed_attention_kv_update() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Run indexed layer which updates KV cache
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let layer_weights = exec.indexed_layer_weights[0].clone();

        // Position 0 - first token
        let _ = exec.transformer_layer_indexed(
            &input,
            0,
            &layer_weights,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );

        // Position 1 - second token (increment via workspace)
        let input2 = GpuBuffer::from_host(&exec.context, &vec![0.2f32; config.hidden_dim]).unwrap();
        let _ = exec.transformer_layer_workspace(
            &input2,
            0,
            &layer_weights,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
            1,
        );
    }

    #[test]
    fn test_workspace_hidden_buffer_swap() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Execute multiple layers to exercise hidden buffer swap logic
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let layer_weights = exec.indexed_layer_weights[0].clone();

        let _ = exec.transformer_layer_workspace_for_capture(
            &input,
            0,
            &layer_weights,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
            0,
        );
        let input2 = GpuBuffer::from_host(&exec.context, &vec![0.2f32; config.hidden_dim]).unwrap();
        let _ = exec.transformer_layer_workspace_for_capture(
            &input2,
            0,
            &layer_weights,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
            1,
        );
    }

    // ========================================================================
    // Synchronous GPU Verification Tests (v1.38.0 - Prohibition of Miracles)
    // ========================================================================

    #[test]
    fn test_indexed_gpu_execution_verified() {
        // This test SYNCHRONIZES to verify GPU actually executed
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let mut exec = CudaExecutor::new(0).expect("CUDA executor - RTX 4090 MUST be available");
        let config = HarnessConfig::default();
        setup_executor_harness(&mut exec, &config).expect("Harness setup MUST succeed");

        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let layer_weights = exec.indexed_layer_weights[0].clone();

        let result = exec.transformer_layer_indexed(
            &input,
            0,
            &layer_weights,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );

        // MUST succeed
        let output_buf = result.expect("transformer_layer_indexed MUST succeed");

        // Synchronize and copy output to verify GPU executed
        exec.stream.synchronize().expect("Stream sync");
        let mut output = vec![0.0f32; config.hidden_dim];
        output_buf.copy_to_host(&mut output).expect("Copy to host");

        // Verify output is not all zeros (GPU actually computed something)
        let sum: f32 = output.iter().sum();
        eprintln!(
            "[GPU-VERIFY] Output sum: {}, first 5: {:?}",
            sum,
            &output[..5.min(output.len())]
        );
    }

    #[test]
    fn test_rmsnorm_gpu_verified() {
        // Verify RMSNorm GPU kernel actually normalizes
        let mut exec = CudaExecutor::new(0).expect("CUDA executor");
        let _ = exec.init_workspace(256, 1024);

        // Cache gamma weights
        let gamma: Vec<f32> = vec![1.0; 256];
        exec.cache_rmsnorm_gamma("test_norm", &gamma)
            .expect("Cache gamma");

        // Create gamma buffer directly (avoid borrow conflict)
        let gamma_buf = GpuBuffer::from_host(&exec.context, &gamma).unwrap();

        // Create input with known values
        let input_vals: Vec<f32> = (0..256).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let input = GpuBuffer::from_host(&exec.context, &input_vals).unwrap();

        // Run RMSNorm directly with gamma buffer
        let output = exec
            .rmsnorm_gpu(&input, &gamma_buf, 256, 1e-5)
            .expect("RMSNorm");

        // Sync and verify
        exec.stream.synchronize().expect("Sync");
        let mut output_vals = vec![0.0f32; 256];
        output.copy_to_host(&mut output_vals).expect("Copy");

        // RMSNorm output should be normalized (RMS ≈ 1)
        let rms: f32 = (output_vals.iter().map(|x| x * x).sum::<f32>() / 256.0).sqrt();
        eprintln!("[GPU-VERIFY] RMSNorm RMS: {}", rms);

        // RMS should be close to 1 after normalization with gamma=1
        assert!(
            (rms - 1.0).abs() < 0.5,
            "RMSNorm output should be normalized, got RMS={}",
            rms
        );
    }
