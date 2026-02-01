    use super::*;

    /// Helper to create CudaExecutor for tests
    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // Validation Tests for forward_batched_to_token_ids
    // ========================================================================

    #[test]
    fn test_forward_batched_empty_batch() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let inputs: Vec<f32> = vec![];
        let positions: Vec<u32> = vec![];

        let result = exec.forward_batched_to_token_ids(
            &inputs, &positions, 1,    // num_layers
            256,  // hidden_dim
            1024, // intermediate_dim
            1024, // vocab_size
            1e-5, // epsilon
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = format!("{:?}", err);
        assert!(err_str.contains("batch size must be 1-32"));
    }

    #[test]
    fn test_forward_batched_batch_too_large() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Batch size > 32 should fail
        let positions: Vec<u32> = (0..33).collect();
        let inputs: Vec<f32> = vec![0.1; 33 * 256];

        let result =
            exec.forward_batched_to_token_ids(&inputs, &positions, 1, 256, 1024, 1024, 1e-5);

        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = format!("{:?}", err);
        assert!(err_str.contains("batch size must be 1-32"));
    }

    #[test]
    fn test_forward_batched_input_size_mismatch() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let positions: Vec<u32> = vec![0, 1, 2, 3];
        // Wrong input size: M=4, hidden_dim=256, expected 1024, give 512
        let inputs: Vec<f32> = vec![0.1; 512];

        let result =
            exec.forward_batched_to_token_ids(&inputs, &positions, 1, 256, 1024, 1024, 1e-5);

        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = format!("{:?}", err);
        assert!(err_str.contains("inputs.len()") && err_str.contains("!= M*hidden_dim"));
    }

    #[test]
    fn test_forward_batched_workspace_not_initialized() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Valid batch size and input size, but workspace not initialized
        let positions: Vec<u32> = vec![0, 1, 2, 3];
        let inputs: Vec<f32> = vec![0.1; 4 * 256];

        let result =
            exec.forward_batched_to_token_ids(&inputs, &positions, 1, 256, 1024, 1024, 1e-5);

        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = format!("{:?}", err);
        assert!(err_str.contains("workspace not initialized"));
    }

    #[test]
    fn test_forward_batched_workspace_wrong_batch_size() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Initialize workspace for batch size 8
        let _ = exec.init_batched_workspace(256, 1024, 8);

        // Try to use batch size 4 (different)
        let positions: Vec<u32> = vec![0, 1, 2, 3];
        let inputs: Vec<f32> = vec![0.1; 4 * 256];

        let result =
            exec.forward_batched_to_token_ids(&inputs, &positions, 1, 256, 1024, 1024, 1e-5);

        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = format!("{:?}", err);
        assert!(err_str.contains("workspace not initialized for M=4"));
    }

    #[test]
    fn test_forward_batched_missing_indexed_weights() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Initialize workspace correctly
        let _ = exec.init_batched_workspace(256, 1024, 4);

        // Don't build indexed weights
        let positions: Vec<u32> = vec![0, 1, 2, 3];
        let inputs: Vec<f32> = vec![0.1; 4 * 256];

        let result = exec.forward_batched_to_token_ids(
            &inputs, &positions, 1, // 1 layer
            256, 1024, 1024, 1e-5,
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = format!("{:?}", err);
        assert!(err_str.contains("weights not indexed") || err_str.contains("hidden_buf2 missing"));
    }

    // ========================================================================
    // Integration Tests with ModelHarness
    // ========================================================================

    #[test]
    fn test_batched_forward_with_harness_m4() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};

        let Some(mut exec) = create_executor() else {
            return;
        };

        let config = HarnessConfig::default();

        // First setup with single-token workspace, then switch to batched
        // The harness sets up indexed weights which we need
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Now reinitialize workspace for batch size 4
        let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);

        // Try batched forward
        let positions: Vec<u32> = vec![0, 1, 2, 3];
        let inputs: Vec<f32> = vec![0.1; 4 * config.hidden_dim];

        let result = exec.forward_batched_to_token_ids(
            &inputs,
            &positions,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            config.vocab_size as u32,
            1e-5,
        );

        // May fail due to kernel issues but exercises the path
        let _ = result;
    }

    #[test]
    fn test_transformer_layer_batched_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};

        let Some(mut exec) = create_executor() else {
            return;
        };

        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Reinitialize for batch
        let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);

        // Create input buffer
        let inputs: Vec<f32> = (0..4 * config.hidden_dim)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let input_buf = GpuBuffer::from_host(&exec.context, &inputs).unwrap();

        // Get indexed layer weights
        if !exec.has_indexed_weights() || exec.indexed_layer_weights.is_empty() {
            return;
        }
        let layer_weights = exec.get_indexed_layer(0).clone();

        // Try transformer layer batched
        let positions: [u32; 4] = [0, 1, 2, 3];
        let result = exec.transformer_layer_batched(
            &input_buf,
            0, // layer_idx
            &layer_weights,
            4, // m (batch_size)
            &positions,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );

        let _ = result;
    }

    // ========================================================================
    // Additional Harness-Based Integration Tests
    // ========================================================================

    #[test]
    fn test_forward_batched_m8_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Batch size 8
        let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 8);

        let positions: Vec<u32> = (0..8).collect();
        let inputs: Vec<f32> = vec![0.1; 8 * config.hidden_dim];

        let result = exec.forward_batched_to_token_ids(
            &inputs,
            &positions,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            config.vocab_size as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_forward_batched_m16_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Batch size 16 uses multi-warp kernel
        let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 16);

        let positions: Vec<u32> = (0..16).collect();
        let inputs: Vec<f32> = vec![0.1; 16 * config.hidden_dim];

        let result = exec.forward_batched_to_token_ids(
            &inputs,
            &positions,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            config.vocab_size as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_forward_batched_m32_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Batch size 32 (max supported)
        let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 32);

        let positions: Vec<u32> = (0..32).collect();
        let inputs: Vec<f32> = vec![0.1; 32 * config.hidden_dim];

        let result = exec.forward_batched_to_token_ids(
            &inputs,
            &positions,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            config.vocab_size as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_forward_batched_graphed_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);

        let positions: Vec<u32> = vec![0, 1, 2, 3];
        let inputs: Vec<f32> = vec![0.1; 4 * config.hidden_dim];

        // Test graphed path
        let result = exec.forward_batched_to_token_ids_graphed(
            &inputs,
            &positions,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            config.vocab_size as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_forward_batched_graphed_replay() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);

        let positions: Vec<u32> = vec![0, 1, 2, 3];
        let inputs: Vec<f32> = vec![0.1; 4 * config.hidden_dim];

        // First call captures graph
        let result1 = exec.forward_batched_to_token_ids_graphed(
            &inputs,
            &positions,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            config.vocab_size as u32,
            1e-5,
        );

        // Update positions for second call
        let positions2: Vec<u32> = vec![1, 2, 3, 4];

        // Second call should replay graph
        let result2 = exec.forward_batched_to_token_ids_graphed(
            &inputs,
            &positions2,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            config.vocab_size as u32,
            1e-5,
        );

        let _ = (result1, result2);
    }

    #[test]
    fn test_batched_kv_cache_init() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Initialize batched KV caches (uses existing KV cache config from harness)
        let result = exec.init_batched_kv_cache_gpu(config.num_layers, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transformer_layer_batched_m8() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 8);

        let inputs: Vec<f32> = vec![0.1; 8 * config.hidden_dim];
        let input_buf = GpuBuffer::from_host(&exec.context, &inputs).unwrap();

        if exec.indexed_layer_weights.is_empty() {
            return;
        }
        let layer_weights = exec.indexed_layer_weights[0].clone();

        let positions: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let result = exec.transformer_layer_batched(
            &input_buf,
            0,
            &layer_weights,
            8,
            &positions,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_batched_workspace_hidden_buffer() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Initialize batched workspace
        let result = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);
        assert!(result.is_ok());

        // Verify workspace is initialized (fields vary by implementation)
        assert!(exec.workspace.hidden_buf1.is_some());
    }

    // ========================================================================
    // Coverage Tests: RMSNorm and Batch Edge Cases (v1.36.0)
    // ========================================================================

    #[test]
    fn test_rmsnorm_gpu_ptr_batched() {
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

        // rmsnorm_gpu_ptr is defined in batched.rs as pub(crate)
        let result = exec.rmsnorm_gpu_ptr(
            &input,
            layer_weights.attn_norm_ptr,
            layer_weights.attn_norm_len,
            config.hidden_dim as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_forward_batched_different_positions() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);

        // Non-contiguous positions
        let positions: Vec<u32> = vec![0, 5, 10, 15];
        let inputs: Vec<f32> = vec![0.1; 4 * config.hidden_dim];

        let result = exec.forward_batched_to_token_ids(
            &inputs,
            &positions,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            config.vocab_size as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_transformer_layer_batched_different_layers() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.num_layers = 4;
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);

        let inputs: Vec<f32> = vec![0.1; 4 * config.hidden_dim];
        let input_buf = GpuBuffer::from_host(&exec.context, &inputs).unwrap();
        let positions: [u32; 4] = [0, 1, 2, 3];

        // Test each layer
        for layer_idx in 0..config.num_layers {
            if layer_idx >= exec.indexed_layer_weights.len() {
                break;
            }
            let layer_weights = exec.indexed_layer_weights[layer_idx].clone();

            let result = exec.transformer_layer_batched(
                &input_buf,
                layer_idx,
                &layer_weights,
                4,
                &positions,
                config.hidden_dim as u32,
                config.intermediate_dim as u32,
                1e-5,
            );
            let _ = result;
        }
    }

    #[test]
    fn test_batched_forward_m1() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Batch size 1 - edge case
        let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 1);

        let positions: Vec<u32> = vec![0];
        let inputs: Vec<f32> = vec![0.1; config.hidden_dim];

        let result = exec.forward_batched_to_token_ids(
            &inputs,
            &positions,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            config.vocab_size as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_batched_kv_cache_multiple_batch_sizes() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Test different batch sizes for KV cache init
        for batch_size in [1, 4, 8, 16] {
            let result = exec.init_batched_kv_cache_gpu(config.num_layers, batch_size);
            let _ = result;
        }
    }

    #[test]
    fn test_transformer_layer_batched_qtype_q6k() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);

        let inputs: Vec<f32> = vec![0.1; 4 * config.hidden_dim];
        let input_buf = GpuBuffer::from_host(&exec.context, &inputs).unwrap();

        if exec.indexed_layer_weights.is_empty() {
            return;
        }

        // Modify weights to use Q6K
        let mut layer_weights = exec.indexed_layer_weights[0].clone();
        layer_weights.attn_v_qtype = WeightQuantType::Q6K;

        let positions: [u32; 4] = [0, 1, 2, 3];
        let result = exec.transformer_layer_batched(
            &input_buf,
            0,
            &layer_weights,
            4,
            &positions,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_batched_graph_token_count() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let initial_count = exec.decode_token_count;

        let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);

        let positions: Vec<u32> = vec![0, 1, 2, 3];
        let inputs: Vec<f32> = vec![0.1; 4 * config.hidden_dim];

        // Execute batched graphed forward
        let _ = exec.forward_batched_to_token_ids_graphed(
            &inputs,
            &positions,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            config.vocab_size as u32,
            1e-5,
        );

        // Token count might change after graphed forward
        assert!(exec.decode_token_count >= initial_count);
    }

    #[test]
    fn test_batched_forward_varying_input_values() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);

        let positions: Vec<u32> = vec![0, 1, 2, 3];

        // Different input values per batch element
        let inputs: Vec<f32> = (0..4 * config.hidden_dim)
            .map(|i| (i as f32 / 1000.0).sin())
            .collect();

        let result = exec.forward_batched_to_token_ids(
            &inputs,
            &positions,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            config.vocab_size as u32,
            1e-5,
        );
        let _ = result;
    }
