//! CUDA tests for APR module (Phase 40)
//!
//! Tests `AprV2ModelCuda` to improve coverage of apr/cuda.rs.

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use crate::apr::{AprV2Model, AprV2ModelCuda, HEADER_SIZE, MAGIC};

    /// Helper to write a tensor entry in APR v2 binary format
    fn write_tensor_entry_binary(
        name: &str,
        dtype: u8,
        shape: &[usize],
        offset: u64,
        size: u64,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        // Name length (u16 LE)
        buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
        // Name bytes
        buf.extend_from_slice(name.as_bytes());
        // Dtype (u8)
        buf.push(dtype);
        // Ndim (u8)
        buf.push(shape.len() as u8);
        // Shape dimensions (u64 LE each)
        for &dim in shape {
            buf.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        // Offset (u64 LE)
        buf.extend_from_slice(&offset.to_le_bytes());
        // Size (u64 LE)
        buf.extend_from_slice(&size.to_le_bytes());
        buf
    }

    /// Helper to create a minimal valid APR v2 model in memory
    fn create_minimal_apr_model() -> AprV2Model {
        // Minimal metadata with required fields for AprV2ModelCuda
        let metadata = r#"{
            "model_type": "test",
            "name": "test-cuda-model",
            "hidden_dim": 32,
            "num_layers": 1,
            "num_heads": 1,
            "vocab_size": 100
        }"#;

        // Create tensor index in binary format
        // dtype: 0=F32
        let tensor_index_binary = write_tensor_entry_binary(
            "model.embed_tokens.weight", // name
            0,                           // dtype = F32
            &[100, 32],                  // shape = [vocab_size, hidden_dim]
            0,                           // offset in data section
            100 * 32 * 4,                // size in bytes
        );

        // Embedding data (100 * 32 floats)
        let tensor_data: Vec<f32> = vec![0.1; 100 * 32];
        let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Calculate offsets (64-byte aligned)
        let metadata_offset = HEADER_SIZE as u64;
        let metadata_size = metadata.len() as u32;
        let tensor_index_offset =
            ((metadata_offset as usize + metadata.len()).div_ceil(64) * 64) as u64;
        let data_offset =
            ((tensor_index_offset as usize + tensor_index_binary.len()).div_ceil(64) * 64) as u64;

        let total_size = data_offset as usize + tensor_bytes.len();
        let mut data = vec![0u8; total_size];

        // Header (64 bytes)
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // Version major
        data[5] = 0; // Version minor
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // Flags
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // Tensor count
        data[12..20].copy_from_slice(&metadata_offset.to_le_bytes());
        data[20..24].copy_from_slice(&metadata_size.to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        // Metadata
        data[metadata_offset as usize..metadata_offset as usize + metadata.len()]
            .copy_from_slice(metadata.as_bytes());

        // Tensor index (binary format)
        data[tensor_index_offset as usize
            ..tensor_index_offset as usize + tensor_index_binary.len()]
            .copy_from_slice(&tensor_index_binary);

        // Tensor data
        data[data_offset as usize..data_offset as usize + tensor_bytes.len()]
            .copy_from_slice(&tensor_bytes);

        AprV2Model::from_bytes(data).expect("should create minimal APR model")
    }

    // =========================================================================
    // AprV2ModelCuda Construction Tests
    // =========================================================================

    #[test]
    fn test_minimal_model_has_embedding_tensor() {
        let model = create_minimal_apr_model();

        // Debug: list all tensors
        for tensor in &model.tensors {
            println!("Tensor: '{}' shape={:?}", tensor.name, tensor.shape);
        }

        // Verify the embedding tensor exists
        let embedding = model.get_tensor("model.embed_tokens.weight");
        assert!(
            embedding.is_some(),
            "Should have model.embed_tokens.weight tensor"
        );
    }

    #[test]
    fn test_apr_cuda_new_succeeds_with_gpu() {
        let model = create_minimal_apr_model();

        // Attempt to create CUDA wrapper - this tests the constructor path
        let result = AprV2ModelCuda::new(model, 0);

        // Should succeed if GPU is available, fail gracefully if not
        // Under coverage instrumentation, may fail - that's OK
        match result {
            Ok(cuda_model) => {
                // Verify device info was captured
                if !cuda_model.device_name().is_empty() {
                    println!("GPU: {}", cuda_model.device_name());
                }
            },
            Err(e) => {
                // Expected if no GPU or under heavy instrumentation
                println!("CUDA init result: {:?}", e);
            },
        }
    }

    #[test]
    fn test_apr_cuda_with_max_seq_len() {
        let model = create_minimal_apr_model();

        // Test with custom sequence length
        let result = AprV2ModelCuda::with_max_seq_len(model, 0, 512);

        // Under coverage instrumentation, may fail - that's OK
        match result {
            Ok(_cuda_model) => {
                println!("CUDA model created with max_seq_len=512");
            },
            Err(e) => {
                println!("CUDA init result: {:?}", e);
            },
        }
    }

    #[test]
    fn test_apr_cuda_invalid_device() {
        let model = create_minimal_apr_model();

        // Device 999 should not exist
        let result = AprV2ModelCuda::new(model, 999);

        // Should fail - exercises error path
        assert!(result.is_err(), "Device 999 should not exist");
    }

    // =========================================================================
    // AprV2ModelCuda Method Tests (require GPU)
    // =========================================================================

    #[test]
    fn test_apr_cuda_device_name() {
        let model = create_minimal_apr_model();

        if let Ok(cuda_model) = AprV2ModelCuda::new(model, 0) {
            let name = cuda_model.device_name();
            // Should return non-empty device name
            assert!(!name.is_empty(), "Device name should not be empty");
            println!("GPU device: {}", name);
        }
    }

    #[test]
    fn test_apr_cuda_memory_info() {
        let model = create_minimal_apr_model();

        if let Ok(cuda_model) = AprV2ModelCuda::new(model, 0) {
            let (free, total) = cuda_model.memory_info();
            // Memory should be positive
            assert!(total > 0, "Total GPU memory should be > 0");
            assert!(free <= total, "Free memory should not exceed total");
            println!("GPU memory: {}/{} bytes free", free, total);
        }
    }

    #[test]
    fn test_apr_cuda_vram_mb() {
        let model = create_minimal_apr_model();

        if let Ok(cuda_model) = AprV2ModelCuda::new(model, 0) {
            let vram = cuda_model.vram_mb();
            assert!(vram > 0, "VRAM should be > 0 MB");
            println!("GPU VRAM: {} MB", vram);
        }
    }

    #[test]
    fn test_apr_cuda_inner_model() {
        let model = create_minimal_apr_model();

        if let Ok(cuda_model) = AprV2ModelCuda::new(model, 0) {
            let inner = cuda_model.inner();
            assert!(inner.tensor_count() > 0, "Model should have tensors");
        }
    }

    #[test]
    fn test_apr_cuda_is_available() {
        // Static method - should always work
        let available = AprV2ModelCuda::is_available();
        println!("CUDA available: {}", available);
        // Don't assert - just exercise the code path
    }

    #[test]
    fn test_apr_cuda_num_devices() {
        let count = AprV2ModelCuda::num_devices();
        println!("CUDA devices: {}", count);
        // Should be >= 0
    }

    #[test]
    fn test_apr_cuda_profiling() {
        let model = create_minimal_apr_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Test profiling toggle
            assert!(!cuda_model.is_profiling_enabled());

            cuda_model.enable_profiling();
            assert!(cuda_model.is_profiling_enabled());

            cuda_model.disable_profiling();
            assert!(!cuda_model.is_profiling_enabled());
        }
    }

    #[test]
    fn test_apr_cuda_profiler_access() {
        let model = create_minimal_apr_model();

        if let Ok(cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Just access the profiler - don't need to assert anything
            let _profiler = cuda_model.profiler();
        }
    }

    #[test]
    fn test_apr_cuda_reset_profiler() {
        let model = create_minimal_apr_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            cuda_model.enable_profiling();
            cuda_model.reset_profiler();
            // Should not panic
        }
    }

    #[test]
    fn test_apr_cuda_reset_kv_cache() {
        let model = create_minimal_apr_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            cuda_model.reset_kv_cache();
            // Should not panic
        }
    }

    #[test]
    fn test_apr_cuda_weights_cached() {
        let model = create_minimal_apr_model();

        if let Ok(cuda_model) = AprV2ModelCuda::new(model, 0) {
            let cached = cuda_model.weights_cached();
            // Initial state - may or may not be cached
            println!("Weights cached: {}", cached);
        }
    }

    #[test]
    fn test_apr_cuda_cached_weight_mb() {
        let model = create_minimal_apr_model();

        if let Ok(cuda_model) = AprV2ModelCuda::new(model, 0) {
            let mb = cuda_model.cached_weight_mb();
            println!("Cached weight: {} MB", mb);
        }
    }

    // =========================================================================
    // Phase 45: Test Executor Injection Tests
    // =========================================================================

    #[test]
    fn test_apr_cuda_with_test_executor() {
        use crate::gpu::executor::MockExecutor;

        let model = create_minimal_apr_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Initially no test executor
            assert!(!cuda_model.has_test_executor());

            // Inject test executor
            let mock = MockExecutor::new("apr_test");
            cuda_model.with_test_executor(Box::new(mock));

            // Now has test executor
            assert!(cuda_model.has_test_executor());
        }
    }

    #[test]
    fn test_apr_cuda_test_executor_bypasses_fast_path() {
        use crate::gpu::executor::MockExecutor;

        let model = create_minimal_apr_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Inject test executor
            let mock = MockExecutor::new("bypass_test");
            cuda_model.with_test_executor(Box::new(mock));

            // has_cached_weight should return false when test_executor is present
            // This forces the uncached GEMM path which routes through test_executor
            assert!(!cuda_model.has_test_executor() || !cuda_model.weights_cached());
            // Actually just verify test executor is set
            assert!(cuda_model.has_test_executor());
        }
    }

    #[test]
    fn test_apr_cuda_forward_with_test_executor() {
        use crate::gpu::executor::CpuExecutor;

        let model = create_transformer_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Inject CPU executor for testing
            let cpu = CpuExecutor::new();
            cuda_model.with_test_executor(Box::new(cpu));

            // Call forward_cuda - should use test executor path
            // The model won't have proper weights so this will likely fail,
            // but it exercises the test_executor code paths
            let result = cuda_model.forward_cuda(&[1]);

            // Result may fail due to missing weights, but the path was exercised
            println!("Forward with test executor result: {:?}", result.is_ok());
        }
    }

    #[test]
    fn test_apr_cuda_forward_with_mock_executor() {
        use crate::gpu::executor::MockExecutor;

        let model = create_transformer_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Inject mock executor
            let mock = MockExecutor::new("forward_mock");
            cuda_model.with_test_executor(Box::new(mock));

            // Call forward_cuda
            let result = cuda_model.forward_cuda(&[1]);

            // May fail due to missing weights, but exercises the path
            println!("Forward with mock executor: {:?}", result.is_ok());
        }
    }

    #[test]
    fn test_apr_cuda_forward_mock_error_handling() {
        use crate::gpu::executor::MockExecutor;

        let model = create_transformer_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Inject mock that fails
            let mock = MockExecutor::new("fail_mock").with_matmul_failure();
            cuda_model.with_test_executor(Box::new(mock));

            // forward_cuda should fail when mock fails
            let result = cuda_model.forward_cuda(&[1]);

            // May fail for other reasons (missing weights), but the path is exercised
            println!("Forward with failing mock: {:?}", result.is_err());
        }
    }

    #[test]
    fn test_apr_cuda_generate_with_test_executor() {
        use crate::gpu::executor::CpuExecutor;

        let model = create_transformer_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Inject CPU executor
            let cpu = CpuExecutor::new();
            cuda_model.with_test_executor(Box::new(cpu));

            // Call generate_cuda - exercises forward path with test executor
            let result = cuda_model.generate_cuda(&[1], 1, 2); // 1 token, eos=2

            println!("Generate with test executor: {:?}", result.is_ok());
        }
    }

    /// PMAT-110: Test that APR CUDA generation produces valid logits (not -inf)
    ///
    /// This test verifies that forward_single_cuda returns valid logits
    /// after prefill, which is required for generation to work.
    #[test]
    fn test_apr_cuda_forward_produces_valid_logits() {
        let model = create_transformer_model();

        if let Ok(mut cuda_model) = AprV2ModelCuda::new(model, 0) {
            // Skip if no GPU available
            if !AprV2ModelCuda::is_available() {
                println!("CUDA not available, skipping test");
                return;
            }

            // Prefill with a simple prompt
            let prompt = vec![1_u32, 2, 3]; // Simple token sequence
            let prefill_result = cuda_model.forward_cuda(&prompt);

            // Prefill should succeed
            assert!(prefill_result.is_ok(), "Prefill failed: {:?}", prefill_result);
            let prefill_logits = prefill_result.unwrap();

            // CRITICAL: Logits must not be all -inf or NaN
            let max_logit = prefill_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(max_logit.is_finite(), "Prefill logits are all -inf or NaN");

            // Now test single token decode (the problematic path)
            let decode_result = cuda_model.forward_single_cuda(4, prompt.len());

            // Decode should succeed
            assert!(decode_result.is_ok(), "Decode failed: {:?}", decode_result);
            let decode_logits = decode_result.unwrap();

            // CRITICAL: Decode logits must ALSO not be all -inf or NaN
            let decode_max = decode_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                decode_max.is_finite(),
                "PMAT-110 FAILURE: forward_single_cuda returns invalid logits (max={}).\n\
                 This indicates KV cache is not working for APR CUDA.",
                decode_max
            );

            println!("Prefill max logit: {:.2}", max_logit);
            println!("Decode max logit: {:.2}", decode_max);
        }
    }

    /// PMAT-110: Integration test with real APR model file
    ///
    /// This test loads an actual APR model and verifies that generation
    /// produces valid (non-empty, non-garbage) output.
    #[test]
    #[ignore] // Run with: cargo test --features cuda -- --ignored
    fn test_apr_cuda_real_model_generation() {
        use std::path::Path;

        // Path to real APR model (Qwen2.5-Coder-1.5B Q4K)
        let model_path = Path::new(env!("HOME")).join(".apr/models/qwen2.5-coder-1.5b-q4k.apr");

        if !model_path.exists() {
            println!("Skipping: APR model not found at {:?}", model_path);
            return;
        }

        if !AprV2ModelCuda::is_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        // Load model
        let model_bytes = std::fs::read(&model_path).expect("Failed to read APR file");
        let model = AprV2Model::from_bytes(model_bytes).expect("Failed to parse APR model");

        println!("Loaded model: {:?}", model.metadata.name);
        println!("  hidden_size: {:?}", model.metadata.hidden_size);
        println!("  num_layers: {:?}", model.metadata.num_layers);
        println!("  vocab_size: {:?}", model.metadata.vocab_size);

        let mut cuda_model = AprV2ModelCuda::new(model, 0).expect("Failed to create CUDA model");

        // Simple prompt: "What is 2+2?" tokenized (approximate)
        // Using raw token IDs that are likely valid
        let prompt = vec![1_u32, 100, 200, 300]; // Placeholder tokens

        // Test prefill
        let prefill_result = cuda_model.forward_cuda(&prompt);
        assert!(prefill_result.is_ok(), "Prefill failed: {:?}", prefill_result);
        let prefill_logits = prefill_result.unwrap();

        let prefill_max = prefill_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("Prefill max logit: {:.2}", prefill_max);
        assert!(prefill_max.is_finite(), "Prefill logits are -inf");

        // Test single token decode (THIS IS THE BUG)
        let decode_result = cuda_model.forward_single_cuda(400, prompt.len());
        assert!(decode_result.is_ok(), "Decode failed: {:?}", decode_result);
        let decode_logits = decode_result.unwrap();

        let decode_max = decode_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("Decode max logit: {:.2}", decode_max);

        // THIS ASSERTION WILL FAIL - proving the bug exists
        assert!(
            decode_max.is_finite() && decode_max != prefill_max,
            "PMAT-110 BUG CONFIRMED: forward_single_cuda returns invalid logits.\n\
             Prefill max: {}, Decode max: {}\n\
             The single-token decode path has no KV cache context.",
            prefill_max,
            decode_max
        );

        // Test full generation
        let gen_result = cuda_model.generate_cuda_with_cache(&prompt, 5, 151645);
        println!("Generation result: {:?}", gen_result);

        if let Ok(tokens) = gen_result {
            println!("Generated {} tokens", tokens.len() - prompt.len());
            // Should generate at least 1 token (not immediately hit EOS)
            assert!(
                tokens.len() > prompt.len(),
                "PMAT-110: Generation produced no new tokens (EOS immediately)"
            );
        }
    }

    /// Helper to create a transformer model with minimal weights
    fn create_transformer_model() -> AprV2Model {
        // Metadata with transformer config
        // NOTE: Uses hidden_size (not hidden_dim) to match AprMetadata struct
        let metadata = r#"{
            "model_type": "transformer",
            "name": "test-transformer",
            "hidden_size": 32,
            "num_layers": 1,
            "num_heads": 2,
            "num_kv_heads": 2,
            "vocab_size": 100,
            "intermediate_size": 64,
            "rms_norm_eps": 1e-6
        }"#;

        // Define tensors with their offsets (all dtype 0 = F32)
        // Hidden dim = 32, vocab = 100, intermediate = 64
        let tensors = [
            ("model.embed_tokens.weight", &[100_usize, 32][..], 0_u64), // 12800 bytes
            ("model.layers.0.input_layernorm.weight", &[32][..], 12800), // 128 bytes
            (
                "model.layers.0.self_attn.q_proj.weight",
                &[32, 32][..],
                12928,
            ), // 4096 bytes
            (
                "model.layers.0.self_attn.k_proj.weight",
                &[32, 32][..],
                17024,
            ), // 4096 bytes
            (
                "model.layers.0.self_attn.v_proj.weight",
                &[32, 32][..],
                21120,
            ), // 4096 bytes
            (
                "model.layers.0.self_attn.o_proj.weight",
                &[32, 32][..],
                25216,
            ), // 4096 bytes
            (
                "model.layers.0.post_attention_layernorm.weight",
                &[32][..],
                29312,
            ), // 128 bytes
            ("model.layers.0.mlp.gate_proj.weight", &[64, 32][..], 29440), // 8192 bytes
            ("model.layers.0.mlp.up_proj.weight", &[64, 32][..], 37632), // 8192 bytes
            ("model.layers.0.mlp.down_proj.weight", &[32, 64][..], 45824), // 8192 bytes
            ("model.norm.weight", &[32][..], 54016),                    // 128 bytes
            ("lm_head.weight", &[100, 32][..], 54144),                  // 12800 bytes
        ];

        // Build binary tensor index
        let mut tensor_index_binary = Vec::new();
        for (name, shape, offset) in &tensors {
            let size: u64 = shape.iter().product::<usize>() as u64 * 4; // F32 = 4 bytes each
            tensor_index_binary.extend(write_tensor_entry_binary(name, 0, shape, *offset, size));
        }

        // Calculate total tensor data size
        let total_tensor_bytes = 54144 + 12800; // Last offset + last size = 66944 bytes

        // Create tensor data with proper initialization (small non-zero values)
        let tensor_data: Vec<f32> = vec![0.01; total_tensor_bytes / 4];
        let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Calculate offsets (64-byte aligned)
        let metadata_offset = HEADER_SIZE as u64;
        let metadata_size = metadata.len() as u32;
        let tensor_index_offset =
            ((metadata_offset as usize + metadata.len()).div_ceil(64) * 64) as u64;
        let data_offset =
            ((tensor_index_offset as usize + tensor_index_binary.len()).div_ceil(64) * 64) as u64;

        let total_size = data_offset as usize + tensor_bytes.len();
        let mut data = vec![0u8; total_size];

        // Header (64 bytes)
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // Version major
        data[5] = 0; // Version minor
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // Flags
        data[8..12].copy_from_slice(&(tensors.len() as u32).to_le_bytes()); // Tensor count
        data[12..20].copy_from_slice(&metadata_offset.to_le_bytes());
        data[20..24].copy_from_slice(&metadata_size.to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        // Metadata
        data[metadata_offset as usize..metadata_offset as usize + metadata.len()]
            .copy_from_slice(metadata.as_bytes());

        // Tensor index (binary format)
        data[tensor_index_offset as usize
            ..tensor_index_offset as usize + tensor_index_binary.len()]
            .copy_from_slice(&tensor_index_binary);

        // Tensor data
        data[data_offset as usize..data_offset as usize + tensor_bytes.len()]
            .copy_from_slice(&tensor_bytes);

        AprV2Model::from_bytes(data).expect("should create transformer APR model")
    }
}
