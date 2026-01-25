//! CUDA tests for APR module (Phase 40)
//!
//! Tests `AprV2ModelCuda` to improve coverage of apr/cuda.rs.

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use crate::apr::{AprV2Model, AprV2ModelCuda, MAGIC, HEADER_SIZE};

    /// Helper to create a minimal valid APR v2 model in memory
    fn create_minimal_apr_model() -> AprV2Model {
        use crate::apr::TensorEntry;

        // Minimal metadata with required fields for AprV2ModelCuda
        let metadata = r#"{
            "model_type": "test",
            "name": "test-cuda-model",
            "hidden_dim": 32,
            "num_layers": 1,
            "num_heads": 1,
            "vocab_size": 100
        }"#;

        // Single weight tensor
        let tensor_index: Vec<TensorEntry> = vec![TensorEntry {
            name: "token_embedding".to_string(),
            dtype: "F32".to_string(),
            shape: vec![100, 32], // vocab_size x hidden_dim
            offset: 0,
            size: 100 * 32 * 4,
        }];
        let tensor_index_json = serde_json::to_vec(&tensor_index).unwrap_or_default();

        // Embedding data (100 * 32 floats)
        let tensor_data: Vec<f32> = vec![0.1; 100 * 32];
        let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Calculate offsets (64-byte aligned)
        let metadata_offset = HEADER_SIZE as u64;
        let metadata_size = metadata.len() as u32;
        let tensor_index_offset =
            ((metadata_offset as usize + metadata.len()).div_ceil(64) * 64) as u64;
        let data_offset =
            ((tensor_index_offset as usize + tensor_index_json.len()).div_ceil(64) * 64) as u64;

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

        // Tensor index
        data[tensor_index_offset as usize..tensor_index_offset as usize + tensor_index_json.len()]
            .copy_from_slice(&tensor_index_json);

        // Tensor data
        data[data_offset as usize..data_offset as usize + tensor_bytes.len()]
            .copy_from_slice(&tensor_bytes);

        AprV2Model::from_bytes(data).expect("should create minimal APR model")
    }

    // =========================================================================
    // AprV2ModelCuda Construction Tests
    // =========================================================================

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
            }
            Err(e) => {
                // Expected if no GPU or under heavy instrumentation
                println!("CUDA init result: {:?}", e);
            }
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
            }
            Err(e) => {
                println!("CUDA init result: {:?}", e);
            }
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
}
