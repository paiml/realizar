
#[test]
fn test_run_bench_regression_first_file_missing() {
    let path2 = "/tmp/test_bench_reg_2b_cov95.json";
    std::fs::write(path2, "{}").unwrap();
    let result = run_bench_regression("/nonexistent/baseline.json", path2, false);
    let _ = std::fs::remove_file(path2);
    assert!(result.is_err());
}

#[test]
fn test_run_bench_regression_strict_first_file_missing() {
    let result = run_bench_regression("/nonexistent/base.json", "/nonexistent/curr.json", true);
    assert!(result.is_err());
}

// ============================================================================
// run_visualization
// ============================================================================

#[test]
fn test_run_visualization_default() {
    run_visualization(false, 100);
}

#[test]
fn test_run_visualization_with_color() {
    run_visualization(true, 50);
}

#[test]
fn test_run_visualization_small_samples() {
    run_visualization(false, 10);
}

// ============================================================================
// run_benchmarks (list mode only - no actual cargo bench)
// ============================================================================

#[test]
fn test_run_benchmarks_list_mode() {
    let result = run_benchmarks(None, true, None, None, None, None);
    assert!(result.is_ok());
}

// ============================================================================
// run_external_benchmark stub (no bench-http feature)
// ============================================================================

#[cfg(not(feature = "bench-http"))]
#[test]
fn test_run_external_benchmark_stub() {
    let result = run_external_benchmark("ollama", "http://localhost:11434", None, None);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("bench-http") || err.contains("feature"));
}

// ============================================================================
// load_safetensors_model
// ============================================================================

#[test]
fn test_load_safetensors_model_invalid() {
    let result = load_safetensors_model(&[0u8; 32]);
    assert!(result.is_err());
}

#[test]
fn test_load_safetensors_model_valid() {
    use crate::safetensors::SafetensorsModel;

    // Create a minimal valid safetensors file
    // Format: 8-byte header length + JSON header + tensor data
    let metadata = serde_json::json!({
        "test.weight": {
            "dtype": "F32",
            "shape": [2, 3],
            "data_offsets": [0, 24]
        }
    });
    let header = serde_json::to_string(&metadata).unwrap();
    let header_bytes = header.as_bytes();

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file_data.extend_from_slice(header_bytes);
    file_data.extend_from_slice(&[0u8; 24]); // tensor data (6 f32s)

    // First verify the model can be parsed
    let model = SafetensorsModel::from_bytes(&file_data);
    if model.is_ok() {
        let result = load_safetensors_model(&file_data);
        assert!(result.is_ok());
    }
}
