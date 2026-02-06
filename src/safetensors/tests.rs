use super::*;

#[test]
fn test_parse_empty_safetensors() {
    // Minimal valid Safetensors: 8-byte header + empty JSON "{}"
    let mut data = Vec::new();
    data.extend_from_slice(&2u64.to_le_bytes()); // metadata_len = 2
    data.extend_from_slice(b"{}"); // Empty JSON

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    assert_eq!(model.tensors.len(), 0);
    assert_eq!(model.data.len(), 0);
}

#[test]
fn test_invalid_header_truncated() {
    // Only 4 bytes (should be 8)
    let data = [0u8; 4];
    let result = SafetensorsModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_empty_file() {
    let data = &[];
    let result = SafetensorsModel::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_parse_single_tensor() {
    // Safetensors with one F32 tensor
    let json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    // Add 24 bytes of dummy tensor data (2*3*4 = 24 bytes for F32)
    data.extend_from_slice(&[0u8; 24]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    assert_eq!(model.tensors.len(), 1);

    let tensor = model.tensors.get("weight").expect("test");
    assert_eq!(tensor.name, "weight");
    assert_eq!(tensor.dtype, SafetensorsDtype::F32);
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.data_offsets, [0, 24]);
}

#[test]
fn test_parse_multiple_tensors() {
    // Safetensors with multiple tensors of different types
    let json = r#"{
            "layer1.weight":{"dtype":"F32","shape":[128,256],"data_offsets":[0,131072]},
            "layer1.bias":{"dtype":"F32","shape":[128],"data_offsets":[131072,131584]}
        }"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    // Add dummy tensor data
    data.extend_from_slice(&vec![0u8; 131_584]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    assert_eq!(model.tensors.len(), 2);

    let weight = model.tensors.get("layer1.weight").expect("test");
    assert_eq!(weight.dtype, SafetensorsDtype::F32);
    assert_eq!(weight.shape, vec![128, 256]);
    assert_eq!(weight.data_offsets, [0, 131_072]);

    let bias = model.tensors.get("layer1.bias").expect("test");
    assert_eq!(bias.dtype, SafetensorsDtype::F32);
    assert_eq!(bias.shape, vec![128]);
    assert_eq!(bias.data_offsets, [131_072, 131_584]);
}

#[test]
fn test_parse_various_dtypes() {
    // Test different data types
    let json = r#"{
            "f32_tensor":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
            "i32_tensor":{"dtype":"I32","shape":[2],"data_offsets":[8,16]},
            "u8_tensor":{"dtype":"U8","shape":[4],"data_offsets":[16,20]}
        }"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 20]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    assert_eq!(model.tensors.len(), 3);

    assert_eq!(
        model.tensors.get("f32_tensor").expect("test").dtype,
        SafetensorsDtype::F32
    );
    assert_eq!(
        model.tensors.get("i32_tensor").expect("test").dtype,
        SafetensorsDtype::I32
    );
    assert_eq!(
        model.tensors.get("u8_tensor").expect("test").dtype,
        SafetensorsDtype::U8
    );
}

#[test]
fn test_invalid_json_error() {
    // Invalid JSON in metadata
    let mut data = Vec::new();
    data.extend_from_slice(&10u64.to_le_bytes()); // metadata_len = 10
    data.extend_from_slice(b"not json!!"); // Invalid JSON

    let result = SafetensorsModel::from_bytes(&data);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        RealizarError::UnsupportedOperation { .. }
    ));
}

#[test]
fn test_truncated_json_error() {
    // Header says JSON is longer than actual data
    let mut data = Vec::new();
    data.extend_from_slice(&100u64.to_le_bytes()); // metadata_len = 100
    data.extend_from_slice(b"{}"); // Only 2 bytes, not 100

    let result = SafetensorsModel::from_bytes(&data);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        RealizarError::UnsupportedOperation { .. }
    ));
}

#[test]
fn test_parse_all_dtypes() {
    // Test all supported data types
    let json = r#"{
            "f32":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
            "f16":{"dtype":"F16","shape":[1],"data_offsets":[4,6]},
            "bf16":{"dtype":"BF16","shape":[1],"data_offsets":[6,8]},
            "i32":{"dtype":"I32","shape":[1],"data_offsets":[8,12]},
            "i64":{"dtype":"I64","shape":[1],"data_offsets":[12,20]},
            "u8":{"dtype":"U8","shape":[1],"data_offsets":[20,21]},
            "bool":{"dtype":"Bool","shape":[1],"data_offsets":[21,22]}
        }"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 22]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    assert_eq!(model.tensors.len(), 7);

    assert_eq!(
        model.tensors.get("f32").expect("test").dtype,
        SafetensorsDtype::F32
    );
    assert_eq!(
        model.tensors.get("f16").expect("test").dtype,
        SafetensorsDtype::F16
    );
    assert_eq!(
        model.tensors.get("bf16").expect("test").dtype,
        SafetensorsDtype::BF16
    );
    assert_eq!(
        model.tensors.get("i32").expect("test").dtype,
        SafetensorsDtype::I32
    );
    assert_eq!(
        model.tensors.get("i64").expect("test").dtype,
        SafetensorsDtype::I64
    );
    assert_eq!(
        model.tensors.get("u8").expect("test").dtype,
        SafetensorsDtype::U8
    );
    assert_eq!(
        model.tensors.get("bool").expect("test").dtype,
        SafetensorsDtype::Bool
    );
}

#[test]
fn test_tensor_data_preserved() {
    // Verify tensor data is correctly preserved
    let json = r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    // Add specific tensor data (two f32s: 1.0 and 2.0)
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&2.0f32.to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    assert_eq!(model.data.len(), 8);

    // Verify we can read back the f32 values
    let val1 = f32::from_le_bytes(model.data[0..4].try_into().expect("test"));
    let val2 = f32::from_le_bytes(model.data[4..8].try_into().expect("test"));
    assert!((val1 - 1.0).abs() < 1e-6);
    assert!((val2 - 2.0).abs() < 1e-6);
}

#[test]
fn test_multidimensional_shapes() {
    // Test tensors with various shapes
    let json = r#"{
            "scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]},
            "vector":{"dtype":"F32","shape":[10],"data_offsets":[4,44]},
            "matrix":{"dtype":"F32","shape":[3,4],"data_offsets":[44,92]},
            "tensor3d":{"dtype":"F32","shape":[2,3,4],"data_offsets":[92,188]}
        }"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 188]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    assert_eq!(model.tensors.len(), 4);

    assert_eq!(
        model.tensors.get("scalar").expect("test").shape,
        Vec::<usize>::new()
    );
    assert_eq!(model.tensors.get("vector").expect("test").shape, vec![10]);
    assert_eq!(model.tensors.get("matrix").expect("test").shape, vec![3, 4]);
    assert_eq!(
        model.tensors.get("tensor3d").expect("test").shape,
        vec![2, 3, 4]
    );
}

#[test]
fn test_aprender_linear_regression_format_compatibility() {
    // Test aprender LinearRegression SafeTensors format compatibility
    // Format: {"coefficients": [n_features], "intercept": [1]}
    // Example model: y = 2.0*x1 + 3.0*x2 + 1.5*x3 + 0.5

    let json = r#"{
            "coefficients":{"dtype":"F32","shape":[3],"data_offsets":[0,12]},
            "intercept":{"dtype":"F32","shape":[1],"data_offsets":[12,16]}
        }"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();

    // Header
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());

    // Metadata
    data.extend_from_slice(json_bytes);

    // Tensor data: coefficients [2.0, 3.0, 1.5]
    data.extend_from_slice(&2.0f32.to_le_bytes());
    data.extend_from_slice(&3.0f32.to_le_bytes());
    data.extend_from_slice(&1.5f32.to_le_bytes());

    // intercept [0.5]
    data.extend_from_slice(&0.5f32.to_le_bytes());

    // Parse with realizar
    let model = SafetensorsModel::from_bytes(&data).expect("test");

    // Verify structure
    assert_eq!(model.tensors.len(), 2);

    // Check coefficients tensor
    let coef = model.tensors.get("coefficients").expect("test");
    assert_eq!(coef.dtype, SafetensorsDtype::F32);
    assert_eq!(coef.shape, vec![3]);
    assert_eq!(coef.data_offsets, [0, 12]);

    // Check intercept tensor
    let intercept = model.tensors.get("intercept").expect("test");
    assert_eq!(intercept.dtype, SafetensorsDtype::F32);
    assert_eq!(intercept.shape, vec![1]);
    assert_eq!(intercept.data_offsets, [12, 16]);

    // Verify we can extract the actual values
    let coef_vals: Vec<f32> = (0..3)
        .map(|i| {
            let offset = i * 4;
            f32::from_le_bytes(model.data[offset..offset + 4].try_into().expect("test"))
        })
        .collect();
    assert!((coef_vals[0] - 2.0).abs() < 1e-6);
    assert!((coef_vals[1] - 3.0).abs() < 1e-6);
    assert!((coef_vals[2] - 1.5).abs() < 1e-6);

    let intercept_val = f32::from_le_bytes(model.data[12..16].try_into().expect("test"));
    assert!((intercept_val - 0.5).abs() < 1e-6);
}

#[test]
fn test_get_tensor_f32_helper() {
    // Test the get_tensor_f32 helper method
    let json = r#"{
            "weights":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},
            "bias":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}
        }"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);

    // weights: [1.0, 2.0, 3.0, 4.0]
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&2.0f32.to_le_bytes());
    data.extend_from_slice(&3.0f32.to_le_bytes());
    data.extend_from_slice(&4.0f32.to_le_bytes());

    // bias: [0.5, 0.25]
    data.extend_from_slice(&0.5f32.to_le_bytes());
    data.extend_from_slice(&0.25f32.to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("test");

    // Test extracting weights
    let weights = model.get_tensor_f32("weights").expect("test");
    assert_eq!(weights.len(), 4);
    assert!((weights[0] - 1.0).abs() < 1e-6);
    assert!((weights[1] - 2.0).abs() < 1e-6);
    assert!((weights[2] - 3.0).abs() < 1e-6);
    assert!((weights[3] - 4.0).abs() < 1e-6);

    // Test extracting bias
    let bias = model.get_tensor_f32("bias").expect("test");
    assert_eq!(bias.len(), 2);
    assert!((bias[0] - 0.5).abs() < 1e-6);
    assert!((bias[1] - 0.25).abs() < 1e-6);

    // Test error: tensor not found
    let result = model.get_tensor_f32("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_get_tensor_f32_wrong_dtype() {
    // Test error when tensor has wrong dtype
    let json = r#"{
            "int_tensor":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}
        }"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&1i32.to_le_bytes());
    data.extend_from_slice(&2i32.to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("test");

    // Should error because dtype is I32, not F32
    let result = model.get_tensor_f32("int_tensor");
    assert!(result.is_err());
}

#[test]
fn test_get_tensor_f32_with_aprender_model() {
    // Use get_tensor_f32 with aprender LinearRegression format
    let json = r#"{
            "coefficients":{"dtype":"F32","shape":[3],"data_offsets":[0,12]},
            "intercept":{"dtype":"F32","shape":[1],"data_offsets":[12,16]}
        }"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&2.0f32.to_le_bytes());
    data.extend_from_slice(&3.0f32.to_le_bytes());
    data.extend_from_slice(&1.5f32.to_le_bytes());
    data.extend_from_slice(&0.5f32.to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("test");

    // Extract using helper method - much cleaner!
    let coefficients = model.get_tensor_f32("coefficients").expect("test");
    assert_eq!(coefficients, vec![2.0, 3.0, 1.5]);

    let intercept = model.get_tensor_f32("intercept").expect("test");
    assert_eq!(intercept, vec![0.5]);
}

// ========== Coverage tests for untested functions ==========

#[test]
fn test_cov_get_tensor_f16_as_f32() {
    let json = r#"{"weights":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    // Two F16 values: 1.0 and 2.0
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data.extend_from_slice(&half::f16::from_f32(2.0).to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let weights = model.get_tensor_f16_as_f32("weights").expect("test");

    assert_eq!(weights.len(), 2);
    assert!((weights[0] - 1.0).abs() < 0.01);
    assert!((weights[1] - 2.0).abs() < 0.01);
}

#[test]
fn test_cov_get_tensor_f16_not_found() {
    let json = r#"{"weights":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 4]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_f16_as_f32("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_cov_get_tensor_f16_wrong_dtype() {
    let json = r#"{"weights":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 8]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_f16_as_f32("weights");
    assert!(result.is_err());
}

#[test]
fn test_cov_get_tensor_f16_data_offset_exceeds() {
    let json = r#"{"weights":{"dtype":"F16","shape":[2],"data_offsets":[0,100]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 4]); // Only 4 bytes, offset says 100

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_f16_as_f32("weights");
    assert!(result.is_err());
}

#[test]
fn test_cov_get_tensor_bf16_as_f32() {
    let json = r#"{"weights":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    // Two BF16 values: 1.0 and 2.0
    data.extend_from_slice(&half::bf16::from_f32(1.0).to_le_bytes());
    data.extend_from_slice(&half::bf16::from_f32(2.0).to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let weights = model.get_tensor_bf16_as_f32("weights").expect("test");

    assert_eq!(weights.len(), 2);
    assert!((weights[0] - 1.0).abs() < 0.01);
    assert!((weights[1] - 2.0).abs() < 0.01);
}

#[test]
fn test_cov_get_tensor_bf16_not_found() {
    let json = r#"{"weights":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 4]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_bf16_as_f32("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_cov_get_tensor_bf16_wrong_dtype() {
    let json = r#"{"weights":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 8]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_bf16_as_f32("weights");
    assert!(result.is_err());
}

#[test]
fn test_cov_get_tensor_bf16_data_offset_exceeds() {
    let json = r#"{"weights":{"dtype":"BF16","shape":[2],"data_offsets":[0,100]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 4]); // Only 4 bytes

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_bf16_as_f32("weights");
    assert!(result.is_err());
}

#[test]
fn test_cov_get_tensor_auto_f32() {
    let json = r#"{"weights":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&2.0f32.to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let weights = model.get_tensor_auto("weights").expect("test");
    assert_eq!(weights, vec![1.0, 2.0]);
}

#[test]
fn test_cov_get_tensor_auto_f16() {
    let json = r#"{"weights":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data.extend_from_slice(&half::f16::from_f32(2.0).to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let weights = model.get_tensor_auto("weights").expect("test");
    assert_eq!(weights.len(), 2);
}

#[test]
fn test_cov_get_tensor_auto_bf16() {
    let json = r#"{"weights":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&half::bf16::from_f32(1.0).to_le_bytes());
    data.extend_from_slice(&half::bf16::from_f32(2.0).to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let weights = model.get_tensor_auto("weights").expect("test");
    assert_eq!(weights.len(), 2);
}

#[test]
fn test_cov_get_tensor_auto_unsupported_dtype() {
    let json = r#"{"weights":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 8]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_auto("weights");
    assert!(result.is_err());
}

#[test]
fn test_cov_get_tensor_auto_not_found() {
    let json = r#"{"weights":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 8]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_auto("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_cov_tensor_names() {
    let json = r#"{
            "weight1":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
            "weight2":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}
        }"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 16]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let names = model.tensor_names();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"weight1"));
    assert!(names.contains(&"weight2"));
}

#[test]
fn test_cov_get_tensor_info() {
    let json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 24]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");

    let info = model.get_tensor_info("weight");
    assert!(info.is_some());
    let info = info.expect("operation failed");
    assert_eq!(info.shape, vec![2, 3]);
    assert_eq!(info.dtype, SafetensorsDtype::F32);

    assert!(model.get_tensor_info("nonexistent").is_none());
}

#[test]
fn test_cov_has_tensor() {
    let json = r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 8]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    assert!(model.has_tensor("weight"));
    assert!(!model.has_tensor("nonexistent"));
}

#[test]
fn test_cov_get_tensor_f32_data_offset_exceeds() {
    let json = r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,100]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 8]); // Only 8 bytes, offset says 100

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_f32("weight");
    assert!(result.is_err());
}

#[test]
fn test_cov_get_tensor_f32_not_multiple_of_4() {
    let json = r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,7]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 7]); // 7 bytes, not a multiple of 4

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_f32("weight");
    assert!(result.is_err());
}

#[test]
fn test_cov_safetensors_config_num_kv_heads() {
    let config = SafetensorsConfig {
        hidden_size: Some(768),
        num_hidden_layers: Some(12),
        num_attention_heads: Some(12),
        num_key_value_heads: Some(4),
        vocab_size: Some(32000),
        intermediate_size: None,
        max_position_embeddings: None,
        rms_norm_eps: None,
        rope_theta: None,
        architectures: None,
        model_type: None,
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: None,
    };

    assert_eq!(config.num_kv_heads(), 4);
}

#[test]
fn test_cov_safetensors_config_num_kv_heads_default() {
    let config = SafetensorsConfig {
        hidden_size: Some(768),
        num_hidden_layers: Some(12),
        num_attention_heads: Some(12),
        num_key_value_heads: None, // Not set, should fall back to attention heads
        vocab_size: Some(32000),
        intermediate_size: None,
        max_position_embeddings: None,
        rms_norm_eps: None,
        rope_theta: None,
        architectures: None,
        model_type: None,
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: None,
    };

    assert_eq!(config.num_kv_heads(), 12);
}

#[test]
fn test_cov_safetensors_config_num_kv_heads_fallback() {
    let config = SafetensorsConfig {
        hidden_size: Some(768),
        num_hidden_layers: Some(12),
        num_attention_heads: None,
        num_key_value_heads: None,
        vocab_size: Some(32000),
        intermediate_size: None,
        max_position_embeddings: None,
        rms_norm_eps: None,
        rope_theta: None,
        architectures: None,
        model_type: None,
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: None,
    };

    assert_eq!(config.num_kv_heads(), 1); // Fallback to 1
}

#[test]
fn test_cov_safetensors_config_architecture_from_architectures() {
    let config = SafetensorsConfig {
        hidden_size: None,
        num_hidden_layers: None,
        num_attention_heads: None,
        num_key_value_heads: None,
        vocab_size: None,
        intermediate_size: None,
        max_position_embeddings: None,
        rms_norm_eps: None,
        rope_theta: None,
        architectures: Some(vec!["LlamaForCausalLM".to_string()]),
        model_type: Some("llama".to_string()),
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: None,
    };

    assert_eq!(config.architecture(), "LlamaForCausalLM");
}

#[test]
fn test_cov_safetensors_config_architecture_from_model_type() {
    let config = SafetensorsConfig {
        hidden_size: None,
        num_hidden_layers: None,
        num_attention_heads: None,
        num_key_value_heads: None,
        vocab_size: None,
        intermediate_size: None,
        max_position_embeddings: None,
        rms_norm_eps: None,
        rope_theta: None,
        architectures: None,
        model_type: Some("llama".to_string()),
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: None,
    };

    assert_eq!(config.architecture(), "llama");
}

#[test]
fn test_cov_safetensors_config_architecture_unknown() {
    let config = SafetensorsConfig {
        hidden_size: None,
        num_hidden_layers: None,
        num_attention_heads: None,
        num_key_value_heads: None,
        vocab_size: None,
        intermediate_size: None,
        max_position_embeddings: None,
        rms_norm_eps: None,
        rope_theta: None,
        architectures: None,
        model_type: None,
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: None,
    };

    assert_eq!(config.architecture(), "unknown");
}

#[test]
fn test_cov_safetensors_config_load_from_sibling_not_found() {
    let path = std::path::Path::new("/nonexistent/model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(path);
    assert!(config.is_none());
}

#[test]
fn test_cov_metadata_key_skipped() {
    // Test that __metadata__ key is skipped
    let json = r#"{
            "__metadata__":{"format":"pt"},
            "weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}
        }"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 8]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    assert_eq!(model.tensors.len(), 1);
    assert!(model.tensors.contains_key("weight"));
    assert!(!model.tensors.contains_key("__metadata__"));
}

#[test]
fn test_cov_json_not_object() {
    // JSON is an array, not an object
    let json = r"[]";
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);

    let result = SafetensorsModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_tensor_metadata_parse_error() {
    // Tensor has invalid metadata (missing dtype)
    let json = r#"{"weight":{"shape":[2],"data_offsets":[0,8]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 8]);

    let result = SafetensorsModel::from_bytes(&data);
    assert!(result.is_err());
}

// ========== MappedSafeTensorsModel tests ==========

#[cfg(not(target_arch = "wasm32"))]
mod mapped_tests {
    use super::*;
    use std::io::Write;

    /// Helper to create a temporary safetensors file
    fn create_temp_safetensors(
        tensors: &[(&str, SafetensorsDtype, &[usize], &[u8])],
    ) -> tempfile::NamedTempFile {
        let mut json_map = serde_json::Map::new();
        let mut tensor_data = Vec::new();
        let mut offset = 0usize;

        for (name, dtype, shape, data) in tensors {
            let dtype_str = match dtype {
                SafetensorsDtype::F32 => "F32",
                SafetensorsDtype::F16 => "F16",
                SafetensorsDtype::BF16 => "BF16",
                SafetensorsDtype::I32 => "I32",
                SafetensorsDtype::I64 => "I64",
                SafetensorsDtype::U8 => "U8",
                SafetensorsDtype::Bool => "Bool",
            };

            let end = offset + data.len();
            json_map.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype_str,
                    "shape": shape,
                    "data_offsets": [offset, end]
                }),
            );

            tensor_data.extend_from_slice(data);
            offset = end;
        }

        let json_str = serde_json::to_string(&json_map).expect("JSON serialization");
        let json_bytes = json_str.as_bytes();

        let mut file = tempfile::NamedTempFile::new().expect("temp file creation");
        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json_bytes).expect("write metadata");
        file.write_all(&tensor_data).expect("write tensor data");
        file.flush().expect("flush file");

        file
    }

    #[test]
    fn test_mapped_load_basic() {
        // Create temp file with one F32 tensor
        let tensor_data: Vec<u8> = [1.0f32, 2.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        assert_eq!(model.tensor_count(), 1);
        assert!(model.has_tensor("weight"));
        assert!(!model.has_tensor("nonexistent"));
    }

    #[test]
    fn test_mapped_file_not_found() {
        let result = MappedSafeTensorsModel::load("/nonexistent/path/model.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_file_too_small() {
        // Create a file with only 4 bytes (less than header size)
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(&[0u8; 4]).expect("write");
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("File too small"),
            "Expected 'File too small' error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_truncated_metadata() {
        // Create a file that claims metadata is 100 bytes but only has 10
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(&100u64.to_le_bytes()).expect("write header");
        file.write_all(b"{}").expect("write short json");
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("truncated"),
            "Expected 'truncated' error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_invalid_json() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let invalid_json = b"not valid json!!";
        file.write_all(&(invalid_json.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(invalid_json).expect("write json");
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_json_not_object() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let json = b"[]"; // Array instead of object
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json).expect("write json");
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("Expected JSON object"),
            "Expected 'Expected JSON object' error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_tensor_metadata_parse_error() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        // Missing dtype field
        let json = r#"{"weight":{"shape":[2],"data_offsets":[0,8]}}"#;
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json.as_bytes()).expect("write json");
        file.write_all(&[0u8; 8]).expect("write data");
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("Failed to parse tensor"),
            "Expected tensor parse error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_get_tensor_bytes() {
        let tensor_data: Vec<u8> = [1.0f32, 2.0f32, 3.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[3], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let bytes = model.get_tensor_bytes("weight").expect("get bytes");
        assert_eq!(bytes.len(), 12); // 3 * 4 bytes
    }

    #[test]
    fn test_mapped_get_tensor_bytes_not_found() {
        let file = create_temp_safetensors(&[(
            "weight",
            SafetensorsDtype::F32,
            &[1],
            &0.0f32.to_le_bytes(),
        )]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_bytes("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_get_tensor_bytes_offset_exceeds() {
        // Create a file with a tensor that claims more data than exists
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let json = r#"{"weight":{"dtype":"F32","shape":[100],"data_offsets":[0,400]}}"#;
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json.as_bytes()).expect("write json");
        file.write_all(&[0u8; 8])
            .expect("write only 8 bytes of data");
        file.flush().expect("flush");

        // GH-213: Truncated files are now caught at load time (Layer 3 safety net)
        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("truncated"),
            "Expected 'truncated' error at load time, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_get_tensor_f32() {
        let tensor_data: Vec<u8> = [1.0f32, 2.0f32, 3.0f32, 4.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[4], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let values = model.get_tensor_f32("weight").expect("get f32");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_mapped_get_tensor_f32_not_found() {
        let file = create_temp_safetensors(&[(
            "weight",
            SafetensorsDtype::F32,
            &[1],
            &1.0f32.to_le_bytes(),
        )]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_f32("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_get_tensor_f32_wrong_dtype() {
        let file = create_temp_safetensors(&[("weight", SafetensorsDtype::I32, &[2], &[0u8; 8])]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_f32("weight");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("expected F32"),
            "Expected wrong dtype error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_get_tensor_f32_not_multiple_of_4() {
        // Create file with misaligned data
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let json = r#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,7]}}"#;
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json.as_bytes()).expect("write json");
        file.write_all(&[0u8; 7]).expect("write 7 bytes");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_f32("weight");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("not a multiple of 4"),
            "Expected alignment error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_get_tensor_f16_bytes() {
        let tensor_data: Vec<u8> = [half::f16::from_f32(1.0), half::f16::from_f32(2.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F16, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let bytes = model.get_tensor_f16_bytes("weight").expect("get f16 bytes");
        assert_eq!(bytes.len(), 4); // 2 * 2 bytes
    }

    #[test]
    fn test_mapped_get_tensor_f16_bytes_not_found() {
        let file = create_temp_safetensors(&[("weight", SafetensorsDtype::F16, &[1], &[0u8; 2])]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_f16_bytes("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_get_tensor_f16_bytes_wrong_dtype() {
        let file = create_temp_safetensors(&[(
            "weight",
            SafetensorsDtype::F32,
            &[1],
            &1.0f32.to_le_bytes(),
        )]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_f16_bytes("weight");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("expected F16"),
            "Expected wrong dtype error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_get_tensor_f16_as_f32() {
        let tensor_data: Vec<u8> = [half::f16::from_f32(1.0), half::f16::from_f32(2.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F16, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let values = model
            .get_tensor_f16_as_f32("weight")
            .expect("get f16 as f32");
        assert_eq!(values.len(), 2);
        assert!((values[0] - 1.0).abs() < 0.01);
        assert!((values[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_mapped_get_tensor_bf16_bytes() {
        let tensor_data: Vec<u8> = [half::bf16::from_f32(1.0), half::bf16::from_f32(2.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::BF16, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let bytes = model
            .get_tensor_bf16_bytes("weight")
            .expect("get bf16 bytes");
        assert_eq!(bytes.len(), 4); // 2 * 2 bytes
    }

    #[test]
    fn test_mapped_get_tensor_bf16_bytes_not_found() {
        let file = create_temp_safetensors(&[("weight", SafetensorsDtype::BF16, &[1], &[0u8; 2])]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_bf16_bytes("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_get_tensor_bf16_bytes_wrong_dtype() {
        let file = create_temp_safetensors(&[(
            "weight",
            SafetensorsDtype::F32,
            &[1],
            &1.0f32.to_le_bytes(),
        )]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_bf16_bytes("weight");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("expected BF16"),
            "Expected wrong dtype error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_get_tensor_bf16_as_f32() {
        let tensor_data: Vec<u8> = [half::bf16::from_f32(1.0), half::bf16::from_f32(2.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::BF16, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let values = model
            .get_tensor_bf16_as_f32("weight")
            .expect("get bf16 as f32");
        assert_eq!(values.len(), 2);
        assert!((values[0] - 1.0).abs() < 0.01);
        assert!((values[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_mapped_get_tensor_auto_f32() {
        let tensor_data: Vec<u8> = [1.0f32, 2.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let values = model.get_tensor_auto("weight").expect("get auto");
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_mapped_get_tensor_auto_f16() {
        let tensor_data: Vec<u8> = [half::f16::from_f32(1.0), half::f16::from_f32(2.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F16, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let values = model.get_tensor_auto("weight").expect("get auto");
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_mapped_get_tensor_auto_bf16() {
        let tensor_data: Vec<u8> = [half::bf16::from_f32(1.0), half::bf16::from_f32(2.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::BF16, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let values = model.get_tensor_auto("weight").expect("get auto");
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_mapped_get_tensor_auto_unsupported() {
        let file = create_temp_safetensors(&[("weight", SafetensorsDtype::I32, &[2], &[0u8; 8])]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_auto("weight");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("Unsupported dtype"),
            "Expected unsupported dtype error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_get_tensor_auto_not_found() {
        let file = create_temp_safetensors(&[(
            "weight",
            SafetensorsDtype::F32,
            &[1],
            &1.0f32.to_le_bytes(),
        )]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_auto("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_tensor_names() {
        let file = create_temp_safetensors(&[
            (
                "weight1",
                SafetensorsDtype::F32,
                &[1],
                &1.0f32.to_le_bytes(),
            ),
            (
                "weight2",
                SafetensorsDtype::F32,
                &[1],
                &2.0f32.to_le_bytes(),
            ),
        ]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let names = model.tensor_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"weight1"));
        assert!(names.contains(&"weight2"));
    }

    #[test]
    fn test_mapped_get_tensor_info() {
        let tensor_data: Vec<u8> = [1.0f32, 2.0f32, 3.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[3], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let info = model.get_tensor_info("weight");
        assert!(info.is_some());
        let info = info.expect("tensor info");
        assert_eq!(info.dtype, SafetensorsDtype::F32);
        assert_eq!(info.shape, vec![3]);

        assert!(model.get_tensor_info("nonexistent").is_none());
    }

    #[test]
    fn test_mapped_path() {
        let file = create_temp_safetensors(&[(
            "weight",
            SafetensorsDtype::F32,
            &[1],
            &1.0f32.to_le_bytes(),
        )]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        assert_eq!(model.path(), file.path());
    }

    #[test]
    fn test_mapped_file_size() {
        let tensor_data: Vec<u8> = [1.0f32, 2.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        // File size = 8 (header) + json length + 8 (tensor data)
        assert!(model.file_size() > 8);
    }

    #[test]
    fn test_mapped_tensor_count() {
        let file = create_temp_safetensors(&[
            (
                "weight1",
                SafetensorsDtype::F32,
                &[1],
                &1.0f32.to_le_bytes(),
            ),
            (
                "weight2",
                SafetensorsDtype::F32,
                &[1],
                &2.0f32.to_le_bytes(),
            ),
            (
                "weight3",
                SafetensorsDtype::F32,
                &[1],
                &3.0f32.to_le_bytes(),
            ),
        ]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        assert_eq!(model.tensor_count(), 3);
    }

    #[test]
    fn test_mapped_metadata_key_skipped() {
        // Test that __metadata__ key is skipped in mapped model
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let json = r#"{
                "__metadata__":{"format":"pt"},
                "weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}
            }"#;
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json.as_bytes()).expect("write json");
        file.write_all(&[0u8; 8]).expect("write data");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        assert_eq!(model.tensor_count(), 1);
        assert!(model.has_tensor("weight"));
        assert!(!model.has_tensor("__metadata__"));
    }

    #[test]
    fn test_mapped_multiple_tensors() {
        let w1: Vec<u8> = [1.0f32, 2.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let w2: Vec<u8> = [half::f16::from_f32(3.0), half::f16::from_f32(4.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let w3: Vec<u8> = [half::bf16::from_f32(5.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let file = create_temp_safetensors(&[
            ("f32_tensor", SafetensorsDtype::F32, &[2], &w1),
            ("f16_tensor", SafetensorsDtype::F16, &[2], &w2),
            ("bf16_tensor", SafetensorsDtype::BF16, &[1], &w3),
        ]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        assert_eq!(model.tensor_count(), 3);

        let f32_vals = model.get_tensor_f32("f32_tensor").expect("f32");
        assert_eq!(f32_vals, vec![1.0, 2.0]);

        let f16_vals = model.get_tensor_f16_as_f32("f16_tensor").expect("f16");
        assert!((f16_vals[0] - 3.0).abs() < 0.01);
        assert!((f16_vals[1] - 4.0).abs() < 0.01);

        let bf16_vals = model.get_tensor_bf16_as_f32("bf16_tensor").expect("bf16");
        assert!((bf16_vals[0] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_mapped_empty_tensors() {
        // Test with empty tensor list (only {} metadata)
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let json = b"{}";
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json).expect("write json");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        assert_eq!(model.tensor_count(), 0);
        assert!(model.tensor_names().is_empty());
    }
}

// ========== Additional SafetensorsConfig coverage ==========

#[test]
fn test_cov_safetensors_config_load_from_sibling_invalid_json() {
    // Create a temp dir with an invalid config.json
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let config_path = temp_dir.path().join("config.json");
    std::fs::write(&config_path, "not valid json").expect("write config");

    let model_path = temp_dir.path().join("model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(&model_path);
    assert!(config.is_none());
}

#[test]
fn test_cov_safetensors_config_load_from_sibling_valid() {
    // Create a temp dir with valid config.json
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let config_path = temp_dir.path().join("config.json");
    let config_json = r#"{
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "vocab_size": 32000,
            "model_type": "llama"
        }"#;
    std::fs::write(&config_path, config_json).expect("write config");

    let model_path = temp_dir.path().join("model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(&model_path);
    assert!(config.is_some());
    let config = config.expect("config");
    assert_eq!(config.hidden_size, Some(768));
    assert_eq!(config.num_hidden_layers, Some(12));
    assert_eq!(config.num_attention_heads, Some(12));
    assert_eq!(config.vocab_size, Some(32000));
    assert_eq!(config.model_type, Some("llama".to_string()));
}

#[test]
fn test_cov_safetensors_config_serde_aliases() {
    // Test that serde aliases work (n_embd, n_layer, n_head, etc.)
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let config_path = temp_dir.path().join("config.json");
    let config_json = r#"{
            "n_embd": 512,
            "n_layer": 6,
            "n_head": 8,
            "n_inner": 2048,
            "n_positions": 1024
        }"#;
    std::fs::write(&config_path, config_json).expect("write config");

    let model_path = temp_dir.path().join("model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(&model_path);
    assert!(config.is_some());
    let config = config.expect("config");
    assert_eq!(config.hidden_size, Some(512)); // n_embd alias
    assert_eq!(config.num_hidden_layers, Some(6)); // n_layer alias
    assert_eq!(config.num_attention_heads, Some(8)); // n_head alias
    assert_eq!(config.intermediate_size, Some(2048)); // n_inner alias
    assert_eq!(config.max_position_embeddings, Some(1024)); // n_positions alias
}

#[test]
fn test_cov_safetensors_config_all_fields() {
    // Test config with all optional fields present
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let config_path = temp_dir.path().join("config.json");
    let config_json = r#"{
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "vocab_size": 32000,
            "intermediate_size": 3072,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "bos_token_id": 1,
            "eos_token_id": 2
        }"#;
    std::fs::write(&config_path, config_json).expect("write config");

    let model_path = temp_dir.path().join("model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(&model_path).expect("config");

    assert_eq!(config.hidden_size, Some(768));
    assert_eq!(config.num_hidden_layers, Some(12));
    assert_eq!(config.num_attention_heads, Some(12));
    assert_eq!(config.num_key_value_heads, Some(4));
    assert_eq!(config.vocab_size, Some(32000));
    assert_eq!(config.intermediate_size, Some(3072));
    assert_eq!(config.max_position_embeddings, Some(2048));
    assert!((config.rms_norm_eps.expect("eps") - 1e-5).abs() < 1e-10);
    assert!((config.rope_theta.expect("theta") - 10000.0).abs() < 1e-3);
    assert_eq!(
        config.architectures,
        Some(vec!["LlamaForCausalLM".to_string()])
    );
    assert_eq!(config.model_type, Some("llama".to_string()));
    assert_eq!(config.bos_token_id, Some(1));
    assert_eq!(config.eos_token_id, Some(2));
}

#[test]
fn test_cov_safetensors_config_architecture_empty_list() {
    // Test architecture() with empty architectures list
    let config = SafetensorsConfig {
        hidden_size: None,
        num_hidden_layers: None,
        num_attention_heads: None,
        num_key_value_heads: None,
        vocab_size: None,
        intermediate_size: None,
        max_position_embeddings: None,
        rms_norm_eps: None,
        rope_theta: None,
        architectures: Some(vec![]), // Empty list
        model_type: Some("gpt2".to_string()),
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: None,
    };

    // Should fall back to model_type when architectures is empty
    assert_eq!(config.architecture(), "gpt2");
}

// ========== Additional SafetensorsTensorInfo tests ==========

#[test]
fn test_cov_tensor_info_clone_and_eq() {
    let info1 = SafetensorsTensorInfo {
        name: "weight".to_string(),
        dtype: SafetensorsDtype::F32,
        shape: vec![2, 3],
        data_offsets: [0, 24],
    };

    let info2 = info1.clone();
    assert_eq!(info1, info2);

    let info3 = SafetensorsTensorInfo {
        name: "weight".to_string(),
        dtype: SafetensorsDtype::F16, // Different dtype
        shape: vec![2, 3],
        data_offsets: [0, 24],
    };
    assert_ne!(info1, info3);
}

#[test]
fn test_cov_safetensors_dtype_clone_and_eq() {
    let dtype1 = SafetensorsDtype::F32;
    let dtype2 = dtype1.clone();
    assert_eq!(dtype1, dtype2);

    let dtype3 = SafetensorsDtype::F16;
    assert_ne!(dtype1, dtype3);
}
