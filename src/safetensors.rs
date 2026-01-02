//! Safetensors parser
//!
//! Pure Rust implementation of Safetensors format reader.
//! Used by `HuggingFace` for safe, zero-copy tensor storage.
//!
//! Format specification: <https://github.com/huggingface/safetensors>
//!
//! ## Format Overview
//!
//! ```text
//! Safetensors := HEADER METADATA TENSOR_DATA
//!
//! HEADER := {
//!   metadata_len: u64 (little-endian)
//! }
//!
//! METADATA := JSON {
//!   "tensor_name": {
//!     "dtype": "F32" | "F16" | "I32" | ...,
//!     "shape": [dim1, dim2, ...],
//!     "data_offsets": [start, end]
//!   },
//!   ...
//! }
//! ```

use std::{
    collections::HashMap,
    io::{Cursor, Read},
};

use serde::{Deserialize, Serialize};

use crate::error::{RealizarError, Result};

/// Safetensors data type
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub enum SafetensorsDtype {
    /// 32-bit float
    F32,
    /// 16-bit float
    F16,
    /// Brain float 16
    BF16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// Boolean
    Bool,
}

/// JSON tensor metadata (internal)
#[derive(Debug, Deserialize)]
struct TensorMetadata {
    dtype: SafetensorsDtype,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// Tensor metadata
#[derive(Debug, Clone, PartialEq)]
pub struct SafetensorsTensorInfo {
    /// Tensor name
    pub name: String,
    /// Data type
    pub dtype: SafetensorsDtype,
    /// Shape (dimensions)
    pub shape: Vec<usize>,
    /// Data offsets in file [start, end)
    pub data_offsets: [usize; 2],
}

/// Safetensors model container
#[derive(Debug, Clone)]
pub struct SafetensorsModel {
    /// Tensor metadata
    pub tensors: HashMap<String, SafetensorsTensorInfo>,
    /// Raw tensor data (not parsed yet)
    pub data: Vec<u8>,
}

impl SafetensorsModel {
    /// Parse Safetensors file from bytes
    ///
    /// # Arguments
    ///
    /// * `data` - Raw Safetensors file bytes
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Invalid header
    /// - Malformed JSON metadata
    /// - Invalid data offsets
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let data = std::fs::read("model.safetensors")?;
    /// let model = SafetensorsModel::from_bytes(&data)?;
    /// println!("Loaded {} tensors", model.tensors.len());
    /// ```
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);

        // Parse header (8-byte metadata length)
        let metadata_len = Self::parse_header(&mut cursor)?;

        // Parse JSON metadata
        let tensors = Self::parse_metadata(&mut cursor, metadata_len)?;

        // Store remaining data
        let data_start =
            usize::try_from(8 + metadata_len).map_err(|_| RealizarError::UnsupportedOperation {
                operation: "convert_data_offset".to_string(),
                reason: format!(
                    "Data offset {} exceeds platform usize limit",
                    8 + metadata_len
                ),
            })?;
        let data = data[data_start..].to_vec();

        Ok(Self { tensors, data })
    }

    /// Extract F32 tensor data by name
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name to extract
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tensor not found
    /// - Tensor dtype is not F32
    /// - Data offsets are invalid
    ///
    /// # Panics
    ///
    /// Never panics. The `unwrap()` in byte conversion is safe because
    /// `chunks_exact(4)` guarantees exactly 4 bytes per chunk.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let model = SafetensorsModel::from_bytes(&data)?;
    /// let weights = model.get_tensor_f32("layer1.weight")?;
    /// println!("Weights: {:?}", weights);
    /// ```
    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        // Find tensor metadata
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        // Verify dtype is F32
        if tensor.dtype != SafetensorsDtype::F32 {
            let dtype = &tensor.dtype;
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Tensor '{name}' has dtype {dtype:?}, expected F32"),
            });
        }

        // Extract data slice
        let [start, end] = tensor.data_offsets;
        if end > self.data.len() {
            let data_len = self.data.len();
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Data offset {end} exceeds data size {data_len}"),
            });
        }

        let bytes = &self.data[start..end];

        // Convert bytes to f32 vector
        if !bytes.len().is_multiple_of(4) {
            let len = bytes.len();
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Data size {len} is not a multiple of 4"),
            });
        }

        let values = bytes
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact(4) guarantees 4-byte slices"),
                )
            })
            .collect();

        Ok(values)
    }

    /// Parse header (8-byte metadata length)
    fn parse_header(cursor: &mut Cursor<&[u8]>) -> Result<u64> {
        let mut buf = [0u8; 8];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_metadata_len".to_string(),
                reason: e.to_string(),
            })?;

        Ok(u64::from_le_bytes(buf))
    }

    /// Parse JSON metadata
    fn parse_metadata(
        cursor: &mut Cursor<&[u8]>,
        len: u64,
    ) -> Result<HashMap<String, SafetensorsTensorInfo>> {
        // Read JSON bytes
        let len_usize = usize::try_from(len).map_err(|_| RealizarError::UnsupportedOperation {
            operation: "convert_metadata_len".to_string(),
            reason: format!("Metadata length {len} exceeds platform usize limit"),
        })?;

        let mut json_bytes = vec![0u8; len_usize];
        cursor
            .read_exact(&mut json_bytes)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_metadata_json".to_string(),
                reason: e.to_string(),
            })?;

        // Parse JSON
        let json_map: HashMap<String, TensorMetadata> = serde_json::from_slice(&json_bytes)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "parse_json".to_string(),
                reason: e.to_string(),
            })?;

        // Convert to SafetensorsTensorInfo
        let mut tensors = HashMap::new();
        for (name, meta) in json_map {
            tensors.insert(
                name.clone(),
                SafetensorsTensorInfo {
                    name,
                    dtype: meta.dtype,
                    shape: meta.shape,
                    data_offsets: meta.data_offsets,
                },
            );
        }

        Ok(tensors)
    }
}

#[cfg(test)]
mod tests {
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
}
