//! Unified Model Loader
//!
//! Per spec §3.2 and §5: Unified loading for APR, GGUF, and SafeTensors formats.
//!
//! ## Jidoka (Built-in Quality)
//!
//! - Format auto-detection from magic bytes
//! - CRC32 verification for APR format
//! - Header validation for SafeTensors
//! - Graceful error handling with detailed messages
//!
//! ## APR Format Support (First-class)
//!
//! Per spec §3.1: APR is the primary format for classical ML models from aprender.
//! Supports all 18 model types:
//!
//! | Type | Description |
//! |------|-------------|
//! | `LinearRegression` | OLS/Ridge/Lasso |
//! | `LogisticRegression` | Binary/Multinomial classification |
//! | `DecisionTree` | CART/ID3 |
//! | `RandomForest` | Bagging ensemble |
//! | `GradientBoosting` | Boosting ensemble |
//! | `KMeans` | Lloyd's clustering |
//! | `PCA` | Dimensionality reduction |
//! | `NaiveBayes` | Gaussian NB |
//! | `KNN` | k-Nearest Neighbors |
//! | `SVM` | Linear SVM |
//! | `NgramLM` | N-gram language model |
//! | `TFIDF` | TF-IDF vectorizer |
//! | `CountVectorizer` | Count vectorizer |
//! | `NeuralSequential` | Feed-forward NN |
//! | `NeuralCustom` | Custom architecture |
//! | `ContentRecommender` | Content-based rec |
//! | `MixtureOfExperts` | Sparse/dense MoE |
//! | `Custom` | User-defined |
//!
//! ## GGUF Support (Backwards Compatible)
//!
//! Per spec §3.3: GGUF for LLM inference with llama.cpp compatibility.
//!
//! ## SafeTensors Support (Backwards Compatible)
//!
//! Per spec §3.4: SafeTensors for HuggingFace model weights.

use std::path::Path;

use crate::format::{detect_and_verify_format, detect_format, FormatError, ModelFormat};

/// Model loading errors
#[derive(Debug, Clone)]
pub enum LoadError {
    /// Format detection failed
    FormatError(FormatError),
    /// File I/O error
    IoError(String),
    /// Model parsing error
    ParseError(String),
    /// Unsupported model type for serving
    UnsupportedType(String),
    /// CRC32 checksum mismatch (APR)
    IntegrityError(String),
    /// Model type mismatch (requested vs detected)
    TypeMismatch {
        /// Expected model type
        expected: String,
        /// Actual model type in file
        actual: String,
    },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FormatError(e) => write!(f, "Format detection error: {e}"),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
            Self::ParseError(msg) => write!(f, "Parse error: {msg}"),
            Self::UnsupportedType(t) => write!(f, "Unsupported model type: {t}"),
            Self::IntegrityError(msg) => write!(f, "Integrity check failed: {msg}"),
            Self::TypeMismatch { expected, actual } => {
                write!(f, "Model type mismatch: expected {expected}, got {actual}")
            },
        }
    }
}

impl std::error::Error for LoadError {}

impl From<FormatError> for LoadError {
    fn from(e: FormatError) -> Self {
        Self::FormatError(e)
    }
}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e.to_string())
    }
}

/// Model metadata extracted during loading
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Detected format
    pub format: ModelFormat,
    /// Model type (if detected)
    pub model_type: Option<String>,
    /// Model version
    pub version: Option<String>,
    /// Input dimensions (for validation)
    pub input_dim: Option<usize>,
    /// Output dimensions
    pub output_dim: Option<usize>,
    /// File size in bytes
    pub file_size: u64,
}

impl ModelMetadata {
    /// Create new metadata with format only
    #[must_use]
    pub fn new(format: ModelFormat) -> Self {
        Self {
            format,
            model_type: None,
            version: None,
            input_dim: None,
            output_dim: None,
            file_size: 0,
        }
    }

    /// Set model type
    #[must_use]
    pub fn with_model_type(mut self, model_type: impl Into<String>) -> Self {
        self.model_type = Some(model_type.into());
        self
    }

    /// Set version
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Set input dimensions
    #[must_use]
    pub fn with_input_dim(mut self, dim: usize) -> Self {
        self.input_dim = Some(dim);
        self
    }

    /// Set output dimensions
    #[must_use]
    pub fn with_output_dim(mut self, dim: usize) -> Self {
        self.output_dim = Some(dim);
        self
    }

    /// Set file size
    #[must_use]
    pub fn with_file_size(mut self, size: u64) -> Self {
        self.file_size = size;
        self
    }
}

/// Detect model format from file path and contents
///
/// Per spec §3.2: Jidoka - verify both path and magic bytes match.
///
/// # Arguments
///
/// * `path` - Path to model file
///
/// # Returns
///
/// Model metadata with detected format
///
/// # Errors
///
/// Returns error if:
/// - File cannot be read
/// - Format cannot be detected
/// - Extension doesn't match magic bytes
///
/// # Example
///
/// ```rust,ignore
/// use realizar::model_loader::detect_model;
/// use std::path::Path;
///
/// let metadata = detect_model(Path::new("model.apr"))?;
/// assert_eq!(metadata.format, ModelFormat::Apr);
/// ```
pub fn detect_model(path: &Path) -> Result<ModelMetadata, LoadError> {
    // Read first 8 bytes for magic detection
    let data = std::fs::read(path)?;
    if data.len() < 8 {
        return Err(LoadError::ParseError(format!(
            "File too small: {} bytes",
            data.len()
        )));
    }

    // Verify format from path and data
    let format = detect_and_verify_format(path, &data[..8])?;

    Ok(ModelMetadata::new(format).with_file_size(data.len() as u64))
}

/// Detect model format from bytes only (no path verification)
///
/// Useful for embedded models via `include_bytes!()`.
///
/// # Arguments
///
/// * `data` - Model file bytes
///
/// # Returns
///
/// Model metadata with detected format
///
/// # Errors
///
/// Returns error if:
/// - Data is too small for format detection (<8 bytes)
/// - Format cannot be detected from magic bytes
///
/// # Example
///
/// ```rust,ignore
/// use realizar::model_loader::detect_model_from_bytes;
///
/// const MODEL: &[u8] = include_bytes!("../models/model.apr");
/// let metadata = detect_model_from_bytes(MODEL)?;
/// ```
pub fn detect_model_from_bytes(data: &[u8]) -> Result<ModelMetadata, LoadError> {
    if data.len() < 8 {
        return Err(LoadError::ParseError(format!(
            "Data too small: {} bytes",
            data.len()
        )));
    }

    let format = detect_format(&data[..8])?;

    Ok(ModelMetadata::new(format).with_file_size(data.len() as u64))
}

/// Load APR model type from metadata bytes
///
/// Reads model type from APR header (bytes 4-6 after magic).
///
/// # Arguments
///
/// * `data` - APR file bytes (at least 8 bytes)
///
/// # Returns
///
/// APR model type string (e.g., "LogisticRegression")
pub fn read_apr_model_type(data: &[u8]) -> Option<String> {
    if data.len() < 8 {
        return None;
    }

    // APR header layout: APRN (4 bytes) + type_id (2 bytes) + version (2 bytes)
    // Per aprender format spec
    let type_id = u16::from_le_bytes([data[4], data[5]]);

    // Map type ID to name (from aprender::format::ModelType)
    let type_name = match type_id {
        0x0001 => "LinearRegression",
        0x0002 => "LogisticRegression",
        0x0003 => "DecisionTree",
        0x0004 => "RandomForest",
        0x0005 => "GradientBoosting",
        0x0006 => "KMeans",
        0x0007 => "PCA",
        0x0008 => "NaiveBayes",
        0x0009 => "KNN",
        0x000A => "SVM",
        0x0010 => "NgramLM",
        0x0011 => "TFIDF",
        0x0012 => "CountVectorizer",
        0x0020 => "NeuralSequential",
        0x0021 => "NeuralCustom",
        0x0030 => "ContentRecommender",
        0x0040 => "MixtureOfExperts",
        0x00FF => "Custom",
        _ => return None,
    };

    Some(type_name.to_string())
}

/// Validate that loaded model matches expected type
///
/// Per Jidoka: fail fast if type mismatch.
///
/// # Arguments
///
/// * `expected` - Expected model type
/// * `actual` - Actual model type from file
///
/// # Returns
///
/// Ok if types match, Err otherwise
///
/// # Errors
///
/// Returns `LoadError::TypeMismatch` if expected and actual types differ.
pub fn validate_model_type(expected: &str, actual: &str) -> Result<(), LoadError> {
    if expected != actual {
        return Err(LoadError::TypeMismatch {
            expected: expected.to_string(),
            actual: actual.to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== EXTREME TDD: LoadError Tests =====

    #[test]
    fn test_load_error_format_error() {
        let err = LoadError::FormatError(FormatError::UnknownFormat);
        assert!(err.to_string().contains("Format detection error"));
        assert!(err.to_string().contains("Unknown"));
    }

    #[test]
    fn test_load_error_io_error() {
        let err = LoadError::IoError("file not found".to_string());
        assert!(err.to_string().contains("I/O error"));
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn test_load_error_parse_error() {
        let err = LoadError::ParseError("invalid header".to_string());
        assert!(err.to_string().contains("Parse error"));
        assert!(err.to_string().contains("invalid header"));
    }

    #[test]
    fn test_load_error_unsupported_type() {
        let err = LoadError::UnsupportedType("UnknownModel".to_string());
        assert!(err.to_string().contains("Unsupported model type"));
        assert!(err.to_string().contains("UnknownModel"));
    }

    #[test]
    fn test_load_error_integrity_error() {
        let err = LoadError::IntegrityError("CRC32 mismatch".to_string());
        assert!(err.to_string().contains("Integrity check failed"));
        assert!(err.to_string().contains("CRC32"));
    }

    #[test]
    fn test_load_error_type_mismatch() {
        let err = LoadError::TypeMismatch {
            expected: "LogisticRegression".to_string(),
            actual: "DecisionTree".to_string(),
        };
        assert!(err.to_string().contains("type mismatch"));
        assert!(err.to_string().contains("LogisticRegression"));
        assert!(err.to_string().contains("DecisionTree"));
    }

    #[test]
    fn test_load_error_from_format_error() {
        let format_err = FormatError::TooShort { len: 3 };
        let load_err: LoadError = format_err.into();
        assert!(matches!(load_err, LoadError::FormatError(_)));
    }

    // ===== EXTREME TDD: ModelMetadata Tests =====

    #[test]
    fn test_model_metadata_new() {
        let meta = ModelMetadata::new(ModelFormat::Apr);
        assert_eq!(meta.format, ModelFormat::Apr);
        assert!(meta.model_type.is_none());
        assert!(meta.version.is_none());
        assert!(meta.input_dim.is_none());
        assert!(meta.output_dim.is_none());
        assert_eq!(meta.file_size, 0);
    }

    #[test]
    fn test_model_metadata_with_model_type() {
        let meta = ModelMetadata::new(ModelFormat::Apr).with_model_type("LogisticRegression");
        assert_eq!(meta.model_type, Some("LogisticRegression".to_string()));
    }

    #[test]
    fn test_model_metadata_with_version() {
        let meta = ModelMetadata::new(ModelFormat::Gguf).with_version("v1.0.0");
        assert_eq!(meta.version, Some("v1.0.0".to_string()));
    }

    #[test]
    fn test_model_metadata_with_input_dim() {
        let meta = ModelMetadata::new(ModelFormat::SafeTensors).with_input_dim(784);
        assert_eq!(meta.input_dim, Some(784));
    }

    #[test]
    fn test_model_metadata_with_output_dim() {
        let meta = ModelMetadata::new(ModelFormat::Apr).with_output_dim(10);
        assert_eq!(meta.output_dim, Some(10));
    }

    #[test]
    fn test_model_metadata_with_file_size() {
        let meta = ModelMetadata::new(ModelFormat::Gguf).with_file_size(1_000_000);
        assert_eq!(meta.file_size, 1_000_000);
    }

    #[test]
    fn test_model_metadata_chained_builders() {
        let meta = ModelMetadata::new(ModelFormat::Apr)
            .with_model_type("RandomForest")
            .with_version("v2.1")
            .with_input_dim(128)
            .with_output_dim(3)
            .with_file_size(50_000);

        assert_eq!(meta.format, ModelFormat::Apr);
        assert_eq!(meta.model_type, Some("RandomForest".to_string()));
        assert_eq!(meta.version, Some("v2.1".to_string()));
        assert_eq!(meta.input_dim, Some(128));
        assert_eq!(meta.output_dim, Some(3));
        assert_eq!(meta.file_size, 50_000);
    }

    // ===== EXTREME TDD: detect_model_from_bytes Tests =====

    #[test]
    fn test_detect_model_from_bytes_apr() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&[0x02, 0x00, 0x01, 0x00]); // LogisticRegression type
        data.extend_from_slice(&[0u8; 100]); // Padding

        let meta = detect_model_from_bytes(&data).expect("Should detect APR");
        assert_eq!(meta.format, ModelFormat::Apr);
        assert_eq!(meta.file_size, 108);
    }

    #[test]
    fn test_detect_model_from_bytes_gguf() {
        let mut data = b"GGUF".to_vec();
        data.extend_from_slice(&[0u8; 100]); // Padding

        let meta = detect_model_from_bytes(&data).expect("Should detect GGUF");
        assert_eq!(meta.format, ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_model_from_bytes_safetensors() {
        let header_size: u64 = 100;
        let mut data = header_size.to_le_bytes().to_vec();
        data.extend_from_slice(&[0u8; 200]);

        let meta = detect_model_from_bytes(&data).expect("Should detect SafeTensors");
        assert_eq!(meta.format, ModelFormat::SafeTensors);
    }

    #[test]
    fn test_detect_model_from_bytes_too_small() {
        let data = b"APR";
        let result = detect_model_from_bytes(data);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LoadError::ParseError(_)));
    }

    // ===== EXTREME TDD: read_apr_model_type Tests =====

    #[test]
    fn test_read_apr_model_type_linear_regression() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0001u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(
            read_apr_model_type(&data),
            Some("LinearRegression".to_string())
        );
    }

    #[test]
    fn test_read_apr_model_type_logistic_regression() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0002u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(
            read_apr_model_type(&data),
            Some("LogisticRegression".to_string())
        );
    }

    #[test]
    fn test_read_apr_model_type_decision_tree() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0003u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("DecisionTree".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_random_forest() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0004u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("RandomForest".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_gradient_boosting() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0005u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(
            read_apr_model_type(&data),
            Some("GradientBoosting".to_string())
        );
    }

    #[test]
    fn test_read_apr_model_type_kmeans() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0006u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("KMeans".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_pca() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0007u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("PCA".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_naive_bayes() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0008u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("NaiveBayes".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_knn() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0009u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("KNN".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_svm() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x000Au16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("SVM".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_ngram_lm() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0010u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("NgramLM".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_tfidf() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0011u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("TFIDF".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_count_vectorizer() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0012u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(
            read_apr_model_type(&data),
            Some("CountVectorizer".to_string())
        );
    }

    #[test]
    fn test_read_apr_model_type_neural_sequential() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0020u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(
            read_apr_model_type(&data),
            Some("NeuralSequential".to_string())
        );
    }

    #[test]
    fn test_read_apr_model_type_neural_custom() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0021u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("NeuralCustom".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_content_recommender() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0030u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(
            read_apr_model_type(&data),
            Some("ContentRecommender".to_string())
        );
    }

    #[test]
    fn test_read_apr_model_type_mixture_of_experts() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0040u16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(
            read_apr_model_type(&data),
            Some("MixtureOfExperts".to_string())
        );
    }

    #[test]
    fn test_read_apr_model_type_custom() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x00FFu16.to_le_bytes());
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), Some("Custom".to_string()));
    }

    #[test]
    fn test_read_apr_model_type_unknown() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0xFFFFu16.to_le_bytes()); // Unknown type
        data.extend_from_slice(&[0, 0]);

        assert_eq!(read_apr_model_type(&data), None);
    }

    #[test]
    fn test_read_apr_model_type_too_short() {
        let data = b"APR\0"; // Only 4 bytes
        assert_eq!(read_apr_model_type(data), None);
    }

    // ===== EXTREME TDD: validate_model_type Tests =====

    #[test]
    fn test_validate_model_type_match() {
        let result = validate_model_type("LogisticRegression", "LogisticRegression");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_model_type_mismatch() {
        let result = validate_model_type("LogisticRegression", "DecisionTree");
        assert!(result.is_err());

        if let Err(LoadError::TypeMismatch { expected, actual }) = result {
            assert_eq!(expected, "LogisticRegression");
            assert_eq!(actual, "DecisionTree");
        } else {
            panic!("Expected TypeMismatch error");
        }
    }

    #[test]
    fn test_validate_model_type_case_sensitive() {
        // Type names are case-sensitive
        let result = validate_model_type("logisticregression", "LogisticRegression");
        assert!(result.is_err());
    }

    // ===== EXTREME TDD: Integration Tests =====

    #[test]
    fn test_detect_and_extract_apr_type() {
        // Simulate APR file with LogisticRegression type
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0002u16.to_le_bytes()); // LogisticRegression
        data.extend_from_slice(&[0, 0]); // version placeholder
        data.extend_from_slice(&[0u8; 100]); // Padding

        let meta = detect_model_from_bytes(&data).expect("Detection should succeed");
        assert_eq!(meta.format, ModelFormat::Apr);

        let model_type = read_apr_model_type(&data).expect("Should extract model type");
        assert_eq!(model_type, "LogisticRegression");
    }

    #[test]
    fn test_full_metadata_extraction() {
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&0x0004u16.to_le_bytes()); // RandomForest
        data.extend_from_slice(&[0, 0]);
        data.extend_from_slice(&[0u8; 500]);

        let meta = detect_model_from_bytes(&data)
            .expect("Detection should succeed")
            .with_model_type(read_apr_model_type(&data).unwrap_or_default())
            .with_version("v1.0")
            .with_input_dim(128);

        assert_eq!(meta.format, ModelFormat::Apr);
        assert_eq!(meta.model_type, Some("RandomForest".to_string()));
        assert_eq!(meta.version, Some("v1.0".to_string()));
        assert_eq!(meta.input_dim, Some(128));
        assert_eq!(meta.file_size, 508);
    }

    // ===== EXTREME TDD: Debug/Error Trait Tests =====

    #[test]
    fn test_load_error_debug() {
        let err = LoadError::IoError("test".to_string());
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("IoError"));
    }

    #[test]
    fn test_model_metadata_debug() {
        let meta = ModelMetadata::new(ModelFormat::Apr);
        let debug_str = format!("{meta:?}");
        assert!(debug_str.contains("Apr"));
    }

    #[test]
    fn test_model_metadata_clone() {
        let meta = ModelMetadata::new(ModelFormat::Gguf)
            .with_model_type("LLM")
            .with_file_size(1000);
        let cloned = meta.clone();

        assert_eq!(cloned.format, ModelFormat::Gguf);
        assert_eq!(cloned.model_type, Some("LLM".to_string()));
        assert_eq!(cloned.file_size, 1000);
    }

    #[test]
    fn test_load_error_clone() {
        let err = LoadError::ParseError("test".to_string());
        let cloned = err.clone();
        assert!(matches!(cloned, LoadError::ParseError(_)));
    }
}
