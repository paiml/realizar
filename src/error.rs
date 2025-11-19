//! Error types for Realizar
//!
//! This module defines all error types used throughout the library.

use thiserror::Error;

/// Result type alias for Realizar operations
pub type Result<T> = std::result::Result<T, RealizarError>;

/// Error type for all Realizar operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum RealizarError {
    /// Shape mismatch between tensors
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape
        actual: Vec<usize>,
    },

    /// Invalid shape specification
    #[error("Invalid shape: {reason}")]
    InvalidShape {
        /// Reason for invalidity
        reason: String,
    },

    /// Data size does not match shape
    #[error("Data size {data_size} does not match shape {shape:?} (expected {expected})")]
    DataShapeMismatch {
        /// Actual data size
        data_size: usize,
        /// Specified shape
        shape: Vec<usize>,
        /// Expected size from shape
        expected: usize,
    },

    /// Invalid dimension for operation
    #[error("Invalid dimension {dim} for tensor with {ndim} dimensions")]
    InvalidDimension {
        /// Requested dimension
        dim: usize,
        /// Number of dimensions
        ndim: usize,
    },

    /// Matrix multiplication dimension mismatch
    #[error("Matrix multiplication dimension mismatch: ({m}×{k}) × ({k2}×{n})")]
    MatmulDimensionMismatch {
        /// Rows in first matrix
        m: usize,
        /// Columns in first matrix / rows in second matrix
        k: usize,
        /// Rows in second matrix (should equal k)
        k2: usize,
        /// Columns in second matrix
        n: usize,
    },

    /// Operation not supported for this tensor type
    #[error("Operation '{operation}' not supported: {reason}")]
    UnsupportedOperation {
        /// Operation name
        operation: String,
        /// Reason it's not supported
        reason: String,
    },

    /// Trueno backend error
    #[error("Trueno backend error: {0}")]
    TruenoError(String),

    /// Index out of bounds
    #[error("Index out of bounds: index {index} for dimension of size {size}")]
    IndexOutOfBounds {
        /// Requested index
        index: usize,
        /// Size of dimension
        size: usize,
    },

    /// Model registry error
    #[error("Model registry error: {0}")]
    RegistryError(String),

    /// Model not found in registry
    #[error("Model '{0}' not found")]
    ModelNotFound(String),

    /// Model already registered
    #[error("Model '{0}' already registered")]
    ModelAlreadyExists(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RealizarError::ShapeMismatch {
            expected: vec![3, 3],
            actual: vec![2, 2],
        };
        assert!(err.to_string().contains("Shape mismatch"));
    }

    #[test]
    fn test_error_equality() {
        let err1 = RealizarError::InvalidShape {
            reason: "Empty shape".to_string(),
        };
        let err2 = RealizarError::InvalidShape {
            reason: "Empty shape".to_string(),
        };
        assert_eq!(err1, err2);
    }
}
