//! Tensor implementation
//!
//! This module provides the core `Tensor` type, which is an N-dimensional array
//! with automatic backend selection for optimal performance.

use std::fmt;

use num_traits::Num;
use serde::{Deserialize, Serialize};

use crate::error::{RealizarError, Result};

/// N-dimensional tensor with automatic backend dispatch
///
/// The tensor automatically selects the optimal execution backend (SIMD, GPU, WASM)
/// based on operation type, data size, and available hardware.
///
/// # Examples
///
/// ```
/// use realizar::Tensor;
///
/// // Create a 2×3 tensor
/// let t = Tensor::from_vec(vec![2, 3], vec![
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
/// ]).expect("test");
///
/// assert_eq!(t.shape(), &[2, 3]);
/// assert_eq!(t.ndim(), 2);
/// assert_eq!(t.size(), 6);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor<T: Num> {
    /// Flattened data in row-major order
    data: Vec<T>,
    /// Shape of the tensor
    shape: Vec<usize>,
}

impl<T: Num + Clone> Tensor<T> {
    /// Create a new tensor from a vector and shape
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimensions of the tensor
    /// * `data` - Flattened data in row-major order
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - Shape is empty
    /// - Data size doesn't match shape
    /// - Shape contains zero
    ///
    /// # Examples
    ///
    /// ```
    /// use realizar::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    /// assert_eq!(t.shape(), &[2, 2]);
    /// ```
    pub fn from_vec(shape: Vec<usize>, data: Vec<T>) -> Result<Self> {
        // Validate shape
        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Shape cannot be empty".to_string(),
            });
        }

        if shape.contains(&0) {
            return Err(RealizarError::InvalidShape {
                reason: "Shape dimensions cannot be zero".to_string(),
            });
        }

        // Calculate expected size
        let expected_size = shape.iter().product();

        // Validate data size
        if data.len() != expected_size {
            return Err(RealizarError::DataShapeMismatch {
                data_size: data.len(),
                shape,
                expected: expected_size,
            });
        }

        Ok(Self { data, shape })
    }

    /// Get the shape of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use realizar::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![3, 4], vec![0.0; 12]).expect("test");
    /// assert_eq!(t.shape(), &[3, 4]);
    /// ```
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use realizar::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![2, 3, 4], vec![0.0; 24]).expect("test");
    /// assert_eq!(t.ndim(), 3);
    /// ```
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use realizar::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).expect("test");
    /// assert_eq!(t.size(), 6);
    /// ```
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get a reference to the underlying data
    ///
    /// # Examples
    ///
    /// ```
    /// use realizar::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("test");
    /// assert_eq!(t.data(), &[1.0, 2.0]);
    /// ```
    #[must_use]
    pub fn data(&self) -> &[T] {
        &self.data
    }
}

impl<T: Num + Clone + fmt::Display> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, data=[", self.shape)?;
        for (i, val) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{val}")?;
        }
        write!(f, "])")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_tensor() {
        let t = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.size(), 6);
    }

    #[test]
    fn test_empty_shape_error() {
        let result = Tensor::from_vec(vec![], vec![1.0, 2.0]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RealizarError::InvalidShape { .. }
        ));
    }

    #[test]
    fn test_zero_dimension_error() {
        let result = Tensor::<f32>::from_vec(vec![2, 0], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_size_mismatch_error() {
        let result = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RealizarError::DataShapeMismatch { .. }
        ));
    }

    #[test]
    fn test_display() {
        let t = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("test");
        let display = format!("{t}");
        assert!(display.contains("shape=[2]"));
        assert!(display.contains('1'));
        assert!(display.contains('2'));
    }
}
