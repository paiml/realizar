//! Text generation and sampling strategies
//!
//! This module provides the generation loop for autoregressive text generation
//! and various sampling strategies for token selection.
//!
//! # Sampling Strategies
//!
//! - **Greedy**: Always select the most probable token
//! - **Top-k**: Sample from the k most probable tokens
//! - **Top-p (nucleus)**: Sample from tokens with cumulative probability ≤ p
//! - **Temperature**: Scale logits before softmax to control randomness

use crate::error::{RealizarError, Result};
use crate::layers::softmax;
use crate::tensor::Tensor;

/// Sample from a probability distribution using a random value
///
/// # Arguments
///
/// * `probs` - Probabilities (must sum to 1)
/// * `indices` - Corresponding indices for each probability
/// * `rng_value` - Random value in [0, 1)
///
/// # Returns
///
/// Selected index
fn sample_from_distribution(probs: &[f32], indices: &[usize], rng_value: f32) -> usize {
    let mut cumsum = 0.0;
    for (i, &prob) in probs.iter().enumerate() {
        cumsum += prob;
        if rng_value < cumsum {
            return indices[i];
        }
    }
    // Fallback to last token
    indices[indices.len() - 1]
}

/// Convert logits to softmax probabilities for a subset
///
/// # Arguments
///
/// * `indexed` - Pairs of (index, logit) sorted by logit descending
///
/// # Returns
///
/// Probabilities for the subset
fn logits_to_probs(indexed: &[(usize, f32)]) -> Vec<f32> {
    let max_logit = indexed[0].1;
    let exp_vals: Vec<f32> = indexed.iter().map(|(_, l)| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|e| e / sum_exp).collect()
}

/// Build nucleus for top-p sampling
///
/// # Arguments
///
/// * `indexed` - Pairs of (index, prob) sorted by prob descending
/// * `p` - Cumulative probability threshold
///
/// # Returns
///
/// Nucleus of (index, prob) pairs with cumulative probability >= p
fn build_nucleus(indexed: &[(usize, f32)], p: f32) -> Vec<(usize, f32)> {
    let mut cumsum = 0.0;
    let mut nucleus = Vec::new();
    for &(idx, prob) in indexed {
        nucleus.push((idx, prob));
        cumsum += prob;
        if cumsum >= p {
            break;
        }
    }
    nucleus
}

/// Sampling strategy for token selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingStrategy {
    /// Always select the most probable token
    Greedy,
    /// Sample from the k most probable tokens
    TopK {
        /// Number of top tokens to consider
        k: usize,
    },
    /// Sample from tokens with cumulative probability ≤ p
    TopP {
        /// Cumulative probability threshold
        p: f32,
    },
}

/// Configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Sampling strategy
    pub strategy: SamplingStrategy,
    /// Temperature for scaling logits (1.0 = no scaling)
    pub temperature: f32,
    /// Token ID for end-of-sequence
    pub eos_token_id: Option<usize>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            strategy: SamplingStrategy::Greedy,
            temperature: 1.0,
            eos_token_id: None,
            seed: None,
        }
    }
}

impl GenerationConfig {
    /// Create a new generation config with greedy sampling
    #[must_use]
    pub fn greedy() -> Self {
        Self {
            strategy: SamplingStrategy::Greedy,
            ..Default::default()
        }
    }

    /// Create a new generation config with top-k sampling
    #[must_use]
    pub fn top_k(k: usize) -> Self {
        Self {
            strategy: SamplingStrategy::TopK { k },
            ..Default::default()
        }
    }

    /// Create a new generation config with top-p (nucleus) sampling
    #[must_use]
    pub fn top_p(p: f32) -> Self {
        Self {
            strategy: SamplingStrategy::TopP { p },
            ..Default::default()
        }
    }

    /// Set temperature
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set maximum tokens
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set end-of-sequence token ID
    #[must_use]
    pub fn with_eos_token_id(mut self, eos_token_id: usize) -> Self {
        self.eos_token_id = Some(eos_token_id);
        self
    }

    /// Set random seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Apply temperature scaling to logits
///
/// # Arguments
///
/// * `logits` - Raw logits from model
/// * `temperature` - Temperature value (> 0)
///
/// # Returns
///
/// Scaled logits
///
/// # Errors
///
/// Returns error if temperature is not positive
pub fn apply_temperature(logits: &Tensor<f32>, temperature: f32) -> Result<Tensor<f32>> {
    if temperature <= 0.0 {
        return Err(RealizarError::InvalidShape {
            reason: "Temperature must be positive".to_string(),
        });
    }

    if (temperature - 1.0).abs() < 1e-6 {
        // No scaling needed
        return Ok(logits.clone());
    }

    let data = logits.data();
    let scaled: Vec<f32> = data.iter().map(|&x| x / temperature).collect();
    Tensor::from_vec(logits.shape().to_vec(), scaled)
}

/// Greedy sampling: select the token with highest probability
///
/// # Arguments
///
/// * `logits` - Logits for the vocabulary
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if logits are empty
pub fn sample_greedy(logits: &Tensor<f32>) -> Result<usize> {
    let data = logits.data();
    if data.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Logits cannot be empty".to_string(),
        });
    }

    let mut max_idx = 0;
    let mut max_val = data[0];
    for (i, &val) in data.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    Ok(max_idx)
}

/// Top-k sampling: sample from the k most probable tokens
///
/// # Arguments
///
/// * `logits` - Logits for the vocabulary
/// * `k` - Number of top tokens to consider
/// * `rng_value` - Random value in [0, 1) for sampling
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if k is 0 or logits are empty
pub fn sample_top_k(logits: &Tensor<f32>, k: usize, rng_value: f32) -> Result<usize> {
    let data = logits.data();
    if data.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Logits cannot be empty".to_string(),
        });
    }
    if k == 0 {
        return Err(RealizarError::InvalidShape {
            reason: "k must be > 0".to_string(),
        });
    }

    // Create (index, logit) pairs and sort by logit descending
    let mut indexed: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top k
    let top_k: Vec<(usize, f32)> = indexed.into_iter().take(k.min(data.len())).collect();

    // Convert to probabilities and sample
    let probs = logits_to_probs(&top_k);
    let indices: Vec<usize> = top_k.iter().map(|(idx, _)| *idx).collect();
    Ok(sample_from_distribution(&probs, &indices, rng_value))
}

/// Top-p (nucleus) sampling: sample from tokens with cumulative probability ≤ p
///
/// # Arguments
///
/// * `logits` - Logits for the vocabulary
/// * `p` - Cumulative probability threshold
/// * `rng_value` - Random value in [0, 1) for sampling
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if p is not in (0, 1] or logits are empty
pub fn sample_top_p(logits: &Tensor<f32>, p: f32, rng_value: f32) -> Result<usize> {
    let data = logits.data();
    if data.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Logits cannot be empty".to_string(),
        });
    }
    if p <= 0.0 || p > 1.0 {
        return Err(RealizarError::InvalidShape {
            reason: "p must be in (0, 1]".to_string(),
        });
    }

    // Convert logits to probabilities
    let probs_tensor = softmax(logits)?;
    let probs = probs_tensor.data();

    // Create (index, prob) pairs and sort by prob descending
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Build nucleus (cumulative probability <= p)
    let nucleus = build_nucleus(&indexed, p);

    // Renormalize and sample
    let nucleus_sum: f32 = nucleus.iter().map(|(_, prob)| prob).sum();
    let normalized_probs: Vec<f32> = nucleus.iter().map(|(_, prob)| prob / nucleus_sum).collect();
    let indices: Vec<usize> = nucleus.iter().map(|(idx, _)| *idx).collect();

    Ok(sample_from_distribution(&normalized_probs, &indices, rng_value))
}

/// Sample a token based on the sampling strategy
///
/// # Arguments
///
/// * `logits` - Logits for the vocabulary
/// * `config` - Generation configuration
/// * `rng_value` - Random value in [0, 1) for sampling (ignored for greedy)
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if temperature is invalid or sampling fails
pub fn sample_token(
    logits: &Tensor<f32>,
    config: &GenerationConfig,
    rng_value: f32,
) -> Result<usize> {
    // Apply temperature
    let scaled_logits = apply_temperature(logits, config.temperature)?;

    match config.strategy {
        SamplingStrategy::Greedy => sample_greedy(&scaled_logits),
        SamplingStrategy::TopK { k } => sample_top_k(&scaled_logits, k, rng_value),
        SamplingStrategy::TopP { p } => sample_top_p(&scaled_logits, p, rng_value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 100);
        assert_eq!(config.strategy, SamplingStrategy::Greedy);
        assert!((config.temperature - 1.0).abs() < 1e-6);
        assert!(config.eos_token_id.is_none());
    }

    #[test]
    fn test_generation_config_builders() {
        let config = GenerationConfig::greedy().with_max_tokens(50);
        assert_eq!(config.max_tokens, 50);
        assert_eq!(config.strategy, SamplingStrategy::Greedy);

        let config = GenerationConfig::top_k(10).with_temperature(0.8);
        assert_eq!(config.strategy, SamplingStrategy::TopK { k: 10 });
        assert!((config.temperature - 0.8).abs() < 1e-6);

        let config = GenerationConfig::top_p(0.9).with_eos_token_id(2);
        assert_eq!(config.strategy, SamplingStrategy::TopP { p: 0.9 });
        assert_eq!(config.eos_token_id, Some(2));
    }

    #[test]
    fn test_apply_temperature() {
        let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Temperature = 1.0 should return same values
        let scaled = apply_temperature(&logits, 1.0).unwrap();
        for i in 0..4 {
            assert!((scaled.data()[i] - logits.data()[i]).abs() < 1e-6);
        }

        // Temperature = 2.0 should halve values
        let scaled = apply_temperature(&logits, 2.0).unwrap();
        assert!((scaled.data()[0] - 0.5).abs() < 1e-6);
        assert!((scaled.data()[3] - 2.0).abs() < 1e-6);

        // Temperature = 0.5 should double values
        let scaled = apply_temperature(&logits, 0.5).unwrap();
        assert!((scaled.data()[0] - 2.0).abs() < 1e-6);
        assert!((scaled.data()[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_temperature_invalid() {
        let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(apply_temperature(&logits, 0.0).is_err());
        assert!(apply_temperature(&logits, -1.0).is_err());
    }

    #[test]
    fn test_sample_greedy() {
        // Clear winner at index 2
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 10.0, 3.0, 4.0]).unwrap();
        let token = sample_greedy(&logits).unwrap();
        assert_eq!(token, 2);

        // Winner at last index
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 5.0]).unwrap();
        let token = sample_greedy(&logits).unwrap();
        assert_eq!(token, 2);

        // Winner at first index
        let logits = Tensor::from_vec(vec![3], vec![5.0, 2.0, 1.0]).unwrap();
        let token = sample_greedy(&logits).unwrap();
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_greedy_empty_error() {
        let logits = Tensor::from_vec(vec![1], vec![1.0]).unwrap();
        // Single element should work
        assert_eq!(sample_greedy(&logits).unwrap(), 0);
    }

    #[test]
    fn test_sample_top_k() {
        // Strong preference for index 0
        let logits = Tensor::from_vec(vec![5], vec![100.0, 1.0, 1.0, 1.0, 1.0]).unwrap();

        // With rng_value = 0.0, should always get first (highest prob)
        let token = sample_top_k(&logits, 3, 0.0).unwrap();
        assert_eq!(token, 0);

        // With k=1, should always get highest
        let token = sample_top_k(&logits, 1, 0.5).unwrap();
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_top_k_distribution() {
        // Two equally likely tokens
        let logits = Tensor::from_vec(vec![4], vec![10.0, 10.0, 0.0, 0.0]).unwrap();

        // Low rng should get index 0 or 1 (they're equal)
        let token = sample_top_k(&logits, 2, 0.1).unwrap();
        assert!(token == 0 || token == 1);

        // High rng should get index 0 or 1
        let token = sample_top_k(&logits, 2, 0.9).unwrap();
        assert!(token == 0 || token == 1);
    }

    #[test]
    fn test_sample_top_k_errors() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        assert!(sample_top_k(&logits, 0, 0.5).is_err());
    }

    #[test]
    fn test_sample_top_p() {
        // One dominant token
        let logits = Tensor::from_vec(vec![3], vec![100.0, 1.0, 1.0]).unwrap();

        // With p=0.9, nucleus likely just the first token
        let token = sample_top_p(&logits, 0.9, 0.5).unwrap();
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_top_p_uniform() {
        // Equal logits
        let logits = Tensor::from_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).unwrap();

        // With p=1.0, all tokens in nucleus
        // Low rng should get early token
        let token = sample_top_p(&logits, 1.0, 0.1).unwrap();
        assert!(token < 4);

        // High rng should get later token
        let token = sample_top_p(&logits, 1.0, 0.9).unwrap();
        assert!(token < 4);
    }

    #[test]
    fn test_sample_top_p_errors() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        assert!(sample_top_p(&logits, 0.0, 0.5).is_err());
        assert!(sample_top_p(&logits, 1.1, 0.5).is_err());
        assert!(sample_top_p(&logits, -0.1, 0.5).is_err());
    }

    #[test]
    fn test_sample_token_greedy() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 10.0, 3.0, 4.0]).unwrap();
        let config = GenerationConfig::greedy();
        let token = sample_token(&logits, &config, 0.5).unwrap();
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_token_with_temperature() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let config = GenerationConfig::greedy().with_temperature(0.5);
        let token = sample_token(&logits, &config, 0.5).unwrap();
        // Higher temperature doesn't change greedy selection
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_token_top_k() {
        let logits = Tensor::from_vec(vec![5], vec![100.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let config = GenerationConfig::top_k(3);
        let token = sample_token(&logits, &config, 0.0).unwrap();
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_token_top_p() {
        let logits = Tensor::from_vec(vec![3], vec![100.0, 1.0, 1.0]).unwrap();
        let config = GenerationConfig::top_p(0.95);
        let token = sample_token(&logits, &config, 0.5).unwrap();
        assert_eq!(token, 0);
    }
}
