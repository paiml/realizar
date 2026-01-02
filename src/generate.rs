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

use crate::{
    error::{RealizarError, Result},
    layers::softmax,
    tensor::Tensor,
};

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

    Ok(sample_from_distribution(
        &normalized_probs,
        &indices,
        rng_value,
    ))
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

// ==================== Advanced Sampling Features ====================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Stop sequence detector for generation termination
///
/// Detects when generated text matches stop sequences and signals termination.
/// Supports both token ID sequences and string patterns.
#[derive(Debug, Clone, Default)]
pub struct StopSequenceDetector {
    /// Token ID sequences to stop on
    token_sequences: Vec<Vec<usize>>,
    /// String patterns to stop on
    string_patterns: Vec<String>,
    /// Buffer for partial matches (token-based)
    token_buffer: Vec<usize>,
    /// Maximum sequence length to track
    max_seq_len: usize,
}

impl StopSequenceDetector {
    /// Create new stop sequence detector
    pub fn new() -> Self {
        Self {
            token_sequences: Vec::new(),
            string_patterns: Vec::new(),
            token_buffer: Vec::new(),
            max_seq_len: 0,
        }
    }

    /// Add a token ID sequence as stop condition
    #[must_use]
    pub fn with_token_sequence(mut self, sequence: Vec<usize>) -> Self {
        if !sequence.is_empty() {
            self.max_seq_len = self.max_seq_len.max(sequence.len());
            self.token_sequences.push(sequence);
        }
        self
    }

    /// Add a string pattern as stop condition
    #[must_use]
    pub fn with_string_pattern(mut self, pattern: impl Into<String>) -> Self {
        let pattern = pattern.into();
        if !pattern.is_empty() {
            self.string_patterns.push(pattern);
        }
        self
    }

    /// Add multiple stop sequences from strings
    #[must_use]
    pub fn with_stop_strings(mut self, stops: Vec<String>) -> Self {
        for stop in stops {
            if !stop.is_empty() {
                self.string_patterns.push(stop);
            }
        }
        self
    }

    /// Check if a new token triggers a stop condition
    ///
    /// Returns true if generation should stop.
    pub fn check_token(&mut self, token_id: usize) -> bool {
        // Add to buffer
        self.token_buffer.push(token_id);

        // Trim buffer to max sequence length
        if self.token_buffer.len() > self.max_seq_len && self.max_seq_len > 0 {
            self.token_buffer.remove(0);
        }

        // Check token sequences
        for seq in &self.token_sequences {
            if self.token_buffer.ends_with(seq) {
                return true;
            }
        }

        false
    }

    /// Check if generated text contains a stop string
    ///
    /// Returns Some(position) if stop found, None otherwise.
    pub fn check_text(&self, text: &str) -> Option<usize> {
        for pattern in &self.string_patterns {
            if let Some(pos) = text.find(pattern) {
                return Some(pos);
            }
        }
        None
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.token_buffer.clear();
    }

    /// Check if detector has any stop conditions configured
    pub fn has_conditions(&self) -> bool {
        !self.token_sequences.is_empty() || !self.string_patterns.is_empty()
    }
}

/// Repetition penalty configuration
///
/// Penalizes tokens that have appeared in the context to reduce repetition.
/// Higher values = stronger penalty (1.0 = no penalty).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepetitionPenaltyConfig {
    /// Penalty multiplier for repeated tokens (1.0 = no penalty, >1.0 = penalty)
    pub penalty: f32,
    /// Number of recent tokens to consider (0 = all)
    pub window_size: usize,
}

impl Default for RepetitionPenaltyConfig {
    fn default() -> Self {
        Self {
            penalty: 1.0, // No penalty by default
            window_size: 64,
        }
    }
}

impl RepetitionPenaltyConfig {
    /// Create with specified penalty
    pub fn new(penalty: f32) -> Self {
        Self {
            penalty,
            window_size: 64,
        }
    }

    /// Set window size for context
    #[must_use]
    pub fn with_window(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Check if penalty is enabled
    pub fn is_enabled(&self) -> bool {
        (self.penalty - 1.0).abs() > 1e-6
    }
}

/// Apply repetition penalty to logits
///
/// Divides logits of tokens that appear in context by the penalty factor.
///
/// # Arguments
///
/// * `logits` - Raw logits from model
/// * `context_tokens` - List of previously generated token IDs
/// * `config` - Repetition penalty configuration
///
/// # Returns
///
/// Logits with repetition penalty applied
pub fn apply_repetition_penalty(
    logits: &Tensor<f32>,
    context_tokens: &[usize],
    config: &RepetitionPenaltyConfig,
) -> Tensor<f32> {
    if !config.is_enabled() || context_tokens.is_empty() {
        return logits.clone();
    }

    let data = logits.data();
    let mut penalized = data.to_vec();
    let vocab_size = data.len();

    // Get relevant context window
    let window_start = if config.window_size > 0 && context_tokens.len() > config.window_size {
        context_tokens.len() - config.window_size
    } else {
        0
    };
    let relevant_tokens = &context_tokens[window_start..];

    // Apply penalty to each token in context
    for &token_id in relevant_tokens {
        if token_id < vocab_size {
            let logit = penalized[token_id];
            // For positive logits, divide by penalty
            // For negative logits, multiply by penalty
            penalized[token_id] = if logit > 0.0 {
                logit / config.penalty
            } else {
                logit * config.penalty
            };
        }
    }

    Tensor::from_vec(logits.shape().to_vec(), penalized)
        .expect("Shape should match original logits")
}

/// Presence and frequency penalty configuration (OpenAI-style)
///
/// - Presence penalty: Constant penalty for tokens that appear at least once
/// - Frequency penalty: Penalty proportional to token frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresenceFrequencyPenalty {
    /// Presence penalty (penalty if token appeared at all)
    pub presence_penalty: f32,
    /// Frequency penalty (penalty per occurrence)
    pub frequency_penalty: f32,
}

impl Default for PresenceFrequencyPenalty {
    fn default() -> Self {
        Self {
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

impl PresenceFrequencyPenalty {
    /// Create new penalty config
    pub fn new(presence: f32, frequency: f32) -> Self {
        Self {
            presence_penalty: presence,
            frequency_penalty: frequency,
        }
    }

    /// Check if any penalty is enabled
    pub fn is_enabled(&self) -> bool {
        self.presence_penalty.abs() > 1e-6 || self.frequency_penalty.abs() > 1e-6
    }
}

/// Apply presence and frequency penalties to logits
///
/// Formula: logit -= presence_penalty * (1 if token in context else 0)
/// Formula: logit -= frequency_penalty * count(token in context)
///
/// # Arguments
///
/// * `logits` - Raw logits from model
/// * `context_tokens` - List of previously generated token IDs
/// * `config` - Presence/frequency penalty configuration
///
/// # Returns
///
/// Logits with penalties applied
pub fn apply_presence_frequency_penalty(
    logits: &Tensor<f32>,
    context_tokens: &[usize],
    config: &PresenceFrequencyPenalty,
) -> Tensor<f32> {
    if !config.is_enabled() || context_tokens.is_empty() {
        return logits.clone();
    }

    let data = logits.data();
    let mut penalized = data.to_vec();
    let vocab_size = data.len();

    // Count token frequencies
    let mut token_counts: HashMap<usize, usize> = HashMap::new();
    for &token_id in context_tokens {
        if token_id < vocab_size {
            *token_counts.entry(token_id).or_insert(0) += 1;
        }
    }

    // Apply penalties
    for (token_id, count) in token_counts {
        let presence = if count > 0 { 1.0 } else { 0.0 };
        penalized[token_id] -= config.presence_penalty * presence;
        penalized[token_id] -= config.frequency_penalty * (count as f32);
    }

    Tensor::from_vec(logits.shape().to_vec(), penalized)
        .expect("Shape should match original logits")
}

/// Logit bias configuration
///
/// Allows adjusting specific token probabilities before sampling.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LogitBias {
    /// Map of token ID to bias value (added to logit)
    biases: HashMap<usize, f32>,
}

impl LogitBias {
    /// Create empty logit bias
    pub fn new() -> Self {
        Self {
            biases: HashMap::new(),
        }
    }

    /// Add bias for a specific token
    #[must_use]
    pub fn with_bias(mut self, token_id: usize, bias: f32) -> Self {
        self.biases.insert(token_id, bias);
        self
    }

    /// Add multiple biases from a map
    #[must_use]
    pub fn with_biases(mut self, biases: HashMap<usize, f32>) -> Self {
        self.biases.extend(biases);
        self
    }

    /// Check if any biases are configured
    pub fn is_empty(&self) -> bool {
        self.biases.is_empty()
    }

    /// Get bias for a token (0.0 if not set)
    pub fn get(&self, token_id: usize) -> f32 {
        self.biases.get(&token_id).copied().unwrap_or(0.0)
    }
}

/// Apply logit bias to logits
///
/// # Arguments
///
/// * `logits` - Raw logits from model
/// * `bias` - Logit bias configuration
///
/// # Returns
///
/// Logits with biases applied
pub fn apply_logit_bias(logits: &Tensor<f32>, bias: &LogitBias) -> Tensor<f32> {
    if bias.is_empty() {
        return logits.clone();
    }

    let data = logits.data();
    let mut biased = data.to_vec();
    let vocab_size = data.len();

    for (&token_id, &bias_value) in &bias.biases {
        if token_id < vocab_size {
            biased[token_id] += bias_value;
        }
    }

    Tensor::from_vec(logits.shape().to_vec(), biased).expect("Shape should match original logits")
}

/// Min-P sampling: filter tokens below a probability threshold relative to max
///
/// Keeps tokens where prob >= min_p * max_prob
///
/// # Arguments
///
/// * `logits` - Logits for the vocabulary
/// * `min_p` - Minimum probability ratio (0.0 to 1.0)
/// * `rng_value` - Random value in [0, 1) for sampling
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if min_p is not in [0, 1] or logits are empty
pub fn sample_min_p(logits: &Tensor<f32>, min_p: f32, rng_value: f32) -> Result<usize> {
    let data = logits.data();
    if data.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Logits cannot be empty".to_string(),
        });
    }
    if !(0.0..=1.0).contains(&min_p) {
        return Err(RealizarError::InvalidShape {
            reason: "min_p must be in [0, 1]".to_string(),
        });
    }

    // Convert to probabilities
    let probs_tensor = softmax(logits)?;
    let probs = probs_tensor.data();

    // Find max probability
    let max_prob = probs.iter().copied().fold(0.0_f32, f32::max);
    let threshold = min_p * max_prob;

    // Keep tokens above threshold
    let mut candidates: Vec<(usize, f32)> = probs
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, p)| *p >= threshold)
        .collect();

    // Sort by probability descending
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if candidates.is_empty() {
        // Fallback to argmax
        return sample_greedy(logits);
    }

    // Renormalize and sample
    let sum: f32 = candidates.iter().map(|(_, p)| p).sum();
    let normalized: Vec<f32> = candidates.iter().map(|(_, p)| p / sum).collect();
    let indices: Vec<usize> = candidates.iter().map(|(idx, _)| *idx).collect();

    Ok(sample_from_distribution(&normalized, &indices, rng_value))
}

/// Mirostat sampling state for adaptive perplexity targeting
///
/// Implements Mirostat 2.0 algorithm from the paper:
/// "Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity"
#[derive(Debug, Clone)]
pub struct MirostatState {
    /// Target surprise value (tau)
    pub tau: f32,
    /// Learning rate (eta)
    pub eta: f32,
    /// Current surprise estimate (mu)
    pub mu: f32,
}

impl Default for MirostatState {
    fn default() -> Self {
        Self {
            tau: 5.0, // Default target surprise
            eta: 0.1, // Learning rate
            mu: 10.0, // Initial mu = 2 * tau
        }
    }
}

impl MirostatState {
    /// Create new Mirostat state with specified tau
    pub fn new(tau: f32) -> Self {
        Self {
            tau,
            eta: 0.1,
            mu: 2.0 * tau,
        }
    }

    /// Set learning rate
    #[must_use]
    pub fn with_eta(mut self, eta: f32) -> Self {
        self.eta = eta;
        self
    }

    /// Update mu based on observed surprise
    pub fn update(&mut self, observed_surprise: f32) {
        self.mu -= self.eta * (observed_surprise - self.tau);
    }
}

/// Mirostat 2.0 sampling: adaptive sampling to target perplexity
///
/// # Arguments
///
/// * `logits` - Logits for the vocabulary
/// * `state` - Mirostat state (will be updated)
/// * `rng_value` - Random value in [0, 1) for sampling
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if logits are empty
pub fn sample_mirostat(
    logits: &Tensor<f32>,
    state: &mut MirostatState,
    rng_value: f32,
) -> Result<usize> {
    let data = logits.data();
    if data.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Logits cannot be empty".to_string(),
        });
    }

    // Convert to probabilities
    let probs_tensor = softmax(logits)?;
    let probs = probs_tensor.data();

    // Sort by probability descending
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Save top candidate for fallback
    let top_candidate = indexed[0];

    // Calculate surprise values and find cutoff
    let mut candidates = Vec::new();
    for (idx, prob) in indexed {
        let surprise = -prob.ln();
        if surprise > state.mu {
            break;
        }
        candidates.push((idx, prob));
    }

    // Ensure at least one candidate
    if candidates.is_empty() {
        candidates.push(top_candidate);
    }

    // Renormalize and sample
    let sum: f32 = candidates.iter().map(|(_, p)| p).sum();
    let normalized: Vec<f32> = candidates.iter().map(|(_, p)| p / sum).collect();
    let indices: Vec<usize> = candidates.iter().map(|(idx, _)| *idx).collect();

    let selected = sample_from_distribution(&normalized, &indices, rng_value);
    let selected_idx = indices.iter().position(|&i| i == selected).unwrap_or(0);
    let selected_prob = candidates[selected_idx].1;

    // Update mu based on observed surprise
    let observed_surprise = -selected_prob.ln();
    state.update(observed_surprise);

    Ok(selected)
}

/// Tail-Free Sampling (TFS): Filter tokens based on probability second derivatives
///
/// TFS analyzes the "tail" of the probability distribution and removes tokens
/// in the low-probability tail. It computes second derivatives to find where
/// the distribution starts to flatten out.
///
/// # Arguments
///
/// * `logits` - Logits for the vocabulary
/// * `z` - TFS parameter (0.0 to 1.0, higher = more tokens kept)
/// * `rng_value` - Random value in [0, 1) for sampling
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if logits are empty
pub fn sample_tfs(logits: &Tensor<f32>, z: f32, rng_value: f32) -> Result<usize> {
    let data = logits.data();
    if data.is_empty() {
        return Err(crate::error::RealizarError::InvalidShape {
            reason: "Logits cannot be empty".to_string(),
        });
    }

    // Convert to probabilities
    let max_logit = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = data.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

    // Sort by probability descending
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if indexed.len() < 3 {
        // Not enough tokens for second derivative, use greedy
        return Ok(indexed[0].0);
    }

    // Compute first derivatives (differences between consecutive probabilities)
    let first_derivatives: Vec<f32> = indexed
        .windows(2)
        .map(|w| (w[0].1 - w[1].1).abs())
        .collect();

    // Compute second derivatives
    let second_derivatives: Vec<f32> = first_derivatives
        .windows(2)
        .map(|w| (w[0] - w[1]).abs())
        .collect();

    // Normalize second derivatives
    let sum_second: f32 = second_derivatives.iter().sum();
    let normalized: Vec<f32> = if sum_second > 1e-9 {
        second_derivatives.iter().map(|&x| x / sum_second).collect()
    } else {
        vec![1.0 / second_derivatives.len() as f32; second_derivatives.len()]
    };

    // Find cumulative sum and cutoff point
    let mut cumsum = 0.0;
    let mut cutoff_idx = indexed.len();
    for (i, &val) in normalized.iter().enumerate() {
        cumsum += val;
        if cumsum > z {
            cutoff_idx = i + 2; // +2 because second derivative is 2 steps behind
            break;
        }
    }

    // Keep tokens up to cutoff
    let kept: Vec<(usize, f32)> = indexed.into_iter().take(cutoff_idx.max(1)).collect();

    // Renormalize and sample
    let sum_kept: f32 = kept.iter().map(|(_, p)| p).sum();
    let normalized_kept: Vec<f32> = kept.iter().map(|(_, p)| p / sum_kept).collect();
    let indices: Vec<usize> = kept.iter().map(|(idx, _)| *idx).collect();

    Ok(sample_from_distribution(
        &normalized_kept,
        &indices,
        rng_value,
    ))
}

/// Locally Typical Sampling: Sample based on local typicality
///
/// Typical sampling selects tokens whose information content is close to
/// the expected information content (entropy) of the distribution.
/// This tends to produce more "typical" text.
///
/// Reference: Meister et al. (2022) "Locally Typical Sampling"
///
/// # Arguments
///
/// * `logits` - Logits for the vocabulary
/// * `p` - Cumulative probability mass to keep (0.0 to 1.0)
/// * `rng_value` - Random value in [0, 1) for sampling
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if logits are empty
pub fn sample_typical(logits: &Tensor<f32>, p: f32, rng_value: f32) -> Result<usize> {
    let data = logits.data();
    if data.is_empty() {
        return Err(crate::error::RealizarError::InvalidShape {
            reason: "Logits cannot be empty".to_string(),
        });
    }

    // Convert to probabilities
    let max_logit = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = data.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

    // Compute entropy (expected information content)
    let entropy: f32 = -probs
        .iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| p * p.ln())
        .sum::<f32>();

    // Compute information content for each token: -log(p)
    // Then compute deviation from entropy: |info - entropy|
    let mut indexed: Vec<(usize, f32, f32)> = probs
        .iter()
        .enumerate()
        .filter(|(_, &prob)| prob > 1e-10)
        .map(|(i, &prob)| {
            let info = -prob.ln();
            let deviation = (info - entropy).abs();
            (i, prob, deviation)
        })
        .collect();

    // Sort by deviation (most typical first)
    indexed.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    // Keep tokens until cumulative probability exceeds p
    let mut cumsum = 0.0;
    let mut kept: Vec<(usize, f32)> = Vec::new();
    for (idx, prob, _) in indexed {
        kept.push((idx, prob));
        cumsum += prob;
        if cumsum >= p {
            break;
        }
    }

    // Ensure at least one token
    if kept.is_empty() {
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        return Ok(max_idx);
    }

    // Renormalize and sample
    let sum_kept: f32 = kept.iter().map(|(_, p)| p).sum();
    let normalized: Vec<f32> = kept.iter().map(|(_, p)| p / sum_kept).collect();
    let indices: Vec<usize> = kept.iter().map(|(idx, _)| *idx).collect();

    Ok(sample_from_distribution(&normalized, &indices, rng_value))
}

/// DRY (Don't Repeat Yourself) sampling configuration
///
/// DRY sampling penalizes n-gram repetitions to prevent the model from
/// generating repetitive sequences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DryConfig {
    /// Multiplier for the penalty (higher = stronger penalty)
    pub multiplier: f32,
    /// Base value for exponential penalty growth
    pub base: f32,
    /// Minimum n-gram length to consider
    pub allowed_length: usize,
    /// Maximum sequence length to check for repetitions
    pub penalty_last_n: usize,
}

impl Default for DryConfig {
    fn default() -> Self {
        Self {
            multiplier: 0.8,
            base: 1.75,
            allowed_length: 2,
            penalty_last_n: 256,
        }
    }
}

impl DryConfig {
    /// Create new DRY config with specified multiplier
    pub fn new(multiplier: f32) -> Self {
        Self {
            multiplier,
            ..Default::default()
        }
    }

    /// Set the base for exponential penalty
    #[must_use]
    pub fn with_base(mut self, base: f32) -> Self {
        self.base = base;
        self
    }

    /// Set minimum n-gram length
    #[must_use]
    pub fn with_allowed_length(mut self, len: usize) -> Self {
        self.allowed_length = len;
        self
    }

    /// Set penalty window size
    #[must_use]
    pub fn with_penalty_last_n(mut self, n: usize) -> Self {
        self.penalty_last_n = n;
        self
    }

    /// Check if DRY is enabled
    pub fn is_enabled(&self) -> bool {
        self.multiplier > 0.0
    }
}

/// Apply DRY (Don't Repeat Yourself) penalty to logits
///
/// Penalizes tokens that would extend n-gram repetitions in the context.
///
/// # Arguments
///
/// * `logits` - Raw logits from model
/// * `context_tokens` - List of previously generated token IDs
/// * `config` - DRY configuration
///
/// # Returns
///
/// Logits with DRY penalty applied
pub fn apply_dry_penalty(
    logits: &Tensor<f32>,
    context_tokens: &[usize],
    config: &DryConfig,
) -> Tensor<f32> {
    if !config.is_enabled() || context_tokens.len() < config.allowed_length {
        return logits.clone();
    }

    let data = logits.data();
    let mut penalized = data.to_vec();

    // Get relevant context window
    let window_start = if context_tokens.len() > config.penalty_last_n {
        context_tokens.len() - config.penalty_last_n
    } else {
        0
    };
    let context = &context_tokens[window_start..];

    // For each possible next token, check if it would extend a repetition
    for (token_id, logit) in penalized.iter_mut().enumerate() {
        let match_len = find_ngram_match_length(context, token_id, config.allowed_length);

        if match_len >= config.allowed_length {
            // Apply exponential penalty based on match length
            let penalty =
                config.multiplier * config.base.powi((match_len - config.allowed_length) as i32);
            *logit -= penalty;
        }
    }

    Tensor::from_vec(logits.shape().to_vec(), penalized)
        .expect("Shape should match original logits")
}

/// Find the length of the longest n-gram that would be repeated if we add this token
fn find_ngram_match_length(context: &[usize], next_token: usize, min_len: usize) -> usize {
    if context.len() < min_len {
        return 0;
    }

    let mut max_match = 0;

    // Build the sequence ending with the potential next token
    // Then search for earlier occurrences
    for end_pos in min_len..=context.len() {
        let search_start = context.len() - end_pos;
        let suffix = &context[search_start..];

        // Look for this suffix earlier in the context
        for start in 0..(context.len() - end_pos) {
            let potential_end = start + end_pos;
            if potential_end >= context.len() {
                continue;
            }

            // Check if suffix matches
            if context[start..potential_end] == *suffix {
                // Check if the next token after this match equals our candidate
                if potential_end < context.len() && context[potential_end] == next_token {
                    max_match = max_match.max(end_pos + 1);
                }
            }
        }
    }

    max_match
}

// ===== XTC (Exclude Top Choices) Sampling =====

/// XTC (Exclude Top Choices) sampling configuration
///
/// XTC removes the most likely tokens with some probability, forcing the model
/// to explore alternative completions. This can increase creativity and diversity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XtcConfig {
    /// Probability of excluding top tokens (0.0 = disabled, 1.0 = always exclude)
    pub probability: f32,
    /// Threshold for excluding tokens (tokens with prob >= threshold may be excluded)
    pub threshold: f32,
    /// Minimum number of tokens to keep after exclusion
    pub min_keep: usize,
}

impl Default for XtcConfig {
    fn default() -> Self {
        Self {
            probability: 0.0,
            threshold: 0.5,
            min_keep: 1,
        }
    }
}

impl XtcConfig {
    /// Create new XTC config with specified probability
    pub fn new(probability: f32) -> Self {
        Self {
            probability,
            ..Default::default()
        }
    }

    /// Set threshold
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set minimum tokens to keep
    #[must_use]
    pub fn with_min_keep(mut self, min_keep: usize) -> Self {
        self.min_keep = min_keep;
        self
    }

    /// Check if XTC is enabled
    pub fn is_enabled(&self) -> bool {
        self.probability > 0.0
    }
}

/// Apply XTC (Exclude Top Choices) sampling
///
/// XTC randomly excludes top tokens to increase diversity.
///
/// # Arguments
///
/// * `logits` - Raw logits from the model
/// * `config` - XTC configuration
/// * `rng_value` - Random value [0, 1) for stochastic exclusion decision
///
/// # Returns
///
/// Modified logits with top choices potentially excluded
pub fn apply_xtc(logits: &Tensor<f32>, config: &XtcConfig, rng_value: f32) -> Tensor<f32> {
    if !config.is_enabled() || rng_value >= config.probability {
        return logits.clone();
    }

    let data = logits.data();
    if data.len() <= config.min_keep {
        return logits.clone();
    }

    // Convert to probabilities
    let max_logit = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = data.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

    // Find tokens above threshold
    let mut excluded_count = 0;
    let mut modified = data.to_vec();

    // Sort by probability descending to find top tokens
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Exclude top tokens above threshold, respecting min_keep
    for (idx, prob) in &indexed {
        if *prob >= config.threshold && data.len() - excluded_count > config.min_keep {
            modified[*idx] = f32::NEG_INFINITY;
            excluded_count += 1;
        }
    }

    Tensor::from_vec(logits.shape().to_vec(), modified).expect("Shape should match original logits")
}

// ===== Eta Sampling =====

/// Eta Sampling (entropy-based truncation)
///
/// Eta sampling dynamically adjusts the truncation threshold based on the
/// entropy of the probability distribution. Higher entropy = more tokens kept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EtaConfig {
    /// Eta parameter (controls sensitivity to entropy)
    pub eta: f32,
    /// Minimum probability to keep (absolute floor)
    pub min_p: f32,
}

impl Default for EtaConfig {
    fn default() -> Self {
        Self {
            eta: 0.3,
            min_p: 0.0001,
        }
    }
}

impl EtaConfig {
    /// Create new Eta config
    pub fn new(eta: f32) -> Self {
        Self {
            eta,
            ..Default::default()
        }
    }

    /// Set minimum probability
    #[must_use]
    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    /// Check if eta sampling is enabled
    pub fn is_enabled(&self) -> bool {
        self.eta > 0.0
    }
}

/// Apply Eta sampling
///
/// # Arguments
///
/// * `logits` - Raw logits from the model
/// * `config` - Eta configuration
/// * `rng_value` - Random value [0, 1) for sampling
///
/// # Returns
///
/// Index of the selected token
///
/// # Errors
///
/// Returns error if logits are empty
pub fn sample_eta(logits: &Tensor<f32>, config: &EtaConfig, rng_value: f32) -> Result<usize> {
    let data = logits.data();
    if data.is_empty() {
        return Err(crate::error::RealizarError::InvalidShape {
            reason: "Logits cannot be empty".to_string(),
        });
    }

    // Convert to probabilities
    let max_logit = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = data.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

    // Compute entropy
    let entropy: f32 = -probs
        .iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| p * p.ln())
        .sum::<f32>();

    // Compute dynamic threshold: eta * exp(-entropy)
    let threshold = (config.eta * (-entropy).exp()).max(config.min_p);

    // Keep tokens above threshold
    let mut indexed: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .filter(|(_, &p)| p >= threshold)
        .map(|(i, &p)| (i, p))
        .collect();

    // Ensure at least one token
    if indexed.is_empty() {
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        return Ok(max_idx);
    }

    // Sort by probability descending
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Renormalize and sample
    let sum_kept: f32 = indexed.iter().map(|(_, p)| p).sum();
    let normalized: Vec<f32> = indexed.iter().map(|(_, p)| p / sum_kept).collect();
    let indices: Vec<usize> = indexed.iter().map(|(idx, _)| *idx).collect();

    Ok(sample_from_distribution(&normalized, &indices, rng_value))
}

// ===== Token Healing =====

/// Token Healing configuration
///
/// Token healing fixes broken token boundaries by backing up and re-tokenizing
/// when a partial token is detected at the prompt boundary.
#[derive(Debug, Clone, Default)]
pub struct TokenHealingConfig {
    /// Enable token healing
    pub enabled: bool,
    /// Maximum characters to back up
    pub max_backup_chars: usize,
}

impl TokenHealingConfig {
    /// Create new token healing config
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            max_backup_chars: 10,
        }
    }

    /// Set max backup characters
    #[must_use]
    pub fn with_max_backup(mut self, chars: usize) -> Self {
        self.max_backup_chars = chars;
        self
    }
}

/// Token healing result
#[derive(Debug, Clone)]
pub struct TokenHealingResult {
    /// Adjusted prompt tokens (may be shorter than original)
    pub adjusted_tokens: Vec<usize>,
    /// Prefix constraint for first generated token
    pub prefix_constraint: Option<String>,
    /// Number of tokens removed from end
    pub tokens_removed: usize,
}

/// Analyze prompt for token healing
///
/// Detects if the last token is a partial token that should be healed.
/// This is a simplified implementation - full implementation requires tokenizer access.
///
/// # Arguments
///
/// * `prompt_tokens` - Original prompt tokens
/// * `last_token_text` - Text of the last token (if available)
///
/// # Returns
///
/// Token healing result with adjusted tokens
pub fn analyze_token_healing(
    prompt_tokens: &[usize],
    last_token_text: Option<&str>,
) -> TokenHealingResult {
    // Simple heuristic: if last token is a partial word (no space, single char),
    // we might want to heal it
    let should_heal = last_token_text.is_some_and(|text| {
        !text.is_empty()
            && !text.starts_with(' ')
            && text.len() <= 3
            && text.chars().all(char::is_alphanumeric)
    });

    if should_heal && !prompt_tokens.is_empty() {
        TokenHealingResult {
            adjusted_tokens: prompt_tokens[..prompt_tokens.len() - 1].to_vec(),
            prefix_constraint: last_token_text.map(String::from),
            tokens_removed: 1,
        }
    } else {
        TokenHealingResult {
            adjusted_tokens: prompt_tokens.to_vec(),
            prefix_constraint: None,
            tokens_removed: 0,
        }
    }
}

// ===== Classifier-Free Guidance (CFG) =====

/// Classifier-Free Guidance configuration
///
/// CFG improves generation quality by comparing conditional and unconditional
/// logits, amplifying the difference to steer generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfgConfig {
    /// Guidance scale (1.0 = no guidance, higher = stronger guidance)
    pub scale: f32,
    /// Negative prompt tokens (for unconditional generation)
    pub negative_prompt_tokens: Vec<usize>,
}

impl Default for CfgConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            negative_prompt_tokens: Vec::new(),
        }
    }
}

impl CfgConfig {
    /// Create new CFG config with specified scale
    pub fn new(scale: f32) -> Self {
        Self {
            scale,
            ..Default::default()
        }
    }

    /// Set negative prompt tokens
    #[must_use]
    pub fn with_negative_prompt(mut self, tokens: Vec<usize>) -> Self {
        self.negative_prompt_tokens = tokens;
        self
    }

    /// Check if CFG is enabled
    pub fn is_enabled(&self) -> bool {
        self.scale > 1.0
    }
}

/// Apply Classifier-Free Guidance
///
/// Combines conditional and unconditional logits using the CFG formula:
/// output = unconditional + scale * (conditional - unconditional)
///
/// # Arguments
///
/// * `conditional_logits` - Logits from the model with the prompt
/// * `unconditional_logits` - Logits from the model with negative/empty prompt
/// * `scale` - Guidance scale
///
/// # Returns
///
/// Guided logits
///
/// # Errors
///
/// Returns error if conditional and unconditional logits have different shapes
pub fn apply_cfg(
    conditional_logits: &Tensor<f32>,
    unconditional_logits: &Tensor<f32>,
    scale: f32,
) -> Result<Tensor<f32>> {
    if conditional_logits.shape() != unconditional_logits.shape() {
        return Err(crate::error::RealizarError::ShapeMismatch {
            expected: conditional_logits.shape().to_vec(),
            actual: unconditional_logits.shape().to_vec(),
        });
    }

    let cond = conditional_logits.data();
    let uncond = unconditional_logits.data();

    // CFG formula: uncond + scale * (cond - uncond)
    let guided: Vec<f32> = cond
        .iter()
        .zip(uncond.iter())
        .map(|(&c, &u)| u + scale * (c - u))
        .collect();

    Tensor::from_vec(conditional_logits.shape().to_vec(), guided)
}

// ===== Prompt Caching =====

/// Prompt cache entry
#[derive(Debug, Clone)]
pub struct PromptCacheEntry {
    /// Token sequence
    pub tokens: Vec<usize>,
    /// Cached KV state (simplified - in practice would be actual KV tensors)
    pub kv_hash: u64,
    /// Number of times this entry has been hit
    pub hit_count: usize,
    /// Last access timestamp
    pub last_access: std::time::Instant,
}

/// Prompt cache for efficient prefix reuse
///
/// Caches prompt prefixes to avoid recomputation when generating multiple
/// completions with the same prefix.
#[derive(Debug)]
pub struct PromptCache {
    /// Cache entries keyed by token sequence hash
    entries: std::collections::HashMap<u64, PromptCacheEntry>,
    /// Maximum cache size
    max_entries: usize,
}

impl Default for PromptCache {
    fn default() -> Self {
        Self::new(100)
    }
}

impl PromptCache {
    /// Create new prompt cache
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            max_entries,
        }
    }

    /// Compute hash for token sequence
    fn hash_tokens(tokens: &[usize]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        tokens.hash(&mut hasher);
        hasher.finish()
    }

    /// Find longest matching prefix in cache
    pub fn find_prefix(&mut self, tokens: &[usize]) -> Option<(usize, u64)> {
        // Try progressively shorter prefixes
        for len in (1..=tokens.len()).rev() {
            let prefix = &tokens[..len];
            let hash = Self::hash_tokens(prefix);
            if let Some(entry) = self.entries.get_mut(&hash) {
                entry.hit_count += 1;
                entry.last_access = std::time::Instant::now();
                return Some((len, entry.kv_hash));
            }
        }
        None
    }

    /// Add entry to cache
    pub fn add(&mut self, tokens: Vec<usize>, kv_hash: u64) {
        // Evict if at capacity
        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        let hash = Self::hash_tokens(&tokens);
        self.entries.insert(
            hash,
            PromptCacheEntry {
                tokens,
                kv_hash,
                hit_count: 0,
                last_access: std::time::Instant::now(),
            },
        );
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) {
        if let Some((&key, _)) = self.entries.iter().min_by_key(|(_, v)| v.last_access) {
            self.entries.remove(&key);
        }
    }

    /// Get cache size
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> PromptCacheStats {
        let total_hits: usize = self.entries.values().map(|e| e.hit_count).sum();
        PromptCacheStats {
            entries: self.entries.len(),
            total_hits,
            max_entries: self.max_entries,
        }
    }
}

/// Prompt cache statistics
#[derive(Debug, Clone)]
pub struct PromptCacheStats {
    /// Number of entries in cache
    pub entries: usize,
    /// Total cache hits
    pub total_hits: usize,
    /// Maximum cache size
    pub max_entries: usize,
}

/// Beam search state for a single hypothesis
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Token sequence generated so far
    pub tokens: Vec<usize>,
    /// Cumulative log probability
    pub score: f32,
    /// Whether this hypothesis has finished (hit EOS)
    pub finished: bool,
}

impl BeamHypothesis {
    /// Create a new hypothesis starting with given tokens
    pub fn new(tokens: Vec<usize>, score: f32) -> Self {
        Self {
            tokens,
            score,
            finished: false,
        }
    }

    /// Extend hypothesis with a new token
    #[must_use]
    pub fn extend(&self, token: usize, log_prob: f32, is_eos: bool) -> Self {
        let mut new_tokens = self.tokens.clone();
        new_tokens.push(token);
        Self {
            tokens: new_tokens,
            score: self.score + log_prob,
            finished: is_eos,
        }
    }

    /// Get length-normalized score
    pub fn normalized_score(&self, length_penalty: f32) -> f32 {
        let len = self.tokens.len() as f32;
        self.score / len.powf(length_penalty)
    }
}

/// Beam search configuration
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of beams (hypotheses) to keep
    pub num_beams: usize,
    /// Length penalty (>1.0 favors longer sequences, <1.0 favors shorter)
    pub length_penalty: f32,
    /// Early stopping: stop when num_beams hypotheses are finished
    pub early_stopping: bool,
    /// Number of beams to return
    pub num_return: usize,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            num_beams: 4,
            length_penalty: 1.0,
            early_stopping: true,
            num_return: 1,
        }
    }
}

impl BeamSearchConfig {
    /// Create new beam search config
    pub fn new(num_beams: usize) -> Self {
        Self {
            num_beams,
            ..Default::default()
        }
    }

    /// Set length penalty
    #[must_use]
    pub fn with_length_penalty(mut self, penalty: f32) -> Self {
        self.length_penalty = penalty;
        self
    }

    /// Set early stopping
    #[must_use]
    pub fn with_early_stopping(mut self, early: bool) -> Self {
        self.early_stopping = early;
        self
    }

    /// Set number of sequences to return
    #[must_use]
    pub fn with_num_return(mut self, n: usize) -> Self {
        self.num_return = n;
        self
    }
}

/// Beam search state manager
#[derive(Debug, Clone)]
pub struct BeamSearchState {
    /// Current hypotheses
    pub hypotheses: Vec<BeamHypothesis>,
    /// Finished hypotheses
    pub finished: Vec<BeamHypothesis>,
    /// Configuration
    pub config: BeamSearchConfig,
}

impl BeamSearchState {
    /// Create new beam search state
    pub fn new(config: BeamSearchConfig, initial_tokens: Vec<usize>) -> Self {
        let hypotheses = vec![BeamHypothesis::new(initial_tokens, 0.0)];
        Self {
            hypotheses,
            finished: Vec::new(),
            config,
        }
    }

    /// Process a step with log probabilities for each hypothesis
    ///
    /// # Arguments
    ///
    /// * `log_probs_per_hyp` - Log probabilities for each token, for each hypothesis
    /// * `eos_token` - Optional end-of-sequence token ID
    pub fn step(&mut self, log_probs_per_hyp: &[Vec<f32>], eos_token: Option<usize>) {
        let mut candidates: Vec<BeamHypothesis> = Vec::new();

        for (hyp_idx, hyp) in self.hypotheses.iter().enumerate() {
            if hyp.finished {
                candidates.push(hyp.clone());
                continue;
            }

            let log_probs = &log_probs_per_hyp[hyp_idx];

            // Get top-k tokens for this hypothesis (k = num_beams * 2 for safety)
            let mut indexed: Vec<(usize, f32)> = log_probs
                .iter()
                .enumerate()
                .map(|(i, &lp)| (i, lp))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for &(token, log_prob) in indexed.iter().take(self.config.num_beams * 2) {
                let is_eos = eos_token == Some(token);
                let new_hyp = hyp.extend(token, log_prob, is_eos);

                if is_eos {
                    self.finished.push(new_hyp);
                } else {
                    candidates.push(new_hyp);
                }
            }
        }

        // Select top num_beams hypotheses by normalized score
        candidates.sort_by(|a, b| {
            let score_a = a.normalized_score(self.config.length_penalty);
            let score_b = b.normalized_score(self.config.length_penalty);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.hypotheses = candidates.into_iter().take(self.config.num_beams).collect();
    }

    /// Check if search should stop
    pub fn should_stop(&self) -> bool {
        if self.config.early_stopping && self.finished.len() >= self.config.num_beams {
            return true;
        }
        self.hypotheses.is_empty() || self.hypotheses.iter().all(|h| h.finished)
    }

    /// Get best completed hypotheses
    pub fn best_hypotheses(&self) -> Vec<BeamHypothesis> {
        let mut all: Vec<_> = self
            .finished
            .iter()
            .chain(self.hypotheses.iter())
            .cloned()
            .collect();
        all.sort_by(|a, b| {
            let score_a = a.normalized_score(self.config.length_penalty);
            let score_b = b.normalized_score(self.config.length_penalty);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all.into_iter().take(self.config.num_return).collect()
    }
}

/// Streaming generation callback type
///
/// The callback receives:
/// - token_id: The generated token ID
/// - token_text: Optional decoded text for the token
/// - is_final: Whether this is the last token
///
/// Returns `true` to continue, `false` to stop generation
pub type StreamCallback = Box<dyn FnMut(usize, Option<&str>, bool) -> bool + Send>;

/// Streaming generation state
#[derive(Debug)]
pub struct StreamingGenerator {
    /// Tokens generated so far
    pub tokens: Vec<usize>,
    /// Generated text so far
    pub text: String,
    /// Whether generation is complete
    pub finished: bool,
    /// Total tokens generated
    pub total_tokens: usize,
}

impl StreamingGenerator {
    /// Create new streaming generator
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            text: String::new(),
            finished: false,
            total_tokens: 0,
        }
    }

    /// Add a generated token
    pub fn add_token(&mut self, token_id: usize, token_text: Option<&str>) {
        self.tokens.push(token_id);
        if let Some(text) = token_text {
            self.text.push_str(text);
        }
        self.total_tokens += 1;
    }

    /// Mark generation as finished
    pub fn finish(&mut self) {
        self.finished = true;
    }

    /// Get current token count
    pub fn token_count(&self) -> usize {
        self.total_tokens
    }
}

impl Default for StreamingGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Extended generation configuration with advanced sampling options
#[derive(Debug, Clone, Default)]
pub struct AdvancedGenerationConfig {
    /// Base generation config
    pub base: GenerationConfig,
    /// Stop sequence detector
    pub stop_detector: Option<StopSequenceDetector>,
    /// Repetition penalty config
    pub repetition_penalty: Option<RepetitionPenaltyConfig>,
    /// Presence/frequency penalties
    pub presence_frequency: Option<PresenceFrequencyPenalty>,
    /// Logit bias
    pub logit_bias: Option<LogitBias>,
}

impl AdvancedGenerationConfig {
    /// Create with base config
    pub fn new(base: GenerationConfig) -> Self {
        Self {
            base,
            ..Default::default()
        }
    }

    /// Add stop sequences
    #[must_use]
    pub fn with_stop_sequences(mut self, stops: Vec<String>) -> Self {
        self.stop_detector = Some(StopSequenceDetector::new().with_stop_strings(stops));
        self
    }

    /// Add repetition penalty
    #[must_use]
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = Some(RepetitionPenaltyConfig::new(penalty));
        self
    }

    /// Add presence/frequency penalties
    #[must_use]
    pub fn with_presence_frequency(mut self, presence: f32, frequency: f32) -> Self {
        self.presence_frequency = Some(PresenceFrequencyPenalty::new(presence, frequency));
        self
    }

    /// Add logit bias
    #[must_use]
    pub fn with_logit_bias(mut self, bias: LogitBias) -> Self {
        self.logit_bias = Some(bias);
        self
    }
}

/// Apply all configured penalties and biases to logits
///
/// # Arguments
///
/// * `logits` - Raw logits from model
/// * `context_tokens` - Previously generated tokens
/// * `config` - Advanced generation configuration
///
/// # Returns
///
/// Logits with all penalties applied
pub fn apply_all_penalties(
    logits: &Tensor<f32>,
    context_tokens: &[usize],
    config: &AdvancedGenerationConfig,
) -> Tensor<f32> {
    let mut result = logits.clone();

    // Apply repetition penalty
    if let Some(ref rep_config) = config.repetition_penalty {
        result = apply_repetition_penalty(&result, context_tokens, rep_config);
    }

    // Apply presence/frequency penalty
    if let Some(ref pf_config) = config.presence_frequency {
        result = apply_presence_frequency_penalty(&result, context_tokens, pf_config);
    }

    // Apply logit bias
    if let Some(ref bias) = config.logit_bias {
        result = apply_logit_bias(&result, bias);
    }

    result
}

// ============================================================================
// Dynamic Temperature (temp_ext) - Entropy-based temperature adjustment
// ============================================================================

/// Configuration for dynamic temperature (temp_ext)
///
/// Adjusts temperature based on the entropy of the probability distribution.
/// When entropy is low (confident), uses higher temperature to increase diversity.
/// When entropy is high (uncertain), uses lower temperature to focus on likely tokens.
///
/// Reference: llama.cpp `llama_sampler_init_temp_ext`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynTempConfig {
    /// Base temperature
    pub temp: f32,
    /// Range around base temperature (min = temp - delta, max = temp + delta)
    pub delta: f32,
    /// Exponent for entropy mapping (higher = more aggressive adjustment)
    pub exponent: f32,
}

impl Default for DynTempConfig {
    fn default() -> Self {
        Self {
            temp: 1.0,
            delta: 0.0,
            exponent: 1.0,
        }
    }
}

impl DynTempConfig {
    /// Create a new dynamic temperature config
    pub fn new(temp: f32, delta: f32, exponent: f32) -> Self {
        Self {
            temp,
            delta,
            exponent,
        }
    }

    /// Create with just temperature (no dynamic adjustment)
    pub fn static_temp(temp: f32) -> Self {
        Self {
            temp,
            delta: 0.0,
            exponent: 1.0,
        }
    }
}

/// Apply dynamic temperature based on entropy
///
/// The algorithm:
/// 1. Calculate max possible entropy: -log(1/n)
/// 2. Calculate actual entropy: -sum(p * log(p))
/// 3. Normalize entropy to [0, 1]
/// 4. Map to temperature: min_temp + (max_temp - min_temp) * pow(norm_entropy, exponent)
/// 5. Apply calculated temperature to logits
///
/// # Arguments
///
/// * `logits` - Raw logits from model
/// * `config` - Dynamic temperature configuration
///
/// # Returns
///
/// Logits with dynamic temperature applied
pub fn apply_dynamic_temperature(logits: &Tensor<f32>, config: &DynTempConfig) -> Tensor<f32> {
    // If no delta, just apply static temperature
    if config.delta <= 0.0 {
        return apply_temperature(logits, config.temp).unwrap_or_else(|_| logits.clone());
    }

    let data = logits.data();
    if data.len() <= 1 {
        return logits.clone();
    }

    // Calculate softmax probabilities
    let max_logit = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = data.iter().map(|x| (x - max_logit).exp()).sum();
    let probs: Vec<f32> = data
        .iter()
        .map(|x| (x - max_logit).exp() / exp_sum)
        .collect();

    // Calculate maximum possible entropy: -log(1/n) = log(n)
    let max_entropy = (data.len() as f32).ln();

    // Calculate actual entropy: -sum(p * log(p))
    let entropy: f32 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();

    // Normalize entropy to [0, 1]
    let normalized_entropy = if max_entropy > 0.0 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Calculate dynamic temperature
    let min_temp = (config.temp - config.delta).max(0.0);
    let max_temp = config.temp + config.delta;
    let dyn_temp = min_temp + (max_temp - min_temp) * normalized_entropy.powf(config.exponent);

    // Apply calculated temperature
    apply_temperature(logits, dyn_temp).unwrap_or_else(|_| logits.clone())
}

// ============================================================================
// Infill/FIM Sampler - Fill-in-the-Middle for code completion
// ============================================================================

/// Configuration for infill/FIM (Fill-in-the-Middle) sampling
///
/// Used for code completion where the model generates text to fill a gap.
/// Handles EOG (End-of-Generation) tokens specially to determine when to stop.
///
/// Reference: llama.cpp `llama_sampler_init_infill`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfillConfig {
    /// EOG (End-of-Generation) token IDs
    pub eog_tokens: Vec<usize>,
    /// Ratio threshold: if 3*p_eog*n > p_txt, force EOG
    pub eog_ratio_threshold: f32,
}

impl Default for InfillConfig {
    fn default() -> Self {
        Self {
            eog_tokens: vec![],
            eog_ratio_threshold: 3.0,
        }
    }
}

impl InfillConfig {
    /// Create a new infill config with EOG tokens
    pub fn new(eog_tokens: Vec<usize>) -> Self {
        Self {
            eog_tokens,
            eog_ratio_threshold: 3.0,
        }
    }

    /// Set the EOG ratio threshold
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.eog_ratio_threshold = threshold;
        self
    }
}

/// Result of infill sampling
#[derive(Debug, Clone)]
pub struct InfillResult {
    /// Modified logits (with non-EOG tokens potentially zeroed)
    pub logits: Tensor<f32>,
    /// Whether to force EOG token
    pub force_eog: bool,
    /// Probability sum of text tokens
    pub p_txt: f32,
    /// Probability sum of EOG tokens
    pub p_eog: f32,
}

/// Apply infill sampling logic
///
/// This determines if the model should stop generating (emit EOG) based on
/// the relative probabilities of EOG vs text tokens.
///
/// # Arguments
///
/// * `logits` - Raw logits from model
/// * `config` - Infill configuration
///
/// # Returns
///
/// `InfillResult` with modified logits and EOG decision
pub fn apply_infill_sampling(logits: &Tensor<f32>, config: &InfillConfig) -> InfillResult {
    let data = logits.data();
    if data.is_empty() || config.eog_tokens.is_empty() {
        return InfillResult {
            logits: logits.clone(),
            force_eog: false,
            p_txt: 1.0,
            p_eog: 0.0,
        };
    }

    // Calculate softmax probabilities
    let max_logit = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = data.iter().map(|x| (x - max_logit).exp()).sum();
    let probs: Vec<f32> = data
        .iter()
        .map(|x| (x - max_logit).exp() / exp_sum)
        .collect();

    // Calculate p_eog and p_txt
    let mut p_eog: f32 = 0.0;
    let mut p_txt: f32 = 0.0;

    for (i, &p) in probs.iter().enumerate() {
        if config.eog_tokens.contains(&i) {
            p_eog += p;
        } else {
            p_txt += p;
        }
    }

    // Check if we should force EOG
    // Condition: 3 * p_eog * n > p_txt
    let n = data.len() as f32;
    let force_eog = config.eog_ratio_threshold * p_eog * n > p_txt;

    if force_eog {
        // Keep only EOG tokens
        let mut new_data = vec![f32::NEG_INFINITY; data.len()];
        let mut eog_sum = 0.0;

        for &eog_id in &config.eog_tokens {
            if eog_id < data.len() {
                new_data[eog_id] = data[eog_id];
                eog_sum += probs[eog_id];
            }
        }

        // Renormalize EOG tokens
        if eog_sum > 0.0 {
            for &eog_id in &config.eog_tokens {
                if eog_id < data.len() && new_data[eog_id] > f32::NEG_INFINITY {
                    // Convert back to logit scale
                    let normalized_p = probs[eog_id] / eog_sum;
                    new_data[eog_id] = normalized_p.ln();
                }
            }
        }

        InfillResult {
            logits: Tensor::from_vec(logits.shape().to_vec(), new_data)
                .unwrap_or_else(|_| logits.clone()),
            force_eog: true,
            p_txt,
            p_eog,
        }
    } else {
        InfillResult {
            logits: logits.clone(),
            force_eog: false,
            p_txt,
            p_eog,
        }
    }
}

// ============================================================================
// Sampler Chain - Composable sampler pipeline
// ============================================================================

/// Trait for samplers that can be chained together
pub trait Sampler: Send + Sync {
    /// Get the sampler name
    fn name(&self) -> &'static str;

    /// Apply the sampler to logits (in-place modification)
    fn apply(&self, logits: &mut Tensor<f32>, context: &SamplerContext);

    /// Clone the sampler (for use in chains)
    fn clone_box(&self) -> Box<dyn Sampler>;
}

/// Context passed to samplers during application
#[derive(Debug, Clone, Default)]
pub struct SamplerContext {
    /// Previously generated tokens
    pub tokens: Vec<usize>,
    /// Random value for stochastic samplers [0, 1)
    pub rng_value: f32,
    /// Current generation step
    pub step: usize,
}

impl SamplerContext {
    /// Create a new sampler context
    pub fn new() -> Self {
        Self::default()
    }

    /// Set tokens
    #[must_use]
    pub fn with_tokens(mut self, tokens: Vec<usize>) -> Self {
        self.tokens = tokens;
        self
    }

    /// Set RNG value
    #[must_use]
    pub fn with_rng(mut self, rng_value: f32) -> Self {
        self.rng_value = rng_value;
        self
    }

    /// Set step
    #[must_use]
    pub fn with_step(mut self, step: usize) -> Self {
        self.step = step;
        self
    }
}

/// A chain of samplers applied in sequence
pub struct SamplerChain {
    samplers: Vec<Box<dyn Sampler>>,
}

impl Default for SamplerChain {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplerChain {
    /// Create a new empty sampler chain
    pub fn new() -> Self {
        Self { samplers: vec![] }
    }

    /// Add a sampler to the chain (builder pattern)
    #[must_use]
    pub fn with_sampler<S: Sampler + 'static>(mut self, sampler: S) -> Self {
        self.samplers.push(Box::new(sampler));
        self
    }

    /// Push a boxed sampler to the chain
    pub fn push(&mut self, sampler: Box<dyn Sampler>) {
        self.samplers.push(sampler);
    }

    /// Get the number of samplers in the chain
    pub fn len(&self) -> usize {
        self.samplers.len()
    }

    /// Check if the chain is empty
    pub fn is_empty(&self) -> bool {
        self.samplers.is_empty()
    }

    /// Get sampler names in order
    pub fn names(&self) -> Vec<&'static str> {
        self.samplers.iter().map(|s| s.name()).collect()
    }

    /// Apply all samplers in sequence
    pub fn apply(&self, logits: &mut Tensor<f32>, context: &SamplerContext) {
        for sampler in &self.samplers {
            sampler.apply(logits, context);
        }
    }

    /// Sample a token after applying all samplers
    ///
    /// # Errors
    ///
    /// Returns error if sampling fails
    pub fn sample(&self, logits: &Tensor<f32>, context: &SamplerContext) -> Result<usize> {
        let mut modified = logits.clone();
        self.apply(&mut modified, context);
        sample_greedy(&modified)
    }
}

impl Clone for SamplerChain {
    fn clone(&self) -> Self {
        Self {
            samplers: self.samplers.iter().map(|s| s.clone_box()).collect(),
        }
    }
}

// Concrete sampler implementations for the chain

/// Temperature sampler
#[derive(Debug, Clone)]
pub struct TemperatureSampler {
    /// Temperature value (1.0 = no change)
    pub temp: f32,
}

impl TemperatureSampler {
    /// Create a new temperature sampler
    pub fn new(temp: f32) -> Self {
        Self { temp }
    }
}

impl Sampler for TemperatureSampler {
    fn name(&self) -> &'static str {
        "temperature"
    }

    fn apply(&self, logits: &mut Tensor<f32>, _context: &SamplerContext) {
        if let Ok(result) = apply_temperature(logits, self.temp) {
            *logits = result;
        }
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

/// Dynamic temperature sampler
#[derive(Debug, Clone)]
pub struct DynTempSampler {
    /// Dynamic temperature configuration
    pub config: DynTempConfig,
}

impl DynTempSampler {
    /// Create a new dynamic temperature sampler
    pub fn new(config: DynTempConfig) -> Self {
        Self { config }
    }
}

impl Sampler for DynTempSampler {
    fn name(&self) -> &'static str {
        "dyn_temp"
    }

    fn apply(&self, logits: &mut Tensor<f32>, _context: &SamplerContext) {
        *logits = apply_dynamic_temperature(logits, &self.config);
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

/// Top-K sampler
#[derive(Debug, Clone)]
pub struct TopKSampler {
    /// Number of top tokens to consider
    pub k: usize,
}

impl TopKSampler {
    /// Create a new top-k sampler
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl Sampler for TopKSampler {
    fn name(&self) -> &'static str {
        "top_k"
    }

    fn apply(&self, logits: &mut Tensor<f32>, _context: &SamplerContext) {
        // Apply top-k by zeroing out tokens outside top-k
        let data = logits.data();
        let mut indexed: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut new_data = vec![f32::NEG_INFINITY; data.len()];
        for (idx, logit) in indexed.iter().take(self.k) {
            new_data[*idx] = *logit;
        }

        if let Ok(result) = Tensor::from_vec(logits.shape().to_vec(), new_data) {
            *logits = result;
        }
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

/// Top-P (nucleus) sampler
#[derive(Debug, Clone)]
pub struct TopPSampler {
    /// Cumulative probability threshold (0.0 to 1.0)
    pub p: f32,
}

impl TopPSampler {
    /// Create a new top-p sampler
    pub fn new(p: f32) -> Self {
        Self { p }
    }
}

impl Sampler for TopPSampler {
    fn name(&self) -> &'static str {
        "top_p"
    }

    fn apply(&self, logits: &mut Tensor<f32>, _context: &SamplerContext) {
        let data = logits.data();

        // Calculate softmax
        let max_logit = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = data.iter().map(|x| (x - max_logit).exp()).sum();
        let mut indexed: Vec<(usize, f32, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, &logit)| (i, logit, (logit - max_logit).exp() / exp_sum))
            .collect();

        // Sort by probability descending
        indexed.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Find cutoff
        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed.len();
        for (i, (_, _, prob)) in indexed.iter().enumerate() {
            cumsum += prob;
            if cumsum >= self.p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out tokens below cutoff
        let mut new_data = vec![f32::NEG_INFINITY; data.len()];
        for (idx, logit, _) in indexed.iter().take(cutoff_idx) {
            new_data[*idx] = *logit;
        }

        if let Ok(result) = Tensor::from_vec(logits.shape().to_vec(), new_data) {
            *logits = result;
        }
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

/// Repetition penalty sampler
#[derive(Debug, Clone)]
pub struct RepetitionPenaltySampler {
    /// Repetition penalty configuration
    pub config: RepetitionPenaltyConfig,
}

impl RepetitionPenaltySampler {
    /// Create a new repetition penalty sampler
    pub fn new(config: RepetitionPenaltyConfig) -> Self {
        Self { config }
    }
}

impl Sampler for RepetitionPenaltySampler {
    fn name(&self) -> &'static str {
        "repetition_penalty"
    }

    fn apply(&self, logits: &mut Tensor<f32>, context: &SamplerContext) {
        *logits = apply_repetition_penalty(logits, &context.tokens, &self.config);
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

/// Infill sampler
#[derive(Debug, Clone)]
pub struct InfillSampler {
    /// Infill/FIM configuration
    pub config: InfillConfig,
}

impl InfillSampler {
    /// Create a new infill sampler
    pub fn new(config: InfillConfig) -> Self {
        Self { config }
    }
}

impl Sampler for InfillSampler {
    fn name(&self) -> &'static str {
        "infill"
    }

    fn apply(&self, logits: &mut Tensor<f32>, _context: &SamplerContext) {
        let result = apply_infill_sampling(logits, &self.config);
        *logits = result.logits;
    }

    fn clone_box(&self) -> Box<dyn Sampler> {
        Box::new(self.clone())
    }
}

// =============================================================================
// LogitProcessor Trait (RLZR-GEN-001)
// =============================================================================
//
// Composable logit processing for text generation pipelines.
// Based on HuggingFace Transformers LogitsProcessor pattern.
//
// References:
// - Holtzman et al. (2020) "The Curious Case of Neural Text Degeneration"
// - Wolf et al. (2020) "Transformers: State-of-the-Art NLP"
// =============================================================================

/// Context available during logit processing
///
/// Provides information about the current generation state to processors.
#[derive(Debug, Clone)]
pub struct LogitProcessorContext<'a> {
    /// Previously generated tokens (including initial prompt)
    pub tokens: &'a [u32],
    /// Current generation step (0-indexed, after initial tokens)
    pub step: usize,
    /// Vocabulary size
    pub n_vocab: usize,
}

impl<'a> LogitProcessorContext<'a> {
    /// Create a new context
    #[must_use]
    pub fn new(tokens: &'a [u32], step: usize, n_vocab: usize) -> Self {
        Self {
            tokens,
            step,
            n_vocab,
        }
    }
}

/// Logit processor trait for composable pre-sampling transforms
///
/// Processors are applied in order before sampling. They can:
/// - Set logits to -inf to suppress tokens
/// - Add penalties (repetition, length)
/// - Scale logits (temperature)
///
/// # Example
///
/// ```rust,ignore
/// use realizar::generate::{LogitProcessor, LogitProcessorContext};
///
/// struct MyProcessor;
///
/// impl LogitProcessor for MyProcessor {
///     fn process(&self, logits: &mut [f32], ctx: &LogitProcessorContext) {
///         // Suppress token 0
///         logits[0] = f32::NEG_INFINITY;
///     }
/// }
/// ```
pub trait LogitProcessor: Send + Sync {
    /// Process logits in-place before sampling
    ///
    /// # Arguments
    ///
    /// * `logits` - Mutable slice of logits to modify
    /// * `ctx` - Context with token history and generation state
    fn process(&self, logits: &mut [f32], ctx: &LogitProcessorContext);

    /// Human-readable name for debugging and tracing
    fn name(&self) -> &'static str {
        "unnamed"
    }
}

/// Suppress specific tokens by setting their logits to -inf
///
/// Use this to prevent certain tokens from being generated, such as:
/// - Special tokens (SOT, PREV, SOLM in Whisper)
/// - Profanity or sensitive content
/// - Invalid tokens for the current context
#[derive(Debug, Clone)]
pub struct TokenSuppressor {
    /// Token IDs to suppress
    suppress_ids: Vec<u32>,
}

impl TokenSuppressor {
    /// Create a new token suppressor
    ///
    /// # Arguments
    ///
    /// * `suppress_ids` - Token IDs to suppress (set to -inf)
    #[must_use]
    pub fn new(suppress_ids: Vec<u32>) -> Self {
        Self { suppress_ids }
    }

    /// Create from a slice of token IDs
    #[must_use]
    pub fn from_slice(suppress_ids: &[u32]) -> Self {
        Self {
            suppress_ids: suppress_ids.to_vec(),
        }
    }
}

impl LogitProcessor for TokenSuppressor {
    fn process(&self, logits: &mut [f32], _ctx: &LogitProcessorContext) {
        for &token_id in &self.suppress_ids {
            if (token_id as usize) < logits.len() {
                logits[token_id as usize] = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &'static str {
        "token_suppressor"
    }
}

/// Penalize repeated tokens to reduce repetitive generation
///
/// Applies a penalty to tokens that have appeared in the recent context.
/// Penalty > 1.0 reduces probability, < 1.0 increases it.
///
/// Based on: Keskar et al. (2019) "CTRL: A Conditional Transformer Language Model"
#[derive(Debug, Clone)]
pub struct RepetitionPenalty {
    /// Penalty multiplier (> 1.0 to penalize, < 1.0 to encourage)
    penalty: f32,
    /// Look-back window size (0 = entire history)
    window: usize,
}

impl RepetitionPenalty {
    /// Create a new repetition penalty processor
    ///
    /// # Arguments
    ///
    /// * `penalty` - Penalty multiplier (typical: 1.0-2.0)
    /// * `window` - Look-back window (0 = use all tokens)
    #[must_use]
    pub fn new(penalty: f32, window: usize) -> Self {
        Self { penalty, window }
    }

    /// Create with default window (entire history)
    #[must_use]
    pub fn with_penalty(penalty: f32) -> Self {
        Self { penalty, window: 0 }
    }
}

impl LogitProcessor for RepetitionPenalty {
    fn process(&self, logits: &mut [f32], ctx: &LogitProcessorContext) {
        // Determine which tokens to consider
        let tokens = if self.window > 0 && ctx.tokens.len() > self.window {
            &ctx.tokens[ctx.tokens.len() - self.window..]
        } else {
            ctx.tokens
        };

        // Apply penalty to tokens that have appeared
        for &token_id in tokens {
            if (token_id as usize) < logits.len() {
                let logit = logits[token_id as usize];
                // Apply penalty: divide positive logits, multiply negative logits
                logits[token_id as usize] = if logit > 0.0 {
                    logit / self.penalty
                } else {
                    logit * self.penalty
                };
            }
        }
    }

    fn name(&self) -> &'static str {
        "repetition_penalty"
    }
}

/// Scale logits by temperature
///
/// Temperature > 1.0 increases randomness (flatter distribution)
/// Temperature < 1.0 decreases randomness (sharper distribution)
/// Temperature = 1.0 has no effect
#[derive(Debug, Clone)]
pub struct TemperatureScaler {
    /// Temperature value (must be > 0)
    temperature: f32,
}

impl TemperatureScaler {
    /// Create a new temperature scaler
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature value (> 0)
    ///
    /// # Panics
    ///
    /// Panics if temperature <= 0
    #[must_use]
    pub fn new(temperature: f32) -> Self {
        assert!(temperature > 0.0, "Temperature must be positive");
        Self { temperature }
    }
}

impl LogitProcessor for TemperatureScaler {
    fn process(&self, logits: &mut [f32], _ctx: &LogitProcessorContext) {
        if (self.temperature - 1.0).abs() > 1e-6 {
            for logit in logits.iter_mut() {
                *logit /= self.temperature;
            }
        }
    }

    fn name(&self) -> &'static str {
        "temperature_scaler"
    }
}

/// Chain of logit processors applied in order
///
/// Allows composing multiple processors into a single processing step.
#[derive(Default)]
pub struct LogitProcessorChain {
    processors: Vec<Box<dyn LogitProcessor>>,
}

impl LogitProcessorChain {
    /// Create an empty processor chain
    #[must_use]
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    /// Add a processor to the chain (builder pattern)
    #[must_use]
    pub fn with_processor<P: LogitProcessor + 'static>(mut self, processor: P) -> Self {
        self.processors.push(Box::new(processor));
        self
    }

    /// Add a boxed processor to the chain (builder pattern)
    #[must_use]
    pub fn with_boxed_processor(mut self, processor: Box<dyn LogitProcessor>) -> Self {
        self.processors.push(processor);
        self
    }

    /// Process logits through all processors in order
    pub fn process(&self, logits: &mut [f32], ctx: &LogitProcessorContext) {
        for processor in &self.processors {
            processor.process(logits, ctx);
        }
    }

    /// Get the number of processors in the chain
    #[must_use]
    pub fn len(&self) -> usize {
        self.processors.len()
    }

    /// Check if the chain is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }

    /// Get processor names for debugging
    #[must_use]
    pub fn processor_names(&self) -> Vec<&str> {
        self.processors.iter().map(|p| p.name()).collect()
    }
}

impl LogitProcessor for LogitProcessorChain {
    fn process(&self, logits: &mut [f32], ctx: &LogitProcessorContext) {
        LogitProcessorChain::process(self, logits, ctx);
    }

    fn name(&self) -> &'static str {
        "processor_chain"
    }
}

/// Model trait for generation pipeline
///
/// Implement this trait to use your model with GenerationPipeline.
pub trait GenerativeModel {
    /// Forward pass producing logits for next token
    ///
    /// # Arguments
    ///
    /// * `tokens` - Current token sequence
    ///
    /// # Returns
    ///
    /// Logits for vocabulary (shape: [vocab_size])
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Reset any cached state (e.g., KV cache)
    fn reset(&mut self) {}
}

/// Generation pipeline with processor chain
///
/// Orchestrates the generation loop with:
/// 1. Model forward pass
/// 2. Logit processing
/// 3. Token sampling
/// 4. EOS detection
///
/// # Example
///
/// ```rust,ignore
/// use realizar::generate::{GenerationPipeline, TokenSuppressor, GenerationConfig};
///
/// let pipeline = GenerationPipeline::new(model)
///     .add_processor(TokenSuppressor::new(vec![0, 1, 2]))
///     .with_config(GenerationConfig::greedy().with_eos_token_id(50256));
///
/// let tokens = pipeline.generate(&[1, 2, 3])?;
/// ```
pub struct GenerationPipeline<M: GenerativeModel> {
    model: M,
    processors: LogitProcessorChain,
    config: GenerationConfig,
}

impl<M: GenerativeModel> GenerationPipeline<M> {
    /// Create a new generation pipeline
    #[must_use]
    pub fn new(model: M) -> Self {
        Self {
            model,
            processors: LogitProcessorChain::new(),
            config: GenerationConfig::default(),
        }
    }

    /// Add a logit processor to the pipeline
    #[must_use]
    pub fn add_processor<P: LogitProcessor + 'static>(mut self, processor: P) -> Self {
        self.processors = self.processors.with_processor(processor);
        self
    }

    /// Set generation configuration
    #[must_use]
    pub fn with_config(mut self, config: GenerationConfig) -> Self {
        self.config = config;
        self
    }

    /// Generate tokens starting from initial sequence
    ///
    /// # Arguments
    ///
    /// * `initial_tokens` - Starting token sequence (prompt)
    ///
    /// # Returns
    ///
    /// Generated token sequence (including initial tokens)
    pub fn generate(&mut self, initial_tokens: &[u32]) -> Result<Vec<u32>> {
        let mut tokens = initial_tokens.to_vec();
        let n_vocab = self.model.vocab_size();
        let eos_token = self.config.eos_token_id;

        // Simple PRNG for sampling (deterministic with seed)
        let mut rng_state = self.config.seed.unwrap_or(42);

        for step in 0..self.config.max_tokens {
            // Forward pass
            let mut logits = self.model.forward(&tokens)?;

            // Apply logit processors
            let ctx = LogitProcessorContext::new(&tokens, step, n_vocab);
            self.processors.process(&mut logits, &ctx);

            // Sample next token
            let logits_tensor = Tensor::from_vec(vec![logits.len()], logits)?;

            // Simple LCG for RNG
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let rng_value = (rng_state >> 33) as f32 / (1u64 << 31) as f32;

            let next_token = sample_token(&logits_tensor, &self.config, rng_value)? as u32;

            tokens.push(next_token);

            // Check for EOS
            if let Some(eos) = eos_token {
                if next_token == eos as u32 {
                    break;
                }
            }
        }

        Ok(tokens)
    }

    /// Get reference to the model
    #[must_use]
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get mutable reference to the model
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Get reference to the processor chain
    #[must_use]
    pub fn processors(&self) -> &LogitProcessorChain {
        &self.processors
    }

    /// Get reference to the config
    #[must_use]
    pub fn config(&self) -> &GenerationConfig {
        &self.config
    }
}

#[cfg(all(test, feature = "heavy-tests"))]
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
        let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");

        // Temperature = 1.0 should return same values
        let scaled = apply_temperature(&logits, 1.0).expect("test");
        for i in 0..4 {
            assert!((scaled.data()[i] - logits.data()[i]).abs() < 1e-6);
        }

        // Temperature = 2.0 should halve values
        let scaled = apply_temperature(&logits, 2.0).expect("test");
        assert!((scaled.data()[0] - 0.5).abs() < 1e-6);
        assert!((scaled.data()[3] - 2.0).abs() < 1e-6);

        // Temperature = 0.5 should double values
        let scaled = apply_temperature(&logits, 0.5).expect("test");
        assert!((scaled.data()[0] - 2.0).abs() < 1e-6);
        assert!((scaled.data()[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_temperature_invalid() {
        let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        assert!(apply_temperature(&logits, 0.0).is_err());
        assert!(apply_temperature(&logits, -1.0).is_err());
    }

    #[test]
    fn test_sample_greedy() {
        // Clear winner at index 2
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 10.0, 3.0, 4.0]).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 2);

        // Winner at last index
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 5.0]).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 2);

        // Winner at first index
        let logits = Tensor::from_vec(vec![3], vec![5.0, 2.0, 1.0]).expect("test");
        let token = sample_greedy(&logits).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_greedy_empty_error() {
        let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
        // Single element should work
        assert_eq!(sample_greedy(&logits).expect("test"), 0);
    }

    #[test]
    fn test_sample_top_k() {
        // Strong preference for index 0
        let logits = Tensor::from_vec(vec![5], vec![100.0, 1.0, 1.0, 1.0, 1.0]).expect("test");

        // With rng_value = 0.0, should always get first (highest prob)
        let token = sample_top_k(&logits, 3, 0.0).expect("test");
        assert_eq!(token, 0);

        // With k=1, should always get highest
        let token = sample_top_k(&logits, 1, 0.5).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_top_k_distribution() {
        // Two equally likely tokens
        let logits = Tensor::from_vec(vec![4], vec![10.0, 10.0, 0.0, 0.0]).expect("test");

        // Low rng should get index 0 or 1 (they're equal)
        let token = sample_top_k(&logits, 2, 0.1).expect("test");
        assert!(token == 0 || token == 1);

        // High rng should get index 0 or 1
        let token = sample_top_k(&logits, 2, 0.9).expect("test");
        assert!(token == 0 || token == 1);
    }

    #[test]
    fn test_sample_top_k_errors() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        assert!(sample_top_k(&logits, 0, 0.5).is_err());
    }

    #[test]
    fn test_sample_top_p() {
        // One dominant token
        let logits = Tensor::from_vec(vec![3], vec![100.0, 1.0, 1.0]).expect("test");

        // With p=0.9, nucleus likely just the first token
        let token = sample_top_p(&logits, 0.9, 0.5).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_top_p_uniform() {
        // Equal logits
        let logits = Tensor::from_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).expect("test");

        // With p=1.0, all tokens in nucleus
        // Low rng should get early token
        let token = sample_top_p(&logits, 1.0, 0.1).expect("test");
        assert!(token < 4);

        // High rng should get later token
        let token = sample_top_p(&logits, 1.0, 0.9).expect("test");
        assert!(token < 4);
    }

    #[test]
    fn test_sample_top_p_errors() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        assert!(sample_top_p(&logits, 0.0, 0.5).is_err());
        assert!(sample_top_p(&logits, 1.1, 0.5).is_err());
        assert!(sample_top_p(&logits, -0.1, 0.5).is_err());
    }

    #[test]
    fn test_sample_token_greedy() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 10.0, 3.0, 4.0]).expect("test");
        let config = GenerationConfig::greedy();
        let token = sample_token(&logits, &config, 0.5).expect("test");
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_token_with_temperature() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let config = GenerationConfig::greedy().with_temperature(0.5);
        let token = sample_token(&logits, &config, 0.5).expect("test");
        // Higher temperature doesn't change greedy selection
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_token_top_k() {
        let logits = Tensor::from_vec(vec![5], vec![100.0, 1.0, 1.0, 1.0, 1.0]).expect("test");
        let config = GenerationConfig::top_k(3);
        let token = sample_token(&logits, &config, 0.0).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_token_top_p() {
        let logits = Tensor::from_vec(vec![3], vec![100.0, 1.0, 1.0]).expect("test");
        let config = GenerationConfig::top_p(0.95);
        let token = sample_token(&logits, &config, 0.5).expect("test");
        assert_eq!(token, 0);
    }

    // =====================================================================
    // Advanced Sampling Feature Tests
    // =====================================================================

    // ----- Stop Sequence Detector Tests -----

    #[test]
    fn test_stop_sequence_detector_new() {
        let detector = StopSequenceDetector::new();
        assert!(!detector.has_conditions());
    }

    #[test]
    fn test_stop_sequence_detector_add_token_sequence() {
        let detector = StopSequenceDetector::new()
            .with_token_sequence(vec![1, 2, 3])
            .with_token_sequence(vec![4, 5]);
        assert!(detector.has_conditions());
    }

    #[test]
    fn test_stop_sequence_detector_add_string_pattern() {
        let detector = StopSequenceDetector::new()
            .with_string_pattern("<|end|>")
            .with_string_pattern("\n\n");
        assert!(detector.has_conditions());
    }

    #[test]
    fn test_stop_sequence_detector_token_match() {
        let mut detector = StopSequenceDetector::new().with_token_sequence(vec![10, 20, 30]);

        // Add tokens one by one
        assert!(!detector.check_token(10)); // Partial match
        assert!(!detector.check_token(20)); // Still partial
        assert!(detector.check_token(30)); // Complete match!
    }

    #[test]
    fn test_stop_sequence_detector_token_no_match() {
        let mut detector = StopSequenceDetector::new().with_token_sequence(vec![10, 20, 30]);

        detector.check_token(10);
        detector.check_token(25); // Wrong token breaks sequence
        assert!(!detector.check_token(30)); // 30 alone doesn't match
    }

    #[test]
    fn test_stop_sequence_detector_string_match() {
        let detector = StopSequenceDetector::new().with_string_pattern("<|end|>");

        assert!(detector.check_text("Hello world").is_none());
        assert!(detector.check_text("Output: <|end|>").is_some());
        assert!(detector.check_text("<|end|> extra").is_some());
    }

    #[test]
    fn test_stop_sequence_detector_buffer_limit() {
        let mut detector = StopSequenceDetector::new().with_token_sequence(vec![1, 2]); // max_seq_len = 2

        // Add many tokens
        for i in 0..100 {
            detector.check_token(i);
        }

        // Detector should still work (buffer is trimmed internally)
        assert!(detector.has_conditions());
    }

    #[test]
    fn test_stop_sequence_detector_reset() {
        let mut detector = StopSequenceDetector::new().with_token_sequence(vec![1, 2, 3]);

        detector.check_token(1);
        detector.check_token(2);
        detector.reset();

        // After reset, need to match sequence again from start
        assert!(!detector.check_token(3)); // Just 3 alone won't match
    }

    // ----- Repetition Penalty Tests -----

    #[test]
    fn test_repetition_penalty_config_default() {
        let config = RepetitionPenaltyConfig::default();
        assert_eq!(config.penalty, 1.0); // No penalty by default
        assert_eq!(config.window_size, 64);
    }

    #[test]
    fn test_repetition_penalty_config_builder() {
        let config = RepetitionPenaltyConfig::new(1.5).with_window(128);
        assert_eq!(config.penalty, 1.5);
        assert_eq!(config.window_size, 128);
    }

    #[test]
    fn test_apply_repetition_penalty_basic() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 3.0, 0.5, -1.0]).expect("test");
        let context = vec![0, 2, 4]; // Penalize tokens 0, 2, 4
        let config = RepetitionPenaltyConfig::new(2.0);

        let result = apply_repetition_penalty(&logits, &context, &config);

        // Positive logits should be divided by penalty
        assert_eq!(result.data()[0], 1.0); // 2.0 / 2.0
        assert_eq!(result.data()[1], 1.0); // Unchanged (not in context)
        assert_eq!(result.data()[2], 1.5); // 3.0 / 2.0

        // Negative logits should be multiplied by penalty
        assert_eq!(result.data()[4], -2.0); // -1.0 * 2.0
    }

    #[test]
    fn test_apply_repetition_penalty_window() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 2.0, 2.0, 2.0, 2.0]).expect("test");
        let context = vec![0, 1, 2, 3, 4]; // All tokens in context
        let config = RepetitionPenaltyConfig::new(2.0).with_window(2); // Only last 2 tokens

        let result = apply_repetition_penalty(&logits, &context, &config);

        // Only tokens 3, 4 should be penalized (last 2 in window)
        assert_eq!(result.data()[0], 2.0); // Unchanged
        assert_eq!(result.data()[1], 2.0); // Unchanged
        assert_eq!(result.data()[2], 2.0); // Unchanged
        assert_eq!(result.data()[3], 1.0); // 2.0 / 2.0
        assert_eq!(result.data()[4], 1.0); // 2.0 / 2.0
    }

    #[test]
    fn test_apply_repetition_penalty_no_penalty() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let context = vec![0, 1, 2];
        let config = RepetitionPenaltyConfig::new(1.0); // No penalty

        let result = apply_repetition_penalty(&logits, &context, &config);

        // No change when penalty is 1.0
        assert_eq!(result.data()[0], 1.0);
        assert_eq!(result.data()[1], 2.0);
        assert_eq!(result.data()[2], 3.0);
    }

    #[test]
    fn test_repetition_penalty_is_enabled() {
        let disabled = RepetitionPenaltyConfig::new(1.0);
        assert!(!disabled.is_enabled());

        let enabled = RepetitionPenaltyConfig::new(1.1);
        assert!(enabled.is_enabled());
    }

    // ----- Presence/Frequency Penalty Tests -----

    #[test]
    fn test_presence_frequency_penalty_default() {
        let config = PresenceFrequencyPenalty::default();
        assert_eq!(config.presence_penalty, 0.0);
        assert_eq!(config.frequency_penalty, 0.0);
    }

    #[test]
    fn test_presence_frequency_penalty_new() {
        let config = PresenceFrequencyPenalty::new(0.5, 0.3);
        assert_eq!(config.presence_penalty, 0.5);
        assert_eq!(config.frequency_penalty, 0.3);
    }

    #[test]
    fn test_apply_presence_penalty() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 10.0, 10.0, 10.0, 10.0]).expect("test");
        let context = vec![0, 0, 1]; // Token 0 appears twice, token 1 once
        let config = PresenceFrequencyPenalty::new(1.0, 0.0);

        let result = apply_presence_frequency_penalty(&logits, &context, &config);

        // Tokens 0 and 1 get presence penalty (constant)
        assert_eq!(result.data()[0], 9.0); // 10.0 - 1.0
        assert_eq!(result.data()[1], 9.0); // 10.0 - 1.0
        assert_eq!(result.data()[2], 10.0); // Unchanged
    }

    #[test]
    fn test_apply_frequency_penalty() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 10.0, 10.0, 10.0, 10.0]).expect("test");
        let context = vec![0, 0, 0, 1]; // Token 0 appears 3x, token 1 once
        let config = PresenceFrequencyPenalty::new(0.0, 1.0);

        let result = apply_presence_frequency_penalty(&logits, &context, &config);

        // Frequency penalty is proportional to count
        assert_eq!(result.data()[0], 7.0); // 10.0 - 3*1.0
        assert_eq!(result.data()[1], 9.0); // 10.0 - 1*1.0
        assert_eq!(result.data()[2], 10.0); // Unchanged
    }

    #[test]
    fn test_apply_combined_penalties() {
        let logits = Tensor::from_vec(vec![3], vec![10.0, 10.0, 10.0]).expect("test");
        let context = vec![0, 0, 1]; // Token 0 appears 2x, token 1 once
        let config = PresenceFrequencyPenalty::new(0.5, 0.5);

        let result = apply_presence_frequency_penalty(&logits, &context, &config);

        // Token 0: 10.0 - 0.5(presence) - 2*0.5(freq) = 10.0 - 1.5 = 8.5
        assert_eq!(result.data()[0], 8.5);
        // Token 1: 10.0 - 0.5(presence) - 1*0.5(freq) = 10.0 - 1.0 = 9.0
        assert_eq!(result.data()[1], 9.0);
    }

    #[test]
    fn test_presence_frequency_is_enabled() {
        let disabled = PresenceFrequencyPenalty::new(0.0, 0.0);
        assert!(!disabled.is_enabled());

        let enabled = PresenceFrequencyPenalty::new(0.1, 0.0);
        assert!(enabled.is_enabled());
    }

    // ----- Logit Bias Tests -----

    #[test]
    fn test_logit_bias_default() {
        let bias = LogitBias::default();
        assert!(bias.is_empty());
    }

    #[test]
    fn test_logit_bias_add() {
        let bias = LogitBias::new().with_bias(10, 5.0).with_bias(20, -100.0);
        assert!(!bias.is_empty());
        assert_eq!(bias.get(10), 5.0);
        assert_eq!(bias.get(20), -100.0);
        assert_eq!(bias.get(30), 0.0); // Not set
    }

    #[test]
    fn test_apply_logit_bias() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let bias = LogitBias::new()
            .with_bias(0, 10.0)
            .with_bias(2, -100.0)
            .with_bias(4, 3.0);

        let result = apply_logit_bias(&logits, &bias);

        assert_eq!(result.data()[0], 11.0); // 1.0 + 10.0
        assert_eq!(result.data()[1], 2.0); // Unchanged
        assert_eq!(result.data()[2], -97.0); // 3.0 - 100.0
        assert_eq!(result.data()[3], 4.0); // Unchanged
        assert_eq!(result.data()[4], 8.0); // 5.0 + 3.0
    }

    #[test]
    fn test_apply_logit_bias_out_of_range() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let bias = LogitBias::new().with_bias(100, 50.0); // Index out of range

        let result = apply_logit_bias(&logits, &bias);

        // Should not panic, just skip out-of-range indices
        assert_eq!(result.data()[0], 1.0);
        assert_eq!(result.data()[1], 2.0);
        assert_eq!(result.data()[2], 3.0);
    }

    // ----- Min-P Sampling Tests -----

    #[test]
    fn test_sample_min_p_basic() {
        // Token 0 has probability ~0.7, token 1 ~0.2, token 2 ~0.1
        let logits = Tensor::from_vec(vec![3], vec![1.0, -0.5, -1.0]).expect("test");

        // With min_p = 0.3 (30% of max), only token 0 should remain
        let token = sample_min_p(&logits, 0.3, 0.5).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_min_p_all_pass() {
        // All tokens have similar logits
        let logits = Tensor::from_vec(vec![3], vec![0.0, 0.0, 0.0]).expect("test");

        // With min_p = 0.9, all tokens should pass (all equal)
        let token = sample_min_p(&logits, 0.9, 0.3).expect("test");
        assert!(token < 3);
    }

    #[test]
    fn test_sample_min_p_low_threshold() {
        let logits = Tensor::from_vec(vec![4], vec![10.0, 1.0, 0.5, 0.1]).expect("test");

        // With very low min_p, all tokens can be sampled
        let token = sample_min_p(&logits, 0.001, 0.99).expect("test");
        assert!(token < 4);
    }

    #[test]
    fn test_sample_min_p_edge_cases() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");

        // min_p = 0 should include all tokens
        let _ = sample_min_p(&logits, 0.0, 0.5).expect("test");

        // min_p = 1.0 should still return something (at least the max)
        let token = sample_min_p(&logits, 1.0, 0.5).expect("test");
        assert_eq!(token, 2); // Highest probability token
    }

    #[test]
    fn test_sample_min_p_rng_boundary() {
        // Test with rng_value at boundary (0.0)
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let token = sample_min_p(&logits, 0.5, 0.0).expect("test");
        assert!(token < 3);
    }

    // ----- Mirostat Sampling Tests -----

    #[test]
    fn test_mirostat_state_default() {
        let state = MirostatState::default();
        assert_eq!(state.tau, 5.0);
        assert_eq!(state.eta, 0.1);
        assert_eq!(state.mu, 10.0);
    }

    #[test]
    fn test_mirostat_state_builder() {
        let state = MirostatState::new(3.0).with_eta(0.2);
        assert_eq!(state.tau, 3.0);
        assert_eq!(state.eta, 0.2);
        assert_eq!(state.mu, 6.0); // 2 * tau
    }

    #[test]
    fn test_mirostat_state_update() {
        let mut state = MirostatState::new(5.0).with_eta(0.1);

        let initial_mu = state.mu;

        // High surprise should decrease mu (mu -= eta * (surprise - tau))
        state.update(10.0); // surprise > tau => mu decreases
        assert!(state.mu < initial_mu);

        // Reset
        state.mu = initial_mu;

        // Low surprise should increase mu
        state.update(2.0); // surprise < tau => mu increases
        assert!(state.mu > initial_mu);
    }

    #[test]
    fn test_sample_mirostat_basic() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 5.0, 1.0, 0.0, -5.0]).expect("test");
        let mut state = MirostatState::default();

        let token = sample_mirostat(&logits, &mut state, 0.5).expect("test");
        assert!(token < 5);
    }

    #[test]
    fn test_sample_mirostat_deterministic() {
        let logits = Tensor::from_vec(vec![3], vec![100.0, 1.0, 1.0]).expect("test");
        let mut state = MirostatState::new(0.1); // Low target perplexity

        // With very low tau, should prefer highest probability token
        let token = sample_mirostat(&logits, &mut state, 0.0).expect("test");
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_mirostat_state_evolution() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 5.0, 1.0, 0.0, -5.0]).expect("test");
        let mut state = MirostatState::default();

        let initial_mu = state.mu;

        // Sample multiple times and verify mu evolves
        for _ in 0..10 {
            let _ = sample_mirostat(&logits, &mut state, 0.5).expect("test");
        }

        // Mu should have changed from initial
        assert_ne!(state.mu, initial_mu);
    }

    #[test]
    fn test_sample_mirostat_rng_boundary() {
        // Test with rng_value at boundary (1.0 - epsilon)
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let mut state = MirostatState::default();
        let token = sample_mirostat(&logits, &mut state, 0.999).expect("test");
        assert!(token < 3);
    }

    // ----- Advanced Generation Config Tests -----

    #[test]
    fn test_advanced_generation_config_default() {
        let config = AdvancedGenerationConfig::default();
        assert!(config.stop_detector.is_none());
        assert!(config.repetition_penalty.is_none());
        assert!(config.presence_frequency.is_none());
        assert!(config.logit_bias.is_none());
    }

    #[test]
    fn test_advanced_generation_config_builder() {
        let config = AdvancedGenerationConfig::new(GenerationConfig::greedy())
            .with_stop_sequences(vec!["<|end|>".to_string()])
            .with_repetition_penalty(1.5)
            .with_presence_frequency(0.5, 0.3)
            .with_logit_bias(LogitBias::new().with_bias(0, 10.0));

        assert!(config.stop_detector.is_some());
        assert!(config.repetition_penalty.is_some());
        assert!(config.presence_frequency.is_some());
        assert!(config.logit_bias.is_some());
    }

    // ----- Apply All Penalties Tests -----

    #[test]
    fn test_apply_all_penalties_empty() {
        let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let original = logits.data().to_vec();
        let context: Vec<usize> = vec![];
        let config = AdvancedGenerationConfig::default();

        let result = apply_all_penalties(&logits, &context, &config);

        // No penalties applied
        assert_eq!(result.data(), original.as_slice());
    }

    #[test]
    fn test_apply_all_penalties_combined() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 10.0, 10.0, 10.0, 10.0]).expect("test");
        let context = vec![0, 0, 1]; // Token 0 twice, token 1 once

        let config = AdvancedGenerationConfig::new(GenerationConfig::greedy())
            .with_repetition_penalty(2.0)
            .with_presence_frequency(1.0, 0.5)
            .with_logit_bias(LogitBias::new().with_bias(4, 100.0));

        let result = apply_all_penalties(&logits, &context, &config);

        // Token 4 should be highest due to bias
        let max_idx = result
            .data()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("test")
            .0;
        assert_eq!(max_idx, 4);

        // Token 0 should be penalized most (repetition + presence + frequency)
        assert!(result.data()[0] < result.data()[2]);
    }

    #[test]
    fn test_stop_sequence_with_stop_strings() {
        let detector = StopSequenceDetector::new()
            .with_stop_strings(vec!["stop".to_string(), "end".to_string()]);

        assert!(detector.check_text("this has stop in it").is_some());
        assert!(detector.check_text("the end").is_some());
        assert!(detector.check_text("nothing here").is_none());
    }

    // ===== Tail-Free Sampling (TFS) Tests =====

    #[test]
    fn test_tfs_basic_filtering() {
        // Create logits with distinct probabilities
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");

        // With z=0.95, should filter some low-probability tokens
        let result = sample_tfs(&logits, 0.95, 0.0);
        assert!(result.is_ok());
        // Should return one of the high-probability tokens
        assert!(result.expect("test") < 5);
    }

    #[test]
    fn test_tfs_z_one_returns_greedy() {
        // z=1.0 should keep all tokens (no filtering)
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");

        // rng=0.0 should select the first valid token after filtering
        let result = sample_tfs(&logits, 1.0, 0.0).expect("test");
        // Should be a valid token
        assert!(result < 5);
    }

    #[test]
    fn test_tfs_z_zero_selects_top() {
        // z=0.0 should filter aggressively, keeping only top tokens
        let logits = Tensor::from_vec(vec![5], vec![10.0, 1.0, 0.5, 0.1, -1.0]).expect("test");

        let result = sample_tfs(&logits, 0.01, 0.0).expect("test");
        // Should select from top tokens
        assert!(result < 3);
    }

    #[test]
    fn test_tfs_single_token() {
        let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
        let result = sample_tfs(&logits, 0.95, 0.5).expect("test");
        assert_eq!(result, 0);
    }

    #[test]
    fn test_tfs_uniform_distribution() {
        // With uniform logits, all tokens have equal second derivative
        let logits = Tensor::from_vec(vec![5], vec![1.0, 1.0, 1.0, 1.0, 1.0]).expect("test");
        let result = sample_tfs(&logits, 0.95, 0.5).expect("test");
        assert!(result < 5);
    }

    #[test]
    fn test_tfs_two_tokens() {
        // Test with minimum viable token count
        let logits = Tensor::from_vec(vec![2], vec![1.0, 0.5]).expect("test");
        let result = sample_tfs(&logits, 0.95, 0.5);
        assert!(result.is_ok());
        assert!(result.expect("test") < 2);
    }

    // ===== Locally Typical Sampling Tests =====

    #[test]
    fn test_typical_basic_sampling() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.5, 1.0, 0.5, 0.0]).expect("test");

        let result = sample_typical(&logits, 0.95, 0.5);
        assert!(result.is_ok());
        assert!(result.expect("test") < 5);
    }

    #[test]
    fn test_typical_p_one_keeps_all() {
        // p=1.0 should keep all tokens
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let result = sample_typical(&logits, 1.0, 0.5).expect("test");
        assert!(result < 5);
    }

    #[test]
    fn test_typical_low_p_selects_typical() {
        // Low p should select only the most typical tokens (closest to entropy)
        let logits = Tensor::from_vec(vec![5], vec![10.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let result = sample_typical(&logits, 0.1, 0.0).expect("test");
        // Should select a token
        assert!(result < 5);
    }

    #[test]
    fn test_typical_single_token() {
        let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
        let result = sample_typical(&logits, 0.95, 0.5).expect("test");
        assert_eq!(result, 0);
    }

    #[test]
    fn test_typical_uniform_distribution() {
        // Uniform distribution - all tokens equally typical
        let logits = Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).expect("test");
        let result = sample_typical(&logits, 0.95, 0.5).expect("test");
        assert!(result < 4);
    }

    #[test]
    fn test_typical_two_tokens() {
        // Test with minimum viable token count
        let logits = Tensor::from_vec(vec![2], vec![1.0, 0.5]).expect("test");
        let result = sample_typical(&logits, 0.95, 0.5);
        assert!(result.is_ok());
        assert!(result.expect("test") < 2);
    }

    // ===== DRY (Don't Repeat Yourself) Sampling Tests =====

    #[test]
    fn test_dry_config_default() {
        let config = DryConfig::default();
        assert_eq!(config.multiplier, 0.8);
        assert_eq!(config.base, 1.75);
        assert_eq!(config.allowed_length, 2);
        assert_eq!(config.penalty_last_n, 256);
        assert!(config.is_enabled()); // Default is enabled
    }

    #[test]
    fn test_dry_config_disabled() {
        let config = DryConfig::new(0.0);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_dry_config_enabled() {
        let config = DryConfig::new(0.5)
            .with_base(1.5)
            .with_allowed_length(3)
            .with_penalty_last_n(64);
        assert!(config.is_enabled());
        assert_eq!(config.base, 1.5);
        assert_eq!(config.allowed_length, 3);
        assert_eq!(config.penalty_last_n, 64);
    }

    #[test]
    fn test_dry_no_penalty_when_disabled() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = DryConfig::new(0.0); // disabled (multiplier=0)
        let context = vec![0, 1, 0, 1, 0];

        let result = apply_dry_penalty(&logits, &context, &config);
        assert_eq!(result.data(), logits.data());
    }

    #[test]
    fn test_dry_penalty_applied() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = DryConfig {
            multiplier: 1.0,
            base: 1.75,
            allowed_length: 2,
            penalty_last_n: 64,
        };
        // Context with repeated pattern: [0, 1, 0, 1] - if next is 0, it continues [0,1] pattern
        let context = vec![0, 1, 0, 1];

        let result = apply_dry_penalty(&logits, &context, &config);
        // Token 0 should be penalized (would create [0,1,0] repetition)
        assert!(result.data()[0] < logits.data()[0]);
    }

    #[test]
    fn test_dry_short_context_no_penalty() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = DryConfig {
            multiplier: 1.0,
            base: 1.75,
            allowed_length: 3,
            penalty_last_n: 64,
        };
        // Context shorter than allowed_length
        let context = vec![0, 1];

        let result = apply_dry_penalty(&logits, &context, &config);
        assert_eq!(result.data(), logits.data());
    }

    #[test]
    fn test_dry_respects_penalty_last_n() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = DryConfig {
            multiplier: 1.0,
            base: 1.75,
            allowed_length: 2,
            penalty_last_n: 3, // Only look at last 3 tokens
        };
        // Repetition is outside the window
        let context = vec![0, 1, 2, 3, 4];

        let result = apply_dry_penalty(&logits, &context, &config);
        // Should not detect repetition from early in context
        // (penalty window is only last 3: [2, 3, 4])
        assert!(result.data().iter().sum::<f32>() > 0.0);
    }

    // ===== Beam Search Tests =====

    #[test]
    fn test_beam_hypothesis_creation() {
        let hyp = BeamHypothesis::new(vec![1, 2, 3], -1.5);
        assert_eq!(hyp.tokens.len(), 3);
        assert!(!hyp.finished);
        assert_eq!(hyp.score, -1.5);
    }

    #[test]
    fn test_beam_hypothesis_extend() {
        let hyp = BeamHypothesis::new(vec![1, 2], -1.0);
        let extended = hyp.extend(3, -0.5, false);
        assert_eq!(extended.tokens, vec![1, 2, 3]);
        assert_eq!(extended.score, -1.5);
        assert!(!extended.finished);
    }

    #[test]
    fn test_beam_hypothesis_extend_with_eos() {
        let hyp = BeamHypothesis::new(vec![1, 2], -1.0);
        let extended = hyp.extend(99, -0.5, true);
        assert_eq!(extended.tokens, vec![1, 2, 99]);
        assert!(extended.finished);
    }

    #[test]
    fn test_beam_hypothesis_normalized_score() {
        let hyp = BeamHypothesis::new(vec![1, 2, 3, 4], -4.0);
        // length_penalty = 1.0 means divide by length
        assert_eq!(hyp.normalized_score(1.0), -1.0);
        // length_penalty = 0.0 means score / 1.0 = score
        assert_eq!(hyp.normalized_score(0.0), -4.0);
    }

    #[test]
    fn test_beam_search_config_default() {
        let config = BeamSearchConfig::default();
        assert_eq!(config.num_beams, 4);
        assert_eq!(config.length_penalty, 1.0);
        assert!(config.early_stopping); // Default is true
        assert_eq!(config.num_return, 1);
    }

    #[test]
    fn test_beam_search_config_new() {
        let config = BeamSearchConfig::new(8);
        assert_eq!(config.num_beams, 8);
        assert!(config.early_stopping);
    }

    #[test]
    fn test_beam_search_config_builder() {
        let config = BeamSearchConfig::new(4)
            .with_length_penalty(0.8)
            .with_early_stopping(false)
            .with_num_return(2);
        assert_eq!(config.num_beams, 4);
        assert_eq!(config.length_penalty, 0.8);
        assert!(!config.early_stopping);
        assert_eq!(config.num_return, 2);
    }

    #[test]
    fn test_beam_search_state_creation() {
        let config = BeamSearchConfig::new(3)
            .with_length_penalty(0.8)
            .with_num_return(2);
        let state = BeamSearchState::new(config, vec![1, 2, 3]);
        assert_eq!(state.hypotheses.len(), 1); // Starts with one hypothesis
        assert!(state.finished.is_empty());
        assert_eq!(state.hypotheses[0].tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_beam_search_state_step() {
        let config = BeamSearchConfig::new(2).with_early_stopping(false);
        let mut state = BeamSearchState::new(config, vec![0]);

        // Create log probabilities for 1 hypothesis, 5 tokens
        let log_probs = vec![vec![-0.1, -0.5, -1.0, -2.0, -3.0]];

        state.step(&log_probs, Some(4)); // EOS token is 4

        // Should have expanded to num_beams hypotheses
        assert!(!state.hypotheses.is_empty());
    }

    #[test]
    fn test_beam_search_state_with_finished() {
        let config = BeamSearchConfig::new(2);
        let mut state = BeamSearchState::new(config, vec![0]);

        // Manually add some hypotheses
        state.hypotheses = vec![
            BeamHypothesis::new(vec![1, 2], -1.0),
            BeamHypothesis::new(vec![1, 3], -2.0),
        ];
        state.finished = vec![BeamHypothesis {
            tokens: vec![1, 2, 4],
            score: -1.5,
            finished: true,
        }];

        assert_eq!(state.hypotheses.len(), 2);
        assert_eq!(state.finished.len(), 1);
    }

    #[test]
    fn test_beam_search_state_should_stop_empty() {
        let config = BeamSearchConfig::new(2).with_early_stopping(false);
        let mut state = BeamSearchState::new(config, vec![0]);
        state.hypotheses.clear();

        // Empty hypotheses means should stop
        assert!(state.should_stop());
    }

    #[test]
    fn test_beam_search_state_should_stop_early() {
        let config = BeamSearchConfig::new(2).with_early_stopping(true);
        let mut state = BeamSearchState::new(config, vec![0]);

        // Not done initially
        assert!(!state.should_stop());

        // Add num_beams finished hypotheses
        state.finished.push(BeamHypothesis {
            tokens: vec![1, 2, 4],
            score: -1.0,
            finished: true,
        });
        state.finished.push(BeamHypothesis {
            tokens: vec![1, 3, 4],
            score: -1.5,
            finished: true,
        });

        // Should be done with early_stopping=true and num_beams finished
        assert!(state.should_stop());
    }

    #[test]
    fn test_beam_search_state_all_finished() {
        let config = BeamSearchConfig::new(2).with_early_stopping(false);
        let mut state = BeamSearchState::new(config, vec![0]);
        state.hypotheses = vec![
            BeamHypothesis {
                tokens: vec![1],
                score: -1.0,
                finished: true,
            },
            BeamHypothesis {
                tokens: vec![2],
                score: -2.0,
                finished: true,
            },
        ];

        // All hypotheses finished
        assert!(state.should_stop());
    }

    // ===== Streaming Generation Tests =====

    #[test]
    fn test_streaming_generator_creation() {
        let generator = StreamingGenerator::new();
        assert!(generator.tokens.is_empty());
        assert!(generator.text.is_empty());
        assert!(!generator.finished);
        assert_eq!(generator.total_tokens, 0);
    }

    #[test]
    fn test_streaming_generator_default() {
        let generator = StreamingGenerator::default();
        assert!(generator.tokens.is_empty());
        assert!(!generator.finished);
    }

    #[test]
    fn test_streaming_generator_add_token() {
        let mut generator = StreamingGenerator::new();
        generator.add_token(1, None);
        generator.add_token(2, Some("hello"));
        assert_eq!(generator.tokens, vec![1, 2]);
        assert_eq!(generator.text, "hello");
        assert_eq!(generator.total_tokens, 2);
    }

    #[test]
    fn test_streaming_generator_add_token_with_text() {
        let mut generator = StreamingGenerator::new();
        generator.add_token(0, Some("Hello "));
        generator.add_token(1, Some("world"));
        generator.add_token(2, Some("!"));
        assert_eq!(generator.text, "Hello world!");
        assert_eq!(generator.token_count(), 3);
    }

    #[test]
    fn test_streaming_generator_token_count() {
        let mut generator = StreamingGenerator::new();
        assert_eq!(generator.token_count(), 0);
        generator.add_token(1, None);
        assert_eq!(generator.token_count(), 1);
        generator.add_token(2, None);
        generator.add_token(3, None);
        assert_eq!(generator.token_count(), 3);
    }

    #[test]
    fn test_streaming_generator_finish() {
        let mut generator = StreamingGenerator::new();
        assert!(!generator.finished);
        generator.add_token(1, Some("test"));
        generator.finish();
        assert!(generator.finished);
    }

    #[test]
    fn test_streaming_generator_accumulates_text() {
        let mut generator = StreamingGenerator::new();
        generator.add_token(1, Some("The "));
        generator.add_token(2, Some("quick "));
        generator.add_token(3, Some("brown "));
        generator.add_token(4, Some("fox"));
        assert_eq!(generator.text, "The quick brown fox");
    }

    #[test]
    fn test_streaming_generator_none_text_no_accumulation() {
        let mut generator = StreamingGenerator::new();
        generator.add_token(1, None);
        generator.add_token(2, None);
        assert!(generator.text.is_empty());
        assert_eq!(generator.tokens, vec![1, 2]);
    }

    // ===== XTC (Exclude Top Choices) Sampling Tests =====

    #[test]
    fn test_xtc_config_default() {
        let config = XtcConfig::default();
        assert_eq!(config.probability, 0.0);
        assert_eq!(config.threshold, 0.5);
        assert_eq!(config.min_keep, 1);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_xtc_config_enabled() {
        let config = XtcConfig::new(0.5).with_threshold(0.3).with_min_keep(2);
        assert!(config.is_enabled());
        assert_eq!(config.probability, 0.5);
        assert_eq!(config.threshold, 0.3);
        assert_eq!(config.min_keep, 2);
    }

    #[test]
    fn test_xtc_disabled_no_change() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let config = XtcConfig::default(); // disabled
        let result = apply_xtc(&logits, &config, 0.5);
        assert_eq!(result.data(), logits.data());
    }

    #[test]
    fn test_xtc_rng_above_probability_no_change() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let config = XtcConfig::new(0.5); // 50% probability
        let result = apply_xtc(&logits, &config, 0.8); // rng > probability
        assert_eq!(result.data(), logits.data());
    }

    #[test]
    fn test_xtc_excludes_top_tokens() {
        let logits = Tensor::from_vec(vec![5], vec![10.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let config = XtcConfig::new(1.0).with_threshold(0.5); // Always exclude, high threshold
        let result = apply_xtc(&logits, &config, 0.0); // rng < probability
                                                       // Top token (index 0) should be excluded (set to NEG_INFINITY)
        assert_eq!(result.data()[0], f32::NEG_INFINITY);
    }

    #[test]
    fn test_xtc_respects_min_keep() {
        let logits = Tensor::from_vec(vec![3], vec![10.0, 9.0, 8.0]).expect("test");
        let config = XtcConfig::new(1.0).with_threshold(0.1).with_min_keep(2);
        let result = apply_xtc(&logits, &config, 0.0);
        // Should keep at least 2 tokens (not set all to NEG_INFINITY)
        let finite_count = result.data().iter().filter(|&&x| x.is_finite()).count();
        assert!(finite_count >= 2);
    }

    // ===== Eta Sampling Tests =====

    #[test]
    fn test_eta_config_default() {
        let config = EtaConfig::default();
        assert_eq!(config.eta, 0.3);
        assert_eq!(config.min_p, 0.0001);
        assert!(config.is_enabled());
    }

    #[test]
    fn test_eta_config_disabled() {
        let config = EtaConfig::new(0.0);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_eta_config_builder() {
        let config = EtaConfig::new(0.5).with_min_p(0.001);
        assert_eq!(config.eta, 0.5);
        assert_eq!(config.min_p, 0.001);
    }

    #[test]
    fn test_eta_sampling_basic() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.0, 0.5, 0.1, -1.0]).expect("test");
        let config = EtaConfig::default();
        let result = sample_eta(&logits, &config, 0.5);
        assert!(result.is_ok());
        assert!(result.expect("test") < 5);
    }

    #[test]
    fn test_eta_sampling_single_token() {
        let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
        let config = EtaConfig::default();
        let result = sample_eta(&logits, &config, 0.5).expect("test");
        assert_eq!(result, 0);
    }

    #[test]
    fn test_eta_sampling_uniform() {
        let logits = Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).expect("test");
        let config = EtaConfig::default();
        let result = sample_eta(&logits, &config, 0.5).expect("test");
        assert!(result < 4);
    }

    // ===== Token Healing Tests =====

    #[test]
    fn test_token_healing_config_default() {
        let config = TokenHealingConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.max_backup_chars, 0);
    }

    #[test]
    fn test_token_healing_config_enabled() {
        let config = TokenHealingConfig::new(true).with_max_backup(15);
        assert!(config.enabled);
        assert_eq!(config.max_backup_chars, 15);
    }

    #[test]
    fn test_token_healing_no_heal_needed() {
        let tokens = vec![1, 2, 3, 4, 5];
        let result = analyze_token_healing(&tokens, Some("hello"));
        assert_eq!(result.adjusted_tokens, tokens);
        assert!(result.prefix_constraint.is_none());
        assert_eq!(result.tokens_removed, 0);
    }

    #[test]
    fn test_token_healing_partial_word() {
        let tokens = vec![1, 2, 3, 4, 5];
        // "wo" is a short alphanumeric token without leading space - should heal
        let result = analyze_token_healing(&tokens, Some("wo"));
        assert_eq!(result.adjusted_tokens, vec![1, 2, 3, 4]);
        assert_eq!(result.prefix_constraint, Some("wo".to_string()));
        assert_eq!(result.tokens_removed, 1);
    }

    #[test]
    fn test_token_healing_empty_tokens() {
        let tokens: Vec<usize> = vec![];
        let result = analyze_token_healing(&tokens, Some("a"));
        assert!(result.adjusted_tokens.is_empty());
        assert!(result.prefix_constraint.is_none());
    }

    #[test]
    fn test_token_healing_space_prefix_no_heal() {
        let tokens = vec![1, 2, 3];
        // Token starting with space - no healing needed
        let result = analyze_token_healing(&tokens, Some(" word"));
        assert_eq!(result.adjusted_tokens, tokens);
        assert!(result.prefix_constraint.is_none());
    }

    // ===== Classifier-Free Guidance (CFG) Tests =====

    #[test]
    fn test_cfg_config_default() {
        let config = CfgConfig::default();
        assert_eq!(config.scale, 1.0);
        assert!(config.negative_prompt_tokens.is_empty());
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_cfg_config_enabled() {
        let config = CfgConfig::new(1.5).with_negative_prompt(vec![1, 2, 3]);
        assert!(config.is_enabled());
        assert_eq!(config.scale, 1.5);
        assert_eq!(config.negative_prompt_tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_cfg_scale_one_no_change() {
        let cond = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        let uncond = Tensor::from_vec(vec![4], vec![0.5, 1.5, 2.5, 3.5]).expect("test");
        let result = apply_cfg(&cond, &uncond, 1.0).expect("test");
        // scale=1.0: uncond + 1.0 * (cond - uncond) = cond
        assert_eq!(result.data(), cond.data());
    }

    #[test]
    fn test_cfg_scale_zero_returns_uncond() {
        let cond = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        let uncond = Tensor::from_vec(vec![4], vec![0.5, 1.5, 2.5, 3.5]).expect("test");
        let result = apply_cfg(&cond, &uncond, 0.0).expect("test");
        // scale=0.0: uncond + 0.0 * (cond - uncond) = uncond
        assert_eq!(result.data(), uncond.data());
    }

    #[test]
    fn test_cfg_amplifies_difference() {
        let cond = Tensor::from_vec(vec![3], vec![2.0, 1.0, 0.0]).expect("test");
        let uncond = Tensor::from_vec(vec![3], vec![1.0, 1.0, 1.0]).expect("test");
        let result = apply_cfg(&cond, &uncond, 2.0).expect("test");
        // scale=2.0: uncond + 2.0 * (cond - uncond)
        // = [1,1,1] + 2*([2,1,0] - [1,1,1])
        // = [1,1,1] + 2*[1,0,-1]
        // = [1,1,1] + [2,0,-2]
        // = [3,1,-1]
        assert_eq!(result.data(), &[3.0, 1.0, -1.0]);
    }

    #[test]
    fn test_cfg_shape_mismatch_error() {
        let cond = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        let uncond = Tensor::from_vec(vec![3], vec![0.5, 1.5, 2.5]).expect("test");
        let result = apply_cfg(&cond, &uncond, 1.5);
        assert!(result.is_err());
    }

    // ===== Prompt Cache Tests =====

    #[test]
    fn test_prompt_cache_creation() {
        let cache = PromptCache::new(50);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_prompt_cache_default() {
        let cache = PromptCache::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_prompt_cache_add_and_find() {
        let mut cache = PromptCache::new(10);
        cache.add(vec![1, 2, 3], 12345);
        assert_eq!(cache.len(), 1);

        // Find exact match
        let result = cache.find_prefix(&[1, 2, 3]);
        assert!(result.is_some());
        let (len, kv_hash) = result.expect("test");
        assert_eq!(len, 3);
        assert_eq!(kv_hash, 12345);
    }

    #[test]
    fn test_prompt_cache_find_prefix() {
        let mut cache = PromptCache::new(10);
        cache.add(vec![1, 2], 111);
        cache.add(vec![1, 2, 3], 222);

        // Should find longer prefix first
        let result = cache.find_prefix(&[1, 2, 3, 4]);
        assert!(result.is_some());
        let (len, _) = result.expect("test");
        assert_eq!(len, 3);
    }

    #[test]
    fn test_prompt_cache_miss() {
        let mut cache = PromptCache::new(10);
        cache.add(vec![1, 2, 3], 12345);

        // No matching prefix
        let result = cache.find_prefix(&[4, 5, 6]);
        assert!(result.is_none());
    }

    #[test]
    fn test_prompt_cache_clear() {
        let mut cache = PromptCache::new(10);
        cache.add(vec![1, 2, 3], 12345);
        cache.add(vec![4, 5, 6], 67890);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_prompt_cache_stats() {
        let mut cache = PromptCache::new(100);
        cache.add(vec![1, 2, 3], 12345);

        // Hit the cache
        cache.find_prefix(&[1, 2, 3]);
        cache.find_prefix(&[1, 2, 3]);

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.total_hits, 2);
        assert_eq!(stats.max_entries, 100);
    }

    #[test]
    fn test_prompt_cache_eviction() {
        let mut cache = PromptCache::new(2);
        cache.add(vec![1], 111);
        cache.add(vec![2], 222);
        assert_eq!(cache.len(), 2);

        // Adding third entry should evict LRU
        cache.add(vec![3], 333);
        assert_eq!(cache.len(), 2);
    }

    // ========================================================================
    // Dynamic Temperature Tests (EXTREME TDD)
    // ========================================================================

    #[test]
    fn test_dyn_temp_config_default() {
        let config = DynTempConfig::default();
        assert!((config.temp - 1.0).abs() < 1e-6);
        assert!((config.delta - 0.0).abs() < 1e-6);
        assert!((config.exponent - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dyn_temp_config_new() {
        let config = DynTempConfig::new(0.8, 0.2, 1.5);
        assert!((config.temp - 0.8).abs() < 1e-6);
        assert!((config.delta - 0.2).abs() < 1e-6);
        assert!((config.exponent - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_dyn_temp_config_static() {
        let config = DynTempConfig::static_temp(0.5);
        assert!((config.temp - 0.5).abs() < 1e-6);
        assert!((config.delta - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dyn_temp_no_delta_uses_static() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = DynTempConfig::static_temp(0.5);

        let result = apply_dynamic_temperature(&logits, &config);
        let static_result = apply_temperature(&logits, 0.5).expect("test");

        // Should be identical to static temperature
        for (a, b) in result.data().iter().zip(static_result.data().iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dyn_temp_single_element() {
        let logits = Tensor::from_vec(vec![1], vec![5.0]).expect("test");
        let config = DynTempConfig::new(1.0, 0.5, 1.0);

        let result = apply_dynamic_temperature(&logits, &config);
        // Single element should return unchanged
        assert!((result.data()[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dyn_temp_low_entropy_higher_temp() {
        // Low entropy (one dominant logit) should use higher temperature
        let logits = Tensor::from_vec(vec![5], vec![10.0, 0.0, 0.0, 0.0, 0.0]).expect("test");
        let config = DynTempConfig::new(1.0, 0.5, 1.0);

        let result = apply_dynamic_temperature(&logits, &config);
        // Result should be scaled, but logits should still be ordered
        assert!(result.data()[0] > result.data()[1]);
    }

    #[test]
    fn test_dyn_temp_high_entropy_lower_temp() {
        // High entropy (uniform logits) should use lower temperature
        let logits = Tensor::from_vec(vec![5], vec![1.0, 1.0, 1.0, 1.0, 1.0]).expect("test");
        let config = DynTempConfig::new(1.0, 0.5, 1.0);

        let result = apply_dynamic_temperature(&logits, &config);
        // With uniform logits and high entropy, should use max temp
        // All values should be close to 1.0 (uniform scaled)
        let sum: f32 = result.data().iter().sum();
        assert!(sum.abs() > 0.0); // Non-degenerate
    }

    #[test]
    fn test_dyn_temp_exponent_affects_scaling() {
        let logits = Tensor::from_vec(vec![5], vec![2.0, 1.5, 1.0, 0.5, 0.0]).expect("test");
        let config_exp1 = DynTempConfig::new(1.0, 0.5, 1.0);
        let config_exp2 = DynTempConfig::new(1.0, 0.5, 2.0);

        let result1 = apply_dynamic_temperature(&logits, &config_exp1);
        let result2 = apply_dynamic_temperature(&logits, &config_exp2);

        // Different exponents should produce different results
        let diff: f32 = result1
            .data()
            .iter()
            .zip(result2.data().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6);
    }

    // ========================================================================
    // Infill/FIM Sampler Tests (EXTREME TDD)
    // ========================================================================

    #[test]
    fn test_infill_config_default() {
        let config = InfillConfig::default();
        assert!(config.eog_tokens.is_empty());
        assert!((config.eog_ratio_threshold - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_infill_config_new() {
        let config = InfillConfig::new(vec![1, 2, 3]);
        assert_eq!(config.eog_tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_infill_config_with_threshold() {
        let config = InfillConfig::new(vec![1]).with_threshold(5.0);
        assert!((config.eog_ratio_threshold - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_infill_empty_eog_tokens() {
        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let config = InfillConfig::default();

        let result = apply_infill_sampling(&logits, &config);
        assert!(!result.force_eog);
        assert!((result.p_txt - 1.0).abs() < 1e-6);
        assert!((result.p_eog - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_infill_no_force_eog_when_text_dominant() {
        // Text tokens have much higher probability than EOG
        let logits = Tensor::from_vec(vec![5], vec![10.0, 10.0, 10.0, 10.0, 0.0]).expect("test");
        let config = InfillConfig::new(vec![4]); // Token 4 is EOG

        let result = apply_infill_sampling(&logits, &config);
        assert!(!result.force_eog);
        assert!(result.p_txt > result.p_eog);
    }

    #[test]
    fn test_infill_force_eog_when_eog_dominant() {
        // EOG token has high probability relative to text
        let logits = Tensor::from_vec(vec![5], vec![0.0, 0.0, 0.0, 0.0, 10.0]).expect("test");
        let config = InfillConfig::new(vec![4]); // Token 4 is EOG

        let result = apply_infill_sampling(&logits, &config);
        assert!(result.force_eog);
        assert!(result.p_eog > 0.5);
    }

    #[test]
    fn test_infill_modified_logits_when_force_eog() {
        let logits = Tensor::from_vec(vec![5], vec![0.0, 0.0, 0.0, 0.0, 10.0]).expect("test");
        let config = InfillConfig::new(vec![4]);

        let result = apply_infill_sampling(&logits, &config);
        if result.force_eog {
            // Non-EOG tokens should be -inf
            assert!(result.logits.data()[0] == f32::NEG_INFINITY);
            assert!(result.logits.data()[1] == f32::NEG_INFINITY);
            // EOG token should remain
            assert!(result.logits.data()[4] > f32::NEG_INFINITY);
        }
    }

    #[test]
    fn test_infill_multiple_eog_tokens() {
        let logits = Tensor::from_vec(vec![5], vec![0.0, 0.0, 0.0, 5.0, 5.0]).expect("test");
        let config = InfillConfig::new(vec![3, 4]); // Tokens 3 and 4 are EOG

        let result = apply_infill_sampling(&logits, &config);
        // Check that both EOG tokens contribute to p_eog
        assert!(result.p_eog > 0.0);
    }

    // ========================================================================
    // Sampler Chain Tests (EXTREME TDD)
    // ========================================================================

    #[test]
    fn test_sampler_chain_new() {
        let chain = SamplerChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_sampler_chain_default() {
        let chain = SamplerChain::default();
        assert!(chain.is_empty());
    }

    #[test]
    fn test_sampler_chain_with_sampler() {
        let chain = SamplerChain::new().with_sampler(TemperatureSampler::new(0.8));
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.names(), vec!["temperature"]);
    }

    #[test]
    fn test_sampler_chain_multiple_samplers() {
        let chain = SamplerChain::new()
            .with_sampler(TemperatureSampler::new(0.8))
            .with_sampler(TopKSampler::new(50))
            .with_sampler(TopPSampler::new(0.9));

        assert_eq!(chain.len(), 3);
        assert_eq!(chain.names(), vec!["temperature", "top_k", "top_p"]);
    }

    #[test]
    fn test_sampler_chain_push() {
        let mut chain = SamplerChain::new();
        chain.push(Box::new(TemperatureSampler::new(0.5)));
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn test_sampler_chain_apply() {
        let chain = SamplerChain::new().with_sampler(TemperatureSampler::new(0.5));

        let mut logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let context = SamplerContext::new();

        chain.apply(&mut logits, &context);

        // Temperature 0.5 should double the logits
        assert!((logits.data()[0] - 2.0).abs() < 1e-6);
        assert!((logits.data()[4] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_sampler_chain_sample() {
        let chain = SamplerChain::new().with_sampler(TemperatureSampler::new(1.0));

        let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        let context = SamplerContext::new();

        let result = chain.sample(&logits, &context).expect("test");
        assert_eq!(result, 4); // Greedy should pick max
    }

    #[test]
    fn test_sampler_chain_clone() {
        let chain = SamplerChain::new()
            .with_sampler(TemperatureSampler::new(0.8))
            .with_sampler(TopKSampler::new(10));

        let cloned = chain.clone();
        assert_eq!(cloned.len(), 2);
        assert_eq!(cloned.names(), vec!["temperature", "top_k"]);
    }

    #[test]
    fn test_sampler_context_default() {
        let ctx = SamplerContext::default();
        assert!(ctx.tokens.is_empty());
        assert!((ctx.rng_value - 0.0).abs() < 1e-6);
        assert_eq!(ctx.step, 0);
    }

    #[test]
    fn test_sampler_context_builders() {
        let ctx = SamplerContext::new()
            .with_tokens(vec![1, 2, 3])
            .with_rng(0.5)
            .with_step(10);

        assert_eq!(ctx.tokens, vec![1, 2, 3]);
        assert!((ctx.rng_value - 0.5).abs() < 1e-6);
        assert_eq!(ctx.step, 10);
    }

    #[test]
    fn test_temperature_sampler() {
        let sampler = TemperatureSampler::new(0.5);
        assert_eq!(sampler.name(), "temperature");
    }

    #[test]
    fn test_dyn_temp_sampler() {
        let sampler = DynTempSampler::new(DynTempConfig::new(1.0, 0.5, 1.0));
        assert_eq!(sampler.name(), "dyn_temp");
    }

    #[test]
    fn test_top_k_sampler() {
        let sampler = TopKSampler::new(10);
        assert_eq!(sampler.name(), "top_k");
        assert_eq!(sampler.k, 10);
    }

    #[test]
    fn test_top_p_sampler() {
        let sampler = TopPSampler::new(0.9);
        assert_eq!(sampler.name(), "top_p");
        assert!((sampler.p - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_sampler() {
        let sampler = RepetitionPenaltySampler::new(RepetitionPenaltyConfig::new(1.2));
        assert_eq!(sampler.name(), "repetition_penalty");
    }

    #[test]
    fn test_infill_sampler() {
        let sampler = InfillSampler::new(InfillConfig::new(vec![1, 2]));
        assert_eq!(sampler.name(), "infill");
    }

    #[test]
    fn test_top_k_sampler_apply() {
        let sampler = TopKSampler::new(2);
        let mut logits = Tensor::from_vec(vec![5], vec![1.0, 5.0, 3.0, 2.0, 4.0]).expect("test");
        let context = SamplerContext::new();

        sampler.apply(&mut logits, &context);

        // Only top 2 (indices 1 and 4) should remain
        let data = logits.data();
        assert!(data[0] == f32::NEG_INFINITY);
        assert!(data[1] > f32::NEG_INFINITY); // 5.0 is top
        assert!(data[2] == f32::NEG_INFINITY);
        assert!(data[3] == f32::NEG_INFINITY);
        assert!(data[4] > f32::NEG_INFINITY); // 4.0 is second
    }

    #[test]
    fn test_top_p_sampler_apply() {
        let sampler = TopPSampler::new(0.5);
        let mut logits = Tensor::from_vec(vec![5], vec![1.0, 5.0, 2.0, 0.0, 0.0]).expect("test");
        let context = SamplerContext::new();

        sampler.apply(&mut logits, &context);

        // Top token (index 1 with 5.0) should definitely remain
        let data = logits.data();
        assert!(data[1] > f32::NEG_INFINITY);
    }

    #[test]
    fn test_full_sampler_pipeline() {
        // Build a realistic pipeline: temp -> top_k -> top_p
        let chain = SamplerChain::new()
            .with_sampler(TemperatureSampler::new(0.8))
            .with_sampler(TopKSampler::new(50))
            .with_sampler(TopPSampler::new(0.95));

        let logits = Tensor::from_vec(
            vec![10],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .expect("test");
        let context = SamplerContext::new();

        let result = chain.sample(&logits, &context).expect("test");
        assert_eq!(result, 9); // Should still pick max after pipeline
    }

    // =========================================================================
    // LogitProcessor Tests (RLZR-GEN-001)
    // =========================================================================

    #[test]
    fn test_logit_processor_context() {
        let tokens = vec![1u32, 2, 3, 4, 5];
        let ctx = LogitProcessorContext::new(&tokens, 3, 1000);

        assert_eq!(ctx.tokens, &[1, 2, 3, 4, 5]);
        assert_eq!(ctx.step, 3);
        assert_eq!(ctx.n_vocab, 1000);
    }

    #[test]
    fn test_token_suppressor_basic() {
        let suppressor = TokenSuppressor::new(vec![0, 5, 9]);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ctx = LogitProcessorContext::new(&[], 0, 10);

        suppressor.process(&mut logits, &ctx);

        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!(logits[5].is_infinite() && logits[5] < 0.0);
        assert!(logits[9].is_infinite() && logits[9] < 0.0);
    }

    #[test]
    fn test_token_suppressor_out_of_bounds() {
        let suppressor = TokenSuppressor::new(vec![100, 200]); // Out of bounds
        let mut logits = vec![1.0, 2.0, 3.0];
        let ctx = LogitProcessorContext::new(&[], 0, 3);

        // Should not panic
        suppressor.process(&mut logits, &ctx);

        // Logits unchanged
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_token_suppressor_name() {
        let suppressor = TokenSuppressor::new(vec![]);
        assert_eq!(suppressor.name(), "token_suppressor");
    }

    #[test]
    fn test_repetition_penalty_basic() {
        let penalty = RepetitionPenalty::with_penalty(2.0);
        let tokens = vec![1u32, 3, 5];
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ctx = LogitProcessorContext::new(&tokens, 0, 6);

        penalty.process(&mut logits, &ctx);

        // Token 1 (logit 2.0) should be halved: 2.0 / 2.0 = 1.0
        assert!((logits[1] - 1.0).abs() < 1e-6);
        // Token 3 (logit 4.0) should be halved: 4.0 / 2.0 = 2.0
        assert!((logits[3] - 2.0).abs() < 1e-6);
        // Token 5 (logit 6.0) should be halved: 6.0 / 2.0 = 3.0
        assert!((logits[5] - 3.0).abs() < 1e-6);
        // Token 0 unchanged
        assert!((logits[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_negative_logits() {
        let penalty = RepetitionPenalty::with_penalty(2.0);
        let tokens = vec![0u32];
        let mut logits = vec![-2.0, 1.0];
        let ctx = LogitProcessorContext::new(&tokens, 0, 2);

        penalty.process(&mut logits, &ctx);

        // Negative logit should be multiplied: -2.0 * 2.0 = -4.0
        assert!((logits[0] - (-4.0)).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_with_window() {
        let penalty = RepetitionPenalty::new(2.0, 2); // Window of 2
        let tokens = vec![1u32, 2, 3, 4]; // Only last 2 (3, 4) should be penalized
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ctx = LogitProcessorContext::new(&tokens, 0, 5);

        penalty.process(&mut logits, &ctx);

        // Token 1, 2 NOT penalized (outside window)
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
        // Token 3, 4 penalized (inside window)
        assert!((logits[3] - 2.0).abs() < 1e-6); // 4.0 / 2.0
        assert!((logits[4] - 2.5).abs() < 1e-6); // 5.0 / 2.0
    }

    #[test]
    fn test_temperature_scaler_basic() {
        let scaler = TemperatureScaler::new(2.0);
        let mut logits = vec![2.0, 4.0, 6.0];
        let ctx = LogitProcessorContext::new(&[], 0, 3);

        scaler.process(&mut logits, &ctx);

        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_temperature_scaler_no_effect_at_1() {
        let scaler = TemperatureScaler::new(1.0);
        let mut logits = vec![1.0, 2.0, 3.0];
        let ctx = LogitProcessorContext::new(&[], 0, 3);

        scaler.process(&mut logits, &ctx);

        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Temperature must be positive")]
    fn test_temperature_scaler_panics_on_zero() {
        let _ = TemperatureScaler::new(0.0);
    }

    #[test]
    fn test_processor_chain_empty() {
        let chain = LogitProcessorChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_processor_chain_add() {
        let chain = LogitProcessorChain::new()
            .with_processor(TokenSuppressor::new(vec![0]))
            .with_processor(RepetitionPenalty::with_penalty(1.5));

        assert_eq!(chain.len(), 2);
        assert!(!chain.is_empty());
    }

    #[test]
    fn test_processor_chain_names() {
        let chain = LogitProcessorChain::new()
            .with_processor(TokenSuppressor::new(vec![0]))
            .with_processor(RepetitionPenalty::with_penalty(1.5))
            .with_processor(TemperatureScaler::new(0.8));

        let names = chain.processor_names();
        assert_eq!(
            names,
            vec![
                "token_suppressor",
                "repetition_penalty",
                "temperature_scaler"
            ]
        );
    }

    #[test]
    fn test_processor_chain_applies_in_order() {
        // Suppress token 0, then apply temp scaling
        let chain = LogitProcessorChain::new()
            .with_processor(TokenSuppressor::new(vec![0]))
            .with_processor(TemperatureScaler::new(2.0));

        let mut logits = vec![10.0, 4.0, 2.0];
        let ctx = LogitProcessorContext::new(&[], 0, 3);

        chain.process(&mut logits, &ctx);

        // Token 0 suppressed (still -inf after scaling)
        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        // Other logits scaled
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_processor_chain_as_logit_processor() {
        let chain = LogitProcessorChain::new().with_processor(TokenSuppressor::new(vec![0]));

        // Use as dyn LogitProcessor
        let processor: &dyn LogitProcessor = &chain;
        assert_eq!(processor.name(), "processor_chain");

        let mut logits = vec![1.0, 2.0];
        let ctx = LogitProcessorContext::new(&[], 0, 2);
        processor.process(&mut logits, &ctx);

        assert!(logits[0].is_infinite());
    }

    // =========================================================================
    // GenerationPipeline Tests
    // =========================================================================

    /// Mock model for testing GenerationPipeline
    struct MockModel {
        vocab_size: usize,
        /// Returns logits with this token as highest
        highest_token: usize,
        call_count: usize,
    }

    impl MockModel {
        fn new(vocab_size: usize, highest_token: usize) -> Self {
            Self {
                vocab_size,
                highest_token,
                call_count: 0,
            }
        }
    }

    impl GenerativeModel for MockModel {
        fn forward(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
            self.call_count += 1;
            let mut logits = vec![0.0f32; self.vocab_size];
            logits[self.highest_token] = 10.0;
            Ok(logits)
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

    #[test]
    fn test_generation_pipeline_basic() {
        let model = MockModel::new(100, 42);
        let mut pipeline = GenerationPipeline::new(model)
            .with_config(GenerationConfig::greedy().with_max_tokens(3));

        let result = pipeline.generate(&[1, 2]).expect("test");

        // Initial tokens + 3 generated
        assert_eq!(result.len(), 5);
        // All generated tokens should be 42 (highest)
        assert_eq!(result[2], 42);
        assert_eq!(result[3], 42);
        assert_eq!(result[4], 42);
    }

    #[test]
    fn test_generation_pipeline_with_eos() {
        // Model that returns EOS token (99) on third call
        struct EosModel {
            call_count: usize,
        }
        impl GenerativeModel for EosModel {
            fn forward(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
                self.call_count += 1;
                let mut logits = vec![0.0f32; 100];
                if self.call_count >= 3 {
                    logits[99] = 10.0; // EOS
                } else {
                    logits[50] = 10.0; // Regular token
                }
                Ok(logits)
            }
            fn vocab_size(&self) -> usize {
                100
            }
        }

        let model = EosModel { call_count: 0 };
        let mut pipeline = GenerationPipeline::new(model).with_config(
            GenerationConfig::greedy()
                .with_max_tokens(10)
                .with_eos_token_id(99),
        );

        let result = pipeline.generate(&[1]).expect("test");

        // Should stop at EOS: [1, 50, 50, 99]
        assert_eq!(result.len(), 4);
        assert_eq!(result[result.len() - 1], 99);
    }

    #[test]
    fn test_generation_pipeline_with_token_suppression() {
        // Model that would return token 0 if not suppressed
        struct ZeroModel;
        impl GenerativeModel for ZeroModel {
            fn forward(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
                let mut logits = vec![0.0f32; 10];
                logits[0] = 10.0; // Token 0 is highest
                logits[5] = 5.0; // Token 5 is second highest
                Ok(logits)
            }
            fn vocab_size(&self) -> usize {
                10
            }
        }

        let model = ZeroModel;
        let mut pipeline = GenerationPipeline::new(model)
            .add_processor(TokenSuppressor::new(vec![0])) // Suppress token 0
            .with_config(GenerationConfig::greedy().with_max_tokens(1));

        let result = pipeline.generate(&[1]).expect("test");

        // Should pick token 5 (second highest) since 0 is suppressed
        assert_eq!(result, vec![1, 5]);
    }

    #[test]
    fn test_generation_pipeline_whisper_use_case() {
        // Simulate Whisper: suppress SOT (50257) to prevent hallucination
        const SOT: u32 = 50257;
        const EOT: u32 = 50256;

        struct WhisperMockModel {
            call_count: usize,
        }
        impl GenerativeModel for WhisperMockModel {
            fn forward(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
                self.call_count += 1;
                let mut logits = vec![0.0f32; 51865];

                // Test scenario: SOT has highest logit (intentional for testing SOT suppression)
                logits[SOT as usize] = 11.0;

                // Text token has second highest
                logits[440] = 10.0; // "The" token

                // EOT after 3 calls
                if self.call_count >= 4 {
                    logits[EOT as usize] = 20.0;
                }

                Ok(logits)
            }
            fn vocab_size(&self) -> usize {
                51865
            }
        }

        let model = WhisperMockModel { call_count: 0 };
        let mut pipeline = GenerationPipeline::new(model)
            .add_processor(TokenSuppressor::new(vec![SOT])) // Suppress SOT
            .with_config(
                GenerationConfig::greedy()
                    .with_max_tokens(10)
                    .with_eos_token_id(EOT as usize),
            );

        let result = pipeline.generate(&[50257, 50258]).expect("test");

        // Should NOT contain SOT (50257) in generated tokens
        for &token in &result[2..] {
            // Skip initial tokens
            assert_ne!(token, SOT, "SOT should be suppressed");
        }

        // Should contain the text token and EOT
        assert!(result.contains(&440), "Should contain text token");
        assert!(result.contains(&EOT), "Should end with EOT");
    }
}
