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

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod generate_tests;
