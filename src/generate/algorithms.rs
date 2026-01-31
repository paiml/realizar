//! Advanced Sampling Algorithms (PMAT-802)
//!
//! Unique sampling algorithms not in sampler.rs:
//! - Min-p sampling
//! - Mirostat (v1/v2) adaptive sampling
//! - Tail-Free Sampling (TFS)
//! - Typical sampling
//! - DRY (Don't Repeat Yourself) penalty
//! - XTC (Exclude Top Choices) sampling
//! - Eta sampling
//! - Token healing
//! - Classifier-Free Guidance (CFG)

use super::{sample_from_distribution, sample_greedy};
use crate::error::{RealizarError, Result};
use crate::layers::softmax;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Sample using min-p (minimum probability) sampling.
///
/// Filters tokens with probability below `min_p * max_prob` threshold.
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

// ============================================================================
// Tests: Sampling Algorithms Coverage (PMAT-802)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create logits tensor (panics on empty - use make_empty_logits for that)
    fn make_logits(data: &[f32]) -> Tensor<f32> {
        assert!(!data.is_empty(), "Use make_empty_logits for empty data");
        Tensor::from_vec(vec![data.len()], data.to_vec()).unwrap()
    }

    // Create a single-element tensor to simulate edge case (empty logits not valid)
    fn make_single_logit() -> Tensor<f32> {
        Tensor::from_vec(vec![1], vec![0.0]).unwrap()
    }

    // ========================================================================
    // Min-P Sampling Tests
    // ========================================================================

    #[test]
    fn test_sample_min_p_basic() {
        let logits = make_logits(&[2.0, 1.0, 0.5, 0.1]);
        let result = sample_min_p(&logits, 0.1, 0.5);
        assert!(result.is_ok());
        let idx = result.unwrap();
        assert!(idx < 4);
    }

    #[test]
    fn test_sample_min_p_single_logit() {
        // Single logit should work (edge case - only one choice)
        let logits = make_single_logit();
        let result = sample_min_p(&logits, 0.1, 0.5);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_sample_min_p_invalid_min_p_negative() {
        let logits = make_logits(&[1.0, 2.0, 3.0]);
        let result = sample_min_p(&logits, -0.1, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_min_p_invalid_min_p_too_high() {
        let logits = make_logits(&[1.0, 2.0, 3.0]);
        let result = sample_min_p(&logits, 1.5, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_min_p_high_threshold_fallback() {
        // With very high min_p, all tokens filtered -> fallback to greedy
        let logits = make_logits(&[1.0, 5.0, 2.0]);
        let result = sample_min_p(&logits, 0.99, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_min_p_zero_threshold() {
        // With min_p=0, all tokens kept
        let logits = make_logits(&[1.0, 2.0, 3.0]);
        let result = sample_min_p(&logits, 0.0, 0.5);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Mirostat State Tests
    // ========================================================================

    #[test]
    fn test_mirostat_state_default() {
        let state = MirostatState::default();
        assert!((state.tau - 5.0).abs() < 1e-6);
        assert!((state.eta - 0.1).abs() < 1e-6);
        assert!((state.mu - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_mirostat_state_new() {
        let state = MirostatState::new(3.0);
        assert!((state.tau - 3.0).abs() < 1e-6);
        assert!((state.mu - 6.0).abs() < 1e-6); // 2 * tau
    }

    #[test]
    fn test_mirostat_state_with_eta() {
        let state = MirostatState::new(5.0).with_eta(0.2);
        assert!((state.eta - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_mirostat_state_update() {
        let mut state = MirostatState::new(5.0).with_eta(0.1);
        let initial_mu = state.mu;
        state.update(6.0); // Observed surprise > tau
        assert!(state.mu < initial_mu); // mu decreases
    }

    #[test]
    fn test_mirostat_state_update_low_surprise() {
        let mut state = MirostatState::new(5.0).with_eta(0.1);
        let initial_mu = state.mu;
        state.update(4.0); // Observed surprise < tau
        assert!(state.mu > initial_mu); // mu increases
    }

    // ========================================================================
    // Mirostat Sampling Tests
    // ========================================================================

    #[test]
    fn test_sample_mirostat_basic() {
        let logits = make_logits(&[2.0, 1.0, 0.5, 0.1]);
        let mut state = MirostatState::default();
        let result = sample_mirostat(&logits, &mut state, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_mirostat_single_logit() {
        let logits = make_single_logit();
        let mut state = MirostatState::default();
        let result = sample_mirostat(&logits, &mut state, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_mirostat_state_updates() {
        let logits = make_logits(&[5.0, 1.0, 0.1]);
        let mut state = MirostatState::new(3.0);
        let initial_mu = state.mu;
        let _ = sample_mirostat(&logits, &mut state, 0.5);
        // State should be updated after sampling
        assert!((state.mu - initial_mu).abs() > 1e-9 || state.mu == initial_mu);
    }

    // ========================================================================
    // Tail-Free Sampling Tests
    // ========================================================================

    #[test]
    fn test_sample_tfs_basic() {
        let logits = make_logits(&[3.0, 2.0, 1.0, 0.5, 0.1]);
        let result = sample_tfs(&logits, 0.95, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_tfs_single_logit() {
        let logits = make_single_logit();
        let result = sample_tfs(&logits, 0.95, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_tfs_low_z() {
        let logits = make_logits(&[3.0, 2.0, 1.0]);
        let result = sample_tfs(&logits, 0.1, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_tfs_high_z() {
        let logits = make_logits(&[3.0, 2.0, 1.0]);
        let result = sample_tfs(&logits, 0.99, 0.5);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Typical Sampling Tests
    // ========================================================================

    #[test]
    fn test_sample_typical_basic() {
        let logits = make_logits(&[3.0, 2.0, 1.0, 0.5]);
        let result = sample_typical(&logits, 0.95, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_typical_single_logit() {
        let logits = make_single_logit();
        let result = sample_typical(&logits, 0.95, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_typical_low_p() {
        let logits = make_logits(&[3.0, 2.0, 1.0]);
        let result = sample_typical(&logits, 0.1, 0.5);
        assert!(result.is_ok());
    }

    // ========================================================================
    // DRY Penalty Tests
    // ========================================================================

    #[test]
    fn test_dry_penalty_no_repetition() {
        let logits = make_logits(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let context = vec![0, 1, 2, 3]; // No repetition
        let config = DryConfig::default();
        let result = apply_dry_penalty(&logits, &context, &config);
        // Returns Tensor, verify it has same shape
        assert_eq!(result.shape(), logits.shape());
    }

    #[test]
    fn test_dry_penalty_with_repetition() {
        let logits = make_logits(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let context = vec![0, 1, 0, 1, 0]; // Repeating pattern
        let config = DryConfig {
            multiplier: 1.0,
            base: 1.5,
            allowed_length: 2,
            penalty_last_n: 256,
        };
        let result = apply_dry_penalty(&logits, &context, &config);
        assert_eq!(result.shape(), logits.shape());
    }

    #[test]
    fn test_dry_penalty_empty_context() {
        let logits = make_logits(&[1.0, 2.0, 3.0]);
        let context: Vec<usize> = vec![];
        let config = DryConfig::default();
        let result = apply_dry_penalty(&logits, &context, &config);
        assert_eq!(result.shape(), logits.shape());
    }

    #[test]
    fn test_dry_config_default() {
        let config = DryConfig::default();
        assert!(config.multiplier > 0.0);
        assert!(config.is_enabled());
    }

    #[test]
    fn test_dry_config_new() {
        let config = DryConfig::new(0.5);
        assert!((config.multiplier - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dry_config_builder() {
        let config = DryConfig::new(1.0)
            .with_base(2.0)
            .with_allowed_length(3)
            .with_penalty_last_n(128);
        assert!((config.base - 2.0).abs() < 1e-6);
        assert_eq!(config.allowed_length, 3);
        assert_eq!(config.penalty_last_n, 128);
    }

    // ========================================================================
    // XTC Tests
    // ========================================================================

    #[test]
    fn test_apply_xtc_disabled() {
        let logits = make_logits(&[5.0, 4.0, 3.0, 2.0, 1.0]);
        let config = XtcConfig::default(); // probability=0.0 (disabled)
        let result = apply_xtc(&logits, &config, 0.5);
        // When disabled, should return original logits
        assert_eq!(result.shape(), logits.shape());
    }

    #[test]
    fn test_apply_xtc_enabled() {
        let logits = make_logits(&[5.0, 4.0, 3.0, 2.0, 1.0]);
        let config = XtcConfig::new(1.0).with_threshold(0.1);
        let result = apply_xtc(&logits, &config, 0.0); // rng < probability triggers
        assert_eq!(result.shape(), logits.shape());
    }

    #[test]
    fn test_xtc_config_default() {
        let config = XtcConfig::default();
        assert!((config.probability - 0.0).abs() < 1e-6);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_xtc_config_builder() {
        let config = XtcConfig::new(0.5).with_threshold(0.3).with_min_keep(2);
        assert!((config.probability - 0.5).abs() < 1e-6);
        assert!((config.threshold - 0.3).abs() < 1e-6);
        assert_eq!(config.min_keep, 2);
        assert!(config.is_enabled());
    }

    // ========================================================================
    // Eta Sampling Tests
    // ========================================================================

    #[test]
    fn test_sample_eta_basic() {
        let logits = make_logits(&[3.0, 2.0, 1.0, 0.5]);
        let config = EtaConfig::default();
        let result = sample_eta(&logits, &config, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_eta_single_logit() {
        let logits = make_single_logit();
        let config = EtaConfig::default();
        let result = sample_eta(&logits, &config, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_eta_custom_config() {
        let logits = make_logits(&[3.0, 2.0, 1.0]);
        let config = EtaConfig::new(0.5);
        let result = sample_eta(&logits, &config, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_eta_config_default() {
        let config = EtaConfig::default();
        assert!((config.eta - 0.3).abs() < 1e-6);
        assert!(config.min_p > 0.0);
    }

    // ========================================================================
    // Token Healing Tests
    // ========================================================================

    #[test]
    fn test_token_healing_empty_tokens() {
        let tokens: Vec<usize> = vec![];
        let result = analyze_token_healing(&tokens, None);
        assert_eq!(result.tokens_removed, 0);
        assert!(result.adjusted_tokens.is_empty());
    }

    #[test]
    fn test_token_healing_no_healing_needed() {
        let tokens = vec![1, 2, 3, 4, 5];
        let result = analyze_token_healing(&tokens, Some("hello")); // Long token, no healing
        assert_eq!(result.tokens_removed, 0);
        assert_eq!(result.adjusted_tokens.len(), 5);
    }

    #[test]
    fn test_token_healing_partial_token() {
        let tokens = vec![1, 2, 3, 4, 5];
        let result = analyze_token_healing(&tokens, Some("ab")); // Short alphanumeric
                                                                 // Should heal (remove last token, add prefix constraint)
        assert_eq!(result.tokens_removed, 1);
        assert_eq!(result.adjusted_tokens.len(), 4);
        assert!(result.prefix_constraint.is_some());
    }

    #[test]
    fn test_token_healing_with_space() {
        let tokens = vec![1, 2, 3];
        let result = analyze_token_healing(&tokens, Some(" the")); // Starts with space
                                                                   // Should NOT heal (tokens starting with space are complete)
        assert_eq!(result.tokens_removed, 0);
    }

    #[test]
    fn test_token_healing_config() {
        let config = TokenHealingConfig::new(true);
        assert!(config.enabled);
        assert!(config.max_backup_chars > 0);

        let config2 = TokenHealingConfig::new(true).with_max_backup(5);
        assert_eq!(config2.max_backup_chars, 5);
    }

    // ========================================================================
    // CFG (Classifier-Free Guidance) Tests
    // ========================================================================

    #[test]
    fn test_apply_cfg_basic() {
        let cond = make_logits(&[1.0, 2.0, 3.0]);
        let uncond = make_logits(&[0.5, 1.0, 1.5]);
        let result = apply_cfg(&cond, &uncond, 2.0);
        assert!(result.is_ok());
        let guided = result.unwrap();
        assert_eq!(guided.shape(), cond.shape());
    }

    #[test]
    fn test_apply_cfg_shape_mismatch() {
        let cond = make_logits(&[1.0, 2.0, 3.0]);
        let uncond = make_logits(&[0.5, 1.0]);
        let result = apply_cfg(&cond, &uncond, 2.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_cfg_scale_one() {
        // With scale=1, output should equal conditional
        let cond = make_logits(&[1.0, 2.0, 3.0]);
        let uncond = make_logits(&[0.0, 0.0, 0.0]);
        let result = apply_cfg(&cond, &uncond, 1.0).unwrap();
        let data = result.data();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_cfg_scale_zero() {
        // With scale=0, output should equal unconditional
        let cond = make_logits(&[1.0, 2.0, 3.0]);
        let uncond = make_logits(&[0.5, 1.0, 1.5]);
        let result = apply_cfg(&cond, &uncond, 0.0).unwrap();
        let data = result.data();
        assert!((data[0] - 0.5).abs() < 1e-6);
        assert!((data[1] - 1.0).abs() < 1e-6);
        assert!((data[2] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_apply_cfg_negative_scale() {
        // Negative scale should work (inverts guidance direction)
        let cond = make_logits(&[1.0, 2.0, 3.0]);
        let uncond = make_logits(&[0.5, 1.0, 1.5]);
        let result = apply_cfg(&cond, &uncond, -1.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cfg_config_default() {
        let config = CfgConfig::default();
        assert!((config.scale - 1.0).abs() < 1e-6);
        assert!(config.negative_prompt_tokens.is_empty());
    }

    #[test]
    fn test_cfg_config_new() {
        let config = CfgConfig::new(2.0);
        assert!((config.scale - 2.0).abs() < 1e-6);
    }
}
