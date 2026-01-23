//! Advanced Sampling Strategies (PMAT-802)
//!
//! Extracted from generate/mod.rs - Advanced sampling algorithms.
//!
//! ## Contents
//! - Stop sequence detection
//! - Repetition, presence/frequency penalties
//! - Min-p, Mirostat, TFS, typical sampling
//! - DRY, XTC, Eta sampling
//! - Token healing

use crate::tensor::Tensor;
use crate::error::Result;
use super::{
    GenerationConfig, apply_temperature, sample_greedy, sample_token,
};
use std::collections::HashMap;

// ==================== Advanced Sampling Features ====================

use serde::{Deserialize, Serialize};

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
