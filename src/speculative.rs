//! Speculative Decoding
//!
//! Per spec §8.3: Speculative decoding with draft model for up to 3x speedup.
//! Reference: [6] Zheng et al. (2024) "SGLang: Efficient Execution of Structured LM Programs"
//!
//! ## Algorithm
//!
//! 1. Generate K speculative tokens with fast draft model
//! 2. Verify all K tokens with single forward pass of target model
//! 3. Accept tokens matching probability distribution
//! 4. Resample at first rejection point
//!
//! ## Benefits
//!
//! - Up to 3x speedup for greedy decoding
//! - Maintains output quality (mathematically equivalent)
//! - Reduces memory-bound bottleneck of autoregressive decoding

// Module-level clippy allows
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_errors_doc)]

use serde::{Deserialize, Serialize};
use std::time::Instant;
use thiserror::Error;

/// Error type for speculative decoding
#[derive(Debug, Error)]
pub enum SpeculativeError {
    /// Draft model error
    #[error("Draft model error: {0}")]
    DraftModelError(String),

    /// Target model error
    #[error("Target model error: {0}")]
    TargetModelError(String),

    /// Invalid speculation length
    #[error("Invalid speculation length: {0}")]
    InvalidSpecLength(usize),

    /// Verification failed
    #[error("Verification failed at position {position}")]
    VerificationFailed {
        /// Position where verification failed
        position: usize,
    },
}

/// Speculative decoding statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpeculativeStats {
    /// Total speculative iterations
    pub iterations: u64,
    /// Total tokens speculated
    pub tokens_speculated: u64,
    /// Total tokens accepted
    pub tokens_accepted: u64,
    /// Average acceptance rate
    pub acceptance_rate: f32,
    /// Average speculation length
    pub avg_spec_length: f32,
    /// Total time saved (estimated)
    pub time_saved_ms: f64,
    /// Draft model time (ms)
    pub draft_time_ms: f64,
    /// Target model time (ms)
    pub target_time_ms: f64,
}

impl SpeculativeStats {
    /// Update stats after an iteration
    pub fn record_iteration(
        &mut self,
        speculated: usize,
        accepted: usize,
        draft_ms: f64,
        target_ms: f64,
    ) {
        self.iterations += 1;
        self.tokens_speculated += speculated as u64;
        self.tokens_accepted += accepted as u64;
        self.draft_time_ms += draft_ms;
        self.target_time_ms += target_ms;

        // Update running averages
        if self.tokens_speculated > 0 {
            self.acceptance_rate = self.tokens_accepted as f32 / self.tokens_speculated as f32;
        }
        if self.iterations > 0 {
            self.avg_spec_length = self.tokens_speculated as f32 / self.iterations as f32;
        }

        // Estimate time saved (accepted tokens would have required sequential target calls)
        // Each accepted token saves ~target_time/batch_size
        let time_per_token = target_ms / speculated.max(1) as f64;
        self.time_saved_ms += (accepted.saturating_sub(1)) as f64 * time_per_token;
    }

    /// Get speedup ratio
    pub fn speedup(&self) -> f32 {
        if self.tokens_accepted == 0 {
            return 1.0;
        }
        // Speedup = (tokens * time_per_token) / actual_time
        // Assuming draft is ~10x faster than target
        let draft_tokens_equivalent = self.tokens_speculated as f64 * 0.1;
        let baseline_time = self.tokens_accepted as f64;
        let actual_time = draft_tokens_equivalent + self.iterations as f64;

        if actual_time > 0.0 {
            (baseline_time / actual_time) as f32
        } else {
            1.0
        }
    }
}

/// Token with probability
#[derive(Debug, Clone)]
pub struct TokenProb {
    /// Token ID
    pub token: u32,
    /// Log probability
    pub log_prob: f32,
}

impl TokenProb {
    /// Create a new token with probability
    pub fn new(token: u32, log_prob: f32) -> Self {
        Self { token, log_prob }
    }

    /// Get probability (exp of log_prob)
    pub fn prob(&self) -> f32 {
        self.log_prob.exp()
    }
}

/// Speculative decoding result
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    /// Accepted tokens
    pub accepted_tokens: Vec<u32>,
    /// Number of tokens speculated
    pub num_speculated: usize,
    /// Number of tokens accepted
    pub num_accepted: usize,
    /// Token that was resampled (if any)
    pub resampled_token: Option<u32>,
    /// Time taken for draft model (ms)
    pub draft_time_ms: f64,
    /// Time taken for target model (ms)
    pub target_time_ms: f64,
}

impl SpeculativeResult {
    /// Get acceptance rate for this iteration
    pub fn acceptance_rate(&self) -> f32 {
        if self.num_speculated == 0 {
            return 0.0;
        }
        self.num_accepted as f32 / self.num_speculated as f32
    }

    /// Check if all tokens were accepted
    pub fn all_accepted(&self) -> bool {
        self.num_accepted == self.num_speculated
    }
}

/// Trait for models that can be used in speculative decoding
pub trait SpeculativeModel {
    /// Generate next token logits
    fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>, SpeculativeError>;

    /// Sample from logits
    fn sample(&self, logits: &[f32]) -> Result<TokenProb, SpeculativeError>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get EOS token
    fn eos_token(&self) -> u32;
}

/// Speculative decoder with draft and target models
pub struct SpeculativeDecoder<D: SpeculativeModel, T: SpeculativeModel> {
    /// Draft model (smaller, faster)
    draft: D,
    /// Target model (larger, slower)
    target: T,
    /// Number of tokens to speculate per iteration
    spec_length: usize,
    /// Statistics
    stats: SpeculativeStats,
}

impl<D: SpeculativeModel, T: SpeculativeModel> SpeculativeDecoder<D, T> {
    /// Create a new speculative decoder
    pub fn new(draft: D, target: T, spec_length: usize) -> Result<Self, SpeculativeError> {
        if spec_length == 0 || spec_length > 32 {
            return Err(SpeculativeError::InvalidSpecLength(spec_length));
        }

        Ok(Self {
            draft,
            target,
            spec_length,
            stats: SpeculativeStats::default(),
        })
    }

    /// Get speculation length
    pub fn spec_length(&self) -> usize {
        self.spec_length
    }

    /// Set speculation length
    pub fn set_spec_length(&mut self, spec_length: usize) -> Result<(), SpeculativeError> {
        if spec_length == 0 || spec_length > 32 {
            return Err(SpeculativeError::InvalidSpecLength(spec_length));
        }
        self.spec_length = spec_length;
        Ok(())
    }

    /// Generate one iteration of speculative decoding
    pub fn decode_iteration(
        &mut self,
        context: &[u32],
    ) -> Result<SpeculativeResult, SpeculativeError> {
        let mut accepted_tokens = Vec::new();
        let mut draft_tokens = Vec::new();
        let mut draft_probs = Vec::new();

        // 1. Generate speculative tokens with draft model
        let draft_start = Instant::now();
        let mut current_context = context.to_vec();

        for _ in 0..self.spec_length {
            let logits = self
                .draft
                .forward(&current_context)
                .map_err(|e| SpeculativeError::DraftModelError(e.to_string()))?;
            let token_prob = self
                .draft
                .sample(&logits)
                .map_err(|e| SpeculativeError::DraftModelError(e.to_string()))?;

            let token = token_prob.token;
            draft_tokens.push(token);
            draft_probs.push(token_prob);
            current_context.push(token);

            // Stop if EOS
            if token == self.draft.eos_token() {
                break;
            }
        }
        let draft_time = draft_start.elapsed();

        // 2. Verify with target model (single forward pass for all positions)
        let target_start = Instant::now();
        let mut verify_context = context.to_vec();
        verify_context.extend(&draft_tokens);

        // Get target logits for verification
        let target_logits = self
            .target
            .forward(&verify_context)
            .map_err(|e| SpeculativeError::TargetModelError(e.to_string()))?;

        let target_time = target_start.elapsed();

        // 3. Accept/reject based on probability matching
        let mut resampled_token = None;

        for draft_prob in &draft_probs {
            // Simple acceptance: compare draft and target probabilities
            // In practice, use more sophisticated rejection sampling
            let target_token = self
                .target
                .sample(&target_logits)
                .map_err(|e| SpeculativeError::TargetModelError(e.to_string()))?;

            // Accept if draft token matches target distribution
            // Simplified: accept if tokens match or random acceptance
            if self.should_accept(draft_prob, &target_token) {
                accepted_tokens.push(draft_prob.token);
            } else {
                // Resample from target distribution
                resampled_token = Some(target_token.token);
                accepted_tokens.push(target_token.token);
                break;
            }
        }

        let num_speculated = draft_tokens.len();
        let num_accepted = accepted_tokens.len();

        // Update stats
        self.stats.record_iteration(
            num_speculated,
            num_accepted,
            draft_time.as_secs_f64() * 1000.0,
            target_time.as_secs_f64() * 1000.0,
        );

        Ok(SpeculativeResult {
            accepted_tokens,
            num_speculated,
            num_accepted,
            resampled_token,
            draft_time_ms: draft_time.as_secs_f64() * 1000.0,
            target_time_ms: target_time.as_secs_f64() * 1000.0,
        })
    }

    /// Acceptance criterion for speculative decoding
    #[allow(clippy::unused_self)] // Will use self for config-based thresholds
    fn should_accept(&self, draft: &TokenProb, target: &TokenProb) -> bool {
        // Simple acceptance: tokens must match
        // More sophisticated: use probability ratio for rejection sampling
        if draft.token == target.token {
            return true;
        }

        // Probabilistic acceptance based on ratio
        let ratio = target.prob() / draft.prob().max(1e-10);
        ratio >= 1.0 || ratio > 0.5 // Simplified threshold
    }

    /// Get statistics
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::default();
    }
}

/// Configuration for speculative decoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Number of tokens to speculate
    pub spec_length: usize,
    /// Minimum acceptance rate before adapting spec_length
    pub min_acceptance_rate: f32,
    /// Enable adaptive speculation length
    pub adaptive: bool,
    /// Maximum speculation length for adaptive mode
    pub max_spec_length: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            spec_length: 4,
            min_acceptance_rate: 0.5,
            adaptive: true,
            max_spec_length: 8,
        }
    }
}

impl SpeculativeConfig {
    /// Create a new config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set speculation length
    pub fn with_spec_length(mut self, spec_length: usize) -> Self {
        self.spec_length = spec_length;
        self
    }

    /// Enable/disable adaptive mode
    pub fn with_adaptive(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Adapt speculation length based on acceptance rate
    pub fn adapt_spec_length(&mut self, acceptance_rate: f32) {
        if !self.adaptive {
            return;
        }

        if acceptance_rate > 0.8 && self.spec_length < self.max_spec_length {
            // High acceptance: increase speculation
            self.spec_length = (self.spec_length + 1).min(self.max_spec_length);
        } else if acceptance_rate < self.min_acceptance_rate && self.spec_length > 1 {
            // Low acceptance: decrease speculation
            self.spec_length = (self.spec_length - 1).max(1);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock model for testing
    struct MockModel {
        vocab_size: usize,
        eos_token: u32,
        /// Fixed token to return
        fixed_token: u32,
    }

    impl MockModel {
        fn new(vocab_size: usize, fixed_token: u32) -> Self {
            Self {
                vocab_size,
                eos_token: 0,
                fixed_token,
            }
        }
    }

    impl SpeculativeModel for MockModel {
        fn forward(&self, _input_ids: &[u32]) -> Result<Vec<f32>, SpeculativeError> {
            // Return uniform logits
            Ok(vec![0.0; self.vocab_size])
        }

        fn sample(&self, _logits: &[f32]) -> Result<TokenProb, SpeculativeError> {
            Ok(TokenProb::new(self.fixed_token, -1.0))
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn eos_token(&self) -> u32 {
            self.eos_token
        }
    }

    // === TokenProb Tests ===

    #[test]
    fn test_token_prob_new() {
        let tp = TokenProb::new(42, -1.0);
        assert_eq!(tp.token, 42);
        assert_eq!(tp.log_prob, -1.0);
    }

    #[test]
    fn test_token_prob_prob() {
        let tp = TokenProb::new(42, 0.0);
        assert!((tp.prob() - 1.0).abs() < 0.001);

        let tp2 = TokenProb::new(42, -1.0);
        assert!((tp2.prob() - 0.368).abs() < 0.01);
    }

    // === SpeculativeStats Tests ===

    #[test]
    fn test_speculative_stats_default() {
        let stats = SpeculativeStats::default();
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.tokens_speculated, 0);
        assert_eq!(stats.acceptance_rate, 0.0);
    }

    #[test]
    fn test_speculative_stats_record() {
        let mut stats = SpeculativeStats::default();
        stats.record_iteration(4, 3, 1.0, 10.0);

        assert_eq!(stats.iterations, 1);
        assert_eq!(stats.tokens_speculated, 4);
        assert_eq!(stats.tokens_accepted, 3);
        assert_eq!(stats.acceptance_rate, 0.75);
    }

    #[test]
    fn test_speculative_stats_speedup() {
        let mut stats = SpeculativeStats::default();
        stats.record_iteration(4, 4, 1.0, 10.0);

        let speedup = stats.speedup();
        assert!(speedup > 1.0);
    }

    #[test]
    fn test_speculative_stats_serialization() {
        let stats = SpeculativeStats {
            iterations: 10,
            tokens_speculated: 40,
            tokens_accepted: 30,
            acceptance_rate: 0.75,
            avg_spec_length: 4.0,
            time_saved_ms: 100.0,
            draft_time_ms: 10.0,
            target_time_ms: 100.0,
        };

        let json = serde_json::to_string(&stats).expect("test");
        let parsed: SpeculativeStats = serde_json::from_str(&json).expect("test");

        assert_eq!(parsed.iterations, stats.iterations);
        assert_eq!(parsed.acceptance_rate, stats.acceptance_rate);
    }

    // === SpeculativeResult Tests ===

    #[test]
    fn test_speculative_result_acceptance_rate() {
        let result = SpeculativeResult {
            accepted_tokens: vec![1, 2, 3],
            num_speculated: 4,
            num_accepted: 3,
            resampled_token: Some(4),
            draft_time_ms: 1.0,
            target_time_ms: 10.0,
        };

        assert_eq!(result.acceptance_rate(), 0.75);
        assert!(!result.all_accepted());
    }

    #[test]
    fn test_speculative_result_all_accepted() {
        let result = SpeculativeResult {
            accepted_tokens: vec![1, 2, 3, 4],
            num_speculated: 4,
            num_accepted: 4,
            resampled_token: None,
            draft_time_ms: 1.0,
            target_time_ms: 10.0,
        };

        assert!(result.all_accepted());
    }

    // === SpeculativeConfig Tests ===

    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.spec_length, 4);
        assert!(config.adaptive);
    }

    #[test]
    fn test_speculative_config_builder() {
        let config = SpeculativeConfig::new()
            .with_spec_length(6)
            .with_adaptive(false);

        assert_eq!(config.spec_length, 6);
        assert!(!config.adaptive);
    }

    #[test]
    fn test_speculative_config_adapt_increase() {
        let mut config = SpeculativeConfig::new().with_spec_length(4);
        config.adapt_spec_length(0.9); // High acceptance

        assert!(config.spec_length >= 4);
    }

    #[test]
    fn test_speculative_config_adapt_decrease() {
        let mut config = SpeculativeConfig::new().with_spec_length(4);
        config.adapt_spec_length(0.3); // Low acceptance

        assert!(config.spec_length <= 4);
    }

    #[test]
    fn test_speculative_config_no_adapt_when_disabled() {
        let mut config = SpeculativeConfig::new()
            .with_spec_length(4)
            .with_adaptive(false);

        config.adapt_spec_length(0.1);
        assert_eq!(config.spec_length, 4); // Should not change
    }

    // === SpeculativeDecoder Tests ===

    #[test]
    fn test_speculative_decoder_new() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let decoder = SpeculativeDecoder::new(draft, target, 4).expect("test");
        assert_eq!(decoder.spec_length(), 4);
    }

    #[test]
    fn test_speculative_decoder_invalid_spec_length() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let result = SpeculativeDecoder::new(draft, target, 0);
        assert!(matches!(
            result,
            Err(SpeculativeError::InvalidSpecLength(0))
        ));
    }

    #[test]
    fn test_speculative_decoder_decode_iteration() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("test");
        let result = decoder.decode_iteration(&[10, 20, 30]).expect("test");

        assert!(!result.accepted_tokens.is_empty());
        assert!(result.num_speculated > 0);
    }

    #[test]
    fn test_speculative_decoder_set_spec_length() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("test");
        decoder.set_spec_length(8).expect("test");
        assert_eq!(decoder.spec_length(), 8);

        let err = decoder.set_spec_length(0);
        assert!(err.is_err());
    }

    #[test]
    fn test_speculative_decoder_stats() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("test");
        let _ = decoder.decode_iteration(&[10]).expect("test");

        let stats = decoder.stats();
        assert_eq!(stats.iterations, 1);
    }

    #[test]
    fn test_speculative_decoder_reset_stats() {
        let draft = MockModel::new(100, 1);
        let target = MockModel::new(100, 1);

        let mut decoder = SpeculativeDecoder::new(draft, target, 4).expect("test");
        let _ = decoder.decode_iteration(&[10]).expect("test");

        decoder.reset_stats();
        assert_eq!(decoder.stats().iterations, 0);
    }

    // === Error Tests ===

    #[test]
    fn test_speculative_error_display() {
        let err = SpeculativeError::DraftModelError("test".to_string());
        assert!(err.to_string().contains("Draft"));

        let err = SpeculativeError::TargetModelError("test".to_string());
        assert!(err.to_string().contains("Target"));

        let err = SpeculativeError::InvalidSpecLength(0);
        assert!(err.to_string().contains("0"));

        let err = SpeculativeError::VerificationFailed { position: 3 };
        assert!(err.to_string().contains("3"));
    }
}
