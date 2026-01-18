//! Constrained sampling with logit processing for grammar-enforced generation.
//!
//! This module provides the [`LogitProcessor`] trait for constraining LLM token generation,
//! along with implementations for JSON grammar enforcement and hybrid sampling that
//! dynamically switches between free-form and grammar-constrained modes.
//!
//! # Architecture
//!
//! The sampling pipeline integrates logit processors between logit computation and token sampling:
//!
//! ```text
//! logits → LogitProcessor::process() → masked_logits → sample_token()
//! ```
//!
//! # Example
//!
//! ```rust
//! use realizar::sampling::{LogitProcessor, JsonGrammarProcessor};
//! use realizar::grammar::{Grammar, GrammarRule, GrammarAlternative, GrammarElement};
//! use std::collections::HashMap;
//!
//! // Create a simple grammar for "true" or "false"
//! let mut grammar = Grammar::with_root("root");
//! grammar.add_rule(GrammarRule::new("root", vec![
//!     GrammarAlternative::new(vec![
//!         GrammarElement::Char('t'), GrammarElement::Char('r'),
//!         GrammarElement::Char('u'), GrammarElement::Char('e'),
//!     ]),
//!     GrammarAlternative::new(vec![
//!         GrammarElement::Char('f'), GrammarElement::Char('a'),
//!         GrammarElement::Char('l'), GrammarElement::Char('s'), GrammarElement::Char('e'),
//!     ]),
//! ]));
//!
//! // Create token vocabulary mapping
//! let mut vocab: HashMap<u32, String> = HashMap::new();
//! vocab.insert(0, "t".to_string());
//! vocab.insert(1, "r".to_string());
//! vocab.insert(2, "u".to_string());
//! vocab.insert(3, "e".to_string());
//! vocab.insert(4, "f".to_string());
//! vocab.insert(5, "a".to_string());
//! vocab.insert(6, "l".to_string());
//! vocab.insert(7, "s".to_string());
//!
//! // Create processor
//! let mut processor = JsonGrammarProcessor::new(grammar, vocab, 99).expect("valid grammar");
//!
//! // Process logits to mask invalid tokens
//! let mut logits = vec![1.0f32; 100];
//! processor.process(&[], &mut logits);
//!
//! // Only 't' and 'f' should have finite logits initially
//! assert!(logits[0].is_finite()); // 't' allowed
//! assert!(logits[4].is_finite()); // 'f' allowed
//! assert!(logits[1].is_infinite()); // 'r' not allowed initially
//! ```

use crate::error::Result;
use crate::grammar::{Grammar, GrammarTokenMasker, ToolCall, ToolCallFormat, ToolCallParser, ToolDefinition};
use std::collections::HashMap;
use std::fmt::Debug;

// =============================================================================
// LOGIT PROCESSOR TRAIT
// =============================================================================

/// Trait for processing logits before token sampling.
///
/// Logit processors modify the raw logit values from the model to enforce
/// constraints, apply biases, or implement custom sampling strategies.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to support concurrent inference.
///
/// # Example
///
/// ```rust
/// use realizar::sampling::LogitProcessor;
///
/// struct TemperatureProcessor {
///     temperature: f32,
/// }
///
/// impl LogitProcessor for TemperatureProcessor {
///     fn process(&mut self, _input_ids: &[u32], logits: &mut [f32]) {
///         for logit in logits.iter_mut() {
///             *logit /= self.temperature;
///         }
///     }
///
///     fn reset(&mut self) {
///         // No state to reset
///     }
///     
///     fn is_complete(&self) -> bool {
///         false // Temperature processor is never "complete"
///     }
/// }
/// ```
pub trait LogitProcessor: Send + Sync + Debug {
    /// Process logits before sampling.
    ///
    /// This method is called with the current input token IDs and mutable logits.
    /// Implementations should modify `logits` in-place to apply their constraints.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Tokens generated so far in the sequence
    /// * `logits` - Mutable slice of logits to modify (vocab_size length)
    fn process(&mut self, input_ids: &[u32], logits: &mut [f32]);

    /// Reset processor state for a new generation sequence.
    ///
    /// Called when starting a new generation to clear any accumulated state.
    fn reset(&mut self);
    
    /// Check if generation is complete according to this processor.
    ///
    /// For grammar processors, returns `true` when a valid end state is reached.
    /// For other processors, may return `false` always.
    fn is_complete(&self) -> bool;
}

// =============================================================================
// JSON GRAMMAR PROCESSOR
// =============================================================================

/// Logit processor that enforces JSON grammar constraints.
///
/// Uses a [`GrammarTokenMasker`] to ensure generated tokens form valid JSON
/// according to the provided grammar. Invalid tokens are masked with negative infinity.
///
/// # Example
///
/// ```rust
/// use realizar::sampling::JsonGrammarProcessor;
/// use realizar::grammar::{Grammar, GrammarRule, GrammarAlternative, GrammarElement};
/// use std::collections::HashMap;
///
/// // Grammar for JSON boolean
/// let mut grammar = Grammar::with_root("bool");
/// grammar.add_rule(GrammarRule::new("bool", vec![
///     GrammarAlternative::new(vec![
///         GrammarElement::Char('t'), GrammarElement::Char('r'),
///         GrammarElement::Char('u'), GrammarElement::Char('e'),
///     ]),
/// ]));
///
/// let vocab = HashMap::from([
///     (0u32, "t".to_string()),
///     (1u32, "r".to_string()),
///     (2u32, "u".to_string()),
///     (3u32, "e".to_string()),
/// ]);
///
/// let processor = JsonGrammarProcessor::new(grammar, vocab, 99);
/// assert!(processor.is_ok());
/// ```
#[derive(Debug)]
pub struct JsonGrammarProcessor {
    /// Token masker for grammar-constrained generation
    masker: GrammarTokenMasker,
    /// Vocabulary size for bounds checking
    vocab_size: usize,
}

impl JsonGrammarProcessor {
    /// Create a new JSON grammar processor.
    ///
    /// # Arguments
    ///
    /// * `grammar` - The grammar to enforce
    /// * `token_strings` - Mapping from token IDs to their string representations
    /// * `eos_token_id` - End-of-sequence token ID
    ///
    /// # Errors
    ///
    /// Returns `RealizarError::InvalidConfiguration` if the grammar is invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use realizar::sampling::JsonGrammarProcessor;
    /// use realizar::grammar::{Grammar, GrammarRule, GrammarAlternative, GrammarElement};
    /// use std::collections::HashMap;
    ///
    /// let mut grammar = Grammar::with_root("root");
    /// grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('x')]));
    ///
    /// let vocab: HashMap<u32, String> = HashMap::from([(0, "x".to_string())]);
    /// let processor = JsonGrammarProcessor::new(grammar, vocab, 1)?;
    /// # Ok::<(), realizar::RealizarError>(())
    /// ```
    pub fn new(
        grammar: Grammar,
        token_strings: HashMap<u32, String>,
        eos_token_id: u32,
    ) -> Result<Self> {
        let vocab_size = token_strings.keys().max().map_or(0, |&m| m as usize + 1);
        let masker = GrammarTokenMasker::new(grammar, token_strings, eos_token_id)?;
        
        Ok(Self { masker, vocab_size })
    }

    /// Advance the grammar state with a token.
    ///
    /// Call this after sampling a token to update the internal state.
    ///
    /// # Returns
    ///
    /// `true` if the token was valid and state was updated, `false` otherwise.
    pub fn advance_token(&mut self, token_id: u32) -> bool {
        self.masker.advance_token(token_id)
    }
}

impl LogitProcessor for JsonGrammarProcessor {
    fn process(&mut self, _input_ids: &[u32], logits: &mut [f32]) {
        let mask = self.masker.get_mask();
        mask.apply_to_logits(logits);
        
        // Handle EOS token
        let eos_id = self.masker.eos_token_id() as usize;
        if eos_id < logits.len() && !mask.allow_eos {
            logits[eos_id] = f32::NEG_INFINITY;
        }
    }

    fn reset(&mut self) {
        self.masker.reset();
    }
    
    fn is_complete(&self) -> bool {
        self.masker.is_complete()
    }
}

// =============================================================================
// TOOL CALL DETECTOR
// =============================================================================

/// Detects tool call patterns in generated text.
///
/// This detector identifies when a model is starting a tool call based on
/// the configured format (OpenAI JSON, Anthropic XML, Hermes XML, or Groq XML).
#[derive(Debug, Clone)]
pub struct ToolCallDetector {
    /// Tool call format to detect
    format: ToolCallFormat,
    /// Buffer of recent tokens for pattern matching
    buffer: String,
    /// Maximum buffer size before truncation
    max_buffer_size: usize,
}

impl ToolCallDetector {
    /// Create a new tool call detector.
    ///
    /// # Arguments
    ///
    /// * `format` - The tool call format to detect
    ///
    /// # Example
    ///
    /// ```rust
    /// use realizar::sampling::ToolCallDetector;
    /// use realizar::grammar::ToolCallFormat;
    ///
    /// let detector = ToolCallDetector::new(ToolCallFormat::Hermes);
    /// ```
    #[must_use]
    pub fn new(format: ToolCallFormat) -> Self {
        Self {
            format,
            buffer: String::new(),
            max_buffer_size: 256,
        }
    }
    
    /// Create a detector for Groq tool-use format (uses Hermes-style `<tool_call>` tags).
    #[must_use]
    pub fn groq() -> Self {
        Self::new(ToolCallFormat::Hermes)
    }

    /// Add a token to the detection buffer.
    pub fn add_token(&mut self, token_str: &str) {
        self.buffer.push_str(token_str);
        
        // Truncate buffer if too large
        if self.buffer.len() > self.max_buffer_size {
            let start = self.buffer.len() - self.max_buffer_size / 2;
            self.buffer = self.buffer[start..].to_string();
        }
    }

    /// Check if a tool call start pattern has been detected.
    ///
    /// # Returns
    ///
    /// `true` if the buffer contains a tool call start pattern.
    #[must_use]
    pub fn detect_tool_call_start(&self) -> bool {
        match self.format {
            ToolCallFormat::OpenAI => {
                // Look for JSON object with "name" field
                self.buffer.contains(r#"{"name":"#) || 
                self.buffer.contains(r#"{"name" :"#) ||
                self.buffer.contains(r#"{ "name":"#)
            },
            ToolCallFormat::Anthropic => {
                self.buffer.contains("<tool_use>")
            },
            ToolCallFormat::Hermes => {
                // Groq model uses this format
                self.buffer.contains("<tool_call>")
            },
        }
    }

    /// Get the current buffer contents.
    #[must_use]
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// =============================================================================
// HYBRID SAMPLER
// =============================================================================

/// Sampling mode for the hybrid sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingMode {
    /// Free-form text generation (no constraints)
    FreeForm,
    /// Grammar-constrained JSON generation
    JsonConstrained,
    /// Detection phase (first few tokens to determine mode)
    Detecting,
}

/// Hybrid sampler that switches between free-form and grammar-constrained generation.
///
/// The sampler starts in detection mode, generating the first few tokens freely
/// to determine if the model is outputting a tool call. Once detected, it switches
/// to grammar-constrained mode to ensure valid JSON output.
///
/// # Detection Strategy
///
/// 1. Generate first `detection_tokens` freely
/// 2. Check for tool call patterns (e.g., `<tool_call>`, `{"name":`)
/// 3. If pattern found, switch to JSON grammar mode
/// 4. Otherwise, continue in free-form mode
///
/// # Example
///
/// ```rust
/// use realizar::sampling::{HybridSampler, SamplingMode};
/// use realizar::grammar::{ToolDefinition, ToolParameter, ToolCallFormat};
///
/// // Define available tools
/// let tools = vec![
///     ToolDefinition::new(
///         "get_weather",
///         "Get current weather",
///         vec![ToolParameter::required_string("location", "City name")],
///     ),
/// ];
///
/// // Create vocabulary (simplified)
/// let vocab: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
///
/// // Create hybrid sampler
/// let sampler = HybridSampler::new(tools, vocab, 128009, ToolCallFormat::Hermes);
/// assert_eq!(sampler.mode(), SamplingMode::Detecting);
/// ```
#[derive(Debug)]
pub struct HybridSampler {
    /// Current sampling mode
    mode: SamplingMode,
    /// JSON grammar processor (activated when tool call detected)
    json_processor: Option<JsonGrammarProcessor>,
    /// Tool call detector
    detector: ToolCallDetector,
    /// Tools for grammar generation
    tools: Vec<ToolDefinition>,
    /// Token vocabulary
    vocab: HashMap<u32, String>,
    /// EOS token ID
    eos_token_id: u32,
    /// Number of tokens for detection phase
    detection_tokens: usize,
    /// Tokens generated so far
    tokens_generated: usize,
    /// Buffer of generated text
    generated_text: String,
    /// Tool call format
    format: ToolCallFormat,
}

impl HybridSampler {
    /// Create a new hybrid sampler.
    ///
    /// # Arguments
    ///
    /// * `tools` - Available tool definitions
    /// * `vocab` - Token ID to string mapping
    /// * `eos_token_id` - End-of-sequence token ID
    /// * `format` - Tool call format to detect and enforce
    ///
    /// # Example
    ///
    /// ```rust
    /// use realizar::sampling::HybridSampler;
    /// use realizar::grammar::{ToolDefinition, ToolCallFormat};
    /// use std::collections::HashMap;
    ///
    /// let tools = vec![ToolDefinition::new("search", "Search web", vec![])];
    /// let vocab = HashMap::new();
    /// let sampler = HybridSampler::new(tools, vocab, 128009, ToolCallFormat::Hermes);
    /// ```
    #[must_use]
    pub fn new(
        tools: Vec<ToolDefinition>,
        vocab: HashMap<u32, String>,
        eos_token_id: u32,
        format: ToolCallFormat,
    ) -> Self {
        Self {
            mode: SamplingMode::Detecting,
            json_processor: None,
            detector: ToolCallDetector::new(format),
            tools,
            vocab,
            eos_token_id,
            detection_tokens: 5, // Check first 5 tokens
            tokens_generated: 0,
            generated_text: String::new(),
            format,
        }
    }
    
    /// Create a hybrid sampler configured for Groq tool-use format.
    ///
    /// The Groq model uses `<tool_call>` XML tags (Hermes format).
    #[must_use]
    pub fn groq(
        tools: Vec<ToolDefinition>,
        vocab: HashMap<u32, String>,
        eos_token_id: u32,
    ) -> Self {
        Self::new(tools, vocab, eos_token_id, ToolCallFormat::Hermes)
    }

    /// Set the number of tokens for the detection phase.
    #[must_use]
    pub fn with_detection_tokens(mut self, n: usize) -> Self {
        self.detection_tokens = n;
        self
    }

    /// Get the current sampling mode.
    #[must_use]
    pub fn mode(&self) -> SamplingMode {
        self.mode
    }

    /// Get the generated text so far.
    #[must_use]
    pub fn generated_text(&self) -> &str {
        &self.generated_text
    }

    /// Record a generated token.
    ///
    /// Call this after sampling each token to update the sampler state.
    pub fn record_token(&mut self, token_id: u32) {
        if let Some(token_str) = self.vocab.get(&token_id) {
            self.generated_text.push_str(token_str);
            self.detector.add_token(token_str);
        }
        
        self.tokens_generated += 1;

        // Check for mode transition during detection phase
        if self.mode == SamplingMode::Detecting {
            if self.detector.detect_tool_call_start() {
                self.switch_to_json_mode();
            } else if self.tokens_generated >= self.detection_tokens {
                // No tool call detected, continue free-form
                self.mode = SamplingMode::FreeForm;
            }
        }

        // If in JSON mode, advance the grammar processor
        if let Some(ref mut processor) = self.json_processor {
            processor.advance_token(token_id);
        }
    }

    /// Switch to JSON-constrained mode.
    fn switch_to_json_mode(&mut self) {
        use crate::grammar::generate_tool_grammar;
        
        let grammar = generate_tool_grammar(&self.tools);
        if let Ok(processor) = JsonGrammarProcessor::new(
            grammar,
            self.vocab.clone(),
            self.eos_token_id,
        ) {
            self.json_processor = Some(processor);
            self.mode = SamplingMode::JsonConstrained;
        }
    }

    /// Parse tool calls from the generated text.
    ///
    /// # Returns
    ///
    /// Vector of parsed tool calls, empty if none found or parsing failed.
    #[must_use]
    pub fn parse_tool_calls(&self) -> Vec<ToolCall> {
        let mut parser = ToolCallParser::new(self.tools.clone()).with_format(self.format);
        parser.parse(&self.generated_text)
    }
}

impl LogitProcessor for HybridSampler {
    fn process(&mut self, input_ids: &[u32], logits: &mut [f32]) {
        match self.mode {
            SamplingMode::FreeForm | SamplingMode::Detecting => {
                // No constraints in free-form or detection mode
            },
            SamplingMode::JsonConstrained => {
                if let Some(ref mut processor) = self.json_processor {
                    processor.process(input_ids, logits);
                }
            },
        }
    }

    fn reset(&mut self) {
        self.mode = SamplingMode::Detecting;
        self.json_processor = None;
        self.detector.reset();
        self.tokens_generated = 0;
        self.generated_text.clear();
    }
    
    fn is_complete(&self) -> bool {
        match self.mode {
            SamplingMode::JsonConstrained => {
                self.json_processor.as_ref().map_or(false, |p| p.is_complete())
            },
            _ => false,
        }
    }
}

// =============================================================================
// COMPOSITE LOGIT PROCESSOR
// =============================================================================

/// A composite processor that applies multiple logit processors in sequence.
///
/// # Example
///
/// ```rust
/// use realizar::sampling::{LogitProcessorChain, LogitProcessor};
///
/// let chain = LogitProcessorChain::new();
/// // Add processors with chain.push(processor)
/// ```
#[derive(Debug, Default)]
pub struct LogitProcessorChain {
    processors: Vec<Box<dyn LogitProcessor>>,
}

impl LogitProcessorChain {
    /// Create a new empty processor chain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    /// Add a processor to the chain.
    pub fn push(&mut self, processor: Box<dyn LogitProcessor>) {
        self.processors.push(processor);
    }

    /// Get the number of processors in the chain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.processors.len()
    }

    /// Check if the chain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }
}

impl LogitProcessor for LogitProcessorChain {
    fn process(&mut self, input_ids: &[u32], logits: &mut [f32]) {
        for processor in &mut self.processors {
            processor.process(input_ids, logits);
        }
    }

    fn reset(&mut self) {
        for processor in &mut self.processors {
            processor.reset();
        }
    }
    
    fn is_complete(&self) -> bool {
        // Chain is complete if any processor is complete
        self.processors.iter().any(|p| p.is_complete())
    }
}

// =============================================================================
// REPETITION PENALTY PROCESSOR
// =============================================================================

/// Logit processor that applies repetition penalty to discourage repeated tokens.
///
/// Implements the repetition penalty from [Keskar et al. 2019](https://arxiv.org/abs/1909.05858).
/// Logits for previously generated tokens are divided by the penalty factor.
///
/// # Example
///
/// ```rust
/// use realizar::sampling::{RepetitionPenaltyProcessor, LogitProcessor};
///
/// let mut processor = RepetitionPenaltyProcessor::new(1.2);
/// let input_ids = vec![1, 2, 3, 1]; // Token 1 appears twice
/// let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
///
/// processor.process(&input_ids, &mut logits);
///
/// // Token 1, 2, 3 should have reduced logits
/// assert!(logits[1] < 2.0);
/// assert!(logits[2] < 3.0);
/// assert!(logits[3] < 4.0);
/// // Token 0 and 4 should be unchanged
/// assert_eq!(logits[0], 1.0);
/// assert_eq!(logits[4], 5.0);
/// ```
#[derive(Debug, Clone)]
pub struct RepetitionPenaltyProcessor {
    /// Penalty factor (> 1.0 reduces probability of repeated tokens)
    penalty: f32,
}

impl RepetitionPenaltyProcessor {
    /// Create a new repetition penalty processor.
    ///
    /// # Arguments
    ///
    /// * `penalty` - Penalty factor (typical range: 1.0 to 1.5)
    #[must_use]
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

impl LogitProcessor for RepetitionPenaltyProcessor {
    fn process(&mut self, input_ids: &[u32], logits: &mut [f32]) {
        for &token_id in input_ids {
            let idx = token_id as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.penalty;
                } else {
                    logits[idx] *= self.penalty;
                }
            }
        }
    }

    fn reset(&mut self) {
        // No state to reset
    }
    
    fn is_complete(&self) -> bool {
        false // Repetition penalty is never "complete"
    }
}

// =============================================================================
// TEMPERATURE PROCESSOR
// =============================================================================

/// Logit processor that applies temperature scaling.
///
/// Temperature controls the randomness of sampling:
/// - `temperature < 1.0`: More deterministic (sharper distribution)
/// - `temperature > 1.0`: More random (flatter distribution)
/// - `temperature = 1.0`: No change
///
/// # Example
///
/// ```rust
/// use realizar::sampling::{TemperatureProcessor, LogitProcessor};
///
/// let mut processor = TemperatureProcessor::new(0.5); // More deterministic
/// let mut logits = vec![1.0, 2.0, 3.0];
///
/// processor.process(&[], &mut logits);
///
/// // Logits should be scaled up (divided by 0.5)
/// assert_eq!(logits, vec![2.0, 4.0, 6.0]);
/// ```
#[derive(Debug, Clone)]
pub struct TemperatureProcessor {
    /// Temperature value (clamped to > 0)
    temperature: f32,
}

impl TemperatureProcessor {
    /// Create a new temperature processor.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature value (clamped to minimum of 0.01)
    #[must_use]
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature: temperature.max(0.01),
        }
    }
}

impl LogitProcessor for TemperatureProcessor {
    fn process(&mut self, _input_ids: &[u32], logits: &mut [f32]) {
        for logit in logits.iter_mut() {
            *logit /= self.temperature;
        }
    }

    fn reset(&mut self) {
        // No state to reset
    }
    
    fn is_complete(&self) -> bool {
        false // Temperature processor is never "complete"
    }
}

// =============================================================================
// TOP-P (NUCLEUS) PROCESSOR
// =============================================================================

/// Logit processor that applies top-p (nucleus) sampling.
///
/// Only keeps tokens with cumulative probability up to `top_p`.
/// This is applied after softmax, so it modifies the effective distribution.
///
/// Note: This processor assumes logits will be converted to probabilities
/// and the actual filtering happens during sampling. This implementation
/// provides the mask for tokens to consider.
#[derive(Debug, Clone)]
pub struct TopPProcessor {
    /// Top-p threshold (0.0 to 1.0)
    top_p: f32,
}

impl TopPProcessor {
    /// Create a new top-p processor.
    ///
    /// # Arguments
    ///
    /// * `top_p` - Cumulative probability threshold (typical: 0.9 to 0.95)
    #[must_use]
    pub fn new(top_p: f32) -> Self {
        Self {
            top_p: top_p.clamp(0.0, 1.0),
        }
    }
    
    /// Get the top-p value.
    #[must_use]
    pub fn top_p(&self) -> f32 {
        self.top_p
    }
}

impl LogitProcessor for TopPProcessor {
    fn process(&mut self, _input_ids: &[u32], logits: &mut [f32]) {
        // Convert to probabilities via softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i, (l - max_logit).exp()))
            .collect();
        
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in &mut probs {
            *p /= sum;
        }
        
        // Sort by probability descending
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Find cutoff
        let mut cumsum = 0.0;
        let mut cutoff_idx = probs.len();
        for (i, (_, prob)) in probs.iter().enumerate() {
            cumsum += prob;
            if cumsum >= self.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        // Mask tokens beyond cutoff
        for (idx, _) in &probs[cutoff_idx..] {
            logits[*idx] = f32::NEG_INFINITY;
        }
    }

    fn reset(&mut self) {
        // No state to reset
    }
    
    fn is_complete(&self) -> bool {
        false // Top-p processor is never "complete"
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::{GrammarAlternative, GrammarElement, GrammarRule};

    fn create_simple_grammar() -> Grammar {
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::new(
            "root",
            vec![
                GrammarAlternative::new(vec![GrammarElement::Char('a')]),
                GrammarAlternative::new(vec![GrammarElement::Char('b')]),
            ],
        ));
        grammar
    }

    fn create_vocab() -> HashMap<u32, String> {
        HashMap::from([
            (0u32, "a".to_string()),
            (1u32, "b".to_string()),
            (2u32, "c".to_string()),
        ])
    }

    #[test]
    fn test_json_grammar_processor_creation() {
        let grammar = create_simple_grammar();
        let vocab = create_vocab();
        
        let processor = JsonGrammarProcessor::new(grammar, vocab, 99);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_json_grammar_processor_masks_invalid() {
        let grammar = create_simple_grammar();
        let vocab = create_vocab();
        
        let mut processor = JsonGrammarProcessor::new(grammar, vocab, 99).expect("valid grammar");
        let mut logits = vec![1.0, 1.0, 1.0];
        
        processor.process(&[], &mut logits);
        
        // Only 'a' and 'b' should be valid
        assert!(logits[0].is_finite()); // 'a' allowed
        assert!(logits[1].is_finite()); // 'b' allowed
        assert!(logits[2].is_infinite()); // 'c' not allowed
    }

    #[test]
    fn test_json_grammar_processor_advance() {
        let grammar = create_simple_grammar();
        let vocab = create_vocab();
        
        let mut processor = JsonGrammarProcessor::new(grammar, vocab, 99).expect("valid grammar");
        
        // Advance with 'a'
        assert!(processor.advance_token(0));
        assert!(processor.is_complete());
    }

    #[test]
    fn test_json_grammar_processor_reset() {
        let grammar = create_simple_grammar();
        let vocab = create_vocab();
        
        let mut processor = JsonGrammarProcessor::new(grammar, vocab, 99).expect("valid grammar");
        
        processor.advance_token(0);
        assert!(processor.is_complete());
        
        processor.reset();
        assert!(!processor.is_complete());
    }

    #[test]
    fn test_tool_call_detector_hermes() {
        let mut detector = ToolCallDetector::new(ToolCallFormat::Hermes);
        
        assert!(!detector.detect_tool_call_start());
        
        detector.add_token("<tool");
        assert!(!detector.detect_tool_call_start());
        
        detector.add_token("_call>");
        assert!(detector.detect_tool_call_start());
    }

    #[test]
    fn test_tool_call_detector_openai() {
        let mut detector = ToolCallDetector::new(ToolCallFormat::OpenAI);
        
        detector.add_token(r#"{"name":"#);
        assert!(detector.detect_tool_call_start());
    }

    #[test]
    fn test_tool_call_detector_anthropic() {
        let mut detector = ToolCallDetector::new(ToolCallFormat::Anthropic);
        
        detector.add_token("<tool_use>");
        assert!(detector.detect_tool_call_start());
    }

    #[test]
    fn test_tool_call_detector_reset() {
        let mut detector = ToolCallDetector::new(ToolCallFormat::Hermes);
        
        detector.add_token("<tool_call>");
        assert!(detector.detect_tool_call_start());
        
        detector.reset();
        assert!(!detector.detect_tool_call_start());
        assert!(detector.buffer().is_empty());
    }

    #[test]
    fn test_hybrid_sampler_creation() {
        let tools = vec![ToolDefinition::new("test", "Test tool", vec![])];
        let vocab = HashMap::new();
        
        let sampler = HybridSampler::new(tools, vocab, 128009, ToolCallFormat::Hermes);
        
        assert_eq!(sampler.mode(), SamplingMode::Detecting);
    }

    #[test]
    fn test_hybrid_sampler_groq() {
        let tools = vec![];
        let vocab = HashMap::new();
        
        let sampler = HybridSampler::groq(tools, vocab, 128009);
        
        assert_eq!(sampler.mode(), SamplingMode::Detecting);
    }

    #[test]
    fn test_hybrid_sampler_detection_phase() {
        let tools = vec![ToolDefinition::new("test", "Test tool", vec![])];
        let vocab = HashMap::from([
            (0u32, "Hello".to_string()),
            (1u32, " world".to_string()),
        ]);
        
        let mut sampler = HybridSampler::new(tools, vocab, 128009, ToolCallFormat::Hermes)
            .with_detection_tokens(3);
        
        // Generate some tokens without tool call pattern
        sampler.record_token(0);
        assert_eq!(sampler.mode(), SamplingMode::Detecting);
        
        sampler.record_token(1);
        assert_eq!(sampler.mode(), SamplingMode::Detecting);
        
        sampler.record_token(0);
        // After 3 tokens without tool call, should switch to free-form
        assert_eq!(sampler.mode(), SamplingMode::FreeForm);
    }

    #[test]
    fn test_hybrid_sampler_reset() {
        let tools = vec![];
        let vocab = HashMap::from([(0u32, "test".to_string())]);
        
        let mut sampler = HybridSampler::new(tools, vocab, 128009, ToolCallFormat::Hermes);
        
        sampler.record_token(0);
        assert!(!sampler.generated_text().is_empty());
        
        sampler.reset();
        assert!(sampler.generated_text().is_empty());
        assert_eq!(sampler.mode(), SamplingMode::Detecting);
    }

    #[test]
    fn test_logit_processor_chain() {
        let mut chain = LogitProcessorChain::new();
        assert!(chain.is_empty());
        
        chain.push(Box::new(TemperatureProcessor::new(1.0)));
        assert_eq!(chain.len(), 1);
        
        chain.push(Box::new(RepetitionPenaltyProcessor::new(1.0)));
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn test_logit_processor_chain_process() {
        let mut chain = LogitProcessorChain::new();
        chain.push(Box::new(TemperatureProcessor::new(0.5)));
        
        let mut logits = vec![1.0, 2.0, 3.0];
        chain.process(&[], &mut logits);
        
        assert_eq!(logits, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_repetition_penalty_processor() {
        let mut processor = RepetitionPenaltyProcessor::new(2.0);
        let input_ids = vec![1, 2];
        let mut logits = vec![1.0, 4.0, 6.0, 8.0];
        
        processor.process(&input_ids, &mut logits);
        
        assert_eq!(logits[0], 1.0); // Unchanged
        assert_eq!(logits[1], 2.0); // Divided by 2
        assert_eq!(logits[2], 3.0); // Divided by 2
        assert_eq!(logits[3], 8.0); // Unchanged
    }

    #[test]
    fn test_repetition_penalty_negative_logits() {
        let mut processor = RepetitionPenaltyProcessor::new(2.0);
        let input_ids = vec![0];
        let mut logits = vec![-2.0, 1.0];
        
        processor.process(&input_ids, &mut logits);
        
        assert_eq!(logits[0], -4.0); // Multiplied by 2 (made more negative)
        assert_eq!(logits[1], 1.0); // Unchanged
    }

    #[test]
    fn test_temperature_processor() {
        let mut processor = TemperatureProcessor::new(2.0);
        let mut logits = vec![2.0, 4.0, 6.0];
        
        processor.process(&[], &mut logits);
        
        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_temperature_processor_clamped() {
        // Temperature should be clamped to minimum 0.01
        let processor = TemperatureProcessor::new(0.0);
        let mut logits = vec![1.0];
        let mut processor_mut = processor;
        processor_mut.process(&[], &mut logits);
        
        // Should not divide by zero
        assert!(logits[0].is_finite());
    }

    #[test]
    fn test_top_p_processor() {
        let mut processor = TopPProcessor::new(0.9);
        assert_eq!(processor.top_p(), 0.9);
        
        // Create logits where one token dominates
        let mut logits = vec![10.0, 0.0, 0.0, 0.0, 0.0];
        processor.process(&[], &mut logits);
        
        // Top token should still be finite
        assert!(logits[0].is_finite());
    }

    #[test]
    fn test_top_p_processor_clamp() {
        let processor = TopPProcessor::new(1.5);
        assert_eq!(processor.top_p(), 1.0);
        
        let processor = TopPProcessor::new(-0.5);
        assert_eq!(processor.top_p(), 0.0);
    }

    #[test]
    fn test_sampling_mode_equality() {
        assert_eq!(SamplingMode::FreeForm, SamplingMode::FreeForm);
        assert_eq!(SamplingMode::JsonConstrained, SamplingMode::JsonConstrained);
        assert_eq!(SamplingMode::Detecting, SamplingMode::Detecting);
        assert_ne!(SamplingMode::FreeForm, SamplingMode::JsonConstrained);
    }
}
