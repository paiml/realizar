//! Tokenizer for text encoding and decoding
//!
//! Implements tokenization for transformer models:
//! - BPE (Byte Pair Encoding) - Used by GPT models
//! - Vocabulary management
//! - Special token handling
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::Tokenizer;
//!
//! let tokenizer = Tokenizer::from_vocab(vocab);
//! let token_ids = tokenizer.encode("Hello, world!")?;
//! let text = tokenizer.decode(&token_ids)?;
//! ```

use std::collections::HashMap;

use crate::error::{RealizarError, Result};

/// Vocabulary mapping between tokens and IDs
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Token to ID mapping
    token_to_id: HashMap<String, u32>,
    /// ID to token mapping
    id_to_token: HashMap<u32, String>,
}

impl Vocabulary {
    /// Create a new vocabulary from token list
    ///
    /// # Arguments
    ///
    /// * `tokens` - List of tokens in order (index = token ID)
    ///
    /// # Errors
    ///
    /// Returns error if tokens list is empty or contains duplicates
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let vocab = Vocabulary::from_tokens(vec![
    ///     "<unk>".to_string(),
    ///     "hello".to_string(),
    ///     "world".to_string(),
    /// ])?;
    /// ```
    pub fn from_tokens(tokens: Vec<String>) -> Result<Self> {
        if tokens.is_empty() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "create_vocabulary".to_string(),
                reason: "Vocabulary cannot be empty".to_string(),
            });
        }

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (id, token) in tokens.into_iter().enumerate() {
            let id = u32::try_from(id).map_err(|_| RealizarError::UnsupportedOperation {
                operation: "convert_token_id".to_string(),
                reason: format!("Token ID {id} exceeds u32 limit"),
            })?;

            if token_to_id.contains_key(&token) {
                return Err(RealizarError::UnsupportedOperation {
                    operation: "create_vocabulary".to_string(),
                    reason: format!("Duplicate token: {token}"),
                });
            }

            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }

        Ok(Self {
            token_to_id,
            id_to_token,
        })
    }

    /// Get token ID for a token
    ///
    /// Returns `None` if token not in vocabulary
    #[must_use]
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token for a token ID
    ///
    /// Returns `None` if ID not in vocabulary
    #[must_use]
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// Get vocabulary size
    #[must_use]
    pub fn size(&self) -> usize {
        self.token_to_id.len()
    }
}

/// Tokenizer for encoding and decoding text
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Vocabulary
    vocab: Vocabulary,
    /// Unknown token ID
    unk_token_id: u32,
}

/// BPE (Byte Pair Encoding) tokenizer
///
/// Implements subword tokenization using byte pair encoding algorithm.
/// Used by GPT-2, GPT-3, and many other models.
#[derive(Debug, Clone)]
pub struct BPETokenizer {
    /// Token to ID mapping
    token_to_id: HashMap<String, u32>,
    /// ID to token mapping
    id_to_token: HashMap<u32, String>,
    /// Merge rules: pairs to merge in order of priority
    merges: Vec<(String, String)>,
    /// Unknown token ID
    unk_token_id: u32,
}

impl BPETokenizer {
    /// Create a new BPE tokenizer
    ///
    /// # Arguments
    ///
    /// * `vocab` - List of tokens (index = token ID)
    /// * `merges` - List of merge pairs in priority order
    /// * `unk_token` - Unknown token string
    ///
    /// # Errors
    ///
    /// Returns error if vocabulary is empty or unknown token not found
    pub fn new(vocab: Vec<String>, merges: Vec<(String, String)>, unk_token: &str) -> Result<Self> {
        if vocab.is_empty() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "create_bpe_tokenizer".to_string(),
                reason: "Vocabulary cannot be empty".to_string(),
            });
        }

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (id, token) in vocab.into_iter().enumerate() {
            let id = u32::try_from(id).map_err(|_| RealizarError::UnsupportedOperation {
                operation: "convert_token_id".to_string(),
                reason: format!("Token ID {id} exceeds u32 limit"),
            })?;
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }

        let unk_token_id =
            *token_to_id
                .get(unk_token)
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "create_bpe_tokenizer".to_string(),
                    reason: format!("Unknown token '{unk_token}' not in vocabulary"),
                })?;

        Ok(Self {
            token_to_id,
            id_to_token,
            merges,
            unk_token_id,
        })
    }

    /// Encode text to token IDs using greedy longest match
    ///
    /// Uses GPT-2 style encoding where spaces become Ġ (U+0120) and
    /// newlines become Ċ (U+010A).
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to encode
    ///
    /// # Returns
    ///
    /// Vector of token IDs
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Convert to GPT-2 encoding: space -> Ġ, newline -> Ċ
        let processed: String = text
            .chars()
            .map(|c| match c {
                ' ' => 'Ġ',  // U+0120
                '\n' => 'Ċ', // U+010A
                '\r' => 'Ḃ', // U+1E02
                _ => c,
            })
            .collect();

        let mut tokens = Vec::new();
        let mut remaining = processed.as_str();

        while !remaining.is_empty() {
            // Greedy longest match
            let mut best_len = 0;
            let mut best_id = None;

            // Collect character byte offsets for proper slicing
            let char_indices: Vec<usize> = remaining
                .char_indices()
                .map(|(i, _)| i)
                .chain(std::iter::once(remaining.len()))
                .collect();

            // Try all prefixes up to 32 chars
            for char_count in 1..=char_indices.len().saturating_sub(1).min(32) {
                let byte_end = char_indices[char_count];
                let prefix = &remaining[..byte_end];
                if let Some(&id) = self.token_to_id.get(prefix) {
                    best_len = byte_end;
                    best_id = Some(id);
                }
            }

            if let Some(id) = best_id {
                tokens.push(id);
                remaining = &remaining[best_len..];
            } else {
                // No match - try byte tokens like <0x48>
                let ch = remaining.chars().next().expect("non-empty");
                let ch_len = ch.len_utf8();

                for byte in remaining[..ch_len].bytes() {
                    let byte_token = format!("<0x{byte:02X}>");
                    if let Some(&id) = self.token_to_id.get(&byte_token) {
                        tokens.push(id);
                    } else {
                        tokens.push(self.unk_token_id);
                    }
                }
                remaining = &remaining[ch_len..];
            }
        }

        tokens
    }

    /// Apply a single merge rule to token list
    fn apply_merge(tokens: &[String], first: &str, second: &str) -> Vec<String> {
        if tokens.len() < 2 {
            return tokens.to_vec();
        }

        let mut result = Vec::new();
        let mut i = 0;

        while i < tokens.len() {
            if i + 1 < tokens.len() && tokens[i] == first && tokens[i + 1] == second {
                // Merge the pair
                result.push(format!("{first}{second}"));
                i += 2;
            } else {
                result.push(tokens[i].clone());
                i += 1;
            }
        }

        result
    }

    /// Decode token IDs to text
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Token IDs to decode
    ///
    /// # Errors
    ///
    /// Returns error if any token ID is invalid
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let mut bytes: Vec<u8> = Vec::new();

        for &id in token_ids {
            let token =
                self.id_to_token
                    .get(&id)
                    .ok_or_else(|| RealizarError::UnsupportedOperation {
                        operation: "decode_bpe_token".to_string(),
                        reason: format!("Invalid token ID: {id}"),
                    })?;

            // Skip special tokens
            if token.starts_with("<|") && token.ends_with("|>") {
                continue;
            }
            if token == "<s>" || token == "</s>" || token == "<unk>" || token == "<pad>" {
                continue;
            }

            // Handle byte tokens like <0xE6>
            if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
                if let Ok(byte_val) = u8::from_str_radix(&token[3..5], 16) {
                    bytes.push(byte_val);
                    continue;
                }
            }

            // Decode GPT-2 style byte-level BPE
            for c in token.chars() {
                match c {
                    'Ġ' => bytes.push(b' '),  // U+0120 -> space
                    'Ċ' => bytes.push(b'\n'), // U+010A -> newline
                    'ċ' => bytes.push(b'\n'), // lowercase variant
                    'Ḃ' => bytes.push(b'\r'), // U+1E02 -> carriage return
                    '▁' => bytes.push(b' '),  // U+2581 SentencePiece -> space
                    _ => {
                        // Try GPT-2 unicode-to-byte mapping
                        if let Some(byte) = Self::gpt2_char_to_byte(c) {
                            bytes.push(byte);
                        } else {
                            // Regular UTF-8 character
                            let mut buf = [0u8; 4];
                            let encoded = c.encode_utf8(&mut buf);
                            bytes.extend_from_slice(encoded.as_bytes());
                        }
                    }
                }
            }
        }

        // Decode as UTF-8, replacing invalid sequences
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Convert GPT-2 unicode character to original byte value
    fn gpt2_char_to_byte(c: char) -> Option<u8> {
        // GPT-2 maps bytes 0-255 to unicode characters
        // Printable ASCII (33-126) maps to itself
        // Other bytes map to unicode range starting at U+0100
        let code = c as u32;
        if (33..=126).contains(&code) || code == 32 {
            Some(code as u8)
        } else if (0x100..=0x100 + 255).contains(&code) {
            // GPT-2 remapped bytes
            let byte = (code - 0x100) as u8;
            // Map back based on GPT-2's byte_encoder
            match byte {
                0..=32 => Some(byte),      // Control chars + space
                127..=160 => Some(byte),   // DEL + extended ASCII
                173 => Some(173),          // Soft hyphen
                _ => None,
            }
        } else {
            None
        }
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Get token ID for a token
    #[must_use]
    pub fn get_token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token for a token ID
    #[must_use]
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }
}

/// Viterbi algorithm state: `(best_score, best_token)` at each position
type ViterbiState = (Vec<f32>, Vec<Option<String>>);

/// `SentencePiece` tokenizer (Unigram model)
///
/// Implements subword tokenization using unigram language model.
/// Used by `LLaMA`, T5, ALBERT, and many other models.
///
/// Unlike BPE which uses greedy merges, `SentencePiece` finds the
/// most likely segmentation using token scores (log probabilities).
#[derive(Debug, Clone)]
pub struct SentencePieceTokenizer {
    /// Token to ID mapping
    token_to_id: HashMap<String, u32>,
    /// ID to token mapping
    id_to_token: HashMap<u32, String>,
    /// Token scores (log probabilities)
    scores: HashMap<String, f32>,
    /// Unknown token ID
    unk_token_id: u32,
}

impl SentencePieceTokenizer {
    /// Create a new `SentencePiece` tokenizer
    ///
    /// # Arguments
    ///
    /// * `vocab` - List of (token, score) pairs where score is log probability
    /// * `unk_token` - Unknown token string
    ///
    /// # Errors
    ///
    /// Returns error if vocabulary is empty or unknown token not found
    pub fn new(vocab: Vec<(String, f32)>, unk_token: &str) -> Result<Self> {
        if vocab.is_empty() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "create_sentencepiece_tokenizer".to_string(),
                reason: "Vocabulary cannot be empty".to_string(),
            });
        }

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut scores = HashMap::new();

        for (id, (token, score)) in vocab.into_iter().enumerate() {
            let id = u32::try_from(id).map_err(|_| RealizarError::UnsupportedOperation {
                operation: "convert_token_id".to_string(),
                reason: format!("Token ID {id} exceeds u32 limit"),
            })?;
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token.clone());
            scores.insert(token, score);
        }

        let unk_token_id =
            *token_to_id
                .get(unk_token)
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "create_sentencepiece_tokenizer".to_string(),
                    reason: format!("Unknown token '{unk_token}' not in vocabulary"),
                })?;

        Ok(Self {
            token_to_id,
            id_to_token,
            scores,
            unk_token_id,
        })
    }

    /// Encode text to token IDs using Viterbi algorithm
    ///
    /// Finds the most likely segmentation based on token scores.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to encode
    ///
    /// # Returns
    ///
    /// Vector of token IDs
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let chars: Vec<char> = text.chars().collect();
        let (_best_score, best_token) = self.viterbi_forward(&chars);
        let tokens = Self::viterbi_backtrack(&chars, &best_token);

        // Convert tokens to IDs
        tokens
            .into_iter()
            .map(|t| {
                self.token_to_id
                    .get(&t)
                    .copied()
                    .unwrap_or(self.unk_token_id)
            })
            .collect()
    }

    /// Viterbi forward pass: find best score and token at each position
    ///
    /// # Arguments
    ///
    /// * `chars` - Character array of input text
    ///
    /// # Returns
    ///
    /// Tuple of `(best_score, best_token)` vectors
    fn viterbi_forward(&self, chars: &[char]) -> ViterbiState {
        let n = chars.len();
        let mut best_score = vec![f32::NEG_INFINITY; n + 1];
        let mut best_token: Vec<Option<String>> = vec![None; n + 1];
        best_score[0] = 0.0;

        for end in 1..=n {
            // Try all possible start positions for token ending at `end`
            for start in 0..end {
                let substr: String = chars[start..end].iter().collect();
                if let Some(&score) = self.scores.get(&substr) {
                    let new_score = best_score[start] + score;
                    if new_score > best_score[end] {
                        best_score[end] = new_score;
                        best_token[end] = Some(substr);
                    }
                }
            }

            // If no token found ending at this position, use single character as unknown
            if best_token[end].is_none() && best_score[end - 1] > f32::NEG_INFINITY {
                let char_str: String = chars[end - 1..end].iter().collect();
                best_score[end] = best_score[end - 1] - 100.0; // Penalty for unknown
                best_token[end] = Some(char_str);
            }
        }

        (best_score, best_token)
    }

    /// Viterbi backtracking: reconstruct token sequence from `best_token`
    ///
    /// # Arguments
    ///
    /// * `chars` - Character array of input text
    /// * `best_token` - Best token at each position (from forward pass)
    ///
    /// # Returns
    ///
    /// Vector of tokens in forward order
    fn viterbi_backtrack(chars: &[char], best_token: &[Option<String>]) -> Vec<String> {
        let n = chars.len();
        let mut tokens = Vec::new();
        let mut pos = n;

        while pos > 0 {
            if let Some(token) = &best_token[pos] {
                tokens.push(token.clone());
                pos -= token.chars().count();
            } else {
                // Fallback: single character
                let char_str: String = chars[pos - 1..pos].iter().collect();
                tokens.push(char_str);
                pos -= 1;
            }
        }

        tokens.reverse();
        tokens
    }

    /// Decode token IDs to text
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Token IDs to decode
    ///
    /// # Errors
    ///
    /// Returns error if any token ID is invalid
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let mut result = String::new();

        for &id in token_ids {
            let token =
                self.id_to_token
                    .get(&id)
                    .ok_or_else(|| RealizarError::UnsupportedOperation {
                        operation: "decode_sentencepiece_token".to_string(),
                        reason: format!("Invalid token ID: {id}"),
                    })?;
            result.push_str(token);
        }

        Ok(result)
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Get token ID for a token
    #[must_use]
    pub fn get_token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token for a token ID
    #[must_use]
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// Get score for a token
    #[must_use]
    pub fn get_score(&self, token: &str) -> Option<f32> {
        self.scores.get(token).copied()
    }
}

impl Tokenizer {
    /// Create a new tokenizer from vocabulary
    ///
    /// # Arguments
    ///
    /// * `vocab` - Vocabulary mapping
    /// * `unk_token` - Unknown token (default: `"<unk>"`)
    ///
    /// # Errors
    ///
    /// Returns error if unknown token not in vocabulary
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let vocab = Vocabulary::from_tokens(vec!["<unk>".to_string(), "hello".to_string()])?;
    /// let tokenizer = Tokenizer::new(vocab, "<unk>")?;
    /// ```
    pub fn new(vocab: Vocabulary, unk_token: &str) -> Result<Self> {
        let unk_token_id =
            vocab
                .get_id(unk_token)
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "create_tokenizer".to_string(),
                    reason: format!("Unknown token '{unk_token}' not in vocabulary"),
                })?;

        Ok(Self {
            vocab,
            unk_token_id,
        })
    }

    /// Encode text to token IDs (simple word-level tokenization)
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to encode
    ///
    /// # Returns
    ///
    /// Vector of token IDs
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let token_ids = tokenizer.encode("hello world")?;
    /// assert_eq!(token_ids, vec![1, 2]);
    /// ```
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|word| self.vocab.get_id(word).unwrap_or(self.unk_token_id))
            .collect()
    }

    /// Decode token IDs to text
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Token IDs to decode
    ///
    /// # Returns
    ///
    /// Decoded text string
    ///
    /// # Errors
    ///
    /// Returns error if any token ID is invalid
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let text = tokenizer.decode(&[1, 2])?;
    /// assert_eq!(text, "hello world");
    /// ```
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let tokens: Result<Vec<&str>> = token_ids
            .iter()
            .map(|&id| {
                self.vocab
                    .get_token(id)
                    .ok_or_else(|| RealizarError::UnsupportedOperation {
                        operation: "decode_token".to_string(),
                        reason: format!("Invalid token ID: {id}"),
                    })
            })
            .collect();

        Ok(tokens?.join(" "))
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_from_tokens() {
        let tokens = vec![
            "<unk>".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];

        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        assert_eq!(vocab.size(), 3);
        assert_eq!(vocab.get_id("<unk>"), Some(0));
        assert_eq!(vocab.get_id("hello"), Some(1));
        assert_eq!(vocab.get_id("world"), Some(2));
        assert_eq!(vocab.get_token(0), Some("<unk>"));
        assert_eq!(vocab.get_token(1), Some("hello"));
        assert_eq!(vocab.get_token(2), Some("world"));
    }

    #[test]
    fn test_vocabulary_empty_error() {
        let result = Vocabulary::from_tokens(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_vocabulary_duplicate_error() {
        let tokens = vec![
            "hello".to_string(),
            "world".to_string(),
            "hello".to_string(), // Duplicate
        ];
        let result = Vocabulary::from_tokens(tokens);
        assert!(result.is_err());
    }

    #[test]
    fn test_vocabulary_get_missing() {
        let tokens = vec!["hello".to_string()];
        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        assert_eq!(vocab.get_id("world"), None);
        assert_eq!(vocab.get_token(999), None);
    }

    #[test]
    fn test_tokenizer_encode_decode() {
        let tokens = vec![
            "<unk>".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];
        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        let tokenizer = Tokenizer::new(vocab, "<unk>").expect("test");

        // Encode known tokens
        let encoded = tokenizer.encode("hello world");
        assert_eq!(encoded, vec![1, 2]);

        // Decode back
        let decoded = tokenizer.decode(&encoded).expect("test");
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_tokenizer_unknown_token() {
        let tokens = vec!["<unk>".to_string(), "hello".to_string()];
        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        let tokenizer = Tokenizer::new(vocab, "<unk>").expect("test");

        // Unknown token should map to <unk> (ID 0)
        let encoded = tokenizer.encode("hello foo");
        assert_eq!(encoded, vec![1, 0]);
    }

    #[test]
    fn test_tokenizer_invalid_unk_token() {
        let tokens = vec!["hello".to_string()];
        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        let result = Tokenizer::new(vocab, "<unk>");
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenizer_decode_invalid_id() {
        let tokens = vec!["<unk>".to_string(), "hello".to_string()];
        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        let tokenizer = Tokenizer::new(vocab, "<unk>").expect("test");

        let result = tokenizer.decode(&[1, 999]); // 999 is invalid
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenizer_empty_string() {
        let tokens = vec!["<unk>".to_string()];
        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        let tokenizer = Tokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode("");
        assert_eq!(encoded, Vec::<u32>::new());

        let decoded = tokenizer.decode(&[]).expect("test");
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_tokenizer_vocab_size() {
        let tokens = vec![
            "<unk>".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];
        let vocab = Vocabulary::from_tokens(tokens).expect("test");
        let tokenizer = Tokenizer::new(vocab, "<unk>").expect("test");

        assert_eq!(tokenizer.vocab_size(), 3);
    }

    // BPE Tokenizer tests

    #[test]
    fn test_bpe_tokenizer_creation() {
        let vocab = vec![
            "<unk>".to_string(),
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "he".to_string(),
            "ll".to_string(),
            "hel".to_string(),
            "hello".to_string(),
        ];
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("l".to_string(), "l".to_string()),
            ("he".to_string(), "l".to_string()),
            ("hel".to_string(), "lo".to_string()),
        ];

        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");
        assert_eq!(tokenizer.vocab_size(), 9);
    }

    #[test]
    fn test_bpe_tokenizer_empty_vocab_error() {
        let result = BPETokenizer::new(vec![], vec![], "<unk>");
        assert!(result.is_err());
    }

    #[test]
    fn test_bpe_tokenizer_invalid_unk_token_error() {
        let vocab = vec!["hello".to_string()];
        let result = BPETokenizer::new(vocab, vec![], "<unk>");
        assert!(result.is_err());
    }

    #[test]
    fn test_bpe_encode_no_merges() {
        // Simple character-level tokenization without merges
        let vocab = vec!["<unk>".to_string(), "h".to_string(), "i".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let encoded = tokenizer.encode("hi");
        assert_eq!(encoded, vec![1, 2]); // h=1, i=2
    }

    #[test]
    fn test_bpe_encode_with_merges() {
        let vocab = vec![
            "<unk>".to_string(),
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "he".to_string(),
            "ll".to_string(),
        ];
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("l".to_string(), "l".to_string()),
        ];

        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");
        let encoded = tokenizer.encode("hello");
        // h+e -> he, l+l -> ll, o stays
        // so: he, ll, o = [5, 6, 4]
        assert_eq!(encoded, vec![5, 6, 4]);
    }

    #[test]
    fn test_bpe_encode_unknown_char() {
        let vocab = vec!["<unk>".to_string(), "h".to_string(), "i".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        // 'x' is not in vocab, should map to <unk>
        let encoded = tokenizer.encode("hix");
        assert_eq!(encoded, vec![1, 2, 0]);
    }

    #[test]
    fn test_bpe_encode_empty_string() {
        let vocab = vec!["<unk>".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let encoded = tokenizer.encode("");
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_bpe_encode_multiple_words() {
        let vocab = vec![
            "<unk>".to_string(),
            "h".to_string(),
            "i".to_string(),
            " ".to_string(),
            " h".to_string(),
        ];
        let merges = vec![(" ".to_string(), "h".to_string())];

        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");
        // "hi hi" -> "hi" + " hi"
        // "hi" -> h, i
        // " hi" -> " " + "h" -> " h", then "i"
        let encoded = tokenizer.encode("hi hi");
        assert_eq!(encoded, vec![1, 2, 4, 2]); // h, i, " h", i
    }

    #[test]
    fn test_bpe_decode() {
        let vocab = vec!["<unk>".to_string(), "hel".to_string(), "lo".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let decoded = tokenizer.decode(&[1, 2]).expect("test");
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_bpe_decode_empty() {
        let vocab = vec!["<unk>".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let decoded = tokenizer.decode(&[]).expect("test");
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_bpe_decode_invalid_id_error() {
        let vocab = vec!["<unk>".to_string(), "hi".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        let result = tokenizer.decode(&[1, 999]);
        assert!(result.is_err());
    }

    #[test]
    fn test_bpe_encode_decode_roundtrip() {
        let vocab = vec![
            "<unk>".to_string(),
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "he".to_string(),
            "ll".to_string(),
            "lo".to_string(),
            "hel".to_string(),
            "hello".to_string(),
        ];
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("l".to_string(), "l".to_string()),
            ("l".to_string(), "o".to_string()),
            ("he".to_string(), "l".to_string()),
            ("hel".to_string(), "lo".to_string()),
        ];

        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");

        let encoded = tokenizer.encode("hello");
        let decoded = tokenizer.decode(&encoded).expect("test");
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_bpe_get_token_methods() {
        let vocab = vec!["<unk>".to_string(), "hello".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

        assert_eq!(tokenizer.get_token_id("hello"), Some(1));
        assert_eq!(tokenizer.get_token_id("world"), None);
        assert_eq!(tokenizer.get_token(1), Some("hello"));
        assert_eq!(tokenizer.get_token(999), None);
    }

    #[test]
    fn test_bpe_multiple_consecutive_merges() {
        // Test that multiple merges are applied correctly
        let vocab = vec![
            "<unk>".to_string(),
            "a".to_string(),
            "b".to_string(),
            "ab".to_string(),
            "abab".to_string(),
        ];
        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("ab".to_string(), "ab".to_string()),
        ];

        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").expect("test");
        let encoded = tokenizer.encode("abab");
        // First: a+b -> ab, a+b -> ab giving [ab, ab]
        // Then: ab+ab -> abab giving [abab]
        assert_eq!(encoded, vec![4]);
    }

    // SentencePiece Tokenizer tests

    #[test]
    fn test_sentencepiece_tokenizer_creation() {
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("hello".to_string(), -1.0),
            ("world".to_string(), -1.5),
        ];

        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");
        assert_eq!(tokenizer.vocab_size(), 3);
    }

    #[test]
    fn test_sentencepiece_empty_vocab_error() {
        let result = SentencePieceTokenizer::new(vec![], "<unk>");
        assert!(result.is_err());
    }

    #[test]
    fn test_sentencepiece_invalid_unk_token_error() {
        let vocab = vec![("hello".to_string(), -1.0)];
        let result = SentencePieceTokenizer::new(vocab, "<unk>");
        assert!(result.is_err());
    }

    #[test]
    fn test_sentencepiece_encode_empty() {
        let vocab = vec![("<unk>".to_string(), 0.0)];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode("");
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_sentencepiece_encode_single_token() {
        let vocab = vec![("<unk>".to_string(), 0.0), ("hello".to_string(), -1.0)];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode("hello");
        assert_eq!(encoded, vec![1]);
    }

    #[test]
    fn test_sentencepiece_encode_prefers_higher_score() {
        // "hello" as single token has score -1.0
        // "hel" + "lo" would have score -2.0 + -2.0 = -4.0
        // So "hello" should be preferred
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("h".to_string(), -5.0),
            ("e".to_string(), -5.0),
            ("l".to_string(), -5.0),
            ("o".to_string(), -5.0),
            ("hel".to_string(), -2.0),
            ("lo".to_string(), -2.0),
            ("hello".to_string(), -1.0),
        ];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode("hello");
        // Should prefer single "hello" token (score -1.0) over subwords
        assert_eq!(encoded, vec![7]);
    }

    #[test]
    fn test_sentencepiece_encode_subwords() {
        // Only subwords available, not full word
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("h".to_string(), -1.0),
            ("e".to_string(), -1.0),
            ("l".to_string(), -1.0),
            ("o".to_string(), -1.0),
            ("he".to_string(), -0.5),
            ("llo".to_string(), -0.5),
        ];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode("hello");
        // "he" (-0.5) + "llo" (-0.5) = -1.0 is better than "h" + "e" + "l" + "l" + "o" = -5.0
        assert_eq!(encoded, vec![5, 6]);
    }

    #[test]
    fn test_sentencepiece_decode() {
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("hel".to_string(), -1.0),
            ("lo".to_string(), -1.0),
        ];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let decoded = tokenizer.decode(&[1, 2]).expect("test");
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_sentencepiece_decode_empty() {
        let vocab = vec![("<unk>".to_string(), 0.0)];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let decoded = tokenizer.decode(&[]).expect("test");
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_sentencepiece_decode_invalid_id_error() {
        let vocab = vec![("<unk>".to_string(), 0.0), ("hi".to_string(), -1.0)];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let result = tokenizer.decode(&[1, 999]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sentencepiece_encode_decode_roundtrip() {
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("h".to_string(), -2.0),
            ("e".to_string(), -2.0),
            ("l".to_string(), -2.0),
            ("o".to_string(), -2.0),
            ("hello".to_string(), -1.0),
        ];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode("hello");
        let decoded = tokenizer.decode(&encoded).expect("test");
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_sentencepiece_get_methods() {
        let vocab = vec![("<unk>".to_string(), 0.0), ("hello".to_string(), -1.5)];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        assert_eq!(tokenizer.get_token_id("hello"), Some(1));
        assert_eq!(tokenizer.get_token_id("world"), None);
        assert_eq!(tokenizer.get_token(1), Some("hello"));
        assert_eq!(tokenizer.get_token(999), None);
        assert!((tokenizer.get_score("hello").expect("test") - (-1.5)).abs() < 1e-6);
        assert_eq!(tokenizer.get_score("world"), None);
    }

    #[test]
    fn test_sentencepiece_unknown_character() {
        // Character not in vocabulary should use unknown penalty
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("h".to_string(), -1.0),
            ("i".to_string(), -1.0),
        ];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        // 'x' is not in vocab, should be tokenized with penalty
        let encoded = tokenizer.encode("hix");
        assert_eq!(encoded.len(), 3);
        assert_eq!(encoded[0], 1); // h
        assert_eq!(encoded[1], 2); // i
                                   // x should map to unk
        assert_eq!(encoded[2], 0);
    }

    #[test]
    fn test_sentencepiece_multiple_words() {
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("hello".to_string(), -1.0),
            (" ".to_string(), -0.5),
            ("world".to_string(), -1.0),
        ];
        let tokenizer = SentencePieceTokenizer::new(vocab, "<unk>").expect("test");

        let encoded = tokenizer.encode("hello world");
        assert_eq!(encoded, vec![1, 2, 3]); // hello, space, world
    }
}
