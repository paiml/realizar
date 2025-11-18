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

        let unk_token_id = *token_to_id.get(unk_token).ok_or_else(|| {
            RealizarError::UnsupportedOperation {
                operation: "create_bpe_tokenizer".to_string(),
                reason: format!("Unknown token '{unk_token}' not in vocabulary"),
            }
        })?;

        Ok(Self {
            token_to_id,
            id_to_token,
            merges,
            unk_token_id,
        })
    }

    /// Encode text to token IDs using BPE
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

        // Split into words, preserving spaces as part of tokens (GPT-2 style)
        let words: Vec<String> = text
            .split(' ')
            .enumerate()
            .flat_map(|(i, word)| {
                if word.is_empty() {
                    vec![]
                } else if i == 0 {
                    vec![word.to_string()]
                } else {
                    // Prepend space to non-first words (GPT-2 convention)
                    vec![format!(" {word}")]
                }
            })
            .collect();

        let mut result = Vec::new();

        for word in words {
            // Start with characters as initial tokens
            let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();

            // Apply merges iteratively
            for (first, second) in &self.merges {
                tokens = Self::apply_merge(&tokens, first, second);
            }

            // Convert tokens to IDs
            for token in tokens {
                let id = self
                    .token_to_id
                    .get(&token)
                    .copied()
                    .unwrap_or(self.unk_token_id);
                result.push(id);
            }
        }

        result
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
        let mut result = String::new();

        for &id in token_ids {
            let token = self.id_to_token.get(&id).ok_or_else(|| {
                RealizarError::UnsupportedOperation {
                    operation: "decode_bpe_token".to_string(),
                    reason: format!("Invalid token ID: {id}"),
                }
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
        let unk_token_id = vocab.get_id(unk_token).ok_or_else(|| {
            RealizarError::UnsupportedOperation {
                operation: "create_tokenizer".to_string(),
                reason: format!("Unknown token '{unk_token}' not in vocabulary"),
            }
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
            .map(|word| {
                self.vocab
                    .get_id(word)
                    .unwrap_or(self.unk_token_id)
            })
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
                self.vocab.get_token(id).ok_or_else(|| {
                    RealizarError::UnsupportedOperation {
                        operation: "decode_token".to_string(),
                        reason: format!("Invalid token ID: {id}"),
                    }
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

        let vocab = Vocabulary::from_tokens(tokens).unwrap();
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
        let vocab = Vocabulary::from_tokens(tokens).unwrap();
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
        let vocab = Vocabulary::from_tokens(tokens).unwrap();
        let tokenizer = Tokenizer::new(vocab, "<unk>").unwrap();

        // Encode known tokens
        let encoded = tokenizer.encode("hello world");
        assert_eq!(encoded, vec![1, 2]);

        // Decode back
        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_tokenizer_unknown_token() {
        let tokens = vec![
            "<unk>".to_string(),
            "hello".to_string(),
        ];
        let vocab = Vocabulary::from_tokens(tokens).unwrap();
        let tokenizer = Tokenizer::new(vocab, "<unk>").unwrap();

        // Unknown token should map to <unk> (ID 0)
        let encoded = tokenizer.encode("hello foo");
        assert_eq!(encoded, vec![1, 0]);
    }

    #[test]
    fn test_tokenizer_invalid_unk_token() {
        let tokens = vec!["hello".to_string()];
        let vocab = Vocabulary::from_tokens(tokens).unwrap();
        let result = Tokenizer::new(vocab, "<unk>");
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenizer_decode_invalid_id() {
        let tokens = vec!["<unk>".to_string(), "hello".to_string()];
        let vocab = Vocabulary::from_tokens(tokens).unwrap();
        let tokenizer = Tokenizer::new(vocab, "<unk>").unwrap();

        let result = tokenizer.decode(&[1, 999]); // 999 is invalid
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenizer_empty_string() {
        let tokens = vec!["<unk>".to_string()];
        let vocab = Vocabulary::from_tokens(tokens).unwrap();
        let tokenizer = Tokenizer::new(vocab, "<unk>").unwrap();

        let encoded = tokenizer.encode("");
        assert_eq!(encoded, Vec::<u32>::new());

        let decoded = tokenizer.decode(&[]).unwrap();
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_tokenizer_vocab_size() {
        let tokens = vec![
            "<unk>".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];
        let vocab = Vocabulary::from_tokens(tokens).unwrap();
        let tokenizer = Tokenizer::new(vocab, "<unk>").unwrap();

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

        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").unwrap();
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
        let vocab = vec![
            "<unk>".to_string(),
            "h".to_string(),
            "i".to_string(),
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").unwrap();

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

        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").unwrap();
        let encoded = tokenizer.encode("hello");
        // h+e -> he, l+l -> ll, o stays
        // so: he, ll, o = [5, 6, 4]
        assert_eq!(encoded, vec![5, 6, 4]);
    }

    #[test]
    fn test_bpe_encode_unknown_char() {
        let vocab = vec![
            "<unk>".to_string(),
            "h".to_string(),
            "i".to_string(),
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").unwrap();

        // 'x' is not in vocab, should map to <unk>
        let encoded = tokenizer.encode("hix");
        assert_eq!(encoded, vec![1, 2, 0]);
    }

    #[test]
    fn test_bpe_encode_empty_string() {
        let vocab = vec!["<unk>".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").unwrap();

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

        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").unwrap();
        // "hi hi" -> "hi" + " hi"
        // "hi" -> h, i
        // " hi" -> " " + "h" -> " h", then "i"
        let encoded = tokenizer.encode("hi hi");
        assert_eq!(encoded, vec![1, 2, 4, 2]); // h, i, " h", i
    }

    #[test]
    fn test_bpe_decode() {
        let vocab = vec![
            "<unk>".to_string(),
            "hel".to_string(),
            "lo".to_string(),
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").unwrap();

        let decoded = tokenizer.decode(&[1, 2]).unwrap();
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_bpe_decode_empty() {
        let vocab = vec!["<unk>".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").unwrap();

        let decoded = tokenizer.decode(&[]).unwrap();
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_bpe_decode_invalid_id_error() {
        let vocab = vec!["<unk>".to_string(), "hi".to_string()];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").unwrap();

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

        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").unwrap();

        let encoded = tokenizer.encode("hello");
        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_bpe_get_token_methods() {
        let vocab = vec![
            "<unk>".to_string(),
            "hello".to_string(),
        ];
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").unwrap();

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

        let tokenizer = BPETokenizer::new(vocab, merges, "<unk>").unwrap();
        let encoded = tokenizer.encode("abab");
        // First: a+b -> ab, a+b -> ab giving [ab, ab]
        // Then: ab+ab -> abab giving [abab]
        assert_eq!(encoded, vec![4]);
    }
}
