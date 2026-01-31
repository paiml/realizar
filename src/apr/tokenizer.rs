//! BPE Tokenizer for APR models (PMAT-802)
//!
//! Byte Pair Encoding tokenizer supporting APR v2 format models.
//!
//! ## Tokenizer Types
//!
//! - `BpeTokenizer`: Full tokenizer with encode/decode, requires tokenizer.json
//! - `SimpleTokenizer`: Decode-only tokenizer using embedded vocabulary (GH-156)
//!
//! For APR models, prefer `SimpleTokenizer` via `AprV2Model::load_embedded_tokenizer()`
//! as it uses the vocabulary embedded in the .apr file - no sibling files needed.

use super::AprV2Model;
use std::collections::HashMap;

// ============================================================================
// SimpleTokenizer (GH-156): Decode-only tokenizer from embedded APR vocabulary
// ============================================================================

/// Simple decode-only tokenizer for APR models with embedded vocabulary.
///
/// Unlike `BpeTokenizer`, this doesn't require tokenizer.json - it uses
/// the vocabulary embedded directly in the APR file's metadata section.
///
/// # Example
///
/// ```rust,ignore
/// let model = AprV2Model::load("model.apr")?;
/// if let Some(tokenizer) = model.load_embedded_tokenizer() {
///     let text = tokenizer.decode(&[1, 2, 3]);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    /// Vocabulary: index = token ID, value = token string
    pub id_to_token: Vec<String>,
    /// Beginning-of-sequence token ID (optional)
    pub bos_token_id: Option<u32>,
    /// End-of-sequence token ID (optional)
    pub eos_token_id: Option<u32>,
}

impl SimpleTokenizer {
    /// Create a new simple tokenizer from vocabulary
    #[must_use]
    pub fn new(vocab: Vec<String>, bos_id: Option<u32>, eos_id: Option<u32>) -> Self {
        Self {
            id_to_token: vocab,
            bos_token_id: bos_id,
            eos_token_id: eos_id,
        }
    }

    /// Decode token IDs to text
    ///
    /// Handles byte-level BPE encoding (Ġ = space prefix, Ċ = newline, etc.)
    #[must_use]
    pub fn decode(&self, token_ids: &[u32]) -> String {
        AprV2Model::decode_tokens(&self.id_to_token, token_ids)
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Check if token ID is end-of-sequence
    #[must_use]
    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_id.is_some_and(|eos| token_id == eos)
    }

    /// Check if token ID is beginning-of-sequence
    #[must_use]
    pub fn is_bos(&self, token_id: u32) -> bool {
        self.bos_token_id.is_some_and(|bos| token_id == bos)
    }
}

// ============================================================================
// BpeTokenizer: Full encode/decode from tokenizer.json
// ============================================================================

/// BPE Tokenizer for encoding and decoding text
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Token string to ID mapping
    pub token_to_id: HashMap<String, u32>,
    /// ID to token string mapping (index = ID)
    pub id_to_token: Vec<String>,
    /// BPE merge rules (first, second) pairs
    pub merge_rules: Vec<(String, String)>,
    /// Beginning-of-sequence token ID
    pub bos_id: Option<u32>,
    /// End-of-sequence token ID
    pub eos_id: Option<u32>,
    /// Special tokens (e.g., <|im_start|>, <|im_end|>) - GH-189 fix
    /// These are tokenized atomically, not split by BPE
    pub special_tokens: HashMap<String, u32>,
}

impl BpeTokenizer {
    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // GH-189: Use special_tokens for atomic tokenization of chat markers
        bpe_encode(text, &self.token_to_id, &self.merge_rules, &self.special_tokens)
    }

    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[u32]) -> String {
        AprV2Model::decode_tokens(&self.id_to_token, token_ids)
    }
}

/// Byte-level BPE encoding with special token support (F-REGR-231 fix)
///
/// Special tokens (like `<|im_start|>`, `<|im_end|>`) are handled atomically
/// and not split by BPE. This is critical for chat template markers.
pub(crate) fn bpe_encode(
    text: &str,
    vocab: &HashMap<String, u32>,
    merges: &[(String, String)],
    special_tokens: &HashMap<String, u32>,
) -> Vec<u32> {
    // F-REGR-231: First, split text by special tokens
    // This ensures tokens like <|im_start|> are not broken into characters
    let segments = split_by_special_tokens(text, special_tokens);

    let mut result = Vec::new();
    for segment in segments {
        match segment {
            TextSegment::Special(id) => {
                result.push(id);
            }
            TextSegment::Regular(s) => {
                result.extend(bpe_encode_segment(&s, vocab, merges));
            }
        }
    }
    result
}

/// Segment type for special token handling
enum TextSegment {
    Special(u32),
    Regular(String),
}

/// Split text by special tokens, preserving order
fn split_by_special_tokens(text: &str, special_tokens: &HashMap<String, u32>) -> Vec<TextSegment> {
    if special_tokens.is_empty() {
        return vec![TextSegment::Regular(text.to_string())];
    }

    // Sort special tokens by length (longest first) to handle overlapping patterns
    let mut sorted_tokens: Vec<(&String, &u32)> = special_tokens.iter().collect();
    sorted_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    let mut segments = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        // Try to match a special token at the start
        let mut matched = false;
        for (token_str, &token_id) in &sorted_tokens {
            if remaining.starts_with(token_str.as_str()) {
                segments.push(TextSegment::Special(token_id));
                remaining = &remaining[token_str.len()..];
                matched = true;
                break;
            }
        }

        if !matched {
            // Find the next special token occurrence
            let mut next_special_pos = remaining.len();
            for (token_str, _) in &sorted_tokens {
                if let Some(pos) = remaining.find(token_str.as_str()) {
                    if pos < next_special_pos {
                        next_special_pos = pos;
                    }
                }
            }

            // Add regular text segment up to the next special token
            if next_special_pos > 0 {
                segments.push(TextSegment::Regular(remaining[..next_special_pos].to_string()));
                remaining = &remaining[next_special_pos..];
            }
        }
    }

    segments
}

/// BPE encode a regular text segment (no special tokens)
fn bpe_encode_segment(
    text: &str,
    vocab: &HashMap<String, u32>,
    merges: &[(String, String)],
) -> Vec<u32> {
    // Convert text to byte-level tokens (GPT-2/Qwen style)
    // Each byte maps to a special unicode char in range U+0100-U+01FF or similar
    let mut tokens: Vec<String> = text
        .chars()
        .map(|c| {
            // Convert character to byte-level BPE token
            // Space becomes Ġ (U+0120 = 288), newline becomes Ċ, etc.
            if c == ' ' {
                "Ġ".to_string()
            } else if c == '\n' {
                "Ċ".to_string()
            } else if c == '\t' {
                "ĉ".to_string()
            } else if c.is_ascii() {
                c.to_string()
            } else {
                // For non-ASCII, encode as bytes
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                s.chars()
                    .map(|byte_char| byte_to_bpe_char(byte_char as u8))
                    .collect()
            }
        })
        .collect();

    // Apply BPE merges iteratively
    for (first, second) in merges {
        let merged = format!("{}{}", first, second);
        loop {
            let mut found = false;
            let mut i = 0;
            while i + 1 < tokens.len() {
                if &tokens[i] == first && &tokens[i + 1] == second {
                    tokens[i].clone_from(&merged);
                    tokens.remove(i + 1);
                    found = true;
                }
                i += 1;
            }
            if !found {
                break;
            }
        }
    }

    // Convert tokens to IDs
    tokens
        .iter()
        .filter_map(|t| vocab.get(t).copied())
        .collect()
}

/// Convert byte to BPE character representation
pub fn byte_to_bpe_char(b: u8) -> String {
    // GPT-2/Qwen byte-level BPE uses specific unicode mappings
    // This is a simplified version - real tokenizers use a full byte-to-unicode table
    match b {
        b' ' => "Ġ".to_string(),
        b'\n' => "Ċ".to_string(),
        b'\t' => "ĉ".to_string(),
        _ if b.is_ascii_graphic() || b.is_ascii_alphanumeric() => (b as char).to_string(),
        _ => format!("<0x{:02X}>", b),
    }
}

// ============================================================================
// Tests for APR Tokenizer (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // SimpleTokenizer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simple_tokenizer_new() {
        let vocab = vec!["hello".to_string(), "world".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, Some(0), Some(1));
        assert_eq!(tokenizer.vocab_size(), 2);
        assert_eq!(tokenizer.bos_token_id, Some(0));
        assert_eq!(tokenizer.eos_token_id, Some(1));
    }

    #[test]
    fn test_simple_tokenizer_vocab_size() {
        let vocab = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);
        assert_eq!(tokenizer.vocab_size(), 3);
    }

    #[test]
    fn test_simple_tokenizer_is_eos() {
        let vocab = vec!["<s>".to_string(), "</s>".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, Some(0), Some(1));
        assert!(tokenizer.is_eos(1));
        assert!(!tokenizer.is_eos(0));
        assert!(!tokenizer.is_eos(2)); // Out of range
    }

    #[test]
    fn test_simple_tokenizer_is_eos_none() {
        let vocab = vec!["hello".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);
        assert!(!tokenizer.is_eos(0));
        assert!(!tokenizer.is_eos(1));
    }

    #[test]
    fn test_simple_tokenizer_is_bos() {
        let vocab = vec!["<s>".to_string(), "</s>".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, Some(0), Some(1));
        assert!(tokenizer.is_bos(0));
        assert!(!tokenizer.is_bos(1));
    }

    #[test]
    fn test_simple_tokenizer_is_bos_none() {
        let vocab = vec!["hello".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);
        assert!(!tokenizer.is_bos(0));
    }

    // -------------------------------------------------------------------------
    // BpeTokenizer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bpe_tokenizer_encode_empty() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec![],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let result = tokenizer.encode("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_encode_simple() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("h".to_string(), 0);
        token_to_id.insert("i".to_string(), 1);

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["h".to_string(), "i".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let result = tokenizer.encode("hi");
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn test_bpe_tokenizer_encode_with_space() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("Ġ".to_string(), 0); // Space
        token_to_id.insert("a".to_string(), 1);

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["Ġ".to_string(), "a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let result = tokenizer.encode(" a");
        assert_eq!(result, vec![0, 1]);
    }

    // -------------------------------------------------------------------------
    // bpe_encode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bpe_encode_empty() {
        let vocab: HashMap<String, u32> = HashMap::new();
        let merges: Vec<(String, String)> = vec![];
        let special: HashMap<String, u32> = HashMap::new();
        let result = bpe_encode("", &vocab, &merges, &special);
        assert!(result.is_empty());
    }

    #[test]
    fn test_bpe_encode_simple_chars() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 1);
        vocab.insert("c".to_string(), 2);
        let special: HashMap<String, u32> = HashMap::new();

        let result = bpe_encode("abc", &vocab, &[], &special);
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_bpe_encode_with_merge() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 1);
        vocab.insert("ab".to_string(), 2);
        let special: HashMap<String, u32> = HashMap::new();

        let merges = vec![("a".to_string(), "b".to_string())];
        let result = bpe_encode("ab", &vocab, &merges, &special);
        assert_eq!(result, vec![2]); // "ab" merged
    }

    #[test]
    fn test_bpe_encode_space_handling() {
        let mut vocab = HashMap::new();
        vocab.insert("Ġ".to_string(), 0); // Space becomes Ġ
        let special: HashMap<String, u32> = HashMap::new();

        let result = bpe_encode(" ", &vocab, &[], &special);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_newline_handling() {
        let mut vocab = HashMap::new();
        vocab.insert("Ċ".to_string(), 0); // Newline becomes Ċ
        let special: HashMap<String, u32> = HashMap::new();

        let result = bpe_encode("\n", &vocab, &[], &special);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_unknown_tokens() {
        let vocab: HashMap<String, u32> = HashMap::new();
        let special: HashMap<String, u32> = HashMap::new();
        let result = bpe_encode("xyz", &vocab, &[], &special);
        assert!(result.is_empty()); // Unknown tokens filtered out
    }

    // -------------------------------------------------------------------------
    // byte_to_bpe_char Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_byte_to_bpe_char_space() {
        assert_eq!(byte_to_bpe_char(b' '), "Ġ");
    }

    #[test]
    fn test_byte_to_bpe_char_newline() {
        assert_eq!(byte_to_bpe_char(b'\n'), "Ċ");
    }

    #[test]
    fn test_byte_to_bpe_char_tab() {
        assert_eq!(byte_to_bpe_char(b'\t'), "ĉ");
    }

    #[test]
    fn test_byte_to_bpe_char_ascii() {
        assert_eq!(byte_to_bpe_char(b'a'), "a");
        assert_eq!(byte_to_bpe_char(b'Z'), "Z");
        assert_eq!(byte_to_bpe_char(b'0'), "0");
        assert_eq!(byte_to_bpe_char(b'!'), "!");
    }

    #[test]
    fn test_byte_to_bpe_char_non_printable() {
        // Non-printable bytes get hex encoding
        assert_eq!(byte_to_bpe_char(0x00), "<0x00>");
        assert_eq!(byte_to_bpe_char(0x7F), "<0x7F>");
        assert_eq!(byte_to_bpe_char(0xFF), "<0xFF>");
    }

    // -------------------------------------------------------------------------
    // GH-189: Special Token Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bpe_tokenizer_with_special_tokens() {
        // GH-189: Verify special tokens are tokenized atomically
        let mut token_to_id = HashMap::new();
        token_to_id.insert("<|im_start|>".to_string(), 100);
        token_to_id.insert("<|im_end|>".to_string(), 101);
        token_to_id.insert("h".to_string(), 0);
        token_to_id.insert("i".to_string(), 1);

        let mut special_tokens = HashMap::new();
        special_tokens.insert("<|im_start|>".to_string(), 100);
        special_tokens.insert("<|im_end|>".to_string(), 101);

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec![
                "h".to_string(),
                "i".to_string(),
                // ... padding ...
            ],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens,
        };

        // Test that special tokens are kept atomic
        let result = tokenizer.encode("<|im_start|>hi<|im_end|>");
        assert_eq!(result.len(), 4); // <|im_start|>, h, i, <|im_end|>
        assert_eq!(result[0], 100); // <|im_start|>
        assert_eq!(result[3], 101); // <|im_end|>
    }

    #[test]
    fn test_bpe_encode_with_special_tokens() {
        // GH-189: Test bpe_encode directly with special tokens
        let mut vocab = HashMap::new();
        vocab.insert("<|im_start|>".to_string(), 100);
        vocab.insert("<|im_end|>".to_string(), 101);
        vocab.insert("H".to_string(), 0);
        vocab.insert("i".to_string(), 1);

        let mut special = HashMap::new();
        special.insert("<|im_start|>".to_string(), 100);
        special.insert("<|im_end|>".to_string(), 101);

        let result = bpe_encode("<|im_start|>Hi<|im_end|>", &vocab, &[], &special);
        // Should be: [100, 0, 1, 101] = <|im_start|>, H, i, <|im_end|>
        assert!(result.contains(&100)); // <|im_start|>
        assert!(result.contains(&101)); // <|im_end|>
        // Verify order: special token should be first
        assert_eq!(result[0], 100);
    }

    #[test]
    fn test_split_by_special_tokens() {
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<|im_start|>".to_string(), 100);
        special_tokens.insert("<|im_end|>".to_string(), 101);

        let text = "<|im_start|>user\nHello<|im_end|>";
        let segments = split_by_special_tokens(text, &special_tokens);

        // Should have 3 segments: Special(100), Regular("user\nHello"), Special(101)
        assert_eq!(segments.len(), 3);
        match &segments[0] {
            TextSegment::Special(id) => assert_eq!(*id, 100),
            TextSegment::Regular(_) => panic!("Expected Special segment"),
        }
        match &segments[1] {
            TextSegment::Regular(s) => assert_eq!(s, "user\nHello"),
            TextSegment::Special(_) => panic!("Expected Regular segment"),
        }
        match &segments[2] {
            TextSegment::Special(id) => assert_eq!(*id, 101),
            TextSegment::Regular(_) => panic!("Expected Special segment"),
        }
    }
}
