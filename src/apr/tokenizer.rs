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

use std::collections::HashMap;
use super::AprV2Model;

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
}

impl BpeTokenizer {
    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        bpe_encode(text, &self.token_to_id, &self.merge_rules)
    }

    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[u32]) -> String {
        AprV2Model::decode_tokens(&self.id_to_token, token_ids)
    }
}

/// Byte-level BPE encoding
pub(crate) fn bpe_encode(text: &str, vocab: &HashMap<String, u32>, merges: &[(String, String)]) -> Vec<u32> {
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

