
impl GGUFModel {

    /// Get RMSNorm epsilon from metadata
    /// Different models use different values (LLaMA: 1e-5, Qwen2: 1e-6)
    pub fn rms_epsilon(&self) -> Option<f32> {
        let arch = self.architecture()?;
        let key = format!("{}.attention.layer_norm_rms_epsilon", arch);
        if let Some(GGUFValue::Float32(eps)) = self.metadata.get(&key) {
            Some(*eps)
        } else {
            None
        }
    }

    /// Get RoPE type from metadata or infer from architecture
    /// Returns: 0 = NORM (adjacent pairs), 2 = NEOX (split halves)
    /// Per llama.cpp: LLAMA_ROPE_TYPE_NORM = 0, LLAMA_ROPE_TYPE_NEOX = 2
    ///
    /// Architecture-based inference matches llama.cpp's llama-model.cpp:7763-7811
    pub fn rope_type(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{}.rope.scaling.type", arch);
        // Try rope type from scaling type first
        if let Some(GGUFValue::String(s)) = self.metadata.get(&key) {
            match s.as_str() {
                "none" | "linear" => return Some(0), // NORM style
                "yarn" | "neox" => return Some(2),   // NEOX style
                _ => {},
            }
        }
        // Infer rope type from architecture (matches llama.cpp llama-model.cpp:7763-7811)
        // NEOX style (type 2): pairs offset by n_rot/2
        let arch_lower = arch.to_lowercase();
        let neox_architectures = [
            "qwen",
            "qwen2",
            "qwen3",
            "stablelm",
            "phi2",
            "phi3",
            "gemma",
            "gemma2",
            "gemma3",
            "starcoder2",
            "gptneox",
            "falcon",
            "codeshell",
            "orion",
            "bert",
            "nomic-bert",
            "dbrx",
            "olmo2",
            "olmoe",
            "plamo",
            "plamo2",
            "openelm",
            "exaone",
            "minicpm3",
            "nemotron",
            "internlm2",
            "deepseek2",
        ];
        for neox_arch in neox_architectures {
            if arch_lower.contains(neox_arch) {
                return Some(2); // NEOX style
            }
        }
        // NORM style (type 0): adjacent pairs - default for LLaMA, TinyLlama
        Some(0)
    }

    /// Get BOS (beginning of sentence) token ID
    #[must_use]
    pub fn bos_token_id(&self) -> Option<u32> {
        if let Some(GGUFValue::UInt32(id)) = self.metadata.get("tokenizer.ggml.bos_token_id") {
            Some(*id)
        } else {
            None
        }
    }

    /// Get EOS (end of sentence) token ID
    #[must_use]
    pub fn eos_token_id(&self) -> Option<u32> {
        if let Some(GGUFValue::UInt32(id)) = self.metadata.get("tokenizer.ggml.eos_token_id") {
            Some(*id)
        } else {
            None
        }
    }

    /// Get vocabulary tokens from metadata
    ///
    /// Returns the token strings indexed by token ID.
    /// Uses "tokenizer.ggml.tokens" key from GGUF metadata.
    #[must_use]
    pub fn vocabulary(&self) -> Option<Vec<String>> {
        if let Some(GGUFValue::Array(arr)) = self.metadata.get("tokenizer.ggml.tokens") {
            let tokens: Vec<String> = arr
                .iter()
                .filter_map(|v| {
                    if let GGUFValue::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .collect();
            if tokens.is_empty() {
                None
            } else {
                Some(tokens)
            }
        } else {
            None
        }
    }

    /// Decode token IDs to text using vocabulary
    ///
    /// Returns decoded string. Unknown tokens are replaced with "�".
    /// Handles BPE markers:
    /// - GPT-2 style: Ġ (U+0120) → space, Ċ (U+010A) → newline
    /// - SentencePiece: ▁ (U+2581) → space
    /// - Byte tokens: <0xHH> → actual byte value
    #[must_use]
    pub fn decode(&self, token_ids: &[u32]) -> String {
        if let Some(vocab) = self.vocabulary() {
            // Detect tokenizer type from metadata
            let is_gpt2_style = self
                .metadata
                .get("tokenizer.ggml.model")
                .is_some_and(|v| matches!(v, GGUFValue::String(s) if s == "gpt2" || s == "bpe"));

            // Collect raw tokens and convert byte tokens to actual bytes
            let mut bytes: Vec<u8> = Vec::new();

            for &id in token_ids {
                let token = vocab
                    .get(id as usize)
                    .map_or("�", std::string::String::as_str);

                // Check if this is a byte token like <0xE6>
                if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
                    if let Ok(byte_val) = u8::from_str_radix(token.get(3..5).expect("byte token <0xNN> has len 6, indices 3..5 always valid"), 16) {
                        bytes.push(byte_val);
                        continue;
                    }
                }

                // For GPT-2 style tokenizers, decode byte-level BPE properly
                // Each unicode character in the token represents a raw byte
                if is_gpt2_style {
                    for c in token.chars() {
                        if let Some(byte) = gpt2_unicode_to_byte(c) {
                            bytes.push(byte);
                        }
                    }
                } else {
                    // SentencePiece style - tokens are regular strings
                    bytes.extend_from_slice(token.as_bytes());
                }
            }

            // Decode bytes as UTF-8 (lossy for invalid sequences)
            let raw = String::from_utf8_lossy(&bytes).into_owned();

            // Post-process BPE markers (only for SentencePiece, GPT-2 already handled)
            if !is_gpt2_style {
                raw.replace('▁', " ") // SentencePiece word boundary
            } else {
                raw
            }
        } else {
            // Fallback to ASCII if no vocabulary
            token_ids
                .iter()
                .map(|&t| char::from_u32(t.min(127)).unwrap_or('?'))
                .collect()
        }
    }

    /// Encode text to token IDs using vocabulary
    ///
    /// Uses greedy longest-match tokenization with special token priority.
    /// Returns None if no vocabulary is available.
    ///
    /// Supports both tokenizer types:
    /// - SentencePiece (llama): Uses `▁` (U+2581) for word boundaries
    /// - GPT-2 (qwen2, gpt2): Uses `Ġ` (U+0120) for space prefixes
    #[must_use]
    pub fn encode(&self, text: &str) -> Option<Vec<u32>> {
        let vocab = self.vocabulary()?;

        // Build reverse lookup: token string -> token ID
        let token_to_id: std::collections::HashMap<&str, u32> = vocab
            .iter()
            .enumerate()
            .map(|(id, token)| (token.as_str(), id as u32))
            .collect();

        // Identify special tokens (high-ID tokens with <|...|> pattern)
        // These need priority matching to avoid being split by greedy algorithm
        let special_tokens: Vec<(&str, u32)> = vocab
            .iter()
            .enumerate()
            .filter(|(id, tok)| *id >= 151643 && tok.starts_with("<|") && tok.ends_with("|>"))
            .map(|(id, tok)| (tok.as_str(), id as u32))
            .collect();

        // Detect tokenizer type from metadata
        // GPT-2 style uses Ġ (U+0120), SentencePiece uses ▁ (U+2581)
        let is_gpt2_style = self
            .metadata
            .get("tokenizer.ggml.model")
            .is_some_and(|v| matches!(v, GGUFValue::String(s) if s == "gpt2" || s == "bpe"));

        let space_char = if is_gpt2_style { '\u{0120}' } else { '▁' };

        // Split text on special tokens first, preserving them
        let mut segments: Vec<(bool, &str)> = Vec::new(); // (is_special, text)
        let mut text_remaining = text;
        while !text_remaining.is_empty() {
            // Find earliest special token match
            let mut earliest_match: Option<(usize, &str, u32)> = None;
            for &(special_tok, special_id) in &special_tokens {
                if let Some(pos) = text_remaining.find(special_tok) {
                    if earliest_match.is_none()
                        || pos < earliest_match.as_ref().map_or(usize::MAX, |m| m.0)
                    {
                        earliest_match = Some((pos, special_tok, special_id));
                    }
                }
            }

            if let Some((pos, special_tok, _)) = earliest_match {
                if pos > 0 {
                    segments.push((false, &text_remaining[..pos]));
                }
                segments.push((true, special_tok));
                text_remaining = &text_remaining[pos + special_tok.len()..];
            } else {
                segments.push((false, text_remaining));
                break;
            }
        }

        let mut tokens = Vec::new();

        for (is_special, segment) in segments {
            if is_special {
                // Direct lookup for special token
                if let Some(&id) = token_to_id.get(segment) {
                    tokens.push(id);
                }
                continue;
            }

            // Process non-special segment with character replacement
            let text_with_prefix = if is_gpt2_style {
                segment.to_string()
            } else if segment.starts_with(' ') {
                segment.to_string()
            } else {
                format!(" {}", segment)
            };

            let processed = if is_gpt2_style {
                text_with_prefix
                    .replace(' ', &space_char.to_string())
                    .replace('\n', "\u{010A}") // Ċ = GPT-2 newline
            } else {
                text_with_prefix.replace(' ', &space_char.to_string())
            };

            let mut remaining = processed.as_str();

            while !remaining.is_empty() {
                // Greedy longest match using character boundaries (not byte indices)
                let mut best_byte_len = 0;
                let mut best_id = None;

                // Collect character byte offsets for proper slicing
                let char_indices: Vec<usize> = remaining
                    .char_indices()
                    .map(|(i, _)| i)
                    .chain(std::iter::once(remaining.len()))
                    .collect();

                // Try all prefixes up to 32 chars (reasonable max token length)
                for char_count in 1..=char_indices.len().saturating_sub(1).min(32) {
                    let byte_end = char_indices[char_count];
                    let prefix = &remaining[..byte_end];
                    if let Some(&id) = token_to_id.get(prefix) {
                        best_byte_len = byte_end;
                        best_id = Some(id);
                    }
                }

                if let Some(id) = best_id {
                    tokens.push(id);
                    remaining = &remaining[best_byte_len..];
                } else {
                    // No match found - try single UTF-8 char as byte tokens
                    // SAFETY: remaining is non-empty (loop condition guarantees this)
                    let ch = remaining
                        .chars()
                        .next()
                        .expect("loop invariant: remaining non-empty");
                    let ch_len = ch.len_utf8();

                    // Look for byte tokens like <0x48> for 'H'
                    for byte in remaining[..ch_len].bytes() {
                        let byte_token = format!("<0x{:02X}>", byte);
                        if let Some(&id) = token_to_id.get(byte_token.as_str()) {
                            tokens.push(id);
                        } else {
                            // Unknown byte - use a common unknown token ID (usually 0 or 1)
                            tokens.push(0);
                        }
                    }
                    remaining = &remaining[ch_len..];
                }
            }
        }

        Some(tokens)
    }
}

use crate::gguf::{
    OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel, OwnedQuantizedTensor,
    QuantizedGGUFTransformer,
};

include!("loader_part_02_part_02.rs");
include!("loader_part_02_part_03.rs");
