//! Token generation for OwnedQuantizedModel
//!
//! Contains generate, generate_with_cache, generate_with_cache_streaming,
//! generate_with_scratch, and sampling methods.

use crate::error::{RealizarError, Result};
use crate::gguf::ops;
use crate::gguf::{
    InferenceScratchBuffer, OwnedQuantizedKVCache, OwnedQuantizedModel, QuantizedGenerateConfig,
};
#[cfg(feature = "gpu")]
use crate::gguf::DispatchMetrics;
use rand::Rng;

impl OwnedQuantizedModel {
    /// Get most likely next token
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails
    pub fn predict_next(&self, token_ids: &[u32]) -> Result<u32> {
        let logits = self.forward(token_ids)?;
        let (max_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Empty logits".to_string(),
            })?;
        Ok(max_idx as u32)
    }

    /// Generate tokens using fused Q4_K operations (IMP-100)
    ///
    /// This is the HTTP serving entry point for quantized inference.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails
    pub fn generate(&self, prompt: &[u32], config: &QuantizedGenerateConfig) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let mut tokens = prompt.to_vec();
        let max_len = prompt.len() + config.max_tokens;

        for _ in 0..config.max_tokens {
            // Forward pass with fused Q4_K ops (1.37x faster)
            let logits = self.forward(&tokens)?;

            // Sample next token
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                // Greedy decoding
                Self::argmax(&logits)
            } else {
                // Temperature + top-k sampling
                Self::sample_topk(&logits, config.temperature, config.top_k)
            };

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // Check max length
            if tokens.len() >= max_len {
                break;
            }
        }

        Ok(tokens)
    }

    /// Greedy argmax over logits
    pub(crate) fn argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx as u32)
    }

    /// Top-k sampling with temperature
    pub fn sample_topk(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
        // Apply temperature
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

        // Get top-k indices
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k);

        // Softmax over top-k
        let max_val = indexed.first().map_or(0.0, |(_, v)| *v);
        let exp_sum: f32 = indexed.iter().map(|(_, v)| (v - max_val).exp()).sum();
        let probs: Vec<(usize, f32)> = indexed
            .iter()
            .map(|(i, v)| (*i, (v - max_val).exp() / exp_sum))
            .collect();

        // Sample from probability distribution with proper randomness
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();

        let mut cumulative = 0.0;
        for &(idx, prob) in &probs {
            cumulative += prob;
            if cumulative >= r {
                return idx as u32;
            }
        }

        probs.last().map_or(0, |(idx, _)| *idx as u32)
    }

    /// Generate tokens using KV cache for efficient autoregressive decoding (IMP-101)
    ///
    /// This is O(n) per token instead of O(n²) due to KV cache reuse.
    ///
    /// # Arguments
    /// * `prompt` - Input token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if forward pass fails
    pub fn generate_with_cache(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let max_seq_len = prompt.len() + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, max_seq_len);
        let mut tokens = prompt.to_vec();

        // Process prompt tokens (prefill), keeping the logits from the last position
        // The logits from processing token[n-1] at position n-1 predict token[n]
        let mut logits = Vec::new();
        for (pos, &token_id) in prompt.iter().enumerate() {
            logits = self.forward_single_with_cache(token_id, &mut cache, pos)?;
        }

        // Generate new tokens
        // First iteration uses logits from prefill, subsequent use logits from forward pass
        for gen_idx in 0..config.max_tokens {
            // DEBUG: Print logits info for first generated token
            if gen_idx == 0 && std::env::var("REALIZAR_DEBUG_LOGITS").is_ok() {
                let sum: f32 = logits.iter().sum();
                let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let min_val = logits.iter().copied().fold(f32::INFINITY, f32::min);
                let top_5: Vec<(usize, f32)> = {
                    let mut indexed: Vec<_> =
                        logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                    indexed.sort_by(|(_, a), (_, b)| {
                        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    indexed.into_iter().take(5).collect()
                };
                eprintln!(
                    "[DEBUG-LOGITS] len={}, sum={:.4}, min={:.4}, max={:.4}",
                    logits.len(),
                    sum,
                    min_val,
                    max_val
                );
                eprintln!("[DEBUG-LOGITS] top 5 token ids and logits: {:?}", top_5);
                eprintln!(
                    "[DEBUG-LOGITS] logits[0..5]: {:?}",
                    &logits[..5.min(logits.len())]
                );
            }

            // Sample next token
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                ops::argmax(&logits)
            } else {
                crate::gguf::OwnedQuantizedModel::sample_topk(&logits, config.temperature, config.top_k)
            };

            // DEBUG: Print selected token
            if gen_idx == 0 && std::env::var("REALIZAR_DEBUG_LOGITS").is_ok() {
                eprintln!(
                    "[DEBUG-LOGITS] selected token: {} (logit={:.4})",
                    next_token,
                    logits.get(next_token as usize).copied().unwrap_or(f32::NAN)
                );
            }

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // Check max length
            if tokens.len() >= max_seq_len {
                break;
            }

            // Get logits for next iteration by forwarding the newly sampled token
            // Position is prompt.len() + gen_idx (where token was just added)
            let position = prompt.len() + gen_idx;
            logits = self.forward_single_with_cache(next_token, &mut cache, position)?;
        }

        Ok(tokens)
    }

    /// Generate tokens with streaming callback (PMAT-087)
    ///
    /// Same as `generate_with_cache` but calls `on_token` after each token
    /// is generated, enabling true streaming to clients.
    ///
    /// # Arguments
    /// * `prompt` - Input token IDs
    /// * `config` - Generation configuration
    /// * `on_token` - Callback called for each generated token. Return `false` to stop.
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if generation fails
    pub fn generate_with_cache_streaming<F>(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let max_seq_len = prompt.len() + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, max_seq_len);
        let mut tokens = prompt.to_vec();

        // Process prompt tokens (prefill)
        let mut logits = Vec::new();
        for (pos, &token_id) in prompt.iter().enumerate() {
            logits = self.forward_single_with_cache(token_id, &mut cache, pos)?;
        }

        // Generate new tokens with streaming
        for gen_idx in 0..config.max_tokens {
            // Sample next token
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                ops::argmax(&logits)
            } else {
                crate::gguf::OwnedQuantizedModel::sample_topk(&logits, config.temperature, config.top_k)
            };

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // PMAT-087: Call streaming callback - stop if it returns false
            if !on_token(next_token) {
                break;
            }

            // Check max length
            if tokens.len() >= max_seq_len {
                break;
            }

            // Get logits for next iteration
            let position = prompt.len() + gen_idx;
            logits = self.forward_single_with_cache(next_token, &mut cache, position)?;
        }

        Ok(tokens)
    }

    /// Generate tokens with zero-allocation inference (IMP-131)
    ///
    /// This is the highest-performance generation path. Uses pre-allocated
    /// scratch buffers to eliminate per-token allocations, providing ~3-4x
    /// speedup over allocating variants.
    ///
    /// Performance characteristics:
    /// - Single allocation at start (scratch buffer + KV cache)
    /// - Zero allocations per generated token
    /// - ~500KB saved per token for TinyLlama-1.1B
    ///
    /// # Arguments
    /// * `prompt` - Input token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if forward pass fails
    pub fn generate_with_scratch(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let max_seq_len = prompt.len() + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, max_seq_len);
        let mut scratch = InferenceScratchBuffer::from_config(&self.config);
        let mut tokens = prompt.to_vec();

        // Process prompt tokens (prefill) - uses scratch buffers
        for (pos, &token_id) in prompt.iter().enumerate() {
            self.forward_single_with_scratch(token_id, &mut cache, pos, &mut scratch)?;
        }

        // Generate new tokens - zero allocations per token
        // PAR-126: Fixed loop structure to match generate_with_cache:
        // 1. Sample from current logits (prefill on first iter, previous forward otherwise)
        // 2. Then run forward on the new token to get logits for next iteration
        for gen_idx in 0..config.max_tokens {
            // Sample next token from current logits (prefill logits on first iter)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                ops::argmax(&scratch.logits)
            } else {
                crate::gguf::OwnedQuantizedModel::sample_topk(&scratch.logits, config.temperature, config.top_k)
            };

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // Check max length
            if tokens.len() >= max_seq_len {
                break;
            }

            // Get logits for next iteration by forwarding the new token
            let position = prompt.len() + gen_idx;
            self.forward_single_with_scratch(next_token, &mut cache, position, &mut scratch)?;
        }

        Ok(tokens)
    }

    /// Generate tokens with adaptive CPU/GPU attention (IMP-125)
    ///
    /// This variant of `generate_with_cache` uses `forward_single_with_cache_adaptive`
    /// to automatically select between CPU and GPU backends based on cache length.
    /// It also records dispatch decisions to the provided metrics tracker.
    ///
    /// # Arguments
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    /// * `metrics` - Dispatch metrics tracker for CPU/GPU decision recording
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if forward pass fails
    #[cfg(feature = "gpu")]
    pub fn generate_with_cache_adaptive(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        metrics: &std::sync::Arc<DispatchMetrics>,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let max_seq_len = prompt.len() + config.max_tokens;
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, max_seq_len);
        let mut tokens = prompt.to_vec();

        // Process prompt tokens (prefill) with adaptive attention
        // Keep the logits from the last position for the first generated token
        let mut logits = Vec::new();
        for (pos, &token_id) in prompt.iter().enumerate() {
            logits = self.forward_single_with_cache_adaptive(token_id, &mut cache, pos, metrics)?;
        }

        // Generate new tokens with adaptive attention
        for gen_idx in 0..config.max_tokens {
            // Sample next token from current logits
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                ops::argmax(&logits)
            } else {
                crate::gguf::OwnedQuantizedModel::sample_topk(&logits, config.temperature, config.top_k)
            };

            // Check stop condition
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);

            // Check max length
            if tokens.len() >= max_seq_len {
                break;
            }

            // Get logits for next iteration by forwarding the newly sampled token
            let position = prompt.len() + gen_idx;
            logits =
                self.forward_single_with_cache_adaptive(next_token, &mut cache, position, metrics)?;
        }

        Ok(tokens)
    }
}
