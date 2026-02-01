//! APR Transformer Generation (PMAT-COMPLY)
//!
//! Extracted from mod.rs for file health compliance.
//! Token generation with KV cache support.

use super::{AprKVCache, AprTransformer, GenerateConfig};
use crate::error::{RealizarError, Result};

/// Common EOS token IDs across model families
const EOS_TOKENS: [u32; 4] = [0, 2, 151645, 151643];

/// Check if a token is an end-of-sequence marker
#[inline]
fn is_eos_token(token: u32) -> bool {
    EOS_TOKENS.contains(&token)
}

/// Sample the next token from logits using temperature scaling
fn sample_from_logits(logits: &[f32], temperature: f32) -> u32 {
    if temperature == 0.0 {
        // Greedy: pick argmax
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx as u32)
    } else {
        // Temperature-scaled sampling (currently picks argmax of probs)
        let scaled: Vec<f32> = logits.iter().map(|l| l / temperature).collect();
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = scaled.iter().map(|s| (s - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|e| e / sum).collect();
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx as u32)
    }
}

/// Generate tokens using KV cache for efficiency (Y4)
///
/// # Arguments
///
/// * `model` - The APR transformer model
/// * `prompt` - Initial token IDs
/// * `config` - Generation configuration
///
/// # Returns
///
/// Generated token sequence (including prompt)
///
/// # Errors
///
/// Returns error if prompt is empty or forward pass fails.
pub fn generate_with_cache(
    model: &AprTransformer,
    prompt: &[u32],
    config: &GenerateConfig,
) -> Result<Vec<u32>> {
    if prompt.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: "Prompt cannot be empty".to_string(),
        });
    }

    let mut cache = AprKVCache::new(&model.config);
    let mut output = prompt.to_vec();

    // PMAT-103 FIX: Process prompt tokens and KEEP the logits from the last one.
    // Previously we threw away all logits (`let _ = ...`) and then reprocessed
    // the last prompt token at the same position, corrupting the KV cache.
    let mut logits = Vec::new();

    // PMAT-103 TRACE: Measure per-token timing to verify O(n) vs O(n²)
    let trace_enabled = std::env::var("REALIZE_TRACE").is_ok();
    if trace_enabled {
        eprintln!("[TRACE] Processing {} prompt tokens...", prompt.len());
    }

    for (pos, &token) in prompt.iter().enumerate() {
        let start = std::time::Instant::now();
        logits = model.forward_with_cache(token, &mut cache, pos)?;
        if trace_enabled {
            eprintln!("[TRACE] Prompt token {}: {:?}", pos, start.elapsed());
        }
    }

    // Generate new tokens using the logits we already have
    for i in 0..config.max_tokens {
        let next_token = sample_from_logits(&logits, config.temperature);
        output.push(next_token);

        if is_eos_token(next_token) {
            break;
        }

        // If we need more tokens, process this one to get logits for the next
        if i < config.max_tokens - 1 {
            // Position is output.len() - 1 = prompt.len() + (i + 1) - 1 = prompt.len() + i
            let start = std::time::Instant::now();
            logits = model.forward_with_cache(next_token, &mut cache, output.len() - 1)?;
            if trace_enabled {
                eprintln!(
                    "[TRACE] Gen token {} (pos {}): {:?}",
                    i,
                    output.len() - 1,
                    start.elapsed()
                );
            }
        }
    }

    if trace_enabled {
        eprintln!(
            "[TRACE] Generation complete. Total output tokens: {}",
            output.len()
        );
    }

    Ok(output)
}
