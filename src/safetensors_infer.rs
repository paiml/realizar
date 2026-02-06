//! SafeTensors Inference Support (PAR-301)
//!
//! Provides SafeTensors model loading and inference for HuggingFace models.
//!
//! ## Architecture
//!
//! SafeTensors files contain only tensor weights, so we need:
//! - `config.json` for model architecture (hidden_size, num_layers, etc.)
//! - `tokenizer.json` for text tokenization
//!
//! The converter loads these from sibling files and builds an AprTransformer.

use crate::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
use crate::error::{RealizarError, Result};
use crate::safetensors::validation::ValidatedAprTransformer;
use crate::safetensors::{
    MappedSafeTensorsModel, SafetensorsConfig, SafetensorsTensorInfo,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::safetensors::ShardedSafeTensorsModel;
use std::path::Path;

/// Trait abstracting tensor access for both single-file and sharded SafeTensors models.
///
/// This enables `SafetensorsToAprConverter` to work identically with
/// `MappedSafeTensorsModel` (single file) and `ShardedSafeTensorsModel` (multi-shard).
pub(crate) trait TensorSource {
    fn get_tensor_auto(&self, name: &str) -> Result<Vec<f32>>;
    fn has_tensor(&self, name: &str) -> bool;
    fn tensor_names(&self) -> Vec<&str>;
    fn get_tensor_info(&self, name: &str) -> Option<&SafetensorsTensorInfo>;
}

#[cfg(not(target_arch = "wasm32"))]
impl TensorSource for MappedSafeTensorsModel {
    fn get_tensor_auto(&self, name: &str) -> Result<Vec<f32>> {
        self.get_tensor_auto(name)
    }
    fn has_tensor(&self, name: &str) -> bool {
        self.has_tensor(name)
    }
    fn tensor_names(&self) -> Vec<&str> {
        self.tensor_names()
    }
    fn get_tensor_info(&self, name: &str) -> Option<&SafetensorsTensorInfo> {
        self.get_tensor_info(name)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl TensorSource for ShardedSafeTensorsModel {
    fn get_tensor_auto(&self, name: &str) -> Result<Vec<f32>> {
        self.get_tensor_auto(name)
    }
    fn has_tensor(&self, name: &str) -> bool {
        self.has_tensor(name)
    }
    fn tensor_names(&self) -> Vec<&str> {
        self.tensor_names()
    }
    fn get_tensor_info(&self, name: &str) -> Option<&SafetensorsTensorInfo> {
        self.get_tensor_info(name)
    }
}

/// SafeTensors to APR Transformer converter
///
/// Converts HuggingFace SafeTensors models to APR Transformer format.
/// Supports BF16, F16, and F32 weights with automatic conversion to F32.
///
/// # Tensor Naming Conventions
///
/// Supports both HuggingFace and GGUF-style tensor naming:
/// - HuggingFace: `model.embed_tokens.weight`, `model.layers.{i}.self_attn.q_proj.weight`
/// - GGUF-style: `token_embd.weight`, `blk.{i}.attn_q.weight`
pub struct SafetensorsToAprConverter;

impl SafetensorsToAprConverter {
    /// Convert SafeTensors model to APR Transformer
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to model.safetensors file
    ///
    /// # Returns
    ///
    /// `AprTransformer` with F32 weights ready for inference
    ///
    /// # Errors
    ///
    /// Returns error if SafeTensors file, config.json, or required tensors are missing
    pub fn convert(model_path: &Path) -> Result<ValidatedAprTransformer> {
        // Load SafeTensors model using mmap for zero-copy access (T-QA-020)
        // This is critical for fast model loading - mmap is O(1) regardless of file size
        let st_model = MappedSafeTensorsModel::load(model_path)?;

        // Load config.json (required for architecture info)
        let config = SafetensorsConfig::load_from_sibling(model_path).ok_or_else(|| {
            RealizarError::UnsupportedOperation {
                operation: "safetensors_convert".to_string(),
                reason: "config.json not found (required for SafeTensors inference)".to_string(),
            }
        })?;

        Self::convert_from_source(&st_model, &config)
    }

    /// Convert a sharded SafeTensors model to APR Transformer (GH-213)
    ///
    /// # Arguments
    ///
    /// * `sharded` - Loaded sharded model (from index.json)
    /// * `config` - Model config.json
    ///
    /// # Errors
    ///
    /// Returns error if required tensors are missing from any shard
    #[cfg(not(target_arch = "wasm32"))]
    pub fn convert_sharded(
        sharded: &ShardedSafeTensorsModel,
        config: &SafetensorsConfig,
    ) -> Result<ValidatedAprTransformer> {
        Self::convert_from_source(sharded, config)
    }

    /// Core conversion logic shared between single-file and sharded paths
    fn convert_from_source<S: TensorSource>(
        source: &S,
        config: &SafetensorsConfig,
    ) -> Result<ValidatedAprTransformer> {
        // Extract architecture parameters
        let hidden_dim = config
            .hidden_size
            .ok_or_else(|| RealizarError::FormatError {
                reason: "config.json missing hidden_size".to_string(),
            })?;
        let num_layers = config
            .num_hidden_layers
            .ok_or_else(|| RealizarError::FormatError {
                reason: "config.json missing num_hidden_layers".to_string(),
            })?;
        let num_heads = config
            .num_attention_heads
            .ok_or_else(|| RealizarError::FormatError {
                reason: "config.json missing num_attention_heads".to_string(),
            })?;
        let num_kv_heads = config.num_kv_heads();
        let vocab_size = config
            .vocab_size
            .ok_or_else(|| RealizarError::FormatError {
                reason: "config.json missing vocab_size".to_string(),
            })?;
        let intermediate_dim = config.intermediate_size.unwrap_or(hidden_dim * 4);
        let context_length = config.max_position_embeddings.unwrap_or(2048);
        let rope_theta = config.rope_theta.unwrap_or(10000.0);
        let eps = config.rms_norm_eps.unwrap_or(1e-6);
        let architecture = config.architecture();

        // Build transformer config
        let apr_config = AprTransformerConfig {
            architecture,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length,
            rope_theta,
            eps,
        };

        // Extract embeddings (HuggingFace: model.embed_tokens.weight, GGUF: token_embd.weight)
        let token_embedding = Self::get_tensor_with_fallback_generic(
            source,
            "model.embed_tokens.weight",
            "token_embd.weight",
        )?;

        // Extract output norm (HuggingFace: model.norm.weight, GGUF: output_norm.weight)
        let output_norm_weight = Self::get_tensor_with_fallback_generic(
            source,
            "model.norm.weight",
            "output_norm.weight",
        )?;

        // F-GT-002 FIX: Check tie_word_embeddings config FIRST, not just tensor existence
        // When tie_word_embeddings=true, HuggingFace may store a placeholder lm_head.weight
        // that's all zeros - we MUST use the embedding matrix instead!
        let use_tied_embeddings = config.tie_word_embeddings.unwrap_or(false);

        let lm_head_weight = if use_tied_embeddings {
            Self::transpose_weight(&token_embedding, vocab_size, hidden_dim)
        } else if Self::has_tensor_with_fallback_generic(source, "lm_head.weight", "output.weight")
        {
            let raw = Self::get_tensor_with_fallback_generic(
                source,
                "lm_head.weight",
                "output.weight",
            )?;
            Self::transpose_weight(&raw, vocab_size, hidden_dim)
        } else {
            // Fallback: assume tied if no lm_head tensor exists
            Self::transpose_weight(&token_embedding, vocab_size, hidden_dim)
        };

        // Extract layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = Self::extract_layer_generic(
                source,
                i,
                hidden_dim,
                num_heads,
                num_kv_heads,
                intermediate_dim,
            )?;
            layers.push(layer);
        }

        let transformer = AprTransformer {
            config: apr_config,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias: None,
            lm_head_weight,
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        };

        // PMAT-235: Validate ALL tensors at construction time
        ValidatedAprTransformer::validate(transformer).map_err(Into::into)
    }

    /// Extract a single transformer layer from SafeTensors (MappedSafeTensorsModel)
    fn extract_layer(
        st_model: &MappedSafeTensorsModel,
        layer_idx: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
    ) -> Result<AprTransformerLayer> {
        Self::extract_layer_generic(st_model, layer_idx, hidden_dim, num_heads, num_kv_heads, intermediate_dim)
    }

    /// Extract a single transformer layer from any TensorSource (GH-213)
    fn extract_layer_generic<S: TensorSource>(
        source: &S,
        layer_idx: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
    ) -> Result<AprTransformerLayer> {
        // Support both HuggingFace and GGUF naming conventions
        let hf_prefix = format!("model.layers.{layer_idx}");
        let gguf_prefix = format!("blk.{layer_idx}");

        let attn_norm_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.input_layernorm.weight"),
            &format!("{gguf_prefix}.attn_norm.weight"),
        )?;

        let q_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.self_attn.q_proj.weight"),
            &format!("{gguf_prefix}.attn_q.weight"),
        )?;
        let k_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.self_attn.k_proj.weight"),
            &format!("{gguf_prefix}.attn_k.weight"),
        )?;
        let v_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.self_attn.v_proj.weight"),
            &format!("{gguf_prefix}.attn_v.weight"),
        )?;

        let head_dim = hidden_dim / num_heads;
        let kv_dim = head_dim * num_kv_heads;
        let qkv_weight =
            Self::concat_qkv_transposed(&q_weight, &k_weight, &v_weight, hidden_dim, kv_dim);

        // QKV bias (optional)
        let qkv_bias =
            Self::try_concat_qkv_bias_dual_generic(source, &hf_prefix, &gguf_prefix, hidden_dim, kv_dim);

        let attn_output_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.self_attn.o_proj.weight"),
            &format!("{gguf_prefix}.attn_output.weight"),
        )?;
        let attn_output_weight = Self::transpose_weight(&attn_output_raw, hidden_dim, hidden_dim);

        let ffn_norm_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.post_attention_layernorm.weight"),
            &format!("{gguf_prefix}.ffn_norm.weight"),
        )?;

        let ffn_gate_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.mlp.gate_proj.weight"),
            &format!("{gguf_prefix}.ffn_gate.weight"),
        )?;
        let ffn_gate_weight = Self::transpose_weight(&ffn_gate_raw, intermediate_dim, hidden_dim);

        let ffn_up_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.mlp.up_proj.weight"),
            &format!("{gguf_prefix}.ffn_up.weight"),
        )?;
        let ffn_up_weight = Self::transpose_weight(&ffn_up_raw, intermediate_dim, hidden_dim);

        let ffn_down_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.mlp.down_proj.weight"),
            &format!("{gguf_prefix}.ffn_down.weight"),
        )?;
        let ffn_down_weight = Self::transpose_weight(&ffn_down_raw, hidden_dim, intermediate_dim);

        Ok(AprTransformerLayer {
            attn_norm_weight,
            attn_norm_bias: None,
            qkv_weight,
            qkv_bias,
            attn_output_weight,
            attn_output_bias: None,
            ffn_gate_weight: Some(ffn_gate_weight),
            ffn_gate_bias: None,
            ffn_up_weight,
            ffn_up_bias: None,
            ffn_down_weight,
            ffn_down_bias: None,
            ffn_norm_weight: Some(ffn_norm_weight),
            ffn_norm_bias: None,
        })
    }

    /// Pass through weight in matvec-optimal [out_dim, in_dim] format
    ///
    /// PMAT-095 FIX: HuggingFace stores Linear weights as [out_features, in_features]
    /// which is EXACTLY what trueno's matvec needs! Previous implementation transposed
    /// twice (here and in matmul), causing O(n²) overhead per forward pass.
    ///
    /// Now we keep HuggingFace format directly - no transposition needed.
    #[allow(clippy::unused_self)]
    pub fn transpose_weight(weight: &[f32], _out_dim: usize, _in_dim: usize) -> Vec<f32> {
        // PMAT-095: Keep [out_dim, in_dim] format - no transposition!
        // This eliminates the 75x performance gap vs GGUF.
        weight.to_vec()
    }

    /// Concatenate Q, K, V weights into combined QKV tensor (matvec-optimal)
    ///
    /// PMAT-095 FIX: Keep [out_dim, in_dim] format from HuggingFace.
    /// For QKV, we concatenate along the output dimension:
    /// - Q: [hidden_dim, hidden_dim]
    /// - K: [kv_dim, hidden_dim]
    /// - V: [kv_dim, hidden_dim]
    ///
    /// Result: [hidden_dim + kv_dim + kv_dim, hidden_dim] in row-major
    pub fn concat_qkv_transposed(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        _hidden_dim: usize,
        _kv_dim: usize,
    ) -> Vec<f32> {
        // PMAT-095: Simple concatenation - weights are already in optimal layout
        // Concatenate [Q; K; V] along output dimension
        let mut qkv = Vec::with_capacity(q.len() + k.len() + v.len());
        qkv.extend_from_slice(q);
        qkv.extend_from_slice(k);
        qkv.extend_from_slice(v);
        qkv
    }

    /// Concatenate Q, K, V weights into combined QKV tensor (legacy, no transpose)
    fn concat_qkv(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        let mut qkv = Vec::with_capacity(q.len() + k.len() + v.len());
        qkv.extend_from_slice(q);
        qkv.extend_from_slice(k);
        qkv.extend_from_slice(v);
        qkv
    }

    /// Try to concatenate Q, K, V biases if they exist
    fn try_concat_qkv_bias(
        st_model: &MappedSafeTensorsModel,
        prefix: &str,
        hidden_dim: usize,
        kv_dim: usize,
    ) -> Option<Vec<f32>> {
        let q_bias = st_model
            .get_tensor_auto(&format!("{prefix}.self_attn.q_proj.bias"))
            .ok()?;
        let k_bias = st_model
            .get_tensor_auto(&format!("{prefix}.self_attn.k_proj.bias"))
            .ok()?;
        let v_bias = st_model
            .get_tensor_auto(&format!("{prefix}.self_attn.v_proj.bias"))
            .ok()?;

        let mut qkv_bias = Vec::with_capacity(hidden_dim + kv_dim + kv_dim);
        qkv_bias.extend_from_slice(&q_bias);
        qkv_bias.extend_from_slice(&k_bias);
        qkv_bias.extend_from_slice(&v_bias);

        Some(qkv_bias)
    }

    /// Try to concatenate Q, K, V biases with dual naming support
    fn try_concat_qkv_bias_dual(
        st_model: &MappedSafeTensorsModel,
        hf_prefix: &str,
        gguf_prefix: &str,
        hidden_dim: usize,
        kv_dim: usize,
    ) -> Option<Vec<f32>> {
        Self::try_concat_qkv_bias_dual_generic(st_model, hf_prefix, gguf_prefix, hidden_dim, kv_dim)
    }

    /// Generic version of QKV bias concatenation (GH-213)
    fn try_concat_qkv_bias_dual_generic<S: TensorSource>(
        source: &S,
        hf_prefix: &str,
        gguf_prefix: &str,
        hidden_dim: usize,
        kv_dim: usize,
    ) -> Option<Vec<f32>> {
        let q_bias = source
            .get_tensor_auto(&format!("{hf_prefix}.self_attn.q_proj.bias"))
            .ok()
            .or_else(|| {
                source
                    .get_tensor_auto(&format!("{gguf_prefix}.attn_q.bias"))
                    .ok()
            })?;
        let k_bias = source
            .get_tensor_auto(&format!("{hf_prefix}.self_attn.k_proj.bias"))
            .ok()
            .or_else(|| {
                source
                    .get_tensor_auto(&format!("{gguf_prefix}.attn_k.bias"))
                    .ok()
            })?;
        let v_bias = source
            .get_tensor_auto(&format!("{hf_prefix}.self_attn.v_proj.bias"))
            .ok()
            .or_else(|| {
                source
                    .get_tensor_auto(&format!("{gguf_prefix}.attn_v.bias"))
                    .ok()
            })?;

        let mut qkv_bias = Vec::with_capacity(hidden_dim + kv_dim + kv_dim);
        qkv_bias.extend_from_slice(&q_bias);
        qkv_bias.extend_from_slice(&k_bias);
        qkv_bias.extend_from_slice(&v_bias);

        Some(qkv_bias)
    }

    /// Get tensor with fallback to alternative naming conventions
    ///
    /// Tries HuggingFace naming first, then GGUF-style naming, then bare name.
    /// This enables loading SafeTensors files regardless of their origin.
    ///
    /// GH-196: Also tries stripping `model.` prefix for APR canonical names,
    /// and adds diagnostic tensor name listing on failure.
    fn get_tensor_with_fallback(
        st_model: &MappedSafeTensorsModel,
        hf_name: &str,
        gguf_name: &str,
    ) -> Result<Vec<f32>> {
        Self::get_tensor_with_fallback_generic(st_model, hf_name, gguf_name)
    }

    /// Generic version of tensor lookup with fallback naming (GH-213)
    fn get_tensor_with_fallback_generic<S: TensorSource>(
        source: &S,
        hf_name: &str,
        gguf_name: &str,
    ) -> Result<Vec<f32>> {
        // Try HuggingFace name first (e.g., "model.norm.weight")
        if let Ok(t) = source.get_tensor_auto(hf_name) {
            return Ok(t);
        }
        // Try GGUF name (e.g., "output_norm.weight")
        if let Ok(t) = source.get_tensor_auto(gguf_name) {
            return Ok(t);
        }
        // Try bare name without "model." prefix (APR canonical names)
        let bare_name = hf_name.strip_prefix("model.").unwrap_or(hf_name);
        if bare_name != hf_name {
            if let Ok(t) = source.get_tensor_auto(bare_name) {
                return Ok(t);
            }
        }

        // Diagnostic: list available tensor names for debugging
        let available = source.tensor_names();
        let sample: Vec<&str> = available.iter().take(5).copied().collect();
        Err(RealizarError::UnsupportedOperation {
            operation: "get_tensor_auto".to_string(),
            reason: format!(
                "Tensor not found with names: '{}', '{}', or '{}'. \
                 Available tensors ({} total): {:?}{}",
                hf_name,
                gguf_name,
                bare_name,
                available.len(),
                sample,
                if available.len() > 5 { ", ..." } else { "" }
            ),
        })
    }

    /// Check if tensor exists with either naming convention
    fn has_tensor_with_fallback(
        st_model: &MappedSafeTensorsModel,
        hf_name: &str,
        gguf_name: &str,
    ) -> bool {
        Self::has_tensor_with_fallback_generic(st_model, hf_name, gguf_name)
    }

    /// Generic version of tensor existence check (GH-213)
    fn has_tensor_with_fallback_generic<S: TensorSource>(
        source: &S,
        hf_name: &str,
        gguf_name: &str,
    ) -> bool {
        if source.has_tensor(hf_name) || source.has_tensor(gguf_name) {
            return true;
        }
        // GH-196: Also check bare name without "model." prefix
        let bare_name = hf_name.strip_prefix("model.").unwrap_or(hf_name);
        bare_name != hf_name && source.has_tensor(bare_name)
    }

    /// Get optional tensor with fallback naming
    fn get_optional_tensor_with_fallback(
        st_model: &MappedSafeTensorsModel,
        hf_name: &str,
        gguf_name: &str,
    ) -> Option<Vec<f32>> {
        st_model
            .get_tensor_auto(hf_name)
            .ok()
            .or_else(|| st_model.get_tensor_auto(gguf_name).ok())
            .or_else(|| {
                // GH-196: Try bare name without "model." prefix
                let bare_name = hf_name.strip_prefix("model.")?;
                st_model.get_tensor_auto(bare_name).ok()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_concat_qkv() {
        let q = vec![1.0, 2.0];
        let k = vec![3.0, 4.0];
        let v = vec![5.0, 6.0];
        let qkv = SafetensorsToAprConverter::concat_qkv(&q, &k, &v);
        assert_eq!(qkv, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // =========================================================================
    // Extended Coverage Tests (15+ tests ending with _ext_cov)
    // =========================================================================

    /// Generate valid non-zero F32 bytes for `count` elements.
    /// Uses sin pattern to pass all validation gates (density, L2, variation).
    fn valid_f32_bytes(count: usize) -> Vec<u8> {
        (0..count)
            .flat_map(|i| ((i as f32 * 0.01).sin() * 0.1 + 0.05).to_le_bytes())
            .collect()
    }

    /// Helper function to create a minimal SafeTensors file with given tensors
    fn create_safetensors_bytes(tensors: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
        use serde_json::json;

        // Calculate tensor data layout
        let mut tensor_entries = serde_json::Map::new();
        let mut offset = 0usize;

        for (name, dtype, shape, data) in tensors {
            let end = offset + data.len();
            tensor_entries.insert(
                (*name).to_string(),
                json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, end]
                }),
            );
            offset = end;
        }

        let json_obj = serde_json::Value::Object(tensor_entries);
        let json_bytes = json_obj.to_string().into_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(&json_bytes);

        // Append tensor data
        for (_, _, _, tensor_data) in tensors {
            data.extend_from_slice(tensor_data);
        }

        data
    }

    /// Helper to create config.json content
    fn create_config_json(
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        vocab_size: usize,
    ) -> String {
        format!(
            r#"{{
                "hidden_size": {},
                "num_hidden_layers": {},
                "num_attention_heads": {},
                "vocab_size": {},
                "intermediate_size": {},
                "max_position_embeddings": 2048,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama"
            }}"#,
            hidden_size,
            num_layers,
            num_heads,
            vocab_size,
            hidden_size * 4
        )
    }

    #[test]
    fn test_convert_file_not_found_ext_cov() {
        let result =
            SafetensorsToAprConverter::convert(Path::new("/nonexistent/model.safetensors"));
        assert!(result.is_err());
        // MappedSafeTensorsModel::load() returns UnsupportedOperation for file open errors
        if let Err(RealizarError::UnsupportedOperation { operation, reason }) = result {
            assert_eq!(operation, "open_safetensors");
            assert!(reason.contains("Failed to open file"));
        } else {
            panic!("Expected UnsupportedOperation error");
        }
    }

    #[test]
    fn test_convert_missing_config_json_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");

        // Create a minimal valid safetensors file
        let data = create_safetensors_bytes(&[]);
        std::fs::write(&model_path, data).expect("write safetensors");

        // No config.json file

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_err());
        if let Err(RealizarError::UnsupportedOperation { operation, reason }) = result {
            assert_eq!(operation, "safetensors_convert");
            assert!(reason.contains("config.json not found"));
        } else {
            panic!("Expected UnsupportedOperation error");
        }
    }

    #[test]
    fn test_convert_missing_hidden_size_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        // Create minimal safetensors
        let data = create_safetensors_bytes(&[]);
        std::fs::write(&model_path, data).expect("write safetensors");

        // Config missing hidden_size
        let config = r#"{"num_hidden_layers": 2, "num_attention_heads": 4, "vocab_size": 100}"#;
        std::fs::write(&config_path, config).expect("write config");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_err());
        if let Err(RealizarError::FormatError { reason }) = result {
            assert!(reason.contains("missing hidden_size"));
        } else {
            panic!("Expected FormatError for missing hidden_size");
        }
    }

    #[test]
    fn test_convert_missing_num_hidden_layers_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let data = create_safetensors_bytes(&[]);
        std::fs::write(&model_path, data).expect("write safetensors");

        // Config missing num_hidden_layers
        let config = r#"{"hidden_size": 64, "num_attention_heads": 4, "vocab_size": 100}"#;
        std::fs::write(&config_path, config).expect("write config");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_err());
        if let Err(RealizarError::FormatError { reason }) = result {
            assert!(reason.contains("missing num_hidden_layers"));
        } else {
            panic!("Expected FormatError for missing num_hidden_layers");
        }
    }

    #[test]
    fn test_convert_missing_num_attention_heads_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let data = create_safetensors_bytes(&[]);
        std::fs::write(&model_path, data).expect("write safetensors");

        // Config missing num_attention_heads
        let config = r#"{"hidden_size": 64, "num_hidden_layers": 2, "vocab_size": 100}"#;
        std::fs::write(&config_path, config).expect("write config");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_err());
        if let Err(RealizarError::FormatError { reason }) = result {
            assert!(reason.contains("missing num_attention_heads"));
        } else {
            panic!("Expected FormatError for missing num_attention_heads");
        }
    }

    #[test]
    fn test_convert_missing_vocab_size_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let data = create_safetensors_bytes(&[]);
        std::fs::write(&model_path, data).expect("write safetensors");

        // Config missing vocab_size
        let config = r#"{"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4}"#;
        std::fs::write(&config_path, config).expect("write config");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_err());
        if let Err(RealizarError::FormatError { reason }) = result {
            assert!(reason.contains("missing vocab_size"));
        } else {
            panic!("Expected FormatError for missing vocab_size");
        }
    }

    #[test]
    fn test_convert_missing_embed_tokens_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        // Safetensors without model.embed_tokens.weight
        let data = create_safetensors_bytes(&[]);
        std::fs::write(&model_path, data).expect("write safetensors");

        // Valid config
        let config = create_config_json(64, 1, 4, 100);
        std::fs::write(&config_path, config).expect("write config");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_err());
        // Should fail because model.embed_tokens.weight is missing
    }

    #[test]
    fn test_concat_qkv_empty_inputs_ext_cov() {
        let q: Vec<f32> = vec![];
        let k: Vec<f32> = vec![];
        let v: Vec<f32> = vec![];
        let qkv = SafetensorsToAprConverter::concat_qkv(&q, &k, &v);
        assert!(qkv.is_empty());
    }

    #[test]
    fn test_concat_qkv_single_elements_ext_cov() {
        let q = vec![1.0];
        let k = vec![2.0];
        let v = vec![3.0];
        let qkv = SafetensorsToAprConverter::concat_qkv(&q, &k, &v);
        assert_eq!(qkv, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_concat_qkv_large_arrays_ext_cov() {
        let q: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let k: Vec<f32> = (1000..2000).map(|i| i as f32).collect();
        let v: Vec<f32> = (2000..3000).map(|i| i as f32).collect();
        let qkv = SafetensorsToAprConverter::concat_qkv(&q, &k, &v);
        assert_eq!(qkv.len(), 3000);
        assert_eq!(qkv[0], 0.0);
        assert_eq!(qkv[1000], 1000.0);
        assert_eq!(qkv[2000], 2000.0);
        assert_eq!(qkv[2999], 2999.0);
    }

    #[test]
    fn test_concat_qkv_asymmetric_ext_cov() {
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![5.0, 6.0];
        let v = vec![7.0];
        let qkv = SafetensorsToAprConverter::concat_qkv(&q, &k, &v);
        assert_eq!(qkv, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_try_concat_qkv_bias_none_when_missing_ext_cov() {
        // Create safetensors model without any biases
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let data = create_safetensors_bytes(&[]);
        std::fs::write(&model_path, data).expect("write safetensors");
        let st_model = MappedSafeTensorsModel::load(&model_path).expect("load safetensors");

        let result =
            SafetensorsToAprConverter::try_concat_qkv_bias(&st_model, "model.layers.0", 64, 64);
        assert!(result.is_none());
    }

    #[test]
    fn test_try_concat_qkv_bias_partial_missing_ext_cov() {
        // Create safetensors with only q_proj.bias (missing k and v)
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let q_bias_data: Vec<u8> = (0..16).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let data = create_safetensors_bytes(&[(
            "model.layers.0.self_attn.q_proj.bias",
            "F32",
            &[4],
            &q_bias_data,
        )]);
        std::fs::write(&model_path, data).expect("write safetensors");
        let st_model = MappedSafeTensorsModel::load(&model_path).expect("load safetensors");

        // Should return None because k_bias and v_bias are missing
        let result =
            SafetensorsToAprConverter::try_concat_qkv_bias(&st_model, "model.layers.0", 4, 4);
        assert!(result.is_none());
    }

    #[test]
    fn test_try_concat_qkv_bias_all_present_ext_cov() {
        // Create F32 byte data for biases
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let q_bias_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let k_bias_data: Vec<u8> = [5.0f32, 6.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        let v_bias_data: Vec<u8> = [7.0f32, 8.0].iter().flat_map(|f| f.to_le_bytes()).collect();

        let data = create_safetensors_bytes(&[
            (
                "model.layers.0.self_attn.q_proj.bias",
                "F32",
                &[4],
                &q_bias_data,
            ),
            (
                "model.layers.0.self_attn.k_proj.bias",
                "F32",
                &[2],
                &k_bias_data,
            ),
            (
                "model.layers.0.self_attn.v_proj.bias",
                "F32",
                &[2],
                &v_bias_data,
            ),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");
        let st_model = MappedSafeTensorsModel::load(&model_path).expect("load safetensors");

        let result =
            SafetensorsToAprConverter::try_concat_qkv_bias(&st_model, "model.layers.0", 4, 2);
        assert!(result.is_some());
        let bias = result.expect("operation failed");
        assert_eq!(bias.len(), 8); // 4 + 2 + 2
        assert_eq!(bias[0], 1.0);
        assert_eq!(bias[4], 5.0);
        assert_eq!(bias[6], 7.0);
    }

    #[test]
    fn test_convert_defaults_intermediate_size_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        // Config without intermediate_size (should default to hidden_size * 4)
        let config = r#"{
            "hidden_size": 64,
            "num_hidden_layers": 0,
            "num_attention_heads": 4,
            "vocab_size": 100
        }"#;
        std::fs::write(&config_path, config).expect("write config");

        // Create safetensors with minimal required tensors for 0 layers
        let embed_data = valid_f32_bytes(100 * 64);
        let norm_data: Vec<u8> = (0..64).flat_map(|_| 1.0f32.to_le_bytes()).collect();

        let data = create_safetensors_bytes(&[
            ("model.embed_tokens.weight", "F32", &[100, 64], &embed_data),
            ("model.norm.weight", "F32", &[64], &norm_data),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok(), "convert failed: {:?}", result.err());
        let transformer = result.expect("operation failed");
        assert_eq!(transformer.config.intermediate_dim, 64 * 4);
    }

    #[test]
    fn test_convert_defaults_context_length_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        // Config without max_position_embeddings (should default to 2048)
        let config = r#"{
            "hidden_size": 64,
            "num_hidden_layers": 0,
            "num_attention_heads": 4,
            "vocab_size": 100,
            "intermediate_size": 256
        }"#;
        std::fs::write(&config_path, config).expect("write config");

        let embed_data = valid_f32_bytes(100 * 64);
        let norm_data: Vec<u8> = (0..64).flat_map(|_| 1.0f32.to_le_bytes()).collect();

        let data = create_safetensors_bytes(&[
            ("model.embed_tokens.weight", "F32", &[100, 64], &embed_data),
            ("model.norm.weight", "F32", &[64], &norm_data),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok(), "convert failed: {:?}", result.err());
        let transformer = result.expect("operation failed");
        assert_eq!(transformer.config.context_length, 2048);
    }

    #[test]
    fn test_convert_tied_embeddings_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let config = create_config_json(64, 0, 4, 100);
        std::fs::write(&config_path, config).expect("write config");

        // Create tensors WITHOUT lm_head.weight (tied embeddings)
        let embed_data: Vec<u8> = (0..(100 * 64))
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let norm_data: Vec<u8> = (0..64).flat_map(|_| 1.0f32.to_le_bytes()).collect();

        let data = create_safetensors_bytes(&[
            ("model.embed_tokens.weight", "F32", &[100, 64], &embed_data),
            ("model.norm.weight", "F32", &[64], &norm_data),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok());
        let transformer = result.expect("operation failed");

        // lm_head_weight should have same dimensions as token_embedding (tied or separate)
        // When tied: lm_head_weight.len() == token_embedding.len()
        // But they may not be equal if transposed or if implementation uses separate weights
        assert!(!transformer.lm_head_weight.is_empty());
        assert!(!transformer.token_embedding.is_empty());
    }

    #[test]
    fn test_convert_separate_lm_head_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let config = create_config_json(64, 0, 4, 100);
        std::fs::write(&config_path, config).expect("write config");

        // Create tensors WITH separate lm_head.weight
        let embed_data: Vec<u8> = (0..(100 * 64))
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let norm_data: Vec<u8> = (0..64).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        let lm_head_data: Vec<u8> = (0..(100 * 64))
            .flat_map(|i| ((i + 1000) as f32).to_le_bytes())
            .collect();

        let data = create_safetensors_bytes(&[
            ("model.embed_tokens.weight", "F32", &[100, 64], &embed_data),
            ("model.norm.weight", "F32", &[64], &norm_data),
            ("lm_head.weight", "F32", &[100, 64], &lm_head_data),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok());
        let transformer = result.expect("operation failed");

        // lm_head_weight should NOT equal token_embedding
        assert_ne!(
            transformer.lm_head_weight[0],
            transformer.token_embedding[0]
        );
    }

    #[test]
    fn test_convert_with_rope_theta_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        // Config with custom rope_theta
        let config = r#"{
            "hidden_size": 64,
            "num_hidden_layers": 0,
            "num_attention_heads": 4,
            "vocab_size": 100,
            "intermediate_size": 256,
            "rope_theta": 500000.0
        }"#;
        std::fs::write(&config_path, config).expect("write config");

        let embed_data = valid_f32_bytes(100 * 64);
        let norm_data: Vec<u8> = (0..64).flat_map(|_| 1.0f32.to_le_bytes()).collect();

        let data = create_safetensors_bytes(&[
            ("model.embed_tokens.weight", "F32", &[100, 64], &embed_data),
            ("model.norm.weight", "F32", &[64], &norm_data),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok(), "convert failed: {:?}", result.err());
        let transformer = result.expect("operation failed");
        assert!((transformer.config.rope_theta - 500000.0).abs() < 1.0);
    }

    #[test]
    fn test_convert_with_rms_norm_eps_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        // Config with custom rms_norm_eps
        let config = r#"{
            "hidden_size": 64,
            "num_hidden_layers": 0,
            "num_attention_heads": 4,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5
        }"#;
        std::fs::write(&config_path, config).expect("write config");

        let embed_data = valid_f32_bytes(100 * 64);
        let norm_data: Vec<u8> = (0..64).flat_map(|_| 1.0f32.to_le_bytes()).collect();

        let data = create_safetensors_bytes(&[
            ("model.embed_tokens.weight", "F32", &[100, 64], &embed_data),
            ("model.norm.weight", "F32", &[64], &norm_data),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok(), "convert failed: {:?}", result.err());
        let transformer = result.expect("operation failed");
        assert!((transformer.config.eps - 1e-5).abs() < 1e-9);
    }

    #[test]
    fn test_safetensors_to_apr_converter_struct_ext_cov() {
        // Test that SafetensorsToAprConverter is a unit struct
        let _converter = SafetensorsToAprConverter;
        // This just ensures the struct exists and can be instantiated
    }

    #[test]
    fn test_convert_architecture_from_config_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let config = create_config_json(64, 0, 4, 100);
        std::fs::write(&config_path, config).expect("write config");

        let embed_data = valid_f32_bytes(100 * 64);
        let norm_data: Vec<u8> = (0..64).flat_map(|_| 1.0f32.to_le_bytes()).collect();

        let data = create_safetensors_bytes(&[
            ("model.embed_tokens.weight", "F32", &[100, 64], &embed_data),
            ("model.norm.weight", "F32", &[64], &norm_data),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok(), "convert failed: {:?}", result.err());
        let transformer = result.expect("operation failed");
        assert_eq!(transformer.config.architecture, "LlamaForCausalLM");
    }

    /// Helper to create all layer tensors for a single transformer layer
    fn create_layer_tensors(
        layer_idx: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Vec<(&'static str, String, Vec<usize>, Vec<u8>)> {
        let prefix = format!("model.layers.{layer_idx}");

        // Calculate tensor sizes
        let attn_norm_size = hidden_dim;
        let q_size = hidden_dim * hidden_dim;
        let k_size = hidden_dim * hidden_dim;
        let v_size = hidden_dim * hidden_dim;
        let o_size = hidden_dim * hidden_dim;
        let ffn_norm_size = hidden_dim;
        let gate_size = hidden_dim * intermediate_dim;
        let up_size = hidden_dim * intermediate_dim;
        let down_size = intermediate_dim * hidden_dim;

        vec![
            (
                "attn_norm",
                format!("{prefix}.input_layernorm.weight"),
                vec![attn_norm_size],
                // Norm vectors: ValidatedVector only checks NaN/Inf/length, 1.0 is safe
                (0..attn_norm_size).flat_map(|_| 1.0f32.to_le_bytes()).collect(),
            ),
            (
                "q_proj",
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![hidden_dim, hidden_dim],
                valid_f32_bytes(q_size),
            ),
            (
                "k_proj",
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![hidden_dim, hidden_dim],
                valid_f32_bytes(k_size),
            ),
            (
                "v_proj",
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![hidden_dim, hidden_dim],
                valid_f32_bytes(v_size),
            ),
            (
                "o_proj",
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![hidden_dim, hidden_dim],
                valid_f32_bytes(o_size),
            ),
            (
                "ffn_norm",
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![ffn_norm_size],
                (0..ffn_norm_size).flat_map(|_| 1.0f32.to_le_bytes()).collect(),
            ),
            (
                "gate_proj",
                format!("{prefix}.mlp.gate_proj.weight"),
                vec![intermediate_dim, hidden_dim],
                valid_f32_bytes(gate_size),
            ),
            (
                "up_proj",
                format!("{prefix}.mlp.up_proj.weight"),
                vec![intermediate_dim, hidden_dim],
                valid_f32_bytes(up_size),
            ),
            (
                "down_proj",
                format!("{prefix}.mlp.down_proj.weight"),
                vec![hidden_dim, intermediate_dim],
                valid_f32_bytes(down_size),
            ),
        ]
    }

    #[test]
    fn test_convert_with_single_layer_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let hidden_dim = 16;
        let intermediate_dim = 64;
        let vocab_size = 50;
        let num_layers = 1;
        let num_heads = 4;

        // Config
        let config = format!(
            r#"{{
                "hidden_size": {},
                "num_hidden_layers": {},
                "num_attention_heads": {},
                "vocab_size": {},
                "intermediate_size": {},
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6
            }}"#,
            hidden_dim, num_layers, num_heads, vocab_size, intermediate_dim
        );
        std::fs::write(&config_path, config).expect("write config");

        // Build layer tensors
        let layer_tensors = create_layer_tensors(0, hidden_dim, intermediate_dim);

        // Build safetensors with all required tensors (valid non-zero data)
        let embed_data = valid_f32_bytes(vocab_size * hidden_dim);
        let norm_data: Vec<u8> = (0..hidden_dim).flat_map(|_| 1.0f32.to_le_bytes()).collect();

        // Create a comprehensive tensor list
        use serde_json::json;
        let mut tensor_entries = serde_json::Map::new();
        let mut all_data = Vec::new();
        let mut offset = 0usize;

        // Add embed_tokens
        tensor_entries.insert(
            "model.embed_tokens.weight".to_string(),
            json!({
                "dtype": "F32",
                "shape": [vocab_size, hidden_dim],
                "data_offsets": [offset, offset + embed_data.len()]
            }),
        );
        all_data.extend(&embed_data);
        offset += embed_data.len();

        // Add norm
        tensor_entries.insert(
            "model.norm.weight".to_string(),
            json!({
                "dtype": "F32",
                "shape": [hidden_dim],
                "data_offsets": [offset, offset + norm_data.len()]
            }),
        );
        all_data.extend(&norm_data);
        offset += norm_data.len();

        // Add layer tensors
        for (_, name, shape, data) in &layer_tensors {
            tensor_entries.insert(
                name.clone(),
                json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [offset, offset + data.len()]
                }),
            );
            all_data.extend(data);
            offset += data.len();
        }

        let json_obj = serde_json::Value::Object(tensor_entries);
        let json_bytes = json_obj.to_string().into_bytes();

        let mut safetensors_data = Vec::new();
        safetensors_data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        safetensors_data.extend_from_slice(&json_bytes);
        safetensors_data.extend_from_slice(&all_data);

        std::fs::write(&model_path, safetensors_data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok(), "Conversion failed: {:?}", result.err());

        let transformer = result.expect("operation failed");
        assert_eq!(transformer.config.hidden_dim, hidden_dim);
        assert_eq!(transformer.config.num_layers, num_layers);
        assert_eq!(transformer.layers.len(), num_layers);

        // Verify layer structure
        let layer = &transformer.layers[0];
        assert_eq!(layer.attn_norm_weight.len(), hidden_dim);
        assert_eq!(layer.qkv_weight.len(), hidden_dim * 3 * hidden_dim);
        assert_eq!(layer.attn_output_weight.len(), hidden_dim * hidden_dim);
        assert_eq!(layer.ffn_up_weight.len(), hidden_dim * intermediate_dim);
        assert_eq!(layer.ffn_down_weight.len(), intermediate_dim * hidden_dim);
        assert!(layer.ffn_gate_weight.is_some());
        assert!(layer.ffn_norm_weight.is_some());
    }

    #[test]
    fn test_convert_with_multiple_layers_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let hidden_dim = 8;
        let intermediate_dim = 32;
        let vocab_size = 20;
        let num_layers = 2;
        let num_heads = 2;

        let config = format!(
            r#"{{
                "hidden_size": {},
                "num_hidden_layers": {},
                "num_attention_heads": {},
                "vocab_size": {},
                "intermediate_size": {}
            }}"#,
            hidden_dim, num_layers, num_heads, vocab_size, intermediate_dim
        );
        std::fs::write(&config_path, config).expect("write config");

        // Build tensors for multiple layers
        use serde_json::json;
        let mut tensor_entries = serde_json::Map::new();
        let mut all_data = Vec::new();
        let mut offset = 0usize;

        // Add embed_tokens (valid non-zero data)
        let embed_data = valid_f32_bytes(vocab_size * hidden_dim);
        tensor_entries.insert(
            "model.embed_tokens.weight".to_string(),
            json!({
                "dtype": "F32",
                "shape": [vocab_size, hidden_dim],
                "data_offsets": [offset, offset + embed_data.len()]
            }),
        );
        all_data.extend(&embed_data);
        offset += embed_data.len();

        // Add norm
        let norm_data: Vec<u8> = (0..hidden_dim).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        tensor_entries.insert(
            "model.norm.weight".to_string(),
            json!({
                "dtype": "F32",
                "shape": [hidden_dim],
                "data_offsets": [offset, offset + norm_data.len()]
            }),
        );
        all_data.extend(&norm_data);
        offset += norm_data.len();

        // Add all layer tensors
        for layer_idx in 0..num_layers {
            let layer_tensors = create_layer_tensors(layer_idx, hidden_dim, intermediate_dim);
            for (_, name, shape, data) in &layer_tensors {
                tensor_entries.insert(
                    name.clone(),
                    json!({
                        "dtype": "F32",
                        "shape": shape,
                        "data_offsets": [offset, offset + data.len()]
                    }),
                );
                all_data.extend(data);
                offset += data.len();
            }
        }

        let json_obj = serde_json::Value::Object(tensor_entries);
        let json_bytes = json_obj.to_string().into_bytes();

        let mut safetensors_data = Vec::new();
        safetensors_data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        safetensors_data.extend_from_slice(&json_bytes);
        safetensors_data.extend_from_slice(&all_data);

        std::fs::write(&model_path, safetensors_data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok(), "Conversion failed: {:?}", result.err());

        let transformer = result.expect("operation failed");
        assert_eq!(transformer.layers.len(), num_layers);
    }

    #[test]
    fn test_extract_layer_missing_input_layernorm_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let config = create_config_json(16, 1, 4, 50);
        std::fs::write(&config_path, config).expect("write config");

        // Missing layer 0 input_layernorm
        let embed_data: Vec<u8> = vec![0u8; 50 * 16 * 4];
        let norm_data: Vec<u8> = vec![0u8; 16 * 4];
        let data = create_safetensors_bytes(&[
            ("model.embed_tokens.weight", "F32", &[50, 16], &embed_data),
            ("model.norm.weight", "F32", &[16], &norm_data),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_with_num_kv_heads_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        // Config with GQA (num_key_value_heads < num_attention_heads)
        let config = r#"{
            "hidden_size": 64,
            "num_hidden_layers": 0,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "vocab_size": 100
        }"#;
        std::fs::write(&config_path, config).expect("write config");

        let embed_data = valid_f32_bytes(100 * 64);
        let norm_data: Vec<u8> = (0..64).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        let data = create_safetensors_bytes(&[
            ("model.embed_tokens.weight", "F32", &[100, 64], &embed_data),
            ("model.norm.weight", "F32", &[64], &norm_data),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok(), "convert failed: {:?}", result.err());
        let transformer = result.expect("operation failed");
        assert_eq!(transformer.config.num_heads, 8);
        assert_eq!(transformer.config.num_kv_heads, 4);
    }
}
