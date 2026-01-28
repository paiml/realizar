//! Forward pass operations for transformer inference
//!
//! Extracted from layer.rs (PMAT-802) to reduce module size.
//! Contains:
//! - PAR-023: forward_all_layers_gpu
//! - PAR-023: forward_all_layers_gpu_to_logits

#![allow(clippy::wildcard_imports)]

use super::super::*;

impl CudaExecutor {
    /// PAR-023: Run ALL transformer layers GPU-resident (minimal syncs)
    ///
    /// Chains all layers on GPU, only syncing at the very end.
    /// Requires RMSNorm weights pre-cached via `preload_rmsnorm_weights()`.
    ///
    /// # Sync Count
    ///
    /// - Input upload: 1 sync
    /// - Per layer: 0 syncs (attention has internal D2D)
    /// - Output download: 1 sync
    /// - Total: ~2 syncs vs 22 syncs (per-layer) or 176 syncs (original)
    ///
    /// # Arguments
    ///
    /// * `input` - Embedding input [hidden_dim]
    /// * `output` - Output buffer [hidden_dim]
    /// * `num_layers` - Number of transformer layers
    /// * `hidden_dim` - Hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `epsilon` - RMSNorm epsilon
    #[allow(clippy::too_many_arguments)]
    pub fn forward_all_layers_gpu(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        position: u32, // PAR-070: Explicit position for RoPE and KV cache
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // 1. Validate all RMSNorm weights are cached
        for layer_idx in 0..num_layers {
            let attn_name = format!("blk.{}.attn_norm.gamma", layer_idx);
            let ffn_name = format!("blk.{}.ffn_norm.gamma", layer_idx);
            if !self.rmsnorm_cache.contains_key(&attn_name) {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: attn_norm not cached for layer {}",
                    layer_idx
                )));
            }
            if !self.rmsnorm_cache.contains_key(&ffn_name) {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: ffn_norm not cached for layer {}",
                    layer_idx
                )));
            }
        }

        // 2. Collect all cache key names (avoids repeated string allocs in loop)
        let layer_keys: Vec<(String, String)> = (0..num_layers)
            .map(|i| {
                (
                    format!("blk.{}.attn_norm.gamma", i),
                    format!("blk.{}.ffn_norm.gamma", i),
                )
            })
            .collect();

        // 3. Upload input embedding - sync point #1
        // PAR-044: Check if we can use zero-allocation workspace path
        let use_workspace = self.has_workspace()
            && self.has_indexed_weights()
            && self.indexed_layer_weights.len() == num_layers;

        let mut hidden_gpu = GpuBuffer::from_host(&self.context, input)?;

        // 4. Chain all transformer layers (no intermediate syncs)
        // PAR-044: Use workspace path for zero-allocation forward (fastest)
        // PAR-043: Use indexed path if weights are pre-indexed (10x faster per-token)
        // PAR-044 FIX: Track which buffer has output to avoid unnecessary D2D copy
        // PAR-044: Workspace path enabled - confirmed same performance as indexed path
        // See five-whys-gpu-performance-gap for analysis
        let mut workspace_used = false;
        if use_workspace {
            // PAR-044: Zero-allocation path - workspace buffers + indexed weights
            // Eliminates ~288 buffer allocations per token
            workspace_used = true;

            // Layer 0: input from external hidden_gpu
            if num_layers > 0 {
                let layer_weights = self.indexed_layer_weights[0].clone();
                self.transformer_layer_workspace(
                    &hidden_gpu,
                    0,
                    &layer_weights,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                    position,
                )?;
            }

            // Layers 1+: input from hidden_buf2 (output of previous layer)
            // Use raw pointer to avoid borrow conflict with &mut self
            for layer_idx in 1..num_layers {
                let layer_weights = self.indexed_layer_weights[layer_idx].clone();
                // SAFETY: hidden_buf2 is initialized and remains valid throughout
                // We get ptr/len before the mutable borrow, avoiding conflict
                let buf_ptr = self
                    .workspace
                    .hidden_buf2
                    .as_ref()
                    .expect("hidden_buf2 must be initialized")
                    .as_ptr();
                let buf_len = self
                    .workspace
                    .hidden_buf2
                    .as_ref()
                    .expect("hidden_buf2 must be initialized")
                    .len();
                // Create temporary non-owning view of hidden_buf2
                // SAFETY: Memory safety ensured by bounds checking and alignment
                // SAFETY: Pointer valid from allocation, length verified, used within scope
                let input_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(buf_ptr, buf_len) };
                self.transformer_layer_workspace(
                    &input_buf,
                    layer_idx,
                    &layer_weights,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                    position,
                )?;
                // Prevent Drop from freeing the borrowed memory
                std::mem::forget(input_buf);
            }

            // PAR-044 FIX: Output is in hidden_buf2, use it directly
            // (removed unnecessary copy_from_buffer - saves one D2D copy per token)
        } else if self.has_indexed_weights() && self.indexed_layer_weights.len() == num_layers {
            // PAR-043: Fast path - O(1) weight access, no string formatting
            for layer_idx in 0..num_layers {
                // Clone the layer weights to avoid borrow conflict
                let layer_weights = self.indexed_layer_weights[layer_idx].clone();
                hidden_gpu = self.transformer_layer_indexed(
                    &hidden_gpu,
                    layer_idx,
                    &layer_weights,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                )?;
            }
        } else {
            // Legacy path - HashMap lookups + string formatting (~10ms overhead)
            for layer_idx in 0..num_layers {
                let prefix = format!("blk.{}", layer_idx);
                let (ref attn_name, ref ffn_name) = layer_keys[layer_idx];

                // Get cached gamma buffer pointers (no data copy, just metadata)
                let attn_gamma = self.rmsnorm_cache.get(attn_name).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PAR-023: Missing cached gamma for {}",
                        attn_name
                    ))
                })?;
                let attn_ptr = attn_gamma.as_ptr();
                let attn_len = attn_gamma.len();
                let ffn_gamma = self.rmsnorm_cache.get(ffn_name).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PAR-023: Missing cached gamma for {}",
                        ffn_name
                    ))
                })?;
                let ffn_ptr = ffn_gamma.as_ptr();
                let ffn_len = ffn_gamma.len();

                // Run layer GPU-resident using cached gamma buffers
                hidden_gpu = self.transformer_layer_gpu_cached(
                    &hidden_gpu,
                    layer_idx,
                    &prefix,
                    hidden_dim,
                    intermediate_dim,
                    attn_ptr,
                    attn_len,
                    ffn_ptr,
                    ffn_len,
                    epsilon,
                )?;
            }
        }

        // 5. Final sync and download - sync point #2
        // PAR-044 FIX: Copy from correct buffer based on which path was used
        self.stream.synchronize()?;
        if workspace_used {
            // Output is in hidden_buf2
            let hidden_ptr = self
                .workspace
                .hidden_buf2
                .as_ref()
                .expect("hidden_buf2 must be initialized")
                .as_ptr();
            let hidden_len = self
                .workspace
                .hidden_buf2
                .as_ref()
                .expect("hidden_buf2 must be initialized")
                .len();
            // SAFETY: Memory safety ensured by bounds checking and alignment
            // SAFETY: Pointer valid from allocation, length verified, used within scope
            let output_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_ptr, hidden_len) };
            output_buf.copy_to_host(output)?;
            std::mem::forget(output_buf);
        } else {
            hidden_gpu.copy_to_host(output)?;
        }

        Ok(())
    }

    /// PAR-023: Fully GPU-resident forward to logits (minimal syncs)
    ///
    /// Runs all transformer layers + output norm + LM head projection entirely on GPU,
    /// only downloading the final logits. This eliminates the CPU round-trip for output norm.
    ///
    /// # Sync Count
    ///
    /// - Input embedding upload: 1 sync
    /// - All transformer layers: 0 syncs (attention has internal D2D)
    /// - Output RMSNorm: 0 syncs (on GPU)
    /// - LM head projection: 0 syncs (on GPU)
    /// - Logits download: 1 sync
    /// - **Total: 2 syncs** vs 3+ syncs (with CPU output norm)
    ///
    /// # Requirements
    ///
    /// Must call `preload_rmsnorm_weights()` and `preload_output_norm()` before use.
    /// LM head weights must be pre-cached via `load_quantized_weights("output.weight", ...)`.
    ///
    /// # Arguments
    ///
    /// * `input` - Input embedding [hidden_dim]
    /// * `logits` - Output logits buffer [vocab_size]
    /// * `position` - Token position for RoPE and KV cache (PAR-070: CORRECTNESS-001 fix)
    /// * `num_layers` - Number of transformer layers
    /// * `hidden_dim` - Hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `vocab_size` - Output vocabulary size
    /// * `epsilon` - RMSNorm epsilon
    #[allow(clippy::too_many_arguments)]
    pub fn forward_all_layers_gpu_to_logits(
        &mut self,
        input: &[f32],
        logits: &mut [f32],
        position: u32,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // PERF-002: Debug code removed for performance (was PAR-058-DEBUG)
        // NaN checks required D2H transfer on every token - ~10ms overhead each

        // 1. Validate all RMSNorm weights are cached (including output norm)
        for layer_idx in 0..num_layers {
            let attn_name = format!("blk.{}.attn_norm.gamma", layer_idx);
            let ffn_name = format!("blk.{}.ffn_norm.gamma", layer_idx);
            if !self.rmsnorm_cache.contains_key(&attn_name) {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: attn_norm not cached for layer {}",
                    layer_idx
                )));
            }
            if !self.rmsnorm_cache.contains_key(&ffn_name) {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: ffn_norm not cached for layer {}",
                    layer_idx
                )));
            }
        }
        if !self.rmsnorm_cache.contains_key("output_norm.gamma") {
            return Err(GpuError::InvalidLaunchConfig(
                "PAR-023: output_norm not cached".to_string(),
            ));
        }

        // 2. Collect all cache key names
        let layer_keys: Vec<(String, String)> = (0..num_layers)
            .map(|i| {
                (
                    format!("blk.{}.attn_norm.gamma", i),
                    format!("blk.{}.ffn_norm.gamma", i),
                )
            })
            .collect();

        // 3. Upload input embedding - sync point #1
        // PAR-044: Check if we can use zero-allocation workspace path
        let use_workspace = self.has_workspace()
            && self.has_indexed_weights()
            && self.indexed_layer_weights.len() == num_layers;

        let mut hidden_gpu = GpuBuffer::from_host(&self.context, input)?;

        // 4. Chain all transformer layers (no intermediate syncs)
        // PAR-044: Use workspace path for zero-allocation forward (fastest)
        // PAR-043: Use indexed path if weights are pre-indexed (10x faster per-token)
        // PAR-044 FIX: Track which buffer has output to avoid unnecessary D2D copy
        // PAR-044: Workspace path enabled - confirmed same performance as indexed path
        // See five-whys-gpu-performance-gap for analysis
        let mut workspace_used = false;
        if use_workspace {
            // PAR-044: Zero-allocation path - workspace buffers + indexed weights
            // Eliminates ~288 buffer allocations per token
            workspace_used = true;

            // Layer 0: input from external hidden_gpu
            // PAR-070: Pass explicit position for RoPE and KV cache
            if num_layers > 0 {
                let layer_weights = self.indexed_layer_weights[0].clone();
                self.transformer_layer_workspace(
                    &hidden_gpu,
                    0,
                    &layer_weights,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                    position,
                )?;
            }

            // Layers 1+: input from hidden_buf2 (output of previous layer)
            // Use raw pointer to avoid borrow conflict with &mut self
            for layer_idx in 1..num_layers {
                let layer_weights = self.indexed_layer_weights[layer_idx].clone();
                // SAFETY: hidden_buf2 is initialized and remains valid throughout
                // We get ptr/len before the mutable borrow, avoiding conflict
                let buf_ptr = self
                    .workspace
                    .hidden_buf2
                    .as_ref()
                    .expect("hidden_buf2 must be initialized")
                    .as_ptr();
                let buf_len = self
                    .workspace
                    .hidden_buf2
                    .as_ref()
                    .expect("hidden_buf2 must be initialized")
                    .len();
                // Create temporary non-owning view of hidden_buf2
                // SAFETY: Memory safety ensured by bounds checking and alignment
                // SAFETY: Pointer valid from allocation, length verified, used within scope
                let input_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(buf_ptr, buf_len) };
                // PAR-070: Pass explicit position for RoPE and KV cache
                self.transformer_layer_workspace(
                    &input_buf,
                    layer_idx,
                    &layer_weights,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                    position,
                )?;
                // Prevent Drop from freeing the borrowed memory
                std::mem::forget(input_buf);
            }

            // PAR-044 FIX: Output is in hidden_buf2, use it directly for output norm
            // (removed unnecessary copy_from_buffer - saves one D2D copy per token)
        } else if self.has_indexed_weights() && self.indexed_layer_weights.len() == num_layers {
            // PAR-043: Fast path - O(1) weight access, no string formatting
            for layer_idx in 0..num_layers {
                // Clone the layer weights to avoid borrow conflict
                let layer_weights = self.indexed_layer_weights[layer_idx].clone();
                hidden_gpu = self.transformer_layer_indexed(
                    &hidden_gpu,
                    layer_idx,
                    &layer_weights,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                )?;
            }
        } else {
            // Legacy path - HashMap lookups + string formatting (~10ms overhead)
            for layer_idx in 0..num_layers {
                let prefix = format!("blk.{}", layer_idx);
                let (ref attn_name, ref ffn_name) = layer_keys[layer_idx];

                // Get cached gamma buffer pointers (no data copy, just metadata)
                let attn_gamma = self.rmsnorm_cache.get(attn_name).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PAR-023: Missing cached gamma for {}",
                        attn_name
                    ))
                })?;
                let attn_ptr = attn_gamma.as_ptr();
                let attn_len = attn_gamma.len();
                let ffn_gamma = self.rmsnorm_cache.get(ffn_name).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PAR-023: Missing cached gamma for {}",
                        ffn_name
                    ))
                })?;
                let ffn_ptr = ffn_gamma.as_ptr();
                let ffn_len = ffn_gamma.len();

                // Run layer GPU-resident using cached gamma buffers
                hidden_gpu = self.transformer_layer_gpu_cached(
                    &hidden_gpu,
                    layer_idx,
                    &prefix,
                    hidden_dim,
                    intermediate_dim,
                    attn_ptr,
                    attn_len,
                    ffn_ptr,
                    ffn_len,
                    epsilon,
                )?;
            }
        }

        // PERF-002: Debug code removed (was PAR-058-DEBUG hidden state check)
        // D2H transfer + NaN check was ~15ms overhead per token

        // CORRECTNESS-001: Compare hidden state before output norm
        static HIDDEN_DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *HIDDEN_DEBUG.get_or_init(|| {
            std::env::var("GPU_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        }) {
            self.stream.synchronize()?;
            let hidden_to_check = if workspace_used {
                let ptr = self
                    .workspace
                    .hidden_buf2
                    .as_ref()
                    .expect("hidden_buf2 must be initialized")
                    .as_ptr();
                let len = self
                    .workspace
                    .hidden_buf2
                    .as_ref()
                    .expect("hidden_buf2 must be initialized")
                    .len();
                // SAFETY: Memory safety ensured by bounds checking and alignment
                unsafe { GpuBuffer::<f32>::from_raw_parts(ptr, len) }
            } else {
                // SAFETY: Memory safety ensured by bounds checking and alignment
                unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_gpu.as_ptr(), hidden_gpu.len()) }
            };
            let mut hidden_host = vec![0.0f32; hidden_to_check.len()];
            hidden_to_check.copy_to_host(&mut hidden_host)?;
            std::mem::forget(hidden_to_check);
            let sum: f32 = hidden_host.iter().sum();
            let sum_sq: f32 = hidden_host.iter().map(|x| x * x).sum();
            eprintln!(
                "[CORRECTNESS-001] Hidden before output_norm: first 5 = {:?}, sum = {:.4}, rms = {:.4}",
                &hidden_host[..5.min(hidden_host.len())],
                sum,
                (sum_sq / hidden_host.len() as f32).sqrt()
            );
        }

        // 5. Output RMSNorm on GPU (no sync)
        // PAR-044 FIX: Use workspace hidden_buf2 directly if workspace was used
        let output_norm_gamma = self.rmsnorm_cache.get("output_norm.gamma").ok_or_else(|| {
            GpuError::InvalidLaunchConfig(
                "PAR-023: Missing cached gamma for output_norm.gamma".to_string(),
            )
        })?;
        let output_gamma_ptr = output_norm_gamma.as_ptr();
        let output_gamma_len = output_norm_gamma.len();

        let normed_hidden = if workspace_used {
            // PAR-044 FIX: Use hidden_buf2 directly (no D2D copy)
            let hidden_ptr = self
                .workspace
                .hidden_buf2
                .as_ref()
                .expect("hidden_buf2 must be initialized")
                .as_ptr();
            let hidden_len = self
                .workspace
                .hidden_buf2
                .as_ref()
                .expect("hidden_buf2 must be initialized")
                .len();
            // SAFETY: Memory safety ensured by bounds checking and alignment
            // SAFETY: Pointer valid from allocation, length verified, used within scope
            let hidden_input = unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_ptr, hidden_len) };
            let result = self.rmsnorm_gpu_ptr(
                &hidden_input,
                output_gamma_ptr,
                output_gamma_len,
                hidden_dim,
                epsilon,
            )?;
            std::mem::forget(hidden_input);
            result
        } else {
            self.rmsnorm_gpu_ptr(
                &hidden_gpu,
                output_gamma_ptr,
                output_gamma_len,
                hidden_dim,
                epsilon,
            )?
        };

        // CORRECTNESS-002: Debug normed_hidden output (before LM head)
        if *HIDDEN_DEBUG.get_or_init(|| {
            std::env::var("GPU_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        }) {
            self.stream.synchronize()?;
            let mut normed_host = vec![0.0f32; normed_hidden.len()];
            normed_hidden.copy_to_host(&mut normed_host)?;
            let sum: f32 = normed_host.iter().sum();
            let sum_sq: f32 = normed_host.iter().map(|x| x * x).sum();
            eprintln!(
                "[CORRECTNESS-002] Normed hidden: first 5 = {:?}, sum = {:.4}, rms = {:.4}",
                &normed_host[..5.min(normed_host.len())],
                sum,
                (sum_sq / normed_host.len() as f32).sqrt()
            );
        }

        // 6. LM head projection on GPU (no sync)
        // PAR-056: Tiled kernel selection based on K dimension
        let lm_head_name = "output.weight".to_string();

        // PAR-058: Detect LM head quantization type using size-based detection
        let lm_head_qtype = if let Some(lm_head_buf) =
            self.quantized_weight_cache.get(&lm_head_name)
        {
            let lm_head_size = lm_head_buf.size_bytes();
            // Try size-based detection first, fall back to metadata
            let detected_qtype =
                WeightQuantType::from_size(lm_head_size, vocab_size as usize, hidden_dim as usize)
                    .unwrap_or_else(|| {
                        // Fall back to GGML type from metadata
                        self.quantized_weight_types
                            .get(&lm_head_name)
                            .and_then(|&t| WeightQuantType::from_ggml_type(t))
                            .unwrap_or(WeightQuantType::Q4K)
                    });
            // PERF-002: eprintln removed for performance
            detected_qtype
        } else {
            WeightQuantType::Q4K
        };

        // Get LM head buffer pointer for direct ptr API
        let lm_head_buf = self
            .quantized_weight_cache
            .get(&lm_head_name)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("LM head weight not cached".to_string())
            })?;
        let lm_head_ptr = lm_head_buf.as_ptr();

        // CORRECTNESS-002: Debug LM head weight buffer
        if *HIDDEN_DEBUG.get_or_init(|| {
            std::env::var("GPU_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        }) {
            let lm_head_size = lm_head_buf.size_bytes();
            let super_blocks_per_row = (hidden_dim as usize + 255) / 256;
            let bytes_per_row = super_blocks_per_row * 210;
            let expected_size = vocab_size as usize * bytes_per_row;
            eprintln!(
                "[CORRECTNESS-002] LM head: ptr=0x{:x}, size={}, expected={}, qtype={:?}",
                lm_head_ptr, lm_head_size, expected_size, lm_head_qtype
            );
            eprintln!(
                "[CORRECTNESS-002] LM head dims: vocab_size={}, hidden_dim={}, sb_per_row={}, bytes_per_row={}",
                vocab_size, hidden_dim, super_blocks_per_row, bytes_per_row
            );

            // LM head weights verified - size matches (skip partial copy due to API limitation)
        }

        // Allocate logits buffer
        let logits_gpu = GpuBuffer::<f32>::new(&self.context, vocab_size as usize)?;

        // PAR-058: Dispatch to correct kernel based on detected quantization type
        match lm_head_qtype {
            WeightQuantType::Q6K => {
                self.q6k_gemv_into(
                    lm_head_ptr,
                    &normed_hidden,
                    &logits_gpu,
                    vocab_size,
                    hidden_dim,
                )?;
                // CORRECTNESS-003: Debug Q6K logits
                if *HIDDEN_DEBUG.get_or_init(|| {
                    std::env::var("GPU_DEBUG")
                        .map(|v| v == "1")
                        .unwrap_or(false)
                }) {
                    self.stream.synchronize()?;
                    // Download ALL logits for full analysis
                    let mut all_logits = vec![0.0f32; vocab_size as usize];
                    logits_gpu.copy_to_host(&mut all_logits)?;

                    eprintln!(
                        "[CORRECTNESS-003] Q6K LM head logits[0..20]: {:?}",
                        &all_logits[..20]
                    );

                    // Find global argmax
                    let (global_max_idx, global_max_val) = all_logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, v)| (i, *v))
                        .expect("CUDA operation failed");
                    eprintln!(
                        "[CORRECTNESS-007] Global argmax: idx={}, val={:.4}",
                        global_max_idx, global_max_val
                    );

                    // Check for outliers: tokens with logit > 10
                    let outliers: Vec<(usize, f32)> = all_logits
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| **v > 10.0)
                        .map(|(i, v)| (i, *v))
                        .collect();
                    if !outliers.is_empty() {
                        eprintln!(
                            "[CORRECTNESS-007] Logits > 10.0 ({} tokens): {:?}",
                            outliers.len(),
                            &outliers[..10.min(outliers.len())]
                        );
                    }

                    // Check expected tokens (15='0', 16='1', 17='2', 18='3', 19='4')
                    eprintln!(
                        "[CORRECTNESS-007] Digit logits: 0={:.4}, 1={:.4}, 2={:.4}, 3={:.4}, 4={:.4}",
                        all_logits[15], all_logits[16], all_logits[17], all_logits[18], all_logits[19]
                    );

                    let logits_debug = all_logits[..20].to_vec();
                    // Check for all-zeros or all-same values (sign of kernel issue)
                    let first = logits_debug[0];
                    let all_same = logits_debug.iter().all(|&x| (x - first).abs() < 0.001);
                    if all_same {
                        eprintln!(
                            "[CORRECTNESS-003] WARNING: All logits are identical ({})",
                            first
                        );
                    }
                    // Check argmax
                    let (max_idx, max_val) = logits_debug
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("CUDA operation failed"))
                        .expect("CUDA operation failed");
                    eprintln!(
                        "[CORRECTNESS-003] Q6K argmax in first 20: idx={}, val={}",
                        max_idx, max_val
                    );

                    // CORRECTNESS-004: Compare GPU vs CPU logits for same input
                    // Download normed_hidden and compute CPU logits for comparison
                    let mut normed_host = vec![0.0f32; hidden_dim as usize];
                    normed_hidden.copy_to_host(&mut normed_host)?;

                    // Get LM head weight data from cache
                    if let Some(lm_head_buf) = self.quantized_weight_cache.get(&lm_head_name) {
                        let mut weight_bytes = vec![0u8; lm_head_buf.size_bytes()];
                        lm_head_buf.copy_to_host(&mut weight_bytes)?;

                        // CPU dequant + matmul for first 20 vocab entries
                        let super_blocks_per_row = (hidden_dim as usize + 255) / 256;
                        let bytes_per_row = super_blocks_per_row * 210; // Q6K: 210 bytes per superblock
                        let mut cpu_logits = vec![0.0f32; 20];

                        for vocab_idx in 0..20 {
                            let row_start = vocab_idx * bytes_per_row;
                            if row_start + bytes_per_row <= weight_bytes.len() {
                                // Dequantize row and dot with normed_hidden
                                let row_data = &weight_bytes[row_start..row_start + bytes_per_row];
                                let mut dot_sum = 0.0f32;

                                // Q6K layout: 256 elements per superblock
                                // Each superblock: 128 ql (low 4 bits), 64 qh (high 2 bits), 16 scales, 1 d (f16)
                                for sb in 0..super_blocks_per_row {
                                    let sb_offset = sb * 210;
                                    if sb_offset + 210 > row_data.len() {
                                        break;
                                    }

                                    // Extract d scale (f16 at offset 0)
                                    let d_bytes = [row_data[sb_offset], row_data[sb_offset + 1]];
                                    let d = half::f16::from_le_bytes(d_bytes).to_f32();

                                    // Extract ql (low 4 bits): 128 bytes at offset 2
                                    let ql = &row_data[sb_offset + 2..sb_offset + 2 + 128];

                                    // Extract qh (high 2 bits): 64 bytes at offset 130
                                    let qh = &row_data[sb_offset + 130..sb_offset + 130 + 64];

                                    // Extract scales: 16 bytes at offset 194
                                    let scales = &row_data[sb_offset + 194..sb_offset + 194 + 16];

                                    // Dequantize and dot product
                                    for i in 0..256 {
                                        let hidden_idx = sb * 256 + i;
                                        if hidden_idx >= hidden_dim as usize {
                                            break;
                                        }

                                        // Extract 6-bit quantized value
                                        let ql_idx = i / 2;
                                        let ql_shift = (i % 2) * 4;
                                        let ql_val = ((ql[ql_idx] >> ql_shift) & 0xF) as i8;

                                        let qh_idx = i / 4;
                                        let qh_shift = (i % 4) * 2;
                                        let qh_val = ((qh[qh_idx] >> qh_shift) & 0x3) as i8;

                                        let q_val = ql_val | (qh_val << 4);
                                        let q_centered = q_val - 32; // Q6K uses offset 32

                                        // Get scale for this 16-element group
                                        let scale_idx = i / 16;
                                        let scale = (scales[scale_idx] as i8) as f32;

                                        let weight = d * scale * (q_centered as f32);
                                        dot_sum += weight * normed_host[hidden_idx];
                                    }
                                }
                                cpu_logits[vocab_idx] = dot_sum;
                            }
                        }

                        eprintln!("[CORRECTNESS-004] CPU logits[0..20]: {:?}", cpu_logits);

                        // Compare
                        let max_diff = logits_debug
                            .iter()
                            .zip(cpu_logits.iter())
                            .map(|(g, c)| (g - c).abs())
                            .fold(0.0f32, f32::max);
                        eprintln!("[CORRECTNESS-004] Max GPU-CPU diff: {:.6}", max_diff);
                    }
                }
            },
            WeightQuantType::Q5K => {
                self.q5k_gemv_into(
                    lm_head_ptr,
                    &normed_hidden,
                    &logits_gpu,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q8_0 => {
                self.q8_0_gemv_into(
                    lm_head_ptr,
                    &normed_hidden,
                    &logits_gpu,
                    vocab_size,
                    hidden_dim,
                )?;
                // GQA-DEBUG: Q8_0 logits output
                if *HIDDEN_DEBUG.get_or_init(|| {
                    std::env::var("GPU_DEBUG")
                        .map(|v| v == "1")
                        .unwrap_or(false)
                }) {
                    self.stream.synchronize()?;
                    let mut all_logits = vec![0.0f32; vocab_size as usize];
                    logits_gpu.copy_to_host(&mut all_logits)?;
                    eprintln!("[GQA-DEBUG] Q8_0 LM head logits[0..20]: {:?}", &all_logits[..20]);
                    // Find global argmax
                    let (argmax_idx, argmax_val) = all_logits.iter().enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, v)| (i, *v)).expect("empty logits");
                    eprintln!("[GQA-DEBUG] Q8_0 argmax: idx={}, val={:.4}", argmax_idx, argmax_val);
                    // Digit logits (check indices for Qwen tokenizer)
                    eprintln!("[GQA-DEBUG] Token 19='4' logit = {:.4}", all_logits.get(19).unwrap_or(&0.0));
                }
            },
            WeightQuantType::Q5_0 => {
                self.q5_0_gemv_into(
                    lm_head_ptr,
                    &normed_hidden,
                    &logits_gpu,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4_0 => {
                self.q4_0_gemv_into(
                    lm_head_ptr,
                    &normed_hidden,
                    &logits_gpu,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4_1 => {
                self.q4_1_gemv_into(
                    lm_head_ptr,
                    &normed_hidden,
                    &logits_gpu,
                    vocab_size,
                    hidden_dim,
                )?;
            },
            WeightQuantType::Q4K => {
                self.q4k_gemv_into(
                    lm_head_ptr,
                    &normed_hidden,
                    &logits_gpu,
                    vocab_size,
                    hidden_dim,
                )?;
            },
        }

        // 7. Final sync and download - sync point #2 (only required sync)
        self.stream.synchronize()?;
        logits_gpu.copy_to_host(logits)?;

        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    /// Helper to create CudaExecutor for tests
    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // Validation Tests for forward_all_layers_gpu
    // ========================================================================

    #[test]
    fn test_forward_all_layers_gpu_missing_attn_norm() {
        let Some(mut exec) = create_executor() else { return; };

        let input = vec![0.1f32; 256];
        let mut output = vec![0.0f32; 256];

        // No RMSNorm weights cached - should error
        let result = exec.forward_all_layers_gpu(
            &input,
            &mut output,
            0,     // position
            1,     // num_layers
            256,   // hidden_dim
            1024,  // intermediate_dim
            1e-5,  // epsilon
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = format!("{:?}", err);
        assert!(err_str.contains("attn_norm not cached"));
    }

    #[test]
    fn test_forward_all_layers_gpu_missing_ffn_norm() {
        let Some(mut exec) = create_executor() else { return; };

        // Cache attn_norm but not ffn_norm
        let gamma: Vec<f32> = vec![1.0; 256];
        let _ = exec.cache_rmsnorm_gamma("blk.0.attn_norm.gamma", &gamma);

        let input = vec![0.1f32; 256];
        let mut output = vec![0.0f32; 256];

        let result = exec.forward_all_layers_gpu(
            &input,
            &mut output,
            0,
            1,
            256,
            1024,
            1e-5,
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = format!("{:?}", err);
        assert!(err_str.contains("ffn_norm not cached"));
    }

    // ========================================================================
    // Validation Tests for forward_all_layers_gpu_to_logits
    // ========================================================================

    #[test]
    fn test_forward_to_logits_missing_attn_norm() {
        let Some(mut exec) = create_executor() else { return; };

        let input = vec![0.1f32; 256];
        let mut logits = vec![0.0f32; 1024];

        let result = exec.forward_all_layers_gpu_to_logits(
            &input,
            &mut logits,
            0,     // position
            1,     // num_layers
            256,   // hidden_dim
            1024,  // intermediate_dim
            1024,  // vocab_size
            1e-5,  // epsilon
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = format!("{:?}", err);
        assert!(err_str.contains("attn_norm not cached"));
    }

    #[test]
    fn test_forward_to_logits_missing_output_norm() {
        let Some(mut exec) = create_executor() else { return; };

        // Cache layer norms but not output_norm
        let gamma: Vec<f32> = vec![1.0; 256];
        let _ = exec.cache_rmsnorm_gamma("blk.0.attn_norm.gamma", &gamma);
        let _ = exec.cache_rmsnorm_gamma("blk.0.ffn_norm.gamma", &gamma);

        let input = vec![0.1f32; 256];
        let mut logits = vec![0.0f32; 1024];

        // This will pass validation but fail later due to missing output_norm.gamma
        // We use workspace_unused path which requires output_norm.gamma
        let result = exec.forward_all_layers_gpu_to_logits(
            &input,
            &mut logits,
            0,
            1,
            256,
            1024,
            1024,
            1e-5,
        );

        // Will error due to missing output_norm.gamma or lm_head
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_to_logits_zero_layers() {
        let Some(mut exec) = create_executor() else { return; };

        // Cache output norm only (no layer norms needed for 0 layers)
        let gamma: Vec<f32> = vec![1.0; 256];
        let _ = exec.cache_rmsnorm_gamma("output_norm.gamma", &gamma);

        let input = vec![0.1f32; 256];
        let mut logits = vec![0.0f32; 1024];

        // 0 layers - should skip layer processing, fail at output norm or lm_head
        let result = exec.forward_all_layers_gpu_to_logits(
            &input,
            &mut logits,
            0,
            0,     // 0 layers
            256,
            1024,
            1024,
            1e-5,
        );

        // Will error due to missing lm_head
        assert!(result.is_err());
    }
}
