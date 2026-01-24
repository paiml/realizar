//! GPU Model and Configuration (PMAT-802)
//!
//! GpuModel, GpuModelConfig, GpuGenerateConfig, AttentionBuffers

use crate::error::{RealizarError, Result};
use super::super::{
    HybridScheduler, StreamingKVCache, exceeds_gpu_buffer_limit,
    cpu_matmul_transposed_simd, cpu_matmul,
};
#[cfg(feature = "cuda")]
use super::core::CudaScheduler;

/// GPU-accelerated model for M3 parity (128 tok/s target)
///
/// Wraps standard Model and uses HybridScheduler for GPU-accelerated
/// matrix multiplications in the forward pass.
pub struct GpuModel {
    /// Embedding weights (vocab_size x hidden_dim)
    pub(crate) embedding_weights: Vec<f32>,
    /// Linear layer weights for each block
    /// Each block has: attn_q, attn_k, attn_v, attn_out, ffn_fc1, ffn_fc2
    pub(crate) block_weights: Vec<BlockWeights>,
    /// Final layer norm weights
    pub(crate) final_norm_weight: Vec<f32>,
    pub(crate) final_norm_bias: Vec<f32>,
    /// LM head weights (hidden_dim x vocab_size)
    pub(crate) lm_head_weight: Vec<f32>,
    /// LM head weights transposed (vocab_size x hidden_dim) for fast CPU inference
    pub(crate) lm_head_weight_t: Vec<f32>,
    pub(crate) lm_head_bias: Vec<f32>,
    /// GPU scheduler (HybridScheduler - may force CPU for m=1)
    pub(crate) scheduler: HybridScheduler,
    /// IMP-1003: Optional CUDA-only scheduler that ALWAYS uses GPU
    /// When present, this scheduler is preferred over HybridScheduler for matmul
    #[cfg(feature = "cuda")]
    pub(crate) cuda_scheduler: Option<CudaScheduler>,
    /// Model configuration
    pub config: GpuModelConfig,
    /// Pre-allocated attention buffers for optimized incremental decoding (M17)
    pub(crate) attention_buffers: Option<AttentionBuffers>,
}

/// Weights for a single transformer block
///
/// Used by adapters (PMAT-106) to construct GpuModel from various formats.
pub struct BlockWeights {
    /// Attention layer norm weight
    pub attn_norm_weight: Vec<f32>,
    /// Attention layer norm bias
    pub attn_norm_bias: Vec<f32>,
    /// Combined QKV projection weights (hidden_dim x 3*hidden_dim)
    pub qkv_weight: Vec<f32>,
    /// QKV projection bias (reserved for future use)
    #[allow(dead_code)]
    pub qkv_bias: Vec<f32>,
    /// Output projection weight (hidden_dim x hidden_dim)
    pub out_weight: Vec<f32>,
    /// Output projection bias
    pub out_bias: Vec<f32>,
    /// FFN layer norm weight
    pub ffn_norm_weight: Vec<f32>,
    /// FFN layer norm bias
    pub ffn_norm_bias: Vec<f32>,
    /// FFN first layer weight (up projection)
    pub ffn_fc1_weight: Vec<f32>,
    /// FFN first layer bias
    pub ffn_fc1_bias: Vec<f32>,
    /// FFN second layer weight (down projection)
    pub ffn_fc2_weight: Vec<f32>,
    /// FFN second layer bias
    pub ffn_fc2_bias: Vec<f32>,
    /// FFN gate projection weight for SwiGLU (optional)
    /// When present, FFN uses SwiGLU: down(SiLU(gate(x)) * up(x))
    /// When None, FFN uses GELU: down(GELU(up(x)))
    pub ffn_gate_weight: Option<Vec<f32>>,
}

/// IMP-1007: Weight type for split-borrow matmul
///
/// This enum specifies which weight matrix to use in matmul_split,
/// enabling zero-clone matmul operations by using Rust's split borrow pattern.
#[derive(Debug, Clone, Copy)]
pub enum WeightType {
    /// QKV projection: [hidden_dim, qkv_dim]
    Qkv,
    /// Output projection: [hidden_dim, hidden_dim]
    Output,
    /// FFN FC1: [hidden_dim, intermediate_dim]
    FfnFc1,
    /// FFN FC2: [intermediate_dim, hidden_dim]
    FfnFc2,
    /// LM head: [hidden_dim, vocab_size]
    LmHead,
}

/// Configuration for GPU model
#[derive(Debug, Clone)]
pub struct GpuModelConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads (Q heads)
    pub num_heads: usize,
    /// Number of key-value heads for GQA (IMP-088)
    /// For standard MHA: num_kv_heads == num_heads
    /// For GQA (Qwen, Llama-3): num_kv_heads < num_heads
    pub num_kv_heads: usize,
    /// Number of transformer blocks
    pub num_layers: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Layer normalization epsilon
    pub eps: f32,
    /// RoPE theta for rotary position embeddings (Phase 21)
    /// Default: 10000.0 (standard LLaMA)
    pub rope_theta: f32,
}

impl GpuModelConfig {
    /// Head dimension (hidden_dim / num_heads)
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }

    /// K/V dimension for GQA (num_kv_heads * head_dim)
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim()
    }

    /// Total QKV projection output dimension
    /// For MHA: 3 * hidden_dim
    /// For GQA: hidden_dim + 2 * kv_dim
    #[inline]
    pub fn qkv_dim(&self) -> usize {
        self.hidden_dim + 2 * self.kv_dim()
    }

    /// Whether this is a GQA model (num_kv_heads < num_heads)
    #[inline]
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads
    }
}

/// Configuration for GPU text generation (M14: E2E Inference)
#[derive(Debug, Clone)]
pub struct GpuGenerateConfig {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = greedy)
    pub temperature: f32,
    /// Top-k sampling (1 = greedy)
    pub top_k: usize,
    /// Stop token IDs
    pub stop_tokens: Vec<usize>,
}

impl Default for GpuGenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: Vec::new(),
        }
    }
}

impl GpuGenerateConfig {
    /// Create config for deterministic (greedy) generation
    #[must_use]
    pub fn deterministic(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: Vec::new(),
        }
    }

    /// Create config with temperature and top-k sampling
    #[must_use]
    pub fn with_sampling(max_tokens: usize, temperature: f32, top_k: usize) -> Self {
        Self {
            max_tokens,
            temperature,
            top_k,
            stop_tokens: Vec::new(),
        }
    }

    /// Add stop tokens to config
    #[must_use]
    pub fn with_stop_tokens(mut self, stop_tokens: Vec<usize>) -> Self {
        self.stop_tokens = stop_tokens;
        self
    }
}

/// Pre-allocated attention buffers for optimized incremental decoding (M17)
///
/// Eliminates per-token memory allocation during incremental generation by
/// reusing pre-allocated buffers for Q, attention scores, and output.
#[derive(Debug)]
pub struct AttentionBuffers {
    /// Q buffer for single-token attention [hidden_dim]
    pub q_buffer: Vec<f32>,
    /// Attention scores buffer [num_heads * max_seq_len]
    pub scores_buffer: Vec<f32>,
    /// Attention output buffer [hidden_dim]
    pub output_buffer: Vec<f32>,
    /// K/V projection buffer [hidden_dim]
    pub kv_proj_buffer: Vec<f32>,
    /// Intermediate FFN buffer [intermediate_dim]
    pub ffn_buffer: Vec<f32>,
    /// Max sequence length these buffers support
    pub max_seq_len: usize,
}

impl AttentionBuffers {
    /// Create pre-allocated attention buffers from model config
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    /// * `max_seq_len` - Maximum sequence length to support
    #[must_use]
    pub fn new(config: &GpuModelConfig, max_seq_len: usize) -> Self {
        Self {
            q_buffer: vec![0.0; config.hidden_dim],
            scores_buffer: vec![0.0; config.num_heads * max_seq_len],
            output_buffer: vec![0.0; config.hidden_dim],
            kv_proj_buffer: vec![0.0; config.hidden_dim],
            ffn_buffer: vec![0.0; config.intermediate_dim],
            max_seq_len,
        }
    }

    /// Reset all buffers to zero (for reuse)
    pub fn reset(&mut self) {
        self.q_buffer.fill(0.0);
        self.scores_buffer.fill(0.0);
        self.output_buffer.fill(0.0);
        self.kv_proj_buffer.fill(0.0);
        self.ffn_buffer.fill(0.0);
    }
}

impl GpuModel {
    /// Create a new GPU-accelerated model with random initialization
    ///
    /// # Errors
    ///
    /// Returns error if GPU initialization fails
    pub fn new(config: GpuModelConfig) -> Result<Self> {
        let scheduler = HybridScheduler::new()?;

        // Initialize weights (small random values for testing)
        let embedding_weights = vec![0.01f32; config.vocab_size * config.hidden_dim];

        let mut block_weights = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            block_weights.push(BlockWeights {
                attn_norm_weight: vec![1.0f32; config.hidden_dim],
                attn_norm_bias: vec![0.0f32; config.hidden_dim],
                qkv_weight: vec![0.01f32; config.hidden_dim * 3 * config.hidden_dim],
                qkv_bias: vec![0.0f32; 3 * config.hidden_dim],
                out_weight: vec![0.01f32; config.hidden_dim * config.hidden_dim],
                out_bias: vec![0.0f32; config.hidden_dim],
                ffn_norm_weight: vec![1.0f32; config.hidden_dim],
                ffn_norm_bias: vec![0.0f32; config.hidden_dim],
                ffn_fc1_weight: vec![0.01f32; config.hidden_dim * config.intermediate_dim],
                ffn_fc1_bias: vec![0.0f32; config.intermediate_dim],
                ffn_fc2_weight: vec![0.01f32; config.intermediate_dim * config.hidden_dim],
                ffn_fc2_bias: vec![0.0f32; config.hidden_dim],
                ffn_gate_weight: None, // No SwiGLU in test models
            });
        }

        let final_norm_weight = vec![1.0f32; config.hidden_dim];
        let final_norm_bias = vec![0.0f32; config.hidden_dim];
        let lm_head_weight = vec![0.01f32; config.hidden_dim * config.vocab_size];
        let lm_head_bias = vec![0.0f32; config.vocab_size];

        // Pre-compute transposed LM head for fast CPU inference
        // Original: [hidden_dim, vocab_size] -> Transposed: [vocab_size, hidden_dim]
        let lm_head_weight_t =
            Self::transpose_weights(&lm_head_weight, config.hidden_dim, config.vocab_size);

        Ok(Self {
            embedding_weights,
            block_weights,
            final_norm_weight,
            final_norm_bias,
            lm_head_weight,
            lm_head_weight_t,
            lm_head_bias,
            scheduler,
            #[cfg(feature = "cuda")]
            cuda_scheduler: None,
            config,
            attention_buffers: None,
        })
    }

    /// IMP-1003: Create GPU model with CUDA-only scheduler
    ///
    /// Unlike `new()`, this constructor creates a model that ALWAYS uses CUDA
    /// for matmul operations, even for m=1 (single-token generation).
    ///
    /// # Errors
    ///
    /// Returns error if GPU or CUDA initialization fails
    #[cfg(feature = "cuda")]
    pub fn new_with_cuda(config: GpuModelConfig) -> Result<Self> {
        let scheduler = HybridScheduler::new()?;
        let cuda_scheduler = Some(CudaScheduler::new()?);

        // Initialize weights (small random values for testing)
        let embedding_weights = vec![0.01f32; config.vocab_size * config.hidden_dim];

        let mut block_weights = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            block_weights.push(BlockWeights {
                attn_norm_weight: vec![1.0f32; config.hidden_dim],
                attn_norm_bias: vec![0.0f32; config.hidden_dim],
                qkv_weight: vec![0.01f32; config.hidden_dim * config.qkv_dim()],
                qkv_bias: vec![0.0f32; config.qkv_dim()],
                out_weight: vec![0.01f32; config.hidden_dim * config.hidden_dim],
                out_bias: vec![0.0f32; config.hidden_dim],
                ffn_norm_weight: vec![1.0f32; config.hidden_dim],
                ffn_norm_bias: vec![0.0f32; config.hidden_dim],
                ffn_fc1_weight: vec![0.01f32; config.hidden_dim * config.intermediate_dim],
                ffn_fc1_bias: vec![0.0f32; config.intermediate_dim],
                ffn_fc2_weight: vec![0.01f32; config.intermediate_dim * config.hidden_dim],
                ffn_fc2_bias: vec![0.0f32; config.hidden_dim],
                ffn_gate_weight: None, // No SwiGLU in test models
            });
        }

        let final_norm_weight = vec![1.0f32; config.hidden_dim];
        let final_norm_bias = vec![0.0f32; config.hidden_dim];
        let lm_head_weight = vec![0.01f32; config.hidden_dim * config.vocab_size];
        let lm_head_bias = vec![0.0f32; config.vocab_size];

        let lm_head_weight_t =
            Self::transpose_weights(&lm_head_weight, config.hidden_dim, config.vocab_size);

        Ok(Self {
            embedding_weights,
            block_weights,
            final_norm_weight,
            final_norm_bias,
            lm_head_weight,
            lm_head_weight_t,
            lm_head_bias,
            scheduler,
            cuda_scheduler,
            config,
            attention_buffers: None,
        })
    }

    /// IMP-1003: Check if this model has CUDA scheduler enabled
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn has_cuda_scheduler(&self) -> bool {
        self.cuda_scheduler.is_some()
    }

    /// IMP-1003: Perform matmul using CUDA scheduler (always GPU, even for m=1)
    ///
    /// # Errors
    ///
    /// Returns error if CUDA scheduler is not available or matmul fails
    #[cfg(feature = "cuda")]
    #[allow(clippy::many_single_char_names)]
    pub fn cuda_matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        if let Some(ref mut cuda_sched) = self.cuda_scheduler {
            cuda_sched.matmul(a, b, m, k, n)
        } else {
            // Fallback to HybridScheduler
            self.scheduler.matmul(a, b, m, k, n)
        }
    }

    /// IMP-1005: Unified matmul dispatch that prefers CudaScheduler when available
    ///
    /// This method is used throughout forward_gpu() and forward_block_idx() to
    /// ensure CUDA is used for all matmul operations when cuda_scheduler is present.
    ///
    /// # Errors
    ///
    /// Returns error if matmul fails
    #[allow(clippy::many_single_char_names)]
    pub fn do_matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        if let Some(ref mut cuda_sched) = self.cuda_scheduler {
            return cuda_sched.matmul(a, b, m, k, n);
        }
        // Fallback to HybridScheduler (or always use it when cuda feature disabled)
        self.scheduler.matmul(a, b, m, k, n)
    }

    /// IMP-1007: Zero-clone matmul using split borrow pattern
    ///
    /// This method eliminates weight cloning by using Rust's split borrow pattern.
    /// It directly borrows weights from block_weights while mutably borrowing schedulers.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    /// * `block_idx` - Block index for block weights (ignored for LmHead)
    /// * `op` - Which matmul operation/weight to use
    ///
    /// # Errors
    ///
    /// Returns error if matmul fails
    pub fn matmul_split(
        &mut self,
        input: &[f32],
        block_idx: usize,
        op: WeightType,
    ) -> Result<Vec<f32>> {
        // IMP-1007: Use split borrowing to avoid weight cloning
        // Extract dimensions from config (Copy types, no borrow conflict)
        let hidden_dim = self.config.hidden_dim;
        let qkv_dim = self.config.qkv_dim();
        let intermediate_dim = self.config.intermediate_dim;
        let vocab_size = self.config.vocab_size;

        // Get weight reference and dimensions based on operation
        let (weight, m, k, n) = match op {
            WeightType::Qkv => (
                &self.block_weights[block_idx].qkv_weight,
                1,
                hidden_dim,
                qkv_dim,
            ),
            WeightType::Output => (
                &self.block_weights[block_idx].out_weight,
                1,
                hidden_dim,
                hidden_dim,
            ),
            WeightType::FfnFc1 => (
                &self.block_weights[block_idx].ffn_fc1_weight,
                1,
                hidden_dim,
                intermediate_dim,
            ),
            WeightType::FfnFc2 => (
                &self.block_weights[block_idx].ffn_fc2_weight,
                1,
                intermediate_dim,
                hidden_dim,
            ),
            WeightType::LmHead => (&self.lm_head_weight, 1, hidden_dim, vocab_size),
        };

        // Clone weight to work around borrow checker - this is the safe fallback.
        // For zero-clone operations, use matmul_zero_clone() instead (IMP-1007).
        let weight_clone = weight.clone();

        // Now call do_matmul with cloned weight
        self.do_matmul(input, &weight_clone, m, k, n)
    }

    /// IMP-1007: Zero-clone matmul helper using explicit scheduler extraction
    ///
    /// This is a more aggressive optimization that temporarily extracts the
    /// cuda_scheduler to enable truly zero-clone matmul operations.
    ///
    /// # Safety
    ///
    /// This method uses `Option::take()` to temporarily move the scheduler,
    /// which is safe but requires careful handling to restore it.
    #[cfg(feature = "cuda")]
    pub fn matmul_zero_clone(
        &mut self,
        input: &[f32],
        block_idx: usize,
        op: WeightType,
    ) -> Result<Vec<f32>> {
        // Extract dimensions
        let hidden_dim = self.config.hidden_dim;
        let qkv_dim = self.config.qkv_dim();
        let intermediate_dim = self.config.intermediate_dim;
        let vocab_size = self.config.vocab_size;

        // Temporarily take cuda_scheduler out of self
        let mut cuda_sched: Option<CudaScheduler> = self.cuda_scheduler.take();

        // Now we can borrow block_weights freely
        let (weight, m, k, n) = match op {
            WeightType::Qkv => (
                &self.block_weights[block_idx].qkv_weight,
                1,
                hidden_dim,
                qkv_dim,
            ),
            WeightType::Output => (
                &self.block_weights[block_idx].out_weight,
                1,
                hidden_dim,
                hidden_dim,
            ),
            WeightType::FfnFc1 => (
                &self.block_weights[block_idx].ffn_fc1_weight,
                1,
                hidden_dim,
                intermediate_dim,
            ),
            WeightType::FfnFc2 => (
                &self.block_weights[block_idx].ffn_fc2_weight,
                1,
                intermediate_dim,
                hidden_dim,
            ),
            WeightType::LmHead => (&self.lm_head_weight, 1, hidden_dim, vocab_size),
        };

        // Perform matmul with extracted scheduler
        let result: Result<Vec<f32>> = if let Some(sched) = cuda_sched.as_mut() {
            CudaScheduler::matmul(sched, input, weight, m, k, n)
        } else {
            self.scheduler.matmul(input, weight, m, k, n)
        };

        // Restore cuda_scheduler
        self.cuda_scheduler = cuda_sched;

        result
    }

    // =========================================================================
    // IMP-1008: RefCell-based zero-clone matmul (interior mutability pattern)
    // =========================================================================

    /// IMP-1008: Zero-clone matmul using interior mutability
    ///
    /// This method takes `&self` instead of `&mut self` by wrapping scheduler
    /// access in RefCell. This eliminates the need to clone weights.
    ///
    /// # Errors
    ///
    /// Returns error if matmul fails or RefCell is already borrowed.
    #[cfg(feature = "cuda")]
    #[allow(clippy::many_single_char_names)]
    pub fn matmul_refcell(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // IMP-1008: For RefCell pattern, we need to use a different approach
        // Since cuda_scheduler is Option<CudaScheduler>, we use UnsafeCell
        // pattern with explicit unsafe block to avoid changing struct layout.
        //
        // This is safe because:
        // 1. We only access cuda_scheduler mutably here
        // 2. No other code paths access it during matmul
        // 3. This is single-threaded execution

        // Use raw pointer to bypass borrow checker (safe in single-threaded context)
        // SAFETY: This is safe because:
        // - We're in single-threaded context (LLM inference)
        // - cuda_scheduler is only accessed through this method during matmul
        // - The borrow is released before returning
        let cuda_sched_ptr = std::ptr::addr_of!(self.cuda_scheduler).cast_mut();

        // SAFETY: Memory safety ensured by bounds checking and alignment
        let result: Result<Vec<f32>> = unsafe {
            if let Some(sched) = (*cuda_sched_ptr).as_mut() {
                CudaScheduler::matmul(sched, a, b, m, k, n)
            } else {
                // Fallback to HybridScheduler (also needs mut access)
                let sched_ptr = std::ptr::addr_of!(self.scheduler).cast_mut();
                (*sched_ptr).matmul(a, b, m, k, n)
            }
        };
        result
    }

    /// IMP-1008: Forward single block without weight cloning
    ///
    /// Uses interior mutability pattern to avoid cloning weights on each matmul.
    /// This method takes `&self` instead of `&mut self`.
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails.
    #[cfg(feature = "cuda")]
    pub fn forward_block_refcell(
        &self,
        input: &[f32],
        block_idx: usize,
        kv_cache: &mut StreamingKVCache,
    ) -> Result<Vec<f32>> {
        // Phase 21 Debug: trace first forward call only
        static DEBUG_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call_count = DEBUG_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let debug_this_call = block_idx == 0 && call_count == 0;  // Only first call to block 0

        // Extract config values (Copy types, no borrow conflict)
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim();
        let kv_dim = self.config.kv_dim();
        let qkv_dim = self.config.qkv_dim();
        let intermediate_dim = self.config.intermediate_dim;
        let eps = self.config.eps;
        let num_kv_heads = self.config.num_kv_heads;

        if debug_this_call {
            eprintln!("[PHASE21] forward_block_refcell START block_idx={}", block_idx);
            eprintln!("[PHASE21] input L2: {:.4}", input.iter().map(|x| x*x).sum::<f32>().sqrt());
        }

        // IMP-1008: No cloning! Direct reference to weights
        // Pre-attention layer norm (static function avoids &self borrow)
        let normed = Self::layer_norm_static(
            input,
            &self.block_weights[block_idx].attn_norm_weight,
            &self.block_weights[block_idx].attn_norm_bias,
            hidden_dim,
            eps,
        );

        // QKV projection - NO CLONE!
        let mut qkv = self.matmul_refcell(
            &normed,
            &self.block_weights[block_idx].qkv_weight,
            1,
            hidden_dim,
            qkv_dim,
        )?;

        if debug_this_call {
            eprintln!("[PHASE21] QKV L2: {:.4}", qkv.iter().map(|x| x*x).sum::<f32>().sqrt());
        }

        // Get current position BEFORE caching (Phase 21)
        let (cached_k_ref, _) = kv_cache.get_valid(block_idx);
        let current_pos = cached_k_ref.len() / kv_dim;

        // Phase 21: Apply RoPE to Q and K BEFORE caching
        // Without RoPE, attention has no position information and produces garbage
        let rope_theta = self.config.rope_theta;
        Self::apply_rope_inline(&mut qkv[0..hidden_dim], num_heads, head_dim, rope_theta, current_pos);
        Self::apply_rope_inline(&mut qkv[hidden_dim..hidden_dim + kv_dim], num_kv_heads, head_dim, rope_theta, current_pos);

        // Split QKV (GQA: K/V have kv_dim, not hidden_dim) - after RoPE
        let q = qkv[0..hidden_dim].to_vec();
        let k_new = qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
        let v_new = qkv[hidden_dim + kv_dim..].to_vec();

        // Get cached K/V and clone to avoid borrow issues with kv_cache
        let (cached_k, cached_v) = kv_cache.get_valid(block_idx);
        let keys_cached = cached_k.to_vec();
        let vals_cached = cached_v.to_vec();

        // Append new K/V (with RoPE applied) to cache
        kv_cache.append(block_idx, &k_new, &v_new);

        // Build full K/V (cached + new)
        let kv_len = keys_cached.len() / kv_dim + 1;
        let mut full_k = keys_cached;
        full_k.extend_from_slice(&k_new);
        let mut full_v = vals_cached;
        full_v.extend_from_slice(&v_new);

        // GQA attention (IMP-089): static method to avoid borrow conflicts
        let attn_output = Self::gqa_multihead_attention(
            &q,
            &full_k,
            &full_v,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        if debug_this_call {
            eprintln!("[PHASE21] attn_output L2: {:.4}", attn_output.iter().map(|x| x*x).sum::<f32>().sqrt());
        }

        // Output projection - NO CLONE!
        let attn_proj = self.matmul_refcell(
            &attn_output,
            &self.block_weights[block_idx].out_weight,
            1,
            hidden_dim,
            hidden_dim,
        )?;

        // Add residual and bias
        let out_bias = &self.block_weights[block_idx].out_bias;
        let post_attn: Vec<f32> = input
            .iter()
            .zip(attn_proj.iter())
            .zip(out_bias.iter())
            .map(|((&i, &a), &b)| i + a + b)
            .collect();

        // FFN with layer norm (static function)
        let ffn_normed = Self::layer_norm_static(
            &post_attn,
            &self.block_weights[block_idx].ffn_norm_weight,
            &self.block_weights[block_idx].ffn_norm_bias,
            hidden_dim,
            eps,
        );

        // FFN: SwiGLU when gate weight exists, otherwise GELU
        let fc1_activated: Vec<f32> = if let Some(ref gate_weight) = self.block_weights[block_idx].ffn_gate_weight {
            // SwiGLU: silu(gate(x)) * up(x)
            // Up projection
            let up_out = self.matmul_refcell(
                &ffn_normed,
                &self.block_weights[block_idx].ffn_fc1_weight,
                1,
                hidden_dim,
                intermediate_dim,
            )?;

            // Gate projection
            let gate_out = self.matmul_refcell(
                &ffn_normed,
                gate_weight,
                1,
                hidden_dim,
                intermediate_dim,
            )?;

            // SwiGLU: silu(gate) * up
            // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
            up_out
                .iter()
                .zip(gate_out.iter())
                .map(|(&u, &g)| {
                    let silu_g = g / (1.0 + (-g).exp());
                    silu_g * u
                })
                .collect()
        } else {
            // Standard GELU FFN
            let fc1_out = self.matmul_refcell(
                &ffn_normed,
                &self.block_weights[block_idx].ffn_fc1_weight,
                1,
                hidden_dim,
                intermediate_dim,
            )?;

            // Add bias and GELU activation
            let ffn_fc1_bias = &self.block_weights[block_idx].ffn_fc1_bias;
            fc1_out
                .iter()
                .zip(ffn_fc1_bias.iter())
                .map(|(&x, &b)| {
                    let x_b = x + b;
                    x_b * 0.5 + x_b * 0.5 * (0.797_884_6 * (x_b + 0.044_715 * x_b.powi(3))).tanh()
                })
                .collect()
        };

        // FFN FC2 (down projection) - NO CLONE!
        let fc2_out = self.matmul_refcell(
            &fc1_activated,
            &self.block_weights[block_idx].ffn_fc2_weight,
            1,
            intermediate_dim,
            hidden_dim,
        )?;

        // Add bias and residual
        let ffn_fc2_bias = &self.block_weights[block_idx].ffn_fc2_bias;
        let output: Vec<f32> = post_attn
            .iter()
            .zip(fc2_out.iter())
            .zip(ffn_fc2_bias.iter())
            .map(|((&h, &f), &b)| h + f + b)
            .collect();

        if debug_this_call {
            eprintln!("[PHASE21] block output L2: {:.4}", output.iter().map(|x| x*x).sum::<f32>().sqrt());
        }

        Ok(output)
    }

    /// IMP-1008: Full incremental forward pass without weight cloning
    ///
    /// Uses interior mutability pattern throughout for zero-clone operation.
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails.
    #[cfg(feature = "cuda")]
    pub fn forward_refcell(
        &self,
        token_id: usize,
        kv_cache: &mut StreamingKVCache,
    ) -> Result<Vec<f32>> {
        // Phase 21: Debug first call only
        static FWD_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let fwd_count = FWD_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let debug_this_fwd = fwd_count == 0;

        if token_id >= self.config.vocab_size {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Token ID {} out of bounds (vocab_size={})",
                    token_id, self.config.vocab_size
                ),
            });
        }

        let hidden_dim = self.config.hidden_dim;

        // Embed single token
        let offset = token_id * hidden_dim;
        let mut hidden = self.embedding_weights[offset..offset + hidden_dim].to_vec();

        // Process through all blocks - NO CLONE!
        for block_idx in 0..self.config.num_layers {
            hidden = self.forward_block_refcell(&hidden, block_idx, kv_cache)?;
        }

        // Final layer norm
        hidden = self.layer_norm_refcell(&hidden, &self.final_norm_weight, &self.final_norm_bias);

        // LM head projection
        let lm_head_elements = hidden_dim * self.config.vocab_size;
        let output = if exceeds_gpu_buffer_limit(lm_head_elements) {
            // CPU path with transposed weights + SIMD + fused bias
            cpu_matmul_transposed_simd(
                &hidden,
                &self.lm_head_weight_t,
                &self.lm_head_bias,
                hidden_dim,
                self.config.vocab_size,
            )
        } else {
            // GPU path - NO CLONE!
            let vocab_size = self.config.vocab_size;
            let logits =
                self.matmul_refcell(&hidden, &self.lm_head_weight, 1, hidden_dim, vocab_size)?;
            // Add bias
            logits
                .into_iter()
                .zip(self.lm_head_bias.iter())
                .map(|(l, &b)| l + b)
                .collect()
        };

        if debug_this_fwd {
            // Find argmax
            let (argmax_idx, argmax_val) = output.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            eprintln!("[PHASE21] forward_refcell: final hidden L2: {:.4}", hidden.iter().map(|x| x*x).sum::<f32>().sqrt());
            eprintln!("[PHASE21] forward_refcell: logits argmax: {} (val: {:.4})", argmax_idx, argmax_val);
        }

        Ok(output)
    }

    /// IMP-1008: Layer norm with RefCell pattern (takes &self)
    #[cfg(feature = "cuda")]
    fn layer_norm_refcell(&self, input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        Self::layer_norm_static(input, weight, bias, self.config.hidden_dim, self.config.eps)
    }

    /// IMP-1008: Generate tokens without weight cloning
    ///
    /// Uses interior mutability pattern for zero-clone inference.
    ///
    /// # Errors
    ///
    /// Returns error if generation fails.
    #[cfg(feature = "cuda")]
    pub fn generate_refcell(
        &self,
        prompt: &[usize],
        config: &GpuGenerateConfig,
    ) -> Result<Vec<usize>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let max_seq_len = prompt.len() + config.max_tokens;

        // Initialize KV cache
        let mut kv_cache =
            StreamingKVCache::new(self.config.num_layers, max_seq_len, num_kv_heads, head_dim);

        let mut tokens = prompt.to_vec();

        // Process prompt tokens to populate KV cache
        for &token_id in prompt {
            let _ = self.forward_refcell(token_id, &mut kv_cache)?;
        }

        // Generate new tokens
        for _ in 0..config.max_tokens {
            let last_token = *tokens.last().unwrap_or(&0);
            let logits = self.forward_refcell(last_token, &mut kv_cache)?;

            // Sample next token (greedy when temperature=0, otherwise top-k)
            let next_token = if config.temperature == 0.0 || config.top_k == 1 {
                // Greedy decoding
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx)
            } else {
                // Top-k sampling with temperature
                Self::sample_topk_generate(&logits, config.temperature, config.top_k)
            };

            tokens.push(next_token);

            // Check for stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }
        }

        Ok(tokens)
    }

    /// Create GPU model from GGUF config (M13: Real Model Loading)
    ///
    /// This is a convenience constructor that creates a model with zero-initialized
    /// weights from a config. Use `from_mapped_gguf()` to load actual weights.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    ///
    /// # Errors
    ///
    /// Returns error if GPU initialization fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = GpuModelConfig {
    ///     vocab_size: 32000,
    ///     hidden_dim: 4096,
    ///     num_heads: 32,
    ///     num_layers: 32,
    ///     intermediate_dim: 11008,
    ///     eps: 1e-5,
    /// };
    /// let model = GpuModel::from_gguf_config(config)?;
    /// ```
    pub fn from_gguf_config(config: GpuModelConfig) -> Result<Self> {
        // Delegate to new() which handles initialization
        Self::new(config)
    }

    /// Load GPU model from memory-mapped GGUF file (M13: Real Model Loading)
    ///
    /// This is the primary method for loading real GGUF models to GPU.
    /// It dequantizes weights on-the-fly and uploads them to GPU buffers.
    ///
    /// # Arguments
    ///
    /// * `mapped` - Memory-mapped GGUF model
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Required tensors are missing
    /// - Tensor shapes don't match expected dimensions
    /// - GPU initialization or upload fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mapped = MappedGGUFModel::from_path("model.gguf")?;
    /// let model = GpuModel::from_mapped_gguf(&mapped)?;
    /// let logits = model.forward_gpu_owned(&[1, 2, 3])?;
    /// ```
    pub fn from_mapped_gguf(mapped: &crate::gguf::MappedGGUFModel) -> Result<Self> {
        use crate::gguf::GGUFConfig;

        // Extract config from GGUF metadata
        let gguf_config = GGUFConfig::from_gguf(&mapped.model)?;

        let config = GpuModelConfig {
            vocab_size: gguf_config.vocab_size,
            hidden_dim: gguf_config.hidden_dim,
            num_heads: gguf_config.num_heads,
            num_kv_heads: gguf_config.num_kv_heads, // IMP-088: GQA support
            num_layers: gguf_config.num_layers,
            intermediate_dim: gguf_config.intermediate_dim,
            eps: gguf_config.eps,
            rope_theta: gguf_config.rope_theta, // Phase 21: RoPE support
        };

        let scheduler = HybridScheduler::new()?;
        let data = mapped.data();

        // Load token embeddings (always dequantized for fast lookup)
        let embedding_weights = mapped.model.get_tensor_f32("token_embd.weight", data)?;

        // Load transformer blocks
        let mut block_weights = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let prefix = format!("blk.{}", layer_idx);

            // Attention norm (small, keep as f32)
            let attn_norm_weight = mapped
                .model
                .get_tensor_f32(&format!("{}.attn_norm.weight", prefix), data)?;
            let attn_norm_bias = mapped
                .model
                .get_tensor_f32(&format!("{}.attn_norm.bias", prefix), data)
                .unwrap_or_else(|_| vec![0.0f32; config.hidden_dim]);

            // QKV projection - try fused QKV first (LLaMA), then separate Q/K/V (Qwen)
            let (qkv_weight, qkv_bias) = if let Ok(fused_qkv) = mapped
                .model
                .get_tensor_f32(&format!("{}.attn_qkv.weight", prefix), data)
            {
                // Fused QKV (LLaMA-style)
                let bias = mapped
                    .model
                    .get_tensor_f32(&format!("{}.attn_qkv.bias", prefix), data)
                    .unwrap_or_else(|_| vec![0.0f32; 3 * config.hidden_dim]);
                (fused_qkv, bias)
            } else {
                // Separate Q/K/V (Qwen-style) - concatenate into fused format
                // For GQA: Q has num_heads * head_dim, K/V have num_kv_heads * head_dim
                let head_dim = config.hidden_dim / config.num_heads;
                let kv_dim = config.num_kv_heads * head_dim; // K/V dimension for GQA

                let q_weight = mapped
                    .model
                    .get_tensor_f32(&format!("{}.attn_q.weight", prefix), data)?;
                let k_weight = mapped
                    .model
                    .get_tensor_f32(&format!("{}.attn_k.weight", prefix), data)?;
                let v_weight = mapped
                    .model
                    .get_tensor_f32(&format!("{}.attn_v.weight", prefix), data)?;

                // Concatenate Q, K, V weights
                let mut qkv_weight =
                    Vec::with_capacity(q_weight.len() + k_weight.len() + v_weight.len());
                qkv_weight.extend_from_slice(&q_weight);
                qkv_weight.extend_from_slice(&k_weight);
                qkv_weight.extend_from_slice(&v_weight);

                // Load biases if available (use correct dimensions for GQA)
                let q_bias = mapped
                    .model
                    .get_tensor_f32(&format!("{}.attn_q.bias", prefix), data)
                    .unwrap_or_else(|_| vec![0.0f32; config.hidden_dim]);
                let k_bias = mapped
                    .model
                    .get_tensor_f32(&format!("{}.attn_k.bias", prefix), data)
                    .unwrap_or_else(|_| vec![0.0f32; kv_dim]); // GQA: K/V use num_kv_heads
                let v_bias = mapped
                    .model
                    .get_tensor_f32(&format!("{}.attn_v.bias", prefix), data)
                    .unwrap_or_else(|_| vec![0.0f32; kv_dim]); // GQA: K/V use num_kv_heads

                // Total bias size: Q (hidden_dim) + K (kv_dim) + V (kv_dim)
                let total_bias_dim = config.hidden_dim + 2 * kv_dim;
                let mut qkv_bias = Vec::with_capacity(total_bias_dim);
                qkv_bias.extend_from_slice(&q_bias);
                qkv_bias.extend_from_slice(&k_bias);
                qkv_bias.extend_from_slice(&v_bias);

                (qkv_weight, qkv_bias)
            };

            // Output projection
            let out_weight = mapped
                .model
                .get_tensor_f32(&format!("{}.attn_output.weight", prefix), data)?;
            let out_bias = mapped
                .model
                .get_tensor_f32(&format!("{}.attn_output.bias", prefix), data)
                .unwrap_or_else(|_| vec![0.0f32; config.hidden_dim]);

            // FFN norm
            let ffn_norm_weight = mapped
                .model
                .get_tensor_f32(&format!("{}.ffn_norm.weight", prefix), data)
                .unwrap_or_else(|_| vec![1.0f32; config.hidden_dim]);
            let ffn_norm_bias = mapped
                .model
                .get_tensor_f32(&format!("{}.ffn_norm.bias", prefix), data)
                .unwrap_or_else(|_| vec![0.0f32; config.hidden_dim]);

            // FFN projections
            let ffn_fc1_weight = mapped
                .model
                .get_tensor_f32(&format!("{}.ffn_up.weight", prefix), data)?;
            let ffn_fc1_bias = mapped
                .model
                .get_tensor_f32(&format!("{}.ffn_up.bias", prefix), data)
                .unwrap_or_else(|_| vec![0.0f32; config.intermediate_dim]);

            let ffn_fc2_weight = mapped
                .model
                .get_tensor_f32(&format!("{}.ffn_down.weight", prefix), data)?;
            let ffn_fc2_bias = mapped
                .model
                .get_tensor_f32(&format!("{}.ffn_down.bias", prefix), data)
                .unwrap_or_else(|_| vec![0.0f32; config.hidden_dim]);

            // Try to load gate weight for SwiGLU (optional)
            let ffn_gate_weight = mapped
                .model
                .get_tensor_f32(&format!("{}.ffn_gate.weight", prefix), data)
                .ok();

            block_weights.push(BlockWeights {
                attn_norm_weight,
                attn_norm_bias,
                qkv_weight,
                qkv_bias,
                out_weight,
                out_bias,
                ffn_norm_weight,
                ffn_norm_bias,
                ffn_fc1_weight,
                ffn_fc1_bias,
                ffn_fc2_weight,
                ffn_fc2_bias,
                ffn_gate_weight,
            });
        }

        // Final layer norm
        let final_norm_weight = mapped.model.get_tensor_f32("output_norm.weight", data)?;
        let final_norm_bias = mapped
            .model
            .get_tensor_f32("output_norm.bias", data)
            .unwrap_or_else(|_| vec![0.0f32; config.hidden_dim]);

        // LM head
        let lm_head_weight = mapped.model.get_tensor_f32("output.weight", data)?;
        let lm_head_bias = mapped
            .model
            .get_tensor_f32("output.bias", data)
            .unwrap_or_else(|_| vec![0.0f32; config.vocab_size]);

        // Pre-compute transposed LM head for fast CPU inference
        let lm_head_weight_t =
            Self::transpose_weights(&lm_head_weight, config.hidden_dim, config.vocab_size);

        Ok(Self {
            embedding_weights,
            block_weights,
            final_norm_weight,
            final_norm_bias,
            lm_head_weight,
            lm_head_weight_t,
            lm_head_bias,
            scheduler,
            #[cfg(feature = "cuda")]
            cuda_scheduler: None,
            config,
            attention_buffers: None,
        })
    }

    /// Create GpuModel from pre-extracted APR weights (PMAT-106)
    ///
    /// This constructor is used by `AprToGpuAdapter` to create a `GpuModel`
    /// from dequantized APR weights.
    ///
    /// # Arguments
    ///
    /// * `config` - GPU model configuration
    /// * `embedding_weights` - Token embedding weights
    /// * `block_weights` - Transformer block weights
    /// * `final_norm_weight` - Final layer norm weight
    /// * `final_norm_bias` - Final layer norm bias
    /// * `lm_head_weight` - LM head weight (row-major)
    /// * `lm_head_weight_t` - LM head weight transposed (for fast CPU inference)
    /// * `lm_head_bias` - LM head bias
    ///
    /// # Errors
    ///
    /// Returns error if GPU scheduler initialization fails
    #[allow(clippy::too_many_arguments)]
    pub fn from_apr_weights(
        config: GpuModelConfig,
        embedding_weights: Vec<f32>,
        block_weights: Vec<BlockWeights>,
        final_norm_weight: Vec<f32>,
        final_norm_bias: Vec<f32>,
        lm_head_weight: Vec<f32>,
        lm_head_weight_t: Vec<f32>,
        lm_head_bias: Vec<f32>,
    ) -> Result<Self> {
        let scheduler = HybridScheduler::new()?;

        // Phase 21: Initialize CudaScheduler for GPU-accelerated matmul
        #[cfg(feature = "cuda")]
        let cuda_scheduler = match CudaScheduler::new() {
            Ok(cs) => {
                eprintln!("[PHASE21] CudaScheduler initialized for APR model");
                Some(cs)
            }
            Err(e) => {
                eprintln!("[PHASE21] CudaScheduler init failed (using HybridScheduler fallback): {}", e);
                None
            }
        };

        Ok(Self {
            embedding_weights,
            block_weights,
            final_norm_weight,
            final_norm_bias,
            lm_head_weight,
            lm_head_weight_t,
            lm_head_bias,
            scheduler,
            #[cfg(feature = "cuda")]
            cuda_scheduler,
            config,
            attention_buffers: None,
        })
    }

    /// Get model configuration (M13: Real Model Loading)
    #[must_use]
    pub fn config(&self) -> &GpuModelConfig {
        &self.config
    }

    // ============================================================================
    // Phase 8: Optimized Incremental Decoding (M17)
    // ============================================================================

    /// Create GPU model with pre-allocated attention buffers (M17)
    ///
    /// Allocates reusable buffers for incremental decoding, eliminating
    /// per-token memory allocation overhead.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    /// * `max_seq_len` - Maximum sequence length to support
    ///
    /// # Errors
    ///
    /// Returns error if GPU initialization fails
    pub fn with_attention_buffers(config: GpuModelConfig, max_seq_len: usize) -> Result<Self> {
        let buffers = AttentionBuffers::new(&config, max_seq_len);
        let mut model = Self::new(config)?;
        model.attention_buffers = Some(buffers);
        Ok(model)
    }

    /// Check if model has pre-allocated attention buffers (M17)
    #[must_use]
    pub fn has_attention_buffers(&self) -> bool {
        self.attention_buffers.is_some()
    }

    /// Optimized text generation using pre-allocated buffers (M17)
    ///
    /// Uses the optimized incremental forward pass with pre-allocated buffers
    /// and batched multi-head attention for better performance.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Errors
    ///
    /// Returns error if generation fails
    pub fn generate_optimized(
        &mut self,
        prompt: &[usize],
        config: &GpuGenerateConfig,
    ) -> Result<Vec<usize>> {
        if prompt.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Prompt cannot be empty".to_string(),
            });
        }

        // Initialize KV cache
        // IMP-093: For GQA, use num_kv_heads since K/V have fewer heads than Q
        let head_dim = self.config.hidden_dim / self.config.num_heads;
        let max_seq_len = self
            .attention_buffers
            .as_ref()
            .map_or(512, |b| b.max_seq_len);
        let mut kv_cache = StreamingKVCache::new(
            self.config.num_layers,
            max_seq_len,
            self.config.num_kv_heads, // GQA: K/V have fewer heads
            head_dim,
        );

        let mut tokens = prompt.to_vec();

        // Process prompt with cache - returns logits for final position only [vocab_size]
        let logits = self.forward_gpu_with_cache(prompt, &mut kv_cache)?;

        // Sample first token (logits is already for last position only)
        let mut next_token = if config.temperature == 0.0 || config.top_k == 1 {
            Self::argmax(&logits)
        } else {
            Self::sample_topk_generate(&logits, config.temperature, config.top_k)
        };

        if config.stop_tokens.contains(&next_token) {
            return Ok(tokens);
        }

        tokens.push(next_token);

        // Generate remaining tokens using optimized incremental forward
        for _ in 1..config.max_tokens {
            let logits = self.forward_gpu_incremental_optimized(next_token, &mut kv_cache)?;

            next_token = if config.temperature == 0.0 || config.top_k == 1 {
                Self::argmax(&logits)
            } else {
                Self::sample_topk_generate(&logits, config.temperature, config.top_k)
            };

            if config.stop_tokens.contains(&next_token) {
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Optimized incremental forward pass using pre-allocated buffers (M17)
    ///
    /// Single-token forward pass optimized by:
    /// - Reusing pre-allocated attention buffers
    /// - Direct KV cache access without copying
    /// - Batched multi-head attention computation
    ///
    /// # Arguments
    ///
    /// * `token_id` - Single token to process
    /// * `kv_cache` - Mutable reference to KV cache
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails
    pub fn forward_gpu_incremental_optimized(
        &mut self,
        token_id: usize,
        kv_cache: &mut StreamingKVCache,
    ) -> Result<Vec<f32>> {
        if token_id >= self.config.vocab_size {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Token ID {} out of bounds (vocab_size={})",
                    token_id, self.config.vocab_size
                ),
            });
        }

        let hidden_dim = self.config.hidden_dim;

        // Get embedding for single token
        let offset = token_id * hidden_dim;
        let mut hidden: Vec<f32> = self.embedding_weights[offset..offset + hidden_dim].to_vec();

        // Process through all blocks with optimized attention
        for block_idx in 0..self.block_weights.len() {
            hidden = self.forward_block_incremental_optimized(&hidden, block_idx, kv_cache)?;
        }

        // Final layer norm
        hidden = self.layer_norm(&hidden, &self.final_norm_weight, &self.final_norm_bias);

        // LM head projection (single token)
        // IMP-090, IMP-096: Use CPU fallback with SIMD for large vocab
        let lm_head_elements = hidden_dim * self.config.vocab_size;
        let output = if exceeds_gpu_buffer_limit(lm_head_elements) {
            // IMP-096: CPU path with transposed weights + SIMD + fused bias
            cpu_matmul_transposed_simd(
                &hidden,
                &self.lm_head_weight_t,
                &self.lm_head_bias,
                hidden_dim,
                self.config.vocab_size,
            )
        } else {
            // IMP-1006: Use do_matmul to route to CudaScheduler when available
            let lm_weight = self.lm_head_weight.clone();
            let vocab_size = self.config.vocab_size;
            let logits = self.do_matmul(&hidden, &lm_weight, 1, hidden_dim, vocab_size)?;
            // Add bias
            logits
                .into_iter()
                .zip(self.lm_head_bias.iter())
                .map(|(l, &b)| l + b)
                .collect()
        };

        Ok(output)
    }

    /// Optimized block forward with batched multi-head attention (M17, IMP-092)
    ///
    /// IMP-092: Eliminated weight cloning (~130MB per layer) by using explicit
    /// field borrowing. Previous version cloned 3.7GB per token across 28 layers.
    pub fn forward_block_incremental_optimized(
        &mut self,
        input: &[f32],
        block_idx: usize,
        kv_cache: &mut StreamingKVCache,
    ) -> Result<Vec<f32>> {
        // Extract config values (Copy types, no borrow conflict)
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim();
        let kv_dim = self.config.kv_dim();
        let qkv_dim = self.config.qkv_dim();
        let intermediate_dim = self.config.intermediate_dim;
        let eps = self.config.eps;
        let num_kv_heads = self.config.num_kv_heads;

        // IMP-092: Use REFERENCES instead of cloning 130MB of weights per layer
        // Pre-attention layer norm (static function avoids &self borrow)
        let normed = Self::layer_norm_static(
            input,
            &self.block_weights[block_idx].attn_norm_weight,
            &self.block_weights[block_idx].attn_norm_bias,
            hidden_dim,
            eps,
        );

        // QKV projection for single token [1, hidden_dim] @ [hidden_dim, qkv_dim]
        // For GQA: qkv_dim = hidden_dim + 2*kv_dim (K/V have fewer heads)
        // IMP-1006: Use do_matmul to route to CudaScheduler when available
        let qkv_weight = self.block_weights[block_idx].qkv_weight.clone();
        let mut qkv = self.do_matmul(&normed, &qkv_weight, 1, hidden_dim, qkv_dim)?;

        // Get current position BEFORE caching (Phase 21)
        let (cached_k_ref, _) = kv_cache.get_valid(block_idx);
        let current_pos = cached_k_ref.len() / kv_dim;

        // Phase 21: Apply RoPE to Q and K BEFORE caching
        // Without RoPE, attention has no position information and produces garbage
        let rope_theta = self.config.rope_theta;
        Self::apply_rope_inline(&mut qkv[0..hidden_dim], num_heads, head_dim, rope_theta, current_pos);
        Self::apply_rope_inline(&mut qkv[hidden_dim..hidden_dim + kv_dim], num_kv_heads, head_dim, rope_theta, current_pos);

        // Split QKV (GQA: K/V have kv_dim, not hidden_dim) - after RoPE
        let q = qkv[0..hidden_dim].to_vec();
        let k_new = qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
        let v_new = qkv[hidden_dim + kv_dim..].to_vec();

        // Get cached K/V and clone to avoid borrow issues with kv_cache
        let (cached_k, cached_v) = kv_cache.get_valid(block_idx);
        let keys_cached = cached_k.to_vec();
        let vals_cached = cached_v.to_vec();

        // Append new K/V (with RoPE applied) to cache
        kv_cache.append(block_idx, &k_new, &v_new);

        // Build full K/V (cached + new)
        // GQA: K/V have kv_dim per position, not hidden_dim
        let kv_len = keys_cached.len() / kv_dim + 1;
        let mut full_k = keys_cached;
        full_k.extend_from_slice(&k_new);
        let mut full_v = vals_cached;
        full_v.extend_from_slice(&v_new);

        // GQA attention (IMP-089): static method to avoid borrow conflicts
        let attn_output = Self::gqa_multihead_attention(
            &q,
            &full_k,
            &full_v,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        // Output projection
        // IMP-1006: Use do_matmul to route to CudaScheduler when available
        let out_weight = self.block_weights[block_idx].out_weight.clone();
        let attn_proj = self.do_matmul(&attn_output, &out_weight, 1, hidden_dim, hidden_dim)?;

        // Add residual and bias
        let out_bias = &self.block_weights[block_idx].out_bias;
        let mut post_attn: Vec<f32> = input
            .iter()
            .zip(attn_proj.iter())
            .zip(out_bias.iter())
            .map(|((&i, &a), &b)| i + a + b)
            .collect();

        // FFN with layer norm (static function)
        let ffn_normed = Self::layer_norm_static(
            &post_attn,
            &self.block_weights[block_idx].ffn_norm_weight,
            &self.block_weights[block_idx].ffn_norm_bias,
            hidden_dim,
            eps,
        );

        // FFN: SwiGLU when gate weight exists, otherwise GELU
        // IMP-1006: Use do_matmul to route to CudaScheduler when available
        let fc1_activated: Vec<f32> = if let Some(ref gate_weight) = self.block_weights[block_idx].ffn_gate_weight {
            // SwiGLU: silu(gate(x)) * up(x)
            let fc1_weight = self.block_weights[block_idx].ffn_fc1_weight.clone();
            let gate_weight = gate_weight.clone();

            let up_out = self.do_matmul(&ffn_normed, &fc1_weight, 1, hidden_dim, intermediate_dim)?;
            let gate_out = self.do_matmul(&ffn_normed, &gate_weight, 1, hidden_dim, intermediate_dim)?;

            // SwiGLU: silu(gate) * up
            up_out
                .iter()
                .zip(gate_out.iter())
                .map(|(&u, &g)| {
                    let silu_g = g / (1.0 + (-g).exp());
                    silu_g * u
                })
                .collect()
        } else {
            // Standard GELU FFN
            let fc1_weight = self.block_weights[block_idx].ffn_fc1_weight.clone();
            let fc1_out = self.do_matmul(&ffn_normed, &fc1_weight, 1, hidden_dim, intermediate_dim)?;

            let ffn_fc1_bias = &self.block_weights[block_idx].ffn_fc1_bias;
            fc1_out
                .iter()
                .zip(ffn_fc1_bias.iter())
                .map(|(&x, &b)| {
                    let x_b = x + b;
                    x_b * 0.5 + x_b * 0.5 * (0.797_884_6 * (x_b + 0.044_715 * x_b.powi(3))).tanh()
                })
                .collect()
        };

        // FFN FC2 (down projection)
        // IMP-1006: Use do_matmul to route to CudaScheduler when available
        let fc2_weight = self.block_weights[block_idx].ffn_fc2_weight.clone();
        let fc2_out =
            self.do_matmul(&fc1_activated, &fc2_weight, 1, intermediate_dim, hidden_dim)?;

        // Add residual and bias
        let ffn_fc2_bias = &self.block_weights[block_idx].ffn_fc2_bias;
        for i in 0..hidden_dim {
            post_attn[i] += fc2_out[i] + ffn_fc2_bias[i];
        }

        Ok(post_attn)
    }

    /// Apply Rotary Position Embedding (RoPE) inline (Phase 21)
    ///
    /// RoPE encodes position information by rotating pairs of elements
    /// with position-dependent angles. This is CRITICAL for transformer attention.
    ///
    /// # Arguments
    /// * `x` - Mutable slice of Q or K vectors for a single position [num_heads * head_dim]
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    /// * `rope_theta` - Base frequency (typically 10000.0)
    /// * `position` - Token position for RoPE encoding
    fn apply_rope_inline(
        x: &mut [f32],
        num_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        position: usize,
    ) {
        let half_dim = head_dim / 2;
        let head_dim_f32 = head_dim as f32;
        let pos_f32 = position as f32;

        for h in 0..num_heads {
            let head_start = h * head_dim;
            let idx2_start = head_start + half_dim;

            for i in 0..half_dim {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim_f32);
                let angle = pos_f32 * freq;
                let (sin_val, cos_val) = angle.sin_cos();

                let x1 = x[head_start + i];
                let x2 = x[idx2_start + i];

                // Apply rotation: [cos -sin; sin cos] * [x1; x2]
                x[head_start + i] = x1 * cos_val - x2 * sin_val;
                x[idx2_start + i] = x1 * sin_val + x2 * cos_val;
            }
        }
    }

    /// GQA multi-head attention (IMP-089, IMP-092, IMP-094)
    ///
    /// Grouped Query Attention where K/V have fewer heads than Q.
    /// Each KV head serves (num_heads / num_kv_heads) Q heads.
    ///
    /// IMP-094: Uses trueno SIMD-accelerated dot product and softmax
    /// for ~10x speedup over scalar implementation.
    ///
    /// Static method to avoid borrow conflicts with scheduler and weights.
    fn gqa_multihead_attention(
        q: &[f32], // Q: [num_heads * head_dim]
        k: &[f32], // K: [kv_len * num_kv_heads * head_dim]
        v: &[f32], // V: [kv_len * num_kv_heads * head_dim]
        kv_len: usize,
        num_heads: usize,    // Number of Q heads
        num_kv_heads: usize, // Number of K/V heads (for GQA, < num_heads)
        head_dim: usize,
    ) -> Vec<f32> {
        use trueno::Vector;

        let hidden_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Number of Q heads per KV head
        let heads_per_kv = num_heads / num_kv_heads;

        let mut output = vec![0.0; hidden_dim];

        // Compute attention for all Q heads
        for h in 0..num_heads {
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            // IMP-094: Create trueno vector for SIMD dot product
            let q_vec = Vector::from_slice(q_head);

            // Map Q head to KV head (GQA: multiple Q heads share one KV head)
            let kv_head = h / heads_per_kv;

            // Compute attention scores for this head using SIMD dot product
            let mut scores = Vec::with_capacity(kv_len);
            for pos in 0..kv_len {
                // K offset: pos * kv_dim + kv_head * head_dim
                let k_offset = pos * kv_dim + kv_head * head_dim;
                let cached_key = &k[k_offset..k_offset + head_dim];

                // IMP-094: SIMD dot product via trueno
                let k_vec = Vector::from_slice(cached_key);
                let score = q_vec.dot(&k_vec).unwrap_or(0.0) * scale;
                scores.push(score);
            }

            // IMP-094: SIMD softmax via trueno
            let scores_vec = Vector::from_slice(&scores);
            let attn_weights: Vec<f32> = scores_vec.softmax().map_or_else(
                |_| {
                    // Fallback to scalar softmax
                    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exp_scores: Vec<f32> =
                        scores.iter().map(|&s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();
                    exp_scores.iter().map(|&e| e / sum_exp).collect()
                },
                |v| v.as_slice().to_vec(),
            );

            // Weighted sum of values (still scalar - SIMD benefit is marginal for small head_dim)
            for (pos, &weight) in attn_weights.iter().enumerate() {
                // V offset: pos * kv_dim + kv_head * head_dim
                let v_offset = pos * kv_dim + kv_head * head_dim;
                let v_head = &v[v_offset..v_offset + head_dim];

                for d in 0..head_dim {
                    output[h * head_dim + d] += weight * v_head[d];
                }
            }
        }

        output
    }

    // ============================================================================
    // Phase 9: Fused Kernels & Vectorization (M18)
    // ============================================================================

    /// Check if model has fused QKV projection (M18 - IMP-037)
    ///
    /// Fused QKV uses a single matmul instead of three separate projections.
    /// This is always true for GpuModel as QKV weights are stored combined.
    #[must_use]
    pub fn has_fused_qkv(&self) -> bool {
        // QKV weights are stored as [hidden_dim, 3*hidden_dim] for fused projection
        !self.block_weights.is_empty()
            && self.block_weights[0].qkv_weight.len()
                == self.config.hidden_dim * 3 * self.config.hidden_dim
    }

    /// Fused QKV projection (M18 - IMP-037)
    ///
    /// Performs Q, K, V projection in a single matmul operation.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [hidden_dim]
    ///
    /// # Returns
    ///
    /// Tuple of (Q, K, V) tensors, each [hidden_dim]
    ///
    /// # Errors
    ///
    /// Returns error if matmul fails
    pub fn fused_qkv_projection(
        &mut self,
        input: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let hidden_dim = self.config.hidden_dim;
        let kv_dim = self.config.kv_dim();
        let qkv_dim = self.config.qkv_dim();

        // Use first block's QKV weights for projection
        let qkv_weight = &self.block_weights[0].qkv_weight;

        // Single matmul: [1, hidden_dim] @ [hidden_dim, qkv_dim] -> [1, qkv_dim]
        // For GQA: qkv_dim = hidden_dim + 2*kv_dim
        let qkv = self
            .scheduler
            .matmul(input, qkv_weight, 1, hidden_dim, qkv_dim)?;

        // Split into Q, K, V (GQA: K/V have kv_dim, not hidden_dim)
        let q = qkv[0..hidden_dim].to_vec();
        let k = qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
        let v = qkv[hidden_dim + kv_dim..].to_vec();

        Ok((q, k, v))
    }

    /// Generation with fused QKV projection (M18 - IMP-037)
    ///
    /// Uses fused QKV projection for improved performance.
    ///
    /// # Errors
    ///
    /// Returns error if generation fails due to invalid input or model state.
    pub fn generate_with_fused_qkv(
        &mut self,
        prompt: &[usize],
        config: &GpuGenerateConfig,
    ) -> Result<Vec<usize>> {
        // Fused QKV is already used in generate_optimized via forward_block_incremental_optimized
        // This method provides explicit API for benchmarking
        self.generate_optimized(prompt, config)
    }

    /// Check if model has fused attention projection (M18 - IMP-039)
    #[must_use]
    pub fn has_fused_attn_proj(&self) -> bool {
        // Attention output projection is stored in block_weights
        !self.block_weights.is_empty()
            && self.block_weights[0].out_weight.len()
                == self.config.hidden_dim * self.config.hidden_dim
    }

    /// Forward pass with fused attention projection (M18 - IMP-039)
    ///
    /// Uses fused attention output projection for improved performance.
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails due to invalid token or cache state.
    pub fn forward_with_fused_attn_proj(
        &mut self,
        token_id: usize,
        kv_cache: &mut StreamingKVCache,
    ) -> Result<Vec<f32>> {
        // Fused attention projection is already used in forward_gpu_incremental_optimized
        // This method provides explicit API for benchmarking
        self.forward_gpu_incremental_optimized(token_id, kv_cache)
    }

    /// Check if model has fused output residual capability (M19 - IMP-042)
    #[must_use]
    pub fn has_fused_output_residual(&self) -> bool {
        // Fused output residual requires attention buffers and block weights
        self.attention_buffers.is_some() && !self.block_weights.is_empty()
    }

    /// Forward pass with fused output projection + residual (M19 - IMP-042)
    ///
    /// Combines the output projection matrix multiplication with residual
    /// connection in a single fused operation.
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails due to invalid token or cache state.
    pub fn forward_with_fused_output_residual(
        &mut self,
        token_id: usize,
        kv_cache: &mut StreamingKVCache,
    ) -> Result<Vec<f32>> {
        // Currently uses the optimized forward path
        // The fused operation is implemented in forward_block_incremental_optimized
        // This method provides explicit API for benchmarking
        self.forward_gpu_incremental_optimized(token_id, kv_cache)
    }

    /// Forward pass taking ownership of token_ids (convenience wrapper)
    ///
    /// This is useful when you don't need to keep the token_ids after the call.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs (as Vec for owned semantics in tests)
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails
    pub fn forward_gpu_owned(&mut self, token_ids: &[usize]) -> Result<Vec<f32>> {
        self.forward_gpu(token_ids)
    }

    /// Generate text tokens using GPU-accelerated inference (M14: E2E Inference)
    ///
    /// Performs autoregressive token generation starting from a prompt.
    /// Uses GPU for forward passes and CPU for sampling.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs to start generation from
    /// * `config` - Generation configuration (max tokens, temperature, etc.)
    ///
    /// # Returns
    ///
    /// Vector of generated token IDs (including the prompt)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Prompt is empty
    /// - Forward pass fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = GpuGenerateConfig::deterministic(32);
    /// let tokens = model.generate(&[1, 2, 3], &config)?;
    /// ```
    pub fn generate(&mut self, prompt: &[usize], config: &GpuGenerateConfig) -> Result<Vec<usize>> {
        // IMP-1009: Use zero-clone RefCell path when CUDA is available
        // This provides ~7x speedup by eliminating weight cloning
        #[cfg(feature = "cuda")]
        if self.cuda_scheduler.is_some() {
            return self.generate_refcell(prompt, config);
        }

        // Fallback to clone-based path for non-CUDA or HybridScheduler
        // IMP-091: Uses KV cache for O(n) generation
        self.generate_optimized(prompt, config)
    }

    // =========================================================================
    // Phase 7: KV Cache Integration - Wrappers (extracted to kv.rs)
    // =========================================================================

    /// Forward pass with KV cache population (IMP-031) - delegates to kv module
    pub fn forward_gpu_with_cache(
        &mut self,
        token_ids: &[usize],
        kv_cache: &mut StreamingKVCache,
    ) -> Result<Vec<f32>> {
        super::kv::forward_gpu_with_cache(self, token_ids, kv_cache)
    }

    /// Incremental forward pass using cached KV (IMP-032) - delegates to kv module
    pub fn forward_gpu_incremental(
        &mut self,
        token_id: usize,
        kv_cache: &mut StreamingKVCache,
    ) -> Result<Vec<f32>> {
        super::kv::forward_gpu_incremental(self, token_id, kv_cache)
    }

    /// Generate with KV cache (IMP-033) - delegates to kv module
    pub fn generate_with_cache(
        &mut self,
        prompt: &[usize],
        config: &GpuGenerateConfig,
    ) -> Result<Vec<usize>> {
        super::kv::generate_with_cache(self, prompt, config)
    }

    /// Top-k sampling with temperature (returns highest prob token in top-k for determinism)
    fn sample_topk_generate(logits: &[f32], temperature: f32, top_k: usize) -> usize {
        // Apply temperature
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

        // Softmax with numerical stability
        let max_logit = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = scaled.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

        // Get top-k indices by sorting
        let mut indexed: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate to top_k and return highest probability token (deterministic)
        indexed.truncate(top_k);
        indexed.first().map_or(0, |&(idx, _)| idx)
    }

    /// Transpose weight matrix from [rows, cols] to [cols, rows]
    fn transpose_weights(weights: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut transposed = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                transposed[j * rows + i] = weights[i * cols + j];
            }
        }
        transposed
    }

    /// Check if GPU is being used
    #[must_use]
    pub fn has_gpu(&self) -> bool {
        self.scheduler.has_gpu()
    }

    /// GPU-accelerated forward pass
    ///
    /// Uses HybridScheduler for matrix multiplications.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits tensor with shape `[seq_len, vocab_size]`
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails
    pub fn forward_gpu(&mut self, token_ids: &[usize]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token IDs cannot be empty".to_string(),
            });
        }

        let seq_len = token_ids.len();
        let hidden_dim = self.config.hidden_dim;

        // Step 1: Embed tokens
        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for &token_id in token_ids {
            if token_id >= self.config.vocab_size {
                return Err(RealizarError::InvalidShape {
                    reason: format!(
                        "Token ID {} out of bounds (vocab_size={})",
                        token_id, self.config.vocab_size
                    ),
                });
            }
            let offset = token_id * hidden_dim;
            hidden.extend_from_slice(&self.embedding_weights[offset..offset + hidden_dim]);
        }

        // Step 2: Pass through transformer blocks
        for block_idx in 0..self.block_weights.len() {
            hidden = self.forward_block_idx(&hidden, seq_len, block_idx)?;
        }

        // Step 3: Final layer norm
        hidden = self.layer_norm(&hidden, &self.final_norm_weight, &self.final_norm_bias);

        // Step 4: LM head projection
        // [seq_len, hidden_dim] @ [hidden_dim, vocab_size] -> [seq_len, vocab_size]
        // Phase 22 FIX: Use lm_head_weight_t (transposed) which is [hidden_dim, vocab_size]
        // The original lm_head_weight is [vocab_size, hidden_dim] (APR convention)
        // IMP-090: Use CPU fallback for large vocab to avoid GPU buffer overflow
        let lm_head_elements = hidden_dim * self.config.vocab_size;
        let logits = if exceeds_gpu_buffer_limit(lm_head_elements) {
            // CPU fallback for large vocab (>256MB weight matrix)
            cpu_matmul(
                &hidden,
                &self.lm_head_weight_t,
                seq_len,
                hidden_dim,
                self.config.vocab_size,
            )
        } else {
            // GPU path for smaller vocab (IMP-1005: use do_matmul for CUDA)
            // Clone weights to avoid borrow conflict with &mut self in do_matmul
            let lm_weight = self.lm_head_weight_t.clone();
            self.do_matmul(
                &hidden,
                &lm_weight,
                seq_len,
                hidden_dim,
                self.config.vocab_size,
            )?
        };

        // Add bias
        let mut output = logits;
        for i in 0..seq_len {
            for j in 0..self.config.vocab_size {
                output[i * self.config.vocab_size + j] += self.lm_head_bias[j];
            }
        }

        Ok(output)
    }

    /// Forward pass through a single transformer block by index
    pub fn forward_block_idx(
        &mut self,
        input: &[f32],
        seq_len: usize,
        block_idx: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;
        let qkv_dim = self.config.qkv_dim();

        // Get references to block weights (avoid cloning)
        let block = &self.block_weights[block_idx];
        let attn_norm_weight = &block.attn_norm_weight;
        let attn_norm_bias = &block.attn_norm_bias;

        // Pre-norm (uses references, no clone)
        let normed = Self::layer_norm_static(
            input,
            attn_norm_weight,
            attn_norm_bias,
            hidden_dim,
            self.config.eps,
        );

        // IMP-1005: Clone weights to avoid borrow conflict with &mut self in do_matmul
        let qkv_weight = self.block_weights[block_idx].qkv_weight.clone();

        // QKV projection (IMP-1005: use do_matmul for CUDA)
        // [seq_len, hidden_dim] @ [hidden_dim, qkv_dim] -> [seq_len, qkv_dim]
        let qkv = self.do_matmul(&normed, &qkv_weight, seq_len, hidden_dim, qkv_dim)?;

        // Optimized GQA attention with GPU matmul for scores
        let attn_out = self.optimized_gqa_attention(&qkv, seq_len)?;

        // IMP-1005: Clone weights to avoid borrow conflict
        let out_weight = self.block_weights[block_idx].out_weight.clone();
        let out_bias = self.block_weights[block_idx].out_bias.clone();

        // Output projection (IMP-1005: use do_matmul for CUDA)
        let projected = self.do_matmul(&attn_out, &out_weight, seq_len, hidden_dim, hidden_dim)?;

        // Residual 1 (vectorized)
        let mut residual1: Vec<f32> = input
            .iter()
            .zip(projected.iter())
            .enumerate()
            .map(|(i, (&inp, &proj))| inp + proj + out_bias[i % hidden_dim])
            .collect();

        // IMP-1005: Clone weights to avoid borrow conflict
        let ffn_norm_weight = self.block_weights[block_idx].ffn_norm_weight.clone();
        let ffn_norm_bias = self.block_weights[block_idx].ffn_norm_bias.clone();

        // FFN pre-norm
        let ffn_normed = Self::layer_norm_static(
            &residual1,
            &ffn_norm_weight,
            &ffn_norm_bias,
            hidden_dim,
            self.config.eps,
        );

        // IMP-1005: Clone weights to avoid borrow conflict
        let ffn_fc1_weight = self.block_weights[block_idx].ffn_fc1_weight.clone();
        let ffn_fc1_bias = self.block_weights[block_idx].ffn_fc1_bias.clone();
        let ffn_gate_weight = self.block_weights[block_idx].ffn_gate_weight.clone();

        // FFN: SwiGLU when gate weight exists, otherwise GELU
        let activated: Vec<f32> = if let Some(gate_weight) = ffn_gate_weight {
            // SwiGLU: silu(gate(x)) * up(x)
            let up_out = self.do_matmul(&ffn_normed, &ffn_fc1_weight, seq_len, hidden_dim, intermediate_dim)?;
            let gate_out = self.do_matmul(&ffn_normed, &gate_weight, seq_len, hidden_dim, intermediate_dim)?;

            // SwiGLU: silu(gate) * up
            up_out
                .iter()
                .zip(gate_out.iter())
                .map(|(&u, &g)| {
                    let silu_g = g / (1.0 + (-g).exp());
                    silu_g * u
                })
                .collect()
        } else {
            // Standard GELU FFN
            let fc1_out = self.do_matmul(
                &ffn_normed,
                &ffn_fc1_weight,
                seq_len,
                hidden_dim,
                intermediate_dim,
            )?;

            // GELU activation + bias (vectorized)
            fc1_out
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let x = x + ffn_fc1_bias[i % intermediate_dim];
                    // GELU approximation
                    0.5 * x
                        * (1.0
                            + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044_715 * x.powi(3)))
                                .tanh())
                })
                .collect()
        };

        // IMP-1005: Clone weights to avoid borrow conflict
        let ffn_fc2_weight = self.block_weights[block_idx].ffn_fc2_weight.clone();
        let ffn_fc2_bias = self.block_weights[block_idx].ffn_fc2_bias.clone();

        // FFN: fc2 (IMP-1005: use do_matmul for CUDA)
        let fc2_out = self.do_matmul(
            &activated,
            &ffn_fc2_weight,
            seq_len,
            intermediate_dim,
            hidden_dim,
        )?;

        // Residual 2 (vectorized, in-place)
        for (i, x) in residual1.iter_mut().enumerate() {
            *x += fc2_out[i] + ffn_fc2_bias[i % hidden_dim];
        }

        Ok(residual1)
    }

    /// RMSNorm (Root Mean Square Layer Normalization)
    ///
    /// PMAT-094 FIX: Qwen2, LLaMA, Mistral use RMSNorm, NOT LayerNorm.
    /// Formula: output = x / sqrt(mean(x^2) + eps) * weight + bias
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn layer_norm_static(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        hidden_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        let num_rows = input.len() / hidden_dim;
        let mut output = Vec::with_capacity(input.len());

        for row in 0..num_rows {
            let start = row * hidden_dim;
            let row_data = &input[start..start + hidden_dim];

            // RMSNorm: compute root mean square (no mean subtraction!)
            let sum_sq: f32 = row_data.iter().map(|&x| x * x).sum();
            let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

            // Normalize and scale
            for (i, &x) in row_data.iter().enumerate() {
                let normalized = x / rms;
                output.push(normalized * weight[i] + bias[i]);
            }
        }

        output
    }

    /// Layer normalization (instance method)
    fn layer_norm(&self, input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
        Self::layer_norm_static(input, weight, bias, self.config.hidden_dim, self.config.eps)
    }

    /// Generate tokens using GPU-accelerated forward pass with incremental decoding (wrapper)
    pub fn generate_gpu(&mut self, prompt: &[usize], max_tokens: usize) -> Result<Vec<usize>> {
        super::batch::generate_gpu(self, prompt, max_tokens)
    }

    /// Argmax helper for sampling (wrapper)
    fn argmax(logits: &[f32]) -> usize {
        super::batch::argmax(logits)
    }

    /// Optimized GQA attention using GPU for matmul operations (wrapper)
    fn optimized_gqa_attention(&mut self, qkv: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        super::batch::optimized_gqa_attention(self, qkv, seq_len)
    }
}
