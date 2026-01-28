//! Inference support types for quantized model execution
//!
//! This module contains pre-allocated buffers and caches for zero-allocation inference.
//!
//! Types included:
//! - `InferenceScratchBuffer`: Scratch buffers for forward passes
//! - `OwnedInferenceScratchBuffer`: Owned variant with Q8K support
//! - `ContiguousKVCache`: Cache-aligned KV cache for efficient attention
//! - `DispatchMetrics`: Thread-safe CPU/GPU dispatch metrics

use super::config::GGUFConfig;

/// Cache line size in bytes (typical x86-64)
const CACHE_LINE_BYTES: usize = 64;

/// Number of f32 elements per cache line (64 bytes / 4 bytes per f32)
const FLOATS_PER_CACHE_LINE: usize = CACHE_LINE_BYTES / std::mem::size_of::<f32>();

// ============================================================================
// InferenceScratchBuffer
// ============================================================================

/// Pre-allocated scratch buffers for zero-allocation forward passes
///
/// ## Buffer Reuse Pattern
///
/// - First use: hidden → normed → qkv → q/k/v → attn_out
/// - FFN pass: normed → ffn_up/ffn_gate → ffn_down → hidden
///
/// PAR-126: Added Q8K scratch buffers for VNNI-accelerated Q4K×Q8K matmul path.
#[derive(Debug)]
pub struct InferenceScratchBuffer {
    /// Hidden state buffer [hidden_dim]
    pub hidden: Vec<f32>,
    /// Normalized hidden state [hidden_dim]
    pub normed: Vec<f32>,
    /// Combined QKV projection [q_dim + k_dim + v_dim]
    pub qkv: Vec<f32>,
    /// Query projection [q_dim]
    pub q: Vec<f32>,
    /// Key projection [k_dim]
    pub k: Vec<f32>,
    /// Value projection [v_dim]
    pub v: Vec<f32>,
    /// Attention output [hidden_dim]
    pub attn_out: Vec<f32>,
    /// Attention projection output [hidden_dim]
    pub attn_proj: Vec<f32>,
    /// FFN up projection [intermediate_dim]
    pub ffn_up: Vec<f32>,
    /// FFN gate projection [intermediate_dim] (for SwiGLU)
    pub ffn_gate: Vec<f32>,
    /// FFN down projection [hidden_dim]
    pub ffn_down: Vec<f32>,
    /// Output logits [vocab_size]
    pub logits: Vec<f32>,
    // PAR-126: Q8K scratch buffers for VNNI-accelerated matmul
    /// Q8K scales for hidden-dim activations [hidden_dim/256]
    pub q8k_hidden_scales: Vec<f32>,
    /// Q8K quants for hidden-dim activations [hidden_dim]
    pub q8k_hidden_quants: Vec<i8>,
    /// Q8K scales for intermediate-dim activations [intermediate_dim/256]
    pub q8k_inter_scales: Vec<f32>,
    /// Q8K quants for intermediate-dim activations [intermediate_dim]
    pub q8k_inter_quants: Vec<i8>,
}

impl InferenceScratchBuffer {
    /// Create scratch buffer from model config
    ///
    /// Pre-allocates all buffers to their maximum required size.
    /// Total memory: ~2.5MB for TinyLlama-1.1B, ~10MB for 7B models.
    #[must_use]
    pub fn from_config(config: &GGUFConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let intermediate_dim = config.intermediate_dim;
        let vocab_size = config.vocab_size;
        let qkv_dim = hidden_dim * 3; // Max for fused QKV

        // PAR-126: Q8K uses 256-element super-blocks for VNNI path
        const QK_K: usize = 256;
        let q8k_hidden_padded = hidden_dim.div_ceil(QK_K) * QK_K;
        let q8k_inter_padded = intermediate_dim.div_ceil(QK_K) * QK_K;

        Self {
            hidden: vec![0.0; hidden_dim],
            normed: vec![0.0; hidden_dim],
            qkv: vec![0.0; qkv_dim],
            q: vec![0.0; hidden_dim],
            k: vec![0.0; hidden_dim],
            v: vec![0.0; hidden_dim],
            attn_out: vec![0.0; hidden_dim],
            attn_proj: vec![0.0; hidden_dim],
            ffn_up: vec![0.0; intermediate_dim],
            ffn_gate: vec![0.0; intermediate_dim],
            ffn_down: vec![0.0; hidden_dim],
            logits: vec![0.0; vocab_size],
            q8k_hidden_scales: vec![0.0f32; q8k_hidden_padded / QK_K],
            q8k_hidden_quants: vec![0i8; q8k_hidden_padded],
            q8k_inter_scales: vec![0.0f32; q8k_inter_padded / QK_K],
            q8k_inter_quants: vec![0i8; q8k_inter_padded],
        }
    }

    /// Reset all buffers to zero for a new forward pass
    #[inline]
    pub fn reset(&mut self) {
        self.hidden.iter_mut().for_each(|x| *x = 0.0);
        self.normed.iter_mut().for_each(|x| *x = 0.0);
    }
}

// ============================================================================
// OwnedInferenceScratchBuffer
// ============================================================================

/// Pre-allocated scratch buffers for OwnedQuantizedModel forward passes
///
/// Eliminates per-token allocations by reusing buffers across forward passes.
/// For Qwen2.5-0.5B with intermediate_dim=4864, this saves ~40KB per token.
///
/// PAR-126: Added Q8K scratch buffers for fused Q4K×Q8K matmul path.
#[derive(Debug)]
pub struct OwnedInferenceScratchBuffer {
    /// QKV output buffer [hidden_dim + 2*kv_dim]
    pub qkv: Vec<f32>,
    /// Attention output buffer [hidden_dim]
    pub attn_out: Vec<f32>,
    /// FFN up projection buffer [intermediate_dim]
    pub ffn_up: Vec<f32>,
    /// FFN gate projection buffer [intermediate_dim]
    pub ffn_gate: Vec<f32>,
    /// FFN down output buffer [hidden_dim]
    pub ffn_down: Vec<f32>,
    /// Expanded V buffer for first token GQA [hidden_dim]
    pub expanded_v: Vec<f32>,
    /// Logits buffer [vocab_size]
    pub logits: Vec<f32>,
    /// Q8 quantization scales scratch [num_blocks]
    pub q8_scales: Vec<f32>,
    /// Q8 quantization values scratch [num_blocks * 32]
    pub q8_quants: Vec<i8>,
    // PAR-126: Q8K scratch buffers for VNNI-accelerated matmul
    /// Q8K scales for hidden-dim activations [hidden_dim/256]
    pub q8k_hidden_scales: Vec<f32>,
    /// Q8K quants for hidden-dim activations [hidden_dim]
    pub q8k_hidden_quants: Vec<i8>,
    /// Q8K scales for intermediate-dim activations [intermediate_dim/256]
    pub q8k_inter_scales: Vec<f32>,
    /// Q8K quants for intermediate-dim activations [intermediate_dim]
    pub q8k_inter_quants: Vec<i8>,
}

impl OwnedInferenceScratchBuffer {
    /// Create scratch buffer from model config
    #[must_use]
    pub fn from_config(config: &GGUFConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = hidden_dim / config.num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_dim = hidden_dim + 2 * kv_dim;
        let intermediate_dim = hidden_dim * 6; // Conservative estimate
        let num_blocks = hidden_dim.div_ceil(32);

        const QK_K: usize = 256;
        let q8k_hidden_padded = hidden_dim.div_ceil(QK_K) * QK_K;
        let q8k_inter_padded = intermediate_dim.div_ceil(QK_K) * QK_K;

        Self {
            qkv: vec![0.0f32; qkv_dim],
            attn_out: vec![0.0f32; hidden_dim],
            ffn_up: vec![0.0f32; intermediate_dim],
            ffn_gate: vec![0.0f32; intermediate_dim],
            ffn_down: vec![0.0f32; hidden_dim],
            expanded_v: vec![0.0f32; hidden_dim],
            logits: vec![0.0f32; config.vocab_size],
            q8_scales: vec![0.0f32; num_blocks],
            q8_quants: vec![0i8; num_blocks * 32],
            q8k_hidden_scales: vec![0.0f32; q8k_hidden_padded / QK_K],
            q8k_hidden_quants: vec![0i8; q8k_hidden_padded],
            q8k_inter_scales: vec![0.0f32; q8k_inter_padded / QK_K],
            q8k_inter_quants: vec![0i8; q8k_inter_padded],
        }
    }

    /// Reset all buffers (clear without deallocating)
    pub fn reset(&mut self) {
        self.qkv.clear();
        self.attn_out.clear();
        self.ffn_up.clear();
        self.ffn_gate.clear();
        self.ffn_down.clear();
        self.expanded_v.clear();
        self.logits.clear();
        self.q8_scales.clear();
        self.q8_quants.clear();
        self.q8k_hidden_scales.clear();
        self.q8k_hidden_quants.clear();
        self.q8k_inter_scales.clear();
        self.q8k_inter_quants.clear();
    }
}

// ============================================================================
// ContiguousKVCache (PARITY-005)
// ============================================================================

/// Contiguous KV cache with 64-byte cache line alignment (PARITY-005)
///
/// This cache uses a single contiguous allocation for all K and V data,
/// aligned to 64-byte cache lines for optimal L2 cache performance.
///
/// ## Memory Layout
///
/// ```text
/// K cache: [layer_0][layer_1]...[layer_n] (all contiguous)
/// V cache: [layer_0][layer_1]...[layer_n] (all contiguous)
///
/// Each layer: [pos_0][pos_1]...[pos_max_seq] where each pos is [hidden_dim]
/// ```
#[derive(Debug)]
pub struct ContiguousKVCache {
    num_layers: usize,
    hidden_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
    layer_stride: usize,
    k_data: Vec<f32>,
    v_data: Vec<f32>,
}

impl ContiguousKVCache {
    /// Create a new contiguous KV cache
    #[must_use]
    pub fn new(num_layers: usize, hidden_dim: usize, max_seq_len: usize) -> Self {
        let raw_layer_size = max_seq_len * hidden_dim;
        let layer_stride = Self::align_to_cache_line(raw_layer_size);
        let total_size = num_layers * layer_stride;

        Self {
            num_layers,
            hidden_dim,
            max_seq_len,
            seq_len: 0,
            layer_stride,
            k_data: vec![0.0f32; total_size],
            v_data: vec![0.0f32; total_size],
        }
    }

    #[inline]
    fn align_to_cache_line(size: usize) -> usize {
        let remainder = size % FLOATS_PER_CACHE_LINE;
        if remainder == 0 {
            size
        } else {
            size + FLOATS_PER_CACHE_LINE - remainder
        }
    }

    /// Create cache from model configuration
    #[must_use]
    pub fn from_config(config: &GGUFConfig, max_seq_len: usize) -> Self {
        Self::new(config.num_layers, config.hidden_dim, max_seq_len)
    }

    /// Check if this cache has contiguous layout
    #[must_use]
    pub const fn is_contiguous(&self) -> bool {
        true
    }

    /// Check if data is cache-line aligned
    #[must_use]
    pub fn is_cache_aligned(&self) -> bool {
        self.layer_stride.is_multiple_of(FLOATS_PER_CACHE_LINE)
    }

    /// Get the layer stride
    #[must_use]
    pub fn layer_stride(&self) -> usize {
        self.layer_stride
    }

    #[inline]
    fn layer_offset(&self, layer: usize) -> usize {
        layer * self.layer_stride
    }

    /// Append K and V vectors for a single position to a layer's cache
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        if layer >= self.num_layers || self.seq_len >= self.max_seq_len {
            return;
        }
        let start = self.layer_offset(layer) + self.seq_len * self.hidden_dim;
        let end = start + self.hidden_dim;
        if end <= self.k_data.len() {
            self.k_data[start..end].copy_from_slice(k);
            self.v_data[start..end].copy_from_slice(v);
        }
    }

    /// Advance the sequence position
    pub fn advance(&mut self) {
        if self.seq_len < self.max_seq_len {
            self.seq_len += 1;
        }
    }

    /// Get cached keys for a layer
    #[must_use]
    pub fn get_k(&self, layer: usize) -> &[f32] {
        if layer >= self.num_layers {
            return &[];
        }
        let start = self.layer_offset(layer);
        &self.k_data[start..start + self.seq_len * self.hidden_dim]
    }

    /// Get cached values for a layer
    #[must_use]
    pub fn get_v(&self, layer: usize) -> &[f32] {
        if layer >= self.num_layers {
            return &[];
        }
        let start = self.layer_offset(layer);
        &self.v_data[start..start + self.seq_len * self.hidden_dim]
    }

    /// Get mutable cached keys for a layer
    pub fn get_k_mut(&mut self, layer: usize) -> &mut [f32] {
        if layer >= self.num_layers {
            return &mut [];
        }
        let start = self.layer_offset(layer);
        let len = self.seq_len * self.hidden_dim;
        &mut self.k_data[start..start + len]
    }

    /// Get mutable cached values for a layer
    pub fn get_v_mut(&mut self, layer: usize) -> &mut [f32] {
        if layer >= self.num_layers {
            return &mut [];
        }
        let start = self.layer_offset(layer);
        let len = self.seq_len * self.hidden_dim;
        &mut self.v_data[start..start + len]
    }

    /// Current sequence length
    #[must_use]
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Reset cache for new generation
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }

    /// Reset cache and zero all data
    pub fn reset_and_zero(&mut self) {
        self.seq_len = 0;
        self.k_data.fill(0.0);
        self.v_data.fill(0.0);
    }

    /// Get maximum sequence length
    #[must_use]
    pub fn max_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get total memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        (self.k_data.len() + self.v_data.len()) * std::mem::size_of::<f32>()
    }

    /// Prefetch K cache for a layer
    #[inline]
    pub fn prefetch_k(&self, layer: usize) {
        if layer < self.num_layers {
            let _ = self.k_data.get(self.layer_offset(layer));
        }
    }

    /// Prefetch V cache for a layer
    #[inline]
    pub fn prefetch_v(&self, layer: usize) {
        if layer < self.num_layers {
            let _ = self.v_data.get(self.layer_offset(layer));
        }
    }
}

// ============================================================================
// DispatchMetrics (IMP-123)
// ============================================================================

/// Thread-safe metrics for tracking CPU vs GPU dispatch decisions
///
/// Tracks dispatch counts and latency histograms for performance analysis.
#[derive(Debug)]
pub struct DispatchMetrics {
    cpu_dispatches: std::sync::atomic::AtomicUsize,
    gpu_dispatches: std::sync::atomic::AtomicUsize,
    cpu_latency_count: std::sync::atomic::AtomicUsize,
    cpu_latency_sum_us: std::sync::atomic::AtomicU64,
    gpu_latency_count: std::sync::atomic::AtomicUsize,
    gpu_latency_sum_us: std::sync::atomic::AtomicU64,
    cpu_latency_buckets: [std::sync::atomic::AtomicUsize; 5],
    gpu_latency_buckets: [std::sync::atomic::AtomicUsize; 5],
    cpu_latency_min_us: std::sync::atomic::AtomicU64,
    cpu_latency_max_us: std::sync::atomic::AtomicU64,
    gpu_latency_min_us: std::sync::atomic::AtomicU64,
    gpu_latency_max_us: std::sync::atomic::AtomicU64,
    cpu_latency_sum_sq_us: std::sync::atomic::AtomicU64,
    gpu_latency_sum_sq_us: std::sync::atomic::AtomicU64,
    start_time_ms: std::sync::atomic::AtomicU64,
}

impl DispatchMetrics {
    /// Histogram bucket boundaries in microseconds
    pub const BUCKET_BOUNDARIES: [u64; 4] = [100, 500, 1000, 5000];

    /// Create new metrics tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            cpu_dispatches: std::sync::atomic::AtomicUsize::new(0),
            gpu_dispatches: std::sync::atomic::AtomicUsize::new(0),
            cpu_latency_count: std::sync::atomic::AtomicUsize::new(0),
            cpu_latency_sum_us: std::sync::atomic::AtomicU64::new(0),
            gpu_latency_count: std::sync::atomic::AtomicUsize::new(0),
            gpu_latency_sum_us: std::sync::atomic::AtomicU64::new(0),
            cpu_latency_buckets: [
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
            ],
            gpu_latency_buckets: [
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
            ],
            cpu_latency_min_us: std::sync::atomic::AtomicU64::new(u64::MAX),
            cpu_latency_max_us: std::sync::atomic::AtomicU64::new(0),
            gpu_latency_min_us: std::sync::atomic::AtomicU64::new(u64::MAX),
            gpu_latency_max_us: std::sync::atomic::AtomicU64::new(0),
            cpu_latency_sum_sq_us: std::sync::atomic::AtomicU64::new(0),
            gpu_latency_sum_sq_us: std::sync::atomic::AtomicU64::new(0),
            start_time_ms: std::sync::atomic::AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
            ),
        }
    }

    fn bucket_index(latency_us: u64) -> usize {
        for (i, &boundary) in Self::BUCKET_BOUNDARIES.iter().enumerate() {
            if latency_us < boundary {
                return i;
            }
        }
        4
    }

    /// Record a CPU dispatch
    pub fn record_cpu_dispatch(&self) {
        self.cpu_dispatches
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record a GPU dispatch
    pub fn record_gpu_dispatch(&self) {
        self.gpu_dispatches
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record CPU dispatch latency
    pub fn record_cpu_latency(&self, latency: std::time::Duration) {
        let latency_us = latency.as_micros() as u64;
        self.cpu_latency_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_sum_us
            .fetch_add(latency_us, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_buckets[Self::bucket_index(latency_us)]
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_min_us
            .fetch_min(latency_us, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_max_us
            .fetch_max(latency_us, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_sum_sq_us.fetch_add(
            latency_us * latency_us,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Record GPU dispatch latency
    pub fn record_gpu_latency(&self, latency: std::time::Duration) {
        let latency_us = latency.as_micros() as u64;
        self.gpu_latency_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_sum_us
            .fetch_add(latency_us, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_buckets[Self::bucket_index(latency_us)]
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_min_us
            .fetch_min(latency_us, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_max_us
            .fetch_max(latency_us, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_sum_sq_us.fetch_add(
            latency_us * latency_us,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Get CPU dispatch count
    #[must_use]
    pub fn cpu_dispatches(&self) -> usize {
        self.cpu_dispatches
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get GPU dispatch count
    #[must_use]
    pub fn gpu_dispatches(&self) -> usize {
        self.gpu_dispatches
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get total dispatches
    #[must_use]
    pub fn total_dispatches(&self) -> usize {
        self.cpu_dispatches() + self.gpu_dispatches()
    }

    /// Get GPU dispatch ratio
    #[must_use]
    pub fn gpu_ratio(&self) -> f64 {
        let total = self.total_dispatches();
        if total == 0 {
            0.0
        } else {
            self.gpu_dispatches() as f64 / total as f64
        }
    }

    /// Get CPU latency count
    #[must_use]
    pub fn cpu_latency_count(&self) -> usize {
        self.cpu_latency_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get GPU latency count
    #[must_use]
    pub fn gpu_latency_count(&self) -> usize {
        self.gpu_latency_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get mean CPU latency in microseconds
    #[must_use]
    pub fn cpu_latency_mean_us(&self) -> f64 {
        let count = self.cpu_latency_count();
        if count == 0 {
            0.0
        } else {
            self.cpu_latency_sum_us
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / count as f64
        }
    }

    /// Get mean GPU latency in microseconds
    #[must_use]
    pub fn gpu_latency_mean_us(&self) -> f64 {
        let count = self.gpu_latency_count();
        if count == 0 {
            0.0
        } else {
            self.gpu_latency_sum_us
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / count as f64
        }
    }

    /// Get CPU latency sum
    #[must_use]
    pub fn cpu_latency_sum_us(&self) -> u64 {
        self.cpu_latency_sum_us
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get GPU latency sum
    #[must_use]
    pub fn gpu_latency_sum_us(&self) -> u64 {
        self.gpu_latency_sum_us
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get CPU latency min
    #[must_use]
    pub fn cpu_latency_min_us(&self) -> u64 {
        if self.cpu_latency_count() == 0 {
            0
        } else {
            self.cpu_latency_min_us
                .load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    /// Get CPU latency max
    #[must_use]
    pub fn cpu_latency_max_us(&self) -> u64 {
        self.cpu_latency_max_us
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get GPU latency min
    #[must_use]
    pub fn gpu_latency_min_us(&self) -> u64 {
        if self.gpu_latency_count() == 0 {
            0
        } else {
            self.gpu_latency_min_us
                .load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    /// Get GPU latency max
    #[must_use]
    pub fn gpu_latency_max_us(&self) -> u64 {
        self.gpu_latency_max_us
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get CPU latency variance
    #[must_use]
    pub fn cpu_latency_variance_us(&self) -> f64 {
        let count = self.cpu_latency_count();
        if count < 2 {
            return 0.0;
        }
        let sum = self
            .cpu_latency_sum_us
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let sum_sq = self
            .cpu_latency_sum_sq_us
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let n = count as f64;
        (sum_sq / n) - (sum / n).powi(2)
    }

    /// Get CPU latency stddev
    #[must_use]
    pub fn cpu_latency_stddev_us(&self) -> f64 {
        self.cpu_latency_variance_us().sqrt()
    }

    /// Get GPU latency variance
    #[must_use]
    pub fn gpu_latency_variance_us(&self) -> f64 {
        let count = self.gpu_latency_count();
        if count < 2 {
            return 0.0;
        }
        let sum = self
            .gpu_latency_sum_us
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let sum_sq = self
            .gpu_latency_sum_sq_us
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let n = count as f64;
        (sum_sq / n) - (sum / n).powi(2)
    }

    /// Get GPU latency stddev
    #[must_use]
    pub fn gpu_latency_stddev_us(&self) -> f64 {
        self.gpu_latency_variance_us().sqrt()
    }

    /// Get CPU latency histogram buckets
    #[must_use]
    pub fn cpu_latency_buckets(&self) -> [usize; 5] {
        [
            self.cpu_latency_buckets[0].load(std::sync::atomic::Ordering::Relaxed),
            self.cpu_latency_buckets[1].load(std::sync::atomic::Ordering::Relaxed),
            self.cpu_latency_buckets[2].load(std::sync::atomic::Ordering::Relaxed),
            self.cpu_latency_buckets[3].load(std::sync::atomic::Ordering::Relaxed),
            self.cpu_latency_buckets[4].load(std::sync::atomic::Ordering::Relaxed),
        ]
    }

    /// Get GPU latency histogram buckets
    #[must_use]
    pub fn gpu_latency_buckets(&self) -> [usize; 5] {
        [
            self.gpu_latency_buckets[0].load(std::sync::atomic::Ordering::Relaxed),
            self.gpu_latency_buckets[1].load(std::sync::atomic::Ordering::Relaxed),
            self.gpu_latency_buckets[2].load(std::sync::atomic::Ordering::Relaxed),
            self.gpu_latency_buckets[3].load(std::sync::atomic::Ordering::Relaxed),
            self.gpu_latency_buckets[4].load(std::sync::atomic::Ordering::Relaxed),
        ]
    }

    fn estimate_percentile_from_buckets(buckets: &[usize; 5], percentile: f64) -> f64 {
        const BUCKET_UPPER_BOUNDS: [f64; 5] = [100.0, 500.0, 1000.0, 5000.0, 10000.0];
        const BUCKET_LOWER_BOUNDS: [f64; 5] = [0.0, 100.0, 500.0, 1000.0, 5000.0];
        let total: usize = buckets.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let target_rank = (percentile / 100.0) * total as f64;
        let mut cumulative: f64 = 0.0;
        for (i, &count) in buckets.iter().enumerate() {
            let prev_cumulative = cumulative;
            cumulative += count as f64;
            if cumulative >= target_rank {
                if count == 0 {
                    return BUCKET_LOWER_BOUNDS[i];
                }
                let fraction = (target_rank - prev_cumulative) / count as f64;
                return BUCKET_LOWER_BOUNDS[i]
                    + fraction * (BUCKET_UPPER_BOUNDS[i] - BUCKET_LOWER_BOUNDS[i]);
            }
        }
        BUCKET_UPPER_BOUNDS[4]
    }

    /// Get CPU p50 latency
    #[must_use]
    pub fn cpu_latency_p50_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.cpu_latency_buckets(), 50.0)
    }

    /// Get CPU p95 latency
    #[must_use]
    pub fn cpu_latency_p95_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.cpu_latency_buckets(), 95.0)
    }

    /// Get CPU p99 latency
    #[must_use]
    pub fn cpu_latency_p99_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.cpu_latency_buckets(), 99.0)
    }

    /// Get GPU p50 latency
    #[must_use]
    pub fn gpu_latency_p50_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.gpu_latency_buckets(), 50.0)
    }

    /// Get GPU p95 latency
    #[must_use]
    pub fn gpu_latency_p95_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.gpu_latency_buckets(), 95.0)
    }

    /// Get GPU p99 latency
    #[must_use]
    pub fn gpu_latency_p99_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.gpu_latency_buckets(), 99.0)
    }

    /// Get bucket boundaries as strings
    #[must_use]
    pub fn bucket_boundaries_us(&self) -> Vec<String> {
        vec![
            format!("0-{}", Self::BUCKET_BOUNDARIES[0]),
            format!(
                "{}-{}",
                Self::BUCKET_BOUNDARIES[0],
                Self::BUCKET_BOUNDARIES[1]
            ),
            format!(
                "{}-{}",
                Self::BUCKET_BOUNDARIES[1],
                Self::BUCKET_BOUNDARIES[2]
            ),
            format!(
                "{}-{}",
                Self::BUCKET_BOUNDARIES[2],
                Self::BUCKET_BOUNDARIES[3]
            ),
            format!("{}+", Self::BUCKET_BOUNDARIES[3]),
        ]
    }

    /// Get start time
    #[must_use]
    pub fn start_time_ms(&self) -> u64 {
        self.start_time_ms
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get elapsed seconds
    #[must_use]
    pub fn elapsed_seconds(&self) -> f64 {
        let start = self.start_time_ms();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        (now.saturating_sub(start)) as f64 / 1000.0
    }

    /// Get throughput
    #[must_use]
    pub fn throughput_rps(&self) -> f64 {
        let elapsed = self.elapsed_seconds();
        if elapsed < 0.001 {
            0.0
        } else {
            self.total_dispatches() as f64 / elapsed
        }
    }

    /// Get CPU latency CV
    #[must_use]
    pub fn cpu_latency_cv(&self) -> f64 {
        let mean = self.cpu_latency_mean_us();
        if mean < 0.001 {
            0.0
        } else {
            (self.cpu_latency_stddev_us() / mean) * 100.0
        }
    }

    /// Get GPU latency CV
    #[must_use]
    pub fn gpu_latency_cv(&self) -> f64 {
        let mean = self.gpu_latency_mean_us();
        if mean < 0.001 {
            0.0
        } else {
            (self.gpu_latency_stddev_us() / mean) * 100.0
        }
    }

    /// Get CPU/GPU speedup
    #[must_use]
    pub fn cpu_gpu_speedup(&self) -> f64 {
        let gpu_mean = self.gpu_latency_mean_us();
        if gpu_mean < 0.001 {
            0.0
        } else {
            self.cpu_latency_mean_us() / gpu_mean
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.cpu_dispatches
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.gpu_dispatches
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_sum_us
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_sum_us
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_min_us
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_max_us
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_min_us
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_max_us
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_sum_sq_us
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_sum_sq_us
            .store(0, std::sync::atomic::Ordering::Relaxed);
        for bucket in &self.cpu_latency_buckets {
            bucket.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        for bucket in &self.gpu_latency_buckets {
            bucket.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.start_time_ms
            .store(now, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Default for DispatchMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> GGUFConfig {
        GGUFConfig {
            architecture: "test".to_string(),
            hidden_dim: 64,
            intermediate_dim: 128,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 2,
            vocab_size: 100,
            rope_theta: 10000.0,
            context_length: 512,
            eps: 1e-5,
            rope_type: 0,
        }
    }

    // ============================================================================
    // InferenceScratchBuffer tests
    // ============================================================================

    #[test]
    fn test_inference_scratch_buffer_from_config() {
        let config = test_config();
        let buf = InferenceScratchBuffer::from_config(&config);

        assert_eq!(buf.hidden.len(), 64);
        assert_eq!(buf.normed.len(), 64);
        assert_eq!(buf.qkv.len(), 64 * 3); // hidden_dim * 3
        assert_eq!(buf.logits.len(), 100); // vocab_size
        assert_eq!(buf.ffn_up.len(), 128); // intermediate_dim
    }

    #[test]
    fn test_inference_scratch_buffer_reset() {
        let config = test_config();
        let mut buf = InferenceScratchBuffer::from_config(&config);

        // Set some values
        buf.hidden[0] = 1.0;
        buf.normed[0] = 2.0;

        buf.reset();

        assert!((buf.hidden[0] - 0.0).abs() < f32::EPSILON);
        assert!((buf.normed[0] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_scratch_buffer_q8k_buffers() {
        let config = test_config();
        let buf = InferenceScratchBuffer::from_config(&config);

        // Q8K uses 256-element super-blocks
        // hidden_dim=64 -> ceil(64/256)=1 super-block -> 1 scale
        assert!(buf.q8k_hidden_scales.len() >= 1);
        assert!(buf.q8k_hidden_quants.len() >= 64);
    }

    #[test]
    fn test_inference_scratch_buffer_debug() {
        let config = test_config();
        let buf = InferenceScratchBuffer::from_config(&config);
        let debug = format!("{:?}", buf);
        assert!(debug.contains("InferenceScratchBuffer"));
    }

    // ============================================================================
    // OwnedInferenceScratchBuffer tests
    // ============================================================================

    #[test]
    fn test_owned_scratch_buffer_from_config() {
        let config = test_config();
        let buf = OwnedInferenceScratchBuffer::from_config(&config);

        // qkv = hidden_dim + 2 * kv_dim
        // kv_dim = num_kv_heads * head_dim = 4 * 16 = 64
        // qkv = 64 + 2 * 64 = 192
        let head_dim = 64 / 4; // 16
        let kv_dim = 4 * head_dim;
        let expected_qkv = 64 + 2 * kv_dim;
        assert_eq!(buf.qkv.len(), expected_qkv);
        assert_eq!(buf.attn_out.len(), 64);
        assert_eq!(buf.logits.len(), 100);
    }

    #[test]
    fn test_owned_scratch_buffer_reset() {
        let config = test_config();
        let mut buf = OwnedInferenceScratchBuffer::from_config(&config);

        // Add some data
        buf.qkv.push(1.0);
        buf.attn_out.push(2.0);

        buf.reset();

        // All vectors should be cleared
        assert!(buf.qkv.is_empty());
        assert!(buf.attn_out.is_empty());
        assert!(buf.ffn_up.is_empty());
        assert!(buf.logits.is_empty());
    }

    #[test]
    fn test_owned_scratch_buffer_debug() {
        let config = test_config();
        let buf = OwnedInferenceScratchBuffer::from_config(&config);
        let debug = format!("{:?}", buf);
        assert!(debug.contains("OwnedInferenceScratchBuffer"));
    }

    // ============================================================================
    // ContiguousKVCache tests
    // ============================================================================

    #[test]
    fn test_contiguous_kv_cache_new() {
        let cache = ContiguousKVCache::new(2, 64, 512);
        assert!(cache.is_contiguous());
        assert!(cache.is_cache_aligned());
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.max_len(), 512);
    }

    #[test]
    fn test_contiguous_kv_cache_from_config() {
        let config = test_config();
        let cache = ContiguousKVCache::from_config(&config, 256);
        assert!(cache.is_contiguous());
        assert_eq!(cache.max_len(), 256);
    }

    #[test]
    fn test_contiguous_kv_cache_append_and_advance() {
        let mut cache = ContiguousKVCache::new(2, 4, 10);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        cache.append(0, &k, &v);
        cache.advance();

        assert_eq!(cache.len(), 1);
        let cached_k = cache.get_k(0);
        assert_eq!(cached_k.len(), 4);
        assert!((cached_k[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_contiguous_kv_cache_get_k_v() {
        let mut cache = ContiguousKVCache::new(2, 4, 10);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        cache.append(0, &k, &v);
        cache.advance();

        let cached_k = cache.get_k(0);
        let cached_v = cache.get_v(0);
        assert_eq!(cached_k.len(), 4);
        assert_eq!(cached_v.len(), 4);
        assert!((cached_v[0] - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_contiguous_kv_cache_get_k_v_mut() {
        let mut cache = ContiguousKVCache::new(1, 4, 10);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        cache.append(0, &k, &v);
        cache.advance();

        let k_mut = cache.get_k_mut(0);
        k_mut[0] = 99.0;

        let cached_k = cache.get_k(0);
        assert!((cached_k[0] - 99.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_contiguous_kv_cache_reset() {
        let mut cache = ContiguousKVCache::new(1, 4, 10);
        cache.append(0, &[1.0; 4], &[2.0; 4]);
        cache.advance();
        assert_eq!(cache.len(), 1);

        cache.reset();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_contiguous_kv_cache_reset_and_zero() {
        let mut cache = ContiguousKVCache::new(1, 4, 10);
        cache.append(0, &[1.0; 4], &[2.0; 4]);
        cache.advance();

        cache.reset_and_zero();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_contiguous_kv_cache_memory_bytes() {
        let cache = ContiguousKVCache::new(2, 64, 128);
        let mem = cache.memory_bytes();
        // At least 2 layers * 128 * 64 * 4 bytes * 2 (k+v) = 131072
        assert!(mem >= 131072);
    }

    #[test]
    fn test_contiguous_kv_cache_layer_stride() {
        let cache = ContiguousKVCache::new(2, 64, 128);
        let stride = cache.layer_stride();
        // Should be cache-line aligned
        assert!(stride % FLOATS_PER_CACHE_LINE == 0);
    }

    #[test]
    fn test_contiguous_kv_cache_invalid_layer() {
        let cache = ContiguousKVCache::new(2, 4, 10);
        // Invalid layer returns empty slice
        assert!(cache.get_k(99).is_empty());
        assert!(cache.get_v(99).is_empty());
    }

    #[test]
    fn test_contiguous_kv_cache_prefetch() {
        let cache = ContiguousKVCache::new(2, 64, 128);
        // Prefetch should not panic
        cache.prefetch_k(0);
        cache.prefetch_v(0);
        cache.prefetch_k(99); // Invalid layer should be safe
    }

    // ============================================================================
    // DispatchMetrics tests
    // ============================================================================

    #[test]
    fn test_dispatch_metrics_new() {
        let metrics = DispatchMetrics::new();
        assert_eq!(metrics.cpu_dispatches(), 0);
        assert_eq!(metrics.gpu_dispatches(), 0);
        assert_eq!(metrics.total_dispatches(), 0);
    }

    #[test]
    fn test_dispatch_metrics_default() {
        let metrics = DispatchMetrics::default();
        assert_eq!(metrics.cpu_dispatches(), 0);
    }

    #[test]
    fn test_dispatch_metrics_record_cpu() {
        let metrics = DispatchMetrics::new();
        metrics.record_cpu_dispatch();
        metrics.record_cpu_dispatch();
        assert_eq!(metrics.cpu_dispatches(), 2);
        assert_eq!(metrics.gpu_dispatches(), 0);
    }

    #[test]
    fn test_dispatch_metrics_record_gpu() {
        let metrics = DispatchMetrics::new();
        metrics.record_gpu_dispatch();
        assert_eq!(metrics.gpu_dispatches(), 1);
    }

    #[test]
    fn test_dispatch_metrics_gpu_ratio() {
        let metrics = DispatchMetrics::new();
        assert!((metrics.gpu_ratio() - 0.0).abs() < f64::EPSILON);

        metrics.record_cpu_dispatch();
        metrics.record_gpu_dispatch();
        assert!((metrics.gpu_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dispatch_metrics_cpu_latency() {
        let metrics = DispatchMetrics::new();
        metrics.record_cpu_latency(std::time::Duration::from_micros(100));
        metrics.record_cpu_latency(std::time::Duration::from_micros(200));

        assert_eq!(metrics.cpu_latency_count(), 2);
        assert!((metrics.cpu_latency_mean_us() - 150.0).abs() < 1.0);
        assert_eq!(metrics.cpu_latency_min_us(), 100);
        assert_eq!(metrics.cpu_latency_max_us(), 200);
    }

    #[test]
    fn test_dispatch_metrics_gpu_latency() {
        let metrics = DispatchMetrics::new();
        metrics.record_gpu_latency(std::time::Duration::from_micros(50));

        assert_eq!(metrics.gpu_latency_count(), 1);
        assert!((metrics.gpu_latency_mean_us() - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_dispatch_metrics_buckets() {
        let metrics = DispatchMetrics::new();
        // <100us -> bucket 0
        metrics.record_cpu_latency(std::time::Duration::from_micros(50));
        // 100-500us -> bucket 1
        metrics.record_cpu_latency(std::time::Duration::from_micros(200));
        // 500-1000us -> bucket 2
        metrics.record_cpu_latency(std::time::Duration::from_micros(700));

        let buckets = metrics.cpu_latency_buckets();
        assert_eq!(buckets[0], 1);
        assert_eq!(buckets[1], 1);
        assert_eq!(buckets[2], 1);
    }

    #[test]
    fn test_dispatch_metrics_variance_and_stddev() {
        let metrics = DispatchMetrics::new();
        // With only 1 sample, variance should be 0
        metrics.record_cpu_latency(std::time::Duration::from_micros(100));
        assert!((metrics.cpu_latency_variance_us() - 0.0).abs() < 0.001);

        // With 2 identical samples, variance should be 0
        metrics.record_cpu_latency(std::time::Duration::from_micros(100));
        assert!(metrics.cpu_latency_variance_us().is_finite());
    }

    #[test]
    fn test_dispatch_metrics_percentiles() {
        let metrics = DispatchMetrics::new();
        // All in first bucket
        for _ in 0..100 {
            metrics.record_cpu_latency(std::time::Duration::from_micros(50));
        }

        let p50 = metrics.cpu_latency_p50_us();
        let p95 = metrics.cpu_latency_p95_us();
        let p99 = metrics.cpu_latency_p99_us();
        assert!(p50 >= 0.0);
        assert!(p95 >= p50);
        assert!(p99 >= p95);
    }

    #[test]
    fn test_dispatch_metrics_bucket_boundaries() {
        let metrics = DispatchMetrics::new();
        let boundaries = metrics.bucket_boundaries_us();
        assert_eq!(boundaries.len(), 5);
        assert!(boundaries[0].contains("0-"));
    }

    #[test]
    fn test_dispatch_metrics_reset() {
        let metrics = DispatchMetrics::new();
        metrics.record_cpu_dispatch();
        metrics.record_gpu_dispatch();
        metrics.record_cpu_latency(std::time::Duration::from_micros(100));

        metrics.reset();

        assert_eq!(metrics.cpu_dispatches(), 0);
        assert_eq!(metrics.gpu_dispatches(), 0);
        assert_eq!(metrics.cpu_latency_count(), 0);
    }

    #[test]
    fn test_dispatch_metrics_speedup() {
        let metrics = DispatchMetrics::new();
        metrics.record_cpu_latency(std::time::Duration::from_micros(1000));
        metrics.record_gpu_latency(std::time::Duration::from_micros(100));

        let speedup = metrics.cpu_gpu_speedup();
        assert!((speedup - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_dispatch_metrics_cv() {
        let metrics = DispatchMetrics::new();
        // No samples -> CV = 0
        assert!((metrics.cpu_latency_cv() - 0.0).abs() < 0.001);
        assert!((metrics.gpu_latency_cv() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_dispatch_metrics_elapsed_and_throughput() {
        let metrics = DispatchMetrics::new();
        metrics.record_cpu_dispatch();

        // Elapsed should be very small
        let elapsed = metrics.elapsed_seconds();
        assert!(elapsed >= 0.0);

        // Throughput calculation
        let _throughput = metrics.throughput_rps();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_dispatch_metrics_debug() {
        let metrics = DispatchMetrics::new();
        let debug = format!("{:?}", metrics);
        assert!(debug.contains("DispatchMetrics"));
    }
}
