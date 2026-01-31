//! CUDA backend for kernel configuration and PTX generation
//!
//! This module provides the `CudaBackend` struct for configuring CUDA kernels
//! and generating PTX code using trueno-gpu primitives.
//!
//! # Features
//!
//! - **IMP-312**: Q4_K quantized GEMM kernel (dequant + matmul fusion)
//! - **IMP-313**: FlashAttention-style tiled attention
//! - **IMP-314**: Paged KV cache memory management
//! - **IMP-315**: CUDA graph capture helpers
//!
//! # Example
//!
//! ```rust,ignore
//! use realizar::gguf::CudaBackend;
//!
//! let cuda = CudaBackend::new(1024, 1024, 4096, 64);
//! let ptx = cuda.q4k_gemm_ptx();  // Get PTX for Q4_K GEMM kernel
//! let attention_ptx = cuda.flash_attention_ptx(2048, 64, true);  // Causal attention
//! ```

use trueno_gpu::kernels::{AttentionKernel, Kernel, QuantizeKernel};

/// CUDA backend for kernel configuration and PTX generation
///
/// Provides dimension-aware kernel generation and launch configuration
/// for CUDA-accelerated inference operations.
///
/// # Example
///
/// ```rust,ignore
/// use realizar::gguf::CudaBackend;
///
/// let cuda = CudaBackend::new(1024, 1024, 4096, 64);
/// let ptx = cuda.q4k_gemm_ptx();  // Get PTX for Q4_K GEMM kernel
/// let attention_ptx = cuda.flash_attention_ptx(2048, 64, true);  // Causal attention
/// ```
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaBackend {
    /// Output rows (M) for GEMM operations
    pub m: u32,
    /// Output columns (N) for GEMM operations
    pub n: u32,
    /// Inner dimension (K) - must be divisible by Q4_K block size (32)
    pub k: u32,
    /// Head dimension for attention (typically 64 or 128)
    pub head_dim: u32,
    /// Number of attention heads
    pub num_heads: u32,
    /// Maximum sequence length for KV cache
    pub max_seq_len: u32,
    /// Cached PTX for Q4_K GEMM kernel (IMP-312)
    q4k_gemm_ptx_cache: std::cell::RefCell<Option<String>>,
    /// Cached PTX for FlashAttention kernel (IMP-313)
    flash_attention_ptx_cache: std::cell::RefCell<Option<String>>,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Create a new CUDA backend with specified dimensions
    ///
    /// # Arguments
    /// * `m` - Output rows for GEMM
    /// * `n` - Output columns for GEMM
    /// * `k` - Inner dimension (should be divisible by 32 for Q4_K)
    /// * `head_dim` - Head dimension for attention (typically 64)
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32, head_dim: u32) -> Self {
        Self {
            m,
            n,
            k,
            head_dim,
            num_heads: 32,     // Default for many models
            max_seq_len: 2048, // Default context length
            q4k_gemm_ptx_cache: std::cell::RefCell::new(None),
            flash_attention_ptx_cache: std::cell::RefCell::new(None),
        }
    }

    /// Set the number of attention heads
    #[must_use]
    pub const fn with_num_heads(mut self, num_heads: u32) -> Self {
        self.num_heads = num_heads;
        self
    }

    /// Set the maximum sequence length for KV cache
    #[must_use]
    pub const fn with_max_seq_len(mut self, max_seq_len: u32) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    // ========================================================================
    // IMP-312: CUDA Q4_K Dequant+Matmul Kernel
    // ========================================================================

    /// Generate PTX for Q4_K quantized GEMM kernel (IMP-312)
    ///
    /// The kernel fuses dequantization with matrix multiplication:
    /// - Dequantization: val = scale * quant + min (per Q4_K block)
    /// - Matrix multiply: C = A × dequant(B)
    ///
    /// # Performance
    /// - Uses warp shuffle for efficient reduction
    /// - Shared memory for dequantized tiles
    /// - Coalesced memory access patterns
    #[must_use]
    pub fn q4k_gemm_ptx(&self) -> String {
        // Check cache first
        if let Some(cached) = self.q4k_gemm_ptx_cache.borrow().as_ref() {
            return cached.clone();
        }

        // Generate PTX using trueno-gpu
        let kernel = QuantizeKernel::new(self.m, self.n, self.k);
        let ptx = kernel.emit_ptx();

        // Cache the result
        *self.q4k_gemm_ptx_cache.borrow_mut() = Some(ptx.clone());
        ptx
    }

    /// Get kernel name for Q4_K GEMM
    #[must_use]
    pub fn q4k_gemm_kernel_name(&self) -> &'static str {
        "q4k_gemm_fused"
    }

    /// Get number of Q4_K blocks per row (K / 32)
    #[must_use]
    pub const fn q4k_blocks_per_row(&self) -> u32 {
        self.k / 32
    }

    /// Estimate Q4_K weight memory size in bytes
    /// Each block: 2 bytes header (scale+min) + 16 bytes data = 18 bytes for 32 weights
    #[must_use]
    pub const fn q4k_weight_bytes(&self) -> usize {
        let blocks_per_row = self.k / 32;
        let bytes_per_row = blocks_per_row * 18;
        (self.n as usize) * (bytes_per_row as usize)
    }

    // ========================================================================
    // IMP-313: CUDA FlashAttention Kernel
    // ========================================================================

    /// Generate PTX for FlashAttention-style tiled attention (IMP-313)
    ///
    /// Implements IO-aware attention per Dao et al. [16]:
    /// - Never materializes the full N×N attention matrix
    /// - Online softmax with running max and sum
    /// - O(N × d) memory instead of O(N²)
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length (N)
    /// * `head_dim` - Head dimension (d)
    /// * `causal` - Enable causal masking for autoregressive models
    #[must_use]
    pub fn flash_attention_ptx(&self, seq_len: u32, head_dim: u32, causal: bool) -> String {
        let kernel = if causal {
            AttentionKernel::new(seq_len, head_dim).with_causal()
        } else {
            AttentionKernel::new(seq_len, head_dim)
        };
        kernel.emit_ptx()
    }

    /// Generate PTX for causal FlashAttention (cached version)
    #[must_use]
    pub fn flash_attention_causal_ptx(&self) -> String {
        // Check cache first
        if let Some(cached) = self.flash_attention_ptx_cache.borrow().as_ref() {
            return cached.clone();
        }

        // Generate causal attention PTX
        let ptx = self.flash_attention_ptx(self.max_seq_len, self.head_dim, true);

        // Cache the result
        *self.flash_attention_ptx_cache.borrow_mut() = Some(ptx.clone());
        ptx
    }

    /// Get kernel name for FlashAttention
    #[must_use]
    pub const fn flash_attention_kernel_name(&self, causal: bool) -> &'static str {
        if causal {
            "flash_attention_causal"
        } else {
            "flash_attention"
        }
    }

    /// Estimate shared memory size for FlashAttention (in bytes)
    /// Uses tiles of Q (B_r × d) and KV (B_c × d × 2)
    #[must_use]
    pub const fn flash_attention_smem_bytes(&self) -> usize {
        let tile_q = 64_u32;
        let tile_kv = 64_u32;
        let d = self.head_dim;
        // Q tile + K tile + V tile, all f32
        ((tile_q * d + tile_kv * d * 2) * 4) as usize
    }

    // ========================================================================
    // IMP-314: CUDA KV Cache with Paged Memory
    // ========================================================================

    /// Calculate KV cache memory size per layer in bytes
    ///
    /// KV cache stores Key and Value tensors for attention:
    /// - K: [num_heads, seq_len, head_dim] × sizeof(f32)
    /// - V: [num_heads, seq_len, head_dim] × sizeof(f32)
    #[must_use]
    pub const fn kv_cache_bytes_per_layer(&self) -> usize {
        let k_size = self.num_heads * self.max_seq_len * self.head_dim * 4;
        let v_size = self.num_heads * self.max_seq_len * self.head_dim * 4;
        (k_size + v_size) as usize
    }

    /// Calculate total KV cache memory for all layers
    #[must_use]
    pub const fn kv_cache_total_bytes(&self, num_layers: u32) -> usize {
        self.kv_cache_bytes_per_layer() * (num_layers as usize)
    }

    /// Get page size for paged KV cache (IMP-314)
    /// Default: 64 tokens per page to balance memory efficiency and fragmentation
    #[must_use]
    pub const fn kv_cache_page_tokens(&self) -> u32 {
        64
    }

    /// Calculate number of pages needed for given sequence length
    #[must_use]
    pub const fn kv_cache_pages_needed(&self, seq_len: u32) -> u32 {
        let page_size = self.kv_cache_page_tokens();
        seq_len.div_ceil(page_size)
    }

    // ========================================================================
    // IMP-315: CUDA Graph Capture Helpers
    // ========================================================================

    /// Get CUDA launch configuration for Q4_K GEMM kernel
    ///
    /// Returns (grid_dim, block_dim) tuple for kernel launch
    #[must_use]
    pub const fn q4k_gemm_launch_config(&self) -> ((u32, u32, u32), (u32, u32, u32)) {
        let tile_size = 32_u32;
        let grid_x = self.n.div_ceil(tile_size);
        let grid_y = self.m.div_ceil(tile_size);
        let grid = (grid_x, grid_y, 1);
        let block = (tile_size * tile_size, 1, 1);
        (grid, block)
    }

    /// Get CUDA launch configuration for FlashAttention kernel
    #[must_use]
    pub const fn flash_attention_launch_config(
        &self,
        seq_len: u32,
    ) -> ((u32, u32, u32), (u32, u32, u32)) {
        let tile_q = 64_u32;
        let num_q_blocks = seq_len.div_ceil(tile_q);
        let grid = (num_q_blocks, self.num_heads, 1);
        let block = (tile_q * self.head_dim, 1, 1);
        (grid, block)
    }

    /// Check if dimensions are valid for CUDA kernels
    #[must_use]
    pub const fn validate_dimensions(&self) -> bool {
        // K must be divisible by Q4_K block size (32)
        let k_valid = self.k.is_multiple_of(32);
        // Head dim should be power of 2 for efficient memory access
        let head_dim_valid = self.head_dim.is_power_of_two();
        // Dimensions must be non-zero
        let non_zero = self.m > 0 && self.n > 0 && self.k > 0 && self.head_dim > 0;
        k_valid && head_dim_valid && non_zero
    }

    /// Get PTX target SM version (default: sm_89 for Ada Lovelace/RTX 4090)
    #[must_use]
    pub const fn ptx_target(&self) -> &'static str {
        "sm_89"
    }

    /// Get PTX version (default: 8.0)
    #[must_use]
    pub const fn ptx_version(&self) -> (u32, u32) {
        (8, 0)
    }
}

// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    // =========================================================================
    // Constructor and Builder Tests
    // =========================================================================

    #[test]
    fn test_cuda_backend_new() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert_eq!(backend.m, 1);
        assert_eq!(backend.n, 4096);
        assert_eq!(backend.k, 4096);
        assert_eq!(backend.head_dim, 64);
        assert_eq!(backend.num_heads, 32); // default
        assert_eq!(backend.max_seq_len, 2048); // default
    }

    #[test]
    fn test_cuda_backend_with_num_heads() {
        let backend = CudaBackend::new(1, 4096, 4096, 64).with_num_heads(16);
        assert_eq!(backend.num_heads, 16);
    }

    #[test]
    fn test_cuda_backend_with_max_seq_len() {
        let backend = CudaBackend::new(1, 4096, 4096, 64).with_max_seq_len(4096);
        assert_eq!(backend.max_seq_len, 4096);
    }

    #[test]
    fn test_cuda_backend_builder_chain() {
        let backend = CudaBackend::new(1, 4096, 4096, 128)
            .with_num_heads(8)
            .with_max_seq_len(1024);
        assert_eq!(backend.num_heads, 8);
        assert_eq!(backend.max_seq_len, 1024);
        assert_eq!(backend.head_dim, 128);
    }

    #[test]
    fn test_cuda_backend_clone() {
        let backend = CudaBackend::new(2, 1024, 2048, 64);
        let cloned = backend.clone();
        assert_eq!(cloned.m, 2);
        assert_eq!(cloned.n, 1024);
        assert_eq!(cloned.k, 2048);
    }

    #[test]
    fn test_cuda_backend_debug() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("CudaBackend"));
        assert!(debug_str.contains("4096"));
    }

    // =========================================================================
    // Q4_K Kernel Tests
    // =========================================================================

    #[test]
    fn test_q4k_gemm_kernel_name() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert_eq!(backend.q4k_gemm_kernel_name(), "q4k_gemm_fused");
    }

    #[test]
    fn test_q4k_blocks_per_row() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert_eq!(backend.q4k_blocks_per_row(), 128); // 4096 / 32
    }

    #[test]
    fn test_q4k_blocks_per_row_small() {
        let backend = CudaBackend::new(1, 1024, 256, 64);
        assert_eq!(backend.q4k_blocks_per_row(), 8); // 256 / 32
    }

    #[test]
    fn test_q4k_weight_bytes() {
        let backend = CudaBackend::new(1, 1024, 1024, 64);
        // blocks_per_row = 1024 / 32 = 32
        // bytes_per_row = 32 * 18 = 576
        // total = 1024 * 576 = 589,824
        assert_eq!(backend.q4k_weight_bytes(), 589_824);
    }

    #[test]
    fn test_q4k_weight_bytes_large() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        // blocks_per_row = 4096 / 32 = 128
        // bytes_per_row = 128 * 18 = 2304
        // total = 4096 * 2304 = 9,437,184
        assert_eq!(backend.q4k_weight_bytes(), 9_437_184);
    }

    #[test]
    fn test_q4k_gemm_ptx_generation() {
        let backend = CudaBackend::new(1, 1024, 256, 64);
        let ptx = backend.q4k_gemm_ptx();
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".version"));
    }

    #[test]
    fn test_q4k_gemm_ptx_caching() {
        let backend = CudaBackend::new(1, 1024, 256, 64);
        let ptx1 = backend.q4k_gemm_ptx();
        let ptx2 = backend.q4k_gemm_ptx(); // Should use cache
        assert_eq!(ptx1, ptx2);
    }

    // =========================================================================
    // FlashAttention Tests
    // =========================================================================

    #[test]
    fn test_flash_attention_kernel_name_causal() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert_eq!(
            backend.flash_attention_kernel_name(true),
            "flash_attention_causal"
        );
    }

    #[test]
    fn test_flash_attention_kernel_name_non_causal() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert_eq!(
            backend.flash_attention_kernel_name(false),
            "flash_attention"
        );
    }

    #[test]
    fn test_flash_attention_smem_bytes() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        // tile_q = 64, tile_kv = 64, d = 64
        // smem = (64*64 + 64*64*2) * 4 = (4096 + 8192) * 4 = 49152
        assert_eq!(backend.flash_attention_smem_bytes(), 49152);
    }

    #[test]
    fn test_flash_attention_smem_bytes_large_head_dim() {
        let backend = CudaBackend::new(1, 4096, 4096, 128);
        // tile_q = 64, tile_kv = 64, d = 128
        // smem = (64*128 + 64*128*2) * 4 = (8192 + 16384) * 4 = 98304
        assert_eq!(backend.flash_attention_smem_bytes(), 98304);
    }

    #[test]
    fn test_flash_attention_ptx_generation() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        let ptx = backend.flash_attention_ptx(512, 64, true);
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".version"));
    }

    #[test]
    fn test_flash_attention_causal_ptx_caching() {
        let backend = CudaBackend::new(1, 4096, 4096, 64).with_max_seq_len(512);
        let ptx1 = backend.flash_attention_causal_ptx();
        let ptx2 = backend.flash_attention_causal_ptx(); // Should use cache
        assert_eq!(ptx1, ptx2);
    }

    // =========================================================================
    // KV Cache Tests
    // =========================================================================

    #[test]
    fn test_kv_cache_bytes_per_layer() {
        let backend = CudaBackend::new(1, 4096, 4096, 64)
            .with_num_heads(32)
            .with_max_seq_len(2048);
        // K: 32 * 2048 * 64 * 4 = 16,777,216
        // V: 32 * 2048 * 64 * 4 = 16,777,216
        // Total: 33,554,432
        assert_eq!(backend.kv_cache_bytes_per_layer(), 33_554_432);
    }

    #[test]
    fn test_kv_cache_total_bytes() {
        let backend = CudaBackend::new(1, 4096, 4096, 64)
            .with_num_heads(32)
            .with_max_seq_len(2048);
        let per_layer = backend.kv_cache_bytes_per_layer();
        assert_eq!(backend.kv_cache_total_bytes(22), per_layer * 22);
    }

    #[test]
    fn test_kv_cache_page_tokens() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert_eq!(backend.kv_cache_page_tokens(), 64);
    }

    #[test]
    fn test_kv_cache_pages_needed_exact() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        // 64 tokens = 1 page
        assert_eq!(backend.kv_cache_pages_needed(64), 1);
        // 128 tokens = 2 pages
        assert_eq!(backend.kv_cache_pages_needed(128), 2);
    }

    #[test]
    fn test_kv_cache_pages_needed_round_up() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        // 65 tokens = 2 pages (rounds up)
        assert_eq!(backend.kv_cache_pages_needed(65), 2);
        // 100 tokens = 2 pages
        assert_eq!(backend.kv_cache_pages_needed(100), 2);
        // 129 tokens = 3 pages
        assert_eq!(backend.kv_cache_pages_needed(129), 3);
    }

    #[test]
    fn test_kv_cache_pages_needed_large() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        // 2048 tokens = 32 pages
        assert_eq!(backend.kv_cache_pages_needed(2048), 32);
    }

    // =========================================================================
    // Launch Configuration Tests
    // =========================================================================

    #[test]
    fn test_q4k_gemm_launch_config() {
        let backend = CudaBackend::new(1, 1024, 4096, 64);
        let (grid, block) = backend.q4k_gemm_launch_config();
        // grid = (n/32, m/32, 1) = (1024/32, 1/32 rounded up, 1) = (32, 1, 1)
        assert_eq!(grid, (32, 1, 1));
        // block = (32*32, 1, 1) = (1024, 1, 1)
        assert_eq!(block, (1024, 1, 1));
    }

    #[test]
    fn test_flash_attention_launch_config() {
        let backend = CudaBackend::new(1, 4096, 4096, 64).with_num_heads(8);
        let (grid, block) = backend.flash_attention_launch_config(256);
        // grid = (256/64, 8, 1) = (4, 8, 1)
        assert_eq!(grid, (4, 8, 1));
        // block = (64 * 64, 1, 1) = (4096, 1, 1)
        assert_eq!(block, (4096, 1, 1));
    }

    // =========================================================================
    // Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_dimensions_valid() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert!(backend.validate_dimensions());
    }

    #[test]
    fn test_validate_dimensions_k_not_divisible() {
        let backend = CudaBackend::new(1, 4096, 4095, 64); // k not divisible by 32
        assert!(!backend.validate_dimensions());
    }

    #[test]
    fn test_validate_dimensions_head_dim_not_power_of_two() {
        let backend = CudaBackend::new(1, 4096, 4096, 80); // 80 is not power of 2
        assert!(!backend.validate_dimensions());
    }

    #[test]
    fn test_validate_dimensions_zero_m() {
        let backend = CudaBackend::new(0, 4096, 4096, 64);
        assert!(!backend.validate_dimensions());
    }

    #[test]
    fn test_validate_dimensions_zero_n() {
        let backend = CudaBackend::new(1, 0, 4096, 64);
        assert!(!backend.validate_dimensions());
    }

    #[test]
    fn test_validate_dimensions_zero_k() {
        let backend = CudaBackend::new(1, 4096, 0, 64);
        assert!(!backend.validate_dimensions());
    }

    #[test]
    fn test_validate_dimensions_zero_head_dim() {
        let backend = CudaBackend::new(1, 4096, 4096, 0);
        assert!(!backend.validate_dimensions());
    }

    // =========================================================================
    // PTX Target Tests
    // =========================================================================

    #[test]
    fn test_ptx_target() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert_eq!(backend.ptx_target(), "sm_89");
    }

    #[test]
    fn test_ptx_version() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert_eq!(backend.ptx_version(), (8, 0));
    }
}
