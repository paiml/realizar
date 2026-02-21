impl CudaKernels {
    /// Create a new CUDA kernel generator
    #[must_use]
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Generate PTX source for the specified kernel
    ///
    /// Returns PTX assembly that can be loaded by the CUDA driver API.
    /// Delegates to category-specific helpers to keep cyclomatic complexity low.
    #[must_use]
    pub fn generate_ptx(&self, kernel_type: &KernelType) -> String {
        Self::generate_gemm_ptx(kernel_type)
            .or_else(|| Self::generate_gemv_ptx(kernel_type))
            .or_else(|| Self::generate_q4k_gemv_ptx(kernel_type))
            .or_else(|| Self::generate_attention_ptx(kernel_type))
            .or_else(|| Self::generate_norm_rope_ptx(kernel_type))
            .or_else(|| Self::generate_activation_misc_ptx(kernel_type))
            .unwrap_or_default()
    }
}

include!("kernels_generate_gemm_cuda.rs");
