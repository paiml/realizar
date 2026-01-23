//! Extracted FFN layer operations for CudaExecutor
//!
//! Split from layer.rs (PMAT-802) to reduce module size while maintaining
//! performance through delegation pattern.

use crate::cuda::executor::{CudaExecutor, GpuBuffer, GpuError};

/// PAR-023: GPU-resident SwiGLU FFN operating entirely on GPU buffers
///
/// Implements LLaMA-style FFN: down(swiglu(gate(x), up(x)))
/// All operations chained without sync - only syncs when output needed.
///
/// PAR-063-V5: Set TRUE_DP4A=1 to use Q8 activation quantization + Q4K×Q8
/// integer dot product for 4x instruction reduction (llama.cpp-style).
///
/// # Arguments
/// * `executor` - CUDA executor with cached weights
/// * `input` - GPU buffer containing hidden state [hidden_dim]
/// * `ffn_gate_name` - Cache key for FFN gate weight
/// * `ffn_up_name` - Cache key for FFN up weight
/// * `ffn_down_name` - Cache key for FFN down weight
/// * `hidden_dim` - Model hidden dimension
/// * `intermediate_dim` - FFN intermediate dimension
///
/// # Returns
/// GPU buffer containing FFN output [hidden_dim] - not synchronized
#[allow(clippy::too_many_arguments)]
pub fn fused_ffn_swiglu_gpu(
    executor: &mut CudaExecutor,
    input: &GpuBuffer<f32>,
    ffn_gate_name: &str,
    ffn_up_name: &str,
    ffn_down_name: &str,
    hidden_dim: u32,
    intermediate_dim: u32,
) -> Result<GpuBuffer<f32>, GpuError> {
    // PAR-063-V5: Environment variable to enable TRUE DP4A path
    // Set TRUE_DP4A=1 to use Q8 activation quantization + Q4K×Q8 integer dot product
    static TRUE_DP4A_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let use_true_dp4a = *TRUE_DP4A_ENABLED.get_or_init(|| {
        std::env::var("TRUE_DP4A")
            .map(|v| v == "1")
            .unwrap_or(false)
    });

    if use_true_dp4a {
        return fused_ffn_swiglu_gpu_true_dp4a(
            executor,
            input,
            ffn_gate_name,
            ffn_up_name,
            ffn_down_name,
            hidden_dim,
            intermediate_dim,
        );
    }

    // PAR-063: Kernel selection for FFN layers
    // Priority order:
    // 1. Dp4aQ4KGemv: Best for aligned K (uses DP4A SIMD, 4x instruction reduction)
    // 2. TiledQ4KGemv: K <= 8192 (32KB shared memory, fits in 48KB limit)
    // 3. Q4KGemv: Fallback for unaligned K or large K > 8192

    // For gate/up projection: K = hidden_dim, N = intermediate_dim
    // For down projection: K = intermediate_dim, N = hidden_dim
    const CHUNK_THRESHOLD: u32 = 8192;

    let hidden_aligned = hidden_dim.is_multiple_of(256);
    let intermediate_aligned = intermediate_dim.is_multiple_of(256);

    // 1. Gate projection: [hidden_dim] -> [intermediate_dim] (no sync)
    // PAR-063: Use DP4A kernel for aligned dimensions (fastest)
    let gate = if hidden_aligned && hidden_dim <= CHUNK_THRESHOLD {
        executor.dp4a_q4k_gemv_cached_async(ffn_gate_name, input, intermediate_dim, hidden_dim)?
    } else {
        executor.q4k_gemv_cached_async(ffn_gate_name, input, intermediate_dim, hidden_dim)?
    };

    // 2. Up projection: [hidden_dim] -> [intermediate_dim] (no sync)
    let up = if hidden_aligned && hidden_dim <= CHUNK_THRESHOLD {
        executor.dp4a_q4k_gemv_cached_async(ffn_up_name, input, intermediate_dim, hidden_dim)?
    } else {
        executor.q4k_gemv_cached_async(ffn_up_name, input, intermediate_dim, hidden_dim)?
    };

    // 3. Fused SwiGLU: silu(gate) * up (no sync)
    let activated = executor.fused_swiglu_gpu(&gate, &up, intermediate_dim)?;

    // 4. Down projection: [intermediate_dim] -> [hidden_dim] (no sync)
    // Note: K = intermediate_dim here (input to down projection)
    let output = if intermediate_aligned && intermediate_dim <= CHUNK_THRESHOLD {
        executor.dp4a_q4k_gemv_cached_async(
            ffn_down_name,
            &activated,
            hidden_dim,
            intermediate_dim,
        )?
    } else {
        executor.q4k_gemv_cached_async(ffn_down_name, &activated, hidden_dim, intermediate_dim)?
    };

    // PAR-023: NO sync here - caller chains more operations or syncs when needed
    Ok(output)
}

/// PAR-063-V5: GPU-resident SwiGLU FFN using TRUE DP4A kernels (async, no sync)
///
/// Uses Q8 activation quantization + Q4K×Q8 integer dot product for 4x instruction reduction.
/// This is the llama.cpp-style approach:
/// 1. Quantize f32 activations to Q8_1 (per-block scale + 32 × int8)
/// 2. Use dp4a.u32.s32 for 4 multiply-adds per instruction
/// 3. Apply scales at the end
///
/// # Arguments
/// * `executor` - CUDA executor with cached weights
/// * `input` - GPU buffer containing hidden state [hidden_dim]
/// * `ffn_gate_name` - Cache key for FFN gate weight
/// * `ffn_up_name` - Cache key for FFN up weight
/// * `ffn_down_name` - Cache key for FFN down weight
/// * `hidden_dim` - Model hidden dimension
/// * `intermediate_dim` - FFN intermediate dimension
#[allow(clippy::too_many_arguments)]
pub fn fused_ffn_swiglu_gpu_true_dp4a(
    executor: &mut CudaExecutor,
    input: &GpuBuffer<f32>,
    ffn_gate_name: &str,
    ffn_up_name: &str,
    ffn_down_name: &str,
    hidden_dim: u32,
    intermediate_dim: u32,
) -> Result<GpuBuffer<f32>, GpuError> {
    // PAR-063-V6: Environment variable to enable packed DP4A kernel
    // Set PACKED_DP4A=1 to use the optimized nibble-packed dp4a.u32.s32 kernel
    static PACKED_DP4A_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let use_packed_dp4a = *PACKED_DP4A_ENABLED.get_or_init(|| {
        std::env::var("PACKED_DP4A")
            .map(|v| v == "1")
            .unwrap_or(false)
    });

    // PAR-063-V5/V6: True DP4A pipeline
    // 1. Quantize input activations to Q8_1 once (shared by gate and up projections)
    let q8_input = executor.q8_quantize_async(input, hidden_dim)?;

    // 2. Gate projection using Q4K × Q8 integer dot product
    let gate = if use_packed_dp4a {
        executor.packed_dp4a_q4k_q8_gemv_async(
            ffn_gate_name,
            &q8_input,
            intermediate_dim,
            hidden_dim,
        )?
    } else {
        executor.q4k_q8_gemv_async(ffn_gate_name, &q8_input, intermediate_dim, hidden_dim)?
    };

    // 3. Up projection using Q4K × Q8 integer dot product
    let up = if use_packed_dp4a {
        executor.packed_dp4a_q4k_q8_gemv_async(
            ffn_up_name,
            &q8_input,
            intermediate_dim,
            hidden_dim,
        )?
    } else {
        executor.q4k_q8_gemv_async(ffn_up_name, &q8_input, intermediate_dim, hidden_dim)?
    };

    // 4. Fused SwiGLU: silu(gate) * up
    let activated = executor.fused_swiglu_gpu(&gate, &up, intermediate_dim)?;

    // 5. Quantize activated values for down projection
    let q8_activated = executor.q8_quantize_async(&activated, intermediate_dim)?;

    // 6. Down projection using Q4K × Q8 integer dot product
    let output = if use_packed_dp4a {
        executor.packed_dp4a_q4k_q8_gemv_async(
            ffn_down_name,
            &q8_activated,
            hidden_dim,
            intermediate_dim,
        )?
    } else {
        executor.q4k_q8_gemv_async(ffn_down_name, &q8_activated, hidden_dim, intermediate_dim)?
    };

    // PAR-063-V5/V6: NO sync here - caller chains more operations or syncs when needed
    Ok(output)
}
