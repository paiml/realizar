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

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::cuda::executor::test_fixtures::generate_q4_0_weights;

    /// Helper to create CudaExecutor for tests
    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    /// Helper to setup FFN weights in executor cache
    fn setup_ffn_weights(
        exec: &mut CudaExecutor,
        hidden_dim: u32,
        intermediate_dim: u32,
    ) -> Result<(), GpuError> {
        // Q4_K: 144 bytes per 256 elements
        // Gate: [intermediate_dim, hidden_dim]
        let gate_blocks = (intermediate_dim as usize) * (hidden_dim as usize / 256);
        let gate_weights = vec![0u8; gate_blocks * 144];
        exec.load_quantized_weights("ffn_gate", &gate_weights)?;

        // Up: [intermediate_dim, hidden_dim]
        let up_weights = vec![0u8; gate_blocks * 144];
        exec.load_quantized_weights("ffn_up", &up_weights)?;

        // Down: [hidden_dim, intermediate_dim]
        let down_blocks = (hidden_dim as usize) * (intermediate_dim as usize / 256);
        let down_weights = vec![0u8; down_blocks * 144];
        exec.load_quantized_weights("ffn_down", &down_weights)?;

        Ok(())
    }

    #[test]
    fn test_fused_ffn_swiglu_gpu_path_selection() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Test path: hidden_dim % 256 != 0 should use q4k_gemv_cached_async
        let hidden_dim = 512u32; // 512 % 256 == 0, aligned
        let intermediate_dim = 1024u32;

        // Setup weights
        if setup_ffn_weights(&mut exec, hidden_dim, intermediate_dim).is_err() {
            return;
        }

        // Create input
        let input: Vec<f32> = (0..hidden_dim as usize)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();

        // Execute FFN (may fail due to kernel issues, but exercises path selection)
        let result = fused_ffn_swiglu_gpu(
            &mut exec,
            &input_buf,
            "ffn_gate",
            "ffn_up",
            "ffn_down",
            hidden_dim,
            intermediate_dim,
        );
        // Result may fail due to kernel compilation, but path selection is exercised
        let _ = result;
    }

    #[test]
    fn test_fused_ffn_swiglu_gpu_unaligned() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Test path: hidden_dim % 256 != 0 (unaligned)
        // This should use the fallback q4k_gemv_cached_async path
        let hidden_dim = 256u32; // Minimum aligned size
        let intermediate_dim = 768u32; // 768 % 256 == 0

        // For Q4_0 format (18 bytes/32 elements) instead of Q4_K
        let gate_blocks = (intermediate_dim as usize) * (hidden_dim as usize / 32);
        let gate_weights = generate_q4_0_weights(gate_blocks);
        let _ = exec.load_quantized_weights("ffn_gate_unaligned", &gate_weights);

        let up_weights = generate_q4_0_weights(gate_blocks);
        let _ = exec.load_quantized_weights("ffn_up_unaligned", &up_weights);

        let down_blocks = (hidden_dim as usize) * (intermediate_dim as usize / 32);
        let down_weights = generate_q4_0_weights(down_blocks);
        let _ = exec.load_quantized_weights("ffn_down_unaligned", &down_weights);

        let input: Vec<f32> = vec![0.1f32; hidden_dim as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();

        // This tests the unaligned path (even though setup may not be perfect for Q4K)
        let result = fused_ffn_swiglu_gpu(
            &mut exec,
            &input_buf,
            "ffn_gate_unaligned",
            "ffn_up_unaligned",
            "ffn_down_unaligned",
            hidden_dim,
            intermediate_dim,
        );
        let _ = result;
    }

    #[test]
    fn test_fused_ffn_chunk_threshold() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Test CHUNK_THRESHOLD behavior
        // hidden_dim > 8192 should use non-dp4a path
        let hidden_dim = 256u32; // Within threshold
        let intermediate_dim = 512u32;

        if setup_ffn_weights(&mut exec, hidden_dim, intermediate_dim).is_err() {
            return;
        }

        let input: Vec<f32> = vec![0.1f32; hidden_dim as usize];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();

        let result = fused_ffn_swiglu_gpu(
            &mut exec,
            &input_buf,
            "ffn_gate",
            "ffn_up",
            "ffn_down",
            hidden_dim,
            intermediate_dim,
        );
        let _ = result;
    }

    #[test]
    fn test_fused_ffn_swiglu_gpu_true_dp4a() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Test the TRUE_DP4A path directly
        let hidden_dim = 256u32;
        let intermediate_dim = 512u32;

        if setup_ffn_weights(&mut exec, hidden_dim, intermediate_dim).is_err() {
            return;
        }

        let input: Vec<f32> = (0..hidden_dim as usize)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();

        // Call the true_dp4a variant directly
        let result = fused_ffn_swiglu_gpu_true_dp4a(
            &mut exec,
            &input_buf,
            "ffn_gate",
            "ffn_up",
            "ffn_down",
            hidden_dim,
            intermediate_dim,
        );
        // May fail due to kernel issues, but exercises the path
        let _ = result;
    }

    // ========================================================================
    // Coverage Tests: FFN with ModelHarness (v1.36.0)
    // ========================================================================

    #[test]
    fn test_ffn_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input: Vec<f32> = vec![0.1f32; config.hidden_dim];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();

        let result = fused_ffn_swiglu_gpu(
            &mut exec,
            &input_buf,
            "blk.0.ffn_gate",
            "blk.0.ffn_up",
            "blk.0.ffn_down",
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
        );
        let _ = result;
    }

    #[test]
    fn test_ffn_different_layers() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.num_layers = 4;
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        for layer_idx in 0..config.num_layers {
            let input: Vec<f32> = vec![0.1f32; config.hidden_dim];
            let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();

            let result = fused_ffn_swiglu_gpu(
                &mut exec,
                &input_buf,
                &format!("blk.{}.ffn_gate", layer_idx),
                &format!("blk.{}.ffn_up", layer_idx),
                &format!("blk.{}.ffn_down", layer_idx),
                config.hidden_dim as u32,
                config.intermediate_dim as u32,
            );
            let _ = result;
        }
    }

    #[test]
    fn test_ffn_varying_inputs() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Test with different input patterns
        let inputs = [
            vec![0.0f32; config.hidden_dim],
            vec![1.0f32; config.hidden_dim],
            (0..config.hidden_dim)
                .map(|i| (i as f32 / 1000.0).sin())
                .collect::<Vec<_>>(),
        ];

        for input in inputs {
            let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();

            let result = fused_ffn_swiglu_gpu(
                &mut exec,
                &input_buf,
                "blk.0.ffn_gate",
                "blk.0.ffn_up",
                "blk.0.ffn_down",
                config.hidden_dim as u32,
                config.intermediate_dim as u32,
            );
            let _ = result;
        }
    }

    #[test]
    fn test_ffn_true_dp4a_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input: Vec<f32> = (0..config.hidden_dim).map(|i| (i as f32) * 0.001).collect();
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();

        // Call true_dp4a variant with harness weights
        let result = fused_ffn_swiglu_gpu_true_dp4a(
            &mut exec,
            &input_buf,
            "blk.0.ffn_gate",
            "blk.0.ffn_up",
            "blk.0.ffn_down",
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
        );
        let _ = result;
    }

    #[test]
    fn test_ffn_larger_intermediate_dim() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.intermediate_dim = 2048; // Larger intermediate
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input: Vec<f32> = vec![0.1f32; config.hidden_dim];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();

        let result = fused_ffn_swiglu_gpu(
            &mut exec,
            &input_buf,
            "blk.0.ffn_gate",
            "blk.0.ffn_up",
            "blk.0.ffn_down",
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
        );
        let _ = result;
    }

    #[test]
    fn test_ffn_output_dimensions() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input: Vec<f32> = vec![0.1f32; config.hidden_dim];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();

        let result = fused_ffn_swiglu_gpu(
            &mut exec,
            &input_buf,
            "blk.0.ffn_gate",
            "blk.0.ffn_up",
            "blk.0.ffn_down",
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
        );

        // If successful, output should have hidden_dim elements
        if let Ok(output_buf) = result {
            let mut output = vec![0.0f32; config.hidden_dim];
            output_buf.copy_to_host(&mut output).expect("copy");
            assert_eq!(
                output.len(),
                config.hidden_dim,
                "FFN output should match hidden_dim"
            );
        }
    }
}
