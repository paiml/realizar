//! Test Q6K GEMV specifically for LM head dimensions (vocab_size=151936, hidden_dim=1536)
//!
//! Run: cargo run --release --features cuda --example test_lm_head_q6k

use realizar::gguf::MappedGGUFModel;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;

    // Find the LM head tensor (output.weight)
    let lm_head_tensor = mapped
        .tensors()
        .iter()
        .find(|t| t.name == "output.weight")
        .ok_or("LM head tensor not found")?;

    eprintln!("LM head tensor: {:?}", lm_head_tensor);
    eprintln!("  shape: {:?}", lm_head_tensor.shape);
    eprintln!("  ggml_type: {} (Q6_K=14)", lm_head_tensor.ggml_type);
    eprintln!("  offset: {}", lm_head_tensor.offset);

    // Verify dimensions
    let vocab_size = lm_head_tensor.shape[0] as usize; // Should be 151936
    let hidden_dim = lm_head_tensor.shape[1] as usize; // Should be 1536
    eprintln!("  vocab_size: {}", vocab_size);
    eprintln!("  hidden_dim: {}", hidden_dim);

    // Q6K super-block size: 256 values, 210 bytes
    let sb_size = 256;
    let sb_bytes = 210;
    let n_sb = (hidden_dim + sb_size - 1) / sb_size; // super-blocks per row
    let bytes_per_row = n_sb * sb_bytes;
    let expected_total = vocab_size * bytes_per_row;

    eprintln!("Q6K layout:");
    eprintln!("  n_sb (per row): {}", n_sb);
    eprintln!("  bytes_per_row: {}", bytes_per_row);
    eprintln!("  expected_total: {}", expected_total);

    // Get the weight data
    let weight_data = mapped.tensor_data(&lm_head_tensor.name)?;
    eprintln!("  actual size: {}", weight_data.len());

    // Calculate size discrepancy
    if weight_data.len() != expected_total {
        eprintln!(
            "WARNING: Size mismatch! expected={}, actual={}",
            expected_total,
            weight_data.len()
        );
        eprintln!(
            "  diff: {} bytes",
            weight_data.len() as isize - expected_total as isize
        );
    } else {
        eprintln!("  size OK");
    }

    // Now let's compare CPU vs GPU Q6K GEMV for a sample input
    eprintln!("\n=== Testing Q6K GEMV ===");

    // Create a test input (the normalized hidden state from the debug output)
    // Using deterministic input for reproducibility
    let mut input = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        input[i] = ((i as f32 * 0.1).sin() * 0.5) as f32; // Deterministic pattern
    }
    let input_sum: f32 = input.iter().sum();
    let input_rms: f32 = (input.iter().map(|x| x * x).sum::<f32>() / hidden_dim as f32).sqrt();
    eprintln!(
        "Input: sum={:.4}, rms={:.4}, first 5: {:?}",
        input_sum,
        input_rms,
        &input[..5]
    );

    // CPU Q6K GEMV
    let start = Instant::now();
    let cpu_output = cpu_q6k_gemv(&weight_data, &input, vocab_size, hidden_dim);
    eprintln!("CPU Q6K took {:?}", start.elapsed());

    let cpu_sum: f32 = cpu_output.iter().sum();
    let cpu_max = cpu_output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let cpu_argmax = cpu_output
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "CPU output: sum={:.4}, max={:.4}, argmax={}",
        cpu_sum, cpu_max, cpu_argmax
    );
    eprintln!("CPU first 10: {:?}", &cpu_output[..10]);

    // GPU Q6K GEMV - using trueno-gpu directly
    use trueno_gpu::driver::{CudaContext, CudaStream, GpuBuffer};
    use trueno_gpu::kernels::{Kernel, Q6KGemvKernel};

    let context = CudaContext::new(0)?;
    let stream = CudaStream::new(&context)?;

    // Upload weight data to GPU
    let weight_gpu = GpuBuffer::from_host_u8(&context, &weight_data)?;
    let input_gpu = GpuBuffer::from_host(&context, &input)?;
    let mut output_gpu = GpuBuffer::new(&context, vocab_size)?;

    // Create and run kernel
    let kernel = Q6KGemvKernel::new(vocab_size as u32, hidden_dim as u32);
    let ptx = kernel.generate_ptx();
    let module = trueno_gpu::driver::CudaModule::from_ptx(&context, &ptx)?;

    let config = kernel.launch_config();
    eprintln!(
        "Kernel launch: grid=({}, {}, {}), block=({}, {}, {})",
        config.grid_dim.0,
        config.grid_dim.1,
        config.grid_dim.2,
        config.block_dim.0,
        config.block_dim.1,
        config.block_dim.2
    );

    let start = Instant::now();
    module.launch_kernel(
        kernel.entry_point(),
        &config,
        &stream,
        &[
            &(weight_gpu.as_ptr() as u64),
            &(input_gpu.as_ptr() as u64),
            &(output_gpu.as_ptr() as u64),
        ],
    )?;
    stream.synchronize()?;
    eprintln!("GPU Q6K took {:?}", start.elapsed());

    // Download output
    let mut gpu_output = vec![0.0f32; vocab_size];
    output_gpu.copy_to_host(&mut gpu_output)?;

    let gpu_sum: f32 = gpu_output.iter().sum();
    let gpu_max = gpu_output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let gpu_argmax = gpu_output
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "GPU output: sum={:.4}, max={:.4}, argmax={}",
        gpu_sum, gpu_max, gpu_argmax
    );
    eprintln!("GPU first 10: {:?}", &gpu_output[..10]);

    // Compare
    eprintln!("\n=== Comparison ===");
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut total_diff = 0.0f32;
    for i in 0..vocab_size {
        let diff = (cpu_output[i] - gpu_output[i]).abs();
        total_diff += diff;
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    let avg_diff = total_diff / vocab_size as f32;
    eprintln!("Max diff: {:.6} at idx {}", max_diff, max_diff_idx);
    eprintln!("Avg diff: {:.6}", avg_diff);

    // Calculate correlation
    let cpu_mean: f32 = cpu_output.iter().sum::<f32>() / vocab_size as f32;
    let gpu_mean: f32 = gpu_output.iter().sum::<f32>() / vocab_size as f32;
    let mut cov = 0.0f32;
    let mut cpu_var = 0.0f32;
    let mut gpu_var = 0.0f32;
    for i in 0..vocab_size {
        let cpu_d = cpu_output[i] - cpu_mean;
        let gpu_d = gpu_output[i] - gpu_mean;
        cov += cpu_d * gpu_d;
        cpu_var += cpu_d * cpu_d;
        gpu_var += gpu_d * gpu_d;
    }
    let corr = cov / (cpu_var.sqrt() * gpu_var.sqrt());
    eprintln!("Correlation: {:.4}", corr);

    if corr > 0.99 && max_diff < 1.0 {
        eprintln!("\n✅ Q6K LM head: CPU and GPU match closely!");
    } else if corr > 0.9 {
        eprintln!("\n⚠️ Q6K LM head: Mostly correlated but with errors");
    } else {
        eprintln!("\n❌ Q6K LM head: CPU and GPU produce different output!");

        // Show some specific values where they differ
        eprintln!("\nValues at max diff idx {}:", max_diff_idx);
        eprintln!("  CPU: {:.6}", cpu_output[max_diff_idx]);
        eprintln!("  GPU: {:.6}", gpu_output[max_diff_idx]);

        // Show argmax values
        eprintln!("\nAt CPU argmax {}:", cpu_argmax);
        eprintln!("  CPU: {:.6}", cpu_output[cpu_argmax]);
        eprintln!("  GPU: {:.6}", gpu_output[cpu_argmax]);

        eprintln!("\nAt GPU argmax {}:", gpu_argmax);
        eprintln!("  CPU: {:.6}", cpu_output[gpu_argmax]);
        eprintln!("  GPU: {:.6}", gpu_output[gpu_argmax]);
    }

    Ok(())
}

/// CPU reference implementation of Q6K GEMV
fn cpu_q6k_gemv(weights: &[u8], input: &[f32], n: usize, k: usize) -> Vec<f32> {
    // Q6K super-block: 256 values in 210 bytes
    // Layout: ql (128 bytes) + qh (64 bytes) + scales (16 bytes) + d (2 bytes f16)
    const SB_SIZE: usize = 256;
    const SB_BYTES: usize = 210;

    let n_sb = (k + SB_SIZE - 1) / SB_SIZE;
    let bytes_per_row = n_sb * SB_BYTES;

    let mut output = vec![0.0f32; n];

    for row in 0..n {
        let row_data = &weights[row * bytes_per_row..(row + 1) * bytes_per_row];
        let mut sum = 0.0f32;

        for sb in 0..n_sb {
            let sb_start = sb * SB_BYTES;
            let ql = &row_data[sb_start..sb_start + 128];
            let qh = &row_data[sb_start + 128..sb_start + 192];
            let scales = &row_data[sb_start + 192..sb_start + 208];
            let d_bytes = &row_data[sb_start + 208..sb_start + 210];

            // d is f16
            let d = f16_to_f32(u16::from_le_bytes([d_bytes[0], d_bytes[1]]));

            // Dequantize 256 values
            for i in 0..256 {
                let base_idx = sb * SB_SIZE + i;
                if base_idx >= k {
                    break;
                }

                // Get the 6-bit quantized value
                // ql contains lower 4 bits
                // qh contains upper 2 bits
                let ql_idx = i / 2;
                let qh_idx = i / 4;
                let qh_bit_offset = (i % 4) * 2;

                let ql_byte = ql[ql_idx];
                let qh_byte = qh[qh_idx];

                let ql_val = if i % 2 == 0 {
                    ql_byte & 0x0F
                } else {
                    (ql_byte >> 4) & 0x0F
                };

                let qh_val = ((qh_byte >> qh_bit_offset) & 0x03) as u8;

                // Combine to get 6-bit value (0-63)
                let q6 = ql_val | (qh_val << 4);

                // Get scale for this value (16 scales for 256 values = 16 values per scale)
                let scale_idx = i / 16;
                let scale = scales[scale_idx] as i8;

                // Dequantize: val = d * scale * (q6 - 32)
                let dequant = d * (scale as f32) * ((q6 as i32 - 32) as f32);

                sum += dequant * input[base_idx];
            }
        }

        output[row] = sum;
    }

    output
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal or zero
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut m = mant;
            let mut e = -14i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let f32_exp = ((e + 127) as u32) & 0xFF;
            let f32_mant = m << 13;
            f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | 0x7F800000 | (mant << 13))
    } else {
        // Normal
        let f32_exp = ((exp - 15 + 127) as u32) & 0xFF;
        let f32_mant = mant << 13;
        f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
    }
}
