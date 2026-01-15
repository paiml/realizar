//! Test individual layer 0 operations to find exact divergence point
//!
//! Tests: RMSNorm, Q4K GEMV (Q projection), RoPE, Attention

#[cfg(feature = "cuda")]
use realizar::gguf::OwnedQuantizedModelCuda;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This example requires the 'cuda' feature");
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        run_layer0_test()
    }
}

#[cfg(feature = "cuda")]
fn run_layer0_test() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = cpu_model.config.hidden_dim;
    let vocab_size = cpu_model.config.vocab_size;
    let num_heads = cpu_model.config.num_heads;
    let num_kv_heads = cpu_model.config.num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let eps = cpu_model.config.eps;

    eprintln!("=== Layer 0 Operation Test ===");
    eprintln!("hidden_dim: {}", hidden_dim);
    eprintln!(
        "num_heads: {}, num_kv_heads: {}, head_dim: {}",
        num_heads, num_kv_heads, head_dim
    );
    eprintln!("epsilon: {}", eps);

    // Get embedding for token 791
    let test_token: u32 = 791;
    let embedding = cpu_model.get_embedding(test_token as usize);

    eprintln!("\n=== Embedding ===");
    let emb_sum: f32 = embedding.iter().sum();
    eprintln!("Embedding sum: {:.6}", emb_sum);
    eprintln!("Embedding[0..4]: {:?}", &embedding[..4]);

    // Test 1: RMSNorm
    eprintln!("\n=== Test 1: RMSNorm ===");
    let attn_norm_gamma = cpu_model.get_layer_attn_norm_gamma(0);
    eprintln!("attn_norm_gamma[0..4]: {:?}", &attn_norm_gamma[..4]);

    // CPU RMSNorm
    let cpu_normed = cpu_rmsnorm(&embedding, &attn_norm_gamma, eps);
    let cpu_normed_sum: f32 = cpu_normed.iter().sum();
    eprintln!("CPU RMSNorm sum: {:.6}", cpu_normed_sum);
    eprintln!("CPU RMSNorm[0..4]: {:?}", &cpu_normed[..4]);

    // GPU RMSNorm - use the model's GPU path
    // We need to initialize the GPU model and run a forward pass to get GPU state
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    // Run GPU RMSNorm directly
    let gpu_normed = cuda_model.test_rmsnorm(&embedding, &attn_norm_gamma, eps)?;
    let gpu_normed_sum: f32 = gpu_normed.iter().sum();
    eprintln!("GPU RMSNorm sum: {:.6}", gpu_normed_sum);
    eprintln!("GPU RMSNorm[0..4]: {:?}", &gpu_normed[..4]);

    let rmsnorm_diff = gpu_normed_sum - cpu_normed_sum;
    eprintln!("RMSNorm sum diff: {:.6}", rmsnorm_diff);

    // Compare element-wise
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    for i in 0..hidden_dim {
        let diff = (gpu_normed[i] - cpu_normed[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    eprintln!(
        "Max element diff: {:.6} at index {}",
        max_diff, max_diff_idx
    );

    if rmsnorm_diff.abs() < 0.001 && max_diff < 0.001 {
        eprintln!("RMSNorm: PASS");
    } else {
        eprintln!("RMSNorm: FAIL - divergence detected");
    }

    // Test 2: Q4K GEMV (Q projection)
    eprintln!("\n=== Test 2: Q4K GEMV (Q projection) ===");
    let q_weight = cpu_model.get_layer_attn_q_weight(0);
    let q_dim = num_heads * head_dim;

    // CPU Q4K dot product
    let cpu_q = cpu_q4k_gemv(&cpu_normed, &q_weight.data, hidden_dim, q_dim)?;
    let cpu_q_sum: f32 = cpu_q.iter().sum();
    eprintln!("CPU Q projection sum: {:.6}", cpu_q_sum);
    eprintln!("CPU Q[0..4]: {:?}", &cpu_q[..4]);

    // GPU Q4K dot product
    let gpu_q = cuda_model.test_q4k_gemv(&gpu_normed, &q_weight.data, q_dim)?;
    let gpu_q_sum: f32 = gpu_q.iter().sum();
    eprintln!("GPU Q projection sum: {:.6}", gpu_q_sum);
    eprintln!("GPU Q[0..4]: {:?}", &gpu_q[..4]);

    let q_diff = gpu_q_sum - cpu_q_sum;
    eprintln!("Q projection sum diff: {:.6}", q_diff);

    if q_diff.abs() < 0.01 {
        eprintln!("Q4K GEMV: PASS");
    } else {
        eprintln!("Q4K GEMV: FAIL - divergence detected");
    }

    Ok(())
}

/// CPU RMSNorm
#[cfg(feature = "cuda")]
fn cpu_rmsnorm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let rms_inv = 1.0 / rms;

    x.iter()
        .zip(gamma.iter())
        .map(|(xi, gi)| xi * rms_inv * gi)
        .collect()
}

/// CPU Q4K GEMV (simplified)
#[cfg(feature = "cuda")]
fn cpu_q4k_gemv(
    input: &[f32],
    weight: &[u8],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let sb_per_row = (in_dim + 255) / 256;
    let bytes_per_row = sb_per_row * 144; // Q4K: 144 bytes per super-block

    let mut output = vec![0.0f32; out_dim];

    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        let row_data = &weight[row_start..row_start + bytes_per_row];
        output[row] = q4k_dot_cpu(row_data, input, in_dim);
    }

    Ok(output)
}

/// CPU Q4K dot product
#[cfg(feature = "cuda")]
fn q4k_dot_cpu(row_data: &[u8], input: &[f32], k: usize) -> f32 {
    let mut result = 0.0f32;
    let num_sb = (k + 255) / 256;

    for sb_idx in 0..num_sb {
        let sb_offset = sb_idx * 144;
        let sb_data = &row_data[sb_offset..sb_offset + 144];

        // Q4K layout: d (f16), dmin (f16), scales[12], qs[128]
        let d = f16_to_f32(u16::from_le_bytes([sb_data[0], sb_data[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([sb_data[2], sb_data[3]]));
        let scales = &sb_data[4..16];
        let qs = &sb_data[16..144];

        // Process 256 values in this super-block
        for val_idx in 0..256 {
            let k_idx = sb_idx * 256 + val_idx;
            if k_idx >= k {
                break;
            }

            let x = input[k_idx];

            // Q4K dequantization
            let block_idx = val_idx / 32;
            let in_block_idx = val_idx % 32;

            // Extract scale and min for this block
            let (scale, min) = extract_scale_min(scales, block_idx);

            // Get quantized value (4 bits per value, packed 2 per byte)
            let byte_idx = val_idx / 2;
            let q_byte = qs[byte_idx];
            let q_val = if val_idx % 2 == 0 {
                q_byte & 0x0F
            } else {
                q_byte >> 4
            };

            // Dequantize
            let dequant = d * scale * (q_val as f32) - dmin * min;
            result += x * dequant;
        }
    }

    result
}

#[cfg(feature = "cuda")]
fn extract_scale_min(scales: &[u8], block_idx: usize) -> (f32, f32) {
    let j = block_idx;
    let (scale_bits, min_bits) = if j < 4 {
        let s = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (s, m)
    } else {
        let s = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (s, m)
    };
    (scale_bits as f32, min_bits as f32)
}

#[cfg(feature = "cuda")]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let f = mant as f32 / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -f } else { f };
    }
    if exp == 31 {
        if mant == 0 {
            return if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
        }
        return f32::NAN;
    }

    let f32_exp = (exp as i32 - 15 + 127) as u32;
    let f32_mant = mant << 13;
    let f32_bits = (sign << 31) | (f32_exp << 23) | f32_mant;
    f32::from_bits(f32_bits)
}
