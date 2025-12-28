use realizar::gguf::{MappedGGUFModel, GGUFModel};
use realizar::quantize::dequantize_q4_0;
use std::fs;

fn main() {
    // Load raw model
    let data = fs::read("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf").unwrap();
    let model = GGUFModel::from_bytes(&data).unwrap();
    
    // Get Q weight tensor from layer 0
    let q_tensor = model.tensors.iter().find(|t| t.name == "blk.0.attn_q.weight").unwrap();
    println!("blk.0.attn_q.weight:");
    println!("  dims (after reverse): {:?}", q_tensor.dims);
    println!("  qtype: {}", q_tensor.qtype);
    
    // Get embedding for token 1 (BOS)
    let embed_tensor = model.tensors.iter().find(|t| t.name == "token_embd.weight").unwrap();
    let embed_data = model.get_tensor_f32("token_embd.weight", &data).unwrap();
    
    let hidden_dim = 2048;
    let bos_embed = &embed_data[hidden_dim..2*hidden_dim]; // Token 1
    
    println!("\nBOS embedding (first 5 values): {:?}", &bos_embed[..5]);
    
    // Dequantize Q weight
    let q_offset = model.tensor_data_start + q_tensor.offset as usize;
    let num_elements: usize = q_tensor.dims.iter().map(|&d| d as usize).product();
    let num_blocks = num_elements.div_ceil(32);
    let byte_size = num_blocks * 18;
    let q_data = &data[q_offset..q_offset + byte_size];
    let q_weights = dequantize_q4_0(q_data).unwrap();
    
    println!("\nQ weights dequantized: {} elements", q_weights.len());
    println!("Expected: {}", num_elements);
    
    let out_dim = q_tensor.dims[0] as usize; // 2048
    let in_dim = q_tensor.dims[1] as usize;  // 2048
    println!("out_dim={}, in_dim={}", out_dim, in_dim);
    
    // Test NORMAL matmul: output[o] = dot(input, weight[o, :])
    // Weight layout: [out_dim, in_dim] row-major
    let mut out_normal = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let w_row = &q_weights[o * in_dim..(o + 1) * in_dim];
        for i in 0..in_dim {
            out_normal[o] += bos_embed[i] * w_row[i];
        }
    }
    
    // Test TRANSPOSED matmul: output[o] = dot(input, weight[:, o])
    // Weight layout: [in_dim, out_dim] row-major (we index column-wise)
    let mut out_transposed = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        for i in 0..in_dim {
            // Weight element [i, o] in [in_dim, out_dim] = weights[i * out_dim + o]
            out_transposed[o] += bos_embed[i] * q_weights[i * out_dim + o];
        }
    }
    
    println!("\nNormal matmul output (first 10):");
    for (i, &v) in out_normal.iter().take(10).enumerate() {
        println!("  [{}] = {:.6}", i, v);
    }
    
    println!("\nTransposed matmul output (first 10):");
    for (i, &v) in out_transposed.iter().take(10).enumerate() {
        println!("  [{}] = {:.6}", i, v);
    }
    
    // Statistics
    fn stats(v: &[f32]) -> (f32, f32, f32) {
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        (min, max, mean)
    }
    
    let (n_min, n_max, n_mean) = stats(&out_normal);
    let (t_min, t_max, t_mean) = stats(&out_transposed);
    
    println!("\nNormal stats: min={:.4}, max={:.4}, mean={:.4}", n_min, n_max, n_mean);
    println!("Transposed stats: min={:.4}, max={:.4}, mean={:.4}", t_min, t_max, t_mean);
}
