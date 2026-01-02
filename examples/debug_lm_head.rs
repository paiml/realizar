//! Debug LM head projection

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q6_k, fused_q6k_parallel_matvec};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn reference_matvec(weight: &[f32], input: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut sum = 0.0f32;
        for i in 0..in_dim {
            sum += weight[o * in_dim + i] * input[i];
        }
        output[o] = sum;
    }
    output
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    println!("=== LM Head Debug ===\n");

    // LM head weight info
    let lm_head = &model.lm_head_weight;
    println!("LM head weight:");
    println!("  in_dim: {} (hidden_dim)", lm_head.in_dim);
    println!("  out_dim: {} (vocab_size)", lm_head.out_dim);
    println!("  qtype: {} (14=Q6_K)", lm_head.qtype);
    println!("  data.len: {}", lm_head.data.len());

    // Create a test input (unit vector with some pattern)
    let in_dim = lm_head.in_dim;
    let out_dim = lm_head.out_dim;

    // Use final_hidden from a known state
    // For now, use a simple pattern
    let mut test_input = vec![0.0f32; in_dim];
    for i in 0..in_dim {
        test_input[i] = (i as f32 / in_dim as f32 - 0.5) * 0.1;
    }
    let input_l2 = l2_norm(&test_input);
    println!("\nTest input L2: {:.6}", input_l2);

    // Dequantize LM head weight
    let dequant = dequantize_q6_k(&lm_head.data).expect("Failed to dequantize");
    println!(
        "Dequantized weight length: {} (expected {})",
        dequant.len(),
        in_dim * out_dim
    );
    println!("Dequantized weight L2: {:.4}", l2_norm(&dequant));

    // Reference matvec
    let ref_output = reference_matvec(&dequant, &test_input, in_dim, out_dim);
    println!("\nReference matvec:");
    println!("  L2: {:.4}", l2_norm(&ref_output));
    println!("  First 5: {:?}", &ref_output[0..5]);

    // Fused matvec
    let fused_output = fused_q6k_parallel_matvec(&lm_head.data, &test_input, in_dim, out_dim)
        .expect("Fused matvec failed");
    println!("\nFused matvec:");
    println!("  L2: {:.4}", l2_norm(&fused_output));
    println!("  First 5: {:?}", &fused_output[0..5]);

    // Compare
    let diff_l2: f32 = ref_output
        .iter()
        .zip(fused_output.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    println!("\nL2 of difference: {:.6}", diff_l2);

    // Now use actual final hidden state (simulate)
    // Use a uniform distribution normalized to L2=81 (what we expect)
    let scale = 81.0 / (in_dim as f32).sqrt();
    let real_input: Vec<f32> = (0..in_dim)
        .map(|i| {
            let x = (i as f32 * 2.718281828).sin(); // Pseudo-random pattern
            x * scale
        })
        .collect();
    println!("\nReal-like input L2: {:.4}", l2_norm(&real_input));

    let real_output = fused_q6k_parallel_matvec(&lm_head.data, &real_input, in_dim, out_dim)
        .expect("Fused matvec failed");
    println!("Real-like output L2: {:.4}", l2_norm(&real_output));
    println!(
        "Expected output L2 (proportional to input): ~{:.0}",
        81.0 * 866.77 / 90.52
    );

    println!("\n=== Complete ===");
}
