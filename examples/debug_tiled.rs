//! Debug tiled matmul implementation

use realizar::quantize::{fused_q4k_parallel_matvec_into, fused_q4k_tiled_matvec_into};
use realizar::RealizarError;

fn main() -> Result<(), RealizarError> {
    // Create a small test case: 4 output rows, 512 input (2 super-blocks)
    let out_dim = 4;
    let in_dim = 512; // 2 super-blocks
    let bytes_per_super_block = 144;
    let super_blocks_per_row = 2;
    let bytes_per_row = super_blocks_per_row * bytes_per_super_block; // 288

    // Create weight data (all zeros except d and dmin)
    let mut weight_data = vec![0u8; out_dim * bytes_per_row];

    // Set d=1.0, dmin=0.0 for each super-block
    for o in 0..out_dim {
        for sb in 0..super_blocks_per_row {
            let offset = o * bytes_per_row + sb * bytes_per_super_block;
            // d (f16) = 1.0 = 0x3C00
            weight_data[offset] = 0x00;
            weight_data[offset + 1] = 0x3C;
            // dmin (f16) = 0.0 = 0x0000
            weight_data[offset + 2] = 0x00;
            weight_data[offset + 3] = 0x00;
            // scales: set first scale to 1 (6 bits)
            weight_data[offset + 4] = 0x01; // scale[0] = 1
        }
    }

    // Create activations (all 1.0)
    let activations = vec![1.0f32; in_dim];

    // Test original
    let mut output_orig = vec![0.0f32; out_dim];
    fused_q4k_parallel_matvec_into(
        &weight_data,
        &activations,
        in_dim,
        out_dim,
        &mut output_orig,
    )?;
    println!("Original output: {:?}", output_orig);

    // Test tiled
    let mut output_tiled = vec![0.0f32; out_dim];
    fused_q4k_tiled_matvec_into(
        &weight_data,
        &activations,
        in_dim,
        out_dim,
        &mut output_tiled,
    )?;
    println!("Tiled output:    {:?}", output_tiled);

    // Check difference
    for (i, (o, t)) in output_orig.iter().zip(output_tiled.iter()).enumerate() {
        println!(
            "Row {}: orig={:.6}, tiled={:.6}, diff={:.6}",
            i,
            o,
            t,
            (o - t).abs()
        );
    }

    Ok(())
}
