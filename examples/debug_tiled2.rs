//! Debug tiled matmul with model weights

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::RealizarError;

fn main() -> Result<(), RealizarError> {
    let model_path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let layer = &model.layers[0];
    let weight = &layer.ffn_down_weight;
    let in_dim = weight.in_dim;
    let out_dim = weight.out_dim;

    println!("FFN down: {}x{}", out_dim, in_dim);
    println!("Super-blocks per row: {}", in_dim / 256);

    // Create simple activations
    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();

    // Test just first 4 output rows
    let test_out_dim = 4;

    let mut output_orig = vec![0.0f32; test_out_dim];
    let mut output_tiled = vec![0.0f32; test_out_dim];

    // Extract weight data for first 4 rows
    let super_blocks_per_row = in_dim / 256;
    let bytes_per_row = super_blocks_per_row * 144;
    let test_weight_data = &weight.data[..test_out_dim * bytes_per_row];

    println!("\nWeight data size: {} bytes", test_weight_data.len());
    println!("Bytes per row: {}", bytes_per_row);

    realizar::quantize::fused_q4k_parallel_matvec_into(
        test_weight_data,
        &activations,
        in_dim,
        test_out_dim,
        &mut output_orig,
    )?;

    realizar::quantize::fused_q4k_tiled_matvec_into(
        test_weight_data,
        &activations,
        in_dim,
        test_out_dim,
        &mut output_tiled,
    )?;

    println!("\nFirst 4 outputs:");
    for (i, (o, t)) in output_orig.iter().zip(output_tiled.iter()).enumerate() {
        println!(
            "Row {}: orig={:>12.4}, tiled={:>12.4}, diff={:>12.4}",
            i,
            o,
            t,
            (o - t).abs()
        );
    }

    // Check individual tile contributions
    println!("\n--- Debugging tile contributions ---");
    let num_input_tiles = (in_dim / 256 + 3) / 4; // ~9 tiles for 8960
    println!("Number of input tiles: {}", num_input_tiles);

    // Manual single-row computation for row 0
    let row_data = &weight.data[0..bytes_per_row];

    // Process tile by tile manually
    let mut manual_sum = 0.0f32;
    for tile_idx in 0..num_input_tiles {
        let sb_start = tile_idx * 4;
        let sb_end = ((tile_idx + 1) * 4).min(super_blocks_per_row);

        let mut tile_sum = 0.0f32;
        for sb in sb_start..sb_end {
            let sb_offset = sb * 144;
            let sb_data = &row_data[sb_offset..sb_offset + 144];

            // Read d and dmin
            let d = read_f16(&sb_data[0..2]);
            let dmin = read_f16(&sb_data[2..4]);

            let act_start = sb * 256;
            let act_end = (act_start + 256).min(in_dim);

            // Just print the super-block info
            if tile_idx == 0 {
                println!(
                    "  SB {}: d={:.6}, dmin={:.6}, act_range=[{}..{}]",
                    sb, d, dmin, act_start, act_end
                );
            }
        }
    }

    Ok(())
}

fn read_f16(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    half::f16::from_bits(bits).to_f32()
}
