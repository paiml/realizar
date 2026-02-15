
impl OwnedQuantizedModel {

    /// Tiled causal attention
    ///
    /// IMP-111c: Flash Attention with causal masking.
    /// For position i, only attends to positions 0..=i.
    #[allow(clippy::too_many_arguments)]
    pub fn tiled_causal_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
        tile_size: usize,
    ) -> Result<Vec<f32>> {
        let tile_size = tile_size.max(1);
        let mut output = vec![0.0f32; seq_len * head_dim];

        // Process each query position
        for i in 0..seq_len {
            let q_i = &q[i * head_dim..(i + 1) * head_dim];

            // Running statistics for online softmax
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;
            let mut running_output = vec![0.0f32; head_dim];

            // Only process K/V up to position i (causal)
            let causal_len = i + 1;

            // Process K/V in tiles
            for tile_start in (0..causal_len).step_by(tile_size) {
                let tile_end = (tile_start + tile_size).min(causal_len);

                // Compute scores for this tile: q_i @ K_tile^T
                let mut tile_scores = Vec::with_capacity(tile_end - tile_start);
                for j in tile_start..tile_end {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_i[d] * k[j * head_dim + d];
                    }
                    tile_scores.push(dot * scale);
                }

                // Find tile max
                let tile_max = tile_scores
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                // Update running statistics
                let new_max = running_max.max(tile_max);

                // Rescale previous output and sum
                if new_max > running_max && running_sum > 0.0 {
                    let rescale = (running_max - new_max).exp();
                    running_sum *= rescale;
                    for out_val in &mut running_output {
                        *out_val *= rescale;
                    }
                }
                running_max = new_max;

                // Accumulate this tile's contribution
                for (idx, &score) in tile_scores.iter().enumerate() {
                    let j = tile_start + idx;
                    let weight = (score - running_max).exp();
                    running_sum += weight;
                    for d in 0..head_dim {
                        running_output[d] += weight * v[j * head_dim + d];
                    }
                }
            }

            // Normalize output
            if running_sum > 0.0 {
                for d in 0..head_dim {
                    output[i * head_dim + d] = running_output[d] / running_sum;
                }
            }
        }

        Ok(output)
    }
}

include!("batch_part_02_part_02.rs");
include!("batch_part_02_part_03.rs");
include!("batch_part_02_part_04.rs");
include!("batch_part_02_part_05.rs");
