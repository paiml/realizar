//! CORRECTNESS-002: Direct Q6K kernel test
//!
//! Bypasses the full forward pass to test Q6K GEMV directly with synthetic data.
//!
//! Run with: CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example debug_q6k_direct

use realizar::quantize::fused_q6k_dot;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("CORRECTNESS-002: Direct Q6K kernel test\n");

    // Small test dimensions: 4 output rows, 256 input values (1 super-block)
    let out_dim = 4;
    let in_dim = 256;
    let super_blocks_per_row = (in_dim + 255) / 256;
    let bytes_per_row = super_blocks_per_row * 210;

    eprintln!("Test dimensions: {}x{}", out_dim, in_dim);
    eprintln!("Super-blocks per row: {}", super_blocks_per_row);
    eprintln!("Bytes per row: {}", bytes_per_row);

    // Create synthetic Q6K weight data with known pattern
    let mut weight_data = Vec::with_capacity(out_dim * bytes_per_row);

    for row in 0..out_dim {
        // Create one super-block per row

        // ql: 128 bytes (low 4 bits of each pair)
        // Using simple pattern: value at position i = (row * 10 + i) % 16
        for i in 0..128 {
            let v1 = ((row * 10 + i * 2) % 16) as u8;
            let v2 = ((row * 10 + i * 2 + 1) % 16) as u8;
            weight_data.push(v1 | (v2 << 4)); // Pack two nibbles
        }

        // qh: 64 bytes (high 2 bits of each quad)
        // Simple pattern: all zeros for simplicity
        for _ in 0..64 {
            weight_data.push(0);
        }

        // scales: 16 bytes (i8 per 16-element sub-block)
        // Use simple scaling: scale[i] = row + 1
        for _ in 0..16 {
            weight_data.push((row + 1) as u8);
        }

        // d: 2 bytes (f16 scale)
        let d = half::f16::from_f32(0.1 + row as f32 * 0.1);
        weight_data.extend_from_slice(&d.to_bits().to_le_bytes());

        assert_eq!(weight_data.len(), (row + 1) * bytes_per_row);
    }

    // Create input vector with known pattern
    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let input_sum: f32 = input.iter().sum();
    eprintln!(
        "\nInput vector: sum={:.4}, first 5={:?}",
        input_sum,
        &input[..5]
    );

    // CPU reference computation
    eprintln!("\n=== CPU Reference ===");
    let mut cpu_output = Vec::with_capacity(out_dim);
    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        let row_end = row_start + bytes_per_row;
        let row_data = &weight_data[row_start..row_end];
        let dot = fused_q6k_dot(row_data, &input)?;
        cpu_output.push(dot);
        eprintln!("[CPU] Row {}: {:.6}", row, dot);
    }

    // GPU computation using CUDA executor directly
    #[cfg(feature = "cuda")]
    {
        eprintln!("\n=== GPU Q6K GEMV ===");

        use realizr::cuda::CudaExecutor;
        use trueno_gpu::CudaContext;

        let context = CudaContext::new(0)?;
        let mut executor = CudaExecutor::new(context)?;

        // Upload weights to GPU
        let weight_buf = executor.allocate_buffer::<u8>(weight_data.len())?;
        weight_buf.copy_from_host(&weight_data)?;
        let weight_ptr = weight_buf.as_ptr();

        // Upload input to GPU
        let input_buf = executor.allocate_buffer::<f32>(in_dim)?;
        input_buf.copy_from_host(&input)?;

        // Allocate output buffer
        let output_buf = executor.allocate_buffer::<f32>(out_dim)?;

        // Run Q6K GEMV
        executor.q6k_gemv_into(
            weight_ptr,
            &input_buf,
            &output_buf,
            out_dim as u32,
            in_dim as u32,
        )?;
        executor.sync()?;

        // Download results
        let mut gpu_output = vec![0.0f32; out_dim];
        output_buf.copy_to_host(&mut gpu_output)?;

        for row in 0..out_dim {
            eprintln!("[GPU] Row {}: {:.6}", row, gpu_output[row]);
        }

        // Compare
        eprintln!("\n=== Comparison ===");
        let mut max_diff = 0.0f32;
        for row in 0..out_dim {
            let diff = (cpu_output[row] - gpu_output[row]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            let rel_err = if cpu_output[row].abs() > 1e-6 {
                diff / cpu_output[row].abs()
            } else {
                diff
            };
            eprintln!(
                "Row {}: CPU={:.6}, GPU={:.6}, diff={:.6}, rel_err={:.2}%",
                row,
                cpu_output[row],
                gpu_output[row],
                diff,
                rel_err * 100.0
            );
        }

        if max_diff < 0.01 {
            eprintln!("\n[OK] GPU matches CPU within tolerance");
        } else {
            eprintln!("\n[FAIL] GPU diverges from CPU");
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("\n[SKIP] CUDA not enabled");
    }

    Ok(())
}
