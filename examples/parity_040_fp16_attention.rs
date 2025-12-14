//! PARITY-040: FP16 Tensor Core Attention Benchmark
//!
//! Tests FP16 attention vs FP32 attention:
//! - FP32: Current FlashAttention (73.9 GFLOPS)
//! - FP16: Tensor Core GEMM for Q@K and attn@V (target: 150 GFLOPS)
//!
//! Run with: cargo run --release --example parity_040_fp16_attention --features cuda

#[cfg(feature = "cuda")]
use std::time::Instant;

fn main() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║       PARITY-040: FP16 Tensor Core Attention Benchmark         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    #[cfg(feature = "cuda")]
    {
        use realizar::cuda::CudaExecutor;

        // Test configurations matching phi-2 attention
        // head_dim must be multiple of 16 for Tensor Cores
        let configs = [
            (64u32, 64u32, "Small (64x64)"),
            (128, 64, "Medium (128x64)"),
            (256, 64, "phi-2 like (256x64)"),
            (512, 64, "Large (512x64)"),
        ];
        let iterations = 10;

        // Initialize CUDA
        println!("[1/4] Initializing CUDA...");
        let start = Instant::now();
        let mut executor = match CudaExecutor::new(0) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("Failed to initialize CUDA: {}", err);
                std::process::exit(1);
            },
        };
        let cuda_init_time = start.elapsed();
        println!("       CUDA init: {:?}", cuda_init_time);

        if let Ok(name) = executor.device_name() {
            println!("       Device: {}", name);
        }

        // FP32 FlashAttention Benchmark
        println!("\n[2/4] Benchmarking FP32 FlashAttention...");
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║ Configuration     │ Time/iter │ GFLOPS │ Status                 ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");

        let mut fp32_results: Vec<(String, std::time::Duration, f64)> = Vec::new();

        for (seq_len, head_dim, name) in &configs {
            let size = (*seq_len * *head_dim) as usize;

            // Generate test data
            let q: Vec<f32> = (0..size).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
            let k: Vec<f32> = (0..size).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
            let v: Vec<f32> = (0..size).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
            let mut output = vec![0.0f32; size];

            let scale = 1.0 / (*head_dim as f32).sqrt();

            // Warmup
            for _ in 0..3 {
                let _ = executor.flash_attention(
                    &q,
                    &k,
                    &v,
                    &mut output,
                    *seq_len,
                    *head_dim,
                    scale,
                    false,
                );
            }

            // Benchmark FP32 FlashAttention
            let start = Instant::now();
            for _ in 0..iterations {
                executor
                    .flash_attention(&q, &k, &v, &mut output, *seq_len, *head_dim, scale, false)
                    .expect("FlashAttention failed");
            }
            let total_time = start.elapsed();
            let per_iter = total_time / iterations as u32;

            // FLOPs: attention is ~4 * N * N * d (Q@K, softmax, @V)
            let flops = 4.0 * (*seq_len as f64) * (*seq_len as f64) * (*head_dim as f64);
            let gflops = flops / per_iter.as_secs_f64() / 1e9;

            let status = if per_iter.as_micros() < 1000 {
                "✓ <1ms"
            } else if per_iter.as_micros() < 5000 {
                "⚠ <5ms"
            } else {
                "✗ slow"
            };

            println!(
                "║ {:<17} │ {:>9?} │ {:>6.1} │ {}                 ║",
                name, per_iter, gflops, status
            );
            fp32_results.push((name.to_string(), per_iter, gflops));
        }
        println!("╚══════════════════════════════════════════════════════════════════╝");

        // FP16 GEMM Benchmark (simulating attention Q@K and attn@V)
        println!("\n[3/4] Benchmarking FP16 Tensor Core GEMM (Q@K simulation)...");
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║ Configuration     │ Time/iter │ GFLOPS │ Status                 ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");

        let mut fp16_results: Vec<(String, std::time::Duration, f64)> = Vec::new();

        for (seq_len, head_dim, name) in &configs {
            // Q@K^T: [seq_len, head_dim] @ [head_dim, seq_len] = [seq_len, seq_len]
            // For Tensor Cores, we need dimensions multiple of 16
            let m = ((*seq_len + 15) / 16) * 16; // Round up to multiple of 16
            let n = ((*seq_len + 15) / 16) * 16;
            let k = ((*head_dim + 15) / 16) * 16;

            let a_size = (m * k) as usize;
            let b_size = (k * n) as usize;
            let c_size = (m * n) as usize;

            // Generate padded test data
            let a: Vec<f32> = (0..a_size)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
                .collect();
            let b: Vec<f32> = (0..b_size)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.01)
                .collect();
            let mut c = vec![0.0f32; c_size];

            // Warmup
            for _ in 0..3 {
                let _ = executor.gemm_fp16(&a, &b, &mut c, m, n, k);
            }

            // Benchmark FP16 GEMM
            let start = Instant::now();
            for _ in 0..iterations {
                executor
                    .gemm_fp16(&a, &b, &mut c, m, n, k)
                    .expect("FP16 GEMM failed");
            }
            let total_time = start.elapsed();
            let per_iter = total_time / iterations as u32;

            // FLOPs for GEMM: 2 * M * N * K
            let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
            let gflops = flops / per_iter.as_secs_f64() / 1e9;

            let status = if per_iter.as_micros() < 1000 {
                "✓ <1ms"
            } else if per_iter.as_micros() < 5000 {
                "⚠ <5ms"
            } else {
                "✗ slow"
            };

            println!(
                "║ {:<17} │ {:>9?} │ {:>6.1} │ {}                 ║",
                name, per_iter, gflops, status
            );
            fp16_results.push((name.to_string(), per_iter, gflops));
        }
        println!("╚══════════════════════════════════════════════════════════════════╝");

        // Comparison
        println!("\n[4/4] Comparison: FP32 FlashAttention vs FP16 GEMM");
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║ Configuration     │ FP32 GFLOPS │ FP16 GFLOPS │ Speedup         ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");

        let mut total_fp32_gflops = 0.0;
        let mut total_fp16_gflops = 0.0;

        for i in 0..configs.len() {
            let (name, _, fp32_gflops) = &fp32_results[i];
            let (_, _, fp16_gflops) = &fp16_results[i];
            let speedup = fp16_gflops / fp32_gflops;

            total_fp32_gflops += fp32_gflops;
            total_fp16_gflops += fp16_gflops;

            let speedup_str = if speedup > 1.0 {
                format!("{:.2}x faster", speedup)
            } else {
                format!("{:.2}x slower", 1.0 / speedup)
            };

            println!(
                "║ {:<17} │ {:>11.1} │ {:>11.1} │ {:>15} ║",
                name, fp32_gflops, fp16_gflops, speedup_str
            );
        }
        println!("╚══════════════════════════════════════════════════════════════════╝");

        let avg_fp32 = total_fp32_gflops / configs.len() as f64;
        let avg_fp16 = total_fp16_gflops / configs.len() as f64;
        let avg_speedup = avg_fp16 / avg_fp32;

        // Token generation estimate
        // phi-2: 32 layers, 32 heads, head_dim=80
        let layers = 32;
        let num_heads = 32;
        let head_dim = 80u32;
        let seq_len = 128u32;

        // Attention FLOPs per token: 4 * seq_len^2 * head_dim * num_heads * layers
        let attn_flops = 4.0
            * (seq_len as f64)
            * (seq_len as f64)
            * (head_dim as f64)
            * (num_heads as f64)
            * (layers as f64);

        // FFN FLOPs per token (from PARITY-038): 6.53ms at 514 GFLOPS
        let ffn_time = 0.00653; // seconds
        let _ffn_gflops = 514.0;

        // Time estimates
        let attn_time_fp32 = attn_flops / (avg_fp32 * 1e9);
        let attn_time_fp16 = attn_flops / (avg_fp16 * 1e9);

        let total_fp32 = attn_time_fp32 + ffn_time;
        let total_fp16 = attn_time_fp16 + ffn_time;

        let tps_fp32 = 1.0 / total_fp32;
        let tps_fp16 = 1.0 / total_fp16;

        println!("\n═══ Token Generation Estimate (phi-2, 32 layers, 32 heads) ═══");
        println!("  FP32 Attention GFLOPS: {:.1}", avg_fp32);
        println!("  FP16 Attention GFLOPS: {:.1}", avg_fp16);
        println!("  Average speedup: {:.2}x", avg_speedup);
        println!();
        println!("  FP32 attention time: {:.2}ms", attn_time_fp32 * 1000.0);
        println!("  FP16 attention time: {:.2}ms", attn_time_fp16 * 1000.0);
        println!("  FFN time (PARITY-038): {:.2}ms", ffn_time * 1000.0);
        println!();
        println!(
            "  FP32 total time: {:.2}ms → {:.1} tok/s",
            total_fp32 * 1000.0,
            tps_fp32
        );
        println!(
            "  FP16 total time: {:.2}ms → {:.1} tok/s",
            total_fp16 * 1000.0,
            tps_fp16
        );

        // M3/M4 status
        let m3_target = 50.6;
        let m4_target = 202.3;
        println!("\n═══ Milestone Status ═══");
        println!(
            "  M3 (<5x gap, >{:.1} tok/s): FP32: {} | FP16: {}",
            m3_target,
            if tps_fp32 > m3_target { "✓" } else { "✗" },
            if tps_fp16 > m3_target { "✓" } else { "✗" }
        );
        println!(
            "  M4 (<1.25x gap, >{:.1} tok/s): FP32: {} | FP16: {}",
            m4_target,
            if tps_fp32 > m4_target { "✓" } else { "✗" },
            if tps_fp16 > m4_target { "✓" } else { "✗" }
        );

        println!("\n═══ Analysis (PARITY-040) ═══");
        println!("  - FP16 GEMM path tested for Q@K simulation");
        println!("  - Current implementation uses tiled GEMM (not true Tensor Cores)");
        println!("  - True Tensor Core requires FP16 buffer support (half crate)");
        println!("  - trueno-gpu has GemmKernel::tensor_core() with WMMA PTX");
        println!("  - Next: Wire trueno-gpu Tensor Core kernel to CudaExecutor");
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled. Run with --features cuda");
    }
}
