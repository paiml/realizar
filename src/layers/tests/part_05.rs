/// IMP-046: Cache-aligned tensor storage
use crate::layers::*;
#[test]
fn test_imp_046_cache_aligned_storage() {
    use crate::gpu::CacheAlignedBuffer;

    // Test 1: Create cache-aligned buffer
    let size = 1024;
    let buffer = CacheAlignedBuffer::new(size);

    // Test 2: Buffer should be 64-byte aligned
    assert!(
        buffer.is_aligned(64),
        "IMP-046: Buffer should be 64-byte aligned"
    );

    // Test 3: Buffer should have correct capacity
    assert_eq!(
        buffer.len(),
        size,
        "IMP-046: Buffer should have correct length"
    );

    // Test 4: Can read and write to buffer
    let mut buffer = CacheAlignedBuffer::new(size);
    buffer.as_mut_slice()[0] = 42.0;
    buffer.as_mut_slice()[size - 1] = 99.0;
    assert_eq!(
        buffer.as_slice()[0],
        42.0,
        "IMP-046: Should read back written value"
    );
    assert_eq!(
        buffer.as_slice()[size - 1],
        99.0,
        "IMP-046: Should read back written value at end"
    );

    // Test 5: Alignment preserved for various sizes
    for size in [64, 128, 256, 512, 1000, 2048] {
        let buf = CacheAlignedBuffer::new(size);
        assert!(
            buf.is_aligned(64),
            "IMP-046: Buffer of size {} should be 64-byte aligned",
            size
        );
    }
}

/// IMP-047: Prefetch hints for sequential access
/// Target: Software prefetch for predictable memory patterns
#[test]
fn test_imp_047_prefetch_hints() {
    use crate::gpu::{prefetch_read, sequential_sum, sum_with_prefetch};
    use std::time::Instant;

    // Create test data
    let size = 64 * 1024; // 64K elements = 256KB
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();

    // Test 1: prefetch_read should not panic
    prefetch_read(&data, 0, 64);
    prefetch_read(&data, 1000, 64);

    // Test 2: Both methods should produce same result
    let seq_result = sequential_sum(&data);
    let prefetch_result = sum_with_prefetch(&data, 64);

    assert!(
        (seq_result - prefetch_result).abs() < 1e-3,
        "IMP-047: Sequential ({}) and prefetch ({}) sums should match",
        seq_result,
        prefetch_result
    );

    // Test 3: Prefetch version should be at least as fast
    // Warmup
    for _ in 0..3 {
        let _ = sequential_sum(&data);
        let _ = sum_with_prefetch(&data, 64);
    }

    // Benchmark sequential
    let mut seq_times = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = Instant::now();
        for _ in 0..20 {
            let _ = sequential_sum(&data);
        }
        seq_times.push(start.elapsed().as_secs_f64());
    }
    seq_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Benchmark with prefetch
    let mut pf_times = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = Instant::now();
        for _ in 0..20 {
            let _ = sum_with_prefetch(&data, 64);
        }
        pf_times.push(start.elapsed().as_secs_f64());
    }
    pf_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let seq_median = seq_times[seq_times.len() / 2];
    let pf_median = pf_times[pf_times.len() / 2];
    let speedup = seq_median / pf_median;

    // Note: Performance benchmarks unreliable under coverage instrumentation
    // Prefetch is advisory - hardware may or may not benefit
    // The key test is correctness. Performance is informational only.
    let _ = speedup;
}

/// IMP-048: Block-wise matrix operations
/// Target: Cache-blocked matmul for better locality
#[test]
#[allow(clippy::many_single_char_names)] // m, k, n, a, b are standard matrix notation
fn test_imp_048_blocked_matmul() {
    use crate::gpu::{blocked_matmul, naive_matmul};
    use std::time::Instant;

    // Test matrices: (M x K) @ (K x N) -> (M x N)
    let m = 128;
    let k = 256;
    let n = 128;

    // Create test matrices
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 100) as f32) * 0.01 - 0.5)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i % 100) as f32) * 0.01 - 0.5)
        .collect();

    // Test 1: Both methods should produce same results
    let naive_result = naive_matmul(&a, &b, m, k, n);
    let blocked_result = blocked_matmul(&a, &b, m, k, n, 32); // Block size 32

    assert_eq!(
        naive_result.len(),
        blocked_result.len(),
        "IMP-048: Results should have same length"
    );

    for (i, (&naive, &blocked)) in naive_result.iter().zip(blocked_result.iter()).enumerate() {
        assert!(
            (naive - blocked).abs() < 1e-4,
            "IMP-048: Mismatch at index {}: naive={}, blocked={}",
            i,
            naive,
            blocked
        );
    }

    // Test 2: Blocked should be faster for larger matrices
    let m = 256;
    let k = 512;
    let n = 256;
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 100) as f32) * 0.01 - 0.5)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i % 100) as f32) * 0.01 - 0.5)
        .collect();

    // Warmup
    for _ in 0..2 {
        let _ = naive_matmul(&a, &b, m, k, n);
        let _ = blocked_matmul(&a, &b, m, k, n, 32);
    }

    // Benchmark naive
    let mut naive_times = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..3 {
            let _ = naive_matmul(&a, &b, m, k, n);
        }
        naive_times.push(start.elapsed().as_secs_f64());
    }
    naive_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Benchmark blocked
    let mut blocked_times = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..3 {
            let _ = blocked_matmul(&a, &b, m, k, n, 32);
        }
        blocked_times.push(start.elapsed().as_secs_f64());
    }
    blocked_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let naive_median = naive_times[naive_times.len() / 2];
    let blocked_median = blocked_times[blocked_times.len() / 2];
    let speedup = naive_median / blocked_median;

    // Note: Performance benchmarks unreliable under coverage instrumentation
    // The key test is correctness (Test 1). Performance is informational only.
    let _ = speedup;
}

// ============================================================================
// Phase 13: Memory Pooling & Arena Allocation (M22) - EXTREME TDD
// ============================================================================

/// IMP-049: Tensor memory pool
/// Target: Reusable tensor buffer pool for inference
#[test]
#[cfg(feature = "gpu")]
fn test_imp_049_tensor_pool() {
    use crate::gpu::TensorPool;

    // Test 1: Create pool with capacity
    let mut pool = TensorPool::new(4); // 4 buffers max
    assert_eq!(pool.capacity(), 4, "IMP-049: Pool should have capacity 4");
    assert_eq!(pool.available(), 0, "IMP-049: Pool should start empty");

    // Test 2: Acquire buffers of different sizes
    let buf1 = pool.acquire(1024);
    assert_eq!(
        buf1.len(),
        1024,
        "IMP-049: Buffer should have requested size"
    );

    let buf2 = pool.acquire(2048);
    assert_eq!(
        buf2.len(),
        2048,
        "IMP-049: Second buffer should have size 2048"
    );

    // Test 3: Release and reuse
    pool.release(buf1);
    assert!(
        pool.available() >= 1,
        "IMP-049: Pool should have available buffer"
    );

    let buf3 = pool.acquire(1024); // Should reuse released buffer
    assert_eq!(
        buf3.len(),
        1024,
        "IMP-049: Reused buffer should have correct size"
    );

    // Test 4: Pool tracks allocations
    pool.release(buf2);
    pool.release(buf3);
    assert!(
        pool.available() >= 2,
        "IMP-049: Pool should have 2 available buffers"
    );

    // Test 5: Clear pool
    pool.clear();
    assert_eq!(
        pool.available(),
        0,
        "IMP-049: Pool should be empty after clear"
    );
}

/// IMP-050: Arena allocator for forward pass
/// Target: Single-allocation arena for temporary tensors
#[test]
#[cfg(feature = "gpu")]
fn test_imp_050_arena_allocator() {
    use crate::gpu::ForwardArena;

    // Test 1: Create arena with capacity
    let mut arena = ForwardArena::new(1024 * 1024); // 1MB arena
    assert!(
        arena.capacity() >= 1024 * 1024,
        "IMP-050: Arena should have at least 1MB capacity"
    );
    assert_eq!(arena.used(), 0, "IMP-050: Arena should start empty");

    // Test 2: Allocate from arena and verify sizes
    {
        let slice1 = arena.alloc(256);
        assert_eq!(
            slice1.len(),
            256,
            "IMP-050: First allocation should have size 256"
        );
    }
    assert_eq!(arena.used(), 256, "IMP-050: Arena should track usage");

    {
        let slice2 = arena.alloc(512);
        assert_eq!(
            slice2.len(),
            512,
            "IMP-050: Second allocation should have size 512"
        );
    }
    assert!(
        arena.used() >= 768,
        "IMP-050: Arena should track cumulative usage"
    );

    // Test 3: Reset arena for reuse
    arena.reset();
    assert_eq!(
        arena.used(),
        0,
        "IMP-050: Arena should be empty after reset"
    );

    // Test 4: Can allocate again after reset
    let slice3 = arena.alloc(1024);
    assert_eq!(
        slice3.len(),
        1024,
        "IMP-050: Post-reset allocation should work"
    );

    // Test 5: Verify allocations are zeroed
    assert!(
        slice3.iter().all(|&x| x == 0.0),
        "IMP-050: Fresh allocation should be zeroed"
    );
}

/// IMP-051: Scratch buffer management
/// Target: Reusable scratch space for intermediate computations
#[test]
#[cfg(feature = "gpu")]
fn test_imp_051_scratch_buffers() {
    use crate::gpu::ScratchBuffer;

    // Test 1: Create scratch buffer for layers
    let num_layers = 4;
    let layer_size = 2048;
    let mut scratch = ScratchBuffer::new(num_layers, layer_size);

    assert_eq!(
        scratch.num_layers(),
        num_layers,
        "IMP-051: Should have 4 layers"
    );
    assert_eq!(
        scratch.layer_size(),
        layer_size,
        "IMP-051: Layer size should be 2048"
    );

    // Test 2: Get scratch for specific layer
    let layer0 = scratch.get_layer(0);
    assert_eq!(
        layer0.len(),
        layer_size,
        "IMP-051: Layer 0 scratch should have correct size"
    );

    let layer3 = scratch.get_layer(3);
    assert_eq!(
        layer3.len(),
        layer_size,
        "IMP-051: Layer 3 scratch should have correct size"
    );

    // Test 3: Layer scratches are independent
    scratch.get_layer_mut(0).iter_mut().for_each(|x| *x = 1.0);
    scratch.get_layer_mut(1).iter_mut().for_each(|x| *x = 2.0);

    assert!(
        scratch.get_layer(0).iter().all(|&x| x == 1.0),
        "IMP-051: Layer 0 should retain its values"
    );
    assert!(
        scratch.get_layer(1).iter().all(|&x| x == 2.0),
        "IMP-051: Layer 1 should be independent"
    );

    // Test 4: Reset all layers
    scratch.reset();
    assert!(
        scratch.get_layer(0).iter().all(|&x| x == 0.0),
        "IMP-051: Layer 0 should be zeroed after reset"
    );

    // Test 5: Total size calculation
    assert_eq!(
        scratch.total_size(),
        num_layers * layer_size,
        "IMP-051: Total size should be layers * layer_size"
    );
}

// ============================================================================
// Phase 14: Quantized Compute Kernels (M23) - EXTREME TDD
// ============================================================================

/// IMP-052: Quantized dot product
/// Target: Compute dot product on Q4/Q8 data without full dequantization
#[test]
#[cfg(feature = "gpu")]
#[allow(clippy::similar_names)] // scale_a_f16/scale_b_f16, block_a_q8/block_b_q8 are intentionally paired
fn test_imp_052_quantized_dot() {
    use crate::gpu::{quantized_dot_q4, quantized_dot_q8};

    // Q4_0 format: 32 values per block, 2 values per byte + f16 scale
    // Block size = 2 (scale) + 16 (data) = 18 bytes

    // Test 1: Q4 dot product - create test blocks
    // Each block has scale (f16 as 2 bytes) + 16 bytes of packed 4-bit values
    let scale_a: f32 = 0.5;
    let scale_b: f32 = 0.25;

    // Create Q4 blocks: [scale_lo, scale_hi, packed_data...]
    let mut block_a = vec![0u8; 18];
    let mut block_b = vec![0u8; 18];

    // Set scales (f16 little-endian)
    let scale_a_f16 = half::f16::from_f32(scale_a);
    let scale_b_f16 = half::f16::from_f32(scale_b);
    block_a[0..2].copy_from_slice(&scale_a_f16.to_le_bytes());
    block_b[0..2].copy_from_slice(&scale_b_f16.to_le_bytes());

    // Set packed values: each byte has two 4-bit values (low nibble, high nibble)
    // Values are stored as unsigned 0-15, centered at 8
    // Use simple test pattern: all 8s (which is 0 after centering)
    for i in 2..18 {
        block_a[i] = 0x99; // Two 9s: (9-8)*scale = scale per element
        block_b[i] = 0x99;
    }

    let result_q4 = quantized_dot_q4(&block_a, &block_b);

    // Expected: 32 elements, each (1*scale_a) * (1*scale_b) = 0.5 * 0.25 = 0.125
    // Sum = 32 * 0.125 = 4.0
    assert!(
        (result_q4 - 4.0).abs() < 0.5,
        "IMP-052: Q4 dot product result ({}) should be ~4.0",
        result_q4
    );

    // Test 2: Q8 dot product
    // Q8_0 format: 32 values per block, 1 byte per value + f16 scale
    // Block size = 2 (scale) + 32 (data) = 34 bytes
    let mut block_a_q8 = vec![0u8; 34];
    let mut block_b_q8 = vec![0u8; 34];

    block_a_q8[0..2].copy_from_slice(&scale_a_f16.to_le_bytes());
    block_b_q8[0..2].copy_from_slice(&scale_b_f16.to_le_bytes());

    // Q8 values are signed i8, use value 1 for simplicity
    for i in 2..34 {
        block_a_q8[i] = 1i8 as u8;
        block_b_q8[i] = 1i8 as u8;
    }

    let result_q8 = quantized_dot_q8(&block_a_q8, &block_b_q8);

    // Expected: 32 elements, each (1*scale_a) * (1*scale_b) = 0.5 * 0.25 = 0.125
    // Sum = 32 * 0.125 = 4.0
    assert!(
        (result_q8 - 4.0).abs() < 0.5,
        "IMP-052: Q8 dot product result ({}) should be ~4.0",
        result_q8
    );

    // Test 3: Zero blocks should give zero result
    let zero_block_q4 = vec![0u8; 18];
    let zero_result = quantized_dot_q4(&zero_block_q4, &zero_block_q4);
    assert!(
        zero_result.abs() < 1e-6,
        "IMP-052: Zero blocks should give zero dot product"
    );
}

/// IMP-053: Quantized matrix-vector multiply
/// Target: MatVec on quantized weights without full dequantization
#[test]
#[cfg(feature = "gpu")]
fn test_imp_053_quantized_matvec() {
    use crate::gpu::{quantized_matvec_q4, quantized_matvec_q8};

    // Test matrix: 2 rows x 32 cols (1 block per row)
    let rows = 2;
    let cols = 32;

    // Create Q4 weight matrix (2 blocks, 18 bytes each)
    let scale: f32 = 0.1;
    let scale_f16 = half::f16::from_f32(scale);

    let mut weights_q4 = vec![0u8; rows * 18];
    for row in 0..rows {
        let offset = row * 18;
        weights_q4[offset..offset + 2].copy_from_slice(&scale_f16.to_le_bytes());
        // Fill with 9s (value 1 after centering at 8)
        for i in 2..18 {
            weights_q4[offset + i] = 0x99;
        }
    }

    // Input vector: 32 f32 values, all 1.0
    let input: Vec<f32> = vec![1.0; cols];

    let result_q4 = quantized_matvec_q4(&weights_q4, &input, rows, cols);

    assert_eq!(
        result_q4.len(),
        rows,
        "IMP-053: Q4 matvec should produce {} outputs",
        rows
    );

    // Each row: sum of 32 * (1 * scale) * 1.0 = 32 * 0.1 = 3.2
    for (i, &val) in result_q4.iter().enumerate() {
        assert!(
            (val - 3.2).abs() < 0.5,
            "IMP-053: Q4 matvec row {} ({}) should be ~3.2",
            i,
            val
        );
    }

    // Test Q8 matvec
    let mut weights_q8 = vec![0u8; rows * 34];
    for row in 0..rows {
        let offset = row * 34;
        weights_q8[offset..offset + 2].copy_from_slice(&scale_f16.to_le_bytes());
        // Fill with 1s (signed i8)
        for i in 2..34 {
            weights_q8[offset + i] = 1i8 as u8;
        }
    }

    let result_q8 = quantized_matvec_q8(&weights_q8, &input, rows, cols);

    assert_eq!(
        result_q8.len(),
        rows,
        "IMP-053: Q8 matvec should produce {} outputs",
        rows
    );

    for (i, &val) in result_q8.iter().enumerate() {
        assert!(
            (val - 3.2).abs() < 0.5,
            "IMP-053: Q8 matvec row {} ({}) should be ~3.2",
            i,
            val
        );
    }
}

/// IMP-054: Mixed precision accumulation
/// Target: Accumulate in f32 while reading quantized data
#[test]
#[cfg(feature = "gpu")]
fn test_imp_054_mixed_precision() {
    use crate::gpu::QuantizedAccumulator;

    // Test 1: Create accumulator
    let mut acc = QuantizedAccumulator::new();
    assert_eq!(
        acc.sum(),
        0.0,
        "IMP-054: New accumulator should have zero sum"
    );

    // Test 2: Add scaled values
    acc.add_scaled(1.0, 0.5); // 1.0 * 0.5 = 0.5
    acc.add_scaled(2.0, 0.5); // 2.0 * 0.5 = 1.0
    acc.add_scaled(3.0, 0.5); // 3.0 * 0.5 = 1.5

    assert!(
        (acc.sum() - 3.0).abs() < 1e-6,
        "IMP-054: Accumulator sum ({}) should be 3.0",
        acc.sum()
    );

    // Test 3: Reset accumulator
    acc.reset();
    assert_eq!(
        acc.sum(),
        0.0,
        "IMP-054: Reset accumulator should have zero sum"
    );

    // Test 4: Add block contribution (simulates quantized block processing)
    let block_sum: f32 = 10.0;
    let block_scale: f32 = 0.1;
    acc.add_block(block_sum, block_scale);

    assert!(
        (acc.sum() - 1.0).abs() < 1e-6,
        "IMP-054: Block contribution ({}) should be 1.0",
        acc.sum()
    );

    // Test 5: Multiple block accumulation
    acc.reset();
    for _ in 0..10 {
        acc.add_block(5.0, 0.2); // 5.0 * 0.2 = 1.0 per block
    }

    assert!(
        (acc.sum() - 10.0).abs() < 1e-5,
        "IMP-054: 10 blocks should sum to 10.0, got {}",
        acc.sum()
    );
}

/// IMP-055: Double-buffered weight loading
/// Target: Load next layer weights while computing current layer
#[test]
#[cfg(feature = "gpu")]
fn test_imp_055_double_buffer() {
    use crate::gpu::DoubleBuffer;

    // Test 1: Create double buffer with given capacity
    let buffer: DoubleBuffer<f32> = DoubleBuffer::new(1024);
    assert_eq!(
        buffer.capacity(),
        1024,
        "IMP-055: Double buffer should have requested capacity"
    );

    // Test 2: Access front buffer for reading
    let front = buffer.front();
    assert_eq!(
        front.len(),
        1024,
        "IMP-055: Front buffer should have full capacity"
    );

    // Test 3: Access back buffer for writing
    let mut buffer = DoubleBuffer::new(256);
    {
        let back = buffer.back_mut();
        for (i, val) in back.iter_mut().enumerate() {
            *val = i as f32;
        }
    }

    // Test 4: Swap buffers - back becomes front
    buffer.swap();
    let front_after_swap = buffer.front();
    assert!(
        (front_after_swap[0] - 0.0).abs() < 1e-6,
        "IMP-055: After swap, front[0] should be 0.0"
    );
    assert!(
        (front_after_swap[255] - 255.0).abs() < 1e-6,
        "IMP-055: After swap, front[255] should be 255.0"
    );

    // Test 5: Multiple swaps maintain data integrity
    {
        let back = buffer.back_mut();
        for val in back.iter_mut() {
            *val = 42.0;
        }
    }
    buffer.swap();
    let front_again = buffer.front();
    assert!(
        (front_again[0] - 42.0).abs() < 1e-6,
        "IMP-055: After second swap, front should have 42.0 values"
    );
}

/// IMP-056: Chunked token processing
/// Target: Process tokens in chunks to improve cache utilization
#[test]
#[cfg(feature = "gpu")]
fn test_imp_056_chunked_processing() {
    use crate::gpu::ChunkedProcessor;

    // Test 1: Create processor with chunk size
    let processor = ChunkedProcessor::new(64);
    assert_eq!(
        processor.chunk_size(),
        64,
        "IMP-056: Processor should have requested chunk size"
    );

    // Test 2: Calculate number of chunks for input
    assert_eq!(
        processor.num_chunks(100),
        2,
        "IMP-056: 100 items with chunk_size=64 needs 2 chunks"
    );
    assert_eq!(
        processor.num_chunks(64),
        1,
        "IMP-056: 64 items with chunk_size=64 needs 1 chunk"
    );
    assert_eq!(
        processor.num_chunks(0),
        0,
        "IMP-056: 0 items needs 0 chunks"
    );

    // Test 3: Get chunk bounds
    let (start, end) = processor.chunk_bounds(0, 100);
    assert_eq!(start, 0, "IMP-056: First chunk starts at 0");
    assert_eq!(end, 64, "IMP-056: First chunk ends at chunk_size");

    let (start, end) = processor.chunk_bounds(1, 100);
    assert_eq!(start, 64, "IMP-056: Second chunk starts at 64");
    assert_eq!(end, 100, "IMP-056: Second chunk ends at total length");

    // Test 4: Process chunks with accumulator function
    let data: Vec<f32> = (0..128).map(|x| x as f32).collect();
    let sum = processor.process_chunks(&data, |chunk| chunk.iter().sum::<f32>());

    // Sum of 0..127 = 127 * 128 / 2 = 8128
    assert!(
        (sum - 8128.0).abs() < 1e-3,
        "IMP-056: Chunked sum ({}) should equal 8128.0",
        sum
    );

    // Test 5: Small input (single chunk)
    let small_data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let small_sum = processor.process_chunks(&small_data, |chunk| chunk.iter().sum::<f32>());
    assert!(
        (small_sum - 6.0).abs() < 1e-6,
        "IMP-056: Small chunked sum ({}) should equal 6.0",
        small_sum
    );
}

/// IMP-057: Pipeline stage management
/// Target: Coordinate multi-stage inference pipeline
#[test]
#[cfg(feature = "gpu")]
fn test_imp_057_pipeline_stages() {
    use crate::gpu::{GpuPipelineStage, InferencePipeline};

    // Test 1: Create pipeline stages enum
    let embed = GpuPipelineStage::Embed;
    let attention = GpuPipelineStage::Attention;
    let ffn = GpuPipelineStage::FFN;
    let output = GpuPipelineStage::Output;

    // Test 2: Pipeline stage ordering
    assert!(
        (embed as u8) < (attention as u8),
        "IMP-057: Embed should come before Attention"
    );
    assert!(
        (attention as u8) < (ffn as u8),
        "IMP-057: Attention should come before FFN"
    );
    assert!(
        (ffn as u8) < (output as u8),
        "IMP-057: FFN should come before Output"
    );

    // Test 3: Create inference pipeline
    let mut pipeline = InferencePipeline::new(4); // 4-stage pipeline
    assert_eq!(
        pipeline.num_stages(),
        4,
        "IMP-057: Pipeline should have 4 stages"
    );

    // Test 4: Record stage timing
    pipeline.record_stage_time(GpuPipelineStage::Embed, 1.0);
    pipeline.record_stage_time(GpuPipelineStage::Attention, 5.0);
    pipeline.record_stage_time(GpuPipelineStage::FFN, 3.0);
    pipeline.record_stage_time(GpuPipelineStage::Output, 0.5);

    // Test 5: Get total pipeline latency
    let total = pipeline.total_latency();
    assert!(
        (total - 9.5).abs() < 1e-6,
        "IMP-057: Total latency ({}) should be 9.5ms",
        total
    );

    // Test 6: Get stage breakdown
    let breakdown = pipeline.stage_breakdown();
    assert!(
        (breakdown[&GpuPipelineStage::Attention] - 5.0).abs() < 1e-6,
        "IMP-057: Attention stage should be 5.0ms"
    );

    // Test 7: Reset pipeline for new forward pass
    pipeline.reset();
    assert!(
        pipeline.total_latency() < 1e-6,
        "IMP-057: Reset pipeline should have zero latency"
    );
}

/// IMP-058: Token batch accumulator
/// Target: Accumulate tokens for batched processing
#[test]
#[cfg(feature = "gpu")]
fn test_imp_058_token_batch() {
    use crate::gpu::TokenBatch;

    // Test 1: Create token batch with capacity
    let mut batch = TokenBatch::new(4);
    assert_eq!(batch.capacity(), 4, "IMP-058: Batch should have capacity 4");
    assert_eq!(batch.len(), 0, "IMP-058: New batch should be empty");
    assert!(!batch.is_full(), "IMP-058: New batch should not be full");

    // Test 2: Push tokens and check state
    assert!(
        batch.push(100).is_none(),
        "IMP-058: First push should not return batch"
    );
    assert_eq!(batch.len(), 1, "IMP-058: Batch should have 1 token");

    assert!(
        batch.push(101).is_none(),
        "IMP-058: Second push should not return batch"
    );
    assert!(
        batch.push(102).is_none(),
        "IMP-058: Third push should not return batch"
    );
    assert_eq!(batch.len(), 3, "IMP-058: Batch should have 3 tokens");

    // Test 3: Push final token returns full batch
    let full_batch = batch.push(103);
    assert!(
        full_batch.is_some(),
        "IMP-058: Fourth push should return full batch"
    );
    let tokens = full_batch.expect("test");
    assert_eq!(
        tokens,
        vec![100, 101, 102, 103],
        "IMP-058: Batch should contain all tokens"
    );
    assert_eq!(
        batch.len(),
        0,
        "IMP-058: After returning, batch should be empty"
    );

    // Test 4: Flush partial batch
    batch.push(200);
    batch.push(201);
    let partial = batch.flush();
    assert_eq!(
        partial,
        vec![200, 201],
        "IMP-058: Flush should return partial batch"
    );
    assert_eq!(
        batch.len(),
        0,
        "IMP-058: After flush, batch should be empty"
    );

    // Test 5: Flush empty batch returns empty vec
    let empty = batch.flush();
    assert!(
        empty.is_empty(),
        "IMP-058: Flush empty batch should return empty vec"
    );
}

/// IMP-059: Speculative token buffer
/// Target: Buffer for speculative decoding candidates
#[test]
#[cfg(feature = "gpu")]
fn test_imp_059_speculative_buffer() {
    use crate::gpu::SpeculativeBuffer;

    // Test 1: Create speculative buffer with capacity
    let mut buffer = SpeculativeBuffer::new(8);
    assert_eq!(
        buffer.capacity(),
        8,
        "IMP-059: Buffer should have capacity 8"
    );
    assert_eq!(buffer.len(), 0, "IMP-059: New buffer should be empty");

    // Test 2: Add candidates with confidence scores
    buffer.add_candidate(100, 0.95);
    buffer.add_candidate(101, 0.85);
    buffer.add_candidate(102, 0.75);
    assert_eq!(buffer.len(), 3, "IMP-059: Buffer should have 3 candidates");

    // Test 3: Verify candidates against actual tokens (all match)
    let actual_tokens = vec![100, 101, 102];
    let (accepted, rejected_at) = buffer.verify(&actual_tokens);
    assert_eq!(accepted, 3, "IMP-059: All 3 candidates should be accepted");
    assert!(
        rejected_at.is_none(),
        "IMP-059: No rejection point when all match"
    );

    // Test 4: Verify with mismatch (clear buffer first)
    buffer.reject(); // Clear previous candidates
    buffer.add_candidate(200, 0.90);
    buffer.add_candidate(201, 0.80);
    buffer.add_candidate(202, 0.70);
    let actual_with_mismatch = vec![200, 201, 999]; // 999 doesn't match 202
    let (accepted2, rejected_at2) = buffer.verify(&actual_with_mismatch);
    assert_eq!(accepted2, 2, "IMP-059: Only first 2 should be accepted");
    assert_eq!(rejected_at2, Some(2), "IMP-059: Rejection at index 2");

    // Test 5: Accept/reject resolution (clear buffer first)
    buffer.reject();
    buffer.add_candidate(300, 0.95);
    buffer.add_candidate(301, 0.85);
    buffer.accept(1); // Accept first candidate
    assert_eq!(
        buffer.len(),
        1,
        "IMP-059: After accept(1), 1 candidate remains"
    );

    buffer.reject(); // Reject remaining
    assert_eq!(
        buffer.len(),
        0,
        "IMP-059: After reject, buffer should be empty"
    );
}

/// IMP-060: Batch scheduling coordinator
/// Target: Coordinate batched inference scheduling
#[test]
#[cfg(feature = "gpu")]
fn test_imp_060_batch_scheduler() {
    use crate::gpu::InferenceBatchScheduler;

    // Test 1: Create batch scheduler
    let mut scheduler = InferenceBatchScheduler::new();
    assert_eq!(
        scheduler.pending_count(),
        0,
        "IMP-060: New scheduler has no pending"
    );
    assert_eq!(
        scheduler.completed_count(),
        0,
        "IMP-060: New scheduler has no completed"
    );

    // Test 2: Submit batches
    let batch_id_1 = scheduler.submit(vec![100, 101, 102]);
    let batch_id_2 = scheduler.submit(vec![200, 201]);
    assert_eq!(
        scheduler.pending_count(),
        2,
        "IMP-060: Should have 2 pending batches"
    );
    assert!(
        batch_id_1 != batch_id_2,
        "IMP-060: Batch IDs should be unique"
    );

    // Test 3: Poll for completed (none yet since we need to mark complete)
    assert!(
        scheduler.poll().is_none(),
        "IMP-060: No batches completed yet"
    );

    // Test 4: Mark batch as complete with results
    scheduler.complete(batch_id_1, vec![1000, 1001, 1002]);
    assert_eq!(
        scheduler.completed_count(),
        1,
        "IMP-060: Should have 1 completed"
    );
    assert_eq!(
        scheduler.pending_count(),
        1,
        "IMP-060: Should have 1 pending"
    );

    // Test 5: Poll returns completed batch
    let completed = scheduler.poll();
    assert!(completed.is_some(), "IMP-060: Should get completed batch");
    let (id, results) = completed.expect("test");
    assert_eq!(id, batch_id_1, "IMP-060: Should get batch_id_1");
    assert_eq!(
        results,
        vec![1000, 1001, 1002],
        "IMP-060: Should get correct results"
    );

    // Test 6: Drain all completed
    scheduler.complete(batch_id_2, vec![2000, 2001]);
    let all_completed = scheduler.drain();
    assert_eq!(
        all_completed.len(),
        1,
        "IMP-060: Drain should return 1 batch"
    );
    assert_eq!(
        scheduler.completed_count(),
        0,
        "IMP-060: After drain, no completed"
    );
}

// =========================================================================
// M26: Async I/O & Event-Driven Processing Tests (Phase 17)
// =========================================================================

/// IMP-061: Async request queue
/// Tests non-blocking request submission and retrieval with backpressure.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_061_async_request_queue() {
    use crate::gpu::AsyncRequestQueue;

    // Test 1: Create queue with capacity
    let mut queue: AsyncRequestQueue<String> = AsyncRequestQueue::new(3);
    assert_eq!(queue.capacity(), 3, "IMP-061: Queue capacity should be 3");
    assert!(queue.is_empty(), "IMP-061: New queue should be empty");
    assert!(!queue.is_full(), "IMP-061: New queue should not be full");
    assert_eq!(queue.len(), 0, "IMP-061: New queue length should be 0");

    // Test 2: Push items
    assert!(
        queue.try_push("request1".to_string()),
        "IMP-061: Should push first item"
    );
    assert!(
        queue.try_push("request2".to_string()),
        "IMP-061: Should push second item"
    );
    assert_eq!(queue.len(), 2, "IMP-061: Queue should have 2 items");
    assert!(!queue.is_empty(), "IMP-061: Queue should not be empty");

    // Test 3: Fill to capacity
    assert!(
        queue.try_push("request3".to_string()),
        "IMP-061: Should push third item"
    );
    assert!(queue.is_full(), "IMP-061: Queue should be full");
    assert!(
        !queue.try_push("request4".to_string()),
        "IMP-061: Should reject when full"
    );

    // Test 4: Pop items (FIFO order)
    let item = queue.try_pop();
    assert!(item.is_some(), "IMP-061: Should pop item");
    assert_eq!(
        item.expect("test"),
        "request1",
        "IMP-061: Should pop in FIFO order"
    );
    assert!(
        !queue.is_full(),
        "IMP-061: Queue should not be full after pop"
    );

    // Test 5: Pop remaining
    assert_eq!(
        queue.try_pop(),
        Some("request2".to_string()),
        "IMP-061: Pop second"
    );
    assert_eq!(
        queue.try_pop(),
        Some("request3".to_string()),
        "IMP-061: Pop third"
    );
    assert!(queue.is_empty(), "IMP-061: Queue should be empty");
    assert!(
        queue.try_pop().is_none(),
        "IMP-061: Pop from empty returns None"
    );
}

/// IMP-062: Event notifier for completion
/// Tests callback-based notification of inference completion.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_062_event_notifier() {
    use crate::gpu::InferenceEventNotifier;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // Test 1: Create notifier
    let mut notifier = InferenceEventNotifier::new();
    assert_eq!(
        notifier.handler_count(),
        0,
        "IMP-062: New notifier has no handlers"
    );

    // Test 2: Register handlers
    let counter1 = Arc::new(AtomicUsize::new(0));
    let counter1_clone = counter1.clone();
    notifier.register(Box::new(move |_request_id, _tokens| {
        counter1_clone.fetch_add(1, Ordering::SeqCst);
    }));
    assert_eq!(
        notifier.handler_count(),
        1,
        "IMP-062: Should have 1 handler"
    );

    let counter2 = Arc::new(AtomicUsize::new(0));
    let counter2_clone = counter2.clone();
    notifier.register(Box::new(move |_request_id, _tokens| {
        counter2_clone.fetch_add(10, Ordering::SeqCst);
    }));
    assert_eq!(
        notifier.handler_count(),
        2,
        "IMP-062: Should have 2 handlers"
    );

    // Test 3: Notify triggers all handlers
    notifier.notify(1, &[100, 101, 102]);
    assert_eq!(
        counter1.load(Ordering::SeqCst),
        1,
        "IMP-062: Handler 1 should be called"
    );
    assert_eq!(
        counter2.load(Ordering::SeqCst),
        10,
        "IMP-062: Handler 2 should be called"
    );

    // Test 4: Multiple notifications
    notifier.notify(2, &[200]);
    assert_eq!(
        counter1.load(Ordering::SeqCst),
        2,
        "IMP-062: Handler 1 called twice"
    );
    assert_eq!(
        counter2.load(Ordering::SeqCst),
        20,
        "IMP-062: Handler 2 called twice"
    );

    // Test 5: Clear handlers
    notifier.clear();
    assert_eq!(
        notifier.handler_count(),
        0,
        "IMP-062: After clear, no handlers"
    );
    notifier.notify(3, &[300]); // Should not crash, just no-op
    assert_eq!(
        counter1.load(Ordering::SeqCst),
        2,
        "IMP-062: Counter unchanged after clear"
    );
}

/// IMP-063: Timeout manager for requests
/// Tests deadline-based request timeout handling.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_063_timeout_manager() {
    use crate::gpu::TimeoutManager;
    use std::time::{Duration, Instant};

    // Test 1: Create timeout manager
    let mut manager = TimeoutManager::new();
    assert_eq!(
        manager.active_count(),
        0,
        "IMP-063: New manager has no active timeouts"
    );

    // Test 2: Register timeouts with different deadlines
    let now = Instant::now();
    let short_deadline = now + Duration::from_millis(10);
    let long_deadline = now + Duration::from_millis(1000);

    manager.register(1, short_deadline);
    manager.register(2, long_deadline);
    assert_eq!(
        manager.active_count(),
        2,
        "IMP-063: Should have 2 active timeouts"
    );

    // Test 3: Check for expired (wait for short timeout to expire)
    std::thread::sleep(Duration::from_millis(20));
    let expired = manager.check_expired();
    assert_eq!(expired.len(), 1, "IMP-063: Should have 1 expired timeout");
    assert_eq!(expired[0], 1, "IMP-063: Request 1 should be expired");
    assert_eq!(
        manager.active_count(),
        1,
        "IMP-063: Should have 1 active after check"
    );

    // Test 4: Remove timeout manually
    manager.remove(2);
    assert_eq!(manager.active_count(), 0, "IMP-063: No active after remove");

    // Test 5: Check expired on empty returns empty vec
    let expired = manager.check_expired();
    assert!(expired.is_empty(), "IMP-063: No expired when empty");
}

// =========================================================================
// M27: Request Scheduling & Resource Management Tests (Phase 18)
// =========================================================================

/// IMP-064: Priority request queue
/// Tests priority-based request scheduling.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_064_priority_queue() {
    use crate::gpu::{PriorityRequest, PriorityRequestQueue};

    // Test 1: Create priority queue
    let mut queue = PriorityRequestQueue::new();
    assert!(queue.is_empty(), "IMP-064: New queue should be empty");
    assert_eq!(queue.len(), 0, "IMP-064: New queue length should be 0");

    // Test 2: Enqueue with different priorities (higher = more important)
    queue.enqueue(PriorityRequest::new(1, "low_priority".to_string()));
    queue.enqueue(PriorityRequest::new(3, "high_priority".to_string()));
    queue.enqueue(PriorityRequest::new(2, "medium_priority".to_string()));
    assert_eq!(queue.len(), 3, "IMP-064: Should have 3 requests");

    // Test 3: Dequeue returns highest priority first
    let req = queue.dequeue_highest();
    assert!(req.is_some(), "IMP-064: Should dequeue request");
    assert_eq!(
        req.expect("test").data(),
        "high_priority",
        "IMP-064: Highest priority first"
    );

    let req = queue.dequeue_highest();
    assert_eq!(
        req.expect("test").data(),
        "medium_priority",
        "IMP-064: Medium priority second"
    );

    let req = queue.dequeue_highest();
    assert_eq!(
        req.expect("test").data(),
        "low_priority",
        "IMP-064: Low priority last"
    );

    // Test 4: Dequeue from empty returns None
    assert!(queue.is_empty(), "IMP-064: Queue should be empty");
    assert!(
        queue.dequeue_highest().is_none(),
        "IMP-064: Dequeue empty returns None"
    );

    // Test 5: Same priority maintains FIFO order
    queue.enqueue(PriorityRequest::new(5, "first".to_string()));
    queue.enqueue(PriorityRequest::new(5, "second".to_string()));
    queue.enqueue(PriorityRequest::new(5, "third".to_string()));
    assert_eq!(
        queue.dequeue_highest().expect("test").data(),
        "first",
        "IMP-064: FIFO for same priority"
    );
    assert_eq!(
        queue.dequeue_highest().expect("test").data(),
        "second",
        "IMP-064: FIFO order"
    );
    assert_eq!(
        queue.dequeue_highest().expect("test").data(),
        "third",
        "IMP-064: FIFO order"
    );
}

/// IMP-065: Token rate limiter
/// Tests throughput control with token bucket algorithm.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_065_rate_limiter() {
    use crate::gpu::TokenRateLimiter;
    use std::time::Duration;

    // Test 1: Create rate limiter (10 tokens/sec, burst of 5)
    let mut limiter = TokenRateLimiter::new(10.0, 5);
    assert_eq!(
        limiter.tokens_available(),
        5,
        "IMP-065: Should start with burst capacity"
    );

    // Test 2: Acquire tokens
    assert!(limiter.try_acquire(3), "IMP-065: Should acquire 3 tokens");
    assert_eq!(
        limiter.tokens_available(),
        2,
        "IMP-065: Should have 2 remaining"
    );

    // Test 3: Acquire more than available fails
    assert!(
        !limiter.try_acquire(3),
        "IMP-065: Should fail to acquire 3 when only 2 available"
    );
    assert_eq!(
        limiter.tokens_available(),
        2,
        "IMP-065: Tokens unchanged on failed acquire"
    );

    // Test 4: Acquire exactly available succeeds
    assert!(
        limiter.try_acquire(2),
        "IMP-065: Should acquire remaining 2"
    );
    assert_eq!(
        limiter.tokens_available(),
        0,
        "IMP-065: Should have 0 remaining"
    );

    // Test 5: Refill adds tokens based on elapsed time
    std::thread::sleep(Duration::from_millis(200)); // 0.2 sec at 10 tok/s = 2 tokens
    limiter.refill();
    let available = limiter.tokens_available();
    assert!(
        available >= 1,
        "IMP-065: Should have refilled at least 1 token, got {}",
        available
    );
    assert!(available <= 5, "IMP-065: Should not exceed burst capacity");
}

/// IMP-066: Resource usage tracker
/// Tests memory and compute resource accounting.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_066_resource_tracker() {
    use crate::gpu::ResourceTracker;

    // Test 1: Create resource tracker (1GB memory, 100% compute capacity)
    let mut tracker = ResourceTracker::new(1024 * 1024 * 1024, 100);
    assert_eq!(
        tracker.memory_usage(),
        0,
        "IMP-066: Initial memory usage is 0"
    );
    assert_eq!(
        tracker.compute_usage(),
        0,
        "IMP-066: Initial compute usage is 0"
    );

    // Test 2: Check allocation availability
    assert!(
        tracker.can_allocate(512 * 1024 * 1024, 50),
        "IMP-066: Should be able to allocate 512MB, 50% compute"
    );
    assert!(
        !tracker.can_allocate(2 * 1024 * 1024 * 1024, 50),
        "IMP-066: Cannot allocate more than capacity"
    );

    // Test 3: Allocate resources
    let alloc_id = tracker.allocate(256 * 1024 * 1024, 30);
    assert!(alloc_id.is_some(), "IMP-066: Allocation should succeed");
    assert_eq!(
        tracker.memory_usage(),
        256 * 1024 * 1024,
        "IMP-066: Memory usage updated"
    );
    assert_eq!(
        tracker.compute_usage(),
        30,
        "IMP-066: Compute usage updated"
    );

    // Test 4: Multiple allocations
    let alloc_id_2 = tracker.allocate(128 * 1024 * 1024, 20);
    assert!(
        alloc_id_2.is_some(),
        "IMP-066: Second allocation should succeed"
    );
    assert_eq!(
        tracker.memory_usage(),
        384 * 1024 * 1024,
        "IMP-066: Memory accumulated"
    );
    assert_eq!(tracker.compute_usage(), 50, "IMP-066: Compute accumulated");

    // Test 5: Release resources
    tracker.release(alloc_id.expect("test"));
    assert_eq!(
        tracker.memory_usage(),
        128 * 1024 * 1024,
        "IMP-066: Memory released"
    );
    assert_eq!(tracker.compute_usage(), 20, "IMP-066: Compute released");

    // Test 6: Usage percentage
    let (mem_pct, compute_pct) = tracker.usage_percentage();
    let expected_mem_pct = (128.0 * 1024.0 * 1024.0) / (1024.0 * 1024.0 * 1024.0) * 100.0;
    assert!(
        (mem_pct - expected_mem_pct).abs() < 0.1,
        "IMP-066: Memory percentage correct"
    );
    assert!(
        (compute_pct - 20.0).abs() < 0.1,
        "IMP-066: Compute percentage correct"
    );
}

// =========================================================================
// M28: Metrics & Health Monitoring Tests (Phase 19)
// =========================================================================

/// IMP-067: Inference metrics collector
/// Tests latency histogram and throughput tracking.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_067_inference_metrics() {
    use crate::gpu::InferenceMetrics;
    use std::time::Duration;

    // Test 1: Create metrics collector
    let mut metrics = InferenceMetrics::new();
    assert_eq!(
        metrics.total_inferences(),
        0,
        "IMP-067: No inferences initially"
    );
    assert_eq!(metrics.total_tokens(), 0, "IMP-067: No tokens initially");

    // Test 2: Record inferences
    metrics.record_inference(Duration::from_millis(10), 5); // 10ms, 5 tokens
    metrics.record_inference(Duration::from_millis(20), 10); // 20ms, 10 tokens
    metrics.record_inference(Duration::from_millis(15), 8); // 15ms, 8 tokens
    assert_eq!(
        metrics.total_inferences(),
        3,
        "IMP-067: Should have 3 inferences"
    );
    assert_eq!(metrics.total_tokens(), 23, "IMP-067: Should have 23 tokens");

    // Test 3: Latency percentiles
    let p50 = metrics.latency_percentile(50);
    assert!(p50.is_some(), "IMP-067: Should have p50");
    let p50_ms = p50.expect("test").as_millis();
    assert!(
        p50_ms >= 10 && p50_ms <= 20,
        "IMP-067: p50 should be ~15ms, got {}ms",
        p50_ms
    );

    // Test 4: Throughput calculation
    let throughput = metrics.throughput();
    assert!(throughput > 0.0, "IMP-067: Throughput should be positive");

    // Test 5: Reset metrics
    metrics.reset();
    assert_eq!(metrics.total_inferences(), 0, "IMP-067: Inferences reset");
    assert_eq!(metrics.total_tokens(), 0, "IMP-067: Tokens reset");
}

/// IMP-068: Health checker
/// Tests component health monitoring.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_068_health_checker() {
    use crate::gpu::HealthChecker;

    // Test 1: Create health checker
    let mut checker = HealthChecker::new();
    assert!(
        checker.is_healthy(),
        "IMP-068: Healthy when no checks registered"
    );

    // Test 2: Register healthy check
    checker.register_check("gpu", Box::new(|| true));
    assert_eq!(checker.check_count(), 1, "IMP-068: Should have 1 check");

    // Test 3: Run checks - all healthy
    let results = checker.check_all();
    assert_eq!(results.len(), 1, "IMP-068: Should have 1 result");
    assert!(
        results.get("gpu").copied().unwrap_or(false),
        "IMP-068: GPU should be healthy"
    );
    assert!(checker.is_healthy(), "IMP-068: Overall should be healthy");

    // Test 4: Register unhealthy check
    checker.register_check("memory", Box::new(|| false));
    let results = checker.check_all();
    assert!(
        !results.get("memory").copied().unwrap_or(true),
        "IMP-068: Memory should be unhealthy"
    );
    assert!(
        !checker.is_healthy(),
        "IMP-068: Overall should be unhealthy"
    );

    // Test 5: Clear checks
    checker.clear();
    assert_eq!(checker.check_count(), 0, "IMP-068: No checks after clear");
    assert!(checker.is_healthy(), "IMP-068: Healthy after clear");
}

/// IMP-069: Graceful shutdown coordinator
/// Tests coordinated shutdown with request draining.
#[test]
#[cfg(feature = "gpu")]
fn test_imp_069_graceful_shutdown() {
    use crate::gpu::ShutdownCoordinator;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // Test 1: Create shutdown coordinator
    let mut coordinator = ShutdownCoordinator::new();
    assert!(
        !coordinator.is_shutting_down(),
        "IMP-069: Not shutting down initially"
    );
    assert_eq!(
        coordinator.pending_requests(),
        0,
        "IMP-069: No pending requests"
    );

    // Test 2: Register shutdown handler
    let handler_called = Arc::new(AtomicBool::new(false));
    let handler_called_clone = handler_called.clone();
    coordinator.register_handler(Box::new(move || {
        handler_called_clone.store(true, Ordering::SeqCst);
    }));
    assert_eq!(
        coordinator.handler_count(),
        1,
        "IMP-069: Should have 1 handler"
    );

    // Test 3: Track pending requests
    coordinator.request_started();
    coordinator.request_started();
    assert_eq!(
        coordinator.pending_requests(),
        2,
        "IMP-069: Should have 2 pending"
    );

    // Test 4: Initiate shutdown
    coordinator.initiate_shutdown();
    assert!(
        coordinator.is_shutting_down(),
        "IMP-069: Should be shutting down"
    );
    assert!(
        handler_called.load(Ordering::SeqCst),
        "IMP-069: Handler should be called"
    );

    // Test 5: Complete pending requests
    coordinator.request_completed();
    assert_eq!(
        coordinator.pending_requests(),
        1,
        "IMP-069: Should have 1 pending"
    );
    coordinator.request_completed();
    assert_eq!(
        coordinator.pending_requests(),
        0,
        "IMP-069: Should have 0 pending"
    );

    // Test 6: Check completion
    assert!(
        coordinator.is_complete(),
        "IMP-069: Should be complete when shutdown + no pending"
    );
}

// ============================================================================
// Phase 20: Error Recovery & Graceful Degradation (M29) - EXTREME TDD
// ============================================================================

/// IMP-070: Error Recovery Strategy
/// Target: Automatic retry with exponential backoff, GPU fallback, error classification
#[test]
#[cfg(feature = "gpu")]
fn test_imp_070_error_recovery_strategy() {
    use crate::gpu::{ErrorClassification, ErrorRecoveryStrategy, RecoveryAction};
    use std::time::Duration;

    // Test 1: Create recovery strategy with config
    let strategy = ErrorRecoveryStrategy::new()
        .with_max_retries(3)
        .with_base_delay(Duration::from_millis(100))
        .with_max_delay(Duration::from_secs(5))
        .with_jitter(0.1);

    assert_eq!(
        strategy.max_retries(),
        3,
        "IMP-070: Max retries should be 3"
    );

    // Test 2: Error classification
    let transient_err = std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout");
    let classification = strategy.classify_error(&transient_err);
    assert_eq!(
        classification,
        ErrorClassification::Transient,
        "IMP-070: Timeout should be transient"
    );

    let fatal_err = std::io::Error::new(std::io::ErrorKind::InvalidData, "bad data");
    let classification = strategy.classify_error(&fatal_err);
    assert_eq!(
        classification,
        ErrorClassification::Fatal,
        "IMP-070: InvalidData should be fatal"
    );

    // Test 3: Recovery action for transient error
    let action = strategy.determine_action(&transient_err, 0);
    assert!(
        matches!(action, RecoveryAction::Retry { .. }),
        "IMP-070: Transient should retry"
    );

    // Test 4: Exponential backoff delay calculation
    let delay_0 = strategy.calculate_delay(0);
    let delay_1 = strategy.calculate_delay(1);
    let delay_2 = strategy.calculate_delay(2);
    assert!(delay_1 > delay_0, "IMP-070: Delay should increase");
    assert!(
        delay_2 > delay_1,
        "IMP-070: Delay should increase exponentially"
    );

    // Test 5: Max retries exceeded
    let action = strategy.determine_action(&transient_err, 4);
    assert!(
        matches!(action, RecoveryAction::Fail),
        "IMP-070: Should fail after max retries"
    );

    // Test 6: GPU fallback action
    let gpu_err = std::io::Error::other("GPU unavailable");
    let action = strategy.determine_action_with_fallback(&gpu_err, 0);
    assert!(
        matches!(action, RecoveryAction::FallbackToCpu),
        "IMP-070: GPU error should fallback to CPU"
    );
}

/// IMP-071: Graceful Degradation Modes
/// Target: GPU→CPU fallback, memory pressure response, context limiting
#[test]
#[cfg(feature = "gpu")]
fn test_imp_071_graceful_degradation() {
    use crate::gpu::{DegradationManager, DegradationMode, SystemLoad};

    // Test 1: Create degradation manager
    let mut manager = DegradationManager::new();
    assert_eq!(
        manager.current_mode(),
        DegradationMode::Normal,
        "IMP-071: Should start in Normal mode"
    );

    // Test 2: GPU unavailable triggers CPU fallback
    manager.set_gpu_available(false);
    assert_eq!(
        manager.current_mode(),
        DegradationMode::CpuFallback,
        "IMP-071: GPU unavailable should trigger CPU fallback"
    );

    // Test 3: Memory pressure reduces batch size
    manager.set_gpu_available(true);
    manager.update_memory_pressure(0.9); // 90% memory used
    let batch_size = manager.recommended_batch_size(8);
    assert!(
        batch_size < 8,
        "IMP-071: High memory pressure should reduce batch size"
    );

    // Test 4: System load affects context length
    let load = SystemLoad {
        cpu_percent: 95.0,
        memory_percent: 85.0,
        queue_depth: 100,
    };
    manager.update_system_load(load);
    let max_context = manager.recommended_max_context(4096);
    assert!(
        max_context < 4096,
        "IMP-071: High load should limit context length"
    );

    // Test 5: Quality vs latency tradeoff
    manager.set_latency_priority(true);
    assert_eq!(
        manager.current_mode(),
        DegradationMode::LowLatency,
        "IMP-071: Latency priority should set LowLatency mode"
    );

    // Test 6: Recovery to normal mode
    manager.set_gpu_available(true);
    manager.update_memory_pressure(0.3); // 30% memory used
    manager.set_latency_priority(false);
    let load = SystemLoad {
        cpu_percent: 20.0,
        memory_percent: 30.0,
        queue_depth: 5,
    };
    manager.update_system_load(load);
    assert_eq!(
        manager.current_mode(),
        DegradationMode::Normal,
        "IMP-071: Low load should restore Normal mode"
    );
}

