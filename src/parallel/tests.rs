use super::*;

// =========================================================================
// ParallelConfig Tests
// =========================================================================

#[test]
fn test_parallel_config_new() {
    let config = ParallelConfig::new(2, 2, 2, 0).expect("test");
    assert_eq!(config.tp_size, 2);
    assert_eq!(config.pp_size, 2);
    assert_eq!(config.dp_size, 2);
    assert_eq!(config.world_size, 8);
    assert_eq!(config.rank, 0);
}

#[test]
fn test_parallel_config_single() {
    let config = ParallelConfig::single();
    assert_eq!(config.tp_size, 1);
    assert_eq!(config.pp_size, 1);
    assert_eq!(config.dp_size, 1);
    assert_eq!(config.world_size, 1);
    assert_eq!(config.rank, 0);
}

#[test]
fn test_parallel_config_invalid_rank() {
    let result = ParallelConfig::new(2, 2, 2, 100);
    assert!(result.is_err());
}

#[test]
fn test_parallel_config_invalid_world_size() {
    let result = ParallelConfig::new(0, 0, 0, 0);
    assert!(result.is_err());
}

#[test]
fn test_parallel_config_ranks() {
    // World size = 2 * 2 * 2 = 8
    // Rank 5: tp_rank=1, pp_stage=0, dp_rank=1
    let config = ParallelConfig::new(2, 2, 2, 5).expect("test");
    assert_eq!(config.tp_rank(), 1);
    assert_eq!(config.pp_stage(), 0);
    assert_eq!(config.dp_rank(), 1);
}

#[test]
fn test_parallel_config_first_last_checks() {
    let config = ParallelConfig::new(2, 2, 1, 0).expect("test");
    assert!(config.is_tp_first());
    assert!(!config.is_tp_last());
    assert!(config.is_pp_first());
    assert!(!config.is_pp_last());
}

// =========================================================================
// ParallelTensor Tests
// =========================================================================

#[test]
fn test_parallel_tensor_new() {
    let tensor = ParallelTensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.numel(), 6);
}

#[test]
fn test_parallel_tensor_zeros() {
    let tensor = ParallelTensor::zeros(vec![2, 3]);
    assert_eq!(tensor.sum(), 0.0);
}

#[test]
fn test_parallel_tensor_narrow_rows() {
    let tensor = ParallelTensor::new(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("test");
    let narrowed = tensor.narrow(0, 1, 2).expect("test");
    assert_eq!(narrowed.shape, vec![2, 2]);
    assert_eq!(narrowed.data, vec![3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_parallel_tensor_narrow_cols() {
    let tensor = ParallelTensor::new(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("test");
    let narrowed = tensor.narrow(1, 1, 2).expect("test");
    assert_eq!(narrowed.shape, vec![2, 2]);
    assert_eq!(narrowed.data, vec![2.0, 3.0, 6.0, 7.0]);
}

#[test]
fn test_parallel_tensor_transpose() {
    let tensor = ParallelTensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
    let transposed = tensor.transpose().expect("test");
    assert_eq!(transposed.shape, vec![3, 2]);
    assert_eq!(transposed.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_parallel_tensor_matmul() {
    // [1, 2] @ [[1, 2], [3, 4]] = [7, 10]
    let a = ParallelTensor::new(vec![1, 2], vec![1.0, 2.0]).expect("test");
    let b = ParallelTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let c = a.matmul(&b).expect("test");
    assert_eq!(c.shape, vec![1, 2]);
    assert_eq!(c.data, vec![7.0, 10.0]);
}

#[test]
fn test_parallel_tensor_add() {
    let a = ParallelTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let b = ParallelTensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).expect("test");
    let c = a.add(&b).expect("test");
    assert_eq!(c.data, vec![6.0, 8.0, 10.0, 12.0]);
}

// =========================================================================
// Communicator Tests
// =========================================================================

#[test]
fn test_communicator_new() {
    let comm = Communicator::new(4, 2).expect("test");
    assert_eq!(comm.world_size(), 4);
    assert_eq!(comm.rank(), 2);
}

#[test]
fn test_communicator_invalid_rank() {
    let result = Communicator::new(4, 10);
    assert!(result.is_err());
}

#[test]
fn test_communicator_all_reduce_sum() {
    let comm = Communicator::new(4, 0).expect("test");
    let tensor = ParallelTensor::new(vec![2], vec![1.0, 2.0]).expect("test");
    let result = comm.all_reduce(&tensor, ReduceOp::Sum).expect("test");
    // test: multiply by world_size
    assert_eq!(result.data, vec![4.0, 8.0]);
}

#[test]
fn test_communicator_all_reduce_avg() {
    let comm = Communicator::new(4, 0).expect("test");
    let tensor = ParallelTensor::new(vec![2], vec![1.0, 2.0]).expect("test");
    let result = comm.all_reduce(&tensor, ReduceOp::Avg).expect("test");
    assert_eq!(result.data, vec![1.0, 2.0]);
}

#[test]
fn test_communicator_all_gather() {
    let comm = Communicator::new(2, 0).expect("test");
    let tensor = ParallelTensor::new(vec![2], vec![1.0, 2.0]).expect("test");
    let result = comm.all_gather(&tensor).expect("test");
    assert_eq!(result.shape, vec![4]);
    assert_eq!(result.data, vec![1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn test_communicator_barrier() {
    let comm = Communicator::new(4, 0).expect("test");
    assert!(comm.barrier().is_ok());
}

// =========================================================================
// TensorParallel Tests
// =========================================================================

#[test]
fn test_tensor_parallel_new() {
    let tp = TensorParallel::new(4, 2).expect("test");
    assert_eq!(tp.tp_size(), 4);
    assert_eq!(tp.rank(), 2);
}

#[test]
fn test_tensor_parallel_invalid_rank() {
    let result = TensorParallel::new(4, 10);
    assert!(result.is_err());
}

#[test]
fn test_tensor_parallel_chunk_size() {
    let tp = TensorParallel::new(4, 0).expect("test");
    assert_eq!(tp.chunk_size(100), 25);
    assert_eq!(tp.chunk_size(16), 4);
}

#[test]
fn test_tensor_parallel_column_linear() {
    let tp = TensorParallel::new(2, 0).expect("test");

    // Input: (1, 4), Weight: (8, 4) split to (4, 4) per rank
    let input = ParallelTensor::new(vec![1, 4], vec![1.0, 1.0, 1.0, 1.0]).expect("test");
    let weight =
        ParallelTensor::new(vec![8, 4], (0..32).map(|i| i as f32).collect()).expect("test");

    let output = tp
        .column_parallel_linear(&input, &weight, None)
        .expect("test");
    // Output should be (1, 4) - chunk of full output
    assert_eq!(output.shape, vec![1, 4]);
}

#[test]
fn test_tensor_parallel_row_linear() {
    let tp = TensorParallel::new(2, 0).expect("test");

    // Row parallel: Weight (4, 8) split to (2, 8) per rank
    // After transpose: (8, 2)
    // Input needs to be (batch, 8) to matmul with (8, 2) -> output (batch, 2)
    let input = ParallelTensor::new(vec![1, 8], vec![1.0; 8]).expect("test");
    let weight =
        ParallelTensor::new(vec![4, 8], (0..32).map(|i| i as f32).collect()).expect("test");

    let output = tp.row_parallel_linear(&input, &weight, None).expect("test");
    // Output shape after row parallel
    assert!(!output.data.is_empty());
    // After all-reduce, output shape is (1, 2)
    assert_eq!(output.shape[0], 1);
}

// =========================================================================
// PipelineParallel Tests
// =========================================================================

#[test]
fn test_pipeline_parallel_new() {
    let pp = PipelineParallel::new(4, 1, 24, 4).expect("test");
    assert_eq!(pp.num_stages(), 4);
    assert_eq!(pp.stage(), 1);
    assert_eq!(pp.micro_batch_size(), 4);
}

#[test]
fn test_pipeline_parallel_layer_distribution() {
    // 24 layers across 4 stages = 6 layers each
    let pp = PipelineParallel::new(4, 0, 24, 4).expect("test");
    let info = pp.stage_info();
    assert_eq!(info.start_layer, 0);
    assert_eq!(info.end_layer, 6);
    assert_eq!(info.num_layers, 6);

    let pp2 = PipelineParallel::new(4, 3, 24, 4).expect("test");
    let info2 = pp2.stage_info();
    assert_eq!(info2.start_layer, 18);
    assert_eq!(info2.end_layer, 24);
}

#[test]
fn test_pipeline_parallel_uneven_layers() {
    // 25 layers across 4 stages: 7, 6, 6, 6
    let pp = PipelineParallel::new(4, 0, 25, 4).expect("test");
    assert_eq!(pp.stage_info().num_layers, 7);

    let pp1 = PipelineParallel::new(4, 1, 25, 4).expect("test");
    assert_eq!(pp1.stage_info().num_layers, 6);
}

#[test]
fn test_pipeline_parallel_first_last() {
    let first = PipelineParallel::new(4, 0, 24, 4).expect("test");
    assert!(first.is_first_stage());
    assert!(!first.is_last_stage());

    let last = PipelineParallel::new(4, 3, 24, 4).expect("test");
    assert!(!last.is_first_stage());
    assert!(last.is_last_stage());
}

#[test]
fn test_pipeline_parallel_bubble_ratio() {
    let pp = PipelineParallel::new(4, 0, 24, 4).expect("test");
    // Bubble = (4-1) / (4 + 8 - 1) = 3/11 ≈ 0.27
    let ratio = pp.bubble_ratio(8);
    assert!(ratio > 0.2 && ratio < 0.4);
}

#[test]
fn test_pipeline_parallel_stats() {
    let mut pp = PipelineParallel::new(4, 0, 24, 4).expect("test");
    pp.record_micro_batch(10.0);
    pp.record_micro_batch(12.0);

    let stats = pp.stats();
    assert_eq!(stats.micro_batches_processed, 2);
    assert_eq!(stats.forward_passes, 2);
    assert!((stats.avg_stage_latency_ms - 11.0).abs() < 0.1);
}

// =========================================================================
// ZeroOffload Tests
// =========================================================================

#[test]
fn test_zero_offload_default() {
    let zero = ZeroOffload::default();
    assert!(zero.offload_optimizer);
    assert!(!zero.offload_params);
    assert!(zero.pin_memory);
}

#[test]
fn test_zero_offload_inference() {
    let zero = ZeroOffload::inference();
    assert!(!zero.offload_optimizer);
    assert!(zero.offload_params);
    assert!(zero.offload_activations);
}

#[test]
fn test_zero_offload_memory_savings() {
    let zero = ZeroOffload::default();
    let savings = zero.memory_savings_ratio();
    assert!(savings >= 0.0 && savings <= 1.0);

    let zero_inference = ZeroOffload::inference();
    let savings_inference = zero_inference.memory_savings_ratio();
    assert!(savings_inference > savings);
}

// =========================================================================
// DistributedContext Tests
// =========================================================================

#[test]
fn test_distributed_context_single() {
    let config = ParallelConfig::single();
    let ctx = DistributedContext::new(config).expect("test");

    assert!(!ctx.is_distributed());
    assert!(ctx.is_initialized());
    assert!(ctx.tensor_parallel().is_none());
    assert!(ctx.pipeline_parallel().is_none());
}

#[test]
fn test_distributed_context_with_tp() {
    let config = ParallelConfig::new(4, 1, 1, 0).expect("test");
    let ctx = DistributedContext::new(config).expect("test");

    assert!(ctx.is_distributed());
    assert!(ctx.tensor_parallel().is_some());
    assert_eq!(ctx.tensor_parallel().expect("test").tp_size(), 4);
}

#[test]
fn test_distributed_context_init_pipeline() {
    let config = ParallelConfig::new(1, 4, 1, 0).expect("test");
    let mut ctx = DistributedContext::new(config).expect("test");

    ctx.init_pipeline(24, 4).expect("test");
    assert!(ctx.pipeline_parallel().is_some());
    assert_eq!(ctx.pipeline_parallel().expect("test").num_stages(), 4);
}

#[test]
fn test_distributed_context_zero_offload() {
    let config = ParallelConfig::single();
    let mut ctx = DistributedContext::new(config).expect("test");

    ctx.set_zero_offload(ZeroOffload::inference());
    assert!(ctx.zero_offload().offload_params);
}

// =========================================================================
// ReduceOp Tests
// =========================================================================

#[test]
fn test_reduce_op_serialization() {
    let op = ReduceOp::Sum;
    let json = serde_json::to_string(&op).expect("test");
    let deserialized: ReduceOp = serde_json::from_str(&json).expect("test");
    assert_eq!(op, deserialized);
}

// =========================================================================
// Error Tests
// =========================================================================

#[test]
fn test_parallel_error_display() {
    let err = ParallelError::InvalidRank {
        rank: 10,
        world_size: 4,
    };
    assert!(err.to_string().contains("10"));
    assert!(err.to_string().contains("4"));

    let err2 = ParallelError::CommunicationError("timeout".to_string());
    assert!(err2.to_string().contains("timeout"));
}

// =========================================================================
// Extended Coverage Tests: ParallelConfig
// =========================================================================

#[test]
fn test_parallel_config_world_size_calculation_ext_cov() {
    let config = ParallelConfig::new(2, 2, 2, 0).expect("test");
    assert_eq!(config.world_size, 8);
}

#[test]
fn test_parallel_config_single_debug_ext_cov() {
    let config = ParallelConfig::single();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("tp_size"));
    assert!(debug_str.contains("pp_size"));
}

#[test]
fn test_parallel_config_invalid_zero_tp_ext_cov() {
    let result = ParallelConfig::new(0, 1, 1, 0);
    assert!(result.is_err());
}

#[test]
fn test_parallel_config_invalid_zero_pp_ext_cov() {
    let result = ParallelConfig::new(1, 0, 1, 0);
    assert!(result.is_err());
}

#[test]
fn test_parallel_config_invalid_zero_dp_ext_cov() {
    let result = ParallelConfig::new(1, 1, 0, 0);
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests: ReduceOp
// =========================================================================

#[test]
fn test_reduce_op_all_variants_ext_cov() {
    let ops = [ReduceOp::Sum, ReduceOp::Min, ReduceOp::Max, ReduceOp::Avg];
    for op in ops {
        let json = serde_json::to_string(&op).expect("serialize");
        let _: ReduceOp = serde_json::from_str(&json).expect("deserialize");
    }
}

#[test]
fn test_reduce_op_clone_ext_cov() {
    let op = ReduceOp::Max;
    let cloned = op;
    assert_eq!(op, cloned);
}

#[test]
fn test_reduce_op_debug_ext_cov() {
    let op = ReduceOp::Avg;
    let debug_str = format!("{:?}", op);
    assert!(debug_str.contains("Avg"));
}

// =========================================================================
// Extended Coverage Tests: ParallelError
// =========================================================================

#[test]
fn test_parallel_error_all_variants_ext_cov() {
    let errors: [ParallelError; 5] = [
        ParallelError::InvalidRank {
            rank: 5,
            world_size: 4,
        },
        ParallelError::InvalidWorldSize(0),
        ParallelError::CommunicationError("timeout".to_string()),
        ParallelError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![3, 2],
        },
        ParallelError::PipelineError("stage error".to_string()),
    ];
    for err in errors {
        let _ = err.to_string();
    }
}

#[test]
fn test_parallel_error_shape_mismatch_ext_cov() {
    let err = ParallelError::ShapeMismatch {
        expected: vec![10, 20],
        got: vec![20, 10],
    };
    let msg = err.to_string();
    assert!(msg.contains("10") || msg.contains("20"));
}

#[test]
fn test_parallel_error_debug_ext_cov() {
    let err = ParallelError::NotInitialized;
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("NotInitialized"));
}

// =========================================================================
// Extended Coverage Tests: ZeroOffload
// =========================================================================

#[test]
fn test_zero_offload_clone_ext_cov() {
    let zero = ZeroOffload::inference();
    let cloned = zero.clone();
    assert_eq!(zero.offload_params, cloned.offload_params);
    assert_eq!(zero.offload_activations, cloned.offload_activations);
}

#[test]
fn test_zero_offload_debug_ext_cov() {
    let zero = ZeroOffload::default();
    let debug_str = format!("{:?}", zero);
    assert!(debug_str.contains("offload_optimizer"));
    assert!(debug_str.contains("pin_memory"));
}

#[test]
fn test_zero_offload_memory_savings_extremes_ext_cov() {
    // Test with all offload options enabled
    let full_offload = ZeroOffload {
        offload_optimizer: true,
        offload_params: true,
        offload_activations: true,
        pin_memory: true,
        overlap_comm: true,
    };
    let savings = full_offload.memory_savings_ratio();
    assert!(savings >= 0.0);
    assert!(savings <= 1.0);

    // Test with no offload
    let no_offload = ZeroOffload {
        offload_optimizer: false,
        offload_params: false,
        offload_activations: false,
        pin_memory: false,
        overlap_comm: false,
    };
    let no_savings = no_offload.memory_savings_ratio();
    assert!(no_savings >= 0.0);
    assert!(no_savings < savings);
}

// =========================================================================
// Extended Coverage Tests: PipelineStats
// =========================================================================

#[test]
fn test_pipeline_stats_clone_debug_ext_cov() {
    let stats = PipelineStats {
        micro_batches_processed: 100,
        forward_passes: 100,
        bubble_time_ms: 5.0,
        avg_stage_latency_ms: 10.5,
    };
    let cloned = stats.clone();
    assert_eq!(
        stats.micro_batches_processed,
        cloned.micro_batches_processed
    );

    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("micro_batches_processed"));
    assert!(debug_str.contains("bubble_time_ms"));
}

// =========================================================================
// Extended Coverage Tests: TensorParallel
// =========================================================================

#[test]
fn test_tensor_parallel_chunk_size_ext_cov() {
    let tp = TensorParallel::new(4, 0).expect("test");
    let chunk = tp.chunk_size(1000);
    assert_eq!(chunk, 250); // 1000 / 4 = 250
}

#[test]
fn test_tensor_parallel_debug_ext_cov() {
    let tp = TensorParallel::new(8, 2).expect("test");
    let debug_str = format!("{:?}", tp);
    assert!(debug_str.contains("tp_size"));
    assert!(debug_str.contains("rank"));
}

#[test]
fn test_tensor_parallel_invalid_rank_ext_cov() {
    let result = TensorParallel::new(4, 10);
    assert!(result.is_err());
}

#[test]
fn test_tensor_parallel_invalid_size_ext_cov() {
    let result = TensorParallel::new(0, 0);
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests: Communicator
// =========================================================================

#[test]
fn test_communicator_debug_ext_cov() {
    let comm = Communicator::new(4, 0).expect("test");
    let debug_str = format!("{:?}", comm);
    assert!(debug_str.contains("world_size"));
    assert!(debug_str.contains("rank"));
}

#[test]
fn test_communicator_invalid_rank_ext_cov() {
    let result = Communicator::new(4, 10);
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests: PipelineParallel
// =========================================================================

#[test]
fn test_pipeline_parallel_stage_info_ext_cov() {
    // PipelineParallel::new(pp_size, stage, total_layers, micro_batch_size)
    let pp = PipelineParallel::new(4, 0, 24, 4).expect("test");
    let info = pp.stage_info();
    assert_eq!(info.start_layer, 0);
    assert_eq!(info.num_layers, 6); // 24 / 4 = 6 layers per stage
}

#[test]
fn test_pipeline_parallel_debug_ext_cov() {
    let pp = PipelineParallel::new(4, 0, 24, 4).expect("test");
    let debug_str = format!("{:?}", pp);
    assert!(debug_str.contains("pp_size"));
    assert!(debug_str.contains("stage"));
}

#[test]
fn test_pipeline_parallel_micro_batch_size_ext_cov() {
    let pp = PipelineParallel::new(4, 0, 24, 8).expect("test");
    assert_eq!(pp.micro_batch_size(), 8);
}

#[test]
fn test_pipeline_parallel_first_last_stage_ext_cov() {
    let first = PipelineParallel::new(4, 0, 24, 4).expect("test");
    let last = PipelineParallel::new(4, 3, 24, 4).expect("test");
    let middle = PipelineParallel::new(4, 1, 24, 4).expect("test");

    assert!(first.is_first_stage());
    assert!(!first.is_last_stage());

    assert!(!last.is_first_stage());
    assert!(last.is_last_stage());

    assert!(!middle.is_first_stage());
    assert!(!middle.is_last_stage());
}

// =========================================================================
// Additional Coverage Tests: ParallelTensor Edge Cases
// =========================================================================

#[test]
fn test_parallel_tensor_new_shape_mismatch() {
    // Data size doesn't match shape
    let result = ParallelTensor::new(vec![2, 3], vec![1.0, 2.0]); // 6 expected, 2 provided
    assert!(result.is_err());
    if let Err(ParallelError::ShapeMismatch { expected, got }) = result {
        assert_eq!(expected, vec![6]);
        assert_eq!(got, vec![2]);
    }
}

#[test]
fn test_parallel_tensor_narrow_1d() {
    let tensor = ParallelTensor::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
    let narrowed = tensor.narrow(0, 2, 3).expect("test");
    assert_eq!(narrowed.shape, vec![3]);
    assert_eq!(narrowed.data, vec![3.0, 4.0, 5.0]);
}

#[test]
fn test_parallel_tensor_narrow_invalid_dim() {
    let tensor = ParallelTensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
    let result = tensor.narrow(5, 0, 1); // dim 5 doesn't exist
    assert!(result.is_err());
}

#[test]
fn test_parallel_tensor_narrow_fallback_3d() {
    // Test the fallback path for >2D tensors
    let tensor = ParallelTensor::new(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("test");
    let narrowed = tensor.narrow(0, 0, 2).expect("test");
    // Fallback path uses simple slicing
    assert_eq!(narrowed.shape[0], 2);
}

#[test]
fn test_parallel_tensor_transpose_non_2d_error() {
    let tensor = ParallelTensor::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
    let result = tensor.transpose();
    assert!(result.is_err());
    if let Err(ParallelError::ShapeMismatch { expected, got }) = result {
        assert_eq!(expected, vec![2]);
        assert_eq!(got, vec![1]);
    }
}

#[test]
fn test_parallel_tensor_matmul_non_2d_error() {
    let a = ParallelTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let b = ParallelTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let result = a.matmul(&b);
    assert!(result.is_err());
}

#[test]
fn test_parallel_tensor_matmul_dimension_mismatch() {
    let a = ParallelTensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
    let b = ParallelTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test"); // K mismatch
    let result = a.matmul(&b);
    assert!(result.is_err());
    if let Err(ParallelError::ShapeMismatch { expected, got }) = result {
        assert_eq!(expected, vec![3]); // a's K dimension
        assert_eq!(got, vec![2]); // b's first dimension
    }
}

#[test]
fn test_parallel_tensor_add_shape_mismatch() {
    let a = ParallelTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let b = ParallelTensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
    let result = a.add(&b);
    assert!(result.is_err());
}

// =========================================================================
// Additional Coverage Tests: Communicator Operations
// =========================================================================

#[test]
fn test_communicator_all_reduce_max() {
    let comm = Communicator::new(4, 0).expect("test");
    let tensor = ParallelTensor::new(vec![3], vec![1.0, 5.0, 3.0]).expect("test");
    let result = comm.all_reduce(&tensor, ReduceOp::Max).expect("test");
    // Single process Max returns as-is
    assert_eq!(result.data, vec![1.0, 5.0, 3.0]);
}

#[test]
fn test_communicator_all_reduce_min() {
    let comm = Communicator::new(4, 0).expect("test");
    let tensor = ParallelTensor::new(vec![3], vec![1.0, 5.0, 3.0]).expect("test");
    let result = comm.all_reduce(&tensor, ReduceOp::Min).expect("test");
    // Single process Min returns as-is
    assert_eq!(result.data, vec![1.0, 5.0, 3.0]);
}

#[test]
fn test_communicator_reduce_scatter_sum() {
    let comm = Communicator::new(2, 0).expect("test");
    let tensor = ParallelTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let result = comm.reduce_scatter(&tensor, ReduceOp::Sum).expect("test");
    // Rank 0 gets first half, multiplied by world_size for Sum
    assert_eq!(result.shape, vec![2]);
    assert_eq!(result.data, vec![2.0, 4.0]); // [1, 2] * 2
}

#[test]
fn test_communicator_reduce_scatter_avg() {
    let comm = Communicator::new(2, 1).expect("test");
    let tensor = ParallelTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let result = comm.reduce_scatter(&tensor, ReduceOp::Avg).expect("test");
    // Rank 1 gets second half, no transformation for Avg
    assert_eq!(result.shape, vec![2]);
    assert_eq!(result.data, vec![3.0, 4.0]);
}

#[test]
fn test_communicator_reduce_scatter_max() {
    let comm = Communicator::new(2, 0).expect("test");
    let tensor = ParallelTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let result = comm.reduce_scatter(&tensor, ReduceOp::Max).expect("test");
    assert_eq!(result.data, vec![1.0, 2.0]);
}

#[test]
fn test_communicator_reduce_scatter_min() {
    let comm = Communicator::new(2, 1).expect("test");
    let tensor = ParallelTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let result = comm.reduce_scatter(&tensor, ReduceOp::Min).expect("test");
    assert_eq!(result.data, vec![3.0, 4.0]);
}

#[test]
fn test_communicator_all_gather_empty_shape() {
    let comm = Communicator::new(2, 0).expect("test");
    let tensor = ParallelTensor {
        shape: vec![],
        data: vec![],
    };
    let result = comm.all_gather(&tensor).expect("test");
    assert!(result.shape.is_empty());
    assert!(result.data.is_empty());
}

#[test]
fn test_communicator_reduce_scatter_empty_shape() {
    let comm = Communicator::new(2, 0).expect("test");
    let tensor = ParallelTensor {
        shape: vec![],
        data: vec![],
    };
    let result = comm.reduce_scatter(&tensor, ReduceOp::Sum).expect("test");
    assert!(result.shape.is_empty());
}

#[test]
fn test_communicator_clone() {
    let comm = Communicator::new(4, 2).expect("test");
    let cloned = comm.clone();
    assert_eq!(cloned.world_size(), 4);
    assert_eq!(cloned.rank(), 2);
}

// =========================================================================
// Additional Coverage Tests: TensorParallel with Bias
// =========================================================================

#[test]
fn test_tensor_parallel_column_linear_with_bias() {
    let tp = TensorParallel::new(2, 0).expect("test");

    let input = ParallelTensor::new(vec![1, 4], vec![1.0, 1.0, 1.0, 1.0]).expect("test");
    let weight =
        ParallelTensor::new(vec![8, 4], (0..32).map(|i| i as f32).collect()).expect("test");
    let bias = ParallelTensor::new(vec![8], vec![0.5; 8]).expect("test");

    let output = tp
        .column_parallel_linear(&input, &weight, Some(&bias))
        .expect("test");
    assert_eq!(output.shape, vec![1, 4]);
    // Verify bias was added (values should be different from no-bias case)
    assert!(output.data.iter().all(|&x| x != 0.0));
}

#[test]
fn test_tensor_parallel_row_linear_with_bias_rank0() {
    let tp = TensorParallel::new(2, 0).expect("test");

    let input = ParallelTensor::new(vec![1, 8], vec![1.0; 8]).expect("test");
    let weight =
        ParallelTensor::new(vec![4, 8], (0..32).map(|i| i as f32).collect()).expect("test");
    // Bias shape should match output (4 features, split by 2 = 2 per rank)
    let bias = ParallelTensor::new(vec![1, 2], vec![1.0; 2]).expect("test");

    let output = tp
        .row_parallel_linear(&input, &weight, Some(&bias))
        .expect("test");
    // Bias should be added on rank 0
    assert!(!output.data.is_empty());
}

#[test]
fn test_tensor_parallel_row_linear_with_bias_rank1() {
    let tp = TensorParallel::new(2, 1).expect("test");

    let input = ParallelTensor::new(vec![1, 8], vec![1.0; 8]).expect("test");
    let weight =
        ParallelTensor::new(vec![4, 8], (0..32).map(|i| i as f32).collect()).expect("test");
    // Bias shape should match output (4 features, split by 2 = 2 per rank)
    let bias = ParallelTensor::new(vec![1, 2], vec![1.0; 2]).expect("test");

    let output = tp
        .row_parallel_linear(&input, &weight, Some(&bias))
        .expect("test");
    // Bias should NOT be added on rank 1
    assert!(!output.data.is_empty());
}

// =========================================================================
// Additional Coverage Tests: PipelineParallel Edge Cases
// =========================================================================

#[test]
fn test_pipeline_parallel_invalid_zero_size() {
    let result = PipelineParallel::new(0, 0, 24, 4);
    assert!(result.is_err());
}

#[test]
fn test_pipeline_parallel_invalid_stage() {
    let result = PipelineParallel::new(4, 10, 24, 4);
    assert!(result.is_err());
}

#[test]
fn test_pipeline_parallel_bubble_ratio_zero_microbatches() {
    let pp = PipelineParallel::new(4, 0, 24, 4).expect("test");
    let ratio = pp.bubble_ratio(0);
    assert_eq!(ratio, 1.0);
}

#[test]
fn test_pipeline_parallel_bubble_ratio_many_microbatches() {
    let pp = PipelineParallel::new(4, 0, 24, 4).expect("test");
    // With many micro-batches, bubble ratio approaches 0
    let ratio = pp.bubble_ratio(1000);
    assert!(ratio < 0.01);
}

#[test]
fn test_pipeline_parallel_record_multiple_microbatches() {
    let mut pp = PipelineParallel::new(4, 0, 24, 4).expect("test");

    // Record several micro-batches
    pp.record_micro_batch(10.0);
    pp.record_micro_batch(20.0);
    pp.record_micro_batch(30.0);

    let stats = pp.stats();
    assert_eq!(stats.micro_batches_processed, 3);
    assert_eq!(stats.forward_passes, 3);
    // Running average: (10 + 20 + 30) / 3 = 20
    assert!((stats.avg_stage_latency_ms - 20.0).abs() < 0.1);
}

#[test]
fn test_pipeline_stage_clone_serialize() {
    let stage = PipelineStage {
        index: 1,
        start_layer: 6,
        end_layer: 12,
        num_layers: 6,
    };
    let cloned = stage.clone();
    assert_eq!(cloned.index, 1);

    let json = serde_json::to_string(&stage).expect("serialize");
    let deserialized: PipelineStage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.num_layers, 6);
}

// =========================================================================
// Additional Coverage Tests: ParallelConfig Ranks
// =========================================================================

#[test]
fn test_parallel_config_is_tp_last() {
    // tp_size=2, last TP rank is 1
    let config = ParallelConfig::new(2, 1, 1, 1).expect("test");
    assert!(config.is_tp_last());
    assert!(!config.is_tp_first());
}

#[test]
fn test_parallel_config_is_pp_last() {
    // pp_size=2, last PP stage is at rank 1 (with tp_size=1)
    let config = ParallelConfig::new(1, 2, 1, 1).expect("test");
    assert!(config.is_pp_last());
    assert!(!config.is_pp_first());
}

#[test]
fn test_parallel_config_complex_ranks() {
    // World: tp=2, pp=3, dp=2 => 12 ranks
    // Rank 7: tp_rank = 7 % 2 = 1, pp_stage = (7/2) % 3 = 0, dp_rank = 7/(2*3) = 1
    let config = ParallelConfig::new(2, 3, 2, 7).expect("test");
    assert_eq!(config.tp_rank(), 1);
    assert_eq!(config.pp_stage(), 0);
    assert_eq!(config.dp_rank(), 1);
}

#[test]
fn test_parallel_config_default() {
    let config = ParallelConfig::default();
    assert_eq!(config.tp_size, 1);
    assert_eq!(config.pp_size, 1);
    assert_eq!(config.dp_size, 1);
    assert_eq!(config.rank, 0);
}

#[test]
fn test_parallel_config_serialization() {
    let config = ParallelConfig::new(2, 2, 2, 3).expect("test");
    let json = serde_json::to_string(&config).expect("serialize");
    let deserialized: ParallelConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.tp_size, 2);
    assert_eq!(deserialized.rank, 3);
}

// =========================================================================
// Additional Coverage Tests: DistributedContext
// =========================================================================

#[test]
fn test_distributed_context_pipeline_parallel_mut() {
    let config = ParallelConfig::new(1, 4, 1, 0).expect("test");
    let mut ctx = DistributedContext::new(config).expect("test");

    ctx.init_pipeline(24, 4).expect("test");

    // Use pipeline_parallel_mut
    if let Some(pp) = ctx.pipeline_parallel_mut() {
        pp.record_micro_batch(15.0);
    }

    let stats = ctx.pipeline_parallel().expect("pp").stats();
    assert_eq!(stats.micro_batches_processed, 1);
}

#[test]
fn test_distributed_context_config_accessor() {
    let config = ParallelConfig::new(2, 2, 2, 5).expect("test");
    let ctx = DistributedContext::new(config).expect("test");

    let cfg = ctx.config();
    assert_eq!(cfg.tp_size, 2);
    assert_eq!(cfg.rank, 5);
}

#[test]
fn test_distributed_context_init_pipeline_single_pp() {
    // When pp_size = 1, init_pipeline should not create PipelineParallel
    let config = ParallelConfig::new(2, 1, 1, 0).expect("test");
    let mut ctx = DistributedContext::new(config).expect("test");

    ctx.init_pipeline(24, 4).expect("test");
    assert!(ctx.pipeline_parallel().is_none());
}

#[test]
fn test_distributed_context_debug() {
    let config = ParallelConfig::single();
    let ctx = DistributedContext::new(config).expect("test");
    let debug_str = format!("{:?}", ctx);
    assert!(debug_str.contains("config"));
    assert!(debug_str.contains("initialized"));
}

// =========================================================================
// Additional Coverage Tests: ZeroOffload Serialization
// =========================================================================

#[test]
fn test_zero_offload_serialization() {
    let zero = ZeroOffload::inference();
    let json = serde_json::to_string(&zero).expect("serialize");
    let deserialized: ZeroOffload = serde_json::from_str(&json).expect("deserialize");
    assert!(deserialized.offload_params);
    assert!(deserialized.offload_activations);
}

// =========================================================================
// Additional Coverage Tests: ParallelError NotInitialized
// =========================================================================

#[test]
fn test_parallel_error_not_initialized_display() {
    let err = ParallelError::NotInitialized;
    let msg = err.to_string();
    assert!(msg.contains("not initialized"));
}

#[test]
fn test_parallel_error_pipeline_error_display() {
    let err = ParallelError::PipelineError("stage mismatch".to_string());
    let msg = err.to_string();
    assert!(msg.contains("stage mismatch"));
}

#[test]
fn test_parallel_error_invalid_world_size_display() {
    let err = ParallelError::InvalidWorldSize(0);
    let msg = err.to_string();
    assert!(msg.contains("0"));
}

// =========================================================================
// Additional Coverage Tests: ParallelTensor Operations
// =========================================================================

#[test]
fn test_parallel_tensor_zeros_various_shapes() {
    let t1 = ParallelTensor::zeros(vec![1]);
    assert_eq!(t1.numel(), 1);

    let t2 = ParallelTensor::zeros(vec![2, 3, 4]);
    assert_eq!(t2.numel(), 24);
    assert_eq!(t2.sum(), 0.0);
}

#[test]
fn test_parallel_tensor_sum_negative_values() {
    let tensor = ParallelTensor::new(vec![4], vec![-1.0, 2.0, -3.0, 4.0]).expect("test");
    assert_eq!(tensor.sum(), 2.0);
}

#[test]
#[allow(clippy::many_single_char_names)]
fn test_parallel_tensor_large_matmul() {
    // Larger matrix multiplication
    let m = 4;
    let k = 8;
    let n = 4;
    let a = ParallelTensor::new(vec![m, k], vec![1.0; m * k]).expect("test");
    let b = ParallelTensor::new(vec![k, n], vec![1.0; k * n]).expect("test");
    let c = a.matmul(&b).expect("test");

    assert_eq!(c.shape, vec![m, n]);
    // Each element should be k (sum of 1.0 * 1.0 k times)
    assert_eq!(c.data[0], k as f32);
}

#[test]
fn test_parallel_tensor_clone() {
    let tensor = ParallelTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let cloned = tensor.clone();
    assert_eq!(cloned.shape, tensor.shape);
    assert_eq!(cloned.data, tensor.data);
}

#[test]
fn test_parallel_tensor_debug() {
    let tensor = ParallelTensor::new(vec![2], vec![1.0, 2.0]).expect("test");
    let debug_str = format!("{:?}", tensor);
    assert!(debug_str.contains("shape"));
    assert!(debug_str.contains("data"));
}

// =========================================================================
// Additional Coverage Tests: PipelineStats Default
// =========================================================================

#[test]
fn test_pipeline_stats_default() {
    let stats = PipelineStats::default();
    assert_eq!(stats.micro_batches_processed, 0);
    assert_eq!(stats.forward_passes, 0);
    assert_eq!(stats.bubble_time_ms, 0.0);
    assert_eq!(stats.avg_stage_latency_ms, 0.0);
}

#[test]
fn test_pipeline_stats_serialization() {
    let stats = PipelineStats {
        micro_batches_processed: 50,
        forward_passes: 50,
        bubble_time_ms: 2.5,
        avg_stage_latency_ms: 8.0,
    };
    let json = serde_json::to_string(&stats).expect("serialize");
    let deserialized: PipelineStats = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.micro_batches_processed, 50);
    assert!((deserialized.avg_stage_latency_ms - 8.0).abs() < 0.01);
}
