//! Multi-GPU and Distributed Inference
//!
//! Per spec §10: Implements parallelism strategies for 70B+ model inference.
//! Reference: [11] Shoeybi et al. (2019) "Megatron-LM: Training Multi-Billion Parameter LMs"
//!
//! ## Parallelism Strategies
//!
//! | Strategy | Description | Use Case | Scaling |
//! |----------|-------------|----------|---------|
//! | Tensor Parallel (TP) | Split tensors across GPUs | Within node | 2-8 GPUs |
//! | Pipeline Parallel (PP) | Split layers across GPUs | Across nodes | 2-64 GPUs |
//! | Data Parallel (DP) | Replicate model, split batches | High throughput | Any |
//!
//! ## Performance Target
//!
//! Per spec §1.3: >85% scaling efficiency for 2-8 GPUs (Amdahl's law measurement)

// Module-level clippy allows
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_errors_doc)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Error type for parallelism operations
#[derive(Debug, Error)]
pub enum ParallelError {
    /// Invalid rank
    #[error("Invalid rank {rank} for world size {world_size}")]
    InvalidRank {
        /// The invalid rank value
        rank: usize,
        /// The total world size
        world_size: usize,
    },

    /// Invalid world size
    #[error("Invalid world size: {0}")]
    InvalidWorldSize(usize),

    /// Communication error
    #[error("Communication error: {0}")]
    CommunicationError(String),

    /// Tensor shape mismatch
    #[error("Tensor shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape
        got: Vec<usize>,
    },

    /// Pipeline stage error
    #[error("Pipeline stage error: {0}")]
    PipelineError(String),

    /// Not initialized
    #[error("Parallel context not initialized")]
    NotInitialized,
}

/// Reduce operation for collective communications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReduceOp {
    /// Sum all values
    Sum,
    /// Take maximum
    Max,
    /// Take minimum
    Min,
    /// Average (Sum / world_size)
    Avg,
}

/// Parallel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Tensor parallel size (within node)
    pub tp_size: usize,
    /// Pipeline parallel size (across nodes)
    pub pp_size: usize,
    /// Data parallel size (batch distribution)
    pub dp_size: usize,
    /// Current global rank
    pub rank: usize,
    /// Total world size
    pub world_size: usize,
}

impl ParallelConfig {
    /// Create a new parallel configuration
    ///
    /// # Arguments
    ///
    /// * `tp_size` - Tensor parallel size (typically 2, 4, or 8)
    /// * `pp_size` - Pipeline parallel size (number of stages)
    /// * `dp_size` - Data parallel size (batch replication)
    /// * `rank` - Current process rank
    pub fn new(
        tp_size: usize,
        pp_size: usize,
        dp_size: usize,
        rank: usize,
    ) -> Result<Self, ParallelError> {
        let world_size = tp_size * pp_size * dp_size;

        if world_size == 0 {
            return Err(ParallelError::InvalidWorldSize(0));
        }

        if rank >= world_size {
            return Err(ParallelError::InvalidRank { rank, world_size });
        }

        Ok(Self {
            tp_size,
            pp_size,
            dp_size,
            rank,
            world_size,
        })
    }

    /// Create single-GPU configuration (no parallelism)
    pub fn single() -> Self {
        Self {
            tp_size: 1,
            pp_size: 1,
            dp_size: 1,
            rank: 0,
            world_size: 1,
        }
    }

    /// Get tensor parallel rank within TP group
    pub fn tp_rank(&self) -> usize {
        self.rank % self.tp_size
    }

    /// Get pipeline parallel stage
    pub fn pp_stage(&self) -> usize {
        (self.rank / self.tp_size) % self.pp_size
    }

    /// Get data parallel rank
    pub fn dp_rank(&self) -> usize {
        self.rank / (self.tp_size * self.pp_size)
    }

    /// Check if this is the first TP rank
    pub fn is_tp_first(&self) -> bool {
        self.tp_rank() == 0
    }

    /// Check if this is the last TP rank
    pub fn is_tp_last(&self) -> bool {
        self.tp_rank() == self.tp_size - 1
    }

    /// Check if this is the first PP stage
    pub fn is_pp_first(&self) -> bool {
        self.pp_stage() == 0
    }

    /// Check if this is the last PP stage
    pub fn is_pp_last(&self) -> bool {
        self.pp_stage() == self.pp_size - 1
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self::single()
    }
}

/// Mock tensor for parallelism testing
/// In production, this would be replaced with trueno::Tensor
#[derive(Debug, Clone)]
pub struct ParallelTensor {
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Data (f32 for simplicity)
    pub data: Vec<f32>,
}

impl ParallelTensor {
    /// Create a new tensor
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, ParallelError> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![expected_size],
                got: vec![data.len()],
            });
        }
        Ok(Self { shape, data })
    }

    /// Create a zeros tensor
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; size],
        }
    }

    /// Get a narrow slice along a dimension
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self, ParallelError> {
        if dim >= self.shape.len() {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![dim],
                got: self.shape.clone(),
            });
        }

        // For 2D tensors (matrices), implement proper narrowing
        if self.shape.len() == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];

            if dim == 0 {
                // Narrow rows
                let mut new_data = Vec::with_capacity(length * cols);
                for row in start..(start + length) {
                    let row_start = row * cols;
                    new_data.extend_from_slice(&self.data[row_start..row_start + cols]);
                }
                let new_shape = vec![length, cols];
                return Ok(Self {
                    shape: new_shape,
                    data: new_data,
                });
            }
            // Narrow columns
            let mut new_data = Vec::with_capacity(rows * length);
            for row in 0..rows {
                let row_start = row * cols;
                new_data
                    .extend_from_slice(&self.data[row_start + start..row_start + start + length]);
            }
            let new_shape = vec![rows, length];
            return Ok(Self {
                shape: new_shape,
                data: new_data,
            });
        }

        // For 1D tensors
        if self.shape.len() == 1 {
            let new_data = self.data[start..start + length].to_vec();
            return Ok(Self {
                shape: vec![length],
                data: new_data,
            });
        }

        // Fallback: simplified implementation
        let new_data = self.data[start..start + length].to_vec();
        let mut new_shape = self.shape.clone();
        new_shape[dim] = length;
        Ok(Self {
            shape: new_shape,
            data: new_data,
        })
    }

    /// Transpose for 2D tensors
    pub fn transpose(&self) -> Result<Self, ParallelError> {
        if self.shape.len() != 2 {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![2],
                got: vec![self.shape.len()],
            });
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut new_data = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                new_data[j * rows + i] = self.data[i * cols + j];
            }
        }

        Ok(Self {
            shape: vec![cols, rows],
            data: new_data,
        })
    }

    /// Matrix multiplication (simplified)
    pub fn matmul(&self, other: &Self) -> Result<Self, ParallelError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![2, 2],
                got: vec![self.shape.len(), other.shape.len()],
            });
        }

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        if k != other.shape[0] {
            return Err(ParallelError::ShapeMismatch {
                expected: vec![k],
                got: vec![other.shape[0]],
            });
        }

        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.data[i * k + l] * other.data[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Ok(Self {
            shape: vec![m, n],
            data: result,
        })
    }

    /// Add another tensor element-wise
    pub fn add(&self, other: &Self) -> Result<Self, ParallelError> {
        if self.shape != other.shape {
            return Err(ParallelError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Ok(Self {
            shape: self.shape.clone(),
            data,
        })
    }

    /// Sum all elements
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Number of elements
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Mock communicator for collective operations
/// In production, this would use NCCL or MPI
#[derive(Debug, Clone)]
pub struct Communicator {
    /// World size
    world_size: usize,
    /// Current rank
    rank: usize,
    /// test buffers for testing
    #[allow(dead_code)]
    buffers: Arc<std::sync::RwLock<HashMap<usize, Vec<f32>>>>,
}

impl Communicator {
    /// Create a new communicator
    pub fn new(world_size: usize, rank: usize) -> Result<Self, ParallelError> {
        if rank >= world_size {
            return Err(ParallelError::InvalidRank { rank, world_size });
        }
        Ok(Self {
            world_size,
            rank,
            buffers: Arc::new(std::sync::RwLock::new(HashMap::new())),
        })
    }

    /// All-reduce operation
    pub fn all_reduce(
        &self,
        tensor: &ParallelTensor,
        op: ReduceOp,
    ) -> Result<ParallelTensor, ParallelError> {
        // In a real implementation, this would use NCCL
        // For testing, we simulate single-process behavior
        match op {
            ReduceOp::Sum => {
                // Single process: multiply by world_size to simulate sum from all ranks
                let data: Vec<f32> = tensor
                    .data
                    .iter()
                    .map(|x| x * self.world_size as f32)
                    .collect();
                Ok(ParallelTensor {
                    shape: tensor.shape.clone(),
                    data,
                })
            },
            ReduceOp::Avg => {
                // Average: no change in single process (sum / world_size = value)
                Ok(tensor.clone())
            },
            ReduceOp::Max | ReduceOp::Min => {
                // Single process: return as-is
                Ok(tensor.clone())
            },
        }
    }

    /// All-gather operation
    pub fn all_gather(&self, tensor: &ParallelTensor) -> Result<ParallelTensor, ParallelError> {
        // Simulate all-gather by replicating data world_size times
        let mut data = Vec::with_capacity(tensor.data.len() * self.world_size);
        for _ in 0..self.world_size {
            data.extend_from_slice(&tensor.data);
        }

        let mut new_shape = tensor.shape.clone();
        if !new_shape.is_empty() {
            new_shape[0] *= self.world_size;
        }

        Ok(ParallelTensor {
            shape: new_shape,
            data,
        })
    }

    /// Reduce-scatter operation
    pub fn reduce_scatter(
        &self,
        tensor: &ParallelTensor,
        op: ReduceOp,
    ) -> Result<ParallelTensor, ParallelError> {
        // Reduce then scatter: each rank gets 1/world_size of the result
        let chunk_size = tensor.data.len() / self.world_size;
        let start = self.rank * chunk_size;
        let end = start + chunk_size;

        let chunk_data: Vec<f32> = match op {
            ReduceOp::Sum => tensor.data[start..end]
                .iter()
                .map(|x| x * self.world_size as f32)
                .collect(),
            ReduceOp::Avg | ReduceOp::Max | ReduceOp::Min => tensor.data[start..end].to_vec(),
        };

        let mut new_shape = tensor.shape.clone();
        if !new_shape.is_empty() {
            new_shape[0] /= self.world_size;
        }

        Ok(ParallelTensor {
            shape: new_shape,
            data: chunk_data,
        })
    }

    /// Barrier synchronization
    pub fn barrier(&self) -> Result<(), ParallelError> {
        // In real implementation, this would synchronize all processes
        Ok(())
    }

    /// Get world size
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.rank
    }
}

/// Tensor Parallelism for multi-GPU inference
/// Reference: [11] Megatron-LM tensor parallelism
#[derive(Debug)]
pub struct TensorParallel {
    /// Number of tensor parallel ranks
    tp_size: usize,
    /// Current rank within TP group
    rank: usize,
    /// Communication group
    comm: Communicator,
}

impl TensorParallel {
    /// Create a new tensor parallel context
    pub fn new(tp_size: usize, rank: usize) -> Result<Self, ParallelError> {
        if tp_size == 0 {
            return Err(ParallelError::InvalidWorldSize(0));
        }
        if rank >= tp_size {
            return Err(ParallelError::InvalidRank {
                rank,
                world_size: tp_size,
            });
        }

        let comm = Communicator::new(tp_size, rank)?;

        Ok(Self {
            tp_size,
            rank,
            comm,
        })
    }

    /// Get chunk size for weight sharding
    pub fn chunk_size(&self, total_size: usize) -> usize {
        total_size / self.tp_size
    }

    /// Column-parallel linear (for MLP first layer, attention QKV)
    ///
    /// Each rank holds weight[:, rank*chunk:(rank+1)*chunk]
    /// No communication needed as outputs are independent
    pub fn column_parallel_linear(
        &self,
        input: &ParallelTensor,
        weight: &ParallelTensor,
        bias: Option<&ParallelTensor>,
    ) -> Result<ParallelTensor, ParallelError> {
        // Get local weight slice
        let output_dim = weight.shape[0];
        let chunk = self.chunk_size(output_dim);
        let local_weight = weight.narrow(0, self.rank * chunk, chunk)?;

        // Transpose weight for matmul: (out_chunk, in) -> (in, out_chunk)
        let weight_t = local_weight.transpose()?;

        // Local matmul: (batch, in) @ (in, out_chunk) -> (batch, out_chunk)
        let mut local_output = input.matmul(&weight_t)?;

        // Add local bias if present
        if let Some(b) = bias {
            let local_bias = b.narrow(0, self.rank * chunk, chunk)?;
            // Broadcast bias addition
            let bias_expanded = ParallelTensor {
                shape: local_output.shape.clone(),
                data: local_output
                    .data
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v + local_bias.data[i % local_bias.data.len()])
                    .collect(),
            };
            local_output = bias_expanded;
        }

        Ok(local_output)
    }

    /// Row-parallel linear (for MLP second layer, attention output)
    ///
    /// Each rank holds weight[rank*chunk:(rank+1)*chunk, :]
    /// Requires all-reduce to sum partial results
    pub fn row_parallel_linear(
        &self,
        input: &ParallelTensor,
        weight: &ParallelTensor,
        bias: Option<&ParallelTensor>,
    ) -> Result<ParallelTensor, ParallelError> {
        // Get local weight slice (rows)
        let input_dim = weight.shape[0];
        let chunk = self.chunk_size(input_dim);
        let local_weight = weight.narrow(0, self.rank * chunk, chunk)?;

        // Transpose for matmul
        let weight_t = local_weight.transpose()?;

        // Local matmul
        let local_output = input.matmul(&weight_t)?;

        // All-reduce to sum partial results
        let mut output = self.comm.all_reduce(&local_output, ReduceOp::Sum)?;

        // Add bias only on rank 0 to avoid double counting
        if self.rank == 0 {
            if let Some(b) = bias {
                output = output.add(b)?;
            }
        }

        Ok(output)
    }

    /// Get TP rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get TP size
    pub fn tp_size(&self) -> usize {
        self.tp_size
    }
}

/// Pipeline stage info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    /// Stage index
    pub index: usize,
    /// Start layer index
    pub start_layer: usize,
    /// End layer index (exclusive)
    pub end_layer: usize,
    /// Number of layers in this stage
    pub num_layers: usize,
}

/// Pipeline Parallelism for multi-node inference
/// Reference: [11] GPipe-style pipeline parallelism
#[derive(Debug)]
pub struct PipelineParallel {
    /// Number of pipeline stages
    pp_size: usize,
    /// Current stage
    stage: usize,
    /// Stage info
    stage_info: PipelineStage,
    /// Micro-batch size for pipelining
    micro_batch_size: usize,
    /// Stats
    stats: PipelineStats,
}

/// Pipeline execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineStats {
    /// Total micro-batches processed
    pub micro_batches_processed: u64,
    /// Total pipeline bubbles (idle time)
    pub bubble_time_ms: f64,
    /// Average stage latency
    pub avg_stage_latency_ms: f64,
    /// Total forward passes
    pub forward_passes: u64,
}

impl PipelineParallel {
    /// Create a new pipeline parallel context
    ///
    /// # Arguments
    ///
    /// * `pp_size` - Number of pipeline stages
    /// * `stage` - Current stage index (0 to pp_size-1)
    /// * `total_layers` - Total number of layers to distribute
    /// * `micro_batch_size` - Size of micro-batches for pipelining
    pub fn new(
        pp_size: usize,
        stage: usize,
        total_layers: usize,
        micro_batch_size: usize,
    ) -> Result<Self, ParallelError> {
        if pp_size == 0 {
            return Err(ParallelError::InvalidWorldSize(0));
        }
        if stage >= pp_size {
            return Err(ParallelError::InvalidRank {
                rank: stage,
                world_size: pp_size,
            });
        }

        // Distribute layers evenly across stages
        let layers_per_stage = total_layers / pp_size;
        let extra_layers = total_layers % pp_size;

        // Earlier stages get extra layers if uneven
        let start_layer = stage * layers_per_stage + stage.min(extra_layers);
        let num_layers = layers_per_stage + usize::from(stage < extra_layers);
        let end_layer = start_layer + num_layers;

        let stage_info = PipelineStage {
            index: stage,
            start_layer,
            end_layer,
            num_layers,
        };

        Ok(Self {
            pp_size,
            stage,
            stage_info,
            micro_batch_size,
            stats: PipelineStats::default(),
        })
    }

    /// Get stage info
    pub fn stage_info(&self) -> &PipelineStage {
        &self.stage_info
    }

    /// Get micro-batch size
    pub fn micro_batch_size(&self) -> usize {
        self.micro_batch_size
    }

    /// Check if this is the first stage
    pub fn is_first_stage(&self) -> bool {
        self.stage == 0
    }

    /// Check if this is the last stage
    pub fn is_last_stage(&self) -> bool {
        self.stage == self.pp_size - 1
    }

    /// Get number of stages
    pub fn num_stages(&self) -> usize {
        self.pp_size
    }

    /// Get current stage index
    pub fn stage(&self) -> usize {
        self.stage
    }

    /// Calculate theoretical bubble ratio (idle time fraction)
    /// Bubble ratio = (pp_size - 1) / (pp_size + num_microbatches - 1)
    pub fn bubble_ratio(&self, num_microbatches: usize) -> f32 {
        if num_microbatches == 0 {
            return 1.0;
        }
        (self.pp_size - 1) as f32 / (self.pp_size + num_microbatches - 1) as f32
    }

    /// Get statistics
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Record a micro-batch processed
    pub fn record_micro_batch(&mut self, stage_latency_ms: f64) {
        self.stats.micro_batches_processed += 1;
        self.stats.forward_passes += 1;

        // Update running average
        let n = self.stats.micro_batches_processed as f64;
        self.stats.avg_stage_latency_ms =
            (self.stats.avg_stage_latency_ms * (n - 1.0) + stage_latency_ms) / n;
    }
}

/// ZeRO-Inference memory offload
/// Reference: [10] Microsoft DeepSpeed ZeRO-Inference
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)] // Config struct - bools are appropriate
pub struct ZeroOffload {
    /// Offload optimizer states to CPU
    pub offload_optimizer: bool,
    /// Offload parameters to CPU
    pub offload_params: bool,
    /// Offload activations to CPU
    pub offload_activations: bool,
    /// Pin memory for faster CPU-GPU transfer
    pub pin_memory: bool,
    /// Overlap compute and communication
    pub overlap_comm: bool,
}

impl Default for ZeroOffload {
    fn default() -> Self {
        Self {
            offload_optimizer: true,
            offload_params: false,
            offload_activations: false,
            pin_memory: true,
            overlap_comm: true,
        }
    }
}

impl ZeroOffload {
    /// Create inference-optimized config (offload everything)
    pub fn inference() -> Self {
        Self {
            offload_optimizer: false, // No optimizer in inference
            offload_params: true,
            offload_activations: true,
            pin_memory: true,
            overlap_comm: true,
        }
    }

    /// Estimate memory savings ratio
    pub fn memory_savings_ratio(&self) -> f32 {
        let mut ratio = 1.0;
        if self.offload_params {
            ratio *= 0.5; // Params on CPU
        }
        if self.offload_activations {
            ratio *= 0.7; // Activations on CPU
        }
        1.0 - ratio
    }
}

/// Distributed inference context combining all parallelism strategies
#[derive(Debug)]
pub struct DistributedContext {
    /// Parallel configuration
    config: ParallelConfig,
    /// Tensor parallelism (if enabled)
    tensor_parallel: Option<TensorParallel>,
    /// Pipeline parallelism (if enabled)
    pipeline_parallel: Option<PipelineParallel>,
    /// ZeRO offload settings
    zero_offload: ZeroOffload,
    /// Initialized flag
    initialized: bool,
}

impl DistributedContext {
    /// Create a new distributed context
    pub fn new(config: ParallelConfig) -> Result<Self, ParallelError> {
        let tensor_parallel = if config.tp_size > 1 {
            Some(TensorParallel::new(config.tp_size, config.tp_rank())?)
        } else {
            None
        };

        // Note: Pipeline parallel requires layer count, initialized separately
        let pipeline_parallel = None;

        Ok(Self {
            config,
            tensor_parallel,
            pipeline_parallel,
            zero_offload: ZeroOffload::default(),
            initialized: true,
        })
    }

    /// Initialize pipeline parallelism
    pub fn init_pipeline(
        &mut self,
        total_layers: usize,
        micro_batch_size: usize,
    ) -> Result<(), ParallelError> {
        if self.config.pp_size > 1 {
            self.pipeline_parallel = Some(PipelineParallel::new(
                self.config.pp_size,
                self.config.pp_stage(),
                total_layers,
                micro_batch_size,
            )?);
        }
        Ok(())
    }

    /// Set ZeRO offload configuration
    pub fn set_zero_offload(&mut self, zero: ZeroOffload) {
        self.zero_offload = zero;
    }

    /// Get parallel configuration
    pub fn config(&self) -> &ParallelConfig {
        &self.config
    }

    /// Get tensor parallel context
    pub fn tensor_parallel(&self) -> Option<&TensorParallel> {
        self.tensor_parallel.as_ref()
    }

    /// Get pipeline parallel context
    pub fn pipeline_parallel(&self) -> Option<&PipelineParallel> {
        self.pipeline_parallel.as_ref()
    }

    /// Get mutable pipeline parallel context
    pub fn pipeline_parallel_mut(&mut self) -> Option<&mut PipelineParallel> {
        self.pipeline_parallel.as_mut()
    }

    /// Get ZeRO offload config
    pub fn zero_offload(&self) -> &ZeroOffload {
        &self.zero_offload
    }

    /// Check if distributed execution is enabled
    pub fn is_distributed(&self) -> bool {
        self.config.world_size > 1
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

#[cfg(test)]
mod tests {
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
        let tensor =
            ParallelTensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
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
        let tensor =
            ParallelTensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
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
}
