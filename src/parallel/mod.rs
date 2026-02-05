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
mod tests;
