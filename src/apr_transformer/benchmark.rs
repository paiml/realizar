//! APR Benchmark Infrastructure (Y6: Format Parity Validation)
//!
//! Provides standardized benchmarking for APR transformers following the benchmark spec:
//! - Dynamic CV-based sampling
//! - Statistical metrics (p50, p99, std_dev)
//! - Throughput and memory measurement
//!
//! Extracted from apr_transformer.rs (PMAT-802)

use crate::error::Result;

use super::{AprTransformer, GenerateConfig};

// ============================================================================
// Y6: APR Benchmark Infrastructure (Format Parity Validation)
// ============================================================================

/// CPU decode threshold: 50 tok/s per spec Y6
pub const APR_CPU_DECODE_THRESHOLD_TOK_S: f64 = 50.0;

/// Prefill threshold: 100 tok/s per spec Y8
pub const APR_PREFILL_THRESHOLD_TOK_S: f64 = 100.0;

/// Parity threshold: 95% of baseline per spec Y6
pub const APR_PARITY_THRESHOLD_PCT: f64 = 95.0;

/// Result of an APR benchmark run
#[derive(Debug, Clone, Default)]
pub struct AprBenchmarkResult {
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Total time in milliseconds
    pub total_time_ms: f64,
    /// Throughput in tokens per second
    pub tokens_per_second: f64,
    /// Median throughput (p50)
    pub throughput_p50: f64,
    /// 99th percentile throughput (worst case)
    pub throughput_p99: f64,
    /// Standard deviation of throughput
    pub throughput_std_dev: f64,
    /// Peak memory usage in MB
    pub peak_memory_mb: f64,
    /// Model memory in MB
    pub model_memory_mb: f64,
}

impl AprBenchmarkResult {
    /// Check if benchmark meets the given throughput threshold
    #[must_use]
    pub fn meets_threshold(&self, threshold_tok_s: f64) -> bool {
        self.tokens_per_second >= threshold_tok_s
    }

    /// Compare this result to a baseline
    #[must_use]
    pub fn compare_to_baseline(&self, baseline: &AprBenchmarkResult) -> AprParityComparison {
        let throughput_ratio = if baseline.tokens_per_second > 0.0 {
            self.tokens_per_second / baseline.tokens_per_second
        } else {
            1.0
        };

        let memory_ratio = if baseline.peak_memory_mb > 0.0 {
            self.peak_memory_mb / baseline.peak_memory_mb
        } else {
            1.0
        };

        AprParityComparison {
            throughput_ratio,
            memory_ratio,
            parity_threshold_pct: APR_PARITY_THRESHOLD_PCT,
        }
    }
}

/// Result of prefill benchmark
#[derive(Debug, Clone, Default)]
pub struct AprPrefillResult {
    /// Number of prompt tokens processed
    pub prompt_tokens: usize,
    /// Prefill time in milliseconds
    pub prefill_time_ms: f64,
    /// Prefill throughput in tokens per second
    pub prefill_tok_s: f64,
}

/// Result of load time benchmark
#[derive(Debug, Clone, Default)]
pub struct AprLoadResult {
    /// Load time in milliseconds
    pub load_time_ms: f64,
}

/// Comparison of APR benchmark to baseline (for parity validation)
#[derive(Debug, Clone)]
pub struct AprParityComparison {
    /// Ratio of APR throughput to baseline
    pub throughput_ratio: f64,
    /// Ratio of APR memory to baseline
    pub memory_ratio: f64,
    /// Parity threshold percentage
    pub parity_threshold_pct: f64,
}

impl AprParityComparison {
    /// Check if APR achieves parity with baseline
    #[must_use]
    pub fn is_parity(&self) -> bool {
        self.throughput_ratio >= (self.parity_threshold_pct / 100.0)
    }
}

/// Benchmark runner for APR transformers (Y6)
///
/// Provides standardized benchmarking following the benchmark spec:
/// - Dynamic CV-based sampling
/// - Statistical metrics (p50, p99, std_dev)
/// - Throughput and memory measurement
#[derive(Debug)]
pub struct AprBenchmarkRunner {
    /// The transformer to benchmark
    transformer: AprTransformer,
    /// Number of warmup iterations
    warmup_iterations: usize,
    /// Number of measurement iterations
    measure_iterations: usize,
}

impl AprBenchmarkRunner {
    /// Create a new benchmark runner for the given transformer
    #[must_use]
    pub fn new(transformer: AprTransformer) -> Self {
        Self {
            transformer,
            warmup_iterations: 3,
            measure_iterations: 10,
        }
    }

    /// Get warmup iterations
    #[must_use]
    pub fn warmup_iterations(&self) -> usize {
        self.warmup_iterations
    }

    /// Get measure iterations
    #[must_use]
    pub fn measure_iterations(&self) -> usize {
        self.measure_iterations
    }

    /// Set warmup iterations
    pub fn set_warmup_iterations(&mut self, n: usize) {
        self.warmup_iterations = n;
    }

    /// Set measure iterations
    pub fn set_measure_iterations(&mut self, n: usize) {
        self.measure_iterations = n.max(1);
    }

    /// Benchmark decode throughput
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `num_tokens` - Number of tokens to generate
    ///
    /// # Returns
    ///
    /// Benchmark result with throughput metrics
    pub fn benchmark_decode(
        &mut self,
        prompt: &[u32],
        num_tokens: usize,
    ) -> Result<AprBenchmarkResult> {
        use std::time::Instant;

        // Warmup
        for _ in 0..self.warmup_iterations {
            let gen_config = GenerateConfig {
                max_tokens: num_tokens.min(5),
                temperature: 0.0,
                ..Default::default()
            };
            let _ = self.transformer.generate_with_cache(prompt, &gen_config)?;
        }

        // Measurement runs
        let mut throughputs = Vec::with_capacity(self.measure_iterations);
        let mut total_tokens = 0usize;
        let mut total_time_ms = 0.0f64;

        for _ in 0..self.measure_iterations {
            let gen_config = GenerateConfig {
                max_tokens: num_tokens,
                temperature: 0.0,
                ..Default::default()
            };

            let start = Instant::now();
            let output = self.transformer.generate_with_cache(prompt, &gen_config)?;
            let elapsed = start.elapsed();

            let generated = output.len().saturating_sub(prompt.len());
            let time_ms = elapsed.as_secs_f64() * 1000.0;
            let throughput = if time_ms > 0.0 {
                (generated as f64) / (time_ms / 1000.0)
            } else {
                0.0
            };

            throughputs.push(throughput);
            total_tokens += generated;
            total_time_ms += time_ms;
        }

        // Calculate statistics
        let mean_throughput = if !throughputs.is_empty() {
            throughputs.iter().sum::<f64>() / throughputs.len() as f64
        } else {
            0.0
        };

        let mut sorted = throughputs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p50 = if !sorted.is_empty() {
            sorted[sorted.len() / 2]
        } else {
            0.0
        };

        let p99_idx =
            ((sorted.len() as f64 * 0.01).floor() as usize).min(sorted.len().saturating_sub(1));
        let p99 = if !sorted.is_empty() {
            sorted[p99_idx]
        } else {
            0.0
        };

        let std_dev = if throughputs.len() > 1 {
            let variance = throughputs
                .iter()
                .map(|t| (t - mean_throughput).powi(2))
                .sum::<f64>()
                / (throughputs.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Memory estimation
        let model_memory_mb = (self.transformer.memory_size() as f64) / (1024.0 * 1024.0);

        Ok(AprBenchmarkResult {
            tokens_generated: total_tokens / self.measure_iterations.max(1),
            total_time_ms: total_time_ms / self.measure_iterations.max(1) as f64,
            tokens_per_second: mean_throughput,
            throughput_p50: p50,
            throughput_p99: p99,
            throughput_std_dev: std_dev,
            peak_memory_mb: model_memory_mb * 1.5, // Estimate: model + KV cache
            model_memory_mb,
        })
    }

    /// Benchmark prefill throughput
    ///
    /// # Arguments
    ///
    /// * `prompt` - Tokens to prefill
    ///
    /// # Returns
    ///
    /// Prefill benchmark result
    pub fn benchmark_prefill(&mut self, prompt: &[u32]) -> Result<AprPrefillResult> {
        use std::time::Instant;

        // Warmup
        for _ in 0..self.warmup_iterations {
            let _ = self.transformer.forward(prompt)?;
        }

        // Measurement runs
        let mut prefill_times_ms = Vec::with_capacity(self.measure_iterations);

        for _ in 0..self.measure_iterations {
            let start = Instant::now();
            let _ = self.transformer.forward(prompt)?;
            let elapsed = start.elapsed();
            prefill_times_ms.push(elapsed.as_secs_f64() * 1000.0);
        }

        let mean_time_ms = if !prefill_times_ms.is_empty() {
            prefill_times_ms.iter().sum::<f64>() / prefill_times_ms.len() as f64
        } else {
            0.0
        };

        let prefill_tok_s = if mean_time_ms > 0.0 {
            (prompt.len() as f64) / (mean_time_ms / 1000.0)
        } else {
            0.0
        };

        Ok(AprPrefillResult {
            prompt_tokens: prompt.len(),
            prefill_time_ms: mean_time_ms,
            prefill_tok_s,
        })
    }

    /// Benchmark model load time
    ///
    /// # Arguments
    ///
    /// * `loader` - Closure that creates the transformer
    ///
    /// # Returns
    ///
    /// Load time result
    pub fn benchmark_load<F>(loader: F) -> Result<AprLoadResult>
    where
        F: Fn() -> AprTransformer,
    {
        use std::time::Instant;

        // Single measurement (load is typically done once)
        let start = Instant::now();
        let _transformer = loader();
        let elapsed = start.elapsed();

        Ok(AprLoadResult {
            load_time_ms: elapsed.as_secs_f64() * 1000.0,
        })
    }
}

// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Constants Tests
    // =========================================================================

    #[test]
    fn test_apr_cpu_decode_threshold() {
        assert!((APR_CPU_DECODE_THRESHOLD_TOK_S - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apr_prefill_threshold() {
        assert!((APR_PREFILL_THRESHOLD_TOK_S - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apr_parity_threshold() {
        assert!((APR_PARITY_THRESHOLD_PCT - 95.0).abs() < f64::EPSILON);
    }

    // =========================================================================
    // AprBenchmarkResult Tests
    // =========================================================================

    #[test]
    fn test_benchmark_result_default() {
        let result = AprBenchmarkResult::default();
        assert_eq!(result.tokens_generated, 0);
        assert_eq!(result.total_time_ms, 0.0);
        assert_eq!(result.tokens_per_second, 0.0);
        assert_eq!(result.throughput_p50, 0.0);
        assert_eq!(result.throughput_p99, 0.0);
        assert_eq!(result.throughput_std_dev, 0.0);
        assert_eq!(result.peak_memory_mb, 0.0);
        assert_eq!(result.model_memory_mb, 0.0);
    }

    #[test]
    fn test_benchmark_result_clone() {
        let result = AprBenchmarkResult {
            tokens_generated: 100,
            total_time_ms: 1000.0,
            tokens_per_second: 100.0,
            throughput_p50: 95.0,
            throughput_p99: 80.0,
            throughput_std_dev: 5.0,
            peak_memory_mb: 512.0,
            model_memory_mb: 256.0,
        };
        let cloned = result.clone();
        assert_eq!(cloned.tokens_generated, 100);
        assert_eq!(cloned.tokens_per_second, 100.0);
    }

    #[test]
    fn test_benchmark_result_debug() {
        let result = AprBenchmarkResult::default();
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("AprBenchmarkResult"));
        assert!(debug_str.contains("tokens_generated"));
    }

    #[test]
    fn test_meets_threshold_pass() {
        let result = AprBenchmarkResult {
            tokens_per_second: 60.0,
            ..Default::default()
        };
        assert!(result.meets_threshold(50.0));
        assert!(result.meets_threshold(60.0)); // equal
    }

    #[test]
    fn test_meets_threshold_fail() {
        let result = AprBenchmarkResult {
            tokens_per_second: 40.0,
            ..Default::default()
        };
        assert!(!result.meets_threshold(50.0));
    }

    #[test]
    fn test_meets_threshold_zero() {
        let result = AprBenchmarkResult::default();
        assert!(!result.meets_threshold(50.0));
        assert!(result.meets_threshold(0.0)); // 0 >= 0
    }

    #[test]
    fn test_compare_to_baseline_normal() {
        let result = AprBenchmarkResult {
            tokens_per_second: 90.0,
            peak_memory_mb: 500.0,
            ..Default::default()
        };
        let baseline = AprBenchmarkResult {
            tokens_per_second: 100.0,
            peak_memory_mb: 400.0,
            ..Default::default()
        };
        let comparison = result.compare_to_baseline(&baseline);
        assert!((comparison.throughput_ratio - 0.9).abs() < 0.001);
        assert!((comparison.memory_ratio - 1.25).abs() < 0.001);
        assert_eq!(comparison.parity_threshold_pct, APR_PARITY_THRESHOLD_PCT);
    }

    #[test]
    fn test_compare_to_baseline_zero_baseline() {
        let result = AprBenchmarkResult {
            tokens_per_second: 100.0,
            peak_memory_mb: 500.0,
            ..Default::default()
        };
        let baseline = AprBenchmarkResult::default(); // zeros
        let comparison = result.compare_to_baseline(&baseline);
        assert_eq!(comparison.throughput_ratio, 1.0);
        assert_eq!(comparison.memory_ratio, 1.0);
    }

    #[test]
    fn test_compare_to_baseline_equal() {
        let result = AprBenchmarkResult {
            tokens_per_second: 100.0,
            peak_memory_mb: 500.0,
            ..Default::default()
        };
        let comparison = result.compare_to_baseline(&result);
        assert!((comparison.throughput_ratio - 1.0).abs() < 0.001);
        assert!((comparison.memory_ratio - 1.0).abs() < 0.001);
    }

    // =========================================================================
    // AprPrefillResult Tests
    // =========================================================================

    #[test]
    fn test_prefill_result_default() {
        let result = AprPrefillResult::default();
        assert_eq!(result.prompt_tokens, 0);
        assert_eq!(result.prefill_time_ms, 0.0);
        assert_eq!(result.prefill_tok_s, 0.0);
    }

    #[test]
    fn test_prefill_result_clone() {
        let result = AprPrefillResult {
            prompt_tokens: 512,
            prefill_time_ms: 50.0,
            prefill_tok_s: 10240.0,
        };
        let cloned = result.clone();
        assert_eq!(cloned.prompt_tokens, 512);
        assert_eq!(cloned.prefill_time_ms, 50.0);
        assert_eq!(cloned.prefill_tok_s, 10240.0);
    }

    #[test]
    fn test_prefill_result_debug() {
        let result = AprPrefillResult::default();
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("AprPrefillResult"));
        assert!(debug_str.contains("prompt_tokens"));
    }

    // =========================================================================
    // AprLoadResult Tests
    // =========================================================================

    #[test]
    fn test_load_result_default() {
        let result = AprLoadResult::default();
        assert_eq!(result.load_time_ms, 0.0);
    }

    #[test]
    fn test_load_result_clone() {
        let result = AprLoadResult {
            load_time_ms: 1234.5,
        };
        let cloned = result.clone();
        assert_eq!(cloned.load_time_ms, 1234.5);
    }

    #[test]
    fn test_load_result_debug() {
        let result = AprLoadResult {
            load_time_ms: 999.0,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("AprLoadResult"));
        assert!(debug_str.contains("load_time_ms"));
    }

    // =========================================================================
    // AprParityComparison Tests
    // =========================================================================

    #[test]
    fn test_parity_comparison_clone() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.96,
            memory_ratio: 1.1,
            parity_threshold_pct: 95.0,
        };
        let cloned = comparison.clone();
        assert!((cloned.throughput_ratio - 0.96).abs() < 0.001);
        assert!((cloned.memory_ratio - 1.1).abs() < 0.001);
    }

    #[test]
    fn test_parity_comparison_debug() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.96,
            memory_ratio: 1.1,
            parity_threshold_pct: 95.0,
        };
        let debug_str = format!("{:?}", comparison);
        assert!(debug_str.contains("AprParityComparison"));
        assert!(debug_str.contains("throughput_ratio"));
    }

    #[test]
    fn test_is_parity_pass() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.96, // 96% >= 95%
            memory_ratio: 1.0,
            parity_threshold_pct: 95.0,
        };
        assert!(comparison.is_parity());
    }

    #[test]
    fn test_is_parity_exact_threshold() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.95, // 95% == 95%
            memory_ratio: 1.0,
            parity_threshold_pct: 95.0,
        };
        assert!(comparison.is_parity());
    }

    #[test]
    fn test_is_parity_fail() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.94, // 94% < 95%
            memory_ratio: 1.0,
            parity_threshold_pct: 95.0,
        };
        assert!(!comparison.is_parity());
    }

    #[test]
    fn test_is_parity_exceed() {
        let comparison = AprParityComparison {
            throughput_ratio: 1.1, // 110% > 95%
            memory_ratio: 0.9,
            parity_threshold_pct: 95.0,
        };
        assert!(comparison.is_parity());
    }

    #[test]
    fn test_is_parity_custom_threshold() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.85,
            memory_ratio: 1.0,
            parity_threshold_pct: 80.0, // 85% >= 80%
        };
        assert!(comparison.is_parity());
    }

    #[test]
    fn test_is_parity_zero_threshold() {
        let comparison = AprParityComparison {
            throughput_ratio: 0.01,
            memory_ratio: 1.0,
            parity_threshold_pct: 0.0, // any positive passes
        };
        assert!(comparison.is_parity());
    }
}
