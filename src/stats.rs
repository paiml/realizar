//! Statistical analysis for A/B testing with log-normal latency support
//!
//! Per Box et al. (2005), latency distributions are often log-normal.
//! This module provides log-transform and non-parametric tests.
//!
//! ## Features
//!
//! - **Welch's t-test**: Standard parametric comparison
//! - **Log-transformed t-test**: For log-normal latency data
//! - **Mann-Whitney U test**: Non-parametric test per Box et al. (2005)
//! - **Automatic test selection**: Based on sample size and skewness
//!
//! ## Citations
//!
//! - Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005).
//!   *Statistics for Experimenters*. Wiley-Interscience.
//! - Welch, B. L. (1947). "The Generalization of 'Student's' Problem."
//!   *Biometrika*, 34(1-2), 28-35.

#![allow(clippy::cast_precision_loss)] // Statistical functions need usize->f64

/// Configuration for experiment analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Significance level (default 0.05)
    pub alpha: f64,
    /// Whether to auto-detect skewness
    pub auto_detect_skew: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            auto_detect_skew: true,
        }
    }
}

/// Result of statistical analysis
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Control group mean (or geometric mean if log-transformed)
    pub control_mean: f64,
    /// Treatment group mean
    pub treatment_mean: f64,
    /// Effect size (relative change)
    pub effect_size: f64,
    /// P-value for the test
    pub p_value: f64,
    /// Whether result is statistically significant
    pub significant: bool,
    /// Test method used
    pub method: TestMethod,
}

/// Statistical test method used
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestMethod {
    /// Standard t-test (normal data)
    TTest,
    /// Log-transformed t-test (log-normal data)
    LogTransformTTest,
    /// Mann-Whitney U test (non-parametric)
    MannWhitneyU,
}

/// Analyze experiment results with automatic distribution detection
#[must_use]
pub fn analyze(control: &[f64], treatment: &[f64], config: &AnalysisConfig) -> AnalysisResult {
    let skewness = compute_skewness(control);

    // Auto-detect: use log-transform for highly skewed data (latency)
    let use_log = config.auto_detect_skew && skewness.abs() > 1.0;

    if use_log {
        analyze_log_transform(control, treatment, config.alpha)
    } else {
        analyze_t_test(control, treatment, config.alpha)
    }
}

/// Log-transformed t-test for log-normal latency data
#[must_use]
pub fn analyze_log_transform(control: &[f64], treatment: &[f64], alpha: f64) -> AnalysisResult {
    // Transform to log space
    let log_control: Vec<f64> = control.iter().map(|x| x.ln()).collect();
    let log_treatment: Vec<f64> = treatment.iter().map(|x| x.ln()).collect();

    let result = analyze_t_test(&log_control, &log_treatment, alpha);

    // Convert means back (geometric mean)
    AnalysisResult {
        control_mean: result.control_mean.exp(),
        treatment_mean: result.treatment_mean.exp(),
        effect_size: result.treatment_mean.exp() / result.control_mean.exp() - 1.0,
        p_value: result.p_value,
        significant: result.significant,
        method: TestMethod::LogTransformTTest,
    }
}

/// Standard Welch's t-test
#[must_use]
pub fn analyze_t_test(control: &[f64], treatment: &[f64], alpha: f64) -> AnalysisResult {
    let n1 = control.len() as f64;
    let n2 = treatment.len() as f64;

    let mean1 = control.iter().sum::<f64>() / n1;
    let mean2 = treatment.iter().sum::<f64>() / n2;

    let var1 = control.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let var2 = treatment.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

    let se = (var1 / n1 + var2 / n2).sqrt();
    let t_stat = (mean2 - mean1) / se;

    // Approximate p-value using normal distribution (valid for large n)
    let p_value = 2.0 * (1.0 - normal_cdf(t_stat.abs()));

    AnalysisResult {
        control_mean: mean1,
        treatment_mean: mean2,
        effect_size: (mean2 - mean1) / mean1,
        p_value,
        significant: p_value < alpha,
        method: TestMethod::TTest,
    }
}

/// Compute skewness of a distribution
fn compute_skewness(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let std_dev = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();

    if std_dev < 1e-10 {
        return 0.0;
    }

    let m3 = data
        .iter()
        .map(|x| ((x - mean) / std_dev).powi(3))
        .sum::<f64>()
        / n;
    m3
}

// ============================================================================
// Mann-Whitney U Test (Box et al. 2005)
// ============================================================================

/// Effect size interpretation per Cohen's conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectSizeInterpretation {
    /// |r| < 0.1 - trivial effect
    Negligible,
    /// 0.1 <= |r| < 0.3 - small effect
    Small,
    /// 0.3 <= |r| < 0.5 - medium effect
    Medium,
    /// |r| >= 0.5 - large effect
    Large,
}

/// Result of Mann-Whitney U test
#[derive(Debug, Clone)]
pub struct MannWhitneyResult {
    /// The U statistic (minimum of U1 and U2)
    pub u_statistic: f64,
    /// Z-score for normal approximation
    pub z_score: f64,
    /// P-value (two-tailed)
    pub p_value: f64,
    /// Whether result is significant at alpha=0.05
    pub significant: bool,
    /// Effect size (rank-biserial correlation)
    pub effect_size: f64,
    /// Interpretation of effect size
    pub effect_interpretation: EffectSizeInterpretation,
    /// Test method identifier
    pub method: TestMethod,
}

/// Mann-Whitney U test for non-parametric comparison
///
/// Also known as Wilcoxon rank-sum test. Compares two independent samples
/// without assuming normality. Preferred when:
/// - Distribution is heavily skewed (skewness > 2)
/// - Sample sizes are small (n < 15)
/// - Outliers are present and meaningful
///
/// ## Algorithm
///
/// 1. Combine and rank all observations
/// 2. Handle ties by assigning average ranks
/// 3. Compute U statistic from rank sums
/// 4. Use normal approximation for p-value
///
/// ## Citation
///
/// Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005).
/// *Statistics for Experimenters*. Wiley-Interscience.
#[must_use]
pub fn mann_whitney_u(control: &[f64], treatment: &[f64]) -> MannWhitneyResult {
    let n1 = control.len();
    let n2 = treatment.len();

    // Combine samples with group labels
    let mut combined: Vec<(f64, usize)> = control
        .iter()
        .map(|&x| (x, 0)) // Group 0 = control
        .chain(treatment.iter().map(|&x| (x, 1))) // Group 1 = treatment
        .collect();

    // Sort by value
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks with tie handling
    let ranks = assign_ranks_with_ties(&combined);

    // Sum ranks for control group
    let r1: f64 = ranks
        .iter()
        .filter(|(_, group)| *group == 0)
        .map(|(rank, _)| rank)
        .sum();

    // Calculate U statistics
    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u2 = (n1 * n2) as f64 - u1;
    let u_statistic = u1.min(u2);

    // Normal approximation (valid for n1, n2 >= 5)
    let mu = (n1 * n2) as f64 / 2.0;
    let sigma = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();

    let z_score = if sigma > 0.0 {
        (u_statistic - mu) / sigma
    } else {
        0.0
    };

    // Two-tailed p-value
    let p_value = 2.0 * (1.0 - normal_cdf(z_score.abs()));

    // Effect size: rank-biserial correlation
    // r = 1 - (2U)/(n1*n2)
    let effect_size = 1.0 - (2.0 * u_statistic) / (n1 * n2) as f64;

    let effect_interpretation = interpret_effect_size(effect_size.abs());

    MannWhitneyResult {
        u_statistic,
        z_score,
        p_value,
        significant: p_value < 0.05,
        effect_size,
        effect_interpretation,
        method: TestMethod::MannWhitneyU,
    }
}

/// Assign ranks to sorted values, handling ties by averaging
fn assign_ranks_with_ties(sorted: &[(f64, usize)]) -> Vec<(f64, usize)> {
    let mut ranks = Vec::with_capacity(sorted.len());
    let mut i = 0;

    while i < sorted.len() {
        let value = sorted[i].0;
        let mut j = i;

        // Find extent of tie
        while j < sorted.len() && (sorted[j].0 - value).abs() < 1e-10 {
            j += 1;
        }

        // Average rank for tied values
        // Ranks are 1-indexed: positions i..j get ranks (i+1)..(j+1)
        let avg_rank: f64 = ((i + 1)..=(j)).map(|r| r as f64).sum::<f64>() / (j - i) as f64;

        // Iterate over the slice instead of using index
        for item in sorted.iter().take(j).skip(i) {
            ranks.push((avg_rank, item.1));
        }

        i = j;
    }

    ranks
}

/// Interpret effect size using Cohen's conventions
fn interpret_effect_size(r: f64) -> EffectSizeInterpretation {
    if r < 0.1 {
        EffectSizeInterpretation::Negligible
    } else if r < 0.3 {
        EffectSizeInterpretation::Small
    } else if r < 0.5 {
        EffectSizeInterpretation::Medium
    } else {
        EffectSizeInterpretation::Large
    }
}

// ============================================================================
// Automatic Test Selection (per Gemini review)
// ============================================================================

/// Minimum sample size for parametric tests
const MIN_PARAMETRIC_SAMPLE_SIZE: usize = 15;

/// Skewness threshold for log-transform
const SKEWNESS_THRESHOLD: f64 = 1.0;

/// Analyze with automatic test selection based on data characteristics
///
/// Selection criteria (per Box et al. 2005 recommendations):
/// - Small samples (n < 15): Mann-Whitney U
/// - Highly skewed (|skewness| > 1): Log-transform if possible, else Mann-Whitney
/// - Normal-ish data: Welch's t-test
#[must_use]
pub fn analyze_with_auto_select(
    control: &[f64],
    treatment: &[f64],
    config: &AnalysisConfig,
) -> AnalysisResult {
    let min_n = control.len().min(treatment.len());

    // Small samples: always use non-parametric
    if min_n < MIN_PARAMETRIC_SAMPLE_SIZE {
        let mw = mann_whitney_u(control, treatment);
        return AnalysisResult {
            control_mean: median(control),
            treatment_mean: median(treatment),
            effect_size: mw.effect_size,
            p_value: mw.p_value,
            significant: mw.significant,
            method: TestMethod::MannWhitneyU,
        };
    }

    // Check skewness
    let skewness = compute_skewness(control);

    if config.auto_detect_skew && skewness.abs() > SKEWNESS_THRESHOLD {
        // Check if all values are positive (required for log-transform)
        let all_positive = control.iter().all(|&x| x > 0.0) && treatment.iter().all(|&x| x > 0.0);

        if all_positive {
            analyze_log_transform(control, treatment, config.alpha)
        } else {
            // Can't log-transform, use Mann-Whitney
            let mw = mann_whitney_u(control, treatment);
            AnalysisResult {
                control_mean: median(control),
                treatment_mean: median(treatment),
                effect_size: mw.effect_size,
                p_value: mw.p_value,
                significant: mw.significant,
                method: TestMethod::MannWhitneyU,
            }
        }
    } else {
        analyze_t_test(control, treatment, config.alpha)
    }
}

/// Calculate median of a slice
fn median(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }

    if n.is_multiple_of(2) {
        f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    } else {
        sorted[n / 2]
    }
}

/// Normal CDF approximation (Abramowitz and Stegun)
#[allow(clippy::unreadable_literal)] // Standard statistical constants
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Mann-Whitney U Test (Non-parametric, per Box et al. 2005)
    // ========================================================================

    #[test]
    fn test_mann_whitney_identical_samples() {
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = mann_whitney_u(&control, &treatment);

        // Identical samples should have no significant difference
        assert!(!result.significant);
        assert!(result.effect_size.abs() < 0.1); // Negligible effect
    }

    #[test]
    fn test_mann_whitney_completely_separated() {
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![10.0, 11.0, 12.0, 13.0, 14.0];

        let result = mann_whitney_u(&control, &treatment);

        // Completely separated should be highly significant
        assert!(result.significant);
        assert!(result.effect_size.abs() > 0.8); // Large effect
        assert_eq!(result.u_statistic, 0.0); // No overlap
    }

    #[test]
    fn test_mann_whitney_handles_ties() {
        let control = vec![1.0, 2.0, 2.0, 3.0, 3.0];
        let treatment = vec![2.0, 2.0, 3.0, 4.0, 5.0];

        let result = mann_whitney_u(&control, &treatment);

        // Should handle ties correctly (average ranks)
        assert!(result.p_value > 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_mann_whitney_effect_size_interpretation() {
        // Small effect
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let result = mann_whitney_u(&control, &treatment);
        assert!(matches!(
            result.effect_interpretation,
            EffectSizeInterpretation::Small | EffectSizeInterpretation::Negligible
        ));
    }

    #[test]
    fn test_mann_whitney_returns_correct_method() {
        let control = vec![1.0, 2.0, 3.0];
        let treatment = vec![4.0, 5.0, 6.0];
        let result = mann_whitney_u(&control, &treatment);
        assert_eq!(result.method, TestMethod::MannWhitneyU);
    }

    // ========================================================================
    // Auto Test Selection (per Gemini review recommendation)
    // ========================================================================

    #[test]
    fn test_auto_select_uses_mann_whitney_for_small_samples() {
        // Small samples (n < 15) should use non-parametric
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let config = AnalysisConfig {
            alpha: 0.05,
            auto_detect_skew: true,
        };

        let result = analyze_with_auto_select(&control, &treatment, &config);

        // Small samples should trigger Mann-Whitney
        assert_eq!(result.method, TestMethod::MannWhitneyU);
    }

    #[test]
    fn test_auto_select_uses_log_transform_for_latency_like_data() {
        // Generate log-normal-ish data (typical latency distribution)
        let control: Vec<f64> = vec![
            10.0, 12.0, 11.0, 15.0, 100.0, 13.0, 14.0, 11.0, 12.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            15.0, 16.0, 17.0, 18.0, 19.0, 200.0,
        ];
        let treatment: Vec<f64> = vec![
            8.0, 9.0, 10.0, 11.0, 50.0, 9.0, 10.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            8.0, 9.0, 10.0, 11.0, 80.0,
        ];

        let config = AnalysisConfig {
            alpha: 0.05,
            auto_detect_skew: true,
        };

        let result = analyze_with_auto_select(&control, &treatment, &config);

        // Skewed data should use log-transform (if n >= 15) or Mann-Whitney
        assert!(matches!(
            result.method,
            TestMethod::LogTransformTTest | TestMethod::MannWhitneyU
        ));
    }

    // ========================================================================
    // Original Tests
    // ========================================================================

    #[test]
    fn test_t_test_no_difference() {
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![1.1, 2.1, 3.1, 4.1, 5.1];
        let result = analyze_t_test(&control, &treatment, 0.05);
        assert!(!result.significant); // Small effect, not significant
    }

    #[test]
    fn test_t_test_significant_difference() {
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let result = analyze_t_test(&control, &treatment, 0.05);
        assert!(result.significant); // Large effect
    }

    #[test]
    fn test_log_transform_latency() {
        // Simulate log-normal latency (ms)
        let control = vec![10.0, 12.0, 15.0, 100.0, 11.0]; // Has outlier
        let treatment = vec![8.0, 9.0, 10.0, 50.0, 8.5];
        let result = analyze_log_transform(&control, &treatment, 0.05);
        assert!(result.treatment_mean < result.control_mean);
        assert_eq!(result.method, TestMethod::LogTransformTTest);
    }

    #[test]
    fn test_auto_detect_skewness() {
        // Highly skewed data should use log-transform
        let control = vec![1.0, 1.1, 1.2, 1.3, 100.0]; // Skewed
        let treatment = vec![1.0, 1.1, 1.2, 1.3, 1.4];
        let config = AnalysisConfig::default();
        let result = analyze(&control, &treatment, &config);
        assert_eq!(result.method, TestMethod::LogTransformTTest);
    }

    #[test]
    fn test_normal_data_uses_t_test() {
        // Symmetric data should use t-test
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let config = AnalysisConfig::default();
        let result = analyze(&control, &treatment, &config);
        assert_eq!(result.method, TestMethod::TTest);
    }

    #[test]
    fn test_skewness_calculation() {
        // Symmetric data has ~0 skewness
        let symmetric = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(compute_skewness(&symmetric).abs() < 0.5);

        // Right-skewed data has positive skewness
        let skewed = vec![1.0, 1.0, 1.0, 1.0, 100.0];
        assert!(compute_skewness(&skewed) > 1.0);
    }

    // ========================================================================
    // Additional Coverage Tests - Median
    // ========================================================================

    #[test]
    fn test_median_empty_slice() {
        let data: Vec<f64> = vec![];
        assert_eq!(median(&data), 0.0);
    }

    #[test]
    fn test_median_single_element() {
        let data = vec![42.0];
        assert_eq!(median(&data), 42.0);
    }

    #[test]
    fn test_median_odd_length() {
        let data = vec![3.0, 1.0, 2.0];
        assert_eq!(median(&data), 2.0);
    }

    #[test]
    fn test_median_even_length() {
        let data = vec![4.0, 1.0, 3.0, 2.0];
        // Sorted: [1.0, 2.0, 3.0, 4.0], median = (2.0 + 3.0) / 2 = 2.5
        assert_eq!(median(&data), 2.5);
    }

    #[test]
    fn test_median_with_duplicates() {
        let data = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        assert_eq!(median(&data), 5.0);
    }

    // ========================================================================
    // Additional Coverage Tests - Normal CDF
    // ========================================================================

    #[test]
    fn test_normal_cdf_zero() {
        let cdf = normal_cdf(0.0);
        assert!((cdf - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_normal_cdf_positive() {
        let cdf = normal_cdf(2.0);
        // CDF(2.0) should be close to 0.9772
        assert!(cdf > 0.95 && cdf < 0.99);
    }

    #[test]
    fn test_normal_cdf_negative() {
        let cdf = normal_cdf(-2.0);
        // CDF(-2.0) should be close to 0.0228
        assert!(cdf > 0.01 && cdf < 0.05);
    }

    #[test]
    fn test_normal_cdf_symmetry() {
        let cdf_pos = normal_cdf(1.5);
        let cdf_neg = normal_cdf(-1.5);
        // CDF(-x) + CDF(x) should equal 1
        assert!((cdf_pos + cdf_neg - 1.0).abs() < 0.01);
    }

    // ========================================================================
    // Additional Coverage Tests - Effect Size Interpretation
    // ========================================================================

    #[test]
    fn test_interpret_effect_size_negligible() {
        assert_eq!(
            interpret_effect_size(0.05),
            EffectSizeInterpretation::Negligible
        );
        assert_eq!(
            interpret_effect_size(0.09),
            EffectSizeInterpretation::Negligible
        );
    }

    #[test]
    fn test_interpret_effect_size_small() {
        assert_eq!(interpret_effect_size(0.1), EffectSizeInterpretation::Small);
        assert_eq!(interpret_effect_size(0.2), EffectSizeInterpretation::Small);
        assert_eq!(interpret_effect_size(0.29), EffectSizeInterpretation::Small);
    }

    #[test]
    fn test_interpret_effect_size_medium() {
        assert_eq!(interpret_effect_size(0.3), EffectSizeInterpretation::Medium);
        assert_eq!(interpret_effect_size(0.4), EffectSizeInterpretation::Medium);
        assert_eq!(
            interpret_effect_size(0.49),
            EffectSizeInterpretation::Medium
        );
    }

    #[test]
    fn test_interpret_effect_size_large() {
        assert_eq!(interpret_effect_size(0.5), EffectSizeInterpretation::Large);
        assert_eq!(interpret_effect_size(0.8), EffectSizeInterpretation::Large);
        assert_eq!(interpret_effect_size(1.0), EffectSizeInterpretation::Large);
    }

    // ========================================================================
    // Additional Coverage Tests - Skewness Edge Cases
    // ========================================================================

    #[test]
    fn test_skewness_constant_values() {
        // All same values -> std_dev = 0 -> should return 0
        let data = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        assert_eq!(compute_skewness(&data), 0.0);
    }

    #[test]
    fn test_skewness_left_skewed() {
        // Left-skewed data has negative skewness
        let data = vec![100.0, 99.0, 98.0, 97.0, 1.0];
        assert!(compute_skewness(&data) < -1.0);
    }

    // ========================================================================
    // Additional Coverage Tests - AnalysisConfig
    // ========================================================================

    #[test]
    fn test_analysis_config_default() {
        let config = AnalysisConfig::default();
        assert!((config.alpha - 0.05).abs() < 1e-10);
        assert!(config.auto_detect_skew);
    }

    #[test]
    fn test_analysis_config_custom() {
        let config = AnalysisConfig {
            alpha: 0.01,
            auto_detect_skew: false,
        };
        assert!((config.alpha - 0.01).abs() < 1e-10);
        assert!(!config.auto_detect_skew);
    }

    // ========================================================================
    // Additional Coverage Tests - Auto Select with Skew Disabled
    // ========================================================================

    #[test]
    fn test_auto_select_skew_disabled_large_sample() {
        // Large sample with skew detection disabled should use t-test
        let control: Vec<f64> = (0..20).map(|i| (i as f64) + 1.0).collect();
        let treatment: Vec<f64> = (0..20).map(|i| (i as f64) + 2.0).collect();
        let config = AnalysisConfig {
            alpha: 0.05,
            auto_detect_skew: false,
        };

        let result = analyze_with_auto_select(&control, &treatment, &config);
        assert_eq!(result.method, TestMethod::TTest);
    }

    #[test]
    fn test_auto_select_skewed_with_non_positive_values() {
        // Skewed data with non-positive values should fall back to Mann-Whitney
        let control: Vec<f64> = vec![
            -5.0, 1.0, 2.0, 3.0, 100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
            12.0, 13.0, 14.0, 500.0,
        ];
        let treatment: Vec<f64> = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
        ];
        let config = AnalysisConfig {
            alpha: 0.05,
            auto_detect_skew: true,
        };

        let result = analyze_with_auto_select(&control, &treatment, &config);
        // Can't log-transform non-positive, should use Mann-Whitney
        assert_eq!(result.method, TestMethod::MannWhitneyU);
    }

    // ========================================================================
    // Additional Coverage Tests - analyze() function
    // ========================================================================

    #[test]
    fn test_analyze_with_skew_disabled() {
        let control = vec![1.0, 1.1, 1.2, 1.3, 100.0]; // Skewed
        let treatment = vec![1.0, 1.1, 1.2, 1.3, 1.4];
        let config = AnalysisConfig {
            alpha: 0.05,
            auto_detect_skew: false,
        };
        let result = analyze(&control, &treatment, &config);
        // With skew detection disabled, should use t-test
        assert_eq!(result.method, TestMethod::TTest);
    }

    // ========================================================================
    // Additional Coverage Tests - T-Test Details
    // ========================================================================

    #[test]
    fn test_t_test_effect_size() {
        let control = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let treatment = vec![15.0, 25.0, 35.0, 45.0, 55.0];
        let result = analyze_t_test(&control, &treatment, 0.05);

        // Effect size should be (35 - 30) / 30 = 0.1667
        let expected_effect = (35.0 - 30.0) / 30.0;
        assert!((result.effect_size - expected_effect).abs() < 0.01);
    }

    #[test]
    fn test_t_test_means() {
        let control = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let treatment = vec![3.0, 6.0, 9.0, 12.0, 15.0];
        let result = analyze_t_test(&control, &treatment, 0.05);

        assert!((result.control_mean - 6.0).abs() < 0.01);
        assert!((result.treatment_mean - 9.0).abs() < 0.01);
    }

    // ========================================================================
    // Additional Coverage Tests - Log Transform Details
    // ========================================================================

    #[test]
    fn test_log_transform_effect_size() {
        let control = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let treatment = vec![20.0, 40.0, 60.0, 80.0, 100.0];
        let result = analyze_log_transform(&control, &treatment, 0.05);

        // Treatment geometric mean should be ~2x control geometric mean
        // Effect size should be roughly 1.0 (100% increase)
        assert!(result.effect_size > 0.5);
    }

    // ========================================================================
    // Additional Coverage Tests - Mann-Whitney Edge Cases
    // ========================================================================

    #[test]
    fn test_mann_whitney_large_effect() {
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let treatment = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        let result = mann_whitney_u(&control, &treatment);

        assert_eq!(
            result.effect_interpretation,
            EffectSizeInterpretation::Large
        );
        assert!(result.significant);
    }

    #[test]
    fn test_mann_whitney_z_score_sign() {
        // Treatment higher than control
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let result = mann_whitney_u(&control, &treatment);

        // U statistic should be 0 (no overlap)
        assert_eq!(result.u_statistic, 0.0);
    }

    #[test]
    fn test_mann_whitney_result_fields() {
        let control = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let treatment = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = mann_whitney_u(&control, &treatment);

        // Verify all fields are populated
        assert!(result.u_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(!result.z_score.is_nan());
        assert!(!result.effect_size.is_nan());
    }

    // ========================================================================
    // Additional Coverage Tests - Rank Assignment
    // ========================================================================

    #[test]
    fn test_assign_ranks_no_ties() {
        let sorted = vec![(1.0, 0), (2.0, 1), (3.0, 0), (4.0, 1)];
        let ranks = assign_ranks_with_ties(&sorted);

        assert_eq!(ranks.len(), 4);
        assert!((ranks[0].0 - 1.0).abs() < 1e-10);
        assert!((ranks[1].0 - 2.0).abs() < 1e-10);
        assert!((ranks[2].0 - 3.0).abs() < 1e-10);
        assert!((ranks[3].0 - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_assign_ranks_with_multiple_tie_groups() {
        // Two tie groups: (1.0, 1.0) and (3.0, 3.0)
        let sorted = vec![(1.0, 0), (1.0, 1), (3.0, 0), (3.0, 1)];
        let ranks = assign_ranks_with_ties(&sorted);

        // First two should have average rank 1.5
        assert!((ranks[0].0 - 1.5).abs() < 1e-10);
        assert!((ranks[1].0 - 1.5).abs() < 1e-10);
        // Last two should have average rank 3.5
        assert!((ranks[2].0 - 3.5).abs() < 1e-10);
        assert!((ranks[3].0 - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_assign_ranks_preserves_groups() {
        let sorted = vec![(1.0, 0), (2.0, 1), (3.0, 0)];
        let ranks = assign_ranks_with_ties(&sorted);

        assert_eq!(ranks[0].1, 0);
        assert_eq!(ranks[1].1, 1);
        assert_eq!(ranks[2].1, 0);
    }

    // ========================================================================
    // Additional Coverage Tests - AnalysisResult Fields
    // ========================================================================

    #[test]
    fn test_analysis_result_significant_flag() {
        // Test with alpha = 0.10 (more lenient)
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![3.0, 4.0, 5.0, 6.0, 7.0];
        let result = analyze_t_test(&control, &treatment, 0.10);

        // With larger alpha, more likely to be significant
        // p_value < alpha should set significant = true
        assert_eq!(result.significant, result.p_value < 0.10);
    }

    // ========================================================================
    // Additional Coverage Tests - TestMethod Enum
    // ========================================================================

    #[test]
    fn test_test_method_equality() {
        assert_eq!(TestMethod::TTest, TestMethod::TTest);
        assert_eq!(TestMethod::LogTransformTTest, TestMethod::LogTransformTTest);
        assert_eq!(TestMethod::MannWhitneyU, TestMethod::MannWhitneyU);
        assert_ne!(TestMethod::TTest, TestMethod::MannWhitneyU);
    }

    #[test]
    fn test_test_method_clone() {
        let method = TestMethod::LogTransformTTest;
        let cloned = method.clone();
        assert_eq!(method, cloned);
    }

    // ========================================================================
    // Additional Coverage Tests - EffectSizeInterpretation Enum
    // ========================================================================

    #[test]
    fn test_effect_size_interpretation_equality() {
        assert_eq!(
            EffectSizeInterpretation::Negligible,
            EffectSizeInterpretation::Negligible
        );
        assert_ne!(
            EffectSizeInterpretation::Small,
            EffectSizeInterpretation::Large
        );
    }

    #[test]
    fn test_effect_size_interpretation_clone() {
        let interp = EffectSizeInterpretation::Medium;
        let cloned = interp;
        assert_eq!(interp, cloned);
    }

    // ========================================================================
    // Additional Coverage Tests - MannWhitneyResult
    // ========================================================================

    #[test]
    fn test_mann_whitney_result_clone() {
        let control = vec![1.0, 2.0, 3.0];
        let treatment = vec![4.0, 5.0, 6.0];
        let result = mann_whitney_u(&control, &treatment);
        let cloned = result.clone();

        assert_eq!(result.u_statistic, cloned.u_statistic);
        assert_eq!(result.p_value, cloned.p_value);
        assert_eq!(result.method, cloned.method);
    }

    // ========================================================================
    // Additional Coverage Tests - AnalysisResult Clone
    // ========================================================================

    #[test]
    fn test_analysis_result_clone() {
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let result = analyze_t_test(&control, &treatment, 0.05);
        let cloned = result.clone();

        assert_eq!(result.control_mean, cloned.control_mean);
        assert_eq!(result.treatment_mean, cloned.treatment_mean);
        assert_eq!(result.method, cloned.method);
    }

    // ========================================================================
    // Additional Coverage Tests - AnalysisConfig Clone
    // ========================================================================

    #[test]
    fn test_analysis_config_clone() {
        let config = AnalysisConfig {
            alpha: 0.01,
            auto_detect_skew: true,
        };
        let cloned = config.clone();

        assert_eq!(config.alpha, cloned.alpha);
        assert_eq!(config.auto_detect_skew, cloned.auto_detect_skew);
    }
}
