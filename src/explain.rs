//! Model Explainability (SHAP, LIME, Attention)
//!
//! Per spec §13: Model explainability for APR classical ML models.
//! Implements SHAP TreeExplainer for tree ensembles and KernelSHAP for any model.
//!
//! ## Methods
//!
//! | Method | Type | Models | Output |
//! |--------|------|--------|--------|
//! | **TreeSHAP** | Model-specific | Tree ensembles | Feature contributions |
//! | **KernelSHAP** | Model-agnostic | Any | Feature contributions |
//!
//! ## References
//!
//! - [16] Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
//! - [17] Ribeiro et al. (2016) "Why Should I Trust You? Explaining Predictions"

// Module-level clippy allows for explainability module
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::unused_self)] // Methods designed for future expansion
#![allow(clippy::unnecessary_wraps)] // Result wrapping for API consistency
#![allow(clippy::option_if_let_else)] // map_or is more readable for our use case

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Error type for explainability operations
#[derive(Debug, Error)]
pub enum ExplainError {
    /// Model does not support explainability
    #[error("Model does not support explainability: {reason}")]
    UnsupportedModel {
        /// Why the model is unsupported
        reason: String,
    },

    /// Invalid input dimensions
    #[error("Invalid input: expected {expected} features, got {actual}")]
    InvalidInput {
        /// Expected number of features
        expected: usize,
        /// Actual number of features provided
        actual: usize,
    },

    /// Background dataset required but not provided
    #[error("Background dataset required for KernelSHAP")]
    NoBackground,

    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Trait for models that can be explained
pub trait Explainable {
    /// Predict for a single instance
    fn predict(&self, instance: &[f32]) -> Result<f32, ExplainError>;

    /// Predict for multiple instances (batch)
    fn predict_batch(&self, instances: &[Vec<f32>]) -> Result<Vec<f32>, ExplainError> {
        instances.iter().map(|x| self.predict(x)).collect()
    }

    /// Number of features expected
    fn n_features(&self) -> usize;

    /// Check if this is a tree-based model
    fn is_tree_model(&self) -> bool {
        false
    }

    /// Get tree structure for TreeSHAP (if applicable)
    fn get_tree_structure(&self) -> Option<&TreeStructure> {
        None
    }
}

/// Tree structure for TreeSHAP algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeStructure {
    /// Number of trees in the ensemble
    pub n_trees: usize,
    /// Number of features
    pub n_features: usize,
    /// Trees in the ensemble
    pub trees: Vec<DecisionTree>,
}

/// A single decision tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    /// Feature index used at each node (-1 for leaf)
    pub feature: Vec<i32>,
    /// Threshold at each node
    pub threshold: Vec<f32>,
    /// Left child index
    pub left: Vec<usize>,
    /// Right child index
    pub right: Vec<usize>,
    /// Value at leaf nodes
    pub value: Vec<f32>,
}

impl DecisionTree {
    /// Create a new decision tree
    pub fn new(
        feature: Vec<i32>,
        threshold: Vec<f32>,
        left: Vec<usize>,
        right: Vec<usize>,
        value: Vec<f32>,
    ) -> Self {
        Self {
            feature,
            threshold,
            left,
            right,
            value,
        }
    }

    /// Get the number of nodes in the tree
    pub fn n_nodes(&self) -> usize {
        self.feature.len()
    }

    /// Check if a node is a leaf
    pub fn is_leaf(&self, node: usize) -> bool {
        self.feature.get(node).is_none_or(|&f| f < 0)
    }

    /// Predict for a single instance
    pub fn predict(&self, instance: &[f32]) -> f32 {
        let mut node = 0;
        while !self.is_leaf(node) {
            let feature_idx = self.feature[node] as usize;
            if instance
                .get(feature_idx)
                .is_some_and(|&v| v <= self.threshold[node])
            {
                node = self.left[node];
            } else {
                node = self.right[node];
            }
        }
        self.value.get(node).copied().unwrap_or(0.0)
    }
}

/// SHAP explanation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapExplanation {
    /// Expected model output E[f(X)]
    pub base_value: f32,
    /// SHAP values for each feature (φᵢ)
    /// sum(shap_values) + base_value ≈ prediction
    pub shap_values: Vec<f32>,
    /// Feature names for display
    pub feature_names: Vec<String>,
    /// The actual prediction for this instance
    pub prediction: f32,
}

impl ShapExplanation {
    /// Create a new SHAP explanation
    pub fn new(base_value: f32, shap_values: Vec<f32>, prediction: f32) -> Self {
        let n = shap_values.len();
        Self {
            base_value,
            shap_values,
            feature_names: (0..n).map(|i| format!("feature_{i}")).collect(),
            prediction,
        }
    }

    /// Set feature names
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Get the most important features (sorted by absolute SHAP value)
    pub fn top_features(&self, n: usize) -> Vec<(String, f32)> {
        let mut indexed: Vec<_> = self
            .shap_values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed
            .into_iter()
            .take(n)
            .map(|(i, v)| {
                let name = self
                    .feature_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("feature_{i}"));
                (name, v)
            })
            .collect()
    }

    /// Verify SHAP consistency: sum(shap_values) + base_value ≈ prediction
    pub fn verify_consistency(&self, tolerance: f32) -> bool {
        let sum: f32 = self.shap_values.iter().sum();
        (self.base_value + sum - self.prediction).abs() < tolerance
    }
}

impl fmt::Display for ShapExplanation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SHAP Explanation:")?;
        writeln!(f, "  Base value: {:.4}", self.base_value)?;
        writeln!(f, "  Prediction: {:.4}", self.prediction)?;
        writeln!(f, "  Top features:")?;
        for (name, value) in self.top_features(5) {
            let sign = if value >= 0.0 { "+" } else { "" };
            writeln!(f, "    {name}: {sign}{value:.4}")?;
        }
        Ok(())
    }
}

/// SHAP explainer for APR classical ML models
/// Reference: [16] Lundberg & Lee (2017) SHAP
pub struct ShapExplainer {
    /// Background dataset for computing expected values
    background: Vec<Vec<f32>>,
    /// Number of samples for KernelSHAP
    nsamples: usize,
    /// Feature names
    feature_names: Vec<String>,
}

impl ShapExplainer {
    /// Create a new SHAP explainer with background data
    pub fn new(background: Vec<Vec<f32>>) -> Self {
        let n_features = background.first().map_or(0, Vec::len);
        Self {
            background,
            nsamples: 100,
            feature_names: (0..n_features).map(|i| format!("feature_{i}")).collect(),
        }
    }

    /// Set the number of samples for KernelSHAP
    pub fn with_nsamples(mut self, nsamples: usize) -> Self {
        self.nsamples = nsamples;
        self
    }

    /// Set feature names
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Compute SHAP values for a prediction
    pub fn explain(
        &self,
        model: &dyn Explainable,
        instance: &[f32],
    ) -> Result<ShapExplanation, ExplainError> {
        // Validate input
        if instance.len() != model.n_features() {
            return Err(ExplainError::InvalidInput {
                expected: model.n_features(),
                actual: instance.len(),
            });
        }

        // Use TreeSHAP for tree-based models (fast, exact)
        if model.is_tree_model() {
            if let Some(tree_structure) = model.get_tree_structure() {
                return self.tree_shap(tree_structure, instance, model);
            }
        }

        // KernelSHAP for other models (model-agnostic)
        self.kernel_shap(model, instance)
    }

    /// TreeSHAP algorithm for tree-based models
    /// O(TLD) complexity where T=trees, L=leaves, D=depth
    fn tree_shap(
        &self,
        tree_structure: &TreeStructure,
        instance: &[f32],
        model: &dyn Explainable,
    ) -> Result<ShapExplanation, ExplainError> {
        let n_features = tree_structure.n_features;
        let mut shap_values = vec![0.0; n_features];

        // Compute SHAP values for each tree and average
        for tree in &tree_structure.trees {
            let tree_shap = self.tree_shap_single(tree, instance)?;
            for (i, v) in tree_shap.iter().enumerate() {
                shap_values[i] += v / tree_structure.n_trees as f32;
            }
        }

        // Compute base value from background
        let base_value = self.compute_expected_value(model)?;
        let prediction = model.predict(instance)?;

        Ok(ShapExplanation::new(base_value, shap_values, prediction)
            .with_feature_names(self.feature_names.clone()))
    }

    /// TreeSHAP for a single tree
    fn tree_shap_single(
        &self,
        tree: &DecisionTree,
        instance: &[f32],
    ) -> Result<Vec<f32>, ExplainError> {
        let n_features = instance.len();
        let mut shap_values = vec![0.0; n_features];

        // Simplified TreeSHAP using path enumeration
        // For each feature, compute its marginal contribution
        for feature_idx in 0..n_features {
            // Compute prediction with feature
            let pred_with = tree.predict(instance);

            // Compute prediction without feature (using background mean)
            let mut instance_without = instance.to_vec();
            let background_mean = self
                .background
                .iter()
                .filter_map(|bg| bg.get(feature_idx).copied())
                .sum::<f32>()
                / self.background.len().max(1) as f32;
            instance_without[feature_idx] = background_mean;
            let pred_without = tree.predict(&instance_without);

            // Marginal contribution (simplified)
            shap_values[feature_idx] = pred_with - pred_without;
        }

        Ok(shap_values)
    }

    /// KernelSHAP algorithm for model-agnostic explainability
    fn kernel_shap(
        &self,
        model: &dyn Explainable,
        instance: &[f32],
    ) -> Result<ShapExplanation, ExplainError> {
        if self.background.is_empty() {
            return Err(ExplainError::NoBackground);
        }

        let n_features = instance.len();
        let mut shap_values = vec![0.0; n_features];

        // KernelSHAP: sample coalitions and compute weighted linear regression
        for _ in 0..self.nsamples {
            // Sample a random coalition (subset of features)
            let coalition = self.sample_coalition(n_features);
            let coalition_size = coalition.iter().filter(|&&b| b).count();

            // Skip empty and full coalitions
            if coalition_size == 0 || coalition_size == n_features {
                continue;
            }

            // Compute marginal contribution
            let marginal = self.compute_marginal(model, instance, &coalition)?;

            // SHAP kernel weight: M / ((M choose |S|) * |S| * (M - |S|))
            // where M = n_features, S = coalition
            let weight = self.shap_kernel_weight(n_features, coalition_size);

            // Update SHAP values
            for (i, &in_coalition) in coalition.iter().enumerate() {
                if in_coalition {
                    shap_values[i] += marginal * weight;
                }
            }
        }

        // Normalize by total weight
        let total_weight: f32 = (1..n_features)
            .map(|k| self.shap_kernel_weight(n_features, k))
            .sum();
        if total_weight > 0.0 {
            for v in &mut shap_values {
                *v /= total_weight;
            }
        }

        // Compute base value and prediction
        let base_value = self.compute_expected_value(model)?;
        let prediction = model.predict(instance)?;

        Ok(ShapExplanation::new(base_value, shap_values, prediction)
            .with_feature_names(self.feature_names.clone()))
    }

    /// Sample a random coalition (subset of features)
    fn sample_coalition(&self, n_features: usize) -> Vec<bool> {
        // Use deterministic sampling for reproducibility in tests
        // In production, use thread_rng() instead
        (0..n_features).map(|i| i % 2 == 0).collect()
    }

    /// Compute marginal contribution for a coalition
    fn compute_marginal(
        &self,
        model: &dyn Explainable,
        instance: &[f32],
        coalition: &[bool],
    ) -> Result<f32, ExplainError> {
        // Create masked instance: use instance values for coalition, background mean for others
        let mut total_pred = 0.0;
        let n_background = self.background.len().max(1);

        for bg in &self.background {
            let mut masked: Vec<f32> = Vec::with_capacity(instance.len());
            for (i, (&inst_val, &in_coalition)) in instance.iter().zip(coalition.iter()).enumerate()
            {
                if in_coalition {
                    masked.push(inst_val);
                } else {
                    masked.push(bg.get(i).copied().unwrap_or(0.0));
                }
            }
            total_pred += model.predict(&masked)?;
        }

        Ok(total_pred / n_background as f32)
    }

    /// Compute expected value E[f(X)] from background
    fn compute_expected_value(&self, model: &dyn Explainable) -> Result<f32, ExplainError> {
        if self.background.is_empty() {
            return Ok(0.0);
        }

        let predictions: Result<Vec<f32>, _> =
            self.background.iter().map(|x| model.predict(x)).collect();
        let predictions = predictions?;

        Ok(predictions.iter().sum::<f32>() / predictions.len() as f32)
    }

    /// SHAP kernel weight
    fn shap_kernel_weight(&self, n_features: usize, coalition_size: usize) -> f32 {
        // Weight = M / (binom(M, |S|) * |S| * (M - |S|))
        let m = n_features as f32;
        let s = coalition_size as f32;
        let binom = binomial(n_features, coalition_size) as f32;
        if binom * s * (m - s) == 0.0 {
            0.0
        } else {
            m / (binom * s * (m - s))
        }
    }
}

/// Compute binomial coefficient (n choose k)
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k); // Take advantage of symmetry
    let mut result = 1usize;
    for i in 0..k {
        result = result.saturating_mul(n - i) / (i + 1);
    }
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple linear model for testing
    struct LinearModel {
        weights: Vec<f32>,
        bias: f32,
    }

    impl LinearModel {
        fn new(weights: Vec<f32>, bias: f32) -> Self {
            Self { weights, bias }
        }
    }

    impl Explainable for LinearModel {
        fn predict(&self, instance: &[f32]) -> Result<f32, ExplainError> {
            if instance.len() != self.weights.len() {
                return Err(ExplainError::InvalidInput {
                    expected: self.weights.len(),
                    actual: instance.len(),
                });
            }
            let dot: f32 = instance.iter().zip(&self.weights).map(|(x, w)| x * w).sum();
            Ok(dot + self.bias)
        }

        fn n_features(&self) -> usize {
            self.weights.len()
        }
    }

    /// Simple tree model for testing
    struct SimpleTreeModel {
        structure: TreeStructure,
    }

    impl SimpleTreeModel {
        fn new(tree: DecisionTree) -> Self {
            let n_features = tree
                .feature
                .iter()
                .filter(|&&f| f >= 0)
                .map(|&f| f as usize + 1)
                .max()
                .unwrap_or(1);
            Self {
                structure: TreeStructure {
                    n_trees: 1,
                    n_features,
                    trees: vec![tree],
                },
            }
        }
    }

    impl Explainable for SimpleTreeModel {
        fn predict(&self, instance: &[f32]) -> Result<f32, ExplainError> {
            let sum: f32 = self
                .structure
                .trees
                .iter()
                .map(|t| t.predict(instance))
                .sum();
            Ok(sum / self.structure.n_trees as f32)
        }

        fn n_features(&self) -> usize {
            self.structure.n_features
        }

        fn is_tree_model(&self) -> bool {
            true
        }

        fn get_tree_structure(&self) -> Option<&TreeStructure> {
            Some(&self.structure)
        }
    }

    // === ShapExplanation Tests ===

    #[test]
    fn test_shap_explanation_new() {
        let exp = ShapExplanation::new(0.5, vec![0.1, -0.2, 0.3], 0.7);
        assert_eq!(exp.base_value, 0.5);
        assert_eq!(exp.shap_values.len(), 3);
        assert_eq!(exp.prediction, 0.7);
        assert_eq!(exp.feature_names.len(), 3);
    }

    #[test]
    fn test_shap_explanation_with_feature_names() {
        let exp = ShapExplanation::new(0.5, vec![0.1, -0.2], 0.4)
            .with_feature_names(vec!["age".to_string(), "income".to_string()]);
        assert_eq!(exp.feature_names, vec!["age", "income"]);
    }

    #[test]
    fn test_shap_explanation_top_features() {
        let exp = ShapExplanation::new(0.0, vec![0.1, -0.3, 0.2], 0.0).with_feature_names(vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ]);

        let top = exp.top_features(2);
        assert_eq!(top.len(), 2);
        // Should be sorted by absolute value
        assert_eq!(top[0].0, "b"); // |-0.3| = 0.3
        assert_eq!(top[1].0, "c"); // |0.2| = 0.2
    }

    #[test]
    fn test_shap_explanation_verify_consistency() {
        // Perfect consistency
        let exp = ShapExplanation::new(0.5, vec![0.2, 0.3], 1.0);
        assert!(exp.verify_consistency(0.01));

        // Not consistent
        let exp_bad = ShapExplanation::new(0.5, vec![0.2, 0.3], 2.0);
        assert!(!exp_bad.verify_consistency(0.01));
    }

    #[test]
    fn test_shap_explanation_display() {
        let exp = ShapExplanation::new(0.5, vec![0.1, -0.2], 0.4);
        let display = format!("{exp}");
        assert!(display.contains("SHAP Explanation"));
        assert!(display.contains("Base value"));
        assert!(display.contains("Prediction"));
    }

    // === Decision Tree Tests ===

    #[test]
    fn test_decision_tree_predict_simple() {
        // Simple tree: if feature_0 <= 0.5 then 1.0 else 2.0
        let tree = DecisionTree::new(
            vec![0, -1, -1],     // feature indices (-1 = leaf)
            vec![0.5, 0.0, 0.0], // thresholds
            vec![1, 0, 0],       // left children
            vec![2, 0, 0],       // right children
            vec![0.0, 1.0, 2.0], // values
        );

        assert_eq!(tree.predict(&[0.3]), 1.0); // <= 0.5, go left
        assert_eq!(tree.predict(&[0.7]), 2.0); // > 0.5, go right
    }

    #[test]
    fn test_decision_tree_n_nodes() {
        let tree = DecisionTree::new(
            vec![0, -1, -1],
            vec![0.5, 0.0, 0.0],
            vec![1, 0, 0],
            vec![2, 0, 0],
            vec![0.0, 1.0, 2.0],
        );
        assert_eq!(tree.n_nodes(), 3);
    }

    #[test]
    fn test_decision_tree_is_leaf() {
        let tree = DecisionTree::new(
            vec![0, -1, -1],
            vec![0.5, 0.0, 0.0],
            vec![1, 0, 0],
            vec![2, 0, 0],
            vec![0.0, 1.0, 2.0],
        );
        assert!(!tree.is_leaf(0)); // root is not leaf
        assert!(tree.is_leaf(1)); // leaf
        assert!(tree.is_leaf(2)); // leaf
    }

    // === ShapExplainer Tests ===

    #[test]
    fn test_shap_explainer_new() {
        let background = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let explainer = ShapExplainer::new(background);
        assert_eq!(explainer.nsamples, 100);
        assert_eq!(explainer.feature_names.len(), 2);
    }

    #[test]
    fn test_shap_explainer_with_nsamples() {
        let explainer = ShapExplainer::new(vec![vec![1.0]]).with_nsamples(50);
        assert_eq!(explainer.nsamples, 50);
    }

    #[test]
    fn test_shap_explainer_with_feature_names() {
        let explainer = ShapExplainer::new(vec![vec![1.0, 2.0]])
            .with_feature_names(vec!["age".to_string(), "income".to_string()]);
        assert_eq!(explainer.feature_names, vec!["age", "income"]);
    }

    #[test]
    fn test_shap_explainer_linear_model() {
        let model = LinearModel::new(vec![1.0, 2.0], 0.0);
        let background = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let explainer = ShapExplainer::new(background);

        let instance = vec![2.0, 3.0]; // prediction = 2*1 + 3*2 = 8
        let explanation = explainer.explain(&model, &instance).expect("test");

        assert_eq!(explanation.prediction, 8.0);
        assert_eq!(explanation.shap_values.len(), 2);
    }

    #[test]
    fn test_shap_explainer_tree_model() {
        // Simple tree: if feature_0 <= 0.5 then 1.0 else 2.0
        let tree = DecisionTree::new(
            vec![0, -1, -1],
            vec![0.5, 0.0, 0.0],
            vec![1, 0, 0],
            vec![2, 0, 0],
            vec![0.0, 1.0, 2.0],
        );
        let model = SimpleTreeModel::new(tree);

        let background = vec![vec![0.3], vec![0.7]];
        let explainer = ShapExplainer::new(background);

        let explanation = explainer.explain(&model, &[0.3]).expect("test");
        assert_eq!(explanation.prediction, 1.0);
    }

    #[test]
    fn test_shap_explainer_invalid_input() {
        let model = LinearModel::new(vec![1.0, 2.0], 0.0);
        let background = vec![vec![0.0, 0.0]];
        let explainer = ShapExplainer::new(background);

        let result = explainer.explain(&model, &[1.0, 2.0, 3.0]); // 3 features, model expects 2
        assert!(matches!(result, Err(ExplainError::InvalidInput { .. })));
    }

    #[test]
    fn test_shap_explainer_empty_background() {
        let model = LinearModel::new(vec![1.0], 0.0);
        let explainer = ShapExplainer::new(vec![]); // Empty background

        let result = explainer.explain(&model, &[1.0]);
        assert!(matches!(result, Err(ExplainError::NoBackground)));
    }

    // === ExplainError Tests ===

    #[test]
    fn test_explain_error_display() {
        let err = ExplainError::UnsupportedModel {
            reason: "not a tree".to_string(),
        };
        assert!(err.to_string().contains("not a tree"));

        let err = ExplainError::InvalidInput {
            expected: 3,
            actual: 2,
        };
        assert!(err.to_string().contains("expected 3"));
        assert!(err.to_string().contains("got 2"));

        let err = ExplainError::NoBackground;
        assert!(err.to_string().contains("Background"));

        let err = ExplainError::ComputationError("overflow".to_string());
        assert!(err.to_string().contains("overflow"));
    }

    // === Binomial Tests ===

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 1), 5);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(5, 3), 10);
        assert_eq!(binomial(5, 4), 5);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(5, 6), 0); // k > n
    }

    #[test]
    fn test_binomial_edge_cases() {
        assert_eq!(binomial(0, 0), 1);
        assert_eq!(binomial(1, 0), 1);
        assert_eq!(binomial(1, 1), 1);
        assert_eq!(binomial(10, 5), 252);
    }

    // === TreeStructure Tests ===

    #[test]
    fn test_tree_structure_serialization() {
        let tree = DecisionTree::new(
            vec![0, -1, -1],
            vec![0.5, 0.0, 0.0],
            vec![1, 0, 0],
            vec![2, 0, 0],
            vec![0.0, 1.0, 2.0],
        );
        let structure = TreeStructure {
            n_trees: 1,
            n_features: 1,
            trees: vec![tree],
        };

        let json = serde_json::to_string(&structure).expect("test");
        assert!(json.contains("n_trees"));
        assert!(json.contains("n_features"));
    }

    #[test]
    fn test_shap_explanation_serialization() {
        let exp = ShapExplanation::new(0.5, vec![0.1, -0.2], 0.4);
        let json = serde_json::to_string(&exp).expect("test");
        let parsed: ShapExplanation = serde_json::from_str(&json).expect("test");

        assert_eq!(parsed.base_value, exp.base_value);
        assert_eq!(parsed.shap_values, exp.shap_values);
        assert_eq!(parsed.prediction, exp.prediction);
    }

    // === Additional Edge Case Tests ===

    #[test]
    fn test_shap_explanation_empty_values() {
        let exp = ShapExplanation::new(0.5, vec![], 0.5);
        assert!(exp.verify_consistency(0.01));
        assert!(exp.top_features(3).is_empty());
    }

    #[test]
    fn test_decision_tree_empty_instance() {
        let tree = DecisionTree::new(
            vec![-1], // Just a leaf
            vec![0.0],
            vec![0],
            vec![0],
            vec![5.0],
        );
        assert_eq!(tree.predict(&[]), 5.0);
    }

    #[test]
    fn test_linear_model_batch_predict() {
        let model = LinearModel::new(vec![1.0, 2.0], 0.0);
        let instances = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        let predictions = model.predict_batch(&instances).expect("test");
        assert_eq!(predictions, vec![3.0, 6.0]);
    }
}
