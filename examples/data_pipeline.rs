//! Data Pipeline Example - Alimentar + Realizar Integration
//!
//! Demonstrates end-to-end ML workflow using:
//! - **alimentar**: Data loading, transforms, quality checks, drift detection
//! - **realizar**: Model inference and serving
//!
//! Uses the classic Iris dataset (embedded, no download required).
//!
//! ## Pipeline Stages
//!
//! 1. Load data with alimentar
//! 2. Apply transforms (normalize, shuffle)
//! 3. Check data quality
//! 4. Split into train/test
//! 5. Run inference with realizar
//! 6. Evaluate predictions
//! 7. Monitor for drift
//!
//! ## Run
//!
//! ```bash
//! cargo run --example data_pipeline --features alimentar-data
//! ```

#![cfg(feature = "alimentar-data")]
// Allow expect in examples for cleaner demo code
#![allow(clippy::expect_used)]

use std::collections::HashMap;

// ============================================================================
// Iris Classifier (Simple Linear Model)
// ============================================================================

/// Iris species
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrisSpecies {
    Setosa = 0,
    Versicolor = 1,
    Virginica = 2,
}

impl IrisSpecies {
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => IrisSpecies::Setosa,
            1 => IrisSpecies::Versicolor,
            _ => IrisSpecies::Virginica,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            IrisSpecies::Setosa => "setosa",
            IrisSpecies::Versicolor => "versicolor",
            IrisSpecies::Virginica => "virginica",
        }
    }
}

/// Iris flower features
#[derive(Debug, Clone)]
pub struct IrisFeatures {
    pub sepal_length: f64,
    pub sepal_width: f64,
    pub petal_length: f64,
    pub petal_width: f64,
}

impl IrisFeatures {
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width,
        ]
    }

    /// Normalize features to [0, 1] using typical Iris ranges
    pub fn normalize(&self) -> Vec<f64> {
        vec![
            (self.sepal_length - 4.3) / (7.9 - 4.3), // 4.3-7.9
            (self.sepal_width - 2.0) / (4.4 - 2.0),  // 2.0-4.4
            (self.petal_length - 1.0) / (6.9 - 1.0), // 1.0-6.9
            (self.petal_width - 0.1) / (2.5 - 0.1),  // 0.1-2.5
        ]
    }
}

/// Iris classifier prediction
#[derive(Debug)]
pub struct IrisPrediction {
    pub species: IrisSpecies,
    pub probabilities: [f64; 3],
    pub confidence: f64,
}

/// Simple Iris classifier using learned decision boundaries
///
/// Uses petal measurements as primary discriminators:
/// - Setosa: petal_length < 2.5
/// - Versicolor: petal_length < 4.9 && petal_width < 1.7
/// - Virginica: otherwise
pub struct IrisClassifier {
    /// Inference count
    inference_count: std::sync::atomic::AtomicU64,
}

impl IrisClassifier {
    pub fn new() -> Self {
        Self {
            inference_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Predict species from features
    pub fn predict(&self, features: &IrisFeatures) -> IrisPrediction {
        self.inference_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Decision tree based on petal measurements
        // (Petal features are most discriminative for Iris)
        let (species, probabilities) = if features.petal_length < 2.5 {
            // Setosa has very small petals
            (IrisSpecies::Setosa, [0.98, 0.01, 0.01])
        } else if features.petal_length < 4.9 && features.petal_width < 1.7 {
            // Versicolor has medium petals
            let conf = if features.petal_width < 1.3 {
                0.92
            } else {
                0.78
            };
            (IrisSpecies::Versicolor, [0.02, conf, 1.0 - conf - 0.02])
        } else {
            // Virginica has large petals
            let conf = if features.petal_width > 1.8 {
                0.95
            } else {
                0.82
            };
            (IrisSpecies::Virginica, [0.01, 1.0 - conf - 0.01, conf])
        };

        let confidence = probabilities[species as usize];

        IrisPrediction {
            species,
            probabilities,
            confidence,
        }
    }

    /// Predict batch of samples
    pub fn predict_batch(&self, samples: &[IrisFeatures]) -> Vec<IrisPrediction> {
        samples.iter().map(|f| self.predict(f)).collect()
    }

    pub fn inference_count(&self) -> u64 {
        self.inference_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Default for IrisClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Metrics
// ============================================================================

/// Classification metrics
#[derive(Debug, Default)]
pub struct ClassificationMetrics {
    pub total: usize,
    pub correct: usize,
    pub per_class: HashMap<String, ClassMetrics>,
}

#[derive(Debug, Default, Clone)]
pub struct ClassMetrics {
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

impl ClassMetrics {
    pub fn precision(&self) -> f64 {
        let denom = self.true_positives + self.false_positives;
        if denom == 0 {
            0.0
        } else {
            self.true_positives as f64 / denom as f64
        }
    }

    pub fn recall(&self) -> f64 {
        let denom = self.true_positives + self.false_negatives;
        if denom == 0 {
            0.0
        } else {
            self.true_positives as f64 / denom as f64
        }
    }

    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

impl ClassificationMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, predicted: &str, actual: &str) {
        self.total += 1;
        if predicted == actual {
            self.correct += 1;
        }

        // Update per-class metrics
        let pred_entry = self
            .per_class
            .entry(predicted.to_string())
            .or_insert_with(ClassMetrics::default);
        if predicted == actual {
            pred_entry.true_positives += 1;
        } else {
            pred_entry.false_positives += 1;
        }

        if predicted != actual {
            let actual_entry = self
                .per_class
                .entry(actual.to_string())
                .or_insert_with(ClassMetrics::default);
            actual_entry.false_negatives += 1;
        }
    }

    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }

    pub fn macro_f1(&self) -> f64 {
        if self.per_class.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.per_class.values().map(|m| m.f1()).sum();
        sum / self.per_class.len() as f64
    }
}

// ============================================================================
// Main Pipeline
// ============================================================================

fn main() {
    println!("=== Data Pipeline: Alimentar + Realizar ===\n");

    // -------------------------------------------------------------------------
    // Stage 1: Load data with alimentar
    // -------------------------------------------------------------------------
    println!("Stage 1: Loading Iris dataset with alimentar...");

    use alimentar::datasets::{iris, CanonicalDataset};

    let dataset = iris().expect("Failed to load Iris dataset");

    println!("  Dataset: {}", dataset.description());
    println!("  Samples: {}", dataset.len());
    println!("  Features: {:?}", dataset.feature_names());
    println!("  Classes: {}", dataset.num_classes());

    // -------------------------------------------------------------------------
    // Stage 2: Data quality check
    // -------------------------------------------------------------------------
    println!("\nStage 2: Data quality check...");

    use alimentar::{QualityChecker, QualityReport};

    let checker = QualityChecker::new();
    let report: QualityReport = checker.check(dataset.data()).expect("Quality check failed");

    println!("  Total rows: {}", report.row_count);
    println!("  Total columns: {}", report.column_count);
    println!("  Issues found: {}", report.issues.len());

    if report.issues.is_empty() {
        println!("  Status: PASSED (no quality issues)");
    } else {
        for issue in &report.issues {
            println!("  Warning: {:?}", issue);
        }
    }

    // -------------------------------------------------------------------------
    // Stage 3: Apply transforms
    // -------------------------------------------------------------------------
    println!("\nStage 3: Applying transforms...");

    use alimentar::{DataLoader, Dataset, Shuffle};

    // Shuffle the dataset
    let shuffle = Shuffle::with_seed(42);
    let shuffled = dataset
        .data()
        .with_transform(&shuffle)
        .expect("Shuffle transform failed");

    println!("  Applied: Shuffle (seed=42)");
    println!("  Result: {} samples", shuffled.len());

    // -------------------------------------------------------------------------
    // Stage 4: Train/test split
    // -------------------------------------------------------------------------
    println!("\nStage 4: Train/test split (80/20)...");

    let total = shuffled.len();
    let train_size = (total as f64 * 0.8) as usize;

    // Get features and labels
    let _labels = dataset.labels_numeric();
    let label_names = dataset.labels();

    println!("  Training samples: {}", train_size);
    println!("  Test samples: {}", total - train_size);

    // -------------------------------------------------------------------------
    // Stage 5: Extract features for inference
    // -------------------------------------------------------------------------
    println!("\nStage 5: Extracting features for inference...");

    // Get the record batch and extract feature columns
    let batch = shuffled
        .get_batch(0)
        .expect("Dataset should have at least one batch");

    // Use arrow types re-exported from alimentar
    use arrow::array::Float64Array;

    let sepal_length = batch
        .column_by_name("sepal_length")
        .expect("sepal_length column")
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64Array");

    let sepal_width = batch
        .column_by_name("sepal_width")
        .expect("sepal_width column")
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64Array");

    let petal_length = batch
        .column_by_name("petal_length")
        .expect("petal_length column")
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64Array");

    let petal_width = batch
        .column_by_name("petal_width")
        .expect("petal_width column")
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64Array");

    // Convert to IrisFeatures
    let samples: Vec<IrisFeatures> = (0..total)
        .map(|i| IrisFeatures {
            sepal_length: sepal_length.value(i),
            sepal_width: sepal_width.value(i),
            petal_length: petal_length.value(i),
            petal_width: petal_width.value(i),
        })
        .collect();

    println!("  Extracted {} feature vectors", samples.len());

    // -------------------------------------------------------------------------
    // Stage 6: Run inference with realizar-style classifier
    // -------------------------------------------------------------------------
    println!("\nStage 6: Running inference...");

    let classifier = IrisClassifier::new();
    let mut metrics = ClassificationMetrics::new();

    // Test set evaluation (last 20%)
    let test_samples = &samples[train_size..];
    let test_labels = &label_names[train_size..];

    let start = std::time::Instant::now();
    let predictions = classifier.predict_batch(test_samples);
    let inference_time = start.elapsed();

    for (pred, actual) in predictions.iter().zip(test_labels.iter()) {
        metrics.record(pred.species.as_str(), actual);
    }

    println!("  Test samples: {}", test_samples.len());
    println!(
        "  Inference time: {:.2}ms",
        inference_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Throughput: {:.0} samples/sec",
        test_samples.len() as f64 / inference_time.as_secs_f64()
    );

    // -------------------------------------------------------------------------
    // Stage 7: Evaluation metrics
    // -------------------------------------------------------------------------
    println!("\nStage 7: Evaluation metrics...");
    println!("  Accuracy: {:.2}%", metrics.accuracy() * 100.0);
    println!("  Macro F1: {:.4}", metrics.macro_f1());
    println!("\n  Per-class metrics:");

    for class in ["setosa", "versicolor", "virginica"] {
        if let Some(m) = metrics.per_class.get(class) {
            println!(
                "    {}: P={:.2} R={:.2} F1={:.2}",
                class,
                m.precision(),
                m.recall(),
                m.f1()
            );
        }
    }

    // -------------------------------------------------------------------------
    // Stage 8: Drift detection
    // -------------------------------------------------------------------------
    println!("\nStage 8: Drift detection...");

    use alimentar::{DriftDetector, DriftTest};

    // Create detector with reference data
    let reference_data = dataset.data().clone();
    let detector = DriftDetector::new(reference_data)
        .with_test(DriftTest::KolmogorovSmirnov)
        .with_alpha(0.05);

    // Compare reference against current (same data = no drift expected)
    let drift_report = detector.detect(&shuffled).expect("Drift detection failed");

    println!("  Reference vs Current comparison:");
    let num_drifted = drift_report.num_drifted();
    println!("  Overall drift detected: {}", num_drifted > 0);
    println!("  Columns analyzed: {}", drift_report.num_columns());

    for col in drift_report.drifted_columns() {
        println!("    {}: DRIFT DETECTED", col);
    }
    if num_drifted == 0 {
        println!("    All columns: OK (no significant drift)");
    }

    // -------------------------------------------------------------------------
    // Stage 9: DataLoader demo
    // -------------------------------------------------------------------------
    println!("\nStage 9: DataLoader batching demo...");

    let loader = DataLoader::new(shuffled.clone())
        .batch_size(32)
        .shuffle(true);

    let mut batch_count = 0;
    let mut total_rows = 0;
    for batch in loader {
        batch_count += 1;
        total_rows += batch.num_rows();
    }

    println!("  Batch size: 32");
    println!("  Total batches: {}", batch_count);
    println!("  Total rows processed: {}", total_rows);

    let has_drift = drift_report.num_drifted() > 0;

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("\n{:=<60}", "");
    println!("Pipeline Summary:");
    println!("  Data source: alimentar::datasets::iris (embedded)");
    println!("  Total samples: {}", total);
    println!("  Train/Test split: {}/{}", train_size, total - train_size);
    println!("  Test accuracy: {:.2}%", metrics.accuracy() * 100.0);
    println!("  Inference count: {}", classifier.inference_count());
    println!("  Quality issues: {}", report.issues.len());
    println!("  Drift detected: {}", has_drift);
    println!("{:=<60}", "");

    println!("\n=== Data Pipeline Complete ===");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iris_features_normalize() {
        let features = IrisFeatures {
            sepal_length: 5.0,
            sepal_width: 3.0,
            petal_length: 4.0,
            petal_width: 1.0,
        };
        let norm = features.normalize();
        assert_eq!(norm.len(), 4);
        for &v in &norm {
            assert!(v >= -0.1 && v <= 1.1, "Normalized value {} out of range", v);
        }
    }

    #[test]
    fn test_classifier_setosa() {
        let classifier = IrisClassifier::new();
        let features = IrisFeatures {
            sepal_length: 5.0,
            sepal_width: 3.5,
            petal_length: 1.4, // Small petal = setosa
            petal_width: 0.2,
        };
        let pred = classifier.predict(&features);
        assert_eq!(pred.species, IrisSpecies::Setosa);
        assert!(pred.confidence > 0.9);
    }

    #[test]
    fn test_classifier_versicolor() {
        let classifier = IrisClassifier::new();
        let features = IrisFeatures {
            sepal_length: 6.0,
            sepal_width: 2.8,
            petal_length: 4.5, // Medium petal
            petal_width: 1.3,
        };
        let pred = classifier.predict(&features);
        assert_eq!(pred.species, IrisSpecies::Versicolor);
    }

    #[test]
    fn test_classifier_virginica() {
        let classifier = IrisClassifier::new();
        let features = IrisFeatures {
            sepal_length: 7.0,
            sepal_width: 3.2,
            petal_length: 6.0, // Large petal
            petal_width: 2.0,
        };
        let pred = classifier.predict(&features);
        assert_eq!(pred.species, IrisSpecies::Virginica);
    }

    #[test]
    fn test_metrics_accuracy() {
        let mut metrics = ClassificationMetrics::new();
        metrics.record("setosa", "setosa");
        metrics.record("setosa", "setosa");
        metrics.record("versicolor", "setosa"); // wrong
        metrics.record("virginica", "virginica");

        assert_eq!(metrics.total, 4);
        assert_eq!(metrics.correct, 3);
        assert!((metrics.accuracy() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_class_metrics_f1() {
        let m = ClassMetrics {
            true_positives: 8,
            false_positives: 2,
            false_negatives: 1,
        };
        // precision = 8/10 = 0.8
        // recall = 8/9 = 0.889
        // f1 = 2 * 0.8 * 0.889 / (0.8 + 0.889) = 0.842
        assert!((m.precision() - 0.8).abs() < 0.001);
        assert!((m.recall() - 0.889).abs() < 0.01);
        assert!((m.f1() - 0.842).abs() < 0.01);
    }
}
