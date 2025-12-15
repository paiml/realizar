//! Build MNIST Model - Generate reproducible `.apr` model file
//!
//! This script trains a LogisticRegression model on test MNIST data
//! and saves it to the `.apr` format for Lambda deployment.
//!
//! ## The `.apr` Format
//!
//! Aprender's proprietary binary format with:
//! - Magic bytes: "APRN" (4 bytes)
//! - CRC32 checksum (integrity verification)
//! - Ed25519 signatures (optional, provenance)
//! - AES-256-GCM encryption (optional, confidentiality)
//! - Zstd compression (optional, efficiency)
//!
//! ## Usage
//!
//! ```bash
//! # Generate model for Lambda embedding
//! cargo run --example build_mnist_model --release --features aprender-serve
//!
//! # Verify model loads correctly
//! ls -la models/mnist_784x2.apr
//! ```

use std::{fs, path::Path};

use aprender::{
    classification::LogisticRegression,
    format::{save, ModelType, SaveOptions},
    primitives::Matrix,
};

// MNIST configuration - 784 features (28x28), binary classification
const INPUT_DIM: usize = 784;
const TRAINING_SAMPLES: usize = 1000;
const SEED: u64 = 42;

fn main() {
    println!("=== Reproducible MNIST Model Builder (.apr format) ===\n");

    // Create models directory
    let models_dir = Path::new("models");
    if !models_dir.exists() {
        fs::create_dir_all(models_dir).expect("Failed to create models directory");
    }

    // Generate deterministic training data
    println!("Step 1: Generating deterministic training data (seed={SEED})...");
    let (x_data, y_data) = generate_mnist_data(SEED);
    let x = Matrix::from_vec(TRAINING_SAMPLES, INPUT_DIM, x_data)
        .expect("Matrix dimensions must match data length");

    println!("  Samples: {TRAINING_SAMPLES}");
    println!("  Features: {INPUT_DIM}");
    println!("  Classes: 2 (digit 0 vs others)");

    // Train model
    println!("\nStep 2: Training LogisticRegression model...");
    let mut model = LogisticRegression::new()
        .with_learning_rate(0.01)
        .with_max_iter(100);

    model.fit(&x, &y_data).expect("Training should succeed");

    // Validate model
    let accuracy = model.score(&x, &y_data);
    println!("  Training accuracy: {:.1}%", accuracy * 100.0);

    if accuracy < 0.85 {
        eprintln!("WARNING: Model accuracy below 85% - consider adjusting hyperparameters");
    }

    // Save to .apr format
    let model_path = models_dir.join("mnist_784x2.apr");
    println!("\nStep 3: Saving model to {}...", model_path.display());

    save(
        &model,
        ModelType::LogisticRegression,
        &model_path,
        SaveOptions::default(),
    )
    .expect("Failed to save model");

    // Verify file
    let metadata = fs::metadata(&model_path).expect("Failed to read model file");
    println!("  File size: {} bytes", metadata.len());
    println!("  CRC32 checksum: embedded ✓");

    // Show file header (first 16 bytes)
    let header = fs::read(&model_path).expect("Failed to read model file");
    print!("  Magic bytes: ");
    for byte in header.iter().take(4) {
        if byte.is_ascii_alphanumeric() {
            print!("{}", *byte as char);
        } else {
            print!("\\x{byte:02x}");
        }
    }
    println!();

    // Verify model loads correctly
    println!("\nStep 4: Verifying model loads correctly...");
    let loaded: LogisticRegression =
        aprender::format::load(&model_path, ModelType::LogisticRegression)
            .expect("Failed to load model - CRC32 check may have failed");

    // Test inference
    let test_input = Matrix::from_vec(1, INPUT_DIM, vec![0.5; INPUT_DIM])
        .expect("Test input dimensions must match");
    let prediction = loaded.predict(&test_input);
    println!("  Test prediction: class {}", prediction[0]);
    println!("  Model verification: PASSED ✓");

    // Summary
    println!("\n=== Build Complete ===");
    println!("Model: {}", model_path.display());
    println!("Format: .apr (Aprender proprietary)");
    println!("Size: {} bytes", metadata.len());
    println!("Accuracy: {:.1}%", accuracy * 100.0);
    println!("\nNext steps:");
    println!(
        "  1. Copy to src/bin/: cp {} src/bin/",
        model_path.display()
    );
    println!(
        "  2. Build Lambda: cargo build --release --bin mnist_lambda --features aprender-serve"
    );
    println!("  3. Deploy to AWS Lambda");
}

/// Generate test MNIST-like data with deterministic seed
///
/// Uses a simple PRNG formula for reproducibility across builds:
/// - pixel = ((sample * 17 + feature * 31) % 256) / 255.0
/// - label = 0 if sample % 10 == 0 else 1
fn generate_mnist_data(seed: u64) -> (Vec<f32>, Vec<usize>) {
    let mut x_data = Vec::with_capacity(TRAINING_SAMPLES * INPUT_DIM);
    let mut y_data = Vec::with_capacity(TRAINING_SAMPLES);

    // Seed is implicit in the deterministic formula
    let _ = seed;

    for i in 0..TRAINING_SAMPLES {
        for j in 0..INPUT_DIM {
            // Deterministic pixel value
            let pixel = ((i * 17 + j * 31) % 256) as f32 / 255.0;
            x_data.push(pixel);
        }
        // Binary classification: digit 0 (10% of samples) vs others (90%)
        y_data.push(if i % 10 == 0 { 0 } else { 1 });
    }

    (x_data, y_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_generation_is_deterministic() {
        let (x1, y1) = generate_mnist_data(42);
        let (x2, y2) = generate_mnist_data(42);

        assert_eq!(x1, x2, "X data should be deterministic");
        assert_eq!(y1, y2, "Y data should be deterministic");
    }

    #[test]
    fn test_data_dimensions() {
        let (x, y) = generate_mnist_data(42);

        assert_eq!(x.len(), TRAINING_SAMPLES * INPUT_DIM);
        assert_eq!(y.len(), TRAINING_SAMPLES);
    }

    #[test]
    fn test_label_distribution() {
        let (_, y) = generate_mnist_data(42);

        let zeros = y.iter().filter(|&&l| l == 0).count();
        let ones = y.iter().filter(|&&l| l == 1).count();

        // 10% should be zeros, 90% ones
        assert_eq!(zeros, TRAINING_SAMPLES / 10);
        assert_eq!(ones, TRAINING_SAMPLES - zeros);
    }
}
