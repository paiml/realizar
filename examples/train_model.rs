//! Train Real .apr Models with Aprender
//!
//! This example trains actual ML models using aprender and saves them
//! in the .apr format for use with realizar inference.
//!
//! ## Models Trained
//!
//! 1. **Wine Quality Regressor** - Linear regression for quality prediction
//!
//! ## Run
//!
//! ```bash
//! cargo run --example train_model --features "aprender-serve"
//! ```
//!
//! ## Output Files (gitignored)
//!
//! - `wine_regressor.apr` - Trained wine quality model

#![cfg(feature = "aprender-serve")]
#![allow(clippy::expect_used)]

use std::path::Path;

fn main() {
    println!("=== Train Real .apr Models ===\n");

    // Train wine regressor
    train_wine_regressor();

    // Verify the model loads
    verify_model();

    println!("\n=== Training Complete ===");
}

/// Train wine quality regressor using aprender
fn train_wine_regressor() {
    println!("Training Wine Quality Regressor...");

    use aprender::{
        format::{save, ModelType, SaveOptions},
        linear_model::LinearRegression,
        primitives::{Matrix, Vector},
        traits::Estimator,
    };

    // Generate synthetic wine data (11 features -> quality score)
    // Features: fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
    //           chlorides, free_so2, total_so2, density, pH, sulphates, alcohol
    let n_samples = 500;
    let n_features = 11;

    // Simple PRNG for reproducibility
    let mut rng_state: u64 = 42;
    let mut rand = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32) / (u32::MAX as f32)
    };

    let mut features = Vec::with_capacity(n_samples * n_features);
    let mut targets = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        // Generate features in typical wine ranges
        let fixed_acidity = 4.0 + rand() * 12.0;
        let volatile_acidity = 0.1 + rand() * 1.5;
        let citric_acid = rand() * 1.0;
        let residual_sugar = 0.9 + rand() * 14.0;
        let chlorides = 0.01 + rand() * 0.5;
        let free_so2 = 1.0 + rand() * 70.0;
        let total_so2 = 6.0 + rand() * 280.0;
        let density = 0.99 + rand() * 0.05;
        let ph = 2.7 + rand() * 1.3;
        let sulphates = 0.3 + rand() * 1.7;
        let alcohol = 8.0 + rand() * 7.0;

        features.extend_from_slice(&[
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_so2,
            total_so2,
            density,
            ph,
            sulphates,
            alcohol,
        ]);

        // Quality influenced by: alcohol (+), volatile_acidity (-), sulphates (+)
        let quality =
            5.0 + alcohol * 0.3 - volatile_acidity * 2.0 + sulphates * 0.5 + (rand() - 0.5) * 1.0; // noise
        targets.push(quality.clamp(3.0, 9.0));
    }

    let x = Matrix::from_vec(n_samples, n_features, features).expect("Matrix failed");
    let y = Vector::from_slice(&targets);

    println!("  Generated {} synthetic samples", n_samples);

    // Train linear regression
    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("Training failed");

    // Evaluate
    let r2 = model.score(&x, &y);
    println!("  R² score: {:.4}", r2);

    // Get coefficients for display
    let coeffs = model.coefficients();
    println!("  Coefficients (top 3 by magnitude):");
    let feature_names = [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_so2",
        "total_so2",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ];

    let mut coef_pairs: Vec<_> = feature_names.iter().zip(coeffs.as_slice().iter()).collect();
    coef_pairs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    for (name, coef) in coef_pairs.iter().take(3) {
        println!("    {}: {:.4}", name, coef);
    }

    // Save model
    let path = "wine_regressor.apr";
    save(
        &model,
        ModelType::LinearRegression,
        path,
        SaveOptions::default(),
    )
    .expect("Save failed");

    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    println!("  Saved: {} ({} bytes)", path, size);
}

/// Load and verify a saved model
fn verify_model() {
    println!("\nVerifying saved model...");

    use aprender::{
        format::{load, ModelType},
        linear_model::LinearRegression,
        primitives::Matrix,
        traits::Estimator,
    };

    if Path::new("wine_regressor.apr").exists() {
        let model: LinearRegression =
            load("wine_regressor.apr", ModelType::LinearRegression).expect("Load failed");
        println!("  Loaded wine_regressor.apr successfully");

        // Quick inference test - predict quality for a sample wine
        let test_wine = Matrix::from_vec(
            1,
            11,
            vec![
                7.0,   // fixed_acidity
                0.3,   // volatile_acidity (low = good)
                0.4,   // citric_acid
                2.0,   // residual_sugar
                0.08,  // chlorides
                15.0,  // free_so2
                40.0,  // total_so2
                0.995, // density
                3.3,   // pH
                0.6,   // sulphates
                12.0,  // alcohol (high = good)
            ],
        )
        .expect("Matrix failed");

        let prediction = model.predict(&test_wine);
        println!(
            "  Test wine prediction: {:.2}/10 quality",
            prediction.as_slice()[0]
        );

        // Test a low quality wine
        let bad_wine = Matrix::from_vec(
            1,
            11,
            vec![
                8.5,  // fixed_acidity
                1.2,  // volatile_acidity (high = bad, vinegar)
                0.1,  // citric_acid
                4.0,  // residual_sugar
                0.15, // chlorides
                5.0,  // free_so2
                20.0, // total_so2
                0.998, 3.5, 0.3, // sulphates (low)
                9.0, // alcohol (low)
            ],
        )
        .expect("Matrix failed");

        let bad_pred = model.predict(&bad_wine);
        println!(
            "  Low quality wine prediction: {:.2}/10",
            bad_pred.as_slice()[0]
        );
    } else {
        println!("  Model file not found (run training first)");
    }
}
