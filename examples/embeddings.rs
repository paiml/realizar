//! Embeddings Example
//!
//! Demonstrates the embedding system for semantic text representations.
//!
//! This example shows:
//! - Embedding configuration and model types
//! - Cosine similarity for semantic search
//! - L2 distance calculations
//! - Pooling strategies (Mean, CLS, LastToken)
//!
//! # Run
//!
//! ```bash
//! # Without real model (uses mock embeddings):
//! cargo run --example embeddings
//!
//! # With real model (requires embeddings feature and model file):
//! cargo run --example embeddings --features embeddings
//! ```

fn main() {
    println!("=== Realizar Embeddings Demo ===\n");

    // -------------------------------------------------------------------------
    // 1. Similarity Functions
    // -------------------------------------------------------------------------
    println!("1. Similarity and Distance Functions\n");
    demo_similarity_functions();

    // -------------------------------------------------------------------------
    // 2. Semantic Search Simulation
    // -------------------------------------------------------------------------
    println!("\n2. Semantic Search (Mock Embeddings)\n");
    demo_semantic_search();

    // -------------------------------------------------------------------------
    // 3. Embedding Configuration
    // -------------------------------------------------------------------------
    #[cfg(feature = "embeddings")]
    {
        println!("\n3. Embedding Configuration\n");
        demo_embedding_config();
    }

    // -------------------------------------------------------------------------
    // 4. Real Model Inference (if available)
    // -------------------------------------------------------------------------
    #[cfg(feature = "embeddings")]
    {
        println!("\n4. Real Model Inference\n");
        demo_real_inference();
    }

    #[cfg(not(feature = "embeddings"))]
    {
        println!("\n3. Real Embedding Features\n");
        println!("  Note: Run with --features embeddings to see full embedding support.");
        println!("  This includes:");
        println!("    - EmbeddingEngine for loading BERT/MiniLM models");
        println!("    - Multiple pooling strategies (Mean, CLS, LastToken)");
        println!("    - Batch embedding generation");
        println!("    - Vector normalization");
    }

    println!("\n=== Demo Complete ===");
}

// =============================================================================
// Demo Functions
// =============================================================================

fn demo_similarity_functions() {
    // Cosine similarity
    println!("  Cosine Similarity:");
    println!("  ------------------");

    let vec_a = vec![1.0, 0.0, 0.0];
    let vec_b = vec![1.0, 0.0, 0.0];
    let vec_c = vec![0.0, 1.0, 0.0];
    let vec_d = vec![-1.0, 0.0, 0.0];

    println!("  vec_a = [1, 0, 0]");
    println!("  vec_b = [1, 0, 0]");
    println!("  vec_c = [0, 1, 0]");
    println!("  vec_d = [-1, 0, 0]");
    println!();

    let sim_ab = cosine_similarity(&vec_a, &vec_b);
    let sim_ac = cosine_similarity(&vec_a, &vec_c);
    let sim_ad = cosine_similarity(&vec_a, &vec_d);

    println!("  cosine(a, b) = {:.4} (identical vectors)", sim_ab);
    println!("  cosine(a, c) = {:.4} (orthogonal vectors)", sim_ac);
    println!("  cosine(a, d) = {:.4} (opposite vectors)", sim_ad);

    // L2 distance
    println!("\n  L2 Distance:");
    println!("  ------------");

    let dist_ab = l2_distance(&vec_a, &vec_b);
    let dist_ac = l2_distance(&vec_a, &vec_c);
    let dist_ad = l2_distance(&vec_a, &vec_d);

    println!("  l2(a, b) = {:.4} (identical)", dist_ab);
    println!("  l2(a, c) = {:.4} (orthogonal)", dist_ac);
    println!("  l2(a, d) = {:.4} (opposite)", dist_ad);
}

fn demo_semantic_search() {
    // Create mock embeddings for demonstration
    // In real use, these would come from an embedding model
    let documents = vec![
        ("The quick brown fox jumps over the lazy dog.", vec![0.8, 0.2, 0.1, 0.3]),
        ("A fast auburn fox leaps above a sleepy canine.", vec![0.75, 0.25, 0.15, 0.28]),
        ("Paris is the capital of France.", vec![0.1, 0.9, 0.05, 0.2]),
        ("Berlin is the capital of Germany.", vec![0.12, 0.88, 0.08, 0.22]),
        ("Machine learning is a subset of AI.", vec![0.3, 0.1, 0.9, 0.4]),
        ("Deep learning uses neural networks.", vec![0.28, 0.12, 0.85, 0.45]),
    ];

    // Query embedding
    let query = "What animal jumps quickly?";
    let query_embedding = vec![0.78, 0.22, 0.12, 0.31]; // Similar to fox sentences

    println!("  Query: \"{}\"", query);
    println!("\n  Document Similarities:");
    println!("  -----------------------");

    // Calculate similarities and rank
    let mut results: Vec<_> = documents
        .iter()
        .map(|(doc, emb)| {
            let sim = cosine_similarity(&query_embedding, emb);
            (doc, sim)
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, (doc, sim)) in results.iter().enumerate() {
        let indicator = if i == 0 { "👑" } else { "  " };
        let truncated = if doc.len() > 40 {
            format!("{}...", &doc[..40])
        } else {
            doc.to_string()
        };
        println!("  {} {:.4}: {}", indicator, sim, truncated);
    }

    println!("\n  Top result is semantically closest to the query!");
}

#[cfg(feature = "embeddings")]
fn demo_embedding_config() {
    use realizar::embeddings::{EmbeddingConfig, EmbeddingModelType, PoolingStrategy};

    println!("  Available Model Types:");
    println!("    - AllMiniLM: Lightweight, 384-dim embeddings");
    println!("    - NomicEmbed: High quality, 768-dim, 8K context");
    println!("    - BgeSmall: BGE series, balanced quality/speed");
    println!("    - Bert: Standard BERT encoder");

    println!("\n  Available Pooling Strategies:");
    println!("    - Mean: Average all token embeddings (most common)");
    println!("    - Cls: Use [CLS] token embedding (BERT-style)");
    println!("    - LastToken: Use last token (for causal models)");

    // Example configuration
    let config = EmbeddingConfig {
        model_type: EmbeddingModelType::AllMiniLM,
        hidden_size: 384,
        vocab_size: 30522,
        max_seq_length: 256,
        pooling: PoolingStrategy::Mean,
        normalize: true,
    };

    println!("\n  Example Configuration (all-MiniLM-L6-v2):");
    println!("    Model type: {:?}", config.model_type);
    println!("    Hidden size: {}", config.hidden_size);
    println!("    Vocab size: {}", config.vocab_size);
    println!("    Max sequence length: {}", config.max_seq_length);
    println!("    Pooling: {:?}", config.pooling);
    println!("    Normalize: {}", config.normalize);
}

#[cfg(feature = "embeddings")]
fn demo_real_inference() {
    use realizar::embeddings::{EmbeddingConfig, EmbeddingEngine, EmbeddingModelType, PoolingStrategy};

    // Check for model path
    let model_path = std::env::var("REALIZAR_EMBEDDING_MODEL");
    
    match model_path {
        Ok(path) => {
            println!("  Loading model from: {}", path);
            
            let config = EmbeddingConfig {
                model_type: EmbeddingModelType::AllMiniLM,
                hidden_size: 384,
                vocab_size: 30522,
                max_seq_length: 256,
                pooling: PoolingStrategy::Mean,
                normalize: true,
            };

            match EmbeddingEngine::load(&path, config) {
                Ok(engine) => {
                    println!("  Model loaded successfully!");
                    
                    let texts = vec![
                        "The weather is beautiful today.",
                        "It's a lovely sunny day outside.",
                        "Machine learning is fascinating.",
                    ];

                    println!("\n  Generating embeddings for {} texts...", texts.len());
                    
                    match engine.embed(&texts) {
                        Ok(embeddings) => {
                            println!("  Generated {} embeddings", embeddings.len());
                            
                            for (i, emb) in embeddings.iter().enumerate() {
                                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                                println!("    Text {}: {} dims, norm = {:.4}", i, emb.len(), norm);
                            }

                            // Calculate similarities
                            println!("\n  Pairwise Similarities:");
                            for i in 0..texts.len() {
                                for j in (i + 1)..texts.len() {
                                    let sim = realizar::embeddings::cosine_similarity(
                                        &embeddings[i],
                                        &embeddings[j],
                                    );
                                    println!("    [{} vs {}]: {:.4}", i, j, sim);
                                }
                            }
                        }
                        Err(e) => println!("  Error generating embeddings: {}", e),
                    }
                }
                Err(e) => println!("  Error loading model: {}", e),
            }
        }
        Err(_) => {
            println!("  No model path specified.");
            println!("  Set REALIZAR_EMBEDDING_MODEL=/path/to/model to run with real embeddings.");
            println!("\n  Recommended models:");
            println!("    - all-MiniLM-L6-v2 (22M params, fast)");
            println!("    - nomic-embed-text-v1.5 (137M params, high quality)");
            println!("    - bge-small-en-v1.5 (33M params, balanced)");
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Calculate cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Calculate L2 (Euclidean) distance between two vectors.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
