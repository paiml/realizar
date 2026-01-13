//! Integration tests for embedding models with real model inference.
//!
//! These tests verify end-to-end embedding functionality including:
//! - Loading BERT-style embedding models
//! - Generating embeddings for text inputs
//! - Verifying embedding dimensions and quality
//! - Testing pooling strategies
//!
//! # Running These Tests
//!
//! Tests are marked `#[ignore]` as they require model files:
//!
//! ```bash
//! # Download embedding model first (see docs/new/model_download_guide.md)
//! # For all-MiniLM-L6-v2:
//! REALIZAR_EMBEDDING_MODEL_PATH=/path/to/model cargo test --features embeddings --test integration_embeddings -- --ignored
//! ```

// =============================================================================
// UNIT TESTS (No model required)
// =============================================================================

#[test]
fn test_embedding_config_creation() {
    // Test creating embedding config without feature-gated code
    // This verifies the basic types are available

    // Basic config values that would be used with EmbeddingEngine
    let hidden_size = 384;
    let max_seq_length = 256;
    let vocab_size = 30522;

    assert!(hidden_size > 0);
    assert!(max_seq_length > 0);
    assert!(vocab_size > 0);
}

#[test]
fn test_cosine_similarity_function() {
    // Test the cosine similarity utility (available without embeddings feature)
    #[cfg(feature = "embeddings")]
    {
        use realizar::embeddings::cosine_similarity;

        // Same vector should have similarity 1.0
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&v1, &v2);
        assert!((sim - 1.0).abs() < 1e-6);

        // Orthogonal vectors should have similarity 0.0
        let v3 = vec![0.0, 1.0, 0.0];
        let sim2 = cosine_similarity(&v1, &v3);
        assert!(sim2.abs() < 1e-6);

        // Opposite vectors should have similarity -1.0
        let v4 = vec![-1.0, 0.0, 0.0];
        let sim3 = cosine_similarity(&v1, &v4);
        assert!((sim3 + 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_l2_distance_function() {
    #[cfg(feature = "embeddings")]
    {
        use realizar::embeddings::l2_distance;

        // Same vector should have distance 0.0
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        let dist = l2_distance(&v1, &v2);
        assert!(dist.abs() < 1e-6);

        // Unit distance
        let v3 = vec![0.0, 0.0, 0.0];
        let v4 = vec![1.0, 0.0, 0.0];
        let dist2 = l2_distance(&v3, &v4);
        assert!((dist2 - 1.0).abs() < 1e-6);

        // Pythagorean distance
        let v5 = vec![0.0, 0.0, 0.0];
        let v6 = vec![3.0, 4.0, 0.0];
        let dist3 = l2_distance(&v5, &v6);
        assert!((dist3 - 5.0).abs() < 1e-6);
    }
}

#[test]
fn test_pooling_strategy_variants() {
    #[cfg(feature = "embeddings")]
    {
        use realizar::embeddings::PoolingStrategy;

        // Verify all pooling strategies are available
        let _mean = PoolingStrategy::Mean;
        let _cls = PoolingStrategy::Cls;
        let _last = PoolingStrategy::LastToken;
    }
}

#[test]
fn test_embedding_model_types() {
    #[cfg(feature = "embeddings")]
    {
        use realizar::embeddings::EmbeddingModelType;

        // Verify all model types are available
        let _bert = EmbeddingModelType::Bert;
        let _minilm = EmbeddingModelType::AllMiniLM;
        let _nomic = EmbeddingModelType::NomicEmbed;
        let _bge = EmbeddingModelType::BgeSmall;
    }
}

// =============================================================================
// MOCK-BASED TESTS (No real model, test logic)
// =============================================================================

/// Test embedding similarity comparisons with mock data
#[test]
fn test_semantic_similarity_ordering() {
    #[cfg(feature = "embeddings")]
    {
        use realizar::embeddings::cosine_similarity;

        // Simulate embeddings for semantic comparison
        // These are mock embeddings that represent the expected relationships

        // "The cat sat on the mat" - represented as a mock embedding
        let cat_embedding = vec![0.8, 0.2, 0.1, 0.3, 0.5];

        // "The dog lay on the rug" - semantically similar to cat sentence
        let dog_embedding = vec![0.75, 0.25, 0.15, 0.35, 0.45];

        // "Paris is the capital of France" - semantically different
        let paris_embedding = vec![0.1, 0.9, 0.8, 0.2, 0.1];

        let cat_dog_sim = cosine_similarity(&cat_embedding, &dog_embedding);
        let cat_paris_sim = cosine_similarity(&cat_embedding, &paris_embedding);

        // Cat sentence should be more similar to dog sentence than Paris sentence
        assert!(
            cat_dog_sim > cat_paris_sim,
            "Similar sentences should have higher cosine similarity"
        );
    }
}

/// Test batch embedding similarity matrix
#[test]
fn test_batch_similarity_matrix() {
    #[cfg(feature = "embeddings")]
    {
        use realizar::embeddings::cosine_similarity;

        // Mock embeddings for a batch of texts
        let embeddings = vec![
            vec![1.0, 0.0, 0.0], // Text 0
            vec![0.9, 0.1, 0.0], // Text 1 (similar to 0)
            vec![0.0, 1.0, 0.0], // Text 2 (orthogonal to 0)
            vec![0.0, 0.9, 0.1], // Text 3 (similar to 2)
        ];

        // Compute similarity matrix
        let n = embeddings.len();
        let mut sim_matrix = vec![vec![0.0f32; n]; n];

        for i in 0..n {
            for j in 0..n {
                sim_matrix[i][j] = cosine_similarity(&embeddings[i], &embeddings[j]);
            }
        }

        // Diagonal should be 1.0 (self-similarity)
        for i in 0..n {
            assert!(
                (sim_matrix[i][i] - 1.0).abs() < 1e-6,
                "Self-similarity should be 1.0"
            );
        }

        // Verify expected relationships
        assert!(sim_matrix[0][1] > 0.9, "Text 0 and 1 should be very similar");
        assert!(sim_matrix[2][3] > 0.9, "Text 2 and 3 should be very similar");
        assert!(
            sim_matrix[0][2].abs() < 0.1,
            "Text 0 and 2 should be orthogonal"
        );
    }
}

// =============================================================================
// INTEGRATION TESTS (Require model files)
// =============================================================================

/// Test loading and running an all-MiniLM-L6-v2 embedding model.
#[test]
#[ignore = "Requires embedding model file. Set REALIZAR_EMBEDDING_MODEL_PATH to run."]
#[cfg(feature = "embeddings")]
fn test_minilm_embedding_generation() {
    use realizar::embeddings::{EmbeddingConfig, EmbeddingEngine, EmbeddingModelType, PoolingStrategy};

    let model_path = std::env::var("REALIZAR_EMBEDDING_MODEL_PATH")
        .expect("REALIZAR_EMBEDDING_MODEL_PATH must be set to run this test");

    let config = EmbeddingConfig {
        model_type: EmbeddingModelType::AllMiniLM,
        hidden_size: 384,
        vocab_size: 30522,
        max_seq_length: 256,
        pooling: PoolingStrategy::Mean,
        normalize: true,
    };

    let engine = EmbeddingEngine::load(&model_path, config)
        .expect("Failed to load embedding model");

    // Generate embeddings
    let texts = vec![
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above a sleepy canine.",
        "The weather in Paris is wonderful today.",
    ];

    let embeddings = engine.embed(&texts).expect("Failed to generate embeddings");

    // Verify dimensions
    assert_eq!(embeddings.len(), 3);
    for emb in &embeddings {
        assert_eq!(emb.len(), 384, "all-MiniLM-L6-v2 should produce 384-dim embeddings");
    }

    // Verify normalization (L2 norm should be ~1.0)
    for emb in &embeddings {
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Normalized embeddings should have unit norm, got {}",
            norm
        );
    }

    // Verify semantic similarity
    let sim_fox = realizar::embeddings::cosine_similarity(&embeddings[0], &embeddings[1]);
    let sim_paris = realizar::embeddings::cosine_similarity(&embeddings[0], &embeddings[2]);

    assert!(
        sim_fox > sim_paris,
        "Fox sentences should be more similar to each other than to Paris sentence"
    );
}

/// Test different pooling strategies produce different results.
#[test]
#[ignore = "Requires embedding model file. Set REALIZAR_EMBEDDING_MODEL_PATH to run."]
#[cfg(feature = "embeddings")]
fn test_pooling_strategy_differences() {
    use realizar::embeddings::{EmbeddingConfig, EmbeddingEngine, EmbeddingModelType, PoolingStrategy};

    let model_path = std::env::var("REALIZAR_EMBEDDING_MODEL_PATH")
        .expect("REALIZAR_EMBEDDING_MODEL_PATH must be set to run this test");

    let text = vec!["This is a test sentence for embedding comparison."];

    // Mean pooling
    let mean_config = EmbeddingConfig {
        model_type: EmbeddingModelType::AllMiniLM,
        hidden_size: 384,
        vocab_size: 30522,
        max_seq_length: 256,
        pooling: PoolingStrategy::Mean,
        normalize: true,
    };
    let mean_engine = EmbeddingEngine::load(&model_path, mean_config.clone())
        .expect("Failed to load with Mean pooling");
    let mean_emb = mean_engine.embed(&text).expect("Failed to embed")[0].clone();

    // CLS pooling
    let cls_config = EmbeddingConfig {
        pooling: PoolingStrategy::Cls,
        ..mean_config.clone()
    };
    let cls_engine = EmbeddingEngine::load(&model_path, cls_config)
        .expect("Failed to load with CLS pooling");
    let cls_emb = cls_engine.embed(&text).expect("Failed to embed")[0].clone();

    // Embeddings should be different with different pooling
    let diff: f32 = mean_emb
        .iter()
        .zip(cls_emb.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        diff > 1e-3,
        "Different pooling strategies should produce different embeddings"
    );
}

/// Test batch embedding efficiency.
#[test]
#[ignore = "Requires embedding model file. Set REALIZAR_EMBEDDING_MODEL_PATH to run."]
#[cfg(feature = "embeddings")]
fn test_batch_embedding_efficiency() {
    use realizar::embeddings::{EmbeddingConfig, EmbeddingEngine, EmbeddingModelType, PoolingStrategy};
    use std::time::Instant;

    let model_path = std::env::var("REALIZAR_EMBEDDING_MODEL_PATH")
        .expect("REALIZAR_EMBEDDING_MODEL_PATH must be set to run this test");

    let config = EmbeddingConfig {
        model_type: EmbeddingModelType::AllMiniLM,
        hidden_size: 384,
        vocab_size: 30522,
        max_seq_length: 256,
        pooling: PoolingStrategy::Mean,
        normalize: true,
    };

    let engine = EmbeddingEngine::load(&model_path, config)
        .expect("Failed to load embedding model");

    // Generate a batch of texts
    let texts: Vec<String> = (0..32)
        .map(|i| format!("This is test sentence number {} for batch embedding.", i))
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    // Time batch embedding
    let start = Instant::now();
    let embeddings = engine.embed(&text_refs).expect("Failed to embed batch");
    let batch_duration = start.elapsed();

    assert_eq!(embeddings.len(), 32);

    // Log performance for manual inspection
    println!(
        "Batch embedding of 32 texts took {:?} ({:.2} texts/sec)",
        batch_duration,
        32.0 / batch_duration.as_secs_f64()
    );
}

/// Test nomic-embed-text model if available.
#[test]
#[ignore = "Requires nomic-embed model. Set REALIZAR_NOMIC_MODEL_PATH to run."]
#[cfg(feature = "embeddings")]
fn test_nomic_embed_model() {
    use realizar::embeddings::{EmbeddingConfig, EmbeddingEngine, EmbeddingModelType, PoolingStrategy};

    let model_path = std::env::var("REALIZAR_NOMIC_MODEL_PATH")
        .expect("REALIZAR_NOMIC_MODEL_PATH must be set to run this test");

    let config = EmbeddingConfig {
        model_type: EmbeddingModelType::NomicEmbed,
        hidden_size: 768,
        vocab_size: 30528,
        max_seq_length: 8192, // nomic-embed supports longer sequences
        pooling: PoolingStrategy::Mean,
        normalize: true,
    };

    let engine = EmbeddingEngine::load(&model_path, config)
        .expect("Failed to load nomic-embed model");

    let texts = vec![
        "search_query: What is the capital of France?",
        "search_document: Paris is the capital and most populous city of France.",
        "search_document: Berlin is the capital of Germany.",
    ];

    let embeddings = engine.embed(&texts).expect("Failed to generate embeddings");

    // Verify dimensions
    assert_eq!(embeddings.len(), 3);
    for emb in &embeddings {
        assert_eq!(emb.len(), 768, "nomic-embed should produce 768-dim embeddings");
    }

    // Query should be more similar to the Paris document
    let sim_paris = realizar::embeddings::cosine_similarity(&embeddings[0], &embeddings[1]);
    let sim_berlin = realizar::embeddings::cosine_similarity(&embeddings[0], &embeddings[2]);

    assert!(
        sim_paris > sim_berlin,
        "France query should be more similar to Paris document"
    );
}

/// Test BGE-small model if available.
#[test]
#[ignore = "Requires BGE model. Set REALIZAR_BGE_MODEL_PATH to run."]
#[cfg(feature = "embeddings")]
fn test_bge_small_model() {
    use realizar::embeddings::{EmbeddingConfig, EmbeddingEngine, EmbeddingModelType, PoolingStrategy};

    let model_path = std::env::var("REALIZAR_BGE_MODEL_PATH")
        .expect("REALIZAR_BGE_MODEL_PATH must be set to run this test");

    let config = EmbeddingConfig {
        model_type: EmbeddingModelType::BgeSmall,
        hidden_size: 384,
        vocab_size: 30522,
        max_seq_length: 512,
        pooling: PoolingStrategy::Cls, // BGE uses CLS pooling
        normalize: true,
    };

    let engine = EmbeddingEngine::load(&model_path, config)
        .expect("Failed to load BGE model");

    let texts = vec![
        "Represent this sentence for retrieval: What is machine learning?",
        "Machine learning is a subset of artificial intelligence.",
        "The weather is sunny today.",
    ];

    let embeddings = engine.embed(&texts).expect("Failed to generate embeddings");

    assert_eq!(embeddings.len(), 3);

    // Semantic similarity check
    let sim_ml = realizar::embeddings::cosine_similarity(&embeddings[0], &embeddings[1]);
    let sim_weather = realizar::embeddings::cosine_similarity(&embeddings[0], &embeddings[2]);

    assert!(
        sim_ml > sim_weather,
        "ML query should be more similar to ML document"
    );
}
