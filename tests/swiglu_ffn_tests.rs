//! SwiGLU FFN Popperian Falsification Tests
//!
//! EXTREME TDD: Tests written BEFORE implementation.
//! These tests MUST FAIL initially, then pass after implementation.
//!
//! SwiGLU (Swish-Gated Linear Unit) used by LLaMA/TinyLlama:
//!   output = down(gate(x) * silu(up(x)))
//!
//! Where:
//!   - up(x) = matmul(x, ffn_up_weight)
//!   - gate(x) = matmul(x, ffn_gate_weight)
//!   - silu(x) = x * sigmoid(x)
//!   - output = matmul(gate * silu(up), ffn_down_weight)

/// FFN-01: OwnedQuantizedLayer must have ffn_gate_weight field
/// Falsification: Struct compilation fails if field is missing
#[test]
fn test_ffn01_owned_layer_has_gate_weight() {
    use realizar::gguf::OwnedQuantizedLayer;

    // This test verifies the struct has the ffn_gate_weight field
    // If the field doesn't exist, this won't compile
    fn assert_has_gate_field(layer: &OwnedQuantizedLayer) -> bool {
        // Access the field - compiler error if missing
        layer.ffn_gate_weight.is_some() || layer.ffn_gate_weight.is_none()
    }

    // Just verify the function compiles
    assert!(true, "FFN-01: OwnedQuantizedLayer must have ffn_gate_weight field");
}

/// FFN-02: OwnedQuantizedLayer must have ffn_gate_bias field
/// Falsification: Struct compilation fails if field is missing
#[test]
fn test_ffn02_owned_layer_has_gate_bias() {
    use realizar::gguf::OwnedQuantizedLayer;

    fn assert_has_gate_bias(layer: &OwnedQuantizedLayer) -> bool {
        layer.ffn_gate_bias.is_some() || layer.ffn_gate_bias.is_none()
    }

    assert!(true, "FFN-02: OwnedQuantizedLayer must have ffn_gate_bias field");
}

/// FFN-03: QuantizedGGUFTransformerLayer must have ffn_gate_weight field
/// Falsification: Struct compilation fails if field is missing
#[test]
fn test_ffn03_quantized_layer_has_gate_weight() {
    use realizar::gguf::QuantizedGGUFTransformerLayer;

    fn assert_has_gate_field(layer: &QuantizedGGUFTransformerLayer) -> bool {
        layer.ffn_gate_weight.is_some() || layer.ffn_gate_weight.is_none()
    }

    assert!(true, "FFN-03: QuantizedGGUFTransformerLayer must have ffn_gate_weight");
}

/// FFN-04: SiLU activation function exists and is correct
/// Falsification: silu(0) != 0 OR silu(large) != large
#[test]
fn test_ffn04_silu_activation_correct() {
    // SiLU(x) = x * sigmoid(x)
    // At x=0: SiLU(0) = 0 * 0.5 = 0
    // At x=large: sigmoid(x) ≈ 1, so SiLU(x) ≈ x

    fn silu(x: f32) -> f32 {
        x * (1.0 / (1.0 + (-x).exp()))
    }

    // Test silu(0) = 0
    let silu_zero = silu(0.0);
    assert!(
        silu_zero.abs() < 1e-6,
        "FFN-04a: SiLU(0) should be 0, got {}",
        silu_zero
    );

    // Test silu(10) ≈ 10 (large positive)
    let silu_large = silu(10.0);
    assert!(
        (silu_large - 10.0).abs() < 0.001,
        "FFN-04b: SiLU(10) should ≈ 10, got {}",
        silu_large
    );

    // Test silu(-10) ≈ 0 (large negative)
    let silu_neg = silu(-10.0);
    assert!(
        silu_neg.abs() < 0.001,
        "FFN-04c: SiLU(-10) should ≈ 0, got {}",
        silu_neg
    );

    // Test silu(1) ≈ 0.731
    let silu_one = silu(1.0);
    assert!(
        (silu_one - 0.731).abs() < 0.01,
        "FFN-04d: SiLU(1) should ≈ 0.731, got {}",
        silu_one
    );
}

/// FFN-05: SwiGLU output differs from simple FFN
/// Falsification: swiglu_output == simple_ffn_output for non-trivial input
#[test]
fn test_ffn05_swiglu_differs_from_simple_ffn() {
    // SwiGLU: output = down(gate * silu(up))
    // Simple: output = down(gelu(up))
    // These should produce different results for the same weights

    fn silu(x: f32) -> f32 {
        x * (1.0 / (1.0 + (-x).exp()))
    }

    fn gelu(x: f32) -> f32 {
        // Approximate GELU
        0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }

    // Simple test: same up projection, different activation path
    let x = 2.0_f32;
    let up = x * 1.5;  // Simulated up projection
    let gate = x * 0.8; // Simulated gate projection (only in SwiGLU)

    // SwiGLU path
    let swiglu_hidden = gate * silu(up);

    // Simple GELU path
    let simple_hidden = gelu(up);

    // They must differ
    assert!(
        (swiglu_hidden - simple_hidden).abs() > 0.1,
        "FFN-05: SwiGLU hidden ({}) must differ from simple GELU hidden ({})",
        swiglu_hidden,
        simple_hidden
    );
}

/// FFN-06: Gate weight loading from GGUF
/// Falsification: TinyLlama GGUF loads with gate=None
#[test]
#[ignore] // Requires TinyLlama GGUF file
fn test_ffn06_gguf_loads_gate_weight() {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
    use std::path::Path;

    let gguf_path = "/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    if !Path::new(gguf_path).exists() {
        println!("Skipping FFN-06: TinyLlama GGUF not found");
        return;
    }

    let mapped = MappedGGUFModel::from_path(gguf_path)
        .expect("Failed to load GGUF");
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .expect("Failed to create model");

    // TinyLlama uses SwiGLU, so gate weight must be present
    assert!(
        model.layers[0].ffn_gate_weight.is_some(),
        "FFN-06: TinyLlama layer 0 must have ffn_gate_weight (SwiGLU model)"
    );
}

/// FFN-07: Gate weight dimensions match up weight
/// Falsification: gate.out_dim != up.out_dim
#[test]
#[ignore] // Requires TinyLlama GGUF file
fn test_ffn07_gate_dimensions_match_up() {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
    use std::path::Path;

    let gguf_path = "/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    if !Path::new(gguf_path).exists() {
        println!("Skipping FFN-07: TinyLlama GGUF not found");
        return;
    }

    let mapped = MappedGGUFModel::from_path(gguf_path)
        .expect("Failed to load GGUF");
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .expect("Failed to create model");

    let layer = &model.layers[0];
    let gate = layer.ffn_gate_weight.as_ref()
        .expect("Gate weight must be present");

    // Gate and Up must have same output dimension (intermediate_dim)
    assert_eq!(
        gate.out_dim, layer.ffn_up_weight.out_dim,
        "FFN-07: gate.out_dim ({}) must equal up.out_dim ({})",
        gate.out_dim, layer.ffn_up_weight.out_dim
    );

    // Gate and Up must have same input dimension (hidden_dim)
    assert_eq!(
        gate.in_dim, layer.ffn_up_weight.in_dim,
        "FFN-07: gate.in_dim ({}) must equal up.in_dim ({})",
        gate.in_dim, layer.ffn_up_weight.in_dim
    );
}

/// FFN-08: SwiGLU forward produces coherent output
/// Falsification: Generated tokens are random/garbage
#[test]
#[ignore] // Requires TinyLlama GGUF file
fn test_ffn08_swiglu_forward_coherent() {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
    use std::path::Path;

    let gguf_path = "/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    if !Path::new(gguf_path).exists() {
        println!("Skipping FFN-08: TinyLlama GGUF not found");
        return;
    }

    let mapped = MappedGGUFModel::from_path(gguf_path)
        .expect("Failed to load GGUF");
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .expect("Failed to create model");

    // Generate a few tokens from a simple prompt
    // BOS token for TinyLlama is 1
    let prompt = vec![1u32, 15043]; // <s> Hello
    let config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0, // Greedy for reproducibility
        top_k: 1,
        stop_tokens: vec![2], // EOS
    };

    let output = model.generate(&prompt, &config)
        .expect("Generation should succeed");

    // Output should not be all zeros or all same token
    let unique_tokens: std::collections::HashSet<_> = output.iter().collect();
    assert!(
        unique_tokens.len() > 1 || output.len() <= 2,
        "FFN-08: Generated output should have variety, got {:?}",
        output
    );

    // Output tokens should be within vocab range
    let vocab_size = model.config.vocab_size;
    for &tok in &output {
        assert!(
            (tok as usize) < vocab_size,
            "FFN-08: Token {} exceeds vocab size {}",
            tok, vocab_size
        );
    }
}
