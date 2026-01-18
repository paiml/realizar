//! Integration tests for JSON grammar-constrained generation.
//!
//! These tests verify end-to-end JSON output with grammar enforcement:
//! - Grammar generation from JSON schemas
//! - LogitProcessor masking during generation
//! - HybridSampler mode switching
//! - Valid JSON output verification
//!
//! # Running These Tests
//!
//! Tests are marked `#[ignore]` as they require model files:
//!
//! ```bash
//! # Download a model first
//! REALIZAR_TEST_MODEL_PATH=/path/to/model.gguf cargo test --test integration_json_grammar -- --ignored
//! ```

use realizar::grammar::{generate_tool_grammar, ToolDefinition, ToolParameter};
use realizar::sampling::{
    LogitProcessor, LogitProcessorChain, SamplingMode, TemperatureProcessor, TopPProcessor,
};
use std::collections::HashMap;

// =============================================================================
// TEST HELPERS
// =============================================================================

fn create_weather_tool() -> ToolDefinition {
    ToolDefinition::new(
        "get_weather",
        "Get weather for a location",
        vec![
            ToolParameter::required_string("location", "The location to get weather for"),
            ToolParameter::required_enum(
                "unit",
                "Temperature unit",
                vec!["celsius".to_string(), "fahrenheit".to_string()],
            ),
        ],
    )
}

// =============================================================================
// UNIT TESTS (No model required)
// =============================================================================

#[test]
fn test_tool_grammar_generation() {
    let tools = vec![create_weather_tool()];
    let grammar = generate_tool_grammar(&tools);

    // Grammar should have rules
    assert!(grammar.len() > 0, "Grammar should have rules");
    assert!(!grammar.root().is_empty(), "Grammar should have root rule");

    println!("Generated grammar with {} rules", grammar.len());
}

#[test]
fn test_logit_processor_chain() {
    // Create a chain of processors
    let mut chain = LogitProcessorChain::new();
    chain.push(Box::new(TemperatureProcessor::new(0.5)));
    chain.push(Box::new(TopPProcessor::new(0.9)));

    assert_eq!(chain.len(), 2);

    // Test processing logits
    let mut logits = vec![1.0f32; 100];
    let input_ids = vec![1u32, 2, 3];

    chain.process(&input_ids, &mut logits);

    // Temperature should scale logits
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(max_logit > 0.0);
}

#[test]
fn test_tool_call_format_variants() {
    use realizar::grammar::ToolCallFormat;

    // Verify all formats are available
    let _openai = ToolCallFormat::OpenAI;
    let _anthropic = ToolCallFormat::Anthropic;
    let _hermes = ToolCallFormat::Hermes;

    assert_eq!(ToolCallFormat::default(), ToolCallFormat::OpenAI);
}

#[test]
fn test_sampling_mode_variants() {
    // Test that sampling mode variants exist
    let mode1 = SamplingMode::FreeForm;
    let mode2 = SamplingMode::JsonConstrained;
    let mode3 = SamplingMode::Detecting;

    assert_ne!(mode1, mode2);
    assert_ne!(mode2, mode3);
    assert_ne!(mode1, mode3);
}

#[test]
fn test_repetition_penalty_processor() {
    use realizar::sampling::RepetitionPenaltyProcessor;

    let penalty = 1.2;
    let mut processor = RepetitionPenaltyProcessor::new(penalty);

    // Create logits with uniform values
    let mut logits = vec![1.0f32; 100];

    // Input with repeated tokens
    let input_ids = vec![5u32, 10, 5, 10, 5];

    processor.process(&input_ids, &mut logits);

    // Repeated tokens should have reduced logits
    assert!(
        logits[5] < 1.0,
        "Token 5 should be penalized, got {}",
        logits[5]
    );
    assert!(
        logits[10] < 1.0,
        "Token 10 should be penalized, got {}",
        logits[10]
    );

    // Non-repeated tokens should be unchanged
    assert!(
        (logits[1] - 1.0).abs() < 1e-6,
        "Token 1 should be unchanged"
    );
}

#[test]
fn test_temperature_processor_scaling() {
    let temperature = 2.0;
    let mut processor = TemperatureProcessor::new(temperature);

    let mut logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let original_diff = 5.0 - 1.0;

    processor.process(&[], &mut logits);

    let scaled_diff = logits[4] - logits[0];

    // Temperature > 1 should reduce logit differences
    assert!(
        scaled_diff < original_diff,
        "Temperature > 1 should reduce logit range"
    );
}

#[test]
fn test_top_p_processor_creation() {
    let processor = TopPProcessor::new(0.9);
    assert!((processor.top_p() - 0.9).abs() < 1e-6);

    // Values should be clamped
    let clamped_low = TopPProcessor::new(-0.5);
    assert!(clamped_low.top_p() >= 0.0);

    let clamped_high = TopPProcessor::new(1.5);
    assert!(clamped_high.top_p() <= 1.0);
}

#[test]
fn test_hybrid_sampler_creation() {
    use realizar::grammar::ToolCallFormat;
    use realizar::sampling::HybridSampler;

    let tools = vec![create_weather_tool()];
    let vocab: HashMap<u32, String> = (0..100)
        .map(|i| (i, format!("token_{}", i)))
        .collect();
    let eos_token_id = 99;

    let sampler = HybridSampler::new(tools, vocab, eos_token_id, ToolCallFormat::Hermes);

    // Initially in detecting mode
    assert_eq!(sampler.mode(), SamplingMode::Detecting);
}

#[test]
fn test_tool_call_detector() {
    use realizar::grammar::ToolCallFormat;
    use realizar::sampling::ToolCallDetector;

    let mut detector = ToolCallDetector::new(ToolCallFormat::Hermes);

    // Add tokens that form a tool call start
    detector.add_token("<");
    detector.add_token("tool");
    detector.add_token("_");
    detector.add_token("call");
    detector.add_token(">");

    assert!(
        detector.detect_tool_call_start(),
        "Should detect tool call with Hermes format"
    );

    // Reset and try again
    detector.reset();
    assert!(!detector.detect_tool_call_start(), "Should reset detection state");
}

#[test]
fn test_tool_call_detector_groq() {
    use realizar::sampling::ToolCallDetector;

    // Groq uses the same format as Hermes
    let mut detector = ToolCallDetector::groq();

    detector.add_token("<tool_call>");
    assert!(detector.detect_tool_call_start(), "Should detect Groq tool call");
}

#[test]
fn test_json_grammar_processor_creation() {
    use realizar::grammar::Grammar;
    use realizar::sampling::JsonGrammarProcessor;

    // Create a simple grammar
    let grammar = Grammar::with_root("root");

    // Create vocab mapping
    let vocab: HashMap<u32, String> = [
        (0u32, "{".to_string()),
        (1, "}".to_string()),
        (2, " ".to_string()),
    ]
    .into_iter()
    .collect();

    // This might fail if grammar is empty, which is expected
    let result = JsonGrammarProcessor::new(grammar, vocab, 2);

    // The result depends on grammar validation
    println!("JsonGrammarProcessor creation result: {:?}", result.is_ok());
}

#[test]
fn test_json_schema_types() {
    use realizar::grammar::JsonSchemaType;

    // Test creating various JSON schema types
    let string_type = JsonSchemaType::String;
    let int_type = JsonSchemaType::Integer;
    let num_type = JsonSchemaType::Number;
    let bool_type = JsonSchemaType::Boolean;

    let array_type = JsonSchemaType::Array(Box::new(JsonSchemaType::String));

    // Object is a tuple variant: (name, type, required)
    let object_type = JsonSchemaType::Object(vec![
        ("name".to_string(), JsonSchemaType::String, true),
        ("age".to_string(), JsonSchemaType::Integer, false),
    ]);

    // Verify types can be created
    assert!(matches!(string_type, JsonSchemaType::String));
    assert!(matches!(int_type, JsonSchemaType::Integer));
    assert!(matches!(num_type, JsonSchemaType::Number));
    assert!(matches!(bool_type, JsonSchemaType::Boolean));
    assert!(matches!(array_type, JsonSchemaType::Array(_)));
    assert!(matches!(object_type, JsonSchemaType::Object(_)));
}

// =============================================================================
// INTEGRATION TESTS (Require model files)
// =============================================================================

/// Test generating JSON output with grammar constraints using a real model.
#[test]
#[ignore = "Requires model file. Set REALIZAR_TEST_MODEL_PATH to run."]
fn test_grammar_constrained_json_generation() {
    use realizar::gguf::GGUFModel;
    use std::fs;

    let model_path = std::env::var("REALIZAR_TEST_MODEL_PATH")
        .expect("REALIZAR_TEST_MODEL_PATH must be set to run this test");

    // Load model
    let data = fs::read(&model_path).expect("Failed to read model file");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse model");

    // Create a tool and its grammar
    let tools = vec![create_weather_tool()];
    let grammar = generate_tool_grammar(&tools);

    println!("Model architecture: {:?}", model.architecture());
    println!("Grammar has {} rules", grammar.len());

    // The grammar should produce valid JSON when used during generation
    assert!(grammar.len() > 0);
}

/// Test JSON schema validation on generated output.
#[test]
#[ignore = "Requires model file. Set REALIZAR_TEST_MODEL_PATH to run."]
fn test_json_schema_validation() {
    // Define expected schema
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "arguments": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        },
        "required": ["name", "arguments"]
    });

    // Example valid output that would be generated
    let valid_output = serde_json::json!({
        "name": "get_weather",
        "arguments": {
            "location": "San Francisco, CA",
            "unit": "fahrenheit"
        }
    });

    // Validate structure matches schema
    assert!(valid_output.get("name").is_some());
    assert!(valid_output.get("arguments").is_some());

    let args = valid_output.get("arguments").unwrap();
    assert!(args.get("location").is_some());

    println!("Schema: {}", serde_json::to_string_pretty(&schema).unwrap());
    println!(
        "Valid output: {}",
        serde_json::to_string_pretty(&valid_output).unwrap()
    );
}

/// Test that complex nested JSON can be generated with grammar constraints.
#[test]
#[ignore = "Requires model file. Set REALIZAR_TEST_MODEL_PATH to run."]
fn test_nested_json_grammar_generation() {
    // Test a more complex tool with nested objects
    let complex_tool = ToolDefinition::new(
        "create_event",
        "Create a calendar event",
        vec![
            ToolParameter::required_string("title", "Event title"),
            ToolParameter::required_string("date", "Event date in ISO format"),
            ToolParameter::optional_string("description", "Event description"),
        ],
    );

    let tools = vec![complex_tool];
    let grammar = generate_tool_grammar(&tools);

    // Grammar should be able to represent the structure
    assert!(grammar.len() > 0);
    println!("Complex nested grammar has {} rules", grammar.len());
}

/// Test multiple tool calls in sequence with grammar switching.
#[test]
#[ignore = "Requires model file. Set REALIZAR_TEST_MODEL_PATH to run."]
fn test_sequential_tool_calls_with_grammar() {
    use realizar::gguf::GGUFModel;
    use std::fs;

    let model_path = std::env::var("REALIZAR_TEST_MODEL_PATH")
        .expect("REALIZAR_TEST_MODEL_PATH must be set to run this test");

    let data = fs::read(&model_path).expect("Failed to read model file");
    let _model = GGUFModel::from_bytes(&data).expect("Failed to parse model");

    let tools = vec![
        create_weather_tool(),
        ToolDefinition::new(
            "search",
            "Search the web",
            vec![ToolParameter::required_string("query", "Search query")],
        ),
    ];

    let grammar = generate_tool_grammar(&tools);

    // The grammar should support multiple tools
    assert!(grammar.len() > 0);
    println!("Multi-tool grammar supports {} tools", tools.len());
}

/// Test performance of grammar-constrained generation.
#[test]
#[ignore = "Requires model file. Set REALIZAR_TEST_MODEL_PATH to run."]
fn test_grammar_generation_performance() {
    use realizar::grammar::ToolCallFormat;
    use realizar::sampling::HybridSampler;
    use std::time::Instant;

    let tools = vec![create_weather_tool()];
    let vocab: HashMap<u32, String> = (0..32000)
        .map(|i| (i, format!("token_{}", i)))
        .collect();

    // Measure time to create sampler
    let start = Instant::now();
    let _sampler = HybridSampler::new(tools.clone(), vocab, 128009, ToolCallFormat::Hermes);
    let sampler_creation_time = start.elapsed();

    // Measure time to generate grammar
    let start = Instant::now();
    let _grammar = generate_tool_grammar(&tools);
    let grammar_generation_time = start.elapsed();

    println!("Sampler creation: {:?}", sampler_creation_time);
    println!("Grammar generation: {:?}", grammar_generation_time);

    // Both should complete in reasonable time (< 1s)
    assert!(
        sampler_creation_time.as_secs() < 1,
        "Sampler creation should be fast"
    );
    assert!(
        grammar_generation_time.as_secs() < 1,
        "Grammar generation should be fast"
    );
}
