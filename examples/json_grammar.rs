//! JSON Grammar-Constrained Generation Example
//!
//! Demonstrates grammar-constrained generation for structured JSON output.
//!
//! This example shows:
//! - Generating grammars from tool definitions
//! - LogitProcessor for constraining generation
//! - HybridSampler for automatic mode switching
//! - Validating generated JSON against schemas
//!
//! # Run
//!
//! ```bash
//! cargo run --example json_grammar
//! ```

use realizar::grammar::{
    generate_tool_grammar, Grammar, GrammarElement, GrammarRule, JsonSchemaType, ToolCallFormat,
    ToolDefinition, ToolParameter,
};
use realizar::sampling::{
    HybridSampler, LogitProcessor, LogitProcessorChain, RepetitionPenaltyProcessor, SamplingMode,
    TemperatureProcessor, ToolCallDetector, TopPProcessor,
};
use std::collections::HashMap;

fn main() {
    println!("=== Realizar JSON Grammar Demo ===\n");

    // -------------------------------------------------------------------------
    // 1. JSON Schema Types
    // -------------------------------------------------------------------------
    println!("1. JSON Schema Types\n");
    demo_json_schema_types();

    // -------------------------------------------------------------------------
    // 2. Tool Grammar Generation
    // -------------------------------------------------------------------------
    println!("\n2. Tool Grammar Generation\n");
    demo_tool_grammar();

    // -------------------------------------------------------------------------
    // 3. LogitProcessor Chain
    // -------------------------------------------------------------------------
    println!("\n3. LogitProcessor Chain\n");
    demo_logit_processor_chain();

    // -------------------------------------------------------------------------
    // 4. Tool Call Detection
    // -------------------------------------------------------------------------
    println!("\n4. Tool Call Detection\n");
    demo_tool_call_detection();

    // -------------------------------------------------------------------------
    // 5. HybridSampler Modes
    // -------------------------------------------------------------------------
    println!("\n5. HybridSampler Modes\n");
    demo_hybrid_sampler();

    // -------------------------------------------------------------------------
    // 6. JSON Validation
    // -------------------------------------------------------------------------
    println!("\n6. JSON Schema Validation\n");
    demo_json_validation();

    println!("\n=== Demo Complete ===");
}

// =============================================================================
// Demo Functions
// =============================================================================

fn demo_json_schema_types() {
    println!("  Available JSON Schema Types:");
    println!("  ----------------------------");

    // Primitive types
    let string_type = JsonSchemaType::String;
    let int_type = JsonSchemaType::Integer;
    let num_type = JsonSchemaType::Number;
    let bool_type = JsonSchemaType::Boolean;

    println!("  Primitive types:");
    println!("    - String: Text values");
    println!("    - Integer: Whole numbers");
    println!("    - Number: Floating point values");
    println!("    - Boolean: true/false");

    // Complex types
    let array_type = JsonSchemaType::Array(Box::new(JsonSchemaType::String));
    let object_type = JsonSchemaType::Object(vec![
        ("name".to_string(), JsonSchemaType::String, true),
        ("age".to_string(), JsonSchemaType::Integer, false),
    ]);

    println!("\n  Complex types:");
    println!("    - Array(String): List of strings");
    println!("    - Object(fields): Structured object with typed fields");

    // Show object structure
    println!("\n  Example Object Schema:");
    println!("    {{");
    println!("      \"name\": String (required)");
    println!("      \"age\": Integer (optional)");
    println!("    }}");
}

fn demo_tool_grammar() {
    // Create tools
    let tools = vec![
        ToolDefinition::new(
            "get_weather",
            "Get weather for a location",
            vec![
                ToolParameter::required_string("location", "City and state"),
                ToolParameter::required_enum(
                    "unit",
                    "Temperature unit",
                    vec!["celsius".to_string(), "fahrenheit".to_string()],
                ),
            ],
        ),
        ToolDefinition::new(
            "search",
            "Search the web",
            vec![
                ToolParameter::required_string("query", "Search query"),
                ToolParameter::optional_string("max_results", "Max results (as string)"),
            ],
        ),
    ];

    println!("  Generating grammar for {} tools...", tools.len());

    let grammar = generate_tool_grammar(&tools);

    println!("  Generated grammar:");
    println!("    Root rule: \"{}\"", grammar.root());
    println!("    Total rules: {}", grammar.len());

    // List rule names
    println!("    Rule names:");
    for name in grammar.rule_names().take(10) {
        println!("      - {}", name);
    }
    if grammar.len() > 10 {
        println!("      ... and {} more", grammar.len() - 10);
    }

    // Grammar can be validated
    match grammar.validate() {
        Ok(_) => println!("\n  Grammar validation: ✓ Valid"),
        Err(e) => println!("\n  Grammar validation: ✗ {}", e),
    }
}

fn demo_logit_processor_chain() {
    // Create a chain of logit processors
    let mut chain = LogitProcessorChain::new();

    // Add temperature scaling (reduces sharpness of distribution)
    chain.push(Box::new(TemperatureProcessor::new(0.7)));
    println!("  Added TemperatureProcessor(0.7)");

    // Add top-p nucleus sampling
    chain.push(Box::new(TopPProcessor::new(0.9)));
    println!("  Added TopPProcessor(0.9)");

    // Add repetition penalty
    chain.push(Box::new(RepetitionPenaltyProcessor::new(1.1)));
    println!("  Added RepetitionPenaltyProcessor(1.1)");

    println!("\n  Chain length: {} processors", chain.len());

    // Demonstrate processing
    let vocab_size = 100;
    let mut logits = vec![1.0f32; vocab_size];

    // Set some tokens to have higher probability
    logits[10] = 5.0; // High probability token
    logits[20] = 4.0;
    logits[30] = 3.0;

    println!("\n  Before processing:");
    println!("    Logit[10] = {:.2}", logits[10]);
    println!("    Logit[20] = {:.2}", logits[20]);
    println!("    Logit[50] = {:.2}", logits[50]);

    // Simulate input tokens with repetition
    let input_ids = vec![10u32, 20, 10, 30, 10]; // Token 10 appears multiple times

    chain.process(&input_ids, &mut logits);

    println!("\n  After processing (with repetition penalty on token 10):");
    println!("    Logit[10] = {:.2} (penalized for repetition)", logits[10]);
    println!("    Logit[20] = {:.2}", logits[20]);
    println!("    Logit[50] = {:.2}", logits[50]);
}

fn demo_tool_call_detection() {
    println!("  Tool Call Formats:");
    println!("  ------------------");

    // OpenAI format
    let openai_examples = [
        (r#"{"name": "func", "arguments": {}}"#, true),
        (r#"Hello world"#, false),
    ];

    println!("\n  OpenAI Format:");
    let mut detector = ToolCallDetector::new(ToolCallFormat::OpenAI);
    for (text, expected) in openai_examples {
        detector.reset();
        detector.add_token(text);
        let detected = detector.detect_tool_call_start();
        let status = if detected == expected { "✓" } else { "✗" };
        println!("    {} \"{}...\" -> {}", status, &text[..20.min(text.len())], detected);
    }

    // Hermes/Groq format
    let hermes_examples = [
        ("<tool_call>{...}</tool_call>", true),
        ("Regular text response", false),
    ];

    println!("\n  Hermes/Groq Format (<tool_call> tags):");
    let mut detector = ToolCallDetector::new(ToolCallFormat::Hermes);
    for (text, expected) in hermes_examples {
        detector.reset();
        detector.add_token(text);
        let detected = detector.detect_tool_call_start();
        let status = if detected == expected { "✓" } else { "✗" };
        println!("    {} \"{}...\" -> {}", status, &text[..25.min(text.len())], detected);
    }

    // Anthropic format
    println!("\n  Anthropic Format (<tool_use> tags):");
    let mut detector = ToolCallDetector::new(ToolCallFormat::Anthropic);
    detector.add_token("<tool_use>");
    println!("    \"<tool_use>\" -> {}", detector.detect_tool_call_start());
}

fn demo_hybrid_sampler() {
    // Create tools
    let tools = vec![ToolDefinition::new(
        "get_weather",
        "Get weather",
        vec![ToolParameter::required_string("location", "Location")],
    )];

    // Create a mock vocabulary
    let vocab: HashMap<u32, String> = (0..100)
        .map(|i| (i, format!("token_{}", i)))
        .collect();

    let eos_token_id = 99;

    // Create hybrid sampler
    let sampler = HybridSampler::new(tools, vocab, eos_token_id, ToolCallFormat::Hermes);

    println!("  Sampling Modes:");
    println!("  ---------------");
    println!("    - Detecting: Looking for tool call start");
    println!("    - FreeForm: Normal text generation");
    println!("    - JsonConstrained: Grammar-enforced JSON");

    println!("\n  Initial mode: {:?}", sampler.mode());

    // The sampler starts in Detecting mode
    assert_eq!(sampler.mode(), SamplingMode::Detecting);
    println!("  Sampler correctly starts in Detecting mode");

    println!("\n  Mode Transitions:");
    println!("    Detecting -> FreeForm: No tool call detected in first N tokens");
    println!("    Detecting -> JsonConstrained: Tool call start detected");
    println!("    JsonConstrained -> FreeForm: Tool call completed");
}

fn demo_json_validation() {
    // Define expected schema
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "arguments": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        },
        "required": ["name", "arguments"]
    });

    println!("  Expected JSON Schema:");
    println!("  ---------------------");
    println!("  {}", serde_json::to_string_pretty(&schema).unwrap());

    // Valid tool call
    let valid_output = r#"{
        "name": "get_weather",
        "arguments": {
            "location": "San Francisco, CA",
            "unit": "fahrenheit"
        }
    }"#;

    // Invalid outputs
    let invalid_missing_name = r#"{
        "arguments": {"location": "NYC"}
    }"#;

    let invalid_wrong_type = r#"{
        "name": 123,
        "arguments": {"location": "NYC"}
    }"#;

    println!("\n  Validation Examples:");
    println!("  --------------------");

    // Validate each
    let examples = [
        ("Valid tool call", valid_output, true),
        ("Missing 'name' field", invalid_missing_name, false),
        ("Wrong type for 'name'", invalid_wrong_type, false),
    ];

    for (label, json, expected_valid) in examples {
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(json);

        let is_valid = match &parsed {
            Ok(v) => {
                v.get("name").map_or(false, |n| n.is_string())
                    && v.get("arguments").map_or(false, |a| a.is_object())
            }
            Err(_) => false,
        };

        let status = if is_valid == expected_valid { "✓" } else { "✗" };
        let result = if is_valid { "valid" } else { "invalid" };
        println!("    {} {}: {}", status, label, result);
    }

    println!("\n  Grammar Enforcement:");
    println!("  --------------------");
    println!("  With grammar constraints, the model can ONLY generate valid JSON.");
    println!("  Invalid token sequences are masked during generation.");
    println!("  This ensures 100% valid output format.");
}
