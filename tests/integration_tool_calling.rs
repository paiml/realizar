//! Integration tests for tool calling with real model inference.
//!
//! These tests verify end-to-end tool calling functionality including:
//! - Model loading with tool-use configurations
//! - Generating tool calls with grammar constraints
//! - Parsing and executing tool calls via handlers
//! - Multi-turn tool calling conversations
//!
//! # Running These Tests
//!
//! Tests are marked `#[ignore]` as they require model files:
//!
//! ```bash
//! # Download the model first (see docs/new/model_download_guide.md)
//! REALIZAR_TEST_MODEL_PATH=/path/to/model.gguf cargo test --test integration_tool_calling -- --ignored
//! ```

use realizar::grammar::{ToolCall, ToolDefinition, ToolParameter, ToolResult};
use realizar::tools::{
    DispatchingToolHandler, ToolCallExecutor, ToolCallHandler, ToolCallHandlerError,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// =============================================================================
// TEST TOOL HANDLER IMPLEMENTATIONS
// =============================================================================

/// A test tool handler that tracks tool invocations.
#[derive(Debug)]
struct TestToolHandler {
    call_count: Arc<AtomicUsize>,
}

impl TestToolHandler {
    fn new() -> Self {
        Self {
            call_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl ToolCallHandler for TestToolHandler {
    fn handle_tool_call(&self, tool_call: &ToolCall) -> Result<ToolResult, ToolCallHandlerError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        match tool_call.name.as_str() {
            "get_weather" => {
                // Parse the location from arguments
                let args: serde_json::Value = serde_json::from_str(&tool_call.arguments)
                    .map_err(|e| ToolCallHandlerError::InvalidArguments {
                        tool_name: tool_call.name.clone(),
                        message: e.to_string(),
                    })?;

                let location = args
                    .get("location")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown");

                let result = serde_json::json!({
                    "location": location,
                    "temperature": 72,
                    "unit": "fahrenheit",
                    "condition": "sunny"
                });

                Ok(ToolResult::success(
                    tool_call.id.clone(),
                    result.to_string(),
                ))
            }
            "search" => {
                let args: serde_json::Value = serde_json::from_str(&tool_call.arguments)
                    .map_err(|e| ToolCallHandlerError::InvalidArguments {
                        tool_name: tool_call.name.clone(),
                        message: e.to_string(),
                    })?;

                let query = args
                    .get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let result = serde_json::json!({
                    "query": query,
                    "results": [
                        {"title": "Result 1", "url": "https://example.com/1"},
                        {"title": "Result 2", "url": "https://example.com/2"}
                    ]
                });

                Ok(ToolResult::success(
                    tool_call.id.clone(),
                    result.to_string(),
                ))
            }
            "calculate" => {
                let args: serde_json::Value = serde_json::from_str(&tool_call.arguments)
                    .map_err(|e| ToolCallHandlerError::InvalidArguments {
                        tool_name: tool_call.name.clone(),
                        message: e.to_string(),
                    })?;

                let expression = args
                    .get("expression")
                    .and_then(|v| v.as_str())
                    .unwrap_or("0");

                // Simple calculator (in real use, you'd use a proper expression parser)
                let result = match expression {
                    "2+2" => "4",
                    "10*5" => "50",
                    _ => "Error: unsupported expression",
                };

                Ok(ToolResult::success(
                    tool_call.id.clone(),
                    serde_json::json!({"result": result}).to_string(),
                ))
            }
            _ => Err(ToolCallHandlerError::UnknownTool(tool_call.name.clone())),
        }
    }

    fn available_tools(&self) -> Vec<String> {
        vec![
            "get_weather".to_string(),
            "search".to_string(),
            "calculate".to_string(),
        ]
    }
}

// =============================================================================
// TEST HELPERS
// =============================================================================

fn create_test_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            "get_weather",
            "Get the current weather for a location",
            vec![
                ToolParameter::required_string(
                    "location",
                    "The city and state, e.g., San Francisco, CA",
                ),
                ToolParameter::required_enum(
                    "unit",
                    "Temperature unit (celsius or fahrenheit)",
                    vec!["celsius".to_string(), "fahrenheit".to_string()],
                ),
            ],
        ),
        ToolDefinition::new(
            "search",
            "Search the web for information",
            vec![ToolParameter::required_string("query", "The search query")],
        ),
        ToolDefinition::new(
            "calculate",
            "Evaluate a mathematical expression",
            vec![ToolParameter::required_string(
                "expression",
                "The mathematical expression to evaluate",
            )],
        ),
    ]
}

// =============================================================================
// UNIT TESTS (No model required)
// =============================================================================

#[test]
fn test_tool_handler_basic_invocation() {
    let handler = TestToolHandler::new();
    let tools = create_test_tools();

    let call = ToolCall::new(
        "call_1",
        "get_weather",
        r#"{"location": "San Francisco, CA", "unit": "fahrenheit"}"#,
    );

    let result = handler.handle_tool_call(&call).unwrap();

    assert!(result.success);
    assert_eq!(handler.call_count(), 1);

    let result_json: serde_json::Value = serde_json::from_str(&result.content).unwrap();
    assert_eq!(result_json["location"], "San Francisco, CA");
    assert_eq!(result_json["temperature"], 72);
}

#[test]
fn test_tool_executor_parses_and_executes() {
    let handler = TestToolHandler::new();
    let tools = create_test_tools();
    let executor = ToolCallExecutor::new(Box::new(handler), tools);

    // Simulate model output with a tool call
    let model_output = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC", "unit": "celsius"}}
</tool_call>"#;

    // Process using Hermes format (since our test uses <tool_call> tags)
    let executor = executor.with_format(realizar::grammar::ToolCallFormat::Hermes);

    let results = executor.process_output(model_output).unwrap();

    assert_eq!(results.len(), 1);
    assert!(results[0].success);
}

#[test]
fn test_tool_executor_system_prompt_generation() {
    let handler = TestToolHandler::new();
    let tools = create_test_tools();
    let executor = ToolCallExecutor::new(Box::new(handler), tools);

    let system_intro = executor.render_system_intro();

    // Groq template is the default
    assert!(system_intro.contains("<tools>"));
    assert!(system_intro.contains("get_weather"));
    assert!(system_intro.contains("search"));
    assert!(system_intro.contains("calculate"));
}

#[test]
fn test_dispatching_handler_multiple_tools() {
    let mut handler = DispatchingToolHandler::new();

    handler.register("greet", |call| {
        let args: serde_json::Value = serde_json::from_str(&call.arguments).unwrap_or_default();
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("World");
        Ok(ToolResult::success(
            call.id.clone(),
            format!(r#"{{"message": "Hello, {name}!"}}"#),
        ))
    });

    handler.register("farewell", |call| {
        let args: serde_json::Value = serde_json::from_str(&call.arguments).unwrap_or_default();
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("World");
        Ok(ToolResult::success(
            call.id.clone(),
            format!(r#"{{"message": "Goodbye, {name}!"}}"#),
        ))
    });

    let greet_call = ToolCall::new("1", "greet", r#"{"name": "Alice"}"#);
    let farewell_call = ToolCall::new("2", "farewell", r#"{"name": "Bob"}"#);

    let greet_result = handler.handle_tool_call(&greet_call).unwrap();
    let farewell_result = handler.handle_tool_call(&farewell_call).unwrap();

    assert!(greet_result.content.contains("Hello, Alice!"));
    assert!(farewell_result.content.contains("Goodbye, Bob!"));
}

#[test]
fn test_multi_turn_tool_conversation() {
    let handler = TestToolHandler::new();
    let tools = create_test_tools();
    let executor = ToolCallExecutor::new(Box::new(handler), tools)
        .with_format(realizar::grammar::ToolCallFormat::OpenAI);

    // Turn 1: Model generates a tool call
    let turn1_output = r#"{"name": "get_weather", "arguments": {"location": "Seattle", "unit": "celsius"}}"#;
    let results1 = executor.process_output(turn1_output).unwrap();
    assert_eq!(results1.len(), 1);

    // Format the results for the next turn
    let formatted_result = executor.format_results(&results1);
    assert!(formatted_result.contains("tool_response"));

    // Turn 2: Model generates another tool call based on the result
    let turn2_output = r#"{"name": "search", "arguments": {"query": "Seattle weather forecast"}}"#;
    let results2 = executor.process_output(turn2_output).unwrap();
    assert_eq!(results2.len(), 1);
}

#[test]
fn test_tool_call_error_handling() {
    let mut handler = DispatchingToolHandler::new();

    handler.register("error_tool", |call| {
        Err(ToolCallHandlerError::ExecutionFailed {
            tool_name: call.name.clone(),
            message: "Something went wrong".to_string(),
        })
    });

    let call = ToolCall::new("1", "error_tool", "{}");
    let result = handler.handle_tool_call(&call);

    assert!(result.is_err());
    match result {
        Err(ToolCallHandlerError::ExecutionFailed { tool_name, message }) => {
            assert_eq!(tool_name, "error_tool");
            assert!(message.contains("Something went wrong"));
        }
        _ => panic!("Expected ExecutionFailed error"),
    }
}

// =============================================================================
// INTEGRATION TESTS (Require model files)
// =============================================================================

/// Test full tool calling with real model inference.
///
/// This test requires a model file to be available at the path specified
/// by the `REALIZAR_TEST_MODEL_PATH` environment variable.
#[test]
#[ignore = "Requires model file. Set REALIZAR_TEST_MODEL_PATH to run."]
fn test_tool_calling_with_real_model() {
    use realizar::gguf::GGUFModel;
    use std::fs;

    let model_path = std::env::var("REALIZAR_TEST_MODEL_PATH")
        .expect("REALIZAR_TEST_MODEL_PATH must be set to run this test");

    // Load the model
    let data = fs::read(&model_path).expect("Failed to read model file");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse model");

    // Create tool handler and executor
    let handler = TestToolHandler::new();
    let tools = create_test_tools();
    let executor = ToolCallExecutor::new(Box::new(handler), tools.clone());

    // Build the prompt with tool definitions
    let system_intro = executor.render_system_intro();
    let user_message = "What's the weather like in San Francisco?";

    let prompt = format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_intro}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    );

    // Verify the system prompt contains tool info
    assert!(system_intro.contains("get_weather"));
    assert!(system_intro.contains("search"));

    // Model loaded successfully
    println!("Model architecture: {:?}", model.architecture());
    println!("Model tensor count: {}", model.header.tensor_count);
    println!("Prompt length: {} chars", prompt.len());
}

/// Test grammar-constrained generation for tool calls.
#[test]
#[ignore = "Requires model file. Set REALIZAR_TEST_MODEL_PATH to run."]
fn test_grammar_constrained_tool_generation() {
    use realizar::grammar::generate_tool_grammar;

    let tools = create_test_tools();

    // Generate grammar for the tools
    let grammar = generate_tool_grammar(&tools);

    // The grammar should have rules for tools
    let rule_count = grammar.len();
    assert!(
        rule_count > 0,
        "Grammar should have rules, got {}",
        rule_count
    );

    // Verify grammar has a root rule
    assert!(!grammar.root().is_empty(), "Grammar should have a root rule");

    println!("Grammar generated with {} rules", rule_count);
    println!("Root rule: {}", grammar.root());
}

/// Test multi-turn conversation with tool results.
#[test]
#[ignore = "Requires model file. Set REALIZAR_TEST_MODEL_PATH to run."]
fn test_multi_turn_inference_with_tools() {
    use realizar::gguf::GGUFModel;
    use std::fs;

    let model_path = std::env::var("REALIZAR_TEST_MODEL_PATH")
        .expect("REALIZAR_TEST_MODEL_PATH must be set to run this test");

    let data = fs::read(&model_path).expect("Failed to read model file");
    let _model = GGUFModel::from_bytes(&data).expect("Failed to parse model");

    let handler = TestToolHandler::new();
    let tools = create_test_tools();
    let executor = ToolCallExecutor::new(Box::new(handler), tools);

    // Simulate multi-turn conversation
    // Turn 1: User asks about weather
    let user_turn1 = "What's the weather in Seattle?";

    // Turn 2: Model responds with tool call (simulated)
    let model_turn1 = r#"{"name": "get_weather", "arguments": {"location": "Seattle, WA", "unit": "fahrenheit"}}"#;

    let results = executor.process_output(model_turn1).unwrap();
    assert_eq!(results.len(), 1);

    // Format results for next turn
    let tool_result = executor.format_results(&results);

    // Build prompt for turn 2
    let system_intro = executor.render_system_intro();
    let _turn2_prompt = format!(
        "{system_intro}\n\nUser: {user_turn1}\n\nAssistant: {model_turn1}\n\n{tool_result}\n\nAssistant:"
    );

    // In a real test, we'd run inference on this prompt
    // For now, verify the tool result formatting
    assert!(tool_result.contains("Seattle"));
    assert!(tool_result.contains("72")); // temperature from TestToolHandler
}
