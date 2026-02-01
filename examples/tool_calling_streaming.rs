//! Streaming Tool Calling Example
//!
//! Demonstrates streaming token generation with tool calling support.
//!
//! This example shows:
//! - Token-by-token generation with streaming events
//! - Real-time tool call detection during generation
//! - Event-driven streaming to clients
//! - Integration with the StreamingAgentExecutor
//!
//! # Run
//!
//! ```bash
//! cargo run --example tool_calling_streaming
//! ```

use realizar::agent::{
    AgentConfig, ClosureGenerator, StreamToken, StreamingAgentEvent, StreamingAgentExecutor,
};
use realizar::error::Result;
use realizar::grammar::{ToolCallFormat, ToolDefinition, ToolParameter, ToolResult};
use realizar::tools::DispatchingToolHandler;
use std::io::{self, Write};
use std::thread;
use std::time::Duration;

fn main() {
    println!("=== Realizar Streaming Tool Calling Demo ===\n");

    // -------------------------------------------------------------------------
    // 1. Basic Streaming
    // -------------------------------------------------------------------------
    println!("1. Basic Token Streaming\n");
    demo_basic_streaming();

    // -------------------------------------------------------------------------
    // 2. Streaming with Tool Calls
    // -------------------------------------------------------------------------
    println!("\n2. Streaming with Tool Call Detection\n");
    demo_streaming_with_tools();

    // -------------------------------------------------------------------------
    // 3. Simulated Chat with Streaming
    // -------------------------------------------------------------------------
    println!("\n3. Simulated Streaming Chat\n");
    demo_streaming_chat();

    println!("\n=== Demo Complete ===");
}

// =============================================================================
// Demo Functions
// =============================================================================

fn demo_basic_streaming() {
    println!("  Simulating token-by-token streaming output:");
    print!("  > ");
    io::stdout().flush().ok();

    // Simulate streaming tokens
    let tokens = ["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", "."];

    for token in tokens {
        print!("{}", token);
        io::stdout().flush().ok();
        thread::sleep(Duration::from_millis(100)); // Simulate generation time
    }
    println!("\n");
}

fn demo_streaming_with_tools() {
    // Create tools
    let tools = vec![
        ToolDefinition::new(
            "get_weather",
            "Get the current weather for a location",
            vec![
                ToolParameter::required_string("location", "The city and state"),
                ToolParameter::required_enum(
                    "unit",
                    "Temperature unit",
                    vec!["celsius".to_string(), "fahrenheit".to_string()],
                ),
            ],
        ),
    ];

    // Create handler
    let mut handler = DispatchingToolHandler::new();
    handler.register("get_weather", |call| {
        // Simulate API latency
        thread::sleep(Duration::from_millis(500));
        
        let args: serde_json::Value = serde_json::from_str(&call.arguments).unwrap_or_default();
        let location = args.get("location").and_then(|v| v.as_str()).unwrap_or("Unknown");
        
        Ok(ToolResult::success(
            call.id.clone(),
            serde_json::json!({
                "location": location,
                "temperature": 72,
                "condition": "sunny",
                "humidity": 45
            }).to_string(),
        ))
    });

    // Create streaming agent
    let config = AgentConfig::default()
        .with_max_iterations(3)
        .with_tool_format(ToolCallFormat::OpenAI);

    let mut agent = StreamingAgentExecutor::new(config, handler, tools);

    // Create a mock streaming generator
    // In real use, this would come from your LLM inference engine
    let tokens = vec![
        "Let", " me", " check", " the", " weather", ".", "\n",
        r#"{"name": "get_weather", "arguments": {"location": "Seattle", "unit": "fahrenheit"}}"#,
    ];
    let token_idx = std::cell::RefCell::new(0);

    let generator = ClosureGenerator::new(move |_prompt, _generated| {
        let mut idx = token_idx.borrow_mut();
        
        if *idx >= tokens.len() {
            return Ok(None);
        }

        let text = tokens[*idx].to_string();
        *idx += 1;
        
        // Simulate generation delay
        thread::sleep(Duration::from_millis(50));

        Ok(Some(StreamToken {
            text,
            token_id: Some(*idx as u32),
            is_final: *idx >= tokens.len(),
        }))
    });

    println!("  Streaming generation with tool call detection:");
    print!("  > ");
    io::stdout().flush().ok();

    let mut tool_call_shown = false;

    let result = agent.run_streaming(
        "What's the weather in Seattle?",
        generator,
        |event| {
            match event {
                StreamingAgentEvent::Token(tok) => {
                    // Stream tokens to output
                    print!("{}", tok.text);
                    io::stdout().flush().ok();
                }
                StreamingAgentEvent::ToolCallDetectionStarted => {
                    if !tool_call_shown {
                        println!("\n  [Tool call detected...]");
                        tool_call_shown = true;
                    }
                }
                StreamingAgentEvent::ToolCallExecuting { tool_call } => {
                    println!("  [Executing: {}({})]", tool_call.name, &tool_call.arguments[..50.min(tool_call.arguments.len())]);
                }
                StreamingAgentEvent::ToolCallCompleted { result } => {
                    let status = if result.success { "✓" } else { "✗" };
                    println!("  [Result {}: {}...]", status, &result.content[..60.min(result.content.len())]);
                }
                StreamingAgentEvent::ContinuingGeneration => {
                    print!("  > ");
                    io::stdout().flush().ok();
                }
                StreamingAgentEvent::Completed { final_response, tool_calls } => {
                    println!("\n  [Completed: {} tool calls, {} chars response]", 
                        tool_calls.len(), final_response.len());
                }
                StreamingAgentEvent::Error(e) => {
                    println!("\n  [Error: {}]", e);
                }
            }
        },
    );

    match result {
        Ok(r) => {
            println!("\n  Result:");
            println!("    Iterations: {}", r.iterations);
            println!("    Tool calls: {}", r.tool_calls.len());
            println!("    Completed: {}", r.completed_normally);
        }
        Err(e) => println!("  Error: {}", e),
    }
}

fn demo_streaming_chat() {
    println!("  Simulating a streaming chat conversation:");
    println!();

    // Create tools
    let tools = vec![
        ToolDefinition::new(
            "get_weather",
            "Get weather for a location",
            vec![ToolParameter::required_string("location", "The location")],
        ),
        ToolDefinition::new(
            "calculate",
            "Calculate a math expression",
            vec![ToolParameter::required_string("expression", "Math expression")],
        ),
    ];

    // Create handler
    let mut handler = DispatchingToolHandler::new();
    handler.register("get_weather", |call| {
        Ok(ToolResult::success(call.id.clone(), r#"{"temp": 72, "condition": "sunny"}"#.to_string()))
    });
    handler.register("calculate", |call| {
        Ok(ToolResult::success(call.id.clone(), r#"{"result": "42"}"#.to_string()))
    });

    let config = AgentConfig::default().with_max_iterations(2);
    let mut agent = StreamingAgentExecutor::new(config, handler, tools);

    // Simulate user message
    println!("  User: What's the weather like today?");
    print!("  Assistant: ");
    io::stdout().flush().ok();

    // Tokens for a simple response (no tool call)
    let response_tokens = vec![
        "I", "'d", " be", " happy", " to", " help", "!", " The", " weather",
        " looks", " beautiful", " today", " -", " sunny", " and", " warm", ".",
    ];
    let token_idx = std::cell::RefCell::new(0);

    let generator = ClosureGenerator::new(move |_prompt, _generated| {
        let mut idx = token_idx.borrow_mut();
        
        if *idx >= response_tokens.len() {
            return Ok(None);
        }

        let text = response_tokens[*idx].to_string();
        *idx += 1;
        
        thread::sleep(Duration::from_millis(30));

        Ok(Some(StreamToken {
            text,
            token_id: Some(*idx as u32),
            is_final: *idx >= response_tokens.len(),
        }))
    });

    let _ = agent.run_streaming(
        "What's the weather like today?",
        generator,
        |event| {
            if let StreamingAgentEvent::Token(tok) = event {
                print!("{}", tok.text);
                io::stdout().flush().ok();
            }
        },
    );

    println!("\n");

    // Second turn
    println!("  User: What is 6 * 7?");
    print!("  Assistant: ");
    io::stdout().flush().ok();

    let calc_tokens = vec![
        "6", " times", " 7", " equals", " 42", ".",
    ];
    let calc_idx = std::cell::RefCell::new(0);

    let calc_generator = ClosureGenerator::new(move |_prompt, _generated| {
        let mut idx = calc_idx.borrow_mut();
        
        if *idx >= calc_tokens.len() {
            return Ok(None);
        }

        let text = calc_tokens[*idx].to_string();
        *idx += 1;
        
        thread::sleep(Duration::from_millis(50));

        Ok(Some(StreamToken {
            text,
            token_id: Some(*idx as u32),
            is_final: *idx >= calc_tokens.len(),
        }))
    });

    // Reuse the agent (would need to recreate in real scenario due to ownership)
    let config2 = AgentConfig::default().with_max_iterations(2);
    let mut handler2 = DispatchingToolHandler::new();
    handler2.register("calculate", |call| {
        Ok(ToolResult::success(call.id.clone(), r#"{"result": "42"}"#.to_string()))
    });
    let tools2 = vec![ToolDefinition::new(
        "calculate",
        "Calculate a math expression",
        vec![ToolParameter::required_string("expression", "Math expression")],
    )];
    let mut agent2 = StreamingAgentExecutor::new(config2, handler2, tools2);

    let _ = agent2.run_streaming(
        "What is 6 * 7?",
        calc_generator,
        |event| {
            if let StreamingAgentEvent::Token(tok) = event {
                print!("{}", tok.text);
                io::stdout().flush().ok();
            }
        },
    );

    println!();
}
