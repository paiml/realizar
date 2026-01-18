//! Tool Calling Example
//!
//! Demonstrates the tool calling system for LLM function calling.
//!
//! This example shows:
//! - Defining tools with parameters
//! - Creating tool handlers
//! - Using the AgentExecutor for automatic multi-turn conversations
//! - Processing tool call results
//!
//! # Run
//!
//! ```bash
//! cargo run --example tool_calling
//! ```

use realizar::agent::{AgentConfig, AgentExecutor, AgentEvent, AgentEventHandler, AgentResult};
use realizar::error::Result;
use realizar::grammar::{ToolCall, ToolDefinition, ToolParameter, ToolResult};
use realizar::tools::{
    DispatchingToolHandler, ToolCallExecutor, ToolCallHandler, ToolCallHandlerError,
    ToolPromptTemplate,
};

fn main() {
    println!("=== Realizar Tool Calling Demo ===\n");

    // -------------------------------------------------------------------------
    // 1. Define Tools
    // -------------------------------------------------------------------------
    println!("1. Defining Tools\n");
    demo_tool_definitions();

    // -------------------------------------------------------------------------
    // 2. Create Tool Handlers
    // -------------------------------------------------------------------------
    println!("\n2. Creating Tool Handlers\n");
    demo_tool_handlers();

    // -------------------------------------------------------------------------
    // 3. Low-Level Tool Execution
    // -------------------------------------------------------------------------
    println!("\n3. Low-Level Tool Execution (ToolCallExecutor)\n");
    demo_tool_executor();

    // -------------------------------------------------------------------------
    // 4. High-Level Agent Execution
    // -------------------------------------------------------------------------
    println!("\n4. High-Level Agent Execution (AgentExecutor)\n");
    demo_agent_executor();

    // -------------------------------------------------------------------------
    // 5. Agent with Event Handling
    // -------------------------------------------------------------------------
    println!("\n5. Agent with Event Handling\n");
    demo_agent_with_events();

    println!("\n=== Demo Complete ===");
}

// =============================================================================
// Demo Functions
// =============================================================================

fn demo_tool_definitions() {
    // Create a weather tool
    let weather_tool = ToolDefinition::new(
        "get_weather",
        "Get the current weather for a location",
        vec![
            ToolParameter::required_string(
                "location",
                "The city and state, e.g., 'San Francisco, CA'",
            ),
            ToolParameter::required_enum(
                "unit",
                "Temperature unit",
                vec!["celsius".to_string(), "fahrenheit".to_string()],
            ),
        ],
    );

    // Create a search tool
    let search_tool = ToolDefinition::new(
        "search",
        "Search the web for information",
        vec![
            ToolParameter::required_string("query", "The search query"),
            ToolParameter::optional_string("max_results", "Maximum number of results (default: 5)"),
        ],
    );

    // Create a calculator tool
    let calculator_tool = ToolDefinition::new(
        "calculate",
        "Evaluate a mathematical expression",
        vec![ToolParameter::required_string(
            "expression",
            "The math expression to evaluate, e.g., '2 + 2 * 3'",
        )],
    );

    println!("  Defined {} tools:", 3);
    println!("    - get_weather: Get weather for a location");
    println!("    - search: Search the web");
    println!("    - calculate: Evaluate math expressions");

    // Show tool definition structure
    println!("\n  Example tool definition (get_weather):");
    println!("    Name: {}", weather_tool.name);
    println!("    Description: {}", weather_tool.description);
    println!("    Parameters:");
    for param in &weather_tool.parameters {
        println!(
            "      - {}: {} (required: {})",
            param.name, param.description, param.required
        );
    }
}

fn demo_tool_handlers() {
    // Create a dispatching handler that routes to different implementations
    let mut handler = DispatchingToolHandler::new();

    // Register the get_weather tool
    handler.register("get_weather", |call| {
        // Parse arguments
        let args: serde_json::Value =
            serde_json::from_str(&call.arguments).map_err(|e| ToolCallHandlerError::InvalidArguments {
                tool_name: call.name.clone(),
                message: e.to_string(),
            })?;

        let location = args
            .get("location")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");
        let unit = args
            .get("unit")
            .and_then(|v| v.as_str())
            .unwrap_or("fahrenheit");

        // Simulate API call
        let temp = if unit == "celsius" { 22 } else { 72 };
        let result = serde_json::json!({
            "location": location,
            "temperature": temp,
            "unit": unit,
            "condition": "sunny",
            "humidity": 45
        });

        Ok(ToolResult::success(call.id.clone(), result.to_string()))
    });

    // Register the search tool
    handler.register("search", |call| {
        let args: serde_json::Value =
            serde_json::from_str(&call.arguments).unwrap_or_default();
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");

        let result = serde_json::json!({
            "query": query,
            "results": [
                {"title": "Result 1", "url": "https://example.com/1", "snippet": "First result..."},
                {"title": "Result 2", "url": "https://example.com/2", "snippet": "Second result..."}
            ]
        });

        Ok(ToolResult::success(call.id.clone(), result.to_string()))
    });

    // Register the calculator tool
    handler.register("calculate", |call| {
        let args: serde_json::Value =
            serde_json::from_str(&call.arguments).unwrap_or_default();
        let expression = args
            .get("expression")
            .and_then(|v| v.as_str())
            .unwrap_or("0");

        // Simple expression evaluation (in production, use a proper parser)
        let result = match expression {
            "2 + 2" => "4",
            "10 * 5" => "50",
            "100 / 4" => "25",
            _ => "Error: expression not supported in demo",
        };

        Ok(ToolResult::success(
            call.id.clone(),
            serde_json::json!({"result": result}).to_string(),
        ))
    });

    println!("  Created DispatchingToolHandler with {} tools", handler.available_tools().len());
    println!("  Available tools: {:?}", handler.available_tools());

    // Test a tool call
    let test_call = ToolCall::new(
        "call_1",
        "get_weather",
        r#"{"location": "New York, NY", "unit": "fahrenheit"}"#,
    );

    let result = handler.handle_tool_call(&test_call);
    println!("\n  Test call to get_weather:");
    println!("    Arguments: {}", test_call.arguments);
    match result {
        Ok(r) => println!("    Result: {}", r.content),
        Err(e) => println!("    Error: {}", e),
    }
}

fn demo_tool_executor() {
    // Create tools
    let tools = create_demo_tools();

    // Create handler
    let mut handler = DispatchingToolHandler::new();
    handler.register("get_weather", |call| {
        Ok(ToolResult::success(
            call.id.clone(),
            r#"{"temperature": 72, "condition": "sunny"}"#.to_string(),
        ))
    });
    handler.register("search", |call| {
        Ok(ToolResult::success(
            call.id.clone(),
            r#"{"results": ["Result 1", "Result 2"]}"#.to_string(),
        ))
    });

    // Create executor
    let executor = ToolCallExecutor::new(Box::new(handler), tools);

    // Get system prompt with tool definitions
    let system_prompt = executor.render_system_intro();
    println!("  System prompt (first 200 chars):");
    println!("    {}...", &system_prompt[..200.min(system_prompt.len())]);

    // Simulate model output containing a tool call
    let model_output = r#"I'll check the weather for you.
{"name": "get_weather", "arguments": {"location": "Seattle, WA", "unit": "celsius"}}"#;

    println!("\n  Simulated model output:");
    println!("    {}", model_output);

    // Process the output
    let results = executor.process_output(model_output).unwrap();
    println!("\n  Parsed {} tool call(s)", results.len());

    for result in &results {
        println!("    Tool call ID: {}", result.tool_call_id);
        println!("    Success: {}", result.success);
        println!("    Content: {}", result.content);
    }

    // Format results for the model
    let formatted = executor.format_results(&results);
    println!("\n  Formatted result for model:");
    println!("    {}", formatted);
}

fn demo_agent_executor() {
    // Create tools and handler
    let tools = create_demo_tools();
    let mut handler = DispatchingToolHandler::new();

    handler.register("get_weather", |call| {
        let args: serde_json::Value = serde_json::from_str(&call.arguments).unwrap_or_default();
        let location = args.get("location").and_then(|v| v.as_str()).unwrap_or("Unknown");

        Ok(ToolResult::success(
            call.id.clone(),
            serde_json::json!({
                "location": location,
                "temperature": 72,
                "condition": "sunny"
            })
            .to_string(),
        ))
    });

    // Create agent with configuration
    let config = AgentConfig::default()
        .with_max_iterations(5)
        .with_tool_trace(true);

    let mut agent = AgentExecutor::new(config, handler, tools);

    println!("  Created AgentExecutor with {} tools", agent.tools().len());
    println!("  Max iterations: 5");

    // Simulate a generation function
    // In real use, this would call your LLM
    let mut iteration = 0;
    let generate_fn = |prompt: &str| -> Result<String> {
        iteration += 1;
        println!("    [Generation iteration {}]", iteration);

        if iteration == 1 {
            // First call: return a tool call
            Ok(r#"Let me check the weather for you.
{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}"#
                .to_string())
        } else {
            // Second call: return final response using the tool result
            Ok("Based on the weather data, it's currently 72°F and sunny in San Francisco. \
                Perfect weather for outdoor activities!".to_string())
        }
    };

    // Run the agent
    let result = agent.run_simple("What's the weather in San Francisco?", generate_fn);

    match result {
        Ok(r) => {
            println!("\n  Agent completed!");
            println!("    Iterations: {}", r.iterations);
            println!("    Tool calls made: {}", r.tool_calls.len());
            println!("    Completed normally: {}", r.completed_normally);
            println!("    Final response: {}", &r.final_response[..100.min(r.final_response.len())]);
            if let Some(trace) = &r.trace {
                println!("\n    Trace (first 300 chars):");
                println!("      {}...", &trace[..300.min(trace.len())]);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
}

fn demo_agent_with_events() {
    // Create tools and handler
    let tools = create_demo_tools();
    let mut handler = DispatchingToolHandler::new();

    handler.register("get_weather", |call| {
        Ok(ToolResult::success(
            call.id.clone(),
            r#"{"temperature": 72, "condition": "sunny"}"#.to_string(),
        ))
    });

    let config = AgentConfig::default().with_max_iterations(3);
    let mut agent = AgentExecutor::new(config, handler, tools);

    // Create a custom event handler
    struct PrintingEventHandler;

    impl AgentEventHandler for PrintingEventHandler {
        fn on_event(&mut self, event: AgentEvent) {
            match event {
                AgentEvent::GenerationStarted { iteration, total_tokens } => {
                    println!("    📝 Starting generation (iteration {}, {} tokens so far)", iteration, total_tokens);
                }
                AgentEvent::ToolCallDetected { tool_call } => {
                    println!("    🔧 Tool call detected: {}", tool_call.name);
                }
                AgentEvent::ToolCallExecuting { tool_name } => {
                    println!("    ⚙️  Executing tool: {}", tool_name);
                }
                AgentEvent::ToolCallCompleted { result } => {
                    let status = if result.success { "✅" } else { "❌" };
                    println!("    {} Tool completed: {}", status, result.tool_call_id);
                }
                AgentEvent::IterationCompleted { had_tool_call, .. } => {
                    let icon = if had_tool_call { "🔄" } else { "✨" };
                    println!("    {} Iteration completed (tool call: {})", icon, had_tool_call);
                }
                AgentEvent::Completed { iterations, total_tokens } => {
                    println!("    🎉 Agent completed: {} iterations, {} tokens", iterations, total_tokens);
                }
                _ => {}
            }
        }
    }

    let mut event_handler = PrintingEventHandler;

    // Simple generate function
    let generate_fn = |_: &str| -> Result<String> {
        Ok("The weather looks great today!".to_string())
    };

    println!("  Running agent with event handler:");
    let _ = agent.run("What's the weather?", generate_fn, &mut event_handler);
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_demo_tools() -> Vec<ToolDefinition> {
    vec![
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
        ToolDefinition::new(
            "search",
            "Search the web for information",
            vec![ToolParameter::required_string("query", "The search query")],
        ),
    ]
}
