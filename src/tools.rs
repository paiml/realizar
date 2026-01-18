//! Tool prompt templates for model-specific tool calling formats.
//!
//! This module provides the [`ToolPromptTemplate`] trait and implementations for
//! different model architectures that support tool/function calling. Each template
//! handles the model-specific prompt formatting for tools.
//!
//! # Supported Formats
//!
//! - **Groq Tool-Use** (default): Uses `<tool_call>` XML tags, best for Llama-3-Groq-8B-Tool-Use
//! - **Llama 3 Instruct**: Uses ipython header format for Meta's Llama 3 Instruct models
//! - **OpenAI**: Standard JSON format for OpenAI-compatible models
//!
//! # Example
//!
//! ```rust
//! use realizar::tools::{ToolPromptTemplate, GroqToolTemplate};
//! use realizar::grammar::{ToolDefinition, ToolParameter, ToolCall, ToolResult};
//!
//! // Create a tool definition
//! let tools = vec![
//!     ToolDefinition::new(
//!         "get_weather",
//!         "Get the current weather in a location",
//!         vec![ToolParameter::required_string("location", "City and state, e.g., San Francisco, CA")],
//!     ),
//! ];
//!
//! // Create the Groq template (default for tool-use)
//! let template = GroqToolTemplate::new();
//!
//! // Render the system prompt with tool definitions
//! let system_prompt = template.render_system_intro(&tools);
//! assert!(system_prompt.contains("<tools>"));
//! assert!(system_prompt.contains("get_weather"));
//! ```

use crate::grammar::{ToolCall, ToolDefinition, ToolResult};
use std::fmt::Debug;

// =============================================================================
// TOOL PROMPT TEMPLATE TRAIT
// =============================================================================

/// Trait for model-specific tool prompt formatting.
///
/// Different LLMs expect different prompt formats for tool calling. This trait
/// abstracts the prompt generation to support multiple model architectures.
///
/// # Required Methods
///
/// - [`render_system_intro`](ToolPromptTemplate::render_system_intro): Format system prompt with tool definitions
/// - [`render_tool_call`](ToolPromptTemplate::render_tool_call): Format a tool call for the conversation
/// - [`render_tool_result`](ToolPromptTemplate::render_tool_result): Format a tool result for the conversation
/// - [`get_stop_tokens`](ToolPromptTemplate::get_stop_tokens): Get stop tokens that indicate end of tool call
/// - [`detect_tool_call_start`](ToolPromptTemplate::detect_tool_call_start): Detect if text starts a tool call
pub trait ToolPromptTemplate: Send + Sync + Debug {
    /// Render the system prompt introduction with tool definitions.
    ///
    /// This should be included at the start of the system message to inform
    /// the model about available tools.
    ///
    /// # Arguments
    ///
    /// * `tools` - Available tool definitions
    ///
    /// # Returns
    ///
    /// Formatted system prompt string
    fn render_system_intro(&self, tools: &[ToolDefinition]) -> String;

    /// Render a tool call for inclusion in the conversation.
    ///
    /// Used when formatting the model's tool call output.
    ///
    /// # Arguments
    ///
    /// * `tool_call` - The tool call to format
    ///
    /// # Returns
    ///
    /// Formatted tool call string
    fn render_tool_call(&self, tool_call: &ToolCall) -> String;

    /// Render a tool result for inclusion in the conversation.
    ///
    /// Used when providing the result of a tool execution back to the model.
    ///
    /// # Arguments
    ///
    /// * `result` - The tool result to format
    ///
    /// # Returns
    ///
    /// Formatted tool result string
    fn render_tool_result(&self, result: &ToolResult) -> String;

    /// Get the stop tokens that indicate the end of a tool call.
    ///
    /// The model should stop generating when these tokens are produced.
    fn get_stop_tokens(&self) -> Vec<String>;

    /// Detect if the given text indicates the start of a tool call.
    ///
    /// Used for hybrid sampling to determine when to switch to grammar-constrained mode.
    ///
    /// # Arguments
    ///
    /// * `text` - The generated text to check
    ///
    /// # Returns
    ///
    /// `true` if the text appears to start a tool call
    fn detect_tool_call_start(&self, text: &str) -> bool;

    /// Get the template name for logging and debugging.
    fn name(&self) -> &'static str;
}

// =============================================================================
// GROQ TOOL-USE TEMPLATE (DEFAULT)
// =============================================================================

/// Tool prompt template for Groq's Llama-3-Groq-8B-Tool-Use model.
///
/// This is the default template for tool calling, designed for the best-performing
/// open-source 8B model for function calling (89.06% BFCL score).
///
/// # Format
///
/// Uses `<tool_call>` XML tags with JSON content:
///
/// ```text
/// <tool_call>
/// {"name": "function_name", "arguments": {"arg1": "value1"}}
/// </tool_call>
/// ```
///
/// # Recommended Settings
///
/// Per the Groq model card:
/// - Temperature: 0.5
/// - Top-P: 0.65
///
/// # Example
///
/// ```rust
/// use realizar::tools::{ToolPromptTemplate, GroqToolTemplate};
/// use realizar::grammar::{ToolDefinition, ToolCall};
///
/// let template = GroqToolTemplate::new();
///
/// // Check stop tokens
/// let stops = template.get_stop_tokens();
/// assert!(stops.contains(&"</tool_call>".to_string()));
///
/// // Detect tool call start
/// assert!(template.detect_tool_call_start("<tool_call>"));
/// assert!(!template.detect_tool_call_start("Hello world"));
/// ```
#[derive(Debug, Clone, Default)]
pub struct GroqToolTemplate {
    _private: (),
}

impl GroqToolTemplate {
    /// Create a new Groq tool template.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Format tool definitions as JSON for the system prompt.
    fn format_tool_definitions(&self, tools: &[ToolDefinition]) -> String {
        let tool_jsons: Vec<String> = tools
            .iter()
            .map(|tool| {
                let params_json = self.format_parameters(&tool.parameters);
                format!(
                    r#"{{
  "name": "{}",
  "description": "{}",
  "parameters": {}
}}"#,
                    tool.name, tool.description, params_json
                )
            })
            .collect();

        tool_jsons.join("\n")
    }

    /// Format tool parameters as JSON schema.
    fn format_parameters(&self, params: &[crate::grammar::ToolParameter]) -> String {
        use crate::grammar::ToolParameterType;

        let props: Vec<String> = params
            .iter()
            .map(|p| {
                let type_str = match &p.param_type {
                    ToolParameterType::String => "\"string\"".to_string(),
                    ToolParameterType::Integer => "\"integer\"".to_string(),
                    ToolParameterType::Number => "\"number\"".to_string(),
                    ToolParameterType::Boolean => "\"boolean\"".to_string(),
                    ToolParameterType::Enum(values) => {
                        let vals: Vec<String> = values.iter().map(|v| format!("\"{v}\"")).collect();
                        format!("\"string\", \"enum\": [{}]", vals.join(", "))
                    },
                    ToolParameterType::Array { .. } => "\"array\"".to_string(),
                    ToolParameterType::Object { .. } => "\"object\"".to_string(),
                };
                format!(
                    r#"    "{}": {{
      "description": "{}",
      "type": {}
    }}"#,
                    p.name, p.description, type_str
                )
            })
            .collect();

        let required: Vec<String> = params
            .iter()
            .filter(|p| p.required)
            .map(|p| format!("\"{}\"", p.name))
            .collect();

        format!(
            r#"{{
  "type": "object",
  "properties": {{
{}
  }},
  "required": [{}]
}}"#,
            props.join(",\n"),
            required.join(", ")
        )
    }
}

impl ToolPromptTemplate for GroqToolTemplate {
    fn render_system_intro(&self, tools: &[ToolDefinition]) -> String {
        let tool_definitions = self.format_tool_definitions(tools);

        format!(
            r#"You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

Here are the available tools:
<tools>
{tool_definitions}
</tools>"#
        )
    }

    fn render_tool_call(&self, tool_call: &ToolCall) -> String {
        format!(
            r#"<tool_call>
{{"id": "{}", "name": "{}", "arguments": {}}}
</tool_call>"#,
            tool_call.id, tool_call.name, tool_call.arguments
        )
    }

    fn render_tool_result(&self, result: &ToolResult) -> String {
        if result.success {
            format!(
                r#"<tool_response>
{{"id": "{}", "result": {}}}
</tool_response>"#,
                result.tool_call_id, result.content
            )
        } else {
            format!(
                r#"<tool_response>
{{"id": "{}", "error": "{}"}}
</tool_response>"#,
                result.tool_call_id, result.content
            )
        }
    }

    fn get_stop_tokens(&self) -> Vec<String> {
        vec![
            "</tool_call>".to_string(),
            "<|eot_id|>".to_string(),
        ]
    }

    fn detect_tool_call_start(&self, text: &str) -> bool {
        text.contains("<tool_call>")
    }

    fn name(&self) -> &'static str {
        "groq_tool"
    }
}

// =============================================================================
// LLAMA 3 INSTRUCT TEMPLATE
// =============================================================================

/// Tool prompt template for Meta's Llama 3 Instruct models.
///
/// Uses the ipython header format for tool/function calling as specified
/// in the Llama 3 documentation.
///
/// # Format
///
/// Uses special tokens for tool interaction:
///
/// ```text
/// <|start_header_id|>ipython<|end_header_id|>
/// {"name": "function_name", "parameters": {...}}
/// ```
///
/// # Example
///
/// ```rust
/// use realizar::tools::{ToolPromptTemplate, Llama3InstructToolTemplate};
/// use realizar::grammar::ToolDefinition;
///
/// let template = Llama3InstructToolTemplate::new();
///
/// let tools = vec![
///     ToolDefinition::new("search", "Search the web", vec![]),
/// ];
///
/// let system = template.render_system_intro(&tools);
/// assert!(system.contains("Environment: ipython"));
/// ```
#[derive(Debug, Clone, Default)]
pub struct Llama3InstructToolTemplate {
    _private: (),
}

impl Llama3InstructToolTemplate {
    /// Create a new Llama 3 Instruct tool template.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Format tools as Python function definitions for the system prompt.
    fn format_tools_as_functions(&self, tools: &[ToolDefinition]) -> String {
        tools
            .iter()
            .map(|tool| {
                let params: Vec<String> = tool
                    .parameters
                    .iter()
                    .map(|p| {
                        let type_hint = self.python_type_hint(&p.param_type);
                        if p.required {
                            format!("{}: {}", p.name, type_hint)
                        } else {
                            format!("{}: {} = None", p.name, type_hint)
                        }
                    })
                    .collect();

                let param_docs: Vec<String> = tool
                    .parameters
                    .iter()
                    .map(|p| format!("        {}: {}", p.name, p.description))
                    .collect();

                format!(
                    r#"def {}({}):
    """{}
    
    Args:
{}
    """
    pass"#,
                    tool.name,
                    params.join(", "),
                    tool.description,
                    param_docs.join("\n")
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Get Python type hint for a parameter type.
    fn python_type_hint(&self, param_type: &crate::grammar::ToolParameterType) -> &'static str {
        use crate::grammar::ToolParameterType;

        match param_type {
            ToolParameterType::String => "str",
            ToolParameterType::Integer => "int",
            ToolParameterType::Number => "float",
            ToolParameterType::Boolean => "bool",
            ToolParameterType::Enum(_) => "str",
            ToolParameterType::Array { .. } => "list",
            ToolParameterType::Object { .. } => "dict",
        }
    }
}

impl ToolPromptTemplate for Llama3InstructToolTemplate {
    fn render_system_intro(&self, tools: &[ToolDefinition]) -> String {
        let functions = self.format_tools_as_functions(tools);

        format!(
            r#"Environment: ipython
Tools: You have access to the following functions. Call them using JSON format with "name" and "parameters" fields.

{functions}

When you need to call a function, respond with a JSON object in the following format:
{{"name": "function_name", "parameters": {{"param1": "value1"}}}}"#
        )
    }

    fn render_tool_call(&self, tool_call: &ToolCall) -> String {
        format!(
            r#"<|start_header_id|>ipython<|end_header_id|>
{{"name": "{}", "parameters": {}}}<|eot_id|>"#,
            tool_call.name, tool_call.arguments
        )
    }

    fn render_tool_result(&self, result: &ToolResult) -> String {
        if result.success {
            format!(
                r#"<|start_header_id|>ipython<|end_header_id|>
{}<|eot_id|>"#,
                result.content
            )
        } else {
            format!(
                r#"<|start_header_id|>ipython<|end_header_id|>
Error: {}<|eot_id|>"#,
                result.content
            )
        }
    }

    fn get_stop_tokens(&self) -> Vec<String> {
        vec![
            "<|eot_id|>".to_string(),
            "<|python_tag|>".to_string(),
        ]
    }

    fn detect_tool_call_start(&self, text: &str) -> bool {
        // Detect JSON-like tool call patterns
        text.contains(r#"{"name":"#) || 
        text.contains(r#"{"name" :"#) ||
        text.contains(r#"{ "name":"#)
    }

    fn name(&self) -> &'static str {
        "llama3_instruct"
    }
}

// =============================================================================
// OPENAI-STYLE TEMPLATE
// =============================================================================

/// Tool prompt template for OpenAI-compatible models.
///
/// Uses standard JSON format for tool calling, compatible with models
/// that follow the OpenAI function calling convention.
///
/// # Format
///
/// ```text
/// {"name": "function_name", "arguments": {"arg1": "value1"}}
/// ```
///
/// # Example
///
/// ```rust
/// use realizar::tools::{ToolPromptTemplate, OpenAIToolTemplate};
/// use realizar::grammar::ToolDefinition;
///
/// let template = OpenAIToolTemplate::new();
/// let tools = vec![ToolDefinition::new("test", "Test function", vec![])];
///
/// let system = template.render_system_intro(&tools);
/// assert!(system.contains("functions"));
/// ```
#[derive(Debug, Clone, Default)]
pub struct OpenAIToolTemplate {
    _private: (),
}

impl OpenAIToolTemplate {
    /// Create a new OpenAI tool template.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Format tool as OpenAI function schema.
    fn format_function_schema(&self, tool: &ToolDefinition) -> String {
        use crate::grammar::ToolParameterType;

        let props: Vec<String> = tool
            .parameters
            .iter()
            .map(|p| {
                let type_str = match &p.param_type {
                    ToolParameterType::String => "string",
                    ToolParameterType::Integer => "integer",
                    ToolParameterType::Number => "number",
                    ToolParameterType::Boolean => "boolean",
                    ToolParameterType::Enum(_) => "string",
                    ToolParameterType::Array { .. } => "array",
                    ToolParameterType::Object { .. } => "object",
                };
                format!(
                    r#"        "{}": {{"type": "{}", "description": "{}"}}"#,
                    p.name, type_str, p.description
                )
            })
            .collect();

        let required: Vec<String> = tool
            .parameters
            .iter()
            .filter(|p| p.required)
            .map(|p| format!("\"{}\"", p.name))
            .collect();

        format!(
            r#"{{
  "name": "{}",
  "description": "{}",
  "parameters": {{
    "type": "object",
    "properties": {{
{}
    }},
    "required": [{}]
  }}
}}"#,
            tool.name,
            tool.description,
            props.join(",\n"),
            required.join(", ")
        )
    }
}

impl ToolPromptTemplate for OpenAIToolTemplate {
    fn render_system_intro(&self, tools: &[ToolDefinition]) -> String {
        let functions: Vec<String> = tools
            .iter()
            .map(|t| self.format_function_schema(t))
            .collect();

        format!(
            r#"You have access to the following functions. To call a function, respond with a JSON object containing "name" and "arguments" fields.

Available functions:
{}

Call functions by responding with:
{{"name": "function_name", "arguments": {{"param1": "value1"}}}}"#,
            functions.join("\n\n")
        )
    }

    fn render_tool_call(&self, tool_call: &ToolCall) -> String {
        format!(
            r#"{{"id": "{}", "name": "{}", "arguments": {}}}"#,
            tool_call.id, tool_call.name, tool_call.arguments
        )
    }

    fn render_tool_result(&self, result: &ToolResult) -> String {
        if result.success {
            format!(
                r#"{{"tool_call_id": "{}", "content": {}}}"#,
                result.tool_call_id, result.content
            )
        } else {
            format!(
                r#"{{"tool_call_id": "{}", "error": "{}"}}"#,
                result.tool_call_id, result.content
            )
        }
    }

    fn get_stop_tokens(&self) -> Vec<String> {
        vec![
            "}\n".to_string(),
            "}}\n".to_string(),
        ]
    }

    fn detect_tool_call_start(&self, text: &str) -> bool {
        text.contains(r#"{"name":"#) || 
        text.contains(r#"{"name" :"#) ||
        text.contains(r#"{ "name":"#)
    }

    fn name(&self) -> &'static str {
        "openai"
    }
}

// =============================================================================
// TEMPLATE FACTORY
// =============================================================================

/// Template type identifier for factory creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolTemplateType {
    /// Groq Tool-Use format (default)
    Groq,
    /// Llama 3 Instruct format
    Llama3Instruct,
    /// OpenAI-compatible format
    OpenAI,
}

impl Default for ToolTemplateType {
    fn default() -> Self {
        Self::Groq
    }
}

/// Create a tool template by type.
///
/// # Example
///
/// ```rust
/// use realizar::tools::{create_template, ToolTemplateType};
///
/// let template = create_template(ToolTemplateType::Groq);
/// assert_eq!(template.name(), "groq_tool");
///
/// let template = create_template(ToolTemplateType::Llama3Instruct);
/// assert_eq!(template.name(), "llama3_instruct");
/// ```
#[must_use]
pub fn create_template(template_type: ToolTemplateType) -> Box<dyn ToolPromptTemplate> {
    match template_type {
        ToolTemplateType::Groq => Box::new(GroqToolTemplate::new()),
        ToolTemplateType::Llama3Instruct => Box::new(Llama3InstructToolTemplate::new()),
        ToolTemplateType::OpenAI => Box::new(OpenAIToolTemplate::new()),
    }
}

/// Detect the best template type for a model name.
///
/// Inspects the model name to determine which tool template format to use.
///
/// # Example
///
/// ```rust
/// use realizar::tools::{detect_template_type, ToolTemplateType};
///
/// assert_eq!(detect_template_type("Llama-3-Groq-8B-Tool-Use"), ToolTemplateType::Groq);
/// assert_eq!(detect_template_type("Meta-Llama-3-8B-Instruct"), ToolTemplateType::Llama3Instruct);
/// assert_eq!(detect_template_type("gpt-4"), ToolTemplateType::OpenAI);
/// ```
#[must_use]
pub fn detect_template_type(model_name: &str) -> ToolTemplateType {
    let name_lower = model_name.to_lowercase();

    if name_lower.contains("groq") && name_lower.contains("tool") {
        ToolTemplateType::Groq
    } else if name_lower.contains("llama-3") || name_lower.contains("llama3") {
        if name_lower.contains("instruct") {
            ToolTemplateType::Llama3Instruct
        } else {
            ToolTemplateType::Groq // Default for Llama 3 is Groq format
        }
    } else if name_lower.contains("gpt") || name_lower.contains("openai") {
        ToolTemplateType::OpenAI
    } else {
        // Default to Groq format for best tool-calling performance
        ToolTemplateType::Groq
    }
}

// =============================================================================
// TOOL CALL HANDLER TRAIT
// =============================================================================

/// Trait for handling tool call execution by clients.
///
/// Clients of this library implement this trait to provide the actual tool
/// implementations. When the model generates a tool call, the handler is
/// invoked to execute it and return a result.
///
/// # Example
///
/// ```rust
/// use realizar::tools::{ToolCallHandler, ToolCallHandlerError};
/// use realizar::grammar::{ToolCall, ToolResult};
///
/// struct MyToolHandler;
///
/// impl ToolCallHandler for MyToolHandler {
///     fn handle_tool_call(&self, tool_call: &ToolCall) -> Result<ToolResult, ToolCallHandlerError> {
///         match tool_call.name.as_str() {
///             "get_weather" => {
///                 // Parse arguments and execute the tool
///                 let result = format!("{{\"temperature\": 72, \"condition\": \"sunny\"}}");
///                 Ok(ToolResult::success(tool_call.id.clone(), result))
///             }
///             _ => Err(ToolCallHandlerError::UnknownTool(tool_call.name.clone())),
///         }
///     }
///
///     fn available_tools(&self) -> Vec<String> {
///         vec!["get_weather".to_string()]
///     }
/// }
/// ```
pub trait ToolCallHandler: Send + Sync + Debug {
    /// Execute a tool call and return the result.
    ///
    /// # Arguments
    ///
    /// * `tool_call` - The tool call parsed from model output
    ///
    /// # Returns
    ///
    /// The result of executing the tool, or an error
    fn handle_tool_call(
        &self,
        tool_call: &crate::grammar::ToolCall,
    ) -> Result<crate::grammar::ToolResult, ToolCallHandlerError>;

    /// Get the list of available tool names.
    ///
    /// Used for validation and to provide tool definitions to the model.
    fn available_tools(&self) -> Vec<String>;

    /// Check if a tool is available.
    fn has_tool(&self, name: &str) -> bool {
        self.available_tools().contains(&name.to_string())
    }
}

/// Error types for tool call handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallHandlerError {
    /// The requested tool is not available
    UnknownTool(
        /// The name of the unknown tool
        String,
    ),
    /// Invalid arguments were provided to the tool
    InvalidArguments {
        /// The name of the tool
        tool_name: String,
        /// Description of the argument error
        message: String,
    },
    /// Tool execution failed
    ExecutionFailed {
        /// The name of the tool that failed
        tool_name: String,
        /// Description of the failure
        message: String,
    },
    /// Tool execution timed out
    Timeout {
        /// The name of the tool that timed out
        tool_name: String,
        /// The timeout duration in milliseconds
        timeout_ms: u64,
    },
    /// Tool call was cancelled
    Cancelled,
    /// Custom error for tool-specific failures
    Custom(
        /// Custom error message
        String,
    ),
}

impl std::fmt::Display for ToolCallHandlerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownTool(name) => write!(f, "Unknown tool: {name}"),
            Self::InvalidArguments { tool_name, message } => {
                write!(f, "Invalid arguments for tool '{tool_name}': {message}")
            }
            Self::ExecutionFailed { tool_name, message } => {
                write!(f, "Tool '{tool_name}' execution failed: {message}")
            }
            Self::Timeout { tool_name, timeout_ms } => {
                write!(f, "Tool '{tool_name}' timed out after {timeout_ms}ms")
            }
            Self::Cancelled => write!(f, "Tool call was cancelled"),
            Self::Custom(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ToolCallHandlerError {}

// =============================================================================
// TOOL CALL EXECUTOR
// =============================================================================

/// Executor for orchestrating multi-turn tool calling conversations.
///
/// The `ToolCallExecutor` manages the lifecycle of tool calling:
/// 1. Parses tool calls from model output
/// 2. Dispatches calls to the registered handler
/// 3. Formats results for the next model turn
/// 4. Tracks conversation state
///
/// # Example
///
/// ```rust,ignore
/// use realizar::tools::{ToolCallExecutor, ToolCallHandler};
/// use realizar::grammar::{ToolDefinition, ToolCallParser};
///
/// // Create executor with handler and tools
/// let executor = ToolCallExecutor::new(
///     Box::new(MyToolHandler),
///     vec![weather_tool, search_tool],
/// );
///
/// // Process model output
/// let model_output = "<tool_call>{\"name\": \"get_weather\", ...}</tool_call>";
/// let results = executor.process_output(model_output)?;
///
/// // Format results for next turn
/// let formatted = executor.format_results(&results);
/// ```
pub struct ToolCallExecutor {
    /// The handler that executes tool calls
    handler: Box<dyn ToolCallHandler>,
    /// Parser for extracting tool calls from model output (wrapped in Mutex for interior mutability)
    parser: std::sync::Mutex<crate::grammar::ToolCallParser>,
    /// Template for formatting tool results
    template: Box<dyn ToolPromptTemplate>,
    /// Available tool definitions
    tools: Vec<crate::grammar::ToolDefinition>,
    /// Maximum number of tool calls per turn
    max_calls_per_turn: usize,
    /// Whether parallel tool execution is allowed
    allow_parallel: bool,
}

impl Debug for ToolCallExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolCallExecutor")
            .field("handler", &self.handler)
            .field("template", &self.template.name())
            .field("tools", &self.tools.iter().map(|t| &t.name).collect::<Vec<_>>())
            .field("max_calls_per_turn", &self.max_calls_per_turn)
            .field("allow_parallel", &self.allow_parallel)
            .finish()
    }
}

impl ToolCallExecutor {
    /// Create a new tool call executor.
    ///
    /// # Arguments
    ///
    /// * `handler` - The handler that will execute tool calls
    /// * `tools` - Available tool definitions
    ///
    /// # Returns
    ///
    /// A new executor configured with Groq template by default
    pub fn new(
        handler: Box<dyn ToolCallHandler>,
        tools: Vec<crate::grammar::ToolDefinition>,
    ) -> Self {
        let parser = crate::grammar::ToolCallParser::new(tools.clone());
        Self {
            handler,
            parser: std::sync::Mutex::new(parser),
            template: Box::new(GroqToolTemplate::new()),
            tools,
            max_calls_per_turn: 10,
            allow_parallel: false,
        }
    }

    /// Set the prompt template to use.
    #[must_use]
    pub fn with_template(mut self, template: Box<dyn ToolPromptTemplate>) -> Self {
        self.template = template;
        self
    }

    /// Set the tool call format for parsing.
    #[must_use]
    pub fn with_format(mut self, format: crate::grammar::ToolCallFormat) -> Self {
        let new_parser = crate::grammar::ToolCallParser::new(self.tools.clone()).with_format(format);
        self.parser = std::sync::Mutex::new(new_parser);
        self
    }

    /// Set the maximum number of tool calls per turn.
    #[must_use]
    pub fn with_max_calls(mut self, max_calls: usize) -> Self {
        self.max_calls_per_turn = max_calls;
        self
    }

    /// Enable or disable parallel tool execution.
    #[must_use]
    pub fn with_parallel(mut self, allow: bool) -> Self {
        self.allow_parallel = allow;
        self
    }

    /// Get the tool definitions.
    pub fn tools(&self) -> &[crate::grammar::ToolDefinition] {
        &self.tools
    }

    /// Get the template being used.
    pub fn template(&self) -> &dyn ToolPromptTemplate {
        self.template.as_ref()
    }

    /// Render the system prompt introduction with tool definitions.
    pub fn render_system_intro(&self) -> String {
        self.template.render_system_intro(&self.tools)
    }

    /// Process model output and execute any tool calls.
    ///
    /// Parses tool calls from the output and executes them via the handler.
    ///
    /// # Arguments
    ///
    /// * `output` - Raw model output text
    ///
    /// # Returns
    ///
    /// A vector of tool results, or an error if parsing/execution fails
    pub fn process_output(
        &self,
        output: &str,
    ) -> Result<Vec<crate::grammar::ToolResult>, ToolExecutorError> {
        // Parse tool calls from output (acquire lock)
        let tool_calls = {
            let mut parser = self.parser.lock().expect("Parser lock poisoned");
            parser.parse(output)
        };

        if tool_calls.is_empty() {
            return Ok(vec![]);
        }

        // Check call limit
        if tool_calls.len() > self.max_calls_per_turn {
            return Err(ToolExecutorError::TooManyCalls {
                count: tool_calls.len(),
                max: self.max_calls_per_turn,
            });
        }

        // Execute each tool call
        let mut results = Vec::with_capacity(tool_calls.len());
        for tool_call in &tool_calls {
            match self.handler.handle_tool_call(tool_call) {
                Ok(result) => results.push(result),
                Err(e) => {
                    // Convert handler error to a failed ToolResult
                    results.push(crate::grammar::ToolResult::error(
                        tool_call.id.clone(),
                        e.to_string(),
                    ));
                }
            }
        }

        Ok(results)
    }

    /// Check if the model output contains any tool calls.
    pub fn has_tool_calls(&self, output: &str) -> bool {
        let mut parser = self.parser.lock().expect("Parser lock poisoned");
        !parser.parse(output).is_empty()
    }

    /// Format tool results for inclusion in the next model turn.
    ///
    /// # Arguments
    ///
    /// * `results` - Tool results to format
    ///
    /// # Returns
    ///
    /// Formatted string suitable for adding to the conversation
    pub fn format_results(&self, results: &[crate::grammar::ToolResult]) -> String {
        results
            .iter()
            .map(|r| self.template.render_tool_result(r))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Execute a single tool call.
    ///
    /// Useful when you've already parsed the tool call yourself.
    ///
    /// # Arguments
    ///
    /// * `tool_call` - The parsed tool call to execute
    ///
    /// # Returns
    ///
    /// The tool result
    pub fn execute_call(
        &self,
        tool_call: &crate::grammar::ToolCall,
    ) -> crate::grammar::ToolResult {
        match self.handler.handle_tool_call(tool_call) {
            Ok(result) => result,
            Err(e) => crate::grammar::ToolResult::error(tool_call.id.clone(), e.to_string()),
        }
    }
}

/// Error types for the tool executor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolExecutorError {
    /// Too many tool calls in a single turn
    TooManyCalls {
        /// Number of tool calls found
        count: usize,
        /// Maximum allowed tool calls
        max: usize,
    },
    /// Failed to parse tool calls from output
    ParseError(
        /// Description of the parse error
        String,
    ),
    /// All tool calls failed
    AllCallsFailed(
        /// List of error messages from each failed call
        Vec<String>,
    ),
}

impl std::fmt::Display for ToolExecutorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyCalls { count, max } => {
                write!(f, "Too many tool calls: {count} (max: {max})")
            }
            Self::ParseError(msg) => write!(f, "Failed to parse tool calls: {msg}"),
            Self::AllCallsFailed(errors) => {
                write!(f, "All tool calls failed: {}", errors.join("; "))
            }
        }
    }
}

impl std::error::Error for ToolExecutorError {}

// =============================================================================
// CONVENIENCE HANDLER IMPLEMENTATIONS
// =============================================================================

/// A function-based tool call handler.
///
/// Useful for simple cases where tools can be implemented as closures.
///
/// # Example
///
/// ```rust
/// use realizar::tools::FnToolHandler;
/// use realizar::grammar::{ToolCall, ToolResult};
///
/// let handler = FnToolHandler::new(|tool_call| {
///     match tool_call.name.as_str() {
///         "greet" => {
///             let name = serde_json::from_str::<serde_json::Value>(&tool_call.arguments)
///                 .ok()
///                 .and_then(|v| v.get("name").and_then(|n| n.as_str()).map(String::from))
///                 .unwrap_or_else(|| "World".to_string());
///             Ok(ToolResult::success(tool_call.id.clone(), format!("Hello, {name}!")))
///         }
///         _ => Ok(ToolResult::error(tool_call.id.clone(), "Unknown tool")),
///     }
/// });
/// ```
pub struct FnToolHandler<F>
where
    F: Fn(&crate::grammar::ToolCall) -> Result<crate::grammar::ToolResult, ToolCallHandlerError>
        + Send
        + Sync,
{
    handler: F,
    tools: Vec<String>,
}

impl<F> Debug for FnToolHandler<F>
where
    F: Fn(&crate::grammar::ToolCall) -> Result<crate::grammar::ToolResult, ToolCallHandlerError>
        + Send
        + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnToolHandler")
            .field("tools", &self.tools)
            .field("handler", &"<function>")
            .finish()
    }
}

impl<F> FnToolHandler<F>
where
    F: Fn(&crate::grammar::ToolCall) -> Result<crate::grammar::ToolResult, ToolCallHandlerError>
        + Send
        + Sync,
{
    /// Create a new function-based handler.
    pub fn new(handler: F) -> Self {
        Self {
            handler,
            tools: vec![],
        }
    }

    /// Set the list of available tools.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools = tools;
        self
    }
}

impl<F> ToolCallHandler for FnToolHandler<F>
where
    F: Fn(&crate::grammar::ToolCall) -> Result<crate::grammar::ToolResult, ToolCallHandlerError>
        + Send
        + Sync
        + 'static,
{
    fn handle_tool_call(
        &self,
        tool_call: &crate::grammar::ToolCall,
    ) -> Result<crate::grammar::ToolResult, ToolCallHandlerError> {
        (self.handler)(tool_call)
    }

    fn available_tools(&self) -> Vec<String> {
        self.tools.clone()
    }
}

/// A handler that dispatches to registered tool implementations.
///
/// Allows registering multiple tools with their implementations.
///
/// # Example
///
/// ```rust
/// use realizar::tools::{DispatchingToolHandler, ToolCallHandlerError};
/// use realizar::grammar::{ToolCall, ToolResult};
///
/// let mut handler = DispatchingToolHandler::new();
///
/// handler.register("get_time", |_call| {
///     Ok(ToolResult::success("1".to_string(), "12:00:00".to_string()))
/// });
///
/// handler.register("get_date", |_call| {
///     Ok(ToolResult::success("2".to_string(), "2025-01-13".to_string()))
/// });
/// ```
pub struct DispatchingToolHandler {
    handlers: std::collections::HashMap<
        String,
        Box<
            dyn Fn(&crate::grammar::ToolCall) -> Result<crate::grammar::ToolResult, ToolCallHandlerError>
                + Send
                + Sync,
        >,
    >,
}

impl Debug for DispatchingToolHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DispatchingToolHandler")
            .field("tools", &self.handlers.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl Default for DispatchingToolHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl DispatchingToolHandler {
    /// Create a new dispatching handler.
    pub fn new() -> Self {
        Self {
            handlers: std::collections::HashMap::new(),
        }
    }

    /// Register a tool implementation.
    ///
    /// # Arguments
    ///
    /// * `name` - The tool name
    /// * `handler` - The handler function
    pub fn register<F>(&mut self, name: &str, handler: F)
    where
        F: Fn(&crate::grammar::ToolCall) -> Result<crate::grammar::ToolResult, ToolCallHandlerError>
            + Send
            + Sync
            + 'static,
    {
        self.handlers.insert(name.to_string(), Box::new(handler));
    }
}

impl ToolCallHandler for DispatchingToolHandler {
    fn handle_tool_call(
        &self,
        tool_call: &crate::grammar::ToolCall,
    ) -> Result<crate::grammar::ToolResult, ToolCallHandlerError> {
        match self.handlers.get(&tool_call.name) {
            Some(handler) => handler(tool_call),
            None => Err(ToolCallHandlerError::UnknownTool(tool_call.name.clone())),
        }
    }

    fn available_tools(&self) -> Vec<String> {
        self.handlers.keys().cloned().collect()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::ToolParameter;

    fn create_test_tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new(
                "get_weather",
                "Get the current weather in a location",
                vec![
                    ToolParameter::required_string("location", "The city and state, e.g., San Francisco, CA"),
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

    // ==================== Groq Template Tests ====================

    #[test]
    fn test_groq_template_system_intro() {
        let template = GroqToolTemplate::new();
        let tools = create_test_tools();

        let system = template.render_system_intro(&tools);

        assert!(system.contains("function calling AI model"));
        assert!(system.contains("<tools>"));
        assert!(system.contains("</tools>"));
        assert!(system.contains("<tool_call>"));
        assert!(system.contains("</tool_call>"));
        assert!(system.contains("get_weather"));
        assert!(system.contains("search"));
    }

    #[test]
    fn test_groq_template_tool_call() {
        let template = GroqToolTemplate::new();
        let call = ToolCall::new("call_1", "get_weather", r#"{"location": "NYC"}"#);

        let rendered = template.render_tool_call(&call);

        assert!(rendered.contains("<tool_call>"));
        assert!(rendered.contains("</tool_call>"));
        assert!(rendered.contains("get_weather"));
        assert!(rendered.contains("call_1"));
    }

    #[test]
    fn test_groq_template_tool_result_success() {
        let template = GroqToolTemplate::new();
        let result = ToolResult::success("call_1", r#"{"temperature": 72}"#);

        let rendered = template.render_tool_result(&result);

        assert!(rendered.contains("<tool_response>"));
        assert!(rendered.contains("</tool_response>"));
        assert!(rendered.contains("result"));
        assert!(rendered.contains("call_1"));
    }

    #[test]
    fn test_groq_template_tool_result_error() {
        let template = GroqToolTemplate::new();
        let result = ToolResult::error("call_1", "API rate limited");

        let rendered = template.render_tool_result(&result);

        assert!(rendered.contains("<tool_response>"));
        assert!(rendered.contains("error"));
        assert!(rendered.contains("API rate limited"));
    }

    #[test]
    fn test_groq_template_stop_tokens() {
        let template = GroqToolTemplate::new();
        let stops = template.get_stop_tokens();

        assert!(stops.contains(&"</tool_call>".to_string()));
        assert!(stops.contains(&"<|eot_id|>".to_string()));
    }

    #[test]
    fn test_groq_template_detect_tool_call() {
        let template = GroqToolTemplate::new();

        assert!(template.detect_tool_call_start("<tool_call>"));
        assert!(template.detect_tool_call_start("Here is the result:\n<tool_call>"));
        assert!(!template.detect_tool_call_start("Hello world"));
        assert!(!template.detect_tool_call_start("{\"name\": \"test\"}"));
    }

    #[test]
    fn test_groq_template_name() {
        let template = GroqToolTemplate::new();
        assert_eq!(template.name(), "groq_tool");
    }

    // ==================== Llama 3 Instruct Template Tests ====================

    #[test]
    fn test_llama3_template_system_intro() {
        let template = Llama3InstructToolTemplate::new();
        let tools = create_test_tools();

        let system = template.render_system_intro(&tools);

        assert!(system.contains("Environment: ipython"));
        assert!(system.contains("def get_weather"));
        assert!(system.contains("def search"));
        assert!(system.contains("location: str"));
    }

    #[test]
    fn test_llama3_template_tool_call() {
        let template = Llama3InstructToolTemplate::new();
        let call = ToolCall::new("call_1", "get_weather", r#"{"location": "NYC"}"#);

        let rendered = template.render_tool_call(&call);

        assert!(rendered.contains("<|start_header_id|>ipython<|end_header_id|>"));
        assert!(rendered.contains("<|eot_id|>"));
        assert!(rendered.contains("get_weather"));
    }

    #[test]
    fn test_llama3_template_tool_result() {
        let template = Llama3InstructToolTemplate::new();
        let result = ToolResult::success("call_1", r#"{"temp": 72}"#);

        let rendered = template.render_tool_result(&result);

        assert!(rendered.contains("<|start_header_id|>ipython<|end_header_id|>"));
        assert!(rendered.contains(r#"{"temp": 72}"#));
    }

    #[test]
    fn test_llama3_template_stop_tokens() {
        let template = Llama3InstructToolTemplate::new();
        let stops = template.get_stop_tokens();

        assert!(stops.contains(&"<|eot_id|>".to_string()));
        assert!(stops.contains(&"<|python_tag|>".to_string()));
    }

    #[test]
    fn test_llama3_template_detect_tool_call() {
        let template = Llama3InstructToolTemplate::new();

        assert!(template.detect_tool_call_start(r#"{"name":"test"}"#));
        assert!(template.detect_tool_call_start(r#"{"name" :"test"}"#));
        assert!(!template.detect_tool_call_start("Hello world"));
        assert!(!template.detect_tool_call_start("<tool_call>"));
    }

    #[test]
    fn test_llama3_template_name() {
        let template = Llama3InstructToolTemplate::new();
        assert_eq!(template.name(), "llama3_instruct");
    }

    // ==================== OpenAI Template Tests ====================

    #[test]
    fn test_openai_template_system_intro() {
        let template = OpenAIToolTemplate::new();
        let tools = create_test_tools();

        let system = template.render_system_intro(&tools);

        assert!(system.contains("functions"));
        assert!(system.contains("get_weather"));
        assert!(system.contains("search"));
        assert!(system.contains("parameters"));
    }

    #[test]
    fn test_openai_template_tool_call() {
        let template = OpenAIToolTemplate::new();
        let call = ToolCall::new("call_1", "search", r#"{"query": "weather"}"#);

        let rendered = template.render_tool_call(&call);

        assert!(rendered.contains("call_1"));
        assert!(rendered.contains("search"));
        assert!(rendered.contains("arguments"));
    }

    #[test]
    fn test_openai_template_name() {
        let template = OpenAIToolTemplate::new();
        assert_eq!(template.name(), "openai");
    }

    // ==================== Factory Tests ====================

    #[test]
    fn test_create_template() {
        let groq = create_template(ToolTemplateType::Groq);
        assert_eq!(groq.name(), "groq_tool");

        let llama = create_template(ToolTemplateType::Llama3Instruct);
        assert_eq!(llama.name(), "llama3_instruct");

        let openai = create_template(ToolTemplateType::OpenAI);
        assert_eq!(openai.name(), "openai");
    }

    #[test]
    fn test_detect_template_type() {
        assert_eq!(
            detect_template_type("Llama-3-Groq-8B-Tool-Use"),
            ToolTemplateType::Groq
        );
        assert_eq!(
            detect_template_type("Meta-Llama-3-8B-Instruct"),
            ToolTemplateType::Llama3Instruct
        );
        assert_eq!(
            detect_template_type("gpt-4-turbo"),
            ToolTemplateType::OpenAI
        );
        assert_eq!(
            detect_template_type("unknown-model"),
            ToolTemplateType::Groq // Default
        );
    }

    #[test]
    fn test_template_type_default() {
        assert_eq!(ToolTemplateType::default(), ToolTemplateType::Groq);
    }

    // ==================== ToolCallHandler Tests ====================

    #[test]
    fn test_dispatching_handler_register_and_call() {
        use crate::grammar::{ToolCall, ToolResult};

        let mut handler = DispatchingToolHandler::new();

        handler.register("greet", |call| {
            Ok(ToolResult::success(call.id.clone(), "Hello!".to_string()))
        });

        let call = ToolCall::new("call_1", "greet", r#"{"name": "Alice"}"#);
        let result = handler.handle_tool_call(&call);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.content, "Hello!");
        assert!(result.success);
    }

    #[test]
    fn test_dispatching_handler_unknown_tool() {
        let handler = DispatchingToolHandler::new();

        let call = crate::grammar::ToolCall::new("call_1", "unknown", "{}");
        let result = handler.handle_tool_call(&call);

        assert!(result.is_err());
        match result {
            Err(ToolCallHandlerError::UnknownTool(name)) => assert_eq!(name, "unknown"),
            _ => panic!("Expected UnknownTool error"),
        }
    }

    #[test]
    fn test_dispatching_handler_available_tools() {
        use crate::grammar::ToolResult;

        let mut handler = DispatchingToolHandler::new();
        handler.register("tool_a", |call| {
            Ok(ToolResult::success(call.id.clone(), "a".to_string()))
        });
        handler.register("tool_b", |call| {
            Ok(ToolResult::success(call.id.clone(), "b".to_string()))
        });

        let tools = handler.available_tools();
        assert_eq!(tools.len(), 2);
        assert!(tools.contains(&"tool_a".to_string()));
        assert!(tools.contains(&"tool_b".to_string()));
    }

    #[test]
    fn test_dispatching_handler_has_tool() {
        use crate::grammar::ToolResult;

        let mut handler = DispatchingToolHandler::new();
        handler.register("my_tool", |call| {
            Ok(ToolResult::success(call.id.clone(), "result".to_string()))
        });

        assert!(handler.has_tool("my_tool"));
        assert!(!handler.has_tool("other_tool"));
    }

    #[test]
    fn test_fn_handler_simple() {
        use crate::grammar::{ToolCall, ToolResult};

        let handler = FnToolHandler::new(|call| {
            Ok(ToolResult::success(call.id.clone(), format!("Called {}", call.name)))
        })
        .with_tools(vec!["test".to_string()]);

        let call = ToolCall::new("1", "test", "{}");
        let result = handler.handle_tool_call(&call).unwrap();

        assert_eq!(result.content, "Called test");
        assert!(handler.has_tool("test"));
    }

    // ==================== ToolCallExecutor Tests ====================

    #[test]
    fn test_executor_creation() {
        let handler = DispatchingToolHandler::new();
        let tools = create_test_tools();

        let executor = ToolCallExecutor::new(Box::new(handler), tools.clone());

        assert_eq!(executor.tools().len(), 2);
        assert_eq!(executor.template().name(), "groq_tool");
    }

    #[test]
    fn test_executor_render_system_intro() {
        let handler = DispatchingToolHandler::new();
        let tools = create_test_tools();

        let executor = ToolCallExecutor::new(Box::new(handler), tools);
        let intro = executor.render_system_intro();

        assert!(intro.contains("<tools>"));
        assert!(intro.contains("get_weather"));
        assert!(intro.contains("search"));
    }

    #[test]
    fn test_executor_process_no_tool_calls() {
        let handler = DispatchingToolHandler::new();
        let tools = create_test_tools();

        let executor = ToolCallExecutor::new(Box::new(handler), tools);
        let results = executor.process_output("Just a normal message, no tool calls here.").unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_executor_process_with_tool_call() {
        use crate::grammar::ToolResult;

        let mut handler = DispatchingToolHandler::new();
        handler.register("get_weather", |call| {
            Ok(ToolResult::success(call.id.clone(), r#"{"temp": 72}"#.to_string()))
        });

        let tools = create_test_tools();
        let executor = ToolCallExecutor::new(Box::new(handler), tools);

        // Simulate model output with tool call (OpenAI format)
        let output = r#"{"name": "get_weather", "arguments": {"location": "NYC", "unit": "fahrenheit"}}"#;
        let results = executor.process_output(output).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].success);
        assert!(results[0].content.contains("72"));
    }

    #[test]
    fn test_executor_process_unknown_tool_in_parser_is_ignored() {
        // The parser only parses tool calls for tools that are defined
        // Unknown tools in the output are simply ignored at the parsing stage
        let handler = DispatchingToolHandler::new();
        let tools = create_test_tools();
        let executor = ToolCallExecutor::new(Box::new(handler), tools);

        // Simulate model output with unknown tool - parser ignores it
        let output = r#"{"name": "unknown_tool", "arguments": {}}"#;
        let results = executor.process_output(output).unwrap();

        // Unknown tools are ignored by the parser, so no results
        assert!(results.is_empty());
    }

    #[test]
    fn test_executor_handler_unknown_tool() {
        use crate::grammar::ToolResult;

        // When the handler is called with an unknown tool (directly, not via parser),
        // it should return an error
        let mut handler = DispatchingToolHandler::new();
        handler.register("known_tool", |call| {
            Ok(ToolResult::success(call.id.clone(), "ok".to_string()))
        });

        let call = crate::grammar::ToolCall::new("1", "unknown_tool", "{}");
        let result = handler.handle_tool_call(&call);

        assert!(result.is_err());
        match result {
            Err(ToolCallHandlerError::UnknownTool(name)) => {
                assert_eq!(name, "unknown_tool");
            }
            _ => panic!("Expected UnknownTool error"),
        }
    }

    #[test]
    fn test_executor_too_many_calls() {
        let handler = DispatchingToolHandler::new();
        let tools = create_test_tools();
        let executor = ToolCallExecutor::new(Box::new(handler), tools)
            .with_max_calls(1);

        // Multiple tool calls in output
        let output = r#"{"name": "get_weather", "arguments": {"location": "NYC"}}
{"name": "search", "arguments": {"query": "weather"}}"#;
        let result = executor.process_output(output);

        assert!(result.is_err());
        match result {
            Err(ToolExecutorError::TooManyCalls { count: 2, max: 1 }) => {}
            _ => panic!("Expected TooManyCalls error"),
        }
    }

    #[test]
    fn test_executor_format_results() {
        use crate::grammar::ToolResult;

        let handler = DispatchingToolHandler::new();
        let tools = create_test_tools();
        let executor = ToolCallExecutor::new(Box::new(handler), tools);

        let results = vec![
            ToolResult::success("call_1", r#"{"temp": 72}"#),
            ToolResult::error("call_2", "API error"),
        ];

        let formatted = executor.format_results(&results);

        // Groq format uses <tool_response>
        assert!(formatted.contains("tool_response"));
        assert!(formatted.contains("call_1"));
        assert!(formatted.contains("call_2"));
    }

    #[test]
    fn test_executor_has_tool_calls() {
        let handler = DispatchingToolHandler::new();
        let tools = create_test_tools();
        let executor = ToolCallExecutor::new(Box::new(handler), tools);

        assert!(executor.has_tool_calls(r#"{"name": "get_weather", "arguments": {}}"#));
        assert!(!executor.has_tool_calls("Just a regular message"));
    }

    #[test]
    fn test_executor_execute_call() {
        use crate::grammar::{ToolCall, ToolResult};

        let mut handler = DispatchingToolHandler::new();
        handler.register("greet", |call| {
            Ok(ToolResult::success(call.id.clone(), "Hello!".to_string()))
        });

        let tools = create_test_tools();
        let executor = ToolCallExecutor::new(Box::new(handler), tools);

        let call = ToolCall::new("call_1", "greet", "{}");
        let result = executor.execute_call(&call);

        assert!(result.success);
        assert_eq!(result.content, "Hello!");
    }

    // ==================== Error Type Tests ====================

    #[test]
    fn test_handler_error_display() {
        let err = ToolCallHandlerError::UnknownTool("foo".to_string());
        assert_eq!(err.to_string(), "Unknown tool: foo");

        let err = ToolCallHandlerError::InvalidArguments {
            tool_name: "bar".to_string(),
            message: "missing required field".to_string(),
        };
        assert!(err.to_string().contains("bar"));
        assert!(err.to_string().contains("missing required field"));

        let err = ToolCallHandlerError::Timeout {
            tool_name: "slow".to_string(),
            timeout_ms: 5000,
        };
        assert!(err.to_string().contains("5000"));
    }

    #[test]
    fn test_executor_error_display() {
        let err = ToolExecutorError::TooManyCalls { count: 5, max: 3 };
        assert!(err.to_string().contains("5"));
        assert!(err.to_string().contains("3"));

        let err = ToolExecutorError::ParseError("invalid JSON".to_string());
        assert!(err.to_string().contains("invalid JSON"));

        let err = ToolExecutorError::AllCallsFailed(vec!["error 1".to_string(), "error 2".to_string()]);
        assert!(err.to_string().contains("error 1"));
    }
}
