# Implementation Plan: Tool Calling, JSON Enforcement, and Embeddings for Realizar

## Executive Summary
This document outlines the technical roadmap for upgrading `realizar` from a basic chat inference engine to a full-featured Agentic AI backend. The primary goals are:
1.  **Tool Calling & JSON Enforcement:** Enabling deterministic function execution, prioritizing Llama 3 architecture.
2.  **Local Embeddings:** Integrating HuggingFace `candle` or `ort` (ONNX Runtime) to serve embeddings.
3.  **Dual Exposure:** Exposing these features via both the OpenAI-compatible REST API and a direct Rust library API for embedding within the Planet application.

---

## Part 1: Architecture Overview

### 1.1 The "Sandwich" Strategy for Tool Calling
To support tool calling across different models (starting with Llama 3) while ensuring reliability, we will implement a two-layer approach:

*   **Top Layer (Prompt Abstraction):** A `ToolPromptTemplate` trait that translates generic tool definitions into model-specific prompt formats (e.g., Llama 3's `<|start_header_id|>ipython...`).
*   **Bottom Layer (Grammar Enforcement):** A `LogitProcessor` that hooks into the sampling loop to mathematically enforce valid JSON output when a tool call is detected.

### 1.2 The Embedding Engine
We will introduce a separate `EmbeddingEngine` struct distinct from the text generation `ModelEngine`. This allows:
*   Running a small, efficient embedding model (e.g., `all-MiniLM-L6-v2` or `nomic-embed-text`) alongside the main LLM.
*   Batch processing capability.

### 1.3 Library vs. API
The core logic will reside in `src/lib.rs` (public API).
*   **REST Server (`src/main.rs` + `src/api.rs`):** Wraps the library components in Axum handlers.
*   **Planet Integration:** Planet imports the library structs directly, managing concurrency via `prometheus-parking-lot`.

---

## Part 2: Detailed Implementation Steps

### Phase 1: Grammar-Constrained Sampling (The "Enforcer")

**Objective:** Prevent the model from generating invalid JSON.

**File:** `src/sampling.rs` (create/modify)

1.  **Add Dependencies:**
    *   `regex-automata` (for regex-guided constraints)
    *   `serde_json` (if not present)

2.  **Define `LogitProcessor` Trait:**
    ```rust
    pub trait LogitProcessor {
        fn process(&mut self, input_ids: &[u32], logits: &mut [f32]);
    }
    ```

3.  **Implement `JsonGrammar`:**
    *   Create a struct `JsonGrammar` that implements `LogitProcessor`.
    *   Use a state machine (or simplified regex logic initially) to mask logits.
    *   *Logic:* If the current token sequence violates JSON syntax, set that token's logit to `-Infinity`.

4.  **Integrate into Inference Loop (`src/model_loader.rs`):**
    *   Modify the `generate` function to accept an optional `LogitProcessor`.
    *   Apply the processor to logits *before* sampling.

### Phase 2: Tool Prompt Abstraction (The "Translator")

**Objective:** Translate `tools` definitions into Llama 3 specific special tokens.

**File:** `src/chat_template.rs` / `src/tools.rs`

1.  **Define Data Structures (`src/tools.rs`):**
    ```rust
    #[derive(Serialize, Deserialize, Clone)]
    pub struct ToolDefinition {
        pub name: String,
        pub description: String,
        pub parameters: serde_json::Value,
    }
    ```

2.  **Create `ToolPromptTemplate` Trait:**
    ```rust
    pub trait ToolPromptTemplate {
        fn render_system_intro(&self, tools: &[ToolDefinition]) -> String;
        fn render_tool_call(&self, tool_call: &ToolCall) -> String;
        fn get_stop_tokens(&self) -> Vec<String>;
    }
    ```

3.  **Implement `Llama3ToolStrategy`:**
    *   **System Prompt:** Inject "Environment: ipython" and JSON definitions.
    *   **Prompt Formatting:** Use `<|start_header_id|>ipython<|end_header_id|>` tags.
    *   **Detection:** If the user request includes `tools`, switch the internal template renderer to this strategy.

### Phase 3: The Hybrid Sampler (Llama 3 Specialization)

**Objective:** Allow Llama 3 to choose between chatting (Text) and acting (JSON) dynamically.

**File:** `src/sampling.rs`

1.  **Implement `HybridSampler`:**
    *   **State:** `is_json_mode: bool`, `buffer: String`.
    *   **Logic:**
        *   Let the model generate 3-5 tokens.
        *   If tokens match a tool call pattern (e.g., `{"name":` or specific Llama 3 tool tokens), enable `JsonGrammar`.
        *   If tokens look like conversational text, disable grammar.

### Phase 4: Embedding Engine Implementation

**Objective:** Add support for BERT/Nomic style embedding models.

**File:** `src/embeddings.rs` (create)

1.  **Dependencies:** Add `candle-nn`, `candle-transformers` (specifically for Bert/Jina support).

2.  **Create `EmbeddingModel` Struct:**
    ```rust
    pub struct EmbeddingModel {
        model: BertModel, // Or generic trait object
        tokenizer: Tokenizer,
    }
    ```

3.  **Implement `embed` function:**
    *   Input: `Vec<String>` (Batch of texts).
    *   Output: `Vec<Vec<f32>>` (Batch of vectors).
    *   **Pooling:** Implement "Mean Pooling" (average of all token embeddings) or "CLS Pooling" depending on the model config.
    *   **Normalization:** specific to model (usually L2 norm).

### Phase 5: Public API & REST Integration

**Objective:** Expose new capabilities to Planet and HTTP clients.

**File:** `src/lib.rs`

1.  **Expose `ModelEngine` (Text Generation):**
    *   Ensure `ModelEngine` is thread-safe (`Arc<Mutex<...>>` or channel-based actor).
    *   Add `generate_with_tools` method.

2.  **Expose `EmbeddingEngine`:**
    *   Add public method `embed(text: Vec<String>) -> Result<Vec<Vec<f32>>>`.

**File:** `src/api.rs` (REST Layer)

1.  **Update `/v1/chat/completions`:**
    *   Deserialize `tools` and `tool_choice` from request.
    *   Pass to `ModelEngine`.
    *   If tool call occurs, format response as `tool_calls` array (OpenAI format).

2.  **Implement `/v1/embeddings`:**
    *   Accept input text.
    *   Call `EmbeddingEngine::embed`.
    *   Return standard OpenAI embedding response JSON.

---

## Part 3: Configuration & Testing Strategy

### 3.1 Configuration (`config.toml`)
Add sections for managing multiple model types:

```toml
[models.llm]
path = "models/Meta-Llama-3-8B-Instruct"
type = "llama-3"

[models.embedding]
path = "models/all-MiniLM-L6-v2"
type = "bert"
```

### 3.2 Testing Plan
1.  **Unit Tests (`cargo test`):**
    *   Test `JsonGrammar` against invalid token sequences.
    *   Test `Llama3ToolStrategy` string formatting.
2.  **Integration Tests:**
    *   Load `all-MiniLM-L6-v2` and verify vector dimensions.
    *   Run a mock tool call: Input "What is the weather in Paris?" -> Expect JSON `{"function": "get_weather", "args": {"city": "Paris"}}`.

## Part 4: Developer Notes for Cursor/Agents

*   **Existing Codebase:** Be careful not to break `src/model_loader.rs` existing logic. Wrap new functionality in `Option<>` or Traits to maintain backward compatibility.
*   **Llama 3 Tokens:** Always verify specific tokenizer configs. Llama 3 uses `128000` vocab size and specific control tokens. Do not hardcode IDs; look them up in the tokenizer.
*   **Candle:** Use `candle-core` and `candle-transformers` for the embedding implementation. Do not pull in `tch` (LibTorch) unless absolutely necessary to keep the binary size manageable.
