# Model Download Guide

This guide covers downloading and configuring models for Realizar's tool calling and embedding capabilities.

## Default Tool Calling Model: Llama-3-Groq-8B-Tool-Use

The default model for tool calling is **Llama-3-Groq-8B-Tool-Use** - the best-performing open-source 8B model for function calling with an 89.06% score on the Berkeley Function Calling Leaderboard (BFCL).

### Model Information

| Property | Value |
|----------|-------|
| **Original Model** | [Groq/Llama-3-Groq-8B-Tool-Use](https://huggingface.co/Groq/Llama-3-Groq-8B-Tool-Use) |
| **GGUF Version** | [bartowski/Llama-3-Groq-8B-Tool-Use-GGUF](https://huggingface.co/bartowski/Llama-3-Groq-8B-Tool-Use-GGUF) |
| **Base Model** | Meta-Llama-3-8B |
| **Training** | Full fine-tuning + Direct Preference Optimization (DPO) |
| **BFCL Score** | 89.06% (best among open-source 8B models) |
| **License** | Meta Llama 3 Community License |

### Quick Start

#### 1. Install HuggingFace CLI

```bash
pip install -U "huggingface_hub[cli]"
```

#### 2. Download the Model

**Recommended (Q4_K_M - balanced quality/size):**

```bash
mkdir -p models
huggingface-cli download bartowski/Llama-3-Groq-8B-Tool-Use-GGUF \
    --include "Llama-3-Groq-8B-Tool-Use-Q4_K_M.gguf" \
    --local-dir ./models/
```

### Available Quantizations

| Quantization | Size | Quality | Recommended For |
|--------------|------|---------|-----------------|
| Q2_K | ~3.2 GB | Lowest | Memory-constrained environments |
| Q3_K_S | ~3.7 GB | Low | Testing only |
| Q3_K_M | ~4.0 GB | Low-Medium | Low-memory systems |
| Q3_K_L | ~4.3 GB | Medium | Balance with limited RAM |
| Q4_0 | ~4.7 GB | Medium | Legacy support |
| **Q4_K_M** | **~4.7 GB** | **Good** | **Recommended default** |
| Q4_K_S | ~4.7 GB | Good | Slightly smaller Q4 |
| Q5_K_M | ~5.4 GB | High | Better quality, more RAM |
| Q5_K_S | ~5.3 GB | High | Slightly smaller Q5 |
| Q6_K | ~6.6 GB | Very High | Quality-focused |
| Q8_0 | ~8.5 GB | Near-Original | Maximum quality |

### Sampling Configuration

The Groq model is sensitive to sampling parameters. Use these recommended settings:

```toml
# config.toml
[models.llm]
path = "models/Llama-3-Groq-8B-Tool-Use-Q4_K_M.gguf"
type = "groq-tool"

[models.llm.sampling]
temperature = 0.5
top_p = 0.65
```

**Important**: Start with `temperature=0.5, top_p=0.65` and adjust as needed.

### Tool Call Format

The Groq model uses XML-wrapped JSON for tool calls:

**System Prompt Format:**
```
You are a function calling AI model. You are provided with function signatures within XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. For each function call return a json object with function name and arguments within XML tags as follows:

<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Here are the available tools:
<tools>
{tool definitions as JSON}
</tools>
```

**Tool Call Response:**
```
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "San Francisco", "unit": "celsius"}}
</tool_call>
```

**Tool Result Format:**
```
<tool_response>
{"id": "call_deok", "result": {"temperature": "72", "unit": "celsius"}}
</tool_response>
```

---

## Alternative: Llama 3 Instruct (ipython format)

For models using the standard Llama 3 Instruct tool format:

```bash
huggingface-cli download bartowski/Meta-Llama-3-8B-Instruct-GGUF \
    --include "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf" \
    --local-dir ./models/
```

Configuration:
```toml
[models.llm]
path = "models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
type = "llama-3"
```

---

## Embedding Models

For the `/v1/embeddings` endpoint, download one of these embedding models:

### Option 1: all-MiniLM-L6-v2 (Recommended - Fast)

384 dimensions, optimized for speed.

```bash
# Clone the model (includes tokenizer and weights)
git lfs install
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 models/all-MiniLM-L6-v2
```

Configuration:
```toml
[models.embedding]
path = "models/all-MiniLM-L6-v2"
type = "bert"
pooling = "mean"
normalize = true
```

### Option 2: nomic-embed-text-v1.5 (High Quality)

768 dimensions, state-of-the-art quality.

```bash
git lfs install
git clone https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 models/nomic-embed-text-v1.5
```

Configuration:
```toml
[models.embedding]
path = "models/nomic-embed-text-v1.5"
type = "nomic"
pooling = "mean"
normalize = true
```

### Option 3: bge-small-en-v1.5 (Compact)

384 dimensions, good quality/size balance.

```bash
git lfs install
git clone https://huggingface.co/BAAI/bge-small-en-v1.5 models/bge-small-en-v1.5
```

Configuration:
```toml
[models.embedding]
path = "models/bge-small-en-v1.5"
type = "bert"
pooling = "cls"
normalize = true
```

---

## Full Configuration Example

```toml
# config.toml - Complete configuration with tool calling and embeddings

[models.llm]
# Llama-3-Groq-8B-Tool-Use: Best open-source 8B for function calling
path = "models/Llama-3-Groq-8B-Tool-Use-Q4_K_M.gguf"
type = "groq-tool"

[models.llm.sampling]
temperature = 0.5
top_p = 0.65
max_tokens = 2048

[models.embedding]
# all-MiniLM-L6-v2: Fast, 384-dimensional embeddings
path = "models/all-MiniLM-L6-v2"
type = "bert"
pooling = "mean"
normalize = true

[server]
host = "0.0.0.0"
port = 8080
```

---

## Troubleshooting

### Model Not Found

Ensure the model file exists at the specified path:

```bash
ls -la models/
```

### Out of Memory

Try a smaller quantization:
- Use Q4_K_M instead of Q8_0
- Use Q3_K_M for very limited RAM
- Close other applications

### Slow Inference

- Enable GPU acceleration with `--gpu` flag
- Use smaller quantization (Q4 instead of Q8)
- Reduce `max_tokens` in config

### Tool Calls Not Working

1. Verify model type is set correctly (`groq-tool` for Groq, `llama-3` for Llama 3 Instruct)
2. Check sampling parameters (temperature=0.5, top_p=0.65 for Groq)
3. Ensure tools are properly defined in the API request

---

## References

- [Groq Model Card](https://huggingface.co/Groq/Llama-3-Groq-8B-Tool-Use)
- [bartowski GGUF Quantizations](https://huggingface.co/bartowski/Llama-3-Groq-8B-Tool-Use-GGUF)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [Meta Llama 3 License](https://llama.meta.com/llama3/license/)
