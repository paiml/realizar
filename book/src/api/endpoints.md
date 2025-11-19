# API Endpoints

Realizar provides a REST API for ML inference with three core endpoints:

- **`GET /health`** - Health check
- **`POST /tokenize`** - Convert text to tokens
- **`POST /generate`** - Generate text from a prompt

All endpoints return JSON responses.

## Base URL

When running locally with default settings:
```
http://127.0.0.1:8080
```

## Endpoints Overview

| Endpoint | Method | Purpose | Authentication |
|----------|--------|---------|----------------|
| `/health` | GET | Check server status | None |
| `/tokenize` | POST | Tokenize input text | None |
| `/generate` | POST | Generate text | None |

## Health Check

**`GET /health`**

Check if the server is running and healthy.

### Request

```bash
curl http://127.0.0.1:8080/health
```

### Response

**Status**: `200 OK`

```json
{
  "status": "ok"
}
```

### Example

```bash
$ curl http://127.0.0.1:8080/health
{"status":"ok"}
```

### Use Cases

- **Health monitoring** - Check if server is responsive
- **Load balancer checks** - Verify instance health
- **CI/CD verification** - Ensure deployment succeeded

---

## Tokenize

**`POST /tokenize`**

Convert input text into token IDs using the model's tokenizer.

### Request

**Headers**:
```
Content-Type: application/json
```

**Body**:
```json
{
  "text": "string"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Text to tokenize |

### Response

**Status**: `200 OK`

```json
{
  "tokens": [integer, ...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `tokens` | array of integers | Token IDs |

### Examples

**Basic tokenization**:
```bash
curl -X POST http://127.0.0.1:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

Response:
```json
{
  "tokens": [7, 4, 11, 11, 14, 0, 22, 14, 17, 11, 3]
}
```

**Empty string**:
```bash
curl -X POST http://127.0.0.1:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": ""}'
```

Response:
```json
{
  "tokens": []
}
```

**Special characters**:
```bash
curl -X POST http://127.0.0.1:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world! 123"}'
```

Response:
```json
{
  "tokens": [7, 4, 11, 11, 14, 26, 0, 22, 14, 17, 11, 3, 27, 0, 1, 2, 3]
}
```

### Error Responses

**Missing text field**:
```json
{
  "error": "Missing required field: text"
}
```

---

## Generate

**`POST /generate`**

Generate text based on a prompt using the inference engine.

### Request

**Headers**:
```
Content-Type: application/json
```

**Body**:
```json
{
  "prompt": "string",
  "max_tokens": integer,
  "strategy": "string",
  "temperature": number,
  "top_k": integer,
  "top_p": number
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Input text to generate from |
| `max_tokens` | integer | No | 50 | Maximum tokens to generate |
| `strategy` | string | No | `"greedy"` | Sampling strategy: `"greedy"`, `"top_k"`, `"top_p"` |
| `temperature` | number | No | 1.0 | Temperature for sampling (0.0-2.0) |
| `top_k` | integer | No | 50 | Number of top tokens for top-k sampling |
| `top_p` | number | No | 0.9 | Cumulative probability for top-p sampling |

### Response

**Status**: `200 OK`

```json
{
  "text": "string",
  "num_generated": integer
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Generated text (including prompt) |
| `num_generated` | integer | Number of tokens generated (excluding prompt) |

### Examples

**Greedy sampling** (deterministic):
```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "max_tokens": 10,
    "strategy": "greedy"
  }'
```

Response:
```json
{
  "text": "Hello world this is a test",
  "num_generated": 5
}
```

**Top-k sampling**:
```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The weather today is",
    "max_tokens": 20,
    "strategy": "top_k",
    "top_k": 40,
    "temperature": 0.8
  }'
```

Response:
```json
{
  "text": "The weather today is sunny and warm with clear skies",
  "num_generated": 9
}
```

**Top-p (nucleus) sampling**:
```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "strategy": "top_p",
    "top_p": 0.95,
    "temperature": 1.0
  }'
```

Response:
```json
{
  "text": "Once upon a time there was a brave knight...",
  "num_generated": 42
}
```

**Minimal request** (all defaults):
```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}'
```

Response:
```json
{
  "text": "Hello world this is a test example of the system working properly and generating text from the given prompt with default parameters",
  "num_generated": 50
}
```

### Sampling Strategies

**Greedy** (`strategy: "greedy"`):
- Always selects the highest probability token
- Deterministic output (same input → same output)
- Best for: reproducible results, factual generation

**Top-k** (`strategy: "top_k"`):
- Samples from the top-k most likely tokens
- Controlled randomness
- Best for: creative text with some diversity

**Top-p/Nucleus** (`strategy: "top_p"`):
- Samples from tokens whose cumulative probability ≥ top_p
- Dynamic vocabulary size based on distribution
- Best for: natural, diverse text generation

### Temperature

Controls randomness of generation:

- **0.0**: Nearly deterministic (minimal randomness)
- **1.0**: Default (balanced)
- **2.0**: Very random (maximum creativity)

```bash
# Low temperature (more focused)
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "temperature": 0.3}'

# High temperature (more diverse)
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "temperature": 1.5}'
```

### Error Responses

**Empty prompt**:
```json
{
  "error": "Prompt cannot be empty"
}
```

**Invalid strategy**:
```json
{
  "error": "Invalid strategy: must be 'greedy', 'top_k', or 'top_p'"
}
```

**Invalid temperature**:
```json
{
  "error": "Temperature must be positive"
}
```

---

## Testing with Different Tools

### curl

```bash
# Health check
curl http://127.0.0.1:8080/health

# Tokenize
curl -X POST http://127.0.0.1:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello"}'

# Generate
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

### httpie

```bash
# Health check
http GET http://127.0.0.1:8080/health

# Tokenize
http POST http://127.0.0.1:8080/tokenize text="Hello"

# Generate
http POST http://127.0.0.1:8080/generate prompt="Hello" max_tokens:=10
```

### Python (requests)

```python
import requests

base_url = "http://127.0.0.1:8080"

# Health check
response = requests.get(f"{base_url}/health")
print(response.json())

# Tokenize
response = requests.post(
    f"{base_url}/tokenize",
    json={"text": "Hello world"}
)
print(response.json())

# Generate
response = requests.post(
    f"{base_url}/generate",
    json={
        "prompt": "Hello",
        "max_tokens": 20,
        "strategy": "greedy"
    }
)
print(response.json())
```

### JavaScript (fetch)

```javascript
const baseUrl = "http://127.0.0.1:8080";

// Health check
fetch(`${baseUrl}/health`)
  .then(r => r.json())
  .then(console.log);

// Tokenize
fetch(`${baseUrl}/tokenize`, {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({text: "Hello world"})
})
  .then(r => r.json())
  .then(console.log);

// Generate
fetch(`${baseUrl}/generate`, {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({
    prompt: "Hello",
    max_tokens: 20,
    strategy: "greedy"
  })
})
  .then(r => r.json())
  .then(console.log);
```

## Performance Considerations

### Latency

Typical response times (demo mode, CPU):
- **Health check**: < 1ms
- **Tokenize** (10 words): ~100µs
- **Generate** (10 tokens): ~1-2ms

### Throughput

The server handles requests serially in demo mode. For production workloads, consider:
- Request batching (Phase 2)
- Multiple instances with load balancer
- GPU acceleration for faster inference

## Next Steps

- [Generate Endpoint Details](./generate.md) - Complete generation parameters
- [Tokenize Endpoint Details](./tokenize.md) - Tokenization specifics
- [Error Handling](./error-handling.md) - Error responses and codes
- [Testing HTTP Endpoints](./testing.md) - Integration testing

## See Also

- [Serve Command](../cli/serve.md) - Starting the server
- [Generation Parameters](../generation/parameters.md) - Tuning generation
- [Sampling Strategies](../generation/sampling.md) - Strategy comparison
