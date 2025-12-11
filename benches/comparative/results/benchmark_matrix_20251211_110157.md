# Benchmark Matrix Results

**Methodology:** CV-based stopping (Hoefler & Belli SC'15)

| Runtime | Backend | p50 Latency | p99 Latency | Throughput | Cold Start |
|---------|---------|-------------|-------------|------------|------------|
| llama-cpp | cpu | - | - | - | - |
| **llama-cpp** | gpu | 161.6ms | 208.3ms | 256.3 tok/s | 162ms |
| **ollama** | gpu | 6.2ms | 6.5ms | 0.0 tok/s | 6ms |
