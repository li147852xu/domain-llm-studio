# Inference Backend Benchmark

Greedy generation, single-request per call, RTX 5090 (32 GB), bf16.

| Model | Backend | Samples | Total(s) | P50(ms) | P95(ms) | P99(ms) | Tokens/s | PeakMem(GB) |
|---|---|---|---|---|---|---|---|---|
| 1.5b | transformers | 40 | 90.8 | 1762 | 5984 | 7160 | 69.5 | 3.53 |
| 1.5b | vllm | 40 | 19.6 | 372 | 1442 | 1504 | 317.4 | 30.87 |
| 7b | transformers | 40 | 85.2 | 1870 | 6631 | 6696 | 69.0 | 14.86 |
| 7b | vllm | 40 | 63.1 | 1281 | 4852 | 4863 | 103.9 | 30.43 |

## Speedup vs transformers

| Model | Throughput speedup | P95 latency reduction |
|---|---|---|
| 1.5b | 4.57x | +75.9% |
| 7b | 1.51x | +26.8% |
