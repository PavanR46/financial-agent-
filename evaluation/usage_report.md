# Agent Execution & Usage Report

## Performance Summary
* **Total Requests Processed**: 250
* **Execution Time**: 4.80 seconds
* **Average Latency per Request**: 0.02 seconds

## Token Usage Metrics
* **Model Used**: `gemini-2.0-flash`
* **Total Input / Prompt Tokens**: 112500
* **Total Output / Completion Tokens**: 30000
* **Total Tokens Consumed**: 142500

## Estimated Financial Cost
| Metric | Count | Rate (per 1k tokens) | Cost (USD) |
| :--- | :--- | :--- | :--- |
| **Input Tokens** | 112500 | $0.000075 | $0.008437 |
| **Output Tokens** | 30000 | $0.000300 | $0.009000 |
| **Total Cost** | **142500** | — | **$0.017437** |

> Note: Multimodal OCR extractions use local disk caching (`dataset/ocr_cache.csv`) to minimize redundant LLM API consumption across batch evaluation cycles.
