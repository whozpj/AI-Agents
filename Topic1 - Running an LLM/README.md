# Running an LLM (Topic 1)

## Table of Contents
| File/Folder         | Description                                                      |
|---------------------|------------------------------------------------------------------|
| README.md           | This file                                                        |
| chat_agent.py       | Simple chat agent using Llama 3.2-1B                             |
| llama_mmlu_eval.py  | MMLU benchmark runner for Llama                                  |
| mistral_mmlu_eval.py| MMLU benchmark runner for Mistral                                |
| qwen_mmlu_eval.py   | MMLU benchmark runner for Qwen                                   |
| results/            | Output JSONs, graphs, and comparison results                     |



### Example Timings
| Model                  | GPU (no quant) | CPU (no quant) | CPU (4-bit) |
|------------------------|---------------|---------------|-------------|
| Llama-3.2-1B-Instruct  | 0:49.4        | 2:15.1        | 1:23.4      |
| Qwen2.5-0.5B           | 0:53.8        | 2:18.2        | 1:41.5      |
| Mistral-7B-Instruct-v0.3| 5:12.8        | 18:47.4       | 10:34.6     |

## Results & Graphs
- All output JSONs and graphs are in the `results/` folder.
- See `mmlu_comparison.png` for accuracy/performance comparison.
- Example output files:
  - `llama_3.2_1b_mmlu_results_full_20260127_172736.json`
  - `Qwen2.5-0.5B_mmlu_results_full_20260127_181519.json`

## Analysis & Discussion
- Both Llama and Qwen perform similarly (~43% accuracy).
- Abstract Algebra is hardest (~24-33%), Astronomy is easiest (~50%).
- Mistakes cluster by subject, not random.
- To check if models fail on the same questions, save per-question results and compare overlap.

## Chat Agent
- `chat_agent.py` implements a simple chat agent using Llama 3.2-1B.
- Maintains conversation context; can toggle history on/off.
- For long conversations, see Llama Chat Context Management Guide for better context handling.
- Try running with and without history to compare multi-turn performance.


