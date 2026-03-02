# Topic 3: Agent Tool Use

## Table of Contents
| File Name                          | Description                                      |
|------------------------------------|--------------------------------------------------|
| README.md                          | This file                                        |
| langgraph_tool_handling.py         | LangGraph tool orchestration example              |
| manual_tool_handling.py            | Manual tool invocation example                   |
| llama_mmlu_eval_astronomy.py       | MMLU eval on Astronomy topic (Ollama & local)     |
| llama_mmlu_eval_business_ethics.py | MMLU eval on Business Ethics topic (Ollama & local)|
| ollama_mmlu_astronomy_output.txt   | Output: Llama MMLU Astronomy (Ollama)            |
| ollama_mmlu_business_ethics_output.txt| Output: Llama MMLU Business Ethics (Ollama)     |
| gpt4o_mini_test_output.txt         | Output: OpenAI GPT-4o Mini test                  |
| tool_use_examples.txt              | Output: Multi-tool queries and chaining           |
| conversation_trace.txt             | Output: Long conversation with checkpointing      |
| parallelization_notes.txt          | Discussion: Parallelization opportunities         |

## Instructions & Observations
- Ollama server was used to run Llama 3.2-1B on Astronomy and Business Ethics topics. Timings and accuracy are in the output files.
- OpenAI GPT-4o Mini was tested for tool use. API key setup instructions are in the main portfolio README.
- Manual and LangGraph tool handling scripts demonstrate calculator, letter count, and custom tools. See outputs for multi-tool queries and chaining.
- Long conversation trace shows context, tool use, and checkpointing.
- Parallelization notes discuss further opportunities for speedup.

## Notes
- All outputs are saved as text files with descriptive names.
- Code files are named by task and purpose.
- See `conversation_trace.txt` for examples of tool use, context, and recovery.
- See `parallelization_notes.txt` for discussion on further parallelization.

