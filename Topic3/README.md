# Topic 3: Agent Tool Use

## Table of Contents
| File/Folder                      | Description                                                      |
|----------------------------------|------------------------------------------------------------------|
| README.md                        | This file                                                        |
| langgraph_tool_handling.py       | LangGraph tool orchestration example                             |
| manual_tool_handling.py          | Manual tool invocation example                                   |
| llama_mmlu_eval_astronomy.py     | MMLU eval on Astronomy topic (Ollama & local)                    |
| llama_mmlu_eval_business_ethics.py| MMLU eval on Business Ethics topic (Ollama & local)              |
| Topic3Tools/                     | Portfolio: all code, outputs, and traces for this topic          |

## Learning Goals
- Run Ollama LLM server and use Llama 3.2-1B locally or on Colab
- Run large commercial models in LangGraph
- Understand OpenAI tool calling format
- Efficiently define and use LangGraph tools

## Tasks & Instructions
1. **Ollama Setup**: Run Llama 3.2-1B on Ollama (local/Colab). Modify and time two MMLU scripts on different topics. Compare sequential vs parallel execution using `time` in the shell.
2. **OpenAI GPT-4o Mini**: Set up API key securely (see below). Test with provided code sample. Explain key lines in README.
3. **Manual Tool Handling**: Download, run, and modify manual-tool-handling.py to add a calculator tool (with geometric functions). Use `json.loads`/`json.dumps` for input/output. Save example outputs.
4. **LangGraph Tool Handling**: Add your calculator, a letter count tool, and a third custom tool. Refactor tool dispatch to use a tool map. Save outputs for multi-tool queries and chaining.
5. **Long Conversation Agent**: Rewrite to use LangGraph nodes/edges for a single long conversation, with checkpointing and recovery. Include a Mermaid diagram and example traces.
6. **Parallelization Opportunity**: Discuss where further parallelization could be added in your agent (no code required).

## API Key Setup
- **Laptop**: Add to `~/.profile`:
  ```bash
  export OPENAI_API_KEY="your-actual-key"
  ```
  In code: `api_key = os.getenv("OPENAI_API_KEY")`
- **Colab**: Use secret manager (see instructions above).

## Example Code Explanation
- `client = OpenAI()` creates the API client.
- `response = client.chat.completions.create(...)` sends a chat request to GPT-4o Mini.

## Outputs & Observations
- All program outputs, timing logs, and traces are saved in `Topic3Tools/` with descriptive filenames.
- Sequential vs parallel execution: Parallel runs are faster if both models are loaded and system resources allow.
- Tool use: Forcing LLMs to use tools may require prompt engineering or system instructions.
- Multi-tool queries: Chaining and parallel tool invocation can be demonstrated with crafted questions.
- Checkpointing: Agent can recover conversation state after interruption.
- Parallelization: Further parallelization could be added in tool execution and response aggregation.

## Portfolio Directory
All code, outputs, and traces for Topic 3 are in `Topic3Tools/`. Each file is named for its task and purpose.

## Resources
- Ollama Guide
- OpenAI GPT-4o Mini Test
- manual-tool-handling.py
- langgraph-tool-handling.py
- Strategies for forcing LLM tool use

