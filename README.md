# AI-Agents
CS 6501 - Workshop on Building AI Agents

Prithvi Raj
Proffessor Kautz


## Topic 3 – Llama vs. Ollama Timing Experiments

- Scripts: `Topic3/llama_mmlu_eval_astronomy.py` and `Topic3/llama_mmlu_eval_business_ethics.py` are trimmed copies of the Topic 1 evaluator with a single MMLU subject each.
- Baseline timing (Hugging Face backend): run `time python Topic3/llama_mmlu_eval_astronomy.py` and `time python Topic3/llama_mmlu_eval_business_ethics.py` individually to capture wall-clock values before switching to Ollama.
- Ollama setup: `ollama pull llama3.2:1b` (short name after pull) and start the server with `ollama serve` (or keep the desktop app running). Update both evaluators to call `ollama.chat` instead of Hugging Face transformers for the second set of measurements.
- Sequential timing command: `time { python Topic3/llama_mmlu_eval_astronomy.py ; python Topic3/llama_mmlu_eval_business_ethics.py ; }`
- Parallel timing command: `time { python Topic3/llama_mmlu_eval_astronomy.py & python Topic3/llama_mmlu_eval_business_ethics.py & wait; }`
- Observation notes placeholder (replace once runs finish):
	- HF inference (per script): `TBD` seconds real time on CPU.
	- Ollama sequential total: `TBD` seconds real time.
	- Ollama parallel total: `TBD` seconds real time.
	- Expected trend: parallel execution should approach the max of individual runtimes when the local machine has enough cores/GPU memory; otherwise contention may erase the gain.

## GPT-4o Mini Setup Checklist

1. **Account and billing**: create/verify the OpenAI account, attach a payment method, and set a monthly budget cap under the Usage settings page.
2. **API key management**:
	 - Laptop: add `export OPENAI_API_KEY="sk-xxxx"` to `~/.profile`, `source ~/.profile`, and access it via `os.getenv("OPENAI_API_KEY")` inside scripts.
	 - Google Colab: Secrets panel (key icon) > add secret `OPENAI_API_KEY` > enable notebook access, then place the snippet below at the top of every Colab runtime:
		 ```python
		 from google.colab import userdata
		 import os

		 os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
		 ```
3. **Library install**: `pip install openai` (or `pip install --upgrade openai` if the SDK pre-exists).
4. **Sanity test**:
	 ```python
	 from openai import OpenAI

	 client = OpenAI()  # creates a client bound to the OPENAI_API_KEY env var defined above
	 response = client.chat.completions.create(
			 model="gpt-4o-mini",
			 messages=[{"role": "user", "content": "Say: Working!"}],
			 max_tokens=5,
	 )
	 print(response.choices[0].message.content)
	 ```
	 - `client = OpenAI()` instantiates the SDK wrapper and automatically reads `OPENAI_API_KEY` plus optional org/project IDs from the environment; no key literal should appear in code.
	 - `client.chat.completions.create(...)` sends a single-turn request to the hosted GPT-4o Mini endpoint with your prompt (`messages`) and decoding parameters (`max_tokens`).

## Manual Tool Handling – Custom Calculator

- Base file: `Topic3/manual_tool_handling.py` (add if missing by copying the sample from the assignment).
- Implement a `geometry_calculator` tool that accepts JSON such as `{ "operation": "area_circle", "radius": 2.5 }`, parses with `json.loads`, evaluates using `numexpr` or `math` + `LiteralEval`, and returns a JSON string via `json.dumps` for structured responses.
- Capture CLI transcripts (planned location: `Topic3/tool_runs/manual_tool_calculator.txt`). Include examples where GPT-4o Mini is forced to call the tool for trigonometric expressions instead of hallucinating arithmetic.
- If the model tries to answer inline, add stronger system instructions (e.g., “Never compute directly; you must call the calculator tool for any numeric result”) and document the prompt tweaks in this README.

## LangGraph Tool Handling Extensions

- Combine the calculator tool with at least two additional tools:
	1. `letter_count` – counts occurrences of a specified character in text; responds with a JSON payload containing both absolute counts and normalized ratios.
	2. `custom_tool` (free-form idea, e.g., `unit_converter` or `date_interval`).
- Store code in `Topic3/langgraph_multi_tool_agent.py` and wire tools via a dispatcher:
	```python
	tools = [calculator_tool, letter_count_tool, my_extra_tool]
	tool_map = {tool.name: tool for tool in tools}
	...
	if function_name in tool_map:
			result = tool_map[function_name].invoke(function_args)
	else:
			result = f"Error: Unknown function {function_name}"
	```
- Portfolio artifacts to capture (placeholders until runs complete):
	- `Topic3/tool_runs/langgraph_multi_tool_examples.txt`
	- A query that triggers nested tool calls such as “What is sin(count_i - count_s) for ‘Mississippi riverboats’?” to demonstrate inner-loop letter counting + outer-loop calculator usage.

## Long-Running LangGraph Conversation with Checkpointing

- Goal: rewrite the manual tool handler so a single LangGraph graph maintains conversation state instead of restarting per question.
- Nodes: `ingest_user_turn`, `decide_tool_call`, `invoke_tool`, `call_gpt4omini`, `summarize_and_checkpoint`.
- Suggested storage: `langgraph.checkpoint.sqlite.SqliteSaver` pointed to `Topic3/checkpoints/tool_agent.sqlite3` for persistence and recovery after crashes.
- Documentation deliverables:
	- Mermaid diagram saved as `Topic3/docs/tool_graph.mmd`.
	- Conversation transcript showing context carry-over, tool use, and a restart scenario where prior checkpoints are loaded.

## Parallelization Opportunity

The current agent evaluates tool invocations strictly sequentially—even when a user question clearly decomposes into independent sub-queries (e.g., counting multiple letters, computing two geometric areas). A straightforward optimization would be to fan out independent tool calls in parallel (async tasks or thread pool) and only serialize when the tool results must be combined. This would reduce turn latency when the tools are I/O bound (GPU calls, API hits) without complicating the LangGraph structure.

