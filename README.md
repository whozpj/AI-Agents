# AI-Agents
CS 6501 – Workshop on Building AI Agents

hi! i’m Prithvi and this repo tracks all the weird experiments we’ve been doing for Prof. Kautz’s class. nothing fancy—just a running diary of what’s working, what’s broken, and the commands i keep forgetting.

---

## Topic 3 – timing llama stuff the low-tech way
- two trimmed scripts live in `Topic3/`: `llama_mmlu_eval_astronomy.py` and `llama_mmlu_eval_business_ethics.py`. each only hits one MMLU subject so i can actually finish a run before coffee gets cold.
- baseline check (hugging face pipeline):
	- `time python Topic3/llama_mmlu_eval_astronomy.py`
	- `time python Topic3/llama_mmlu_eval_business_ethics.py`
- switch to ollama by running `ollama pull llama3.2:1b` and keeping `ollama serve` alive (or just leave the desktop app open). update the scripts to call `ollama.chat(...)` instead of transformers.
- timing commands the assignment wants:
	- sequential: `time { python Topic3/llama_mmlu_eval_astronomy.py ; python Topic3/llama_mmlu_eval_business_ethics.py ; }`
	- parallel: `time { python Topic3/llama_mmlu_eval_astronomy.py & python Topic3/llama_mmlu_eval_business_ethics.py & wait; }`
- results i still need to plug in: `TBD` for HF alone, `TBD` for ollama sequential, `TBD` for ollama parallel. pretty sure parallel will only help if my cpu/gpu isn’t already crying.

---

## GPT-4o mini plan (aka please don’t leak the api key)
1. make sure the openai account has billing + a sane monthly cap so i don’t accidentally buy a yacht worth of tokens.
2. secrets:
	 - local laptop: drop `export OPENAI_API_KEY="sk-..."` into `~/.profile`, reload, and grab it in python with `os.getenv("OPENAI_API_KEY")`.
	 - colab: secrets tab (little key icon) → add `OPENAI_API_KEY` → enable notebook access → stick this at the top:
		 ```python
		 from google.colab import userdata
		 import os

		 os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
		 ```
3. install `openai` (`pip install --upgrade openai` if needed).
4. smoke test:
	 ```python
	 from openai import OpenAI

	 client = OpenAI()  # reads the env var, no hard-coded key
	 response = client.chat.completions.create(
			 model="gpt-4o-mini",
			 messages=[{"role": "user", "content": "Say: Working!"}],
			 max_tokens=5,
	 )
	 print(response.choices[0].message.content)
	 ```
	 - `client = OpenAI()` = “hey sdk, use whatever key/org/project i set in env”.
	 - `client.chat.completions.create(...)` = sends one shot prompt to gpt-4o mini with a short max token budget.

---

## manual tool handling homework (custom calculator)
- base file: `Topic3/manual_tool_handling.py`. grab the starter from the docs if it’s missing.
- build a `geometry_calculator` tool that eats JSON like `{ "operation": "area_circle", "radius": 2.5 }`, parse with `json.loads`, crunch numbers with `math`/`numexpr`/`ast.literal_eval`, then return nice JSON via `json.dumps`.
- logs go in `Topic3/tool_runs/manual_tool_calculator.txt`. need at least a couple runs showing GPT-4o mini actually CALLS the tool when i yell “no mental math allowed”.

---

## langgraph multi-tool mashup
- stick everything in `Topic3/langgraph_multi_tool_agent.py`.
- tools i’m wiring in:
	1. geometry calculator from above (so trig/areas/etc are handled consistently).
	2. `letter_count` that answers questions like “how many s are in Mississippi riverboats”. returns raw counts + normalized stats.
	3. some extra tool (maybe quick unit converter or date gap calculator).
- dispatch pattern (no more giant if/else):
	```python
	tools = [calculator_tool, letter_count_tool, bonus_tool]
	tool_map = {tool.name: tool for tool in tools}

	if function_name in tool_map:
			result = tool_map[function_name].invoke(function_args)
	else:
			result = f"Error: Unknown function {function_name}"
	```
- sample prompts i want to capture in `Topic3/tool_runs/langgraph_multi_tool_examples.txt`:
	- “Are there more i’s than s’s in Mississippi riverboats?” → should call `letter_count` twice.
	- “What’s the sin of the difference between the number of i’s and s’s…?” → letter counts (twice) + calculator.
	- stretch: question that forces all tools + tries to hit the 5-turn limit.

---

## long-running langgraph convo w/ checkpoints
- rebuild the manual tool handler as a LangGraph state machine so the chat keeps context instead of rebooting every question.
- rough node list: `ingest_user_turn → decide_tool_call → invoke_tool → call_gpt4omini → summarize_and_checkpoint` (loop until user quits or step limit hits).
- persistence: use `langgraph.checkpoint.sqlite.SqliteSaver` pointing at `Topic3/checkpoints/tool_agent.sqlite3` so i can kill the process and resume.
- documentation to drop later:
	- Mermaid diagram saved at `Topic3/docs/tool_graph.mmd`.
	- transcript showing tool use + a recovery scenario where the graph reloads from checkpoint.

---

## random optimization idea
right now every tool call runs one after another even if they’re totally independent (e.g., counting multiple letters, running two geometry ops). future me should fan those out in parallel—spawn async tasks, gather the results, then move on. would slash latency when the tools are i/o bound (gpu or external apis) without wrecking the graph logic.

