"""
Exercise C: Asta-Powered Research Chatbot
Chatbot that fetches tool schemas from MCP and uses GPT-4o mini to decide tool calls.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

MCP_URL = "https://asta-tools.allen.ai/mcp/v1"
HEADERS = {
	"Content-Type": "application/json",
	"Accept": "application/json, text/event-stream",
	"x-api-key": os.environ.get("ASTA_API_KEY", "")
}
if not HEADERS["x-api-key"]:
	raise RuntimeError("ASTA_API_KEY environment variable not set. Please set it in your .env file or environment.")

def get_asta_tools():
	"""Fetch tool schemas from MCP and convert to OpenAI format."""
	payload = {
		"jsonrpc": "2.0",
		"id": 1,
		"method": "tools/list",
		"params": {}
	}
	resp = requests.post(MCP_URL, headers=HEADERS, json=payload)
	resp.raise_for_status()
	# Handle SSE response
	lines = resp.text.splitlines()
	json_data = None
	for line in lines:
		if line.startswith('data:'):
			json_str = line[len('data:'):].strip()
			try:
				json_data = json.loads(json_str)
				break
			except Exception:
				continue
	if json_data is None:
		raise RuntimeError("No 'data:' line found in SSE response.")
	tools = json_data["result"]["tools"]
	openai_tools = []
	for tool in tools:
		openai_tools.append({
			"type": "function",
			"function": {
				"name": tool["name"],
				"description": tool.get("description", ""),
				"parameters": tool.get("inputSchema", {})
			}
		})
	return openai_tools

def call_asta_tool(name, arguments):
	"""Execute a tools/call and return the text content or error."""
	payload = {
		"jsonrpc": "2.0",
		"id": 2,
		"method": "tools/call",
		"params": {
			"name": name,
			"arguments": arguments
		}
	}
	try:
		resp = requests.post(MCP_URL, headers=HEADERS, json=payload)
		resp.raise_for_status()
		lines = resp.text.splitlines()
		json_data = None
		for line in lines:
			if line.startswith('data:'):
				json_str = line[len('data:'):].strip()
				try:
					json_data = json.loads(json_str)
					break
				except Exception:
					continue
		if json_data is None:
			return {"error": "No 'data:' line found in SSE response."}
		content = json_data["result"]["content"][0]["text"]
		return {"content": content}
	except Exception as e:
		return {"error": str(e)}

# Dummy GPT-4o mini function-calling simulation (replace with OpenAI API in real use)
def gpt4o_mini(messages, tools):
	"""Simulate GPT-4o mini function-calling. Replace with OpenAI API for real use."""
	# For demo: parse user message and emit a tool call
	user = messages[-1]["content"].lower()
	if "large language model agents" in user:
		return {"tool_calls": [{"name": "search_papers_by_relevance", "arguments": {"keyword": "large language model agents", "fields": "title,year,authors", "limit": 3}}]}
	if "attention is all you need" in user:
		return {"tool_calls": [
			{"name": "search_paper_by_title", "arguments": {"title": "Attention is All You Need", "fields": "title,authors"}},
			{"name": "get_author_papers", "arguments": {"author_id": "1741100", "paper_fields": "title,year", "limit": 3}}
		]}
	if "bert" in user and "cite" in user:
		return {"tool_calls": [{"name": "get_citations", "arguments": {"paper_id": "ARXIV:1810.04805", "fields": "title,year", "limit": 3}}]}
	if "react paper" in user and "references" in user:
		return {"tool_calls": [{"name": "get_references", "arguments": {"paper_id": "ARXIV:2210.03629", "fields": "title,year"}}]}
	return {"content": "I'm not sure which tool to use."}

def chat(user_message, messages, tools):
	"""One turn of the chatbot loop, handling tool calls."""
	if user_message.strip().lower() in ["what tools are available", "list tools", "show tools", "tools?"]:
		print("\nAvailable tools:")
		for t in tools:
			fn = t["function"]
			print(f"- {fn['name']}: {fn.get('description', '').splitlines()[0]}")
		return
	messages.append({"role": "user", "content": user_message})
	# System prompt
	if len(messages) == 1:
		messages.insert(0, {"role": "system", "content": "You are a research assistant with access to Semantic Scholar tools via MCP."})
	# Call GPT-4o mini (replace with OpenAI API in real use)
	model_response = gpt4o_mini(messages, tools)
	if "tool_calls" in model_response:
		for call in model_response["tool_calls"]:
			name = call["name"]
			arguments = call["arguments"]
			print(f"[TOOL CALL] {name}({json.dumps(arguments)})")
			result = call_asta_tool(name, arguments)
			if "error" in result:
				print(f"[TOOL ERROR] {result['error']}")
				messages.append({"role": "tool", "content": f"Error: {result['error']}"})
			else:
				print(f"[TOOL RESULT] {result['content']}")
				messages.append({"role": "tool", "content": result["content"]})
		# Simulate model producing a final answer after tool calls
		print("[BOT] (Final answer would be generated here based on tool results.)")
	else:
		print(f"[BOT] {model_response['content']}")

if __name__ == "__main__":
	tools = get_asta_tools()
	messages = []
	print("Type a research question (or 'exit' to quit):")
	while True:
		user_message = input("You: ")
		if user_message.strip().lower() == "exit":
			break
		chat(user_message, messages, tools)
