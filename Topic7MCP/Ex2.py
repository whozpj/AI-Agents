"""
Exercise B: Direct Asta Tool Calls — Three Focused Drills
Drill 1 — search_papers: Find recent LLM agent papers
Drill 2 — get_citations: Trace impact of a landmark paper
Drill 3 — get_references: Understand a paper's intellectual foundation
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

headers = {
	"Content-Type": "application/json",
	"Accept": "application/json, text/event-stream",
	"x-api-key": os.environ.get("ASTA_API_KEY", "")
}
if not headers["x-api-key"]:
	raise RuntimeError("ASTA_API_KEY environment variable not set. Please set it in your .env file or environment.")

def call_asta_tool(arguments, request_id=2):
	url = "https://asta-tools.allen.ai/mcp/v1"
	payload = {
		"jsonrpc": "2.0",
		"id": request_id,
		"method": "tools/call",
		"params": arguments
	}
	resp = requests.post(url, headers=headers, json=payload)
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
	return json_data

def drill1_search_papers():
	print("\nDrill 1 — search_papers: Find recent LLM agent papers")
	arguments = {
		"name": "search_papers_by_relevance",
		"arguments": {
			"keyword": "large language model agents",
			"fields": "title,abstract,year,authors",
			"limit": 5
		}
	}
	data = call_asta_tool(arguments, request_id=2)
	content = data["result"]["content"][0]["text"]
	papers = json.loads(content)
	if isinstance(papers, dict):
		# Print the single paper
		print(f"1. {papers.get('title', '(no title)')} ({papers.get('year', '?')})")
		return
	if not isinstance(papers, list):
		print("[DEBUG] Unexpected papers type:", type(papers), papers)
		return
	if len(papers) > 0 and isinstance(papers[0], str):
		print("[DEBUG] papers list contains strings, printing all:")
		for p in papers:
			print(f"Could not parse paper: {p}")
		return
	for idx, paper in enumerate(papers, 1):
		title = paper.get("title", "(no title)")
		year = paper.get("year", "?")
		print(f"{idx}. {title} ({year})")

def drill2_get_citations():
	print("\nDrill 2 — get_citations: Trace impact of a landmark paper")
	arguments = {
		"name": "get_citations",
		"arguments": {
			"paper_id": "ARXIV:1810.04805",
			"fields": "title,year,authors",
			"limit": 10,
			"publication_date_range": "2023-01-01:"
		}
	}
	data = call_asta_tool(arguments, request_id=3)
	content = data["result"]["content"][0]["text"]
	citing_papers = json.loads(content)
	# Sometimes the result is a dict, not a list
	if isinstance(citing_papers, dict):
		print("[DEBUG] citing_papers is a dict:", citing_papers)
		papers_list = list(citing_papers.values())
	elif isinstance(citing_papers, list):
		papers_list = citing_papers
	else:
		print("[DEBUG] Unexpected citing_papers type:", type(citing_papers), citing_papers)
		papers_list = []
	print(f"Citations since 2023: {len(papers_list)}")
	for idx, paper in enumerate(papers_list[:5], 1):
		title = paper.get("title", "(no title)")
		print(f"{idx}. {title}")

def drill3_get_references():
	print("\nDrill 3 — get_references: Understand a paper's intellectual foundation")
	arguments = {
		"name": "get_references",
		"arguments": {
			"paper_id": "ARXIV:2210.03629",
			"fields": "title,year"
		}
	}
	data = call_asta_tool(arguments, request_id=4)
	content = data["result"]["content"][0]["text"]
	if not content.strip():
		print("[DEBUG] get_references returned empty content.")
		return
	try:
		references = json.loads(content)
	except Exception as e:
		print(f"[DEBUG] Could not parse references JSON: {e}\nRaw content: {content}")
		return
	if isinstance(references, dict):
		references = [references]
	references = sorted(references, key=lambda x: x.get("year", 0) or 0)
	for ref in references:
		title = ref.get("title", "(no title)")
		year = ref.get("year", "?")
		print(f"{year}: {title}")

if __name__ == "__main__":
	drill1_search_papers()
	time.sleep(1)
	drill2_get_citations()
	time.sleep(1)
	drill3_get_references()
