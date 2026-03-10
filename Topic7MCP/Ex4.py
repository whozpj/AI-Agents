"""
Exercise D: Citation Network Explorer Agent
Given a seed paper by ArXiv ID, build a citation neighborhood and produce a structured markdown report.
"""

import os
import sys
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

def call_asta_tool(name, arguments):
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments
        }
    }
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
        raise RuntimeError("No 'data:' line found in SSE response.")
    content = json_data["result"]["content"][0]["text"]
    if not content.strip():
        return None
    try:
        return json.loads(content)
    except Exception as e:
        print(f"[DEBUG] Could not parse JSON for {name}: {e}\nRaw content: {content}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python Ex4.py <ARXIV_ID>")
        sys.exit(1)
    paper_id = sys.argv[1]

    # 1. Retrieve full metadata for the seed paper
    paper = call_asta_tool("get_paper", {"paper_id": paper_id, "fields": "title,abstract,year,authors,fieldsOfStudy"})
    title = paper.get("title", "?")
    abstract = paper.get("abstract", "?")
    year = paper.get("year", "?")
    authors = paper.get("authors", [])
    fields = paper.get("fieldsOfStudy", [])

    # 2. Fetch references and retrieve abstracts for 5 most-cited
    references = call_asta_tool("get_references", {"paper_id": paper_id, "fields": "title,year,abstract,citationCount,authors"})
    if not references:
        print("[DEBUG] No references found or could not parse references.")
        references = []
    elif isinstance(references, dict):
        references = [references]
    references = sorted(references, key=lambda x: x.get("citationCount", 0), reverse=True)[:5]

    # 3. Fetch recent citing papers (last 3 years)
    citations = call_asta_tool("get_citations", {"paper_id": paper_id, "fields": "title,year,abstract,authors", "limit": 20, "publication_date_range": f"{year-3 if isinstance(year, int) else ''}:"})
    if isinstance(citations, dict):
        citations = [citations]
    citations = sorted(citations, key=lambda x: x.get("year", 0), reverse=True)[:5]

    # 4. For each author, retrieve their most-cited other work
    author_profiles = []
    for author in authors:
        author_id = author.get("authorId")
        if not author_id:
            continue
        papers = call_asta_tool("get_author_papers", {"author_id": author_id, "paper_fields": "title,year,citationCount", "limit": 10})
        if isinstance(papers, dict):
            papers = [papers]
        # Exclude the seed paper itself
        papers = [p for p in papers if p.get("title") != title]
        if papers:
            most_cited = max(papers, key=lambda x: x.get("citationCount", 0))
            author_profiles.append({"name": author.get("name", "?"), "work": most_cited})
        else:
            author_profiles.append({"name": author.get("name", "?"), "work": None})

    # 5. Generate markdown report
    print(f"# Citation Network Report for {title}\n")
    print(f"**Year:** {year}  ")
    print(f"**Authors:** {', '.join(a.get('name', '?') for a in authors)}  ")
    print(f"**Fields of Study:** {', '.join(fields)}\n")
    print(f"## Summary\n{abstract}\n")
    print("## Foundational Works\n")
    for ref in references:
        print(f"- **{ref.get('title', '?')}** ({ref.get('year', '?')}) — Citations: {ref.get('citationCount', 0)}\n  {ref.get('abstract', '')[:300]}{'...' if ref.get('abstract') and len(ref.get('abstract')) > 300 else ''}")
    print("\n## Recent Developments\n")
    for cit in citations:
        print(f"- **{cit.get('title', '?')}** ({cit.get('year', '?')})\n  {cit.get('abstract', '')[:300]}{'...' if cit.get('abstract') and len(cit.get('abstract')) > 300 else ''}")
    print("\n## Author Profiles\n")
    for profile in author_profiles:
        work = profile["work"]
        if work:
            print(f"- **{profile['name']}**: {work.get('title', '?')} ({work.get('year', '?')}) — Citations: {work.get('citationCount', 0)}")
        else:
            print(f"- **{profile['name']}**: No other notable works found.")

if __name__ == "__main__":
    main()
