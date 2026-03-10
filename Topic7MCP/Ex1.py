# Exercise A: Discover the Asta MCP Tools
#
# ANSWERS:
# Q: Which tool would you use to find all papers about "transformer attention mechanisms"?
# A: search_papers — it accepts a keyword/semantic query and returns matching papers.
#
# Q: Which tool would you use to find who else published in the same area as a specific author?
# A: get_author_papers (to fetch that author's papers) combined with search_papers
#    using their topic keywords, OR get_paper_authors on their papers to surface
#    co-authors. If available, a dedicated "find similar authors" or
#    "get_author_details" tool would be ideal — but the workflow is:
#    get_author_papers → inspect fields/topics → search_papers on those topics.

import dotenv
import requests
import os
import json

from dotenv import load_dotenv
load_dotenv()

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ.get("ASTA_API_KEY", "")
}

if not headers["x-api-key"]:
    raise RuntimeError("ASTA_API_KEY environment variable not set. Please set it in your .env file or environment.")

payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {}
}

resp = requests.post(
    "https://asta-tools.allen.ai/mcp/v1",
    headers=headers,
    json=payload
)


resp.raise_for_status()
try:
    # Handle Server-Sent Events (SSE) response
    lines = resp.text.splitlines()
    json_data = None
    for line in lines:
        if line.startswith('data:'):
            json_str = line[len('data:'):].strip()
            try:
                json_data = json.loads(json_str)
                break
            except Exception as e:
                print(f"Failed to parse JSON from data line: {e}")
                print(f"Data line: {json_str}")
                raise
    if json_data is None:
        print("No 'data:' line found in response. Full response:")
        print(resp.text)
        raise RuntimeError("No 'data:' line found in SSE response.")
    data = json_data
except Exception as e:
    print(f"Failed to process SSE response: {e}")
    print(f"Raw response text: {resp.text}")
    raise

tools = data["result"]["tools"]

for tool in tools:
    name = tool.get("name", "unknown")
    description = tool.get("description", "No description provided.")
    # Grab just the first sentence / line for a one-liner
    one_line = description.split("\n")[0].split(". ")[0]

    schema = tool.get("inputSchema", {})
    properties = schema.get("properties", {})
    required_params = set(schema.get("required", []))

    required_list = []
    optional_list = []

    for param_name, param_schema in properties.items():
        param_type = param_schema.get("type", "any")
        if param_name in required_params:
            required_list.append(f"{param_name} ({param_type})")
        else:
            optional_list.append(f"{param_name} ({param_type})")

    print(f"Tool: {name}")
    print(f"  Description: {one_line}")
    if required_list:
        print(f"  Required: {', '.join(required_list)}")
    else:
        print(f"  Required: (none)")
    if optional_list:
        print(f"  Optional: {', '.join(optional_list)}")
    print()