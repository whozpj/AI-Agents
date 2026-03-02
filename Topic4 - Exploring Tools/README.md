## TEAM  
- Ariful Islam
- Prithvi Raj 
- Shaina	Kumar


## 2-Hour Agent Project

The Smart Travel Planner:

- Accepts destination + travel dates
- Calls OpenWeatherMap API
- Generates:
  - Weather snapshot
  - Packing list (with reasons)
  - Activity recommendations
- Handles:
  - Invalid city
  - Invalid date ranges
  - API errors
  - `verbose`, `quiet`, `exit` commands
 
The Smart Travel Planner uses a LangGraph conversation wrapper around a LangChain ReAct agent. It defines a get_weather_forecast tool that calls the OpenWeatherMap API for a specific city and date range, then formats the results. The graph controls the flow: an input node handles user commands (verbose, quiet, exit), the agent node generates responses using the weather tool, an output node prints the result, and a trim node limits conversation history. A system prompt enforces a structured response format (weather snapshot, packing list, activities). Because the free OpenWeatherMap API tier only provides a 5-day forecast, the planner can only return weather data within that window. If a user provides a longer trip, the tool warns that only the first five days are available and returns data for the supported range. The --test-tool mode runs the weather tool independently without the agent. It executes predefined test cases to verify API connectivity, unit handling, date validation, and error handling before full agent integration.

---

Run full agent:

    python weather_travel_planner.py

Run tool test mode:

    python weather_travel_planner.py --test-tool

Link to Google Colab Notebook: https://colab.research.google.com/drive/1wn5cdGza_iPSRci_FpkEWOaOcooBOvN0?usp=sharing


# Topic 4: Exploring Tools

## Directory Table of Contents
| Name                   | Description                                      |
|------------------------|--------------------------------------------------|
| README.md              | This file                                        |
| react_agent_example.py | React agent example for LangChain abstraction    |
| toolnode_example.py    | ToolNode example for parallel tool dispatch      |
| 2hr project/           | Folder for 2-Hour Agent Project work             |

