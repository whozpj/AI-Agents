# Mermaid Diagram: LangGraph Agent System

```mermaid
graph TD
    START --> get_user_input
    get_user_input -->|if quit| END
    get_user_input -->|else| dispatch
    dispatch --> call_llama
    dispatch --> call_qwen
    call_llama --> join_responses
    call_qwen --> join_responses
    join_responses --> print_response
    print_response --> get_user_input
    END[End]
```

This diagram shows the flow of nodes and edges in the LangGraph agent system, including user input, model calls, response joining, and looping with checkpointing.
