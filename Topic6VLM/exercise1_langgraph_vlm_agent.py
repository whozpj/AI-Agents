"""Exercise 1 – Vision-Language LangGraph Chat Agent.

This script launches either a CLI or Gradio chat experience that lets you hold a
multi-turn conversation about an uploaded image. LangGraph manages the
conversation state, while the Llava model served through Ollama answers
questions grounded in the provided photo.

Usage examples
--------------
# Launch Gradio (default interface)
python Topic6VLM/exercise1_langgraph_vlm_agent.py --image Topic6VLM/data/sample.jpg

# Stay in the terminal (CLI) and attach a different image later on
python Topic6VLM/exercise1_langgraph_vlm_agent.py --interface cli
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

import ollama
from langgraph.graph import END, START, StateGraph
from PIL import Image

try:
    import gradio as gr  # Imported lazily when launching the UI
except Exception:  # pragma: no cover - Gradio optional for CLI
    gr = None

SYSTEM_PROMPT = (
    "You are a meticulous vision-language analyst. Always tie your answers to "
    "concrete visual evidence. If the user uploads a new picture, mention that "
    "you are now looking at the updated scene. If you cannot see something in "
    "the image, clearly state that limitation instead of guessing."
)


def _encode_image(image: Image.Image, max_size: int) -> str:
    """Resize (if needed) and return the base64 encoding of a PIL image."""
    # Preserve aspect ratio when resizing very large pictures
    width, height = image.size
    largest_edge = max(width, height)
    if largest_edge > max_size:
        scale = max_size / float(largest_edge)
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size)
    with io.BytesIO() as buffer:
        image.convert("RGB").save(buffer, format="JPEG", quality=92)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _load_image_file(path: Path, max_size: int) -> str:
    with Image.open(path) as img:
        return _encode_image(img, max_size)


class AgentState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    image_b64: Optional[str]
    image_path: Optional[str]
    incoming_user_message: Optional[str]
    incoming_image_b64: Optional[str]
    assistant_response: Optional[str]
    should_exit: bool
    skip_llm: bool
    chat_history: List[List[str]]


def ensure_system_message(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": SYSTEM_PROMPT}] + messages


def prepare_turn(state: AgentState) -> AgentState:
    state.pop("skip_llm", None)
    state["messages"] = ensure_system_message(state.get("messages", []))
    incoming_text = (state.pop("incoming_user_message", "") or "").strip()
    incoming_image = state.pop("incoming_image_b64", None)

    if incoming_image:
        state["image_b64"] = incoming_image
        state["image_path"] = state.get("image_path") or "<uploaded via UI>"

    if not incoming_text:
        state["skip_llm"] = True
        state["assistant_response"] = (
            "Please describe what you would like to learn about the image."
        )
        return state

    lowered = incoming_text.lower()
    if lowered in {"quit", "exit", "q", ":q"}:
        state["should_exit"] = True
        state["skip_llm"] = True
        state["assistant_response"] = "Session closed."
        return state

    image_b64 = state.get("image_b64")
    msg_content = incoming_text
    if state.get("image_path"):
        msg_content = f"[Image source: {state['image_path']}]\n{incoming_text}"

    user_message: Dict[str, Any] = {"role": "user", "content": msg_content}
    if image_b64:
        user_message["images"] = [image_b64]

    state["messages"].append(user_message)
    state["skip_llm"] = False
    return state


def call_vlm(state: AgentState) -> AgentState:
    if state.get("skip_llm"):
        return state

    try:
        response = ollama.chat(model="llava", messages=state["messages"])
        state["assistant_response"] = response["message"]["content"].strip()
    except Exception as exc:  # pragma: no cover - network/runtime errors
        state["assistant_response"] = f"Error contacting Llava: {exc}"
        state["skip_llm"] = True
    return state


def update_history(state: AgentState) -> AgentState:
    if state.get("skip_llm"):
        return state

    assistant_message = {
        "role": "assistant",
        "content": state.get("assistant_response", "") or "(no response)",
    }
    state.setdefault("messages", []).append(assistant_message)
    return state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("prepare_turn", prepare_turn)
    graph.add_node("call_vlm", call_vlm)
    graph.add_node("update_history", update_history)
    graph.add_edge(START, "prepare_turn")

    def _route(state: AgentState) -> Literal["call_vlm", "end"]:
        if state.get("should_exit") or state.get("skip_llm"):
            return "end"
        return "call_vlm"

    graph.add_conditional_edges(
        "prepare_turn",
        _route,
        {
            "call_vlm": "call_vlm",
            "end": END,
        },
    )
    graph.add_edge("call_vlm", "update_history")
    graph.add_edge("update_history", END)
    return graph.compile()


def build_initial_state(image_path: Optional[Path], max_size: int) -> AgentState:
    if image_path is None:
        return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}]}
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_b64 = _load_image_file(image_path, max_size)
    return {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "image_b64": image_b64,
        "image_path": str(image_path),
    }


def run_cli(app, state: AgentState, max_size: int) -> None:
    print("Vision-Language LangGraph Chat Agent")
    print("Type '/image <path>' to swap the image or 'quit' to exit.\n")
    while True:
        user_input = input("You> ").strip()
        if not user_input:
            continue
        if user_input.startswith("/image"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 1:
                print("Usage: /image path/to/photo.jpg")
                continue
            new_path = Path(parts[1]).expanduser()
            if not new_path.exists():
                print(f"File not found: {new_path}")
                continue
            try:
                state["image_b64"] = _load_image_file(new_path, max_size)
                state["image_path"] = str(new_path)
                print(f"Loaded image {new_path}")
            except Exception as exc:
                print(f"Failed to load image: {exc}")
            continue

        graph_input = {**state, "incoming_user_message": user_input}
        result = app.invoke(graph_input)
        response_text = result.get("assistant_response", "(no response)")
        print(f"Llava> {response_text}\n")
        if result.get("should_exit"):
            break
        state = result


def launch_gradio(app, initial_state: AgentState, max_size: int) -> None:
    if gr is None:
        raise RuntimeError("Gradio is not installed. Reinstall requirements or use CLI mode.")

    def _handle_chat(user_text: str, image: Optional[Image.Image], state_dict: Optional[AgentState]):
        state_dict = state_dict or initial_state.copy()
        graph_input = dict(state_dict)
        graph_input["incoming_user_message"] = user_text
        if image is not None:
            graph_input["incoming_image_b64"] = _encode_image(image, max_size)
            graph_input["image_path"] = state_dict.get("image_path") or "<uploaded via UI>"
        result = app.invoke(graph_input)
        chat_history = state_dict.get("chat_history", [])
        chat_history.append([user_text, result.get("assistant_response", "(no response)")])
        result["chat_history"] = chat_history
        return chat_history, result, None, ""

    def _reset_state():
        return initial_state.copy(), []

    with gr.Blocks(title="Topic 6 – Vision-Language LangGraph Agent") as demo:
        gr.Markdown("## Topic 6 – Vision-Language LangGraph Chat Agent")
        chatbot = gr.Chatbot(height=420)
        image_input = gr.Image(label="Upload/Replace Image", type="pil")
        user_box = gr.Textbox(label="Ask about the picture", placeholder="e.g., What is happening in the photo?", lines=2)
        send_btn = gr.Button("Send")
        clear_btn = gr.Button("Reset Conversation")
        state = gr.State(initial_state.copy())

        send_btn.click(
            fn=_handle_chat,
            inputs=[user_box, image_input, state],
            outputs=[chatbot, state, image_input, user_box],
        )
        user_box.submit(
            fn=_handle_chat,
            inputs=[user_box, image_input, state],
            outputs=[chatbot, state, image_input, user_box],
        )
        clear_btn.click(fn=_reset_state, outputs=[state, chatbot])

    demo.launch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Topic 6 – Vision-Language LangGraph Agent")
    parser.add_argument("--image", type=str, help="Optional path to an image to preload", default=None)
    parser.add_argument("--interface", choices=["gradio", "cli"], default="gradio", help="Choose gradio UI or CLI mode")
    parser.add_argument("--max-size", type=int, default=1024, help="Maximum image edge in pixels after resizing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser() if args.image else None
    try:
        initial_state = build_initial_state(image_path, args.max_size)
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    app = build_graph()
    if args.interface == "cli":
        run_cli(app, initial_state, args.max_size)
    else:
        launch_gradio(app, initial_state, args.max_size)


if __name__ == "__main__":
    main()
