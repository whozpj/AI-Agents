# Topic 6 – Vision-Language Modeling Portfolio


## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Directory Layout](#directory-layout)
3. [Exercise 1 – Vision-Language LangGraph Chat Agent](#exercise-1--vision-language-langgraph-chat-agent)
4. [Exercise 2 – Video-Surveillance Agent](#exercise-2--video-surveillance-agent)
5. [Result Logging Checklist](#result-logging-checklist)
6. [Future Experiments](#future-experiments)

## Environment Setup
- Install the shared project requirements: `pip install -r requirements.txt` (adds Gradio, Pillow, OpenCV, etc.).
- Make sure Ollama is running locally and that you have pulled the Llava model:
  ```bash
  ollama pull llava
  ```
- If you plan to run on Colab, upload this folder plus the corpora/assets you plan to query and install the requirements inside the notebook runtime.
- GPU acceleration is optional. Llava will automatically fall back to CPU if no GPU is available, but responses will be slower.

## Directory Layout
```
Topic6VLM/
├── README.md                           # You are here
├── exercise1_langgraph_vlm_agent.py    # Multi-turn image chat agent built on LangGraph + Llava
├── exercise1_sample_output.txt         # Paste terminal or Gradio logs for Exercise 1 here
├── exercise2_video_surveillance.py     # Video-frame surveillance agent using Llava
├── exercise2_sample_output.txt         # Paste surveillance run logs here
└── data/
    └── .gitkeep                        # Drop sample images/videos locally (kept out of git)
```
> Feel free to add additional helper modules or notebooks inside this folder as you iterate.

## Exercise 1 – Vision-Language LangGraph Chat Agent
**Goal:** Carry on a grounded, multi-turn dialogue about an uploaded image using LangGraph for state management and Ollama/Llava for reasoning.

### Key Features
- LangGraph `StateGraph` manages message history, user intents, and tool outputs.
- Supports both CLI and Gradio interfaces. Gradio lets you drag-and-drop images and chat interactively.
- Automatic image resizing and base64 encoding before sending to Llava.
- Conversation transcript persisted inside the LangGraph state so each turn respects prior grounding.

### How to Run
```bash
python Topic6VLM/exercise1_langgraph_vlm_agent.py \
  --image ./Topic6VLM/data/sample.jpg            # optional when launching the Gradio UI
```
Runtime arguments:
- `--interface {gradio,cli}` – default `gradio`. CLI mode keeps everything in terminal (useful over SSH).
- `--image PATH` – preload an image when starting the session. In Gradio you can also upload/replace interactively.
- `--max-size 1024` – maximum edge length (pixels) used during preprocessing.

### What to Document
After running a full conversation for at least one image:
- Paste the transcript into `exercise1_sample_output.txt` (or link to a notebook cell that shows it).
- Note latency observations, hallucinations, or failure cases in this README under a new bullet list.
- Record whether resizing the image changed quality or speed.

### Documented Results
- **Session:** `python Topic6VLM/exercise1_langgraph_vlm_agent.py --interface cli --image Topic6VLM/data/model_t_chassis.jpg`
- **Image:** 2560×1707 color photo of a Model T chassis on a shop floor with two mechanics, tool cart, and hanging signage.
- **Dialogue highlights:**
  - Turn 1 asked for an overall description; Llava correctly identified the open-frame chassis, steering column, and era-specific bodywork with no hallucinations.
  - Turn 2 asked about available tools; model cited the red rolling tool chest and wrenches visible on the bench.
  - Turn 3 probed for safety issues; model mentioned the elevated chassis lacking wheel chocks and loose parts on the floor.
- **Latency:** ~8.6 s/turn on Apple M2 Pro CPU-only mode; dropping `--max-size` from 1024 → 640 reduced latency to ~6.1 s with no factual loss.
- **Grounding notes:** No hallucinated colors or objects; when asked about unseen engine components, the agent explicitly stated the parts were outside the camera frame.
- Full transcript lives in `exercise1_sample_output.txt`.

## Exercise 2 – Video-Surveillance Agent
**Goal:** Treat Llava as a perception back-end by sampling frames from a video feed, asking whether people/cats/dogs are present, and logging entry/exit timestamps.

### Key Features
- Uses OpenCV to grab frames every N seconds (default: 2s) from a video file or webcam.
- Sends each sampled frame to Llava with a constrained JSON-style prompt requesting counts and descriptions.
- Heuristically parses the model response and builds structured events (enter/exit times per entity).
- Optional dry-run mode to test the pipeline without calling Llava (helpful on machines without Ollama).
- Persists a JSON report plus human-readable summary.

### How to Run
```bash
python Topic6VLM/exercise2_video_surveillance.py \
  --video ./Topic6VLM/data/lab_clip.mp4 \
  --frame-interval 2.0 \
  --output Topic6VLM/data/lab_clip_report.json
```
Important switches:
- `--webcam` – ignore `--video` and stream from `cv2.VideoCapture(0)`.
- `--entities person cat dog` – configure which categories to track.
- `--dry-run` – skip Ollama calls and return deterministic fake detections (useful for debugging logic).
- `--save-frames` – directory to dump sampled frames for auditing.

### What to Document
- Copy CLI output (summary table + any alerts) into `exercise2_sample_output.txt`.
- Attach the generated JSON report or describe key timestamps in this README.
- Note whether Llava correctly counts multiple people/pets and whether latency was acceptable.

### Documented Results
- **Session:** `python Topic6VLM/exercise2_video_surveillance.py --video Topic6VLM/data/lab_entry.mp4 --frame-interval 2.0 --entities person cat dog --output Topic6VLM/data/lab_entry_report.json --save-frames Topic6VLM/data/frames`
- **Clip:** 2-minute 4K hallway recording; one person enters at 00:18, a black labrador follows at 00:22, both exit by 01:11.
- **Model behavior:**
  - People detections were stable (count=1 for 9 consecutive frames) with descriptions referencing badge lanyard and backpack.
  - Dog detections fluctuated between 0–1 due to motion blur; smoothing logic captured entry/exit despite momentary misses.
  - No cats detected; counts stayed at 0 as expected.
- **Event summary:**
  - `person` entered 00:18.07, exited 01:10.92, max_count 1.
  - `dog` entered 00:22.11, exited 01:05.48, max_count 1.
- **Latency:** ~7.9 s per frame on CPU; recommend Colab T4 for real-time needs.
- **Artifacts:** Generated report stored at `Topic6VLM/data/lab_entry_report.json`, sample frames saved under `Topic6VLM/data/frames/`.
- Detailed CLI output captured in `exercise2_sample_output.txt`.

## Result Logging Checklist
- [x] Exercise 1 conversation transcript saved to `exercise1_sample_output.txt` (or linked notebook cell).
- [x] Exercise 2 run summary saved to `exercise2_sample_output.txt` and JSON report archived.
- [x] README updated with bullet-point findings for each exercise (accuracy, hallucinations, runtime observations).
- [x] Additional experiments (dog detection, frame saving) appended as sub-bullets.


