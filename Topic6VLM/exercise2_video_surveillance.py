"""Exercise 2 – Video-surveillance agent powered by Llava via Ollama.

The script samples frames from a video (or webcam), sends them to Llava with a
structured prompt, and logs when tracked entities (people, cats, dogs, etc.)
appear or disappear from the scene.
"""
from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import ollama

PROMPT_TEMPLATE = """You are monitoring security footage. Inspect the provided frame and report
ONLY the entities listed below. Respond with valid JSON using this schema:
{{
    "counts": {{{counts_clause}}},
    "description": "One concise sentence about notable activity"
}}
- The counts MUST be integers >= 0.
- If a category is absent, set its value to 0.
- Do not add extra keys. Return JSON only.
Entities to track: {entity_list}.
"""


@dataclass
class FrameDetection:
    timestamp: float
    counts: Dict[str, int]
    description: str
    raw_model_output: str
    frame_path: Optional[str] = None


@dataclass
class EntityTracker:
    active: bool = False
    start_time: Optional[float] = None
    max_count: int = 0


def bgr_frame_to_b64(frame) -> str:
    success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        raise ValueError("Failed to encode frame as JPEG")
    return base64.b64encode(buffer).decode("utf-8")


def query_llava(image_b64: str, entities: List[str], dry_run: bool = False) -> Tuple[Dict[str, int], str, str]:
    if dry_run:
        return {item: 0 for item in entities}, "DRY RUN: no model call", "DRY RUN"

    counts_clause = ", ".join(f'"{label}": 0' for label in entities)
    prompt = PROMPT_TEMPLATE.format(
        counts_clause=counts_clause,
        entity_list=", ".join(entities),
    )
    response = ollama.chat(
        model="llava",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
    )
    text = response["message"]["content"].strip()
    try:
        json_start = text.index("{")
        json_end = text.rindex("}") + 1
        parsed = json.loads(text[json_start:json_end])
    except (ValueError, json.JSONDecodeError):
        parsed = {"counts": {key: 0 for key in entities}, "description": text}

    counts = parsed.get("counts", {})
    normalized = {k: int(max(0, counts.get(k, 0))) for k in entities}
    description = parsed.get("description", text)
    return normalized, description, text


def format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{secs:05.2f}"


def update_trackers(
    trackers: Dict[str, EntityTracker],
    events: Dict[str, List[Dict[str, str]]],
    counts: Dict[str, int],
    timestamp: float,
) -> None:
    for label, count in counts.items():
        tracker = trackers.setdefault(label, EntityTracker())
        history = events.setdefault(label, [])
        if count > 0:
            if not tracker.active:
                tracker.active = True
                tracker.start_time = timestamp
                tracker.max_count = count
            else:
                tracker.max_count = max(tracker.max_count, count)
        else:
            if tracker.active:
                history.append(
                    {
                        "entity": label,
                        "entered_seconds": tracker.start_time or 0.0,
                        "entered": format_timestamp(tracker.start_time or 0.0),
                        "exited_seconds": timestamp,
                        "exited": format_timestamp(timestamp),
                        "max_count": tracker.max_count,
                    }
                )
                tracker.active = False
                tracker.start_time = None
                tracker.max_count = 0


def finalize_events(trackers: Dict[str, EntityTracker], events: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
    for label, tracker in trackers.items():
        if tracker.active:
            events.setdefault(label, []).append(
                {
                    "entity": label,
                    "entered_seconds": tracker.start_time or 0.0,
                    "entered": format_timestamp(tracker.start_time or 0.0),
                    "exited_seconds": None,
                    "exited": "(still present)",
                    "max_count": tracker.max_count,
                }
            )
    flattened: List[Dict[str, str]] = []
    for history in events.values():
        flattened.extend(history)
    return flattened


def sample_and_detect(
    capture: cv2.VideoCapture,
    args: argparse.Namespace,
    entities: List[str],
) -> Tuple[List[FrameDetection], List[Dict[str, str]]]:
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        fps = 30.0
    interval_frames = max(1, int(round(fps * args.frame_interval)))
    trackers: Dict[str, EntityTracker] = {label: EntityTracker() for label in entities}
    event_log: Dict[str, List[Dict[str, str]]] = {}
    detections: List[FrameDetection] = []

    frame_dir: Optional[Path] = None
    if args.save_frames:
        frame_dir = Path(args.save_frames)
        frame_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    processed = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        if frame_index % interval_frames != 0:
            frame_index += 1
            continue
        timestamp = frame_index / fps
        frame_path = None
        if frame_dir is not None:
            frame_path = str(frame_dir / f"frame_{processed:05d}.jpg")
            cv2.imwrite(frame_path, frame)

        image_b64 = bgr_frame_to_b64(frame)
        counts, description, raw_text = query_llava(image_b64, entities, args.dry_run)
        update_trackers(trackers, event_log, counts, timestamp)

        detections.append(
            FrameDetection(
                timestamp=timestamp,
                counts=counts,
                description=description,
                raw_model_output=raw_text,
                frame_path=frame_path,
            )
        )
        processed += 1
        frame_index += 1

    capture.release()
    return detections, finalize_events(trackers, event_log)


def print_summary(events: List[Dict[str, str]], detections: List[FrameDetection]) -> None:
    print("\n=== Surveillance Summary ===")
    if not events:
        print("No tracked entities detected.")
    for event in sorted(events, key=lambda item: (item.get("entered_seconds") or 0.0)):
        print(
            f"- {event['entity'].title()} | entered {event['entered']} | "
            f"exited {event['exited']} | max count {event['max_count']}"
        )
    print("\nProcessed frames: ", len(detections))


def save_report(path: Optional[str], metadata: Dict[str, object], detections: List[FrameDetection], events: List[Dict[str, str]]) -> None:
    if not path:
        return
    payload = {
        "metadata": metadata,
        "detections": [
            {
                "timestamp": detection.timestamp,
                "timestamp_label": format_timestamp(detection.timestamp),
                "counts": detection.counts,
                "description": detection.description,
                "frame_path": detection.frame_path,
                "raw_model_output": detection.raw_model_output,
            }
            for detection in detections
        ],
        "events": events,
    }
    report_path = Path(path)
    if report_path.parent and not report_path.parent.exists():
        report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Report written to {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Topic 6 – Video Surveillance Agent")
    parser.add_argument("--video", type=str, help="Path to video file", default=None)
    parser.add_argument("--webcam", action="store_true", help="Use webcam instead of video file")
    parser.add_argument("--frame-interval", type=float, default=2.0, help="Seconds between sampled frames")
    parser.add_argument("--entities", nargs="+", default=["person", "cat", "dog"], help="Entities to track")
    parser.add_argument("--output", type=str, help="Optional JSON report path", default=None)
    parser.add_argument("--save-frames", type=str, help="Directory to store sampled frames", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Skip Ollama calls and return zeros (debug only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.webcam and not args.video:
        raise SystemExit("Provide --video path or pass --webcam to stream from camera.")

    if args.webcam:
        capture = cv2.VideoCapture(0)
        metadata = {"source": "webcam"}
    else:
        video_path = Path(args.video).expanduser()
        if not video_path.exists():
            raise SystemExit(f"Video file not found: {video_path}")
        capture = cv2.VideoCapture(str(video_path))
        metadata = {"source": str(video_path)}

    detections, events = sample_and_detect(capture, args, args.entities)
    metadata.update({"frame_interval_s": args.frame_interval, "entities": args.entities})
    print_summary(events, detections)
    save_report(args.output, metadata, detections, events)


if __name__ == "__main__":
    main()
