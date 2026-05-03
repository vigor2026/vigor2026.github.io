#!/usr/bin/env python3
"""Blur faces using manual keyframes and interpolation.

This is for cases where automatic tracking is unreliable. You annotate face
boxes on a few frames, then the script linearly interpolates the boxes between
those frames and blurs only those regions.

Install once if needed:
  /usr/bin/python3 -m pip install opencv-python

Run:
  /usr/bin/python3 scripts/blur_factorized_faces_keyframes.py
"""

from __future__ import annotations

import argparse
import bisect
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    print(
        "Missing dependency: cv2\n\n"
        "Install OpenCV, then rerun:\n"
        "  /usr/bin/python3 -m pip install opencv-python\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


DEFAULT_INPUT = Path("static/videos/factorized/factorized_data_gen.mov")
DEFAULT_OUTPUT = Path("static/videos/factorized/factorized_data_gen_blurred.mp4")
DEFAULT_KEYFRAMES = Path("static/videos/factorized/factorized_face_keyframes.json")

Box = tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blur faces with manual keyframe interpolation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--keyframes", type=Path, default=DEFAULT_KEYFRAMES)
    parser.add_argument("--step-seconds", type=float, default=2.0, help="Annotation spacing.")
    parser.add_argument("--blur-kernel", type=int, default=31, help="Odd Gaussian blur kernel.")
    parser.add_argument("--padding", type=float, default=0.25, help="Extra padding around each box.")
    parser.add_argument("--reuse-keyframes", action="store_true", help="Reuse saved JSON annotations.")
    return parser.parse_args()


def odd(value: int) -> int:
    value = max(3, value)
    return value if value % 2 else value + 1


def clamp_box(box: Box, width: int, height: int) -> Box:
    x, y, w, h = box
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return x, y, w, h


def expand_box(box: Box, padding: float, width: int, height: int) -> Box:
    x, y, w, h = box
    px = int(w * padding)
    py = int(h * padding)
    return clamp_box((x - px, y - py, w + 2 * px, h + 2 * py), width, height)


def read_frame(cap, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def draw_text(frame, text: str, y: int) -> None:
    cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 1, cv2.LINE_AA)


def annotate_frame(frame, frame_index: int, total_frames: int, previous: list[Box]) -> list[Box]:
    window = "Annotate face boxes"
    boxes: list[Box] = []
    drawing = False
    start: tuple[int, int] | None = None

    def redraw(current: tuple[int, int] | None = None) -> None:
        preview = frame.copy()
        for x, y, w, h in previous:
            cv2.rectangle(preview, (x, y), (x + w, y + h), (160, 160, 160), 1)
        for x, y, w, h in boxes:
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if drawing and start and current:
            cv2.rectangle(preview, start, current, (0, 180, 255), 2)

        draw_text(preview, f"Frame {frame_index}/{total_frames - 1}", 34)
        draw_text(preview, "Drag boxes. q/Enter=save, u=use previous, r=reset, Esc=cancel", 68)
        cv2.imshow(window, preview)

    def on_mouse(event, x, y, _flags, _param) -> None:
        nonlocal drawing, start
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start = (x, y)
            redraw((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            redraw((x, y))
        elif event == cv2.EVENT_LBUTTONUP and drawing and start:
            drawing = False
            x1, y1 = start
            left, right = sorted((x1, x))
            top, bottom = sorted((y1, y))
            if right - left > 2 and bottom - top > 2:
                boxes.append((left, top, right - left, bottom - top))
            start = None
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), ord("\r"), ord("\n"), 13, 10):
            break
        if key == ord("u"):
            boxes = previous[:]
            break
        if key == ord("r"):
            boxes.clear()
            redraw()
        if key == 27:
            cv2.destroyWindow(window)
            raise SystemExit("Cancelled annotation.")

    cv2.destroyWindow(window)
    return boxes


def keyframe_indices(total_frames: int, fps: float, step_seconds: float) -> list[int]:
    step = max(1, int(round(fps * step_seconds)))
    last_frame = max(0, total_frames - 2)
    indices = list(range(0, last_frame + 1, step))
    if last_frame not in indices:
        indices.append(last_frame)
    return indices


def collect_keyframes(
    cap,
    total_frames: int,
    fps: float,
    step_seconds: float,
    keyframes_path: Path,
) -> dict[int, list[Box]]:
    annotations: dict[int, list[Box]] = {}
    previous: list[Box] = []
    for frame_index in keyframe_indices(total_frames, fps, step_seconds):
        frame = read_frame(cap, frame_index)
        if frame is None:
            print(f"Skipping unreadable frame {frame_index}", file=sys.stderr)
            continue
        print(f"Annotating frame {frame_index}/{total_frames - 1}", file=sys.stderr)
        boxes = annotate_frame(frame, frame_index, total_frames, previous)
        annotations[frame_index] = boxes
        if boxes:
            previous = boxes
        save_keyframes(keyframes_path, annotations)
    return annotations


def save_keyframes(path: Path, annotations: dict[int, list[Box]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {str(frame): [list(box) for box in boxes] for frame, boxes in annotations.items()}
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_keyframes(path: Path) -> dict[int, list[Box]]:
    data = json.loads(path.read_text())
    return {int(frame): [tuple(box) for box in boxes] for frame, boxes in data.items()}


def box_at(frame_index: int, annotations: dict[int, list[Box]]) -> list[Box]:
    frames = sorted(annotations)
    position = bisect.bisect_left(frames, frame_index)

    if position == 0:
        return annotations[frames[0]]
    if position == len(frames):
        return annotations[frames[-1]]

    left_frame = frames[position - 1]
    right_frame = frames[position]
    left_boxes = annotations[left_frame]
    right_boxes = annotations[right_frame]
    count = min(len(left_boxes), len(right_boxes))
    if count == 0:
        return left_boxes or right_boxes

    alpha = (frame_index - left_frame) / max(1, right_frame - left_frame)
    boxes: list[Box] = []
    for left, right in zip(left_boxes[:count], right_boxes[:count]):
        boxes.append(tuple(round(left[i] + (right[i] - left[i]) * alpha) for i in range(4)))
    return boxes


def blur_boxes(frame, boxes: list[Box], padding: float, kernel: int) -> None:
    height, width = frame.shape[:2]
    for box in boxes:
        x, y, w, h = expand_box(box, padding, width, height)
        roi = frame[y : y + h, x : x + w]
        if roi.size:
            frame[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (kernel, kernel), 0)


def mux_audio(input_path: Path, video_without_audio: Path, output_path: Path) -> None:
    if not shutil.which("ffmpeg"):
        video_without_audio.replace(output_path)
        print("ffmpeg not found; wrote video without audio.", file=sys.stderr)
        return

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_without_audio),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )


def render(input_path: Path, output_path: Path, annotations: dict[int, list[Box]], blur_kernel: int, padding: float) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video = Path(temp_dir) / "keyframed_blur_no_audio.mp4"
        writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise SystemExit(f"Could not create temporary video: {temp_video}")

        kernel = odd(blur_kernel)
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            blur_boxes(frame, box_at(frame_index, annotations), padding, kernel)
            writer.write(frame)
            frame_index += 1
            if total and frame_index % 60 == 0:
                print(f"Processed {frame_index}/{total} frames...")

        cap.release()
        writer.release()
        mux_audio(input_path, temp_video, output_path)


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser()
    output_path = args.output.expanduser()
    keyframes_path = args.keyframes.expanduser()

    if not input_path.exists():
        print(f"Input video not found: {input_path}", file=sys.stderr)
        return 1

    if args.reuse_keyframes:
        annotations = load_keyframes(keyframes_path)
    else:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Could not open video: {input_path}", file=sys.stderr)
            return 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        annotations = collect_keyframes(cap, total, fps, args.step_seconds, keyframes_path)
        cap.release()
        print(f"Saved keyframes: {keyframes_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    render(input_path, output_path, annotations, args.blur_kernel, args.padding)
    print(f"Wrote blurred video: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
