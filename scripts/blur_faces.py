#!/usr/bin/env python3
"""Blur faces in a video using OpenCV Haar cascades.

Usage:
  python3 scripts/blur_faces.py --input input.mp4 --output output.mp4
"""

import argparse
import sys
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Blur faces in a video file.")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output video")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Haar scale factor")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Haar min neighbors")
    parser.add_argument("--min-size", type=int, default=25, help="Min face size (pixels)")
    parser.add_argument("--blur-kernel", type=int, default=61, help="Odd kernel size for Gaussian blur")
    return parser.parse_args()


def ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def primary_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def is_drastic_change(prev, curr) -> bool:
    px, py, pw, ph = prev
    cx, cy, cw, ch = curr
    prev_cx = px + pw / 2.0
    prev_cy = py + ph / 2.0
    curr_cx = cx + cw / 2.0
    curr_cy = cy + ch / 2.0
    dx = curr_cx - prev_cx
    dy = curr_cy - prev_cy
    dist = (dx * dx + dy * dy) ** 0.5
    size_ref = max(pw, ph)
    return dist > size_ref * 0.7


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: cannot open input video: {args.input}", file=sys.stderr)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: cannot open output video: {args.output}", file=sys.stderr)
        return 1

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Error: failed to load Haar cascade", file=sys.stderr)
        return 1

    blur_k = ensure_odd(max(3, args.blur_kernel))

    last_faces = []
    last_seen = 0
    hold_frames = int(max(1, fps * 1.5))  # hold last detections for ~1.5s
    expand = 0.5  # expand box by 30% on each side

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=args.scale_factor,
            minNeighbors=args.min_neighbors,
            minSize=(args.min_size, args.min_size),
        )

        if len(faces) > 0:
            if len(last_faces) > 0:
                prev = primary_face(last_faces)
                curr = primary_face(faces)
                if prev is not None and curr is not None and is_drastic_change(prev, curr):
                    last_seen += 1
                    faces = last_faces
                else:
                    last_faces = faces
                    last_seen = 0
            else:
                last_faces = faces
                last_seen = 0
        else:
            last_seen += 1
            if last_faces is not None and last_seen <= hold_frames:
                faces = last_faces

        for (x, y, w, h) in faces:
            ex = int(w * expand)
            ey = int(h * expand)
            x1 = max(0, x - ex)
            y1 = max(0, y - ey)
            x2 = min(width, x + w + ex)
            y2 = min(height, y + h + ey)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            blurred = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)
            frame[y1:y2, x1:x2] = blurred

        out.write(frame)

    cap.release()
    out.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
