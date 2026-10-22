#!/usr/bin/env python3
"""Capture one frame per resolution from the same camera and tile them into a single image.

Run on the Orange Pi:

    python3 camera_resolution_grid.py --camera 0
    python3 camera_resolution_grid.py --camera 0 --resolutions 640x360,640x480,1280x720
    python3 camera_resolution_grid.py --camera /dev/video2 --fourcc MJPG --out grid.jpg

Notes:
  * The camera is reopened for every resolution. V4L2 carries the previous
    format state over, and changing the size on the same handle often fails
    silently.
  * cap.set() is only a request, not a command. Each cell shows
    "requested -> actual", and a mismatch is drawn in orange.
  * Frames are scaled to fit and padded, never stretched, so different aspect
    ratios stay comparable.
  * The real capture frame rate is measured for every mode.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# The Windows console defaults to cp950 and raises on non-ASCII output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

DEFAULT_RESOLUTIONS = [
    (320, 180), (640, 360),
    (640, 480), (800, 600),
    (960, 540), (1280, 720),
    (1280, 960), (1920, 1080),
]

CELL_W, CELL_H = 480, 300
LABEL_H = 26
PAD = 6


def fourcc_str(value: int) -> str:
    value = int(value)
    text = "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))
    return text if text.isprintable() and text.strip() else "????"


def open_camera(source, width: int, height: int, fourcc: str | None):
    """Open a fresh capture and request the given mode.

    Order matters on V4L2: set the pixel format first, then the size, then the
    size again, because some drivers reset the size when the format is accepted.
    """
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    try:
        cap = cv2.VideoCapture(source, backend)
    except Exception:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    if fourcc:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        except Exception:
            pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def capture_mode(source, width: int, height: int, fourcc: str | None,
                 frames: int, warmup: float):
    """Grab one frame at the requested mode and report what the driver really gave."""
    cap = open_camera(source, width, height, fourcc)
    if cap is None:
        return None

    reported_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    reported_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fourcc = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))

    # Flush stale buffers before sampling.
    deadline = time.monotonic() + warmup
    while time.monotonic() < deadline:
        cap.read()

    frame = None
    count = 0
    start = time.perf_counter()
    for _ in range(max(1, frames)):
        ok, got = cap.read()
        if ok and got is not None and got.size:
            frame = got
            count += 1
    elapsed = time.perf_counter() - start
    fps = count / elapsed if elapsed > 0 and count else 0.0
    cap.release()

    if frame is None:
        return None
    return {
        "requested": (width, height),
        "frame": frame,
        "actual": (frame.shape[1], frame.shape[0]),
        "reported": (reported_w, reported_h),
        "fourcc": actual_fourcc,
        "fps": fps,
    }


def fit_letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale to fit and pad with grey - never stretch."""
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return np.full((height, width, 3), 40, np.uint8)
    scale = min(width / w, height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 40, np.uint8)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture one frame per resolution and tile them into one image")
    parser.add_argument("--camera", default="2", help="camera index or /dev/videoN")
    parser.add_argument("--fourcc", default="MJPG",
                        help="pixel format to request (MJPG/YUYV); empty string to skip")
    parser.add_argument("--resolutions", default=None,
                        help="comma separated WxH list, e.g. 640x360,1280x720")
    parser.add_argument("--frames", type=int, default=10,
                        help="frames to time per mode")
    parser.add_argument("--warmup", type=float, default=0.2,
                        help="seconds to flush stale buffers before sampling")
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("camera_resolutions.jpg"))
    parser.add_argument("--rotate", choices=["none", "90", "180", "270"], default="none")
    args = parser.parse_args()

    if args.resolutions:
        modes = []
        for token in args.resolutions.split(","):
            token = token.strip().lower()
            if not token:
                continue
            width, height = token.split("x")
            modes.append((int(width), int(height)))
    else:
        modes = DEFAULT_RESOLUTIONS

    try:
        source = int(args.camera)
    except ValueError:
        source = args.camera
    fourcc = args.fourcc.strip() or None

    print(f"camera={source}  fourcc={fourcc or '-'}  modes={len(modes)}")
    print(f"{'requested':>12} {'actual':>12} {'reported':>12} {'fcc':>5} {'fps':>7}")

    results = []
    for width, height in modes:
        result = capture_mode(source, width, height, fourcc, args.frames, args.warmup)
        if result is None:
            print(f"{f'{width}x{height}':>12} {'<failed>':>12}")
            results.append({"requested": (width, height), "frame": None})
            continue
        actual_w, actual_h = result["actual"]
        reported_w, reported_h = result["reported"]
        print(f"{f'{width}x{height}':>12} {f'{actual_w}x{actual_h}':>12} "
              f"{f'{reported_w}x{reported_h}':>12} {result['fourcc']:>5} {result['fps']:7.1f}")
        results.append(result)

    if not any(r.get("frame") is not None for r in results):
        print("\nNo frames captured: check the camera index, whether another "
              "process holds it, and permissions.")
        return 1

    cols = max(1, args.cols)
    rows = (len(results) + cols - 1) // cols
    cell_h = LABEL_H + CELL_H
    sheet = np.full((rows * cell_h, cols * CELL_W, 3), 24, np.uint8)

    for index, result in enumerate(results):
        row, col = divmod(index, cols)
        ox, oy = col * CELL_W, row * cell_h
        req_w, req_h = result["requested"]

        if result.get("frame") is None:
            cv2.rectangle(sheet, (ox + 2, oy + 2), (ox + CELL_W - 2, oy + cell_h - 2),
                          (0, 0, 140), 1)
            cv2.putText(sheet, f"{req_w}x{req_h}  <failed>", (ox + 8, oy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 255), 1, cv2.LINE_AA)
            continue

        frame = result["frame"]
        if args.rotate == "90":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == "180":
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif args.rotate == "270":
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        actual_w, actual_h = result["actual"]
        matched = (actual_w, actual_h) == (req_w, req_h)
        colour = (80, 220, 80) if matched else (0, 170, 255)
        label = (f"req {req_w}x{req_h} -> got {actual_w}x{actual_h} "
                 f"{result['fourcc']} {result['fps']:.0f}fps")

        cell = fit_letterbox(frame, CELL_W - 2 * PAD, CELL_H - 2 * PAD)
        sheet[oy + LABEL_H + PAD:oy + LABEL_H + PAD + cell.shape[0],
              ox + PAD:ox + PAD + cell.shape[1]] = cell
        cv2.putText(sheet, label, (ox + 8, oy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        cv2.rectangle(sheet, (ox, oy), (ox + CELL_W - 1, oy + cell_h - 1),
                      colour, 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite cannot open non-ASCII paths on Windows; encode then write bytes.
    ok, buffer = cv2.imencode(args.out.suffix or ".jpg", sheet)
    if not ok:
        print(f"Failed to encode the sheet: {args.out}")
        return 1
    buffer.tofile(str(args.out))
    print(f"\nSaved: {args.out.resolve()}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print("Green = actual matches requested, orange = the driver picked another mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
