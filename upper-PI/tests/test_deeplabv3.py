"""Run the DeepLabV3 RKNN model with a camera on RK3588."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


DEFAULT_MODEL = Path(__file__).with_name("deeplabv3_mobilenet_roboflow_voc_640x480_native_fp16.rknn")


def parse_args():
    parser = argparse.ArgumentParser(description="DeepLabV3 RKNN camera inference")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--camera", default="0", help="camera index or video path")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="foreground probability threshold for a two-class model")
    parser.add_argument("--flip", choices=("none", "-1", "0", "1"), default="none")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def load_runtime(model_path):
    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(str(model_path)) != 0:
        raise RuntimeError(f"failed to load RKNN model: {model_path}")

    core_mask = getattr(RKNNLite, "NPU_CORE_0_1_2", RKNNLite.NPU_CORE_AUTO)
    if rknn.init_runtime(core_mask=core_mask) != 0:
        rknn.release()
        raise RuntimeError("failed to initialize RKNN runtime")
    return rknn


def prepare_input(frame_bgr, width, height):
    """Return contiguous NHWC uint8 RGB input with batch size 1."""
    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb[np.newaxis, ...], dtype=np.uint8)


def decode_output(output, threshold):
    """Decode NCHW/NHWC logits or a 2-D class mask."""
    array = np.asarray(output)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise RuntimeError(f"only batch=1 is supported, got {array.shape}")
        array = array[0]

    if array.ndim == 2:
        return array.astype(np.uint8), None

    if array.ndim != 3:
        raise RuntimeError(f"unsupported output shape: {array.shape}")

    if array.shape[0] <= 64:  # NCHW
        logits = array.astype(np.float32, copy=False)
        if logits.shape[0] == 2:
            delta = np.clip(logits[1] - logits[0], -30.0, 30.0)
            foreground_probability = 1.0 / (1.0 + np.exp(-delta))
            return (foreground_probability >= threshold).astype(np.uint8), foreground_probability
        return np.argmax(logits, axis=0).astype(np.uint8), None

    if array.shape[-1] <= 64:  # NHWC
        logits = array.astype(np.float32, copy=False)
        if logits.shape[-1] == 2:
            delta = np.clip(logits[..., 1] - logits[..., 0], -30.0, 30.0)
            foreground_probability = 1.0 / (1.0 + np.exp(-delta))
            return (foreground_probability >= threshold).astype(np.uint8), foreground_probability
        return np.argmax(logits, axis=-1).astype(np.uint8), None

    raise RuntimeError(f"cannot determine output layout: {array.shape}")


def make_overlay(frame, mask, fps, probability, threshold):
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(
            mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    color = np.zeros_like(frame)
    color[mask == 1] = (0, 255, 0)
    overlay = cv2.addWeighted(frame, 0.65, color, 0.35, 0.0)
    foreground_percent = float(np.mean(mask == 1) * 100.0)

    label = f"FPS {fps:.1f} | foreground {foreground_percent:.1f}% | threshold {threshold:.2f}"
    if probability is not None:
        label += f" | max {float(probability.max()):.2f}"
    cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay, foreground_percent


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")

    rknn = load_runtime(model_path)
    source = int(args.camera) if args.camera.isdigit() else args.camera
    backend = cv2.CAP_V4L2 if isinstance(source, int) else cv2.CAP_ANY
    camera = cv2.VideoCapture(source, backend)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_FPS, args.camera_fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        rknn.release()
        raise RuntimeError(f"failed to open camera/video: {source}")

    print(f"model: {model_path}")
    print(f"input: RGB uint8 NHWC, 1x{args.height}x{args.width}x3")
    print("press Q or Esc to exit")

    last_time = time.perf_counter()
    smooth_fps = 0.0
    frame_count = 0
    output_shape_printed = False

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("warning: failed to read camera frame")
                continue

            if args.flip != "none":
                frame = cv2.flip(frame, int(args.flip))

            model_input = prepare_input(frame, args.width, args.height)
            outputs = rknn.inference(inputs=[model_input], data_format=["nhwc"])
            if not outputs:
                raise RuntimeError("RKNN returned no output")

            if not output_shape_printed:
                output = np.asarray(outputs[0])
                print(f"output: shape={output.shape}, dtype={output.dtype}")
                output_shape_printed = True

            mask, probability = decode_output(outputs[0], args.threshold)

            now = time.perf_counter()
            current_fps = 1.0 / max(now - last_time, 1e-9)
            last_time = now
            smooth_fps = current_fps if smooth_fps == 0.0 else smooth_fps * 0.9 + current_fps * 0.1

            overlay, foreground_percent = make_overlay(
                frame, mask, smooth_fps, probability, args.threshold
            )
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"FPS={smooth_fps:.1f}, foreground={foreground_percent:.2f}%")

            if args.no_display:
                continue

            cv2.imshow("DeepLabV3 RKNN", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        rknn.release()
        print("resources released")


if __name__ == "__main__":
    main()
