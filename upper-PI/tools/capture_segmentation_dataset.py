"""低位摄像头 16:9、640x360 语义分割数据集采集工具。"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2


def parse_flip(value: str):
    value = str(value).strip().lower()
    if value in ("none", "off", "no", "raw"):
        return None
    parsed = int(value)
    if parsed not in (-1, 0, 1):
        raise argparse.ArgumentTypeError("--flip 只能是 none、-1、0 或 1")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="2", help="摄像头编号或设备路径")
    parser.add_argument("--output", default="datasets")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip", type=parse_flip, default=None)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--auto", action="store_true", help="启动后立即自动采集")
    args = parser.parse_args()
    if (args.width, args.height) != (1280, 720):
        raise ValueError("该采集脚本固定保存 1280x720 的 16:9 数据；请不要修改 --width/--height")

    session = datetime.now().strftime("road_%Y%m%d_%H%M%S")
    image_dir = Path(args.output) / session / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    source = int(args.camera) if str(args.camera).isdigit() else args.camera
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {source}")

    print(f"[DATASET] 保存目录: {image_dir}")
    print("[DATASET] 摄像头采集 16:9，保存 1280x720；s/空格=拍一张 a=自动 q/Esc=退出")
    auto_capture, last_saved, saved_count = args.auto, 0.0, 0

    def save_frame(raw_frame):
        nonlocal saved_count, last_saved
        if raw_frame.shape[:2] != (args.height, args.width):
            raw_frame = cv2.resize(raw_frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = image_dir / f"road_{stamp}_{saved_count:06d}.jpg"
        ok = cv2.imwrite(str(path), raw_frame,
                         [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, args.quality))])
        if ok:
            saved_count += 1; last_saved = time.monotonic(); print(f"[SAVE] {saved_count}: {path.name}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.03); continue
            if args.flip is not None: frame = cv2.flip(frame, args.flip)
            if frame.shape[:2] != (args.height, args.width):
                preview = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            else: preview = frame
            now = time.monotonic()
            if auto_capture and now - last_saved >= max(0.05, args.interval): save_frame(frame)
            display = preview.copy(); mode = "AUTO" if auto_capture else "MANUAL"
            cv2.putText(display, f"{mode} saved={saved_count} 1280x720", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .58, (0,255,0) if auto_capture else (0,255,255), 2)
            cv2.imshow("Dataset Capture 1280x720", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27): break
            if key in (ord("s"), ord(" ")): save_frame(frame)
            elif key == ord("a"):
                auto_capture = not auto_capture; last_saved = 0.0
    finally:
        cap.release(); cv2.destroyAllWindows()
        print(f"[DATASET] 结束，共保存 {saved_count} 张图片")
        print(f"[DATASET] 数据集目录: {image_dir.parent}")


if __name__ == "__main__": main()
