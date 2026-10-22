#!/usr/bin/env python3
"""纯视觉巡线与路口识别测试，不连接串口，也不控制底盘。"""

from __future__ import annotations

import argparse
import os
import time

import cv2

from line_detect import DEFAULT_MODEL, LineTracker


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no"}


def parse_camera_source(value: object) -> int | str:
    """支持摄像头数字编号或 /dev/v4l 下的设备路径。"""
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def parse_flip_code(value: object) -> int | None:
    """解析 OpenCV 翻转参数；none 表示不翻转。"""
    text = str(value).strip().lower()
    if text in {"none", "off", "false", "no"}:
        return None
    code = int(text)
    if code not in (-1, 0, 1):
        raise ValueError("翻转参数只能是 -1、0、1 或 none")
    return code


def open_camera(source: int | str, width: int, height: int, fps: int):
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    try:
        camera = cv2.VideoCapture(source, backend)
    except Exception:
        camera = cv2.VideoCapture(source)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def wait_for_camera(camera, timeout: float) -> bool:
    """Match full_map_walk_test: wait until the camera yields a valid frame."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if camera.isOpened():
            ok, frame = camera.read()
            if ok and frame is not None and frame.size:
                return True
        time.sleep(0.05)
    return False


def read_road(camera, tracker: LineTracker, flip_code: int | None):
    """Read, orient and infer one frame using the full-map call sequence."""
    ok, frame = camera.read()
    if not ok or frame is None or frame.size == 0:
        return None
    if flip_code in (-1, 0, 1):
        frame = cv2.flip(frame, flip_code)
    return tracker.get_road_info(frame)


def run(args: argparse.Namespace) -> int:
    try:
        source = parse_camera_source(args.camera)
        flip_code = parse_flip_code(args.flip)
    except ValueError as exc:
        print(f"[配置错误] {exc}")
        return 2

    cross_count = 0
    tracker = None
    camera = None

    try:
        debug_mode = not args.no_window and env_bool("LINE_DEBUG", True)
        tracker = LineTracker(model_path=args.model, debug_mode=debug_mode)
        tracker.set_junction_mode(args.junction_mode)
        camera = open_camera(source, args.width, args.height, args.fps)
        if not wait_for_camera(camera, args.camera_startup_timeout):
            print(
                f"[摄像头错误] {args.camera_startup_timeout:.1f} 秒内"
                f"未从摄像头 {source} 取得有效画面"
            )
            return 1

        print("[视觉测试] 纯视觉模式：不连接串口，不控制底盘")
        print(f"[视觉测试] 语义分割模型={tracker.model_path}")
        print(
            f"[视觉测试] 摄像头={source}，翻转={flip_code}，"
            f"分辨率={args.width}x{args.height}，帧率={args.fps}"
        )
        if tracker.debug_mode:
            print("[视觉测试] 按 q、Esc 或 Ctrl+C 退出")
        else:
            print("[视觉测试] 无窗口模式，按 Ctrl+C 退出")

        started_at = time.monotonic()
        fps_started_at = started_at
        processed_frames = 0
        displayed_fps = 0.0
        camera_misses = 0
        cross_active = False

        while True:
            road = read_road(camera, tracker, flip_code)
            if road is None:
                camera_misses += 1
                if camera_misses >= args.max_camera_misses:
                    print("\n[摄像头错误] 连续取帧失败次数过多")
                    return 3
                time.sleep(0.02)
                continue

            camera_misses = 0
            road_type, offset = road
            processed_frames += 1

            now = time.monotonic()
            fps_elapsed = now - fps_started_at
            if fps_elapsed >= 1.0:
                displayed_fps = processed_frames / fps_elapsed
                processed_frames = 0
                fps_started_at = now

            if road_type == "CROSS":
                if not cross_active:
                    cross_count += 1
                    print(f"\n[路口] 已确认第 {cross_count} 个路口")
                cross_active = True
            else:
                cross_active = False

            status_text = {
                "LINE": "巡线",
                "CROSS": "路口",
                "WAITING": "等待画面",
            }.get(road_type, road_type)

            print(
                f"[视觉] 状态={status_text:<5} 偏差={offset:>4} "
                f"路口确认={tracker.cross_counter}/"
                f"{tracker.cross_confirm_threshold} "
                f"路口数={cross_count} 帧率={displayed_fps:>4.1f}",
                end="\r",
                flush=True,
            )

            if tracker.debug_mode and tracker.last_key in (ord("q"), 27):
                print("\n[视觉测试] 收到退出按键")
                return 0
            if args.duration > 0 and now - started_at >= args.duration:
                print(f"\n[视觉测试] 已达到设定时长：{args.duration:.1f} 秒")
                return 0
    except KeyboardInterrupt:
        print("\n[视觉测试] 用户已停止测试")
        return 0
    finally:
        if camera is not None:
            camera.release()
        if tracker is not None:
            tracker.release()
        cv2.destroyAllWindows()
        print(f"[视觉测试] 摄像头和识别器已释放；确认路口数={cross_count}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="仅测试巡线视觉与路口识别，不会控制底盘"
    )
    parser.add_argument(
        "--camera",
        default=os.environ.get("NAV_CAMERA", "0"),
        help="摄像头编号或设备路径；默认读取 NAV_CAMERA 或使用 0",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DEEPLAB_MODEL") or str(DEFAULT_MODEL),
        help="DeepLabV3 RKNN 语义分割模型；默认读取 DEEPLAB_MODEL 或使用内置路径",
    )
    parser.add_argument(
        "--flip",
        default=os.environ.get("NAV_CAMERA_FLIP", "none"),
        help="画面翻转：默认读取 NAV_CAMERA_FLIP 或不翻转；可选 -1、0、1、none",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--junction-mode",
        choices=("NORMAL", "L"),
        default=os.environ.get("VISION_JUNCTION_MODE", "NORMAL").upper(),
        help="路口检测模式：NORMAL 或 L；默认 NORMAL",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="测试时长（秒）；0 表示持续运行，直到手动停止",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="不打开 OpenCV 画面窗口；使用 Ctrl+C 退出",
    )
    parser.add_argument(
        "--camera-startup-timeout",
        type=float,
        default=float(os.environ.get("CAMERA_STARTUP_TIMEOUT", "10")),
        help="等待摄像头产生首个有效画面的最长秒数",
    )
    parser.add_argument(
        "--max-camera-misses",
        type=int,
        default=int(os.environ.get("FULL_MAP_MAX_CAMERA_MISSES", "10")),
    )
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("宽度、高度和帧率必须大于 0")
    if args.duration < 0:
        parser.error("测试时长不能为负数")
    if args.camera_startup_timeout <= 0:
        parser.error("摄像头启动超时必须大于 0")
    if args.max_camera_misses < 1:
        parser.error("最大连续丢帧次数至少为 1")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
