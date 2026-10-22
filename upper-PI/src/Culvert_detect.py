#!/usr/bin/env python3
"""高位摄像头涵洞亮度/纹理检测及下位机扫描联动测试。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SERIAL = os.environ.get("CHASSIS_PORT", "/dev/ttyUSB0")






class CulvertBrightnessDetector:
    """通过涵洞内画面的低纹理、亮度均匀性变化触发检测。"""

    def __init__(self, consistency_threshold=0.12, edge_threshold=0.012,
                 min_consecutive=3):
        self.consistency_threshold = float(consistency_threshold)
        self.edge_threshold = float(edge_threshold)
        self.min_consecutive = max(2, int(min_consecutive))
        self.candidate_streak = 0
        self.triggered = False
        self.last_consistency = 1.0
        self.last_edge_density = 1.0

    def reset(self):
        self.candidate_streak = 0
        self.triggered = False

    def inspect(self, frame):
        if frame is None or frame.size == 0 or self.triggered:
            return False, []
        height, width = frame.shape[:2]
        roi = frame[int(height * 0.10):int(height * 0.90),
                    int(width * 0.05):int(width * 0.95)]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mean = float(gray.mean())
        self.last_consistency = float(gray.std() / max(mean, 1.0))
        gx = np.abs(np.diff(gray.astype(np.int16), axis=1))
        gy = np.abs(np.diff(gray.astype(np.int16), axis=0))
        self.last_edge_density = float(
            ((gx > 20).mean() + (gy > 20).mean()) / 2.0
        )
        candidate = (
            self.last_consistency <= self.consistency_threshold
            and self.last_edge_density <= self.edge_threshold
        )
        if candidate:
            self.candidate_streak += 1
        else:
            self.candidate_streak = 0
        confirmed = self.candidate_streak >= self.min_consecutive
        if confirmed:
            self.triggered = True
        return confirmed, []

    def annotate(self, frame, confirmed=False):
        output = frame.copy()
        status = "CULVERT" if confirmed else "UNIFORM" if self.candidate_streak else "NORMAL"
        cv2.putText(
            output,
            f"{status} consistency={self.last_consistency:.3f} "
            f"edges={self.last_edge_density:.3f} hits={self.candidate_streak}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 0, 255) if confirmed else (0, 220, 255), 2,
        )
        return output


class ChassisLink:
    def __init__(self, port: str, baudrate: int = 115200, dry_run: bool = False):
        self.dry_run = dry_run
        self.serial = None
        if not dry_run:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError("缺少 pyserial，请先安装 python3-serial") from exc
            self.serial = serial.Serial(port, baudrate, timeout=0)
            print(f"[SERIAL] connected: {port}")

    def send(self, command: str) -> None:
        print(f"[SERIAL TX] {command}")
        if self.serial is not None:
            self.serial.write(command.encode("ascii"))
            self.serial.flush()

    def trigger_culvert(self) -> None:
        # 指令 5 在下位机接收任务中具有最高优先级，可立即中断当前运动。
        self.send("@|5|0|0|0#")
        time.sleep(0.15)
        self.send("@|10|0|0|0#")

    def read_lines(self) -> Iterable[str]:
        if self.serial is None:
            return []
        messages: list[str] = []
        while self.serial.in_waiting:
            line = self.serial.readline().decode("utf-8", errors="ignore").strip()
            if line:
                messages.append(line)
        return messages

    def close(self) -> None:
        if self.serial is not None:
            self.serial.close()


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def configure_desktop_display(show: bool) -> None:
    """让 VS Code/SSH 终端启动的 OpenCV 窗口显示到本机桌面。"""
    if not show or os.environ.get("DISPLAY"):
        return
    runtime_dir = Path(f"/run/user/{os.getuid()}")
    auth_files = sorted(
        runtime_dir.glob(".mutter-Xwaylandauth.*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if Path("/tmp/.X11-unix/X0").exists() and auth_files:
        os.environ["DISPLAY"] = ":0"
        os.environ["XAUTHORITY"] = str(auth_files[0])
        os.environ.setdefault("XDG_RUNTIME_DIR", str(runtime_dir))
        print(f"[DISPLAY] using :0 ({auth_files[0]})", flush=True)
        return
    raise RuntimeError(
        "No desktop DISPLAY found. Run with --no-show or start from the local desktop."
    )


def open_camera(source: int | str, width: int, height: int) -> cv2.VideoCapture:
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    cap = cv2.VideoCapture(source, backend)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {source}")
    print(
        f"[CAMERA] opened source={source} "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"fps={cap.get(cv2.CAP_PROP_FPS):.1f}",
        flush=True,
    )
    return cap


def run_self_test() -> int:
    detector = CulvertBrightnessDetector(min_consecutive=3)
    uniform = np.full((240, 320, 3), 180, dtype=np.uint8)
    assert [detector.inspect(uniform)[0] for _ in range(4)] == [False, False, True, False]
    detector.reset()
    textured = np.zeros_like(uniform)
    textured[:, ::8] = 255
    assert not any(detector.inspect(textured)[0] for _ in range(6))
    detector.reset()
    assert detector.inspect(uniform)[0] is False
    assert detector.annotate(uniform).shape == uniform.shape
    print("[SELF TEST] PASS: brightness/texture, confirmation, reset, annotation")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="涵洞亮度/纹理检测联动测试")
    parser.add_argument("--camera", default=os.environ.get("DETECT_CAMERA", "0"))
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--flip", choices=("none", "-1", "0", "1"), default="none",
                        help="画面翻转方式；高位摄像头默认不翻转")
    parser.add_argument("--stop-delay", type=float, default=0.6,
                        help="确认涵洞后继续行驶的秒数")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示带检测标注的实时摄像头画面（默认开启，--no-show 关闭）",
    )
    parser.add_argument("--dry-run", action="store_true", help="不打开串口、不控制底盘")
    parser.add_argument(
        "--repeat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="收到 Y_END 后自动重新启用涵洞检测（默认开启，--no-repeat 关闭）",
    )
    parser.add_argument("--save-dir", default="", help="触发时保存标注图片的目录")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    configure_desktop_display(args.show)
    detector = CulvertBrightnessDetector(
        consistency_threshold=float(os.environ.get("CULVERT_BRIGHTNESS_THRESHOLD", "0.11")),
        edge_threshold=float(os.environ.get("CULVERT_EDGE_THRESHOLD", "0.012")),
        min_consecutive=3,
    )
    cap = open_camera(parse_source(args.camera), args.width, args.height)
    if args.show:
        cv2.namedWindow("Culvert detector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Culvert detector", args.width * 2, args.height * 2)
        print("[DISPLAY] preview window created; press q or Esc to exit", flush=True)
    chassis = ChassisLink(args.serial, dry_run=args.dry_run)
    triggered = False
    detection_enabled = True
    flip_code = None if args.flip == "none" else int(args.flip)
    frame_count = 0
    first_frame_reported = False
    print("[SYS] 涵洞识别开始；亮度/纹理连续确认涵洞后将急停并启动嫌疑人扫描")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            if not first_frame_reported:
                print(f"[CAMERA] first frame received: {frame.shape}", flush=True)
                first_frame_reported = True
            if flip_code in (-1, 0, 1):
                frame = cv2.flip(frame, flip_code)
            if detection_enabled:
                confirmed, _ = detector.inspect(frame)
            else:
                # 涵洞触发后由下位机云台和成熟上位机负责嫌疑人识别，
                # 此时暂停涵洞检测，等待扫描结束。
                confirmed = False
            frame_count += 1

            if confirmed and not triggered:
                triggered = True
                detection_enabled = False
                print(
                    f"[CULVERT] confirmed at frame {frame_count}; "
                    f"continue {args.stop_delay:.2f}s then stop"
                )
                time.sleep(max(0.0, args.stop_delay))
                chassis.trigger_culvert()
                annotated = detector.annotate(frame, True)
                if args.save_dir:
                    save_dir = Path(args.save_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    path = save_dir / f"culvert_{int(time.time())}.jpg"
                    cv2.imwrite(str(path), annotated)
                    print(f"[DEBUG] saved: {path}")

            for message in chassis.read_lines():
                print(f"[SERIAL RX] {message}")
                if message == "Y_START":
                    print("[SUSPECT] lower controller entered suspect-camera mode")
                elif message in {"Y_L", "Y_R"}:
                    print(f"[SUSPECT] gimbal position {message}; run suspect recognition now")
                elif message == "Y_END" and args.repeat:
                    detector.reset()
                    triggered = False
                    detection_enabled = True
                    print("[CULVERT] re-armed")

            if args.show:
                annotated = detector.annotate(frame, confirmed)
                if not detection_enabled:
                    cv2.putText(annotated, "SUSPECT MODE - culvert detection OFF",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                (255, 80, 80), 2)
                cv2.imshow("Culvert detector", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        print("\n[SYS] stopped")
    finally:
        cap.release()
        chassis.close()
        if args.show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
