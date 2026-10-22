#!/usr/bin/env python3
"""Standalone visual line-following and single-intersection test."""

from __future__ import annotations

import signal
import time
import os
from dataclasses import dataclass

import cv2

from old.SRC.Serial import ChassisController
from old.SRC.line_detect import LineTracker


@dataclass
class TestConfig:
    # ==================== 只需要修改这里 ====================
    # straight: 定时直道巡线；cross: 找到路口后执行 turn。
    mode: str = "cross"
    # cross 模式可选 forward、left、right。
    turn: str = "forward"

    duration: float = 10.0
    cross_timeout: float = 30.0
    action_timeout: float = 10.0
    speed: int = 5
    kp: float = 0.35
    kd: float = 0.10
    max_angle: float = 30.0
    countdown: int = 3
    max_camera_misses: int = 10

    camera: int | str = int(os.environ.get("NAV_CAMERA", "0"))
    flip: int | None = None
    serial_port: str = os.environ.get(
        "CHASSIS_PORT",
        "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A5069RR4-if00-port0",
    )
    baudrate: int = 115200
    debug: bool = os.environ.get("LINE_DEBUG", "0").lower() in {
        "1", "true", "on", "yes"
    }
    # ========================================================


class LineFollowTest:
    def __init__(self, args: TestConfig):
        self.args = args
        self.running = True
        self.last_offset = 0
        self.chassis = ChassisController(args.serial_port, args.baudrate)
        self.tracker = LineTracker()
        self.tracker.debug_mode = args.debug

        backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        self.camera = cv2.VideoCapture(args.camera, backend)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def validate_hardware(self) -> None:
        if not self.chassis.serial or not self.chassis.serial.is_open:
            raise RuntimeError(f"底盘串口不可用: {self.args.serial_port}")
        if not self.camera.isOpened():
            raise RuntimeError(f"巡线摄像头不可用: {self.args.camera}")

    def stop(self, *_args: object) -> None:
        self.running = False
        self.chassis.stop()

    def read_road(self) -> tuple[str, int] | None:
        ok, frame = self.camera.read()
        if not ok or frame is None or frame.size == 0:
            return None
        if self.args.flip in (-1, 0, 1):
            frame = cv2.flip(frame, self.args.flip)
        return self.tracker.get_road_info(frame)

    def follow_once(self, offset: int) -> None:
        correction = (
            self.args.kp * offset
            + self.args.kd * (offset - self.last_offset)
        )
        self.last_offset = offset
        correction = max(-self.args.max_angle, min(self.args.max_angle, correction))
        self.chassis.drive_steer(self.args.speed, correction)

    def wait_for_completion(self) -> bool:
        deadline = time.monotonic() + self.args.action_timeout
        while self.running and time.monotonic() < deadline:
            message = self.chassis.read_message()
            if message == "@":
                return True
            if message and message.startswith("ERR:"):
                print(f"[WARN] 下位机动作失败: {message}")
                return False
            time.sleep(0.01)
        return False

    def execute_cross_action(self) -> bool:
        action = self.args.turn
        # 防止之前命令的 ACK 被误认为当前路口动作已经完成。
        self.chassis.clear_pending_input()
        print(f"\n[TEST] 执行路口动作: {action}")
        if action == "forward":
            self.chassis.go_cross()
        elif action == "left":
            self.chassis.ccw()
        elif action == "right":
            self.chassis.cw()
        else:
            raise ValueError(f"未知路口动作: {action}")

        completed = self.wait_for_completion()
        if completed:
            print("[TEST] 下位机已确认动作完成")
        else:
            print("[WARN] 等待动作完成超时，立即停车")
        return completed

    def run_straight(self) -> int:
        print(f"[TEST] 直道巡线 {self.args.duration:.1f} 秒，Ctrl+C 可停车")
        deadline = time.monotonic() + self.args.duration
        missed_frames = 0
        while self.running and time.monotonic() < deadline:
            road = self.read_road()
            if road is None:
                missed_frames += 1
                if missed_frames >= self.args.max_camera_misses:
                    print("\n[ERROR] 摄像头连续取帧失败，停车")
                    return 2
                time.sleep(0.02)
                continue

            missed_frames = 0
            road_type, offset = road
            self.follow_once(offset)
            print(
                f"[LINE] 状态={road_type:<5} 偏差={offset:>4} ",
                end="\r",
                flush=True,
            )
            if self.tracker.last_key == ord("q"):
                break
            time.sleep(0.02)
        print("\n[TEST] 直道测试结束")
        return 0

    def run_cross(self) -> int:
        print(
            f"[TEST] 寻找第一个路口，检测后执行 {self.args.turn}，"
            f"最长等待 {self.args.cross_timeout:.1f} 秒"
        )
        deadline = time.monotonic() + self.args.cross_timeout
        missed_frames = 0
        while self.running and time.monotonic() < deadline:
            road = self.read_road()
            if road is None:
                missed_frames += 1
                if missed_frames >= self.args.max_camera_misses:
                    print("\n[ERROR] 摄像头连续取帧失败，停车")
                    return 2
                time.sleep(0.02)
                continue

            missed_frames = 0
            road_type, offset = road
            if road_type == "CROSS":
                # 使用无 ACK 的零速度巡线指令停车。急停指令会返回 @，容易被
                # 后续 wait_for_completion() 误认为转向动作的完成应答。
                self.chassis.drive_steer(0, 0)
                time.sleep(0.2)
                return 0 if self.execute_cross_action() else 3

            self.follow_once(offset)
            print(f"[LINE] 偏差={offset:>4}，等待路口 ", end="\r", flush=True)
            if self.tracker.last_key == ord("q"):
                break
            time.sleep(0.02)

        if self.running:
            print("\n[WARN] 在限定时间内没有检测到路口")
            return 4
        return 0

    def run(self) -> int:
        try:
            self.validate_hardware()
            print(
                f"[READY] 摄像头={self.args.camera} 串口={self.args.serial_port} "
                f"速度={self.args.speed}"
            )
            for remaining in range(self.args.countdown, 0, -1):
                print(f"[READY] {remaining} 秒后启动...", end="\r", flush=True)
                time.sleep(1)
            print(" " * 40, end="\r")
            if self.args.mode == "straight":
                return self.run_straight()
            return self.run_cross()
        finally:
            self.chassis.stop()
            time.sleep(0.1)
            self.camera.release()
            cv2.destroyAllWindows()
            self.chassis.close()
            print("[SAFE] 电机已停车，摄像头和串口已释放")


def main() -> int:
    args = TestConfig()
    if args.mode not in {"straight", "cross"}:
        raise SystemExit("mode 只能填写 straight 或 cross")
    if args.turn not in {"forward", "left", "right"}:
        raise SystemExit("turn 只能填写 forward、left 或 right")
    if args.flip not in {None, -1, 0, 1}:
        raise SystemExit("flip 只能填写 None、-1、0 或 1")
    if args.duration <= 0 or args.cross_timeout <= 0 or args.action_timeout <= 0:
        raise SystemExit("duration 和 timeout 必须大于 0")
    if args.countdown < 0 or args.max_camera_misses < 1:
        raise SystemExit("countdown 必须非负，max-camera-misses 必须大于 0")

    test = LineFollowTest(args)
    signal.signal(signal.SIGINT, test.stop)
    signal.signal(signal.SIGTERM, test.stop)
    return test.run()


if __name__ == "__main__":
    raise SystemExit(main())
