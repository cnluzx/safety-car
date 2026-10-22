"""带随机涵洞和随机障碍处理的全地图行走测试。

该测试沿用 ``line_follow_test.py`` 的硬件调用方式：巡线模块直接读取
摄像头，底盘动作直接通过 ``ChassisController`` 发送，并在每个动作后轮询
下位机的 ``@`` 完成应答。地图规划只使用物理道路和 N1～N12 固定点，完成
条件为覆盖全部道路、经过所有固定点并返回出发区。
"""

from __future__ import annotations

import argparse
from collections import deque
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from Serial import ChassisController
from line_detect import LineTracker
from navigator import Navigator
from topo_map import TopoMap
from odometry import Odometry
from rfid import RFIDReader
try:
    from yolo_detect import YoloDetector
except Exception as exc:
    YoloDetector = None
    print(f"[YOLO WARN] 无法导入目标识别模块: {exc}")
try:
    from oled import OLEDDisplay
except Exception as exc:
    OLEDDisplay = None
    print(f"[OLED WARN] 无法导入显示模块: {exc}")
try:
    from broadcast import Broadcaster
except Exception as exc:
    Broadcaster = None
    print(f"[Audio WARN] 无法导入播报模块: {exc}")

_TEST_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_code")
if _TEST_CODE_DIR not in sys.path:
    sys.path.insert(0, _TEST_CODE_DIR)
from Culvert_detect import CulvertBrightnessDetector
from obstacle_color_test import ColorObstacleDetector


SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# These physical outer-road segments use the L-junction detector in both
# directions. Asymmetric inner-to-outer approaches are listed separately below.
# T1-T4 are tunnel identifiers on physical edges, not navigation nodes.
L_JUNCTION_SEGMENTS = frozenset({
    frozenset(("J_TOP", "N1")),
    frozenset(("J_TOP", "N12")),
    frozenset(("N2", "N3")),
    frozenset(("N3", "N4")),
    frozenset(("N4", "N5")),
    frozenset(("N5", "N6")),
    frozenset(("N6", "N7")),
    frozenset(("N7", "N8")),
    frozenset(("N8", "N9")),
    frozenset(("N9", "N10")),
    frozenset(("N10", "N11")),
})

# These inner-to-outer approaches are asymmetric: the outer-bound direction
# looks like an L junction, while returning from the outer node is a normal
# crossroad approach.
L_JUNCTION_DIRECTIONS = frozenset({
    ("I1_1", "N1"),
    ("I1_1", "N2"),
    ("I12_1", "N12"),
    ("I12_1", "N11"),
})

# OpenCV 在运行时提供该 C 扩展函数，但部分 cv2 类型存根没有声明它。
_video_writer_fourcc = getattr(cv2, "VideoWriter_fourcc")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _parse_camera_source(value: str) -> int | str:
    value = str(value).strip()
    try:
        return int(value)
    except ValueError:
        return value


def _parse_flip(value: str) -> int | None:
    value = str(value).strip().lower()
    if value in {"none", "off", "false", "no"}:
        return None
    code = int(value)
    if code not in (-1, 0, 1):
        raise ValueError("flip 只能填写 None、-1、0 或 1")
    return code


@dataclass
class FullMapConfig:
    camera: int | str = _parse_camera_source(os.environ.get("NAV_CAMERA", "2"))
    flip: int | None = _parse_flip(os.environ.get("NAV_CAMERA_FLIP", "-1"))
    serial_port: str = os.environ.get(
        "CHASSIS_PORT",
        "/dev/ttyUSB0",
    )
    baudrate: int = 115200
    speed: int = int(os.environ.get("FULL_MAP_LINE_SPEED", "9"))
    kp: float = float(os.environ.get("FULL_MAP_KP", "0.5"))
    kd: float = float(os.environ.get("FULL_MAP_KD", "0.10"))
    max_angle: float = float(os.environ.get("FULL_MAP_MAX_ANGLE", "30"))
    action_timeout: float = float(os.environ.get("FULL_MAP_ACTION_TIMEOUT", "10"))
    camera_startup_timeout: float = float(
        os.environ.get("CAMERA_STARTUP_TIMEOUT", "10")
    )
    system_ready_timeout: float = float(
        os.environ.get("SYSTEM_READY_TIMEOUT", "20")
    )
    startup_min_road_fps: float = float(
        os.environ.get("STARTUP_MIN_ROAD_FPS", "10")
    )
    startup_stable_frames: int = max(
        2, int(os.environ.get("STARTUP_STABLE_FRAMES", "15"))
    )
    camera_stale_timeout: float = float(
        os.environ.get("CAMERA_STALE_TIMEOUT", "1.0")
    )
    road_stale_timeout: float = float(
        os.environ.get("ROAD_STALE_TIMEOUT", "1.5")
    )
    start_action_timeout: float = float(
        os.environ.get("START_ACTION_TIMEOUT", "8")
    )
    start_line_required_frames: int = max(
        1, int(os.environ.get("START_LINE_STABLE_FRAMES", "30"))
    )
    start_right_turn_angle: int = int(
        os.environ.get("START_RIGHT_TURN_ANGLE", "80")
    )
    start_forward_speed: int = int(os.environ.get("START_FORWARD_SPEED", "11"))
    start_forward_distance: int = int(
        os.environ.get("START_FORWARD_DISTANCE", "100")
    )
    return_start_turn_angle: int = int(
        os.environ.get("RETURN_START_TURN_ANGLE", "75")
    )
    return_start_approach_speed: int = int(os.environ.get("RETURN_START_APPROACH_SPEED", "11"))
    return_start_approach_distance: int = int(os.environ.get("RETURN_START_APPROACH_DISTANCE", "100"))
    return_start_speed: int = int(os.environ.get("RETURN_START_SPEED", "11"))
    return_start_distance: int = int(
        os.environ.get("RETURN_START_DISTANCE", "95")
    )
    return_start_timeout: float = float(
        os.environ.get("RETURN_START_TIMEOUT", "6")
    )
    cross_timeout: float = float(os.environ.get("FULL_MAP_CROSS_TIMEOUT", "100"))
    countdown: int = int(os.environ.get("FULL_MAP_COUNTDOWN", "1"))
    max_camera_misses: int = int(os.environ.get("FULL_MAP_MAX_CAMERA_MISSES", "10"))
    debug: bool = _env_bool("LINE_DEBUG", False)
    culvert_camera: int | str = _parse_camera_source(os.environ.get("CULVERT_CAMERA", "0"))
    culvert_flip: int | None = _parse_flip(os.environ.get("CULVERT_CAMERA_FLIP", "none"))
    rfid_port: str = os.environ.get("RFID_PORT", "/dev/ttyACM0")
    rfid_baudrate: int = int(os.environ.get("RFID_BAUDRATE", "115200"))
    culvert_min_length_ratio: float = float(
        os.environ.get("CULVERT_MIN_LENGTH_RATIO", "0.35")
    )
    culvert_stop_delay: float = float(
        os.environ.get("CULVERT_STOP_DELAY", "0")
    )
    culvert_save_photos: bool = _env_bool("CULVERT_SAVE_PHOTOS", True)
    culvert_detect_interval: int = max(
        1, int(os.environ.get("CULVERT_DETECT_INTERVAL", "1"))
    )
    # 高位涵洞画面默认不弹窗，避免与底部巡线调试窗口争用 Qt 线程。
    culvert_debug: bool = _env_bool("CULVERT_DEBUG", False)
    culvert_scan_timeout: float = float(
        os.environ.get("CULVERT_SCAN_TIMEOUT", "20")
    )
    culvert_rearm_delay: float = float(
        os.environ.get("CULVERT_REARM_DELAY", "3")
    )
    obstacle_enabled: bool = _env_bool("OBSTACLE_DETECT_ENABLED", True)
    obstacle_debug: bool = _env_bool("OBSTACLE_DEBUG", False)
    obstacle_min_area: float = float(os.environ.get("OBSTACLE_MIN_AREA", "200"))
    obstacle_min_area_ratio: float = float(
        os.environ.get("OBSTACLE_MIN_AREA_RATIO", "0.02")
    )
    obstacle_roi_left: float = float(os.environ.get("OBSTACLE_ROI_LEFT", "0.40"))
    obstacle_roi_top: float = float(os.environ.get("OBSTACLE_ROI_TOP", "0.75"))
    obstacle_roi_right: float = float(os.environ.get("OBSTACLE_ROI_RIGHT", "0.50"))
    obstacle_roi_bottom: float = float(os.environ.get("OBSTACLE_ROI_BOTTOM", "1.0"))
    obstacle_detect_interval: float = max(
        0.02, float(os.environ.get("OBSTACLE_DETECT_INTERVAL", "0.04"))
    )
    obstacle_confirm_frames: int = max(
        2, int(os.environ.get("OBSTACLE_CONFIRM_FRAMES", "2"))
    )
    obstacle_cooldown: float = max(
        0.0, float(os.environ.get("OBSTACLE_COOLDOWN", "1.5"))
    )
    ack_min_latency: float = max(
        0.0, float(os.environ.get("ACK_MIN_LATENCY", "0.05"))
    )
    ack_debounce: float = max(
        0.0, float(os.environ.get("ACK_DEBOUNCE", "0.15"))
    )


def build_test_navigator():
    """建立不含绿色任务点、但要求覆盖全部道路的导航器。"""
    topo_map = TopoMap()
    topo_map.build_competition_track(green_count=0)
    navigator = Navigator(topo_map)
    navigator.init_mission(
        start_node="START_AREA",
        mission_plan=topo_map.fixed_patrol_nodes,
        expected_green_count=int(os.environ.get("CULVERT_EXPECTED_COUNT", "8")),
        cover_all_edges=True,
    )
    # 将最终返航支路延后覆盖：第一次到达 N12 时原地掉头，继续巡检
    # 主地图；其他道路完成后才走 N12 → J_TOP → START_AREA。
    navigator.defer_coverage_edges([("N12", "J_TOP")])
    return topo_map, navigator


def calculate_plan():
    """离线推进规划器，得到本次测试会覆盖的完整节点路线。"""
    topo_map, navigator = build_test_navigator()
    route: list[str] = [str(navigator.current_node)]
    safety_limit = max(100, len(topo_map.nodes) * len(topo_map.edges) * 4)

    while not navigator.mission_complete and len(route) < safety_limit:
        segment = navigator.get_next_segment()
        if not segment:
            break
        reached = navigator.advance_path()
        if reached is None:
            break
        route.append(str(reached))

    return route, navigator.get_progress()


def print_plan() -> bool:
    route, progress = calculate_plan()
    print("[FULL MAP TEST] 随机涵洞、随机障碍全道路巡检任务")
    print("[FULL MAP TEST] 路线: " + " -> ".join(route))
    print(
        "[FULL MAP TEST] 道路覆盖: "
        f"{progress['covered_edges']}/{progress['required_edges']} "
        f"({progress['coverage_percent']:.1f}%)"
    )
    print(
        "[FULL MAP TEST] 固定点: "
        f"{progress['visited']}/{progress['total']}; "
        f"缺失={progress['missing']}"
    )
    print(f"[FULL MAP TEST] 返回出发区并完成: {progress['complete']}")
    return bool(progress["complete"])


class FullMapWalkTest:
    """使用 line_follow_test 同款接口运行地图巡线状态机。"""

    def __init__(self, args: FullMapConfig):
        self.args = args
        self.running = True
        self.stop_requested = False
        self.last_offset = 0
        self.last_action_error = None
        self._cross_event_latched = False
        # Armed after a tunnel segment; applies the stricter profile until the
        # first junction on the road after the tunnel.
        self._tunnel_exit_junction_pending = False
        self._last_line_log_time = 0.0
        self._road_frame_times = deque(maxlen=args.startup_stable_frames)
        self._road_fps_stable = threading.Event()
        self._startup_road_fps = 0.0
        self._last_camera_frame_at = 0.0
        self._last_culvert_camera_frame_at = 0.0
        self._last_road_result_at = 0.0
        self._road_resume_grace_until = 0.0
        self._camera_thread_started = threading.Event()
        self._culvert_camera_thread_started = threading.Event()
        self._road_thread_started = threading.Event()
        self._culvert_thread_started = threading.Event()
        self._obstacle_thread_started = threading.Event()
        self._rfid_thread_started = threading.Event()
        self._broadcast_thread_started = threading.Event()
        self.start_line_frames = 0
        self.cross_event_count = 0
        self.action_count = 0
        self.ack_received_count = 0
        self.ack_accepted_count = 0
        self.ack_debounced_count = 0
        self._ack_lock = threading.Lock()
        self.state = "START"
        self.odometry = Odometry()
        self._culvert_detector = CulvertBrightnessDetector(
            consistency_threshold=float(os.environ.get("CULVERT_BRIGHTNESS_THRESHOLD", "0.12")),
            edge_threshold=float(os.environ.get("CULVERT_EDGE_THRESHOLD", "0.012")),
            min_consecutive=2,
        )
        self._culvert_camera = None
        self._culvert_camera_lock = threading.Lock()
        self._culvert_camera_ready = threading.Event()
        self._culvert_camera_first_frame = threading.Event()
        self._culvert_camera_frame = None
        self._culvert_camera_sequence = 0
        self._culvert_frame_history = {}
        self._culvert_lock = threading.Lock()
        self._culvert_triggered = threading.Event()
        self._culvert_active = threading.Event()
        self._culvert_road_lock = threading.Lock()
        self._culvert_road = None
        self._culvert_scanned_edges = set()
        self._culvert_rearm_at = 0.0
        self._obstacle_detector = ColorObstacleDetector(
            min_area=args.obstacle_min_area,
            min_area_ratio=args.obstacle_min_area_ratio,
            roi_left_ratio=args.obstacle_roi_left,
            roi_top_ratio=args.obstacle_roi_top,
            roi_right_ratio=args.obstacle_roi_right,
            roi_bottom_ratio=args.obstacle_roi_bottom,
        )
        self._obstacle_pending = threading.Event()
        self._obstacle_pending_lock = threading.Lock()
        self._obstacle_pending_data = None
        self._obstacle_debug_frame = None
        self._obstacle_debug_window_ready = False
        self._obstacle_debug_next_at = 0.0
        self._obstacle_rearm_at = 0.0
        self._obstacle_recovering = False
        self._obstacle_return_from = None
        self._npu_lock = threading.Lock()
        self.chassis = ChassisController(args.serial_port, args.baudrate)
        self.tracker = LineTracker(debug_mode=args.debug)
        self._yolo = None
        self._culvert_photo_dir = Path(
            os.environ.get(
                "CULVERT_YOLO_SAVE_DIR",
                str(Path(__file__).resolve().parent.parent / "logs" / "culvert_yolo"),
            )
        )
        self._culvert_photo_count = 0
        if YoloDetector is not None:
            try:
                self._yolo = YoloDetector(
                    preset_classes=tuple(range(1, 21)),
                    show_window=False,
                    save_debug=False,
                )
                if self._yolo.rknn is None:
                    self._yolo = None
                    print("[YOLO WARN] 目标模型预加载失败，涵洞扫描时将重试")
                else:
                    print("[YOLO] 目标模型已预加载")
            except Exception as exc:
                self._yolo = None
                print(f"[YOLO WARN] 目标模型预加载失败，涵洞扫描时将重试: {exc}")

        backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        try:
            self.camera = cv2.VideoCapture(args.camera, backend)
        except Exception:
            self.camera = cv2.VideoCapture(args.camera)
        # Match line_detect.py's standalone full-view camera input. LineTracker
        # directly resizes the actual camera frame to its configured model input.
        self.camera.set(cv2.CAP_PROP_FOURCC, _video_writer_fourcc(*"MJPG"))
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 544)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Capture frames independently; the main loop always consumes the latest frame.
        self._camera_lock = threading.Lock()
        self._camera_ready = threading.Event()
        # Startup latch is independent of the consumable new-frame event.
        self._camera_first_frame = threading.Event()
        self._camera_frame = None
        self._camera_sequence = 0
        self._processed_camera_sequence = 0
        self._camera_thread = threading.Thread(
            target=self._camera_capture_loop,
            name="camera-capture",
            daemon=True,
        )
        self._camera_thread.start()
        self._road_lock = threading.Lock()
        self._road_ready = threading.Event()
        self._road_result: tuple[str, int] | None = None
        self._road_result_sequence = 0
        self._consumed_road_sequence = 0

        # 高位摄像头独立采集，涵洞识别线程不会占用巡线摄像头帧。
        backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        try:
            self._culvert_camera = cv2.VideoCapture(args.culvert_camera, backend)
        except Exception:
            self._culvert_camera = cv2.VideoCapture(args.culvert_camera)
        # Both UVC devices expose 544x288 MJPG reliably when used together.
        # The detector resizes this frame to its 640x360 model input.
        self._culvert_camera.set(
            cv2.CAP_PROP_FOURCC, _video_writer_fourcc(*"MJPG")
        )
        self._culvert_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 544)
        self._culvert_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)
        self._culvert_camera.set(cv2.CAP_PROP_FPS, 15)
        self._culvert_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._culvert_camera_thread = threading.Thread(
            target=self._culvert_camera_capture_loop,
            name="culvert-camera-capture",
            daemon=True,
        )
        self._culvert_camera_thread.start()
        self._rfid_reader = RFIDReader(port=args.rfid_port, baudrate=args.rfid_baudrate)
        self._broadcast_queue: queue.Queue[tuple[str, int]] = queue.Queue()
        self._broadcaster = Broadcaster() if Broadcaster is not None else None
        self._oled_lock = threading.Lock()
        self._oled = None
        if OLEDDisplay is not None:
            try:
                self._oled = OLEDDisplay()
            except Exception as exc:
                print(f"[OLED WARN] 显示模块初始化失败: {exc}")

        self.topo_map, self.navigator = build_test_navigator()
        self.current_node = "START_AREA"
        self.previous_node = None
        self.current_path = None
        self.current_target_node = None
        self._sync_navigation_path()
        self._road_thread = threading.Thread(
            target=self._road_processing_loop, name="road-processing", daemon=True
        )
        self._culvert_thread = threading.Thread(target=self._culvert_loop, name="culvert-detect", daemon=True)
        self._obstacle_thread = threading.Thread(
            target=self._obstacle_loop, name="obstacle-detect", daemon=True
        )
        self._rfid_thread = threading.Thread(target=self._rfid_loop, name="rfid-reader", daemon=True)
        self._broadcast_thread = threading.Thread(target=self._broadcast_loop, name="broadcast", daemon=True)
        self._culvert_thread.start()
        self._obstacle_thread.start()
        self._rfid_thread.start()
        self._road_thread.start()
        self._broadcast_thread.start()
        self._show_oled("全地图测试\n正在准备系统", duration=1200)

    def _show_oled(self, message: str, duration: int = 1200) -> None:
        """非阻塞显示运行状态；显示故障不影响底盘安全控制。"""
        if self._oled is None:
            return
        with self._oled_lock:
            try:
                self._oled.show_text(message, duration=duration)
            except Exception as exc:
                print(f"[OLED WARN] 显示失败: {exc}")

    def _camera_capture_loop(self) -> None:
        self._camera_thread_started.set()
        reported_size = None
        while self.running:
            ok, frame = self.camera.read()
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.005)
                continue
            if self.args.flip in (-1, 0, 1):
                frame = cv2.flip(frame, self.args.flip)
            actual_size = (frame.shape[1], frame.shape[0])
            if actual_size != reported_size:
                print(f"[LINE CAMERA] requested=640x360 MJPG | "
                      f"actual={actual_size[0]}x{actual_size[1]} | "
                      f"model={self.tracker.input_width}x{self.tracker.input_height} | "
                      "preview=direct (no padding/crop)")
                if actual_size != (544, 288):
                    print("[LINE CAMERA WARN] 摄像头未提供请求的 544x288，按实际画面处理，与独立测试一致")
                reported_size = actual_size
            with self._camera_lock:
                self._camera_frame = frame
                self._camera_sequence += 1
                self._last_camera_frame_at = time.monotonic()
            self._camera_first_frame.set()
            self._camera_ready.set()

    def _culvert_camera_capture_loop(self) -> None:
        """持续清空高位UVC缓存，并保存带序号的最新帧。"""
        self._culvert_camera_thread_started.set()
        next_capture_at = 0.0
        reported_open = False
        read_failures = 0
        backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        while self.running:
            now = time.monotonic()
            if now < next_capture_at:
                time.sleep(min(0.01, next_capture_at - now))
                continue
            camera = self._culvert_camera
            if camera is None or not camera.isOpened():
                if not reported_open:
                    print(f"[CULVERT CAMERA] source={self.args.culvert_camera} not opened; retrying", flush=True)
                    reported_open = True
                time.sleep(0.2)
                continue
            if not reported_open:
                print(f"[CULVERT CAMERA] source={self.args.culvert_camera} opened", flush=True)
                reported_open = True
            ok, frame = camera.read()
            if not ok or frame is None or frame.size == 0:
                read_failures += 1
                if read_failures == 1 or read_failures % 5 == 0:
                    print(f"[CULVERT CAMERA] source={self.args.culvert_camera} frame read failed ({read_failures}); retrying", flush=True)
                # Some UVC drivers report open=True but stop delivering frames
                # after the other camera is initialized. Recreate the stream.
                if read_failures >= 5:
                    try:
                        camera.release()
                    except Exception:
                        pass
                    time.sleep(0.2)
                    try:
                        camera = cv2.VideoCapture(self.args.culvert_camera, backend)
                    except Exception:
                        camera = cv2.VideoCapture(self.args.culvert_camera)
                    camera.set(cv2.CAP_PROP_FOURCC, _video_writer_fourcc(*"MJPG"))
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 544)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)
                    camera.set(cv2.CAP_PROP_FPS, 15)
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self._culvert_camera = camera
                    read_failures = 0
                    reported_open = False
                time.sleep(0.05)
                continue
            read_failures = 0
            with self._culvert_camera_lock:
                self._culvert_camera_frame = frame
                self._culvert_camera_sequence += 1
                self._culvert_frame_history[self._culvert_camera_sequence] = frame.copy()
                if len(self._culvert_frame_history) > 12:
                    oldest = min(self._culvert_frame_history)
                    del self._culvert_frame_history[oldest]
                self._last_culvert_camera_frame_at = time.monotonic()
            self._culvert_camera_first_frame.set()
            self._culvert_camera_ready.set()
            # 高位摄像头始终保持15FPS，减少云台到位后的取帧延迟。
            next_capture_at = time.monotonic() + (1.0 / 15.0)

    def _ensure_yolo(self) -> bool:
        """确认预加载模型可用；启动时加载失败则在扫描阶段重试。"""
        if self._yolo is not None and self._yolo.rknn is not None:
            return True
        if YoloDetector is None:
            print("[YOLO ERROR] 目标识别模块不可用")
            return False
        try:
            self._yolo = YoloDetector(
                preset_classes=tuple(range(1, 21)),
                show_window=False,
                save_debug=False,
            )
            ready = self._yolo.rknn is not None
            print(f"[YOLO] 扫描阶段重试加载 {'成功' if ready else '失败'}")
            return ready
        except Exception as exc:
            self._yolo = None
            print(f"[YOLO ERROR] 目标模型按需加载失败: {exc}")
            return False

    def _release_yolo(self) -> None:
        yolo, self._yolo = self._yolo, None
        if yolo is not None:
            yolo.release()
            print("[YOLO] 目标模型资源已释放")

    def _latest_culvert_frame(
        self, after_sequence: int = -1, min_new_frames: int = 1,
        timeout: float = 0.8,
    ):
        """返回 after_sequence 之后至少 min_new_frames 的高位新帧。"""
        target_sequence = after_sequence + max(1, min_new_frames)
        deadline = time.monotonic() + timeout
        while self.running and time.monotonic() < deadline:
            with self._culvert_camera_lock:
                sequence = self._culvert_camera_sequence
                frame = self._culvert_camera_frame
                if sequence >= target_sequence and frame is not None:
                    # 等够指定帧数后，从这批新帧中选 Laplacian 方差最高的一帧。
                    candidates = [(frame.copy(), sequence)]
                    start_sequence = after_sequence + 1
                    for seq in range(start_sequence, sequence):
                        cached = self._culvert_frame_history.get(seq)
                        if cached is not None:
                            candidates.append((cached.copy(), seq))
                    best_frame, best_sequence = max(
                        candidates,
                        key=lambda item: float(
                            cv2.Laplacian(
                                cv2.cvtColor(item[0], cv2.COLOR_BGR2GRAY),
                                cv2.CV_64F,
                            ).var()
                        ),
                    )
                    return best_frame, best_sequence
                # 当前帧尚未达到目标序号，清除旧通知后等待采集线程发布
                # 真正的新帧。原逻辑未清除Event，会在两帧之间持续空转。
                self._culvert_camera_ready.clear()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                self._culvert_camera_ready.wait(min(0.1, remaining))
        return None, after_sequence

    def _road_processing_loop(self) -> None:
        """后台执行 RKNN 巡线推理，主线程只读取最新结果。"""
        self._road_thread_started.set()
        while self.running:
            if self._culvert_active.is_set():
                time.sleep(0.02)
                continue
            frame = self._latest_camera_frame(timeout=0.5)
            if frame is None:
                continue
            debug_from, debug_to = self._current_physical_segment()
            self.tracker.set_debug_segment(debug_from, debug_to)
            self.tracker.set_junction_mode(self._expected_junction_mode())
            with self._npu_lock:
                result = self.tracker.get_road_info(frame)
                with self._road_lock:
                    self._road_result = result
                    self._road_result_sequence += 1
                    self._last_road_result_at = time.monotonic()
                    self._road_ready.set()
                now = time.monotonic()
                self._road_frame_times.append(now)
                if len(self._road_frame_times) == self.args.startup_stable_frames:
                    stable_elapsed = self._road_frame_times[-1] - self._road_frame_times[0]
                    if stable_elapsed > 0:
                        self._startup_road_fps = (
                            (len(self._road_frame_times) - 1) / stable_elapsed
                        )
                        if self._startup_road_fps >= self.args.startup_min_road_fps:
                            self._road_fps_stable.set()

    def _culvert_loop(self) -> None:
        self._culvert_thread_started.set()
        frame_count = 0
        last_camera_sequence = -1
        debug_enabled = self.args.culvert_debug
        if debug_enabled:
            try:
                cv2.namedWindow("Culvert high camera", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Culvert high camera", 960, 540)
            except cv2.error as exc:
                print(f"[CULVERT] 调试窗口不可用: {exc}")
                debug_enabled = False
        while self.running:
            frame, last_camera_sequence = self._latest_culvert_frame(
                after_sequence=last_camera_sequence, timeout=0.5
            )
            if frame is None:
                continue
            if self.args.culvert_flip in (-1, 0, 1):
                frame = cv2.flip(frame, self.args.culvert_flip)
            with self._culvert_road_lock:
                active_road = self._culvert_road
            if active_road is None:
                with self._culvert_lock:
                    self._culvert_detector.reset()
                continue
            if time.monotonic() < self._culvert_rearm_at:
                with self._culvert_lock:
                    self._culvert_detector.reset()
                time.sleep(0.05)
                continue
            frame_count += 1
            if frame_count % self.args.culvert_detect_interval:
                if debug_enabled:
                    try:
                        cv2.imshow("Culvert high camera", frame)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            self.args.culvert_debug = False
                            debug_enabled = False
                    except cv2.error:
                        debug_enabled = False
                continue
            with self._culvert_lock:
                confirmed, candidates = self._culvert_detector.inspect(frame)
            if debug_enabled:
                try:
                    with self._culvert_lock:
                        annotated = self._culvert_detector.annotate(frame, confirmed)
                    cv2.imshow("Culvert high camera", annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        self.args.culvert_debug = False
                        debug_enabled = False
                except cv2.error:
                    debug_enabled = False
            if not confirmed or self._culvert_triggered.is_set():
                continue
            self._culvert_triggered.set()
            print(
                f"[CULVERT] 高位摄像头确认涵洞，道路={active_road[0]} -> {active_road[1]}，"
                f"候选线={len(candidates)}"
            )
            self._run_culvert_scan(*active_road)
        if self.args.culvert_debug:
            try:
                cv2.destroyWindow("Culvert high camera")
            except cv2.error:
                pass

    def _obstacle_segment(self):
        """Return the eligible directed road currently being driven."""
        if (
            not self.args.obstacle_enabled
            or self.state != "FOLLOW_LINE"
            or self._culvert_active.is_set()
            or self._obstacle_recovering
        ):
            return None
        from_node = self.current_node
        to_node = self.current_target_node
        if not from_node or not to_node:
            return None
        if not self.topo_map.is_configured_culvert_edge(from_node, to_node):
            return None
        approach = self.topo_map.directed_edge_key(from_node, to_node)
        with self.navigator.lock:
            if approach in self.navigator.inspected_blocked_approaches:
                return None
        return from_node, to_node

    def _obstacle_loop(self) -> None:
        """Detect cones on shared navigation frames and notify the main loop."""
        self._obstacle_thread_started.set()
        last_sequence = -1
        consecutive = 0
        confirming_segment = None
        display_available = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        debug_enabled = self.args.obstacle_debug and display_available
        window_name = "Obstacle camera 0"
        if self.args.obstacle_debug and not display_available:
            print("[OBSTACLE] 未检测到 DISPLAY，跳过调试窗口")
        while self.running:
            frame, last_sequence = self._latest_camera_frame_after(
                last_sequence, timeout=0.5
            )
            if frame is None:
                self._obstacle_detector.reset()
                consecutive = 0
                confirming_segment = None
                continue
            segment = self._obstacle_segment()
            detection_blocked = (
                segment is None
                or self._obstacle_pending.is_set()
                or time.monotonic() < self._obstacle_rearm_at
            )
            result = None
            if detection_blocked or segment != confirming_segment:
                self._obstacle_detector.reset()
                consecutive = 0
                confirming_segment = segment if not detection_blocked else None
            try:
                if debug_enabled or not detection_blocked:
                    result = self._obstacle_detector.detect(frame)
            except (ValueError, cv2.error) as exc:
                print(f"[OBSTACLE WARN] 单帧检测失败: {exc}")
                consecutive = 0
                confirming_segment = None
                continue
            if debug_enabled and self.args.obstacle_debug and result is not None:
                # Publish only the newest preview. GUI work belongs to the main
                # thread and must not delay obstacle confirmation in this worker.
                self._obstacle_debug_frame = (frame, result, segment, detection_blocked)
            if detection_blocked:
                consecutive = 0
                confirming_segment = None
                continue
            if result is None:
                continue
            # Count raw hits exactly once. result.obstacle already contains
            # the detector's two-frame gate; counting that again adds latency.
            if not result.raw_obstacle:
                consecutive = 0
                confirming_segment = None
                time.sleep(self.args.obstacle_detect_interval)
                continue
            if segment != confirming_segment:
                confirming_segment = segment
                consecutive = 1
            else:
                consecutive += 1
            if consecutive < self.args.obstacle_confirm_frames:
                time.sleep(self.args.obstacle_detect_interval)
                continue

            # Recheck the directed segment after confirmation so a junction
            # transition cannot attach an old frame to the new road.
            if self._obstacle_segment() != segment:
                consecutive = 0
                confirming_segment = None
                continue
            with self._obstacle_pending_lock:
                if not self._obstacle_pending.is_set():
                    self._obstacle_pending_data = (segment, result)
                    self._obstacle_pending.set()
                    # Wake read_road() without waiting for the next NPU result.
                    self._road_ready.set()
                    print(
                        f"[OBSTACLE] 连续{consecutive}帧确认障碍物，"
                        f"道路={segment[0]} -> {segment[1]}，"
                        f"颜色={','.join(result.colors) or 'red/yellow'}，"
                        f"面积={result.combined_area:.0f}"
                    )
            consecutive = 0
            confirming_segment = None
            time.sleep(self.args.obstacle_detect_interval)

    def _pump_obstacle_debug(self) -> None:
        """Render the latest obstacle preview on the main thread, at most 10 Hz."""
        if (not self.args.obstacle_debug or self._obstacle_debug_frame is None
                or self._obstacle_pending.is_set()
                or time.monotonic() < self._obstacle_debug_next_at):
            return
        frame, result, segment, blocked = self._obstacle_debug_frame
        self._obstacle_debug_next_at = time.monotonic() + 0.1
        window_name = "Obstacle camera 0"
        try:
            if not self._obstacle_debug_window_ready:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 960, 540)
                self._obstacle_debug_window_ready = True
            annotated = self._obstacle_detector.draw_result(frame, result)
            label = (f"ACTIVE {segment[0]}->{segment[1]}"
                     if not blocked and segment else "INACTIVE (map/state)")
            cv2.putText(annotated, label, (8, 74), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2)
            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                cv2.destroyWindow(window_name)
                self.args.obstacle_debug = False
                self._obstacle_debug_window_ready = False
        except cv2.error as exc:
            print(f"[OBSTACLE] 调试窗口已关闭: {exc}")
            self.args.obstacle_debug = False

    def _rfid_loop(self) -> None:
        self._rfid_thread_started.set()
        while self.running:
            try:
                card_id = self._rfid_reader.wait_for_card(timeout=0.2)
                if card_id is not None:
                    self.odometry.calibrate_by_rfid(card_id)
                    self._broadcast_queue.put(("point", int(card_id)))
                    self._show_oled(f"到达固定点\nID: {card_id}")
                    print(
                        f"[RFID] 读取任务卡 ID={card_id}，仅用于播报，"
                        "不修改导航当前位置"
                    )
            except Exception as exc:
                print(f"[RFID] 读取线程异常: {exc}")

    def _broadcast_loop(self) -> None:
        self._broadcast_thread_started.set()
        while self.running or not self._broadcast_queue.empty():
            try:
                event_type, event_id = self._broadcast_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self._broadcaster is not None:
                    if event_type == "point":
                        self._broadcaster.say_point(event_id)
                    elif event_type == "target":
                        self._broadcaster.say_target(event_id)
                    else:
                        print(f"[Audio WARN] 未知播报事件: {event_type}")
                else:
                    print(f"[Audio] 播报模块不可用，跳过 {event_type}:{event_id}")
            except Exception as exc:
                print(f"[Audio Error] {event_type}:{event_id} 播报失败: {exc}")
            finally:
                self._broadcast_queue.task_done()

    def _latest_camera_frame_after(self, after_sequence: int, timeout: float = 0.15):
        """Read a new shared navigation frame without consuming another worker's frame."""
        deadline = time.monotonic() + timeout
        while self.running:
            with self._camera_lock:
                sequence = self._camera_sequence
                if sequence != after_sequence and self._camera_frame is not None:
                    frame = self._camera_frame.copy()
                    return frame, sequence
                self._camera_ready.clear()
            if time.monotonic() >= deadline:
                return None, after_sequence
            self._camera_ready.wait(min(0.05, deadline - time.monotonic()))
        return None, after_sequence

    def _latest_camera_frame(self, timeout: float = 0.15):
        frame, sequence = self._latest_camera_frame_after(
            self._processed_camera_sequence, timeout
        )
        if frame is not None:
            self._processed_camera_sequence = sequence
        return frame

    def validate_hardware(self) -> None:
        if not self.chassis.serial or not self.chassis.serial.is_open:
            raise RuntimeError(f"底盘串口不可用: {self.args.serial_port}")
        if not self.camera.isOpened():
            raise RuntimeError(f"巡线摄像头不可用: {self.args.camera}")
        if self._culvert_camera is None or not self._culvert_camera.isOpened():
            raise RuntimeError(f"高位涵洞摄像头不可用: {self.args.culvert_camera}")

    def _wait_until_system_ready(self) -> bool:
        """保持停车，直到所有模块、工作线程和巡线帧率均已就绪。"""
        checks = (
            ("巡线采集线程", self._camera_thread_started, lambda: self._camera_thread.is_alive()),
            ("高位采集线程", self._culvert_camera_thread_started, lambda: self._culvert_camera_thread.is_alive()),
            ("巡线推理线程", self._road_thread_started, lambda: self._road_thread.is_alive()),
            ("涵洞识别线程", self._culvert_thread_started, lambda: self._culvert_thread.is_alive()),
            ("障碍识别线程", self._obstacle_thread_started, lambda: self._obstacle_thread.is_alive()),
            ("RFID线程", self._rfid_thread_started, lambda: self._rfid_thread.is_alive()),
            ("播报线程", self._broadcast_thread_started, lambda: self._broadcast_thread.is_alive()),
            ("巡线摄像头画面", self._camera_first_frame,
             lambda: time.monotonic() - self._last_camera_frame_at
             <= self.args.camera_stale_timeout),
            ("高位摄像头画面", self._culvert_camera_first_frame, lambda: True),
            ("巡线推理帧率", self._road_fps_stable, lambda: True),
        )
        modules = (
            (
                "RFID",
                self._rfid_reader is not None
                and self._rfid_reader.ser is not None
                and self._rfid_reader.ser.is_open,
            ),
            (
                "播报",
                self._broadcaster is not None
                and not getattr(self._broadcaster, "is_mock", True),
            ),
            (
                "OLED",
                self._oled is not None
                and getattr(self._oled, "is_ready", lambda: False)(),
            ),
            ("YOLO", self._yolo is not None and self._yolo.rknn is not None),
        )
        deadline = time.monotonic() + self.args.system_ready_timeout
        last_pending = None
        self._show_oled("全地图测试\n等待系统就绪", duration=1200)
        while self.running and time.monotonic() < deadline:
            self.tracker.pump_debug_window()
            self.chassis.stop()
            pending = [name for name, ready in modules if not ready]
            pending.extend(
                name for name, event, alive in checks if not event.is_set() or not alive()
            )
            if not pending:
                print(
                    f"[STARTUP READY] 全部模块和线程已就绪；"
                    f"巡线最近{self.args.startup_stable_frames}帧="
                    f"{self._startup_road_fps:.1f} FPS"
                )
                self._show_oled("全地图测试\n系统已就绪", duration=1200)
                return True
            pending_text = ", ".join(pending)
            if pending_text != last_pending:
                print(f"[STARTUP WAIT] 等待: {pending_text}")
                last_pending = pending_text
            time.sleep(0.1)

        self.chassis.stop()
        pending = [name for name, ready in modules if not ready]
        pending.extend(
            name for name, event, alive in checks if not event.is_set() or not alive()
        )
        print(f"[STARTUP ERROR] 系统未全部就绪: {', '.join(pending) or '任务被中止'}")
        self._show_oled("启动失败\n系统未就绪", duration=4000)
        return False

    def _runtime_health_error(self) -> str | None:
        """Return the failed critical component, or None while operation is safe."""
        serial_port = getattr(self.chassis, "serial", None)
        if serial_port is None or not serial_port.is_open:
            return "底盘串口已断开"

        workers = (
            ("巡线采集线程", self._camera_thread),
            ("高位采集线程", self._culvert_camera_thread),
            ("巡线推理线程", self._road_thread),
            ("涵洞识别线程", self._culvert_thread),
            ("障碍识别线程", self._obstacle_thread),
            ("RFID线程", self._rfid_thread),
            ("播报线程", self._broadcast_thread),
        )
        for name, worker in workers:
            if not worker.is_alive():
                return f"{name}已退出"

        rfid_serial = getattr(self._rfid_reader, "ser", None)
        if rfid_serial is None or not rfid_serial.is_open:
            return "RFID串口已断开"
        if self._broadcaster is None or getattr(self._broadcaster, "is_mock", True):
            return "播报模块不可用"
        if self._oled is None or not getattr(self._oled, "is_ready", lambda: False)():
            return "OLED显示进程已退出"

        now = time.monotonic()
        if now - self._last_camera_frame_at > self.args.camera_stale_timeout:
            return "巡线摄像头画面超时"
        if now - self._last_culvert_camera_frame_at > self.args.camera_stale_timeout:
            return "高位摄像头画面超时"
        if (
            not self._culvert_active.is_set()
            and now >= self._road_resume_grace_until
            and now - self._last_road_result_at > self.args.road_stale_timeout
        ):
            return "巡线推理结果超时"
        return None

    def stop(self, *_args: object) -> None:
        self.stop_requested = True
        self.running = False
        self.chassis.stop()

    def read_road(self) -> tuple[str, int] | None:
        # RKNN在双摄场景下单帧可能超过150ms；等待略放宽，避免把正常
        # 推理误判为连续丢帧。该等待不改变摄像头/推理线程的实际帧率。
        if not self._road_ready.wait(0.30):
            return None
        if self._obstacle_pending.is_set():
            return None
        with self._road_lock:
            if self._road_result_sequence == self._consumed_road_sequence:
                self._road_ready.clear()
                return None
            result = self._road_result
            self._consumed_road_sequence = self._road_result_sequence
            # 事件只表示“存在尚未消费的新结果”。生产者和消费者都在
            # _road_lock 内修改它，避免新结果到达时被错误清除。
            self._road_ready.clear()
            if result is None:
                return None
            return result

    def _current_physical_segment(self) -> tuple[str | None, str | None]:
        """Return the physical road currently occupied by the robot."""
        if self._obstacle_recovering and self._obstacle_return_from:
            return self._obstacle_return_from, self.current_node
        return self.current_node, self.current_target_node

    def _expected_junction_mode(self) -> str:
        """Select the junction profile for the current physical approach."""
        if self.state == "START_ALIGN":
            return "NORMAL"
        if self._tunnel_exit_junction_pending:
            return "TUNNEL_EXIT"
        from_node, to_node = self._current_physical_segment()
        if (
            frozenset((from_node, to_node)) in L_JUNCTION_SEGMENTS
            or (from_node, to_node) in L_JUNCTION_DIRECTIONS
        ):
            return "L"
        return "NORMAL"

    def follow_once(self, offset: int) -> None:
        correction = self.args.kp * offset + self.args.kd * (
            offset - self.last_offset
        )
        self.last_offset = offset
        correction = max(-self.args.max_angle, min(self.args.max_angle, correction))
        self.chassis.drive_steer(self.args.speed, correction)

    @staticmethod
    def _is_completion_ack(message: str | None) -> bool:
        """下位机完成应答固定为独立一行 ``@``。"""
        return bool(message) and message.strip() == "@"

    def _record_ack(
        self,
        *,
        action_id: int | None,
        action_name: str,
        sent_at: float | None,
        already_accepted: bool,
    ) -> bool:
        """记录并打印 ACK；返回当前 ACK 是否可完成对应动作。"""
        now = time.monotonic()
        elapsed = None if sent_at is None else max(0.0, now - sent_at)
        if action_id is None:
            status = "UNEXPECTED"
            accepted = False
        elif already_accepted:
            status = "DEBOUNCED_DUPLICATE"
            accepted = False
        elif elapsed is not None and elapsed < self.args.ack_min_latency:
            status = "DEBOUNCED_EARLY"
            accepted = False
        else:
            status = "ACCEPTED"
            accepted = True

        with self._ack_lock:
            self.ack_received_count += 1
            if accepted:
                self.ack_accepted_count += 1
            else:
                self.ack_debounced_count += 1
            received = self.ack_received_count
            accepted_total = self.ack_accepted_count
            debounced_total = self.ack_debounced_count

        elapsed_text = "-" if elapsed is None else f"{elapsed:.3f}s"
        action_text = "-" if action_id is None else f"#{action_id:03d}"
        print(
            f"[ACK RX #{received:04d}] action={action_text} "
            f"name={action_name} elapsed={elapsed_text} status={status}; "
            f"accepted={accepted_total} debounced={debounced_total}"
        )
        return accepted

    def wait_for_completion(
        self,
        timeout: float | None = None,
        *,
        action_id: int,
        action_name: str,
        sent_at: float,
    ) -> bool:
        """等待动作 ACK，并在接受后继续收集短时间内的重复 ACK。"""
        self.last_action_error = None
        deadline = time.monotonic() + (
            self.args.action_timeout if timeout is None else timeout
        )
        accepted_at = None
        while self.running:
            now = time.monotonic()
            if accepted_at is None:
                if now >= deadline:
                    break
            elif now - accepted_at >= self.args.ack_debounce:
                return True
            self.tracker.pump_debug_window()
            message = self.chassis.read_message()
            if message and message.startswith("E:"):
                try:
                    left, right = message[2:].split(",", 1)
                    self.odometry.update_encoders(int(left), int(right))
                except (TypeError, ValueError):
                    print(f"[ODOMETRY] 忽略无法解析的编码器回报: {message}")
            if message and message.startswith("ERR:"):
                self.last_action_error = message
                print(f"[WARN] 下位机动作失败: {message}")
                return False
            if self._is_completion_ack(message):
                accepted = self._record_ack(
                    action_id=action_id,
                    action_name=action_name,
                    sent_at=sent_at,
                    already_accepted=accepted_at is not None,
                )
                if accepted:
                    accepted_at = time.monotonic()
            time.sleep(0.01)
        return accepted_at is not None

    def _run_action(self, name: str, command, timeout: float | None = None) -> bool:
        # 清除旧 ACK，避免上一条动作的完成消息被当前动作误收。
        self.action_count += 1
        action_id = self.action_count
        print(f"[ACTION #{action_id:03d} SEND] {name}")
        self.chassis.clear_pending_input()
        sent_at = time.monotonic()
        command()
        if self.wait_for_completion(
            timeout,
            action_id=action_id,
            action_name=name,
            sent_at=sent_at,
        ):
            self._road_resume_grace_until = time.monotonic() + max(
                2.0, self.args.road_stale_timeout
            )
            print(f"[ACTION #{action_id:03d} DONE] {name}")
            return True
        if self.last_action_error:
            print(
                f"[ACTION #{action_id:03d} ERROR] {name} 被下位机拒绝: "
                f"{self.last_action_error}，立即停车"
            )
        else:
            print(f"[ACTION #{action_id:03d} TIMEOUT] {name}，立即停车")
        self.chassis.stop()
        return False

    def _set_culvert_road(self, from_node: str | None, to_node: str | None) -> None:
        """转向完成后更新识别场景；隧道关闭涵洞和嫌疑人检测。"""
        tunnel_id = (
            self.topo_map.tunnel_id_for_segment(from_node, to_node)
            if from_node and to_node
            else None
        )
        if tunnel_id:
            with self._culvert_road_lock:
                self._culvert_road = None
            with self._culvert_lock:
                self._culvert_detector.reset()
            print(
                f"[TUNNEL] {tunnel_id} 进入 {from_node} -> {to_node}，"
                "涵洞线检测=OFF，嫌疑人识别=OFF"
            )
            return
        road = None
        if from_node and to_node and self.topo_map.is_configured_culvert_edge(from_node, to_node):
            edge = self.topo_map.edge_key(from_node, to_node)
            if edge not in self.topo_map.culvert_edges and edge not in self._culvert_scanned_edges:
                road = (from_node, to_node)
        with self._culvert_road_lock:
            changed = road != self._culvert_road
            self._culvert_road = road
        if changed:
            with self._culvert_lock:
                self._culvert_detector.reset()
            print(f"[CULVERT ROAD] 当前道路={road or '-'} 识别={'ON' if road else 'OFF'}")

    def _run_culvert_scan(self, from_node: str, to_node: str) -> None:
        """普通道路确认涵洞后停车扫描嫌疑人并记录道路（支持同侧双目标）。"""
        edge = self.topo_map.edge_key(from_node, to_node)
        if edge in self._culvert_scanned_edges:
            print(f"[CULVERT] 道路 {from_node} -> {to_node} 已扫描，忽略重复触发")
            return
        self._culvert_scanned_edges.add(edge)
        self._culvert_active.set()
        scan_results: dict[str, list[int]] = {"left": [], "right": []}
        try:
            # 涵洞确认后再延迟约100 ms，给检测结果和画面留出稳定时间。
            if self.args.culvert_stop_delay > 0:
                time.sleep(self.args.culvert_stop_delay)
            self.chassis.stop()
            self._show_oled(f"发现涵洞\n{from_node} -> {to_node}", duration=1500)
            self.navigator.mark_culvert_edge(from_node, to_node)
            self.chassis._send_command("@|5|0|0|0#")
            time.sleep(0.15)
            self._ensure_yolo()
            self.chassis._send_command("@|10|0|0|0#")
            captured_sides: set[str] = set()
            deadline = time.monotonic() + self.args.culvert_scan_timeout
            while self.running and time.monotonic() < deadline:
                message = self.chassis.read_message()
                if message:
                    message = message.strip()
                    print(f"[CULVERT SERIAL RX] {message}")
                    if self._is_completion_ack(message):
                        self._record_ack(
                            action_id=None,
                            action_name="CULVERT_SCAN",
                            sent_at=None,
                            already_accepted=False,
                        )
                        continue
                    if message == "Y_START":
                        self._show_oled("云台扫描\n准备识别目标", duration=1500)
                    # 实车安装方向：Y_L(180度)朝小车右侧，Y_R(0度)朝小车左侧。
                    elif message == "Y_L" and "right" not in captured_sides:
                        captured_sides.add("right")
                        with self._culvert_camera_lock:
                            signal_sequence = self._culvert_camera_sequence
                        scan_frame, frame_sequence = self._latest_culvert_frame(
                            after_sequence=signal_sequence, min_new_frames=5, timeout=0.7
                        )
                        print(f"[YOLO SYNC] Y_L signal_seq={signal_sequence} capture_seq={frame_sequence}")
                        scan_results["right"] = self._detect_scan_target("左侧", "Y_L", scan_frame)
                    elif message == "Y_R" and "left" not in captured_sides:
                        captured_sides.add("left")
                        with self._culvert_camera_lock:
                            signal_sequence = self._culvert_camera_sequence
                        scan_frame, frame_sequence = self._latest_culvert_frame(
                            after_sequence=signal_sequence, min_new_frames=5, timeout=0.7
                        )
                        print(f"[YOLO SYNC] Y_R signal_seq={signal_sequence} capture_seq={frame_sequence}")
                        scan_results["left"] = self._detect_scan_target("右侧", "Y_R", scan_frame)
                    if message == "Y_END":
                        print(f"[CULVERT] 扫描完成，收到 Y_END，道路={from_node} -> {to_node}")
                        break
                time.sleep(0.02)
            else:
                print("[CULVERT WARN] 等待 Y_END 超时，恢复巡线")
            self._publish_scan_results(scan_results["left"], scan_results["right"])
        finally:
            with self._culvert_lock:
                self._culvert_detector.reset()
            self._culvert_rearm_at = time.monotonic() + max(0.0, self.args.culvert_rearm_delay)
            # 涵洞扫描主动暂停巡线 NPU；解除暂停后给推理线程一个短暂
            # 恢复窗口，避免第一帧结果尚未产出就被健康检查误停车。
            self._road_resume_grace_until = time.monotonic() + max(
                2.0, self.args.road_stale_timeout
            )
            print(f"[CULVERT] 扫描结束，冷却 {self.args.culvert_rearm_delay:.1f}s，期间禁止重复触发")
            self._culvert_triggered.clear()
            self._culvert_active.clear()
            self._set_culvert_road(None, None)

    @staticmethod
    def _target_text(target_id: int) -> str:
        if target_id < 1:
            return "未识别"
        if target_id <= 10:
            return f"{target_id}号嫌疑人"
        return f"{target_id - 10}号刀具"

    @classmethod
    def _targets_text(cls, target_ids: list[int] | int) -> str:
        """将单个或多个目标业务号转换为可读文本（例如 '3号嫌疑人+5号刀具'）。"""
        if isinstance(target_ids, int):
            target_ids = [target_ids] if target_ids > 0 else []
        valid = [int(tid) for tid in target_ids if int(tid) > 0]
        if not valid:
            return "未识别"
        return "+".join(cls._target_text(tid) for tid in valid)

    def _capture_culvert_frame(self):
        """收到 Y_L/Y_R 后立即抓取第一张有效帧。

        下位机是在云台到位后才发送 Y_L/Y_R；收到信号后只等待第一张
        真正的新帧，避免使用信号前的缓存画面，也不额外等待多帧。
        """
        with self._culvert_camera_lock:
            sequence = self._culvert_camera_sequence
        frame, _ = self._latest_culvert_frame(
            after_sequence=sequence, min_new_frames=1, timeout=0.3
        )
        if frame is None:
            return None
        if self.args.culvert_flip in (-1, 0, 1):
            frame = cv2.flip(frame, self.args.culvert_flip)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        print(f"[YOLO FRAME] 信号后立即抓拍，清晰度={score:.1f}")
        return frame

    def _detect_scan_target(
        self, side_name: str, trigger_message: str = "", frame=None
    ) -> list[int]:
        """执行单侧识别，返回本侧检出的有效目标列表（0~2个业务号）。"""
        supplied_frame = frame is not None
        if frame is None:
            frame = self._capture_culvert_frame()
        if frame is None:
            print(f"[YOLO ERROR] {side_name}拍照失败")
            return []
        else:
            frame = frame.copy()
            if supplied_frame and self.args.culvert_flip in (-1, 0, 1):
                frame = cv2.flip(frame, self.args.culvert_flip)
        print(
            f"[YOLO FRAME] {side_name}触发={trigger_message or '-'} "
            f"shape={frame.shape[1]}x{frame.shape[0]} "
            f"flip={self.args.culvert_flip} "
            f"save={'ON' if self.args.culvert_save_photos else 'OFF'}"
        )
        if self._yolo is None or self._yolo.rknn is None:
            print(f"[YOLO ERROR] {side_name}目标模型不可用")
            return []
        with self._npu_lock:
            if hasattr(self._yolo, "detect_targets"):
                raw_ids = self._yolo.detect_targets(frame)
                target_ids = [int(tid) for tid in raw_ids if int(tid) > 0]
            else:
                single_id = int(self._yolo.detect_target(frame))
                target_ids = [single_id] if single_id > 0 else []
            annotated = getattr(self._yolo, "last_annotated_frame", None)
            if annotated is None or annotated.size == 0:
                annotated = frame.copy()
            else:
                annotated = annotated.copy()

        text = self._targets_text(target_ids)
        id_label = ",".join(str(t) for t in target_ids) if target_ids else "-1"
        if self.args.culvert_save_photos:
            self._culvert_photo_dir.mkdir(parents=True, exist_ok=True)
            self._culvert_photo_count += 1
            stamp = time.strftime("%Y%m%d_%H%M%S")
            photo_path = self._culvert_photo_dir / (
                f"{stamp}_{self._culvert_photo_count:03d}_{trigger_message or 'scan'}_{side_name}.jpg"
            )
            cv2.putText(
                annotated,
                f"{side_name} {text} (ID={id_label})",
                (10, max(28, annotated.shape[0] - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255) if target_ids else (0, 165, 255),
                2,
            )
            if cv2.imwrite(str(photo_path), annotated):
                print(f"[YOLO PHOTO] 已保存识别效果照片: {photo_path}")
            else:
                print(f"[YOLO PHOTO WARN] 保存失败: {photo_path}")
        print(f"[YOLO] {side_name}识别结果: {text} (business_ids={target_ids})")
        self._show_oled(f"{side_name}识别\n{text}", duration=1500)
        return target_ids

    def _publish_scan_results(
        self, left_ids: list[int] | int, right_ids: list[int] | int
    ) -> None:
        if isinstance(left_ids, int):
            left_ids = [left_ids] if left_ids > 0 else []
        if isinstance(right_ids, int):
            right_ids = [right_ids] if right_ids > 0 else []

        left_text = self._targets_text(left_ids)
        right_text = self._targets_text(right_ids)
        self._show_oled(
            f"右:{left_text}\n左:{right_text}", duration=2500
        )
        detected: list[int] = []
        for target_id in (*left_ids, *right_ids):
            if target_id > 0 and target_id not in detected:
                detected.append(int(target_id))

        # 丢弃尚未播放的旧目标播报，避免新一轮扫描播报上一次结果。
        with self._broadcast_queue.mutex:
            pending = list(self._broadcast_queue.queue)
            self._broadcast_queue.queue.clear()
            self._broadcast_queue.queue.extend(
                event for event in pending if event[0] != "target"
            )
        if detected:
            for target_id in detected:
                self._broadcast_queue.put(("target", target_id))
        else:
            self._broadcast_queue.put(("target", -1))

    def execute_cross_action(
        self,
        action: str,
        cross_id: int,
        at_node: str,
        next_node: str,
    ) -> bool:
        planned_action = action

        if action == "FORWARD":
            # Code 7 confirms encoder distance with ACK; line following
            # resumes immediately at the same forward speed.
            self.action_count += 1
            action_id = self.action_count
            print(
                f"[CROSS #{cross_id:03d} DISPATCH] node={at_node} "
                f"planned={planned_action} dispatched={action} "
                f"method=chassis.go_cross(speed={self.args.speed}, distance_multiplier=1)"
            )
            print(
                f"[ACTION #{action_id:03d} SEND] CROSS#{cross_id:03d} "
                f"node={at_node} FORWARD"
            )
            self.chassis.clear_pending_input()
            action_name = f"CROSS#{cross_id:03d} node={at_node} FORWARD"
            sent_at = time.monotonic()
            self.chassis.go_cross(speed=self.args.speed, distance_multiplier=1)
            if not self.wait_for_completion(
                self.args.action_timeout,
                action_id=action_id,
                action_name=action_name,
                sent_at=sent_at,
            ):
                print(
                    f"[ACTION #{action_id:03d} TIMEOUT] CROSS#{cross_id:03d} "
                    f"node={at_node} FORWARD，立即停车"
                )
                self.chassis.stop()
                return False
            print(
                f"[ACTION #{action_id:03d} DONE] CROSS#{cross_id:03d} "
                f"node={at_node} FORWARD（编码器闭环）"
            )
            self._road_resume_grace_until = time.monotonic() + max(
                2.0, self.args.road_stale_timeout
            )
            # Resume without a settling pause; firmware code 7 keeps moving.
            self.chassis.drive_steer(self.args.speed, 0)
            self.last_offset = 0
            self._set_culvert_road(at_node, next_node)
            return True

        # J_TOP 返回出发区的转弯使用独立角度，和出发区一样走纯角度指令；
        # 其他普通路口仍保持底盘固定的 ±90° 转向。
        if next_node == "START_AREA" and action in {"LEFT", "RIGHT"}:
            signed_angle = (
                -self.args.return_start_turn_angle
                if action == "LEFT"
                else self.args.return_start_turn_angle
            )
            commands = {
                action: (
                    lambda angle=signed_angle: self.chassis.turn_pure(angle),
                    f"chassis.turn_pure({signed_angle})",
                )
            }
        else:
            commands = {
                "LEFT": (self.chassis.ccw, "chassis.ccw()"),
                "RIGHT": (self.chassis.cw, "chassis.cw()"),
                "U_TURN": (lambda: self.chassis.turn(180), "chassis.turn(180)"),
            }
        if next_node == "START_AREA" and self.args.return_start_approach_distance > 0:
            if not self._run_action(f"返回出发区转弯前进 speed={self.args.return_start_approach_speed} distance={self.args.return_start_approach_distance}", lambda: self.chassis.distance(val=self.args.return_start_approach_speed, dis=self.args.return_start_approach_distance), self.args.return_start_timeout):
                return False
        command_spec = commands.get(action)
        if command_spec is None:
            raise ValueError(f"未知路口动作: {action}")
        command, method = command_spec
        print(
            f"[CROSS #{cross_id:03d} DISPATCH] node={at_node} "
            f"planned={planned_action} dispatched={action} method={method}"
        )
        completed = self._run_action(
            f"CROSS#{cross_id:03d} node={at_node} {action} method={method}",
            command,
        )
        if completed:
            # 与原版 ReconRobot._execute_cross_action 保持相同的机械稳定时间，
            # 避免下一个路口命令紧跟在底盘完成 ACK 后发送。
            time.sleep(0.3)
            self._set_culvert_road(at_node, next_node)
        return completed

    def _sync_navigation_path(self) -> None:
        self.current_node = self.navigator.current_node
        self.previous_node = self.navigator.previous_node
        self.current_path = self.navigator.get_next_segment()
        self.current_target_node = (
            self.current_path[1]
            if self.current_path and len(self.current_path) > 1
            else None
        )

    def _handle_pending_obstacle(self) -> bool:
        """Stop, block the road, U-turn, and prepare the replanned detour."""
        with self._obstacle_pending_lock:
            pending = self._obstacle_pending_data
            self._obstacle_pending_data = None
            self._obstacle_pending.clear()
        if pending is None:
            return True
        (blocked_from, blocked_to), _result = pending
        if (
            self.state != "FOLLOW_LINE"
            or self.current_node != blocked_from
            or self.current_target_node != blocked_to
        ):
            print(
                f"[OBSTACLE] 丢弃过期事件 {blocked_from} -> {blocked_to}；"
                f"当前={self.current_node} -> {self.current_target_node}"
            )
            return True

        self.chassis.stop()
        self._set_culvert_road(None, None)
        print(f"[OBSTACLE] 封锁道路 {blocked_from} <-> {blocked_to}，开始重规划")
        if not self.navigator.on_obstacle(blocked_from, blocked_to):
            print("[OBSTACLE ERROR] 封锁后没有可执行的剩余任务路线")
            self.chassis.stop()
            return False

        self._sync_navigation_path()
        self._obstacle_recovering = True
        self._obstacle_return_from = blocked_to
        print(f"[OBSTACLE] 绕行路线={self.current_path}")
        if not self._run_action(
            f"障碍物前掉头 {blocked_from} <- {blocked_to}",
            lambda: self.chassis.turn_all(180),
        ):
            self._obstacle_recovering = False
            self._obstacle_return_from = None
            return False
        self._obstacle_rearm_at = time.monotonic() + self.args.obstacle_cooldown
        print(
            f"[OBSTACLE] 已掉头，沿线返回 {blocked_from}；"
            "返回路口时再执行绕行转向"
        )
        return True

    def _confirm_next_node(self, cross_id: int | None = None) -> str | None:
        expected = self.current_target_node
        from_node = self.current_node
        reached = self.navigator.advance_path()
        self._sync_navigation_path()
        prefix = (
            f"[CROSS #{cross_id:03d} ARRIVAL]"
            if cross_id is not None
            else "[ARRIVAL]"
        )
        if reached:
            progress = self.navigator.get_progress()
            print(
                f"{prefix} from={from_node} expected={expected} reached={reached} "
                f"next={self.current_target_node or '-'}; "
                f"固定点 {progress['visited']}/{progress['total']}; "
                f"道路覆盖 {progress['covered_edges']}/"
                f"{progress['required_edges']}"
            )
        elif expected:
            print(
                f"{prefix} from={from_node} expected={expected} reached=None "
                "status=FAILED"
            )
        return reached

    def _run_start_actions(self) -> bool:
        print("\n[START_STATE] 全地图测试出发区固定动作...")
        self._show_oled("全地图测试\n正在出发", duration=1200)
        actions = (
            (
                f"出发区前进 speed={self.args.start_forward_speed} "
                f"distance={self.args.start_forward_distance}",
                lambda: self.chassis.distance(
                    val=self.args.start_forward_speed,
                    dis=self.args.start_forward_distance,
                ),
            ),
            (
                f"出发区右转 angle={self.args.start_right_turn_angle} "
                "method=chassis.turn_pure()",
                lambda: self.chassis.turn_pure(self.args.start_right_turn_angle),
            ),
        )
        for name, command in actions:
            if not self._run_action(name, command, self.args.start_action_timeout):
                return False
            time.sleep(0.2)

        self.start_line_frames = 0
        self.state = "START_ALIGN"
        print(
            "[START_STATE] 前进和右转完成，进入姿态巡线；稳定后同步 J_TOP，"
            "J_TOP 转向由巡线完成，不再额外发送路口动作；"
            f"稳定帧数={self.args.start_line_required_frames}"
        )
        return True

    def _run_start_align(self, road_type: str, offset: int) -> None:
        self.follow_once(offset)
        if road_type not in {"LINE", "CROSS"}:
            return
        self.start_line_frames += 1
        if self.start_line_frames >= self.args.start_line_required_frames:
            # 实车会在该阶段依靠道路偏差自然完成 J_TOP 左转，且该过程不
            # 产生 CROSS 事件。这里同步逻辑节点，避免下一个实体路口再次
            # 被误认成 J_TOP，导致后续动作整体落后一站。
            reached = self._confirm_next_node()
            if reached != "J_TOP":
                print(
                    f"[START_SYNC ERROR] expected=J_TOP reached={reached}，停车"
                )
                self.chassis.stop()
                self.running = False
                return
            print(
                "[START_SYNC] logical_node=J_TOP next=N1; "
                "J_TOP 左转由视觉巡线完成，未发送额外路口动作"
            )
            self._cross_event_latched = road_type == "CROSS"
            print("[START_ALIGN] 姿态巡线稳定，开放后续路口识别")
            self._set_culvert_road("J_TOP", "N1")
            self.state = "FOLLOW_LINE"

    def _handle_crossroad(self) -> bool:
        self.cross_event_count += 1
        cross_id = self.cross_event_count
        print(
            f"\n[CROSS #{cross_id:03d} DETECT] "
            f"logical_previous={self.previous_node or '-'} "
            f"logical_current={self.current_node} "
            f"target={self.current_target_node or '-'}"
        )
        # The current detection uses the tunnel-exit profile; consume it so the
        # next junction returns to the normal or L profile.
        tunnel_exit_profile_used = self._tunnel_exit_junction_pending
        self._tunnel_exit_junction_pending = False
        if tunnel_exit_profile_used:
            print("[TUNNEL EXIT] 使用隧道出口专用路口阈值")
        if self._obstacle_recovering:
            if not self._obstacle_return_from or not self.current_target_node:
                print("[OBSTACLE ERROR] 返回路口时缺少绕行方向")
                self.chassis.stop()
                return False
            action = self.topo_map.turn_action(
                self._obstacle_return_from,
                self.current_node,
                self.current_target_node,
            )
            print(
                f"[OBSTACLE] 已返回 {self.current_node}；从封锁方向 "
                f"{self._obstacle_return_from} 转入 {self.current_target_node}，"
                f"action={action}"
            )
            if not self.execute_cross_action(
                action, cross_id, self.current_node, self.current_target_node
            ):
                return False
            # This CROSS is the junction we returned to after the obstacle,
            # not the target at the far end of the newly selected road. The
            # turn only enters current_node -> current_target_node; commit the
            # target after the next physical CROSS is actually observed.
            print(
                f"[OBSTACLE ROUTE ENTERED] current={self.current_node} "
                f"target={self.current_target_node}; position_commit=no"
            )
            self._obstacle_recovering = False
            self._obstacle_return_from = None
            self.state = "FOLLOW_LINE"
            return True
        expected = self.current_target_node
        if not expected:
            print("[NAVIGATION ERROR] 没有待确认的目标节点")
            return False

        arrived_from = self.current_node
        if arrived_from is None:
            print(
                f"[NAVIGATION ERROR] 当前节点缺失，"
                "无法计算路口动作"
            )
            self.chassis.stop()
            return False
        preview_reached, preview_path, preview_complete = (
            self.navigator.preview_after_advance()
        )
        if preview_reached != expected:
            print(
                f"[NAVIGATION ERROR] 位置预演失败: expected={expected} "
                f"reached={preview_reached}"
            )
            self.chassis.stop()
            return False
        next_node = (
            preview_path[1]
            if preview_path and len(preview_path) > 1
            else None
        )
        if not next_node:
            print(
                f"[NAVIGATION ERROR] 到达 {expected} 后没有后续路线; "
                f"preview_complete={preview_complete}"
            )
            self.chassis.stop()
            return False

        action = self.topo_map.turn_action(arrived_from, expected, next_node)
        print(
            f"[CROSS #{cross_id:03d} PLAN] {arrived_from} -> {expected} -> "
            f"{next_node} action={action}"
        )
        exited_tunnel = self.topo_map.is_tunnel_segment(arrived_from, expected)
        self._set_culvert_road(None, None)
        if not self.execute_cross_action(action, cross_id, expected, next_node):
            return False
        if exited_tunnel:
            self._tunnel_exit_junction_pending = True
            print(
                "[TUNNEL EXIT] 已离开隧道，下一段道路使用 "
                "TUNNEL_EXIT_CROSS_WIDTH_RATIO 路口阈值"
            )

        # Commit the logical node only after the corresponding chassis action
        # has completed successfully.
        reached = self._confirm_next_node(cross_id)
        if reached is None or reached != expected:
            return False
        print(
            f"[POSITION COMMIT] motion complete; current={reached} "
            f"next={self.current_target_node or '-'}"
        )
        if next_node == "START_AREA":
            # J_TOP 的转向完成后只走一小段回到出发区，随后结束任务。
            return self._run_return_start()
        self.state = "FOLLOW_LINE"
        return True

    def _run_return_start(self) -> bool:
        print(
            "[FINISH] J_TOP 转向完成，短距离返回出发区: "
            f"speed={self.args.return_start_speed}, "
            f"distance={self.args.return_start_distance}"
        )
        command = lambda: self.chassis.distance(
            val=self.args.return_start_speed,
            dis=self.args.return_start_distance,
        )
        if not self._run_action(
            f"最终返回出发区 speed={self.args.return_start_speed} "
            f"distance={self.args.return_start_distance}",
            command,
            self.args.return_start_timeout,
        ):
            return False
        reached = self._confirm_next_node()
        if reached != "START_AREA" or not self.navigator.mission_complete:
            print(
                f"[MISSION ERROR] 返回确认失败: reached={reached}, "
                f"complete={self.navigator.mission_complete}"
            )
            return False
        print("[MISSION] 全地图测试完成")
        self._show_oled("全地图完成\n已返回出发区", duration=4000)
        self.state = "STOP"
        return True

    def run_mission(self) -> int:
        if not self._wait_until_system_ready():
            return 2

        for remaining in range(self.args.countdown, 0, -1):
            self.chassis.stop()
            print(f"[READY] {remaining} 秒后启动...", end="\r", flush=True)
            time.sleep(1)
        print(" " * 50, end="\r")

        if not self._run_start_actions():
            self._show_oled("启动失败\n底盘动作异常", duration=4000)
            return 3

        self._show_oled("全地图测试\n巡检进行中", duration=1200)

        missed_frames = 0
        while self.running and self.state != "STOP":
            self._pump_obstacle_debug()
            self.tracker.pump_debug_window()
            health_error = self._runtime_health_error()
            if health_error is not None:
                print(f"\n[HEALTH ERROR] {health_error}，立即停车")
                self.chassis.stop()
                self._show_oled(f"运行停止\n{health_error}", duration=4000)
                return 5
            if self._culvert_active.is_set():
                # 涵洞扫描期间禁止主线程发送转向/巡线指令，避免抢占下位机任务。
                time.sleep(0.05)
                continue
            if self._obstacle_pending.is_set():
                if not self._handle_pending_obstacle():
                    return 6
                continue
            road = self.read_road()
            # An obstacle may have arrived while waiting for road inference.
            # Handle it before consuming an old road result or counting a miss.
            if self._obstacle_pending.is_set():
                if not self._handle_pending_obstacle():
                    return 6
                continue
            if road is None:
                missed_frames += 1
                if missed_frames >= self.args.max_camera_misses:
                    print("\n[ERROR] 摄像头连续取帧失败，停车")
                    self._show_oled("测试停止\n摄像头取帧失败", duration=4000)
                    return 2
                time.sleep(0.02)
                continue

            missed_frames = 0
            road_type, offset = road
            if self.state == "START_ALIGN":
                self._run_start_align(road_type, offset)
                continue

            if road_type == "CROSS":
                if self._cross_event_latched:
                    # The same physical junction may remain visible for many
                    # frames. Keep driving out of it, but dispatch only one
                    # chassis action and therefore only one position commit.
                    self.follow_once(offset)
                    time.sleep(0.02)
                    continue
                self._cross_event_latched = True
                # 直接执行路口动作，避免状态切换时先发零速造成停顿。
                if not self._handle_crossroad():
                    return 4
                continue

            if road_type == "LINE" and self._cross_event_latched:
                self._cross_event_latched = False
                print("[CROSS GATE] LINE observed; next junction armed")
            self.follow_once(offset)
            now = time.monotonic()
            if now - self._last_line_log_time >= 0.25:
                print(
                    f"[LINE] 状态={road_type:<5} 偏差={offset:>4} "
                    f"目标={self.current_target_node or '-'} "
                    f"路口模式={self.tracker.junction_mode}",
                    end="\r",
                    flush=True,
                )
                self._last_line_log_time = now
            if self.tracker.last_key in (ord("q"), ord("Q"), 27):
                break
            time.sleep(0.02)

        return 0 if self.state == "STOP" and self.navigator.mission_complete else 1

    def close(self) -> None:
        self.running = False
        try:
            self.chassis.stop()
        finally:
            time.sleep(0.1)
            self.camera.release()
            if self._culvert_camera is not None:
                self._culvert_camera.release()
            if self._camera_thread.is_alive():
                self._camera_thread.join(timeout=1.0)
            if self._culvert_camera_thread.is_alive():
                self._culvert_camera_thread.join(timeout=1.0)
            if self._road_thread.is_alive():
                self._road_thread.join(timeout=1.0)
            if self._culvert_thread.is_alive():
                self._culvert_thread.join(timeout=1.0)
            if self._obstacle_thread.is_alive():
                self._obstacle_thread.join(timeout=1.0)
            if self._rfid_thread.is_alive():
                self._rfid_thread.join(timeout=1.0)
            # 退出时不再启动新的播报；否则清理完成后后台线程仍可能
            # 访问已经关闭的音频设备。
            while True:
                try:
                    self._broadcast_queue.get_nowait()
                    self._broadcast_queue.task_done()
                except queue.Empty:
                    break
            if self._broadcast_thread.is_alive():
                self._broadcast_thread.join(timeout=5.0)
            self._rfid_reader.disconnect()
            self.tracker.release()
            if self._yolo is not None:
                self._release_yolo()
            if self._oled is not None:
                self._oled.close()
            cv2.destroyAllWindows()
            self.chassis.close()
            with self._ack_lock:
                print(
                    f"[ACK SUMMARY] received={self.ack_received_count} "
                    f"accepted={self.ack_accepted_count} "
                    f"debounced={self.ack_debounced_count}"
                )
            print("[SAFE] 电机已停车，摄像头、识别器和串口已释放")

    def run(self) -> int:
        try:
            self.validate_hardware()
            print(
                f"[READY] 摄像头={self.args.camera} 串口={self.args.serial_port} "
                f"巡线速度={self.args.speed} "
                f"障碍识别={'ON' if self.args.obstacle_enabled else 'OFF'}"
            )
            route, progress = calculate_plan()
            print(
                f"[FULL MAP TEST] 规划节点数={len(route)}，"
                f"需覆盖道路={progress['required_edges']}"
            )
            result = self.run_mission()
            if result != 0:
                if self.stop_requested:
                    self._show_oled("全地图测试\n用户已中止", duration=2500)
                else:
                    self._show_oled(
                        f"全地图测试停止\n错误码: {result}", duration=4000
                    )
            return result
        finally:
            self.close()


def run_live() -> int:
    robot = None
    try:
        robot = FullMapWalkTest(FullMapConfig())
        signal.signal(signal.SIGINT, robot.stop)
        signal.signal(signal.SIGTERM, robot.stop)
        return robot.run()
    except KeyboardInterrupt:
        print("\n[FULL MAP TEST] 用户中止")
        if robot is not None:
            robot._show_oled("全地图测试\n用户已中止", duration=2500)
        return 130
    except Exception:
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # run() normally owns cleanup; this also handles constructor failures.
        if robot is not None and robot.running:
            robot.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="只打印完整覆盖路线，不初始化摄像头、串口或识别器",
    )
    args = parser.parse_args()
    if args.plan_only:
        return 0 if print_plan() else 1

    return run_live()


if __name__ == "__main__":
    raise SystemExit(main())