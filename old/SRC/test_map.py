"""
带随机涵洞和随机障碍处理的全地图行走测试。

该测试沿用 line_follow_test.py 的硬件调用方式：巡线模块直接读取
摄像头，底盘动作直接通过 ChassisController 发送，并在每个动作后轮询
下位机的 @ 完成应答。地图规划只使用物理道路和 N1～N12 固定点，完成
条件为覆盖全部道路、经过所有固定点并返回出发区。
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass

import cv2

from old.SRC.Serial import ChassisController
from old.SRC.line_detect import LineTracker
from old.SRC.navigator import Navigator
from old.SRC.topo_map import TopoMap
from old.SRC.odometry import Odometry
from old.SRC.rfid import RFIDReader

try:
    from old.SRC.yolo_detect import YoloDetector
except Exception as exc:
    YoloDetector = None
    print(f"[YOLO WARN] 无法导入目标识别模块: {exc}")

try:
    from old.SRC.oled import OLEDDisplay
except Exception as exc:
    OLEDDisplay = None
    print(f"[OLED WARN] 无法导入显示模块: {exc}")

try:
    from old.SRC.broadcast import Broadcaster
except Exception as exc:
    Broadcaster = None
    print(f"[Audio WARN] 无法导入播报模块: {exc}")

# 添加 test_code 目录到 sys.path（涵洞检测和障碍检测）
_TEST_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_code")
if _TEST_CODE_DIR not in sys.path:
    sys.path.insert(0, _TEST_CODE_DIR)

from old.SRC.Culvert_detect import CulvertBrightnessDetector
from old.SRC.obstacle_color_test import ColorObstacleDetector


# ============================================================================
# 第一部分：辅助函数和工具
# ============================================================================

def _env_bool(name: str, default: bool) -> bool:
    """从环境变量读取布尔值。"""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _parse_camera_source(value: str) -> int | str:
    """解析摄像头来源（整数索引或设备路径）。"""
    value = str(value).strip()
    try:
        return int(value)
    except ValueError:
        return value


def _parse_flip(value: str) -> int | None:
    """解析摄像头翻转参数。"""
    value = str(value).strip().lower()
    if value in {"none", "off", "false", "no"}:
        return None
    code = int(value)
    if code not in (-1, 0, 1):
        raise ValueError("flip 只能填写 None、-1、0 或 1")
    return code


# ============================================================================
# 第二部分：配置类
# ============================================================================

@dataclass
class FullMapConfig:
    """全地图测试配置参数，全部从环境变量读取。"""
    
    # ---- 摄像头配置 ----
    camera: int | str = _parse_camera_source(os.environ.get("NAV_CAMERA", "0"))                # 巡线摄像头
    flip: int | None = _parse_flip(os.environ.get("NAV_CAMERA_FLIP", "none"))                  # 图像翻转
    
    # ---- 串口配置 ----
    serial_port: str = os.environ.get("CHASSIS_PORT", "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A5069RR4-if00-port0")
    baudrate: int = 115200
    
    # ---- 巡线PID控制 ----
    speed: int = int(os.environ.get("FULL_MAP_LINE_SPEED", "10"))                              # 巡线速度
    kp: float = float(os.environ.get("FULL_MAP_KP", "0.35"))                                   # 比例系数
    kd: float = float(os.environ.get("FULL_MAP_KD", "0.10"))                                   # 微分系数
    max_angle: float = float(os.environ.get("FULL_MAP_MAX_ANGLE", "30"))                       # 最大转向角度
    
    # ---- 超时配置 ----
    action_timeout: float = float(os.environ.get("FULL_MAP_ACTION_TIMEOUT", "10"))             # 动作超时
    system_ready_timeout: float = float(os.environ.get("SYSTEM_READY_TIMEOUT", "20"))          # 系统就绪超时
    camera_stale_timeout: float = float(os.environ.get("CAMERA_STALE_TIMEOUT", "1.0"))         # 摄像头画面超时
    road_stale_timeout: float = float(os.environ.get("ROAD_STALE_TIMEOUT", "1.5"))             # 巡线结果超时
    
    # ---- 帧率检测 ----
    startup_min_road_fps: float = float(os.environ.get("STARTUP_MIN_ROAD_FPS", "10"))          # 最低帧率要求
    startup_stable_frames: int = max(2, int(os.environ.get("STARTUP_STABLE_FRAMES", "15")))    # 采样帧数
    
    # ---- 出发区动作 ----
    start_action_timeout: float = float(os.environ.get("START_ACTION_TIMEOUT", "8"))           # 出发动作超时
    start_line_required_frames: int = max(1, int(os.environ.get("START_LINE_STABLE_FRAMES", "30")))  # 姿态稳定帧数
    start_right_turn_angle: int = int(os.environ.get("START_RIGHT_TURN_ANGLE", "80"))          # 右转角度
    start_forward_speed: int = int(os.environ.get("START_FORWARD_SPEED", "10"))                # 前进速度
    start_forward_distance: int = int(os.environ.get("START_FORWARD_DISTANCE", "120"))         # 前进距离
    
    # ---- 返回出发区 ----
    return_start_speed: int = int(os.environ.get("RETURN_START_SPEED", "10"))
    return_start_distance: int = int(os.environ.get("RETURN_START_DISTANCE", "100"))
    return_start_timeout: float = float(os.environ.get("RETURN_START_TIMEOUT", "6"))
    
    # ---- 路口控制（关键参数） ----
    cross_rearm_delay: float = max(0.0, float(os.environ.get("FULL_MAP_CROSS_REARM_DELAY", "0.5")))          # 路口重新激活延迟
    cross_rearm_line_frames: int = max(1, int(os.environ.get("FULL_MAP_CROSS_REARM_LINE_FRAMES", "2")))   # 需要连续LINE帧数
    cross_rearm_max_offset: int = max(0, int(os.environ.get("FULL_MAP_CROSS_REARM_MAX_OFFSET", "35")))    # 允许最大偏移
    cross_forward_distance_multiplier: int = max(1, int(os.environ.get("FULL_MAP_CROSS_FORWARD_DISTANCE", "1")))  # 直行距离倍数
    
    # ---- 通用 ----
    countdown: int = int(os.environ.get("FULL_MAP_COUNTDOWN", "3"))
    max_camera_misses: int = int(os.environ.get("FULL_MAP_MAX_CAMERA_MISSES", "10"))
    debug: bool = _env_bool("LINE_DEBUG", False)
    
    # ---- 涵洞检测 ----
    culvert_camera: int | str = _parse_camera_source(os.environ.get("CULVERT_CAMERA", "2"))    # 高位涵洞摄像头
    culvert_flip: int | None = _parse_flip(os.environ.get("CULVERT_CAMERA_FLIP", "none"))
    culvert_min_length_ratio: float = float(os.environ.get("CULVERT_MIN_LENGTH_RATIO", "0.35")) # 涵洞最小长度比例
    culvert_detect_interval: int = max(1, int(os.environ.get("CULVERT_DETECT_INTERVAL", "1")))  # 检测间隔（帧）
    culvert_debug: bool = _env_bool("CULVERT_DEBUG", False)
    culvert_scan_timeout: float = float(os.environ.get("CULVERT_SCAN_TIMEOUT", "20"))           # 扫描超时
    culvert_rearm_delay: float = float(os.environ.get("CULVERT_REARM_DELAY", "3"))              # 冷却时间
    
    # ---- 障碍物检测 ----
    obstacle_enabled: bool = _env_bool("OBSTACLE_DETECT_ENABLED", True)
    obstacle_debug: bool = _env_bool("OBSTACLE_DEBUG", False)
    obstacle_min_area: float = float(os.environ.get("OBSTACLE_MIN_AREA", "200"))                # 最小面积（像素）
    obstacle_roi_left: float = float(os.environ.get("OBSTACLE_ROI_LEFT", "0.10"))               # ROI区域
    obstacle_roi_top: float = float(os.environ.get("OBSTACLE_ROI_TOP", "0.80"))
    obstacle_roi_right: float = float(os.environ.get("OBSTACLE_ROI_RIGHT", "0.90"))
    obstacle_roi_bottom: float = float(os.environ.get("OBSTACLE_ROI_BOTTOM", "1.0"))
    obstacle_confirm_frames: int = max(1, int(os.environ.get("OBSTACLE_CONFIRM_FRAMES", "3")))  # 确认所需连续帧数
    obstacle_cooldown: float = max(0.0, float(os.environ.get("OBSTACLE_COOLDOWN", "1.5")))      # 冷却时间
    
    # ---- RFID ----
    rfid_port: str = os.environ.get("RFID_PORT", "/dev/ttyUSB2")
    rfid_baudrate: int = int(os.environ.get("RFID_BAUDRATE", "115200"))


# ============================================================================
# 第三部分：L型路口配置（常量）
# ============================================================================

# L型路口：需要特殊转弯逻辑的路段
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

# 不对称L型方向：向外方向用L型，返回时用普通十字路口
L_JUNCTION_DIRECTIONS = frozenset({
    ("I1_1", "N1"),
    ("I12_1", "N12"),
})

# OpenCV VideoWriter fourcc 函数
_video_writer_fourcc = getattr(cv2, "VideoWriter_fourcc")


# ============================================================================
# 第四部分：控制台过滤
# ============================================================================

def _console_line_allowed(line: str) -> bool:
    """控制台输出过滤：只显示转向和障碍物相关消息。"""
    if "[OBSTACLE" in line or "障碍" in line:
        return True
    if any(token in line for token in ("LEFT", "RIGHT", "U_TURN", "ccw()", "cw()", "turn_pure", "左转", "右转", "掉头", "转向")):
        return True
    keep_errors = os.environ.get("FULL_MAP_OUTPUT_KEEP_ERRORS", "0").strip().lower()
    return keep_errors not in {"0", "false", "off", "no"} and any(
        token in line for token in ("[ERROR", "[WARN", "Traceback", "Exception")
    )


class _FilteredConsole:
    """过滤控制台输出的包装类。"""
    def __init__(self, target):
        self._target = target
        self._buffer = ""

    def write(self, data):
        if not data:
            return 0
        self._buffer += str(data)
        while True:
            separators = [idx for idx in (self._buffer.find("\n"), self._buffer.find("\r")) if idx >= 0]
            if not separators:
                break
            idx = min(separators)
            line = self._buffer[:idx]
            self._buffer = self._buffer[idx + 1:]
            if _console_line_allowed(line):
                self._target.write(line + "\n")
        return len(data)

    def flush(self):
        if self._buffer:
            if _console_line_allowed(self._buffer):
                self._target.write(self._buffer)
            self._buffer = ""
        self._target.flush()

    def isatty(self):
        return self._target.isatty()

    def __getattr__(self, name):
        return getattr(self._target, name)


def _install_console_filter() -> None:
    """安装控制台过滤器。"""
    mode = os.environ.get("FULL_MAP_OUTPUT", "turn_obstacle").strip().lower()
    if mode in {"all", "full", "verbose", "0", "off"}:
        return
    sys.stdout = _FilteredConsole(sys.stdout)
    sys.stderr = _FilteredConsole(sys.stderr)


_install_console_filter()
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


@contextmanager
def _silence_native_console():
    """静音原生库直接写入stdout/stderr的输出。"""
    mode = os.environ.get("FULL_MAP_OUTPUT", "turn_obstacle").strip().lower()
    if mode in {"all", "full", "verbose", "0", "off"}:
        yield
        return
    stdout_fd = sys.__stdout__.fileno()
    stderr_fd = sys.__stderr__.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        with open(os.devnull, "w", encoding="utf-8") as sink:
            os.dup2(sink.fileno(), stdout_fd)
            os.dup2(sink.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


# ============================================================================
# 第五部分：导航器构建和路线规划
# ============================================================================

def build_test_navigator():
    """
    建立测试导航器。
    - 无绿色任务点
    - 要求覆盖全部道路
    - N12→J_TOP 返航支路延后覆盖
    """
    topo_map = TopoMap()
    topo_map.build_competition_track(green_count=0)
    navigator = Navigator(topo_map)
    navigator.init_mission(
        start_node="START_AREA",
        mission_plan=topo_map.fixed_patrol_nodes,
        expected_green_count=int(os.environ.get("CULVERT_EXPECTED_COUNT", "8")),
        cover_all_edges=True,
    )
    navigator.defer_coverage_edges([("N12", "J_TOP")])
    return topo_map, navigator


def calculate_plan():
    """离线计算完整路线。"""
    topo_map, navigator = build_test_navigator()
    route = [str(navigator.current_node)]
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
    """打印规划路线。"""
    route, progress = calculate_plan()
    print("[FULL MAP TEST] 随机涵洞、随机障碍全道路巡检任务")
    print("[FULL MAP TEST] 路线: " + " -> ".join(route))
    print(f"[FULL MAP TEST] 道路覆盖: {progress['covered_edges']}/{progress['required_edges']} ({progress['coverage_percent']:.1f}%)")
    print(f"[FULL MAP TEST] 固定点: {progress['visited']}/{progress['total']}; 缺失={progress['missing']}")
    print(f"[FULL MAP TEST] 返回出发区并完成: {progress['complete']}")
    return bool(progress["complete"])


# ============================================================================
# 第六部分：主测试类 FullMapWalkTest
# ============================================================================

class FullMapWalkTest:
    """
    全地图行走测试主类。
    包含：摄像头采集、巡线推理、涵洞检测、障碍物检测、RFID、播报等多线程。
    """
    
    # ------------------------------------------------------------------------
    # 6.1 初始化
    # ------------------------------------------------------------------------
    
    def __init__(self, args: FullMapConfig):
        self.args = args
        
        # ---- 运行状态 ----
        self.running = True                     # 主循环运行标志
        self.stop_requested = False             # 用户请求停止
        self.state = "START"                    # 状态机：START → START_ALIGN → FOLLOW_LINE → STOP
        
        # ---- 巡线状态 ----
        self.last_offset = 0                    # 上一次偏移量（微分计算用）
        self.last_action_error = None           # 最后一次动作错误
        
        # ---- 路口门控 ----
        self._cross_event_latched = False       # 路口事件锁，防止重复触发
        self._cross_clear_frames = 0            # 连续清晰帧计数
        self._cross_rearm_at = 0.0              # 重新激活时间
        
        # ---- 统计 ----
        self.cross_event_count = 0              # 路口事件计数
        self.action_count = 0                   # 动作计数
        self.start_line_frames = 0              # 出发区稳定帧计数
        
        # ---- 帧率检测 ----
        self._road_frame_times = deque(maxlen=args.startup_stable_frames)
        self._road_fps_stable = threading.Event()
        self._startup_road_fps = 0.0
        
        # ---- 健康检查 ----
        self._last_camera_frame_at = 0.0
        self._last_culvert_camera_frame_at = 0.0
        self._last_road_result_at = 0.0
        self._road_resume_grace_until = 0.0     # 恢复后的宽限期
        
        # ---- 线程启动信号 ----
        self._camera_thread_started = threading.Event()
        self._culvert_camera_thread_started = threading.Event()
        self._road_thread_started = threading.Event()
        self._culvert_thread_started = threading.Event()
        self._obstacle_thread_started = threading.Event()
        self._rfid_thread_started = threading.Event()
        self._broadcast_thread_started = threading.Event()
        
        # ---- 里程计 ----
        self.odometry = Odometry()
        
        # ---- 涵洞检测器 ----
        self._culvert_detector = CulvertBrightnessDetector(
            consistency_threshold=float(os.environ.get("CULVERT_BRIGHTNESS_THRESHOLD", "0.12")),
            edge_threshold=float(os.environ.get("CULVERT_EDGE_THRESHOLD", "0.012")),
            min_consecutive=3,
        )
        self._culvert_camera = None
        self._culvert_camera_lock = threading.Lock()
        self._culvert_camera_ready = threading.Event()
        self._culvert_camera_first_frame = threading.Event()
        self._culvert_camera_frame = None
        self._culvert_camera_sequence = 0
        
        # ---- 涵洞控制 ----
        self._culvert_lock = threading.Lock()
        self._culvert_triggered = threading.Event()     # 涵洞触发标志
        self._culvert_active = threading.Event()        # 涵洞扫描进行中
        self._culvert_road_lock = threading.Lock()
        self._culvert_road = None                       # 当前涵洞道路
        self._culvert_scanned_edges = set()             # 已扫描的道路
        self._culvert_rearm_at = 0.0
        
        # ---- 障碍物检测 ----
        self._obstacle_detector = ColorObstacleDetector(
            min_area=args.obstacle_min_area,
            min_area_ratio=args.obstacle_min_area_ratio,
            roi_left_ratio=args.obstacle_roi_left,
            roi_top_ratio=args.obstacle_roi_top,
            roi_right_ratio=args.obstacle_roi_right,
            roi_bottom_ratio=args.obstacle_roi_bottom,
        )
        self._obstacle_pending = threading.Event()      # 有障碍物待处理
        self._obstacle_pending_lock = threading.Lock()
        self._obstacle_pending_data = None
        self._obstacle_rearm_at = 0.0
        self._obstacle_recovering = False               # 正在绕行恢复
        self._obstacle_return_from = None               # 障碍物方向
        
        # ---- NPU锁 ----
        self._npu_lock = threading.Lock()
        
        # ---- 硬件初始化 ----
        self.chassis = ChassisController(args.serial_port, args.baudrate)
        with _silence_native_console():
            self.tracker = LineTracker(debug_mode=args.debug)
        
        # ---- YOLO目标检测 ----
        self._yolo = None
        if YoloDetector is not None:
            try:
                with _silence_native_console():
                    self._yolo = YoloDetector(preset_classes=tuple(range(1, 21)), show_window=False, save_debug=False)
                if self._yolo.rknn is None:
                    self._yolo = None
                    print("[YOLO WARN] 目标模型预加载失败，涵洞扫描时将重试")
                else:
                    print("[YOLO] 目标模型已预加载")
            except Exception as exc:
                self._yolo = None
                print(f"[YOLO WARN] 目标模型预加载失败，涵洞扫描时将重试: {exc}")
        
        # ---- 巡线摄像头 ----
        backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        try:
            self.camera = cv2.VideoCapture(args.camera, backend)
        except Exception:
            self.camera = cv2.VideoCapture(args.camera)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_FOURCC, _video_writer_fourcc(*"MJPG"))
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # ---- 巡线摄像头共享帧 ----
        self._camera_lock = threading.Lock()
        self._camera_ready = threading.Event()
        self._camera_frame = None
        self._camera_sequence = 0
        self._processed_camera_sequence = 0
        
        # ---- 巡线结果共享 ----
        self._road_lock = threading.Lock()
        self._road_ready = threading.Event()
        self._road_result = None
        self._road_result_sequence = 0
        self._consumed_road_sequence = 0
        
        # ---- 高位涵洞摄像头 ----
        backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        try:
            self._culvert_camera = cv2.VideoCapture(args.culvert_camera, backend)
        except Exception:
            self._culvert_camera = cv2.VideoCapture(args.culvert_camera)
        self._culvert_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._culvert_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self._culvert_camera.set(cv2.CAP_PROP_FPS, 15)
        self._culvert_camera.set(cv2.CAP_PROP_FOURCC, _video_writer_fourcc(*"MJPG"))
        self._culvert_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # ---- RFID ----
        self._rfid_reader = RFIDReader(port=args.rfid_port, baudrate=args.rfid_baudrate)
        
        # ---- 播报 ----
        self._broadcast_queue = queue.Queue()
        self._broadcaster = Broadcaster() if Broadcaster is not None else None
        
        # ---- OLED ----
        self._oled_lock = threading.Lock()
        self._oled = None
        if OLEDDisplay is not None:
            try:
                self._oled = OLEDDisplay()
            except Exception as exc:
                print(f"[OLED WARN] 显示模块初始化失败: {exc}")
        
        # ---- 导航器 ----
        self.topo_map, self.navigator = build_test_navigator()
        self.current_node = "START_AREA"
        self.previous_node = None
        self.current_path = None
        self.current_target_node = None
        self._sync_navigation_path()
        
        # ---- 启动所有工作线程 ----
        self._camera_thread = threading.Thread(target=self._camera_capture_loop, name="camera-capture", daemon=True)
        self._culvert_camera_thread = threading.Thread(target=self._culvert_camera_capture_loop, name="culvert-camera-capture", daemon=True)
        self._road_thread = threading.Thread(target=self._road_processing_loop, name="road-processing", daemon=True)
        self._culvert_thread = threading.Thread(target=self._culvert_loop, name="culvert-detect", daemon=True)
        self._obstacle_thread = threading.Thread(target=self._obstacle_loop, name="obstacle-detect", daemon=True)
        self._rfid_thread = threading.Thread(target=self._rfid_loop, name="rfid-reader", daemon=True)
        self._broadcast_thread = threading.Thread(target=self._broadcast_loop, name="broadcast", daemon=True)
        
        self._camera_thread.start()
        self._culvert_camera_thread.start()
        self._road_thread.start()
        self._culvert_thread.start()
        self._obstacle_thread.start()
        self._rfid_thread.start()
        self._broadcast_thread.start()
        
        self._show_oled("全地图测试\n正在准备系统", duration=1200)
    
    # ------------------------------------------------------------------------
    # 6.2 辅助方法
    # ------------------------------------------------------------------------
    
    def _show_oled(self, message: str, duration: int = 1200) -> None:
        """显示OLED信息。"""
        if self._oled is None:
            return
        with self._oled_lock:
            try:
                self._oled.show_text(message, duration=duration)
            except Exception as exc:
                print(f"[OLED WARN] 显示失败: {exc}")
    
    def _sync_navigation_path(self) -> None:
        """同步导航路径到本地变量。"""
        self.current_node = self.navigator.current_node
        self.previous_node = self.navigator.previous_node
        self.current_path = self.navigator.get_next_segment()
        self.current_target_node = self.current_path[1] if self.current_path and len(self.current_path) > 1 else None
    
    # ------------------------------------------------------------------------
    # 6.3 摄像头采集线程
    # ------------------------------------------------------------------------
    
    def _camera_capture_loop(self) -> None:
        """巡线摄像头采集线程。"""
        self._camera_thread_started.set()
        while self.running:
            ok, frame = self.camera.read()
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.005)
                continue
            if self.args.flip in (-1, 0, 1):
                frame = cv2.flip(frame, self.args.flip)
            with self._camera_lock:
                self._camera_frame = frame
                self._camera_sequence += 1
                self._last_camera_frame_at = time.monotonic()
            self._camera_ready.set()
    
    def _culvert_camera_capture_loop(self) -> None:
        """高位涵洞摄像头采集线程，15FPS固定帧率。"""
        self._culvert_camera_thread_started.set()
        next_capture_at = 0.0
        while self.running:
            now = time.monotonic()
            if now < next_capture_at:
                time.sleep(min(0.01, next_capture_at - now))
                continue
            camera = self._culvert_camera
            if camera is None or not camera.isOpened():
                time.sleep(0.05)
                continue
            ok, frame = camera.read()
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.01)
                continue
            with self._culvert_camera_lock:
                self._culvert_camera_frame = frame
                self._culvert_camera_sequence += 1
                self._last_culvert_camera_frame_at = time.monotonic()
            self._culvert_camera_first_frame.set()
            self._culvert_camera_ready.set()
            next_capture_at = time.monotonic() + (1.0 / 15.0)
    
    # ------------------------------------------------------------------------
    # 6.4 巡线推理线程
    # ------------------------------------------------------------------------
    
    def _road_processing_loop(self) -> None:
        """后台执行RKNN巡线推理。"""
        self._road_thread_started.set()
        while self.running:
            if self._culvert_active.is_set():                   # 涵洞扫描期间暂停推理
                time.sleep(0.02)
                continue
            frame = self._latest_camera_frame(timeout=0.5)      # 获取最新帧
            if frame is None:
                continue
            debug_from, debug_to = self._current_physical_segment()
            self.tracker.set_debug_segment(debug_from, debug_to)
            self.tracker.set_junction_mode(self._expected_junction_mode())
            with self._npu_lock:
                result = self.tracker.get_road_info(frame)      # RKNN推理
                with self._road_lock:
                    self._road_result = result
                    self._road_result_sequence += 1
                    self._last_road_result_at = time.monotonic()
                    self._road_ready.set()
                now = time.monotonic()
                self._road_frame_times.append(now)
                if len(self._road_frame_times) == self.args.startup_stable_frames:
                    elapsed = self._road_frame_times[-1] - self._road_frame_times[0]
                    if elapsed > 0:
                        self._startup_road_fps = (len(self._road_frame_times) - 1) / elapsed
                        if self._startup_road_fps >= self.args.startup_min_road_fps:
                            self._road_fps_stable.set()         # 帧率达标，系统就绪
    
    # ------------------------------------------------------------------------
    # 6.5 涵洞检测线程
    # ------------------------------------------------------------------------
    
    def _culvert_loop(self) -> None:
        """涵洞检测线程：从高位摄像头检测涵洞。"""
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
            if time.monotonic() < self._culvert_rearm_at:      # 冷却期间跳过
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
                confirmed, candidates = self._culvert_detector.inspect(frame)  # 涵洞检测
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
            print(f"[CULVERT] 确认涵洞，道路={active_road[0]} -> {active_road[1]}")
            self._run_culvert_scan(*active_road)                # 执行涵洞扫描
        
        if self.args.culvert_debug:
            try:
                cv2.destroyWindow("Culvert high camera")
            except cv2.error:
                pass
    
    # ------------------------------------------------------------------------
    # 6.6 障碍物检测线程
    # ------------------------------------------------------------------------
    
    def _obstacle_segment(self):
        """返回当前路段是否允许障碍物检测。"""
        if (not self.args.obstacle_enabled or self.state != "FOLLOW_LINE" or
            self._culvert_active.is_set() or self._obstacle_recovering):
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
        """障碍物检测线程：检测锥桶并通知主循环。"""
        self._obstacle_thread_started.set()
        last_sequence = -1
        consecutive = 0
        confirming_segment = None
        display_available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        debug_enabled = self.args.obstacle_debug and display_available
        window_name = "Obstacle camera 0"
        if self.args.obstacle_debug and not display_available:
            print("[OBSTACLE] 未检测到 DISPLAY，跳过调试窗口")
        if debug_enabled:
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 960, 720)
            except cv2.error as exc:
                print(f"[OBSTACLE] 调试窗口不可用: {exc}")
                debug_enabled = False
        
        while self.running:
            frame, last_sequence = self._latest_camera_frame_after(last_sequence, timeout=0.5)
            if frame is None:
                continue
            
            segment = self._obstacle_segment()
            detection_blocked = (segment is None or self._obstacle_pending.is_set() or
                                 time.monotonic() < self._obstacle_rearm_at)
            
            result = None
            try:
                if debug_enabled or not detection_blocked:
                    result = self._obstacle_detector.detect(frame)     # 颜色检测
            except (ValueError, cv2.error) as exc:
                print(f"[OBSTACLE WARN] 单帧检测失败: {exc}")
                consecutive = 0
                confirming_segment = None
                continue
            
            if debug_enabled and result is not None:
                try:
                    annotated = self._obstacle_detector.draw_result(frame, result)
                    cv2.imshow(window_name, annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        cv2.destroyWindow(window_name)
                        self.args.obstacle_debug = False
                        debug_enabled = False
                except cv2.error as exc:
                    print(f"[OBSTACLE] 调试窗口已关闭: {exc}")
                    debug_enabled = False
            
            if detection_blocked:
                consecutive = 0
                confirming_segment = None
                continue
            if result is None or not result.obstacle:
                consecutive = 0
                confirming_segment = None
                time.sleep(self.args.obstacle_detect_interval)
                continue
            
            if segment != confirming_segment:
                confirming_segment = segment
                consecutive = 1
            else:
                consecutive += 1
            if consecutive < self.args.obstacle_confirm_frames:     # 连续N帧确认
                time.sleep(self.args.obstacle_detect_interval)
                continue
            
            if self._obstacle_segment() != segment:                 # 再次校验路段
                consecutive = 0
                confirming_segment = None
                continue
            
            with self._obstacle_pending_lock:
                if not self._obstacle_pending.is_set():
                    self._obstacle_pending_data = (segment, result)
                    self._obstacle_pending.set()                    # 通知主循环
                    print(f"[OBSTACLE] 连续{consecutive}帧确认障碍物，道路={segment[0]} -> {segment[1]}")
            
            consecutive = 0
            confirming_segment = None
            time.sleep(self.args.obstacle_detect_interval)
        
        if debug_enabled:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
    
    # ------------------------------------------------------------------------
    # 6.7 RFID和播报线程
    # ------------------------------------------------------------------------
    
    def _rfid_loop(self) -> None:
        """RFID读取线程。"""
        self._rfid_thread_started.set()
        while self.running:
            try:
                card_id = self._rfid_reader.wait_for_card(timeout=0.2)
                if card_id is not None:
                    self.odometry.calibrate_by_rfid(card_id)
                    self._broadcast_queue.put(("point", int(card_id)))
                    self._show_oled(f"到达固定点\nID: {card_id}")
            except Exception as exc:
                print(f"[RFID] 读取线程异常: {exc}")
    
    def _broadcast_loop(self) -> None:
        """语音播报线程。"""
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
            except Exception as exc:
                print(f"[Audio Error] {event_type}:{event_id} 播报失败: {exc}")
            finally:
                self._broadcast_queue.task_done()
    
    # ------------------------------------------------------------------------
    # 6.8 帧获取工具方法
    # ------------------------------------------------------------------------
    
    def _latest_camera_frame_after(self, after_sequence: int, timeout: float = 0.15):
        """获取after_sequence之后的新帧。"""
        deadline = time.monotonic() + timeout
        while self.running:
            with self._camera_lock:
                seq = self._camera_sequence
                if seq != after_sequence and self._camera_frame is not None:
                    return self._camera_frame.copy(), seq
                self._camera_ready.clear()
            if time.monotonic() >= deadline:
                return None, after_sequence
            self._camera_ready.wait(min(0.05, deadline - time.monotonic()))
        return None, after_sequence
    
    def _latest_camera_frame(self, timeout: float = 0.15):
        """获取最新帧并更新处理序列号。"""
        frame, seq = self._latest_camera_frame_after(self._processed_camera_sequence, timeout)
        if frame is not None:
            self._processed_camera_sequence = seq
        return frame
    
    def _latest_culvert_frame(self, after_sequence: int = -1, min_new_frames: int = 1, timeout: float = 0.8):
        """获取高位摄像头新帧。"""
        target_sequence = after_sequence + max(1, min_new_frames)
        deadline = time.monotonic() + timeout
        while self.running and time.monotonic() < deadline:
            with self._culvert_camera_lock:
                seq = self._culvert_camera_sequence
                frame = self._culvert_camera_frame
                if seq >= target_sequence and frame is not None:
                    return frame.copy(), seq
                self._culvert_camera_ready.clear()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                self._culvert_camera_ready.wait(min(0.1, remaining))
        return None, after_sequence
    
    def read_road(self) -> tuple[str, int] | None:
        """读取最新的巡线结果。"""
        if not self._road_ready.wait(0.30):
            return None
        with self._road_lock:
            if self._road_result_sequence == self._consumed_road_sequence:
                self._road_ready.clear()
                return None
            result = self._road_result
            self._consumed_road_sequence = self._road_result_sequence
            self._road_ready.clear()
            return result
    
    # ------------------------------------------------------------------------
    # 6.9 路口和路段相关
    # ------------------------------------------------------------------------
    
    def _current_physical_segment(self) -> tuple[str | None, str | None]:
        """返回当前物理路段端点，跳过虚拟隧道节点。"""
        if self._obstacle_recovering and self._obstacle_return_from:
            return self._obstacle_return_from, self.current_node
        target = self.current_target_node
        for candidate in (self.current_path or [])[1:]:
            node = self.topo_map.get_node(candidate)
            if node is None or node.type != self.topo_map.NODE_TUNNEL:
                target = candidate
                break
        return self.current_node, target
    
    def _expected_junction_mode(self) -> str:
        """判断路口模式：L型还是普通十字路口。"""
        if self.state == "START_ALIGN":
            return "NORMAL"
        from_node, to_node = self._current_physical_segment()
        if (frozenset((from_node, to_node)) in L_JUNCTION_SEGMENTS or
            (from_node, to_node) in L_JUNCTION_DIRECTIONS):
            return "L"
        return "NORMAL"
    
    def _set_culvert_road(self, from_node: str | None, to_node: str | None) -> None:
        """设置当前涵洞道路，启用/禁用涵洞识别。"""
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
    
    # ------------------------------------------------------------------------
    # 6.10 巡线控制
    # ------------------------------------------------------------------------
    
    def follow_once(self, offset: int) -> None:
        """执行单帧巡线控制。"""
        correction = self.args.kp * offset + self.args.kd * (offset - self.last_offset)
        self.last_offset = offset
        correction = max(-self.args.max_angle, min(self.args.max_angle, correction))
        self.chassis.drive_steer(self.args.speed, correction)
    
    # ------------------------------------------------------------------------
    # 6.11 动作执行
    # ------------------------------------------------------------------------
    
    def wait_for_completion(self, timeout: float | None = None) -> bool:
        """等待下位机返回@完成信号。"""
        self.last_action_error = None
        deadline = time.monotonic() + (self.args.action_timeout if timeout is None else timeout)
        while self.running and time.monotonic() < deadline:
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
            if message and "@" in message:      # @ 表示动作完成
                return True
            time.sleep(0.01)
        return False
    
    def _run_action(self, name: str, command, timeout: float | None = None) -> bool:
        """执行动作并等待完成。"""
        self.action_count += 1
        action_id = self.action_count
        print(f"[ACTION #{action_id:03d} SEND] {name}")
        self.chassis.clear_pending_input()
        command()
        if self.wait_for_completion(timeout):
            self._road_resume_grace_until = time.monotonic() + max(2.0, self.args.road_stale_timeout)
            print(f"[ACTION #{action_id:03d} DONE] {name}")
            return True
        if self.last_action_error:
            print(f"[ACTION #{action_id:03d} ERROR] {name} 被下位机拒绝: {self.last_action_error}，立即停车")
        else:
            print(f"[ACTION #{action_id:03d} TIMEOUT] {name}，立即停车")
        self.chassis.stop()
        return False
    
    # ------------------------------------------------------------------------
    # 6.12 路口动作
    # ------------------------------------------------------------------------
    
    def execute_cross_action(self, action: str, cross_id: int, at_node: str, next_node: str) -> bool:
        """执行路口动作（左转/右转/掉头/直行）。"""
        if action == "FORWARD":
            self.action_count += 1
            action_id = self.action_count
            print(f"[CROSS #{cross_id:03d} DISPATCH] node={at_node} FORWARD")
            self.chassis.clear_pending_input()
            self.chassis.go_cross(speed=self.args.speed, distance_multiplier=self.args.cross_forward_distance_multiplier)
            if not self.wait_for_completion(self.args.action_timeout):
                print(f"[ACTION #{action_id:03d} TIMEOUT] FORWARD，立即停车")
                self.chassis.stop()
                return False
            print(f"[ACTION #{action_id:03d} DONE] FORWARD（编码器闭环）")
            self._road_resume_grace_until = time.monotonic() + max(2.0, self.args.road_stale_timeout)
            time.sleep(0.3)
            self._set_culvert_road(at_node, next_node)
            self._begin_cross_rearm(cross_id=cross_id, action=action, at_node=at_node, next_node=next_node)
            return True
        
        commands = {
            "LEFT": (self.chassis.ccw, "chassis.ccw()"),                     # 左转
            "RIGHT": (self.chassis.cw, "chassis.cw()"),                      # 右转
            "U_TURN": (lambda: self.chassis.turn(180), "chassis.turn(180)"), # 掉头
        }
        cmd, method = commands.get(action)
        if cmd is None:
            raise ValueError(f"未知路口动作: {action}")
        print(f"[CROSS #{cross_id:03d} DISPATCH] node={at_node} {action} method={method}")
        completed = self._run_action(f"CROSS#{cross_id:03d} node={at_node} {action}", cmd)
        if completed:
            time.sleep(0.3)
            self._set_culvert_road(at_node, next_node)
            self._begin_cross_rearm(cross_id=cross_id, action=action, at_node=at_node, next_node=next_node)
        return completed
    
    def _begin_cross_rearm(self, *, cross_id: int, action: str, at_node: str, next_node: str) -> None:
        """路口动作完成后锁定，防止重复触发。"""
        self._cross_event_latched = True
        self._cross_clear_frames = 0
        self._cross_rearm_at = time.monotonic() + self.args.cross_rearm_delay
        print(f"[CROSS GATE] CROSS#{cross_id:03d} locked; rearm_after={self.args.cross_rearm_delay:.2f}s")
    
    def _advance_virtual_targets(self, cross_id: int) -> bool:
        """推进T1-T4虚拟节点（不消耗视觉路口事件）。"""
        while self.current_target_node:
            target = self.topo_map.get_node(self.current_target_node)
            if target is None or target.type != self.topo_map.NODE_TUNNEL:
                break
            expected = self.current_target_node
            reached = self.navigator.advance_path()
            self._sync_navigation_path()
            if reached != expected:
                print(f"[NAVIGATION ERROR] 无法推进虚拟节点 {expected}: reached={reached}")
                return False
            print(f"[CROSS #{cross_id:03d} VIRTUAL] reached={reached} (无视觉路口)")
        return True
    
    # ------------------------------------------------------------------------
    # 6.13 障碍物处理（核心）
    # ------------------------------------------------------------------------
    
    def _handle_pending_obstacle(self) -> bool:
        """
        处理障碍物：封锁道路 → 重规划 → 掉头 → 返回路口。
        这是障碍物处理的核心函数。
        """
        # 1. 取出待处理障碍物事件
        with self._obstacle_pending_lock:
            pending = self._obstacle_pending_data
            self._obstacle_pending_data = None
            self._obstacle_pending.clear()
        if pending is None:
            return True
        
        # 2. 解析障碍物事件，校验是否过期
        (blocked_from, blocked_to), _result = pending
        if (self.state != "FOLLOW_LINE" or self.current_node != blocked_from or
            self.current_target_node != blocked_to):
            print(f"[OBSTACLE] 丢弃过期事件 {blocked_from} -> {blocked_to}")
            return True
        
        # 3. 停车，清除涵洞模式
        self.chassis.stop()
        self._set_culvert_road(None, None)
        print(f"[OBSTACLE] 封锁道路 {blocked_from} <-> {blocked_to}，开始重规划")
        
        # 4. 通知导航器封锁道路，重新规划
        if not self.navigator.on_obstacle(blocked_from, blocked_to):
            print("[OBSTACLE ERROR] 封锁后没有可执行的剩余任务路线")
            self.chassis.stop()
            return False
        
        # 5. 同步导航路径，设置恢复状态
        self._sync_navigation_path()
        self._obstacle_recovering = True
        self._obstacle_return_from = blocked_to
        print(f"[OBSTACLE] 绕行路线={self.current_path}")
        
        # 6. 执行原地掉头
        if not self._run_action(
            f"障碍物前掉头 {blocked_from} <- {blocked_to}",
            lambda: self.chassis.turn_pure(180),
        ):
            self._obstacle_recovering = False
            self._obstacle_return_from = None
            return False
        
        # 7. 设置冷却时间，返回（掉头后继续巡线返回路口）
        self._obstacle_rearm_at = time.monotonic() + self.args.obstacle_cooldown
        print(f"[OBSTACLE] 已掉头，沿线返回 {blocked_from}；返回路口时再执行绕行转向")
        return True
    
    # ------------------------------------------------------------------------
    # 6.14 涵洞扫描
    # ------------------------------------------------------------------------
    
    def _run_culvert_scan(self, from_node: str, to_node: str) -> None:
        """涵洞确认后执行扫描（YOLO目标识别）。"""
        edge = self.topo_map.edge_key(from_node, to_node)
        if edge in self._culvert_scanned_edges:
            print(f"[CULVERT] 道路 {from_node} -> {to_node} 已扫描，忽略")
            return
        
        self._culvert_scanned_edges.add(edge)
        self._culvert_active.set()
        scan_results = {"left": -1, "right": -1}
        
        try:
            self.chassis.stop()
            self._show_oled(f"发现涵洞\n{from_node} -> {to_node}", duration=1500)
            self.navigator.mark_culvert_edge(from_node, to_node)
            
            # 启动云台
            self.chassis._send_command("@|5|0|0|0#")
            time.sleep(0.15)
            self._ensure_yolo()
            self.chassis._send_command("@|10|0|0|0#")
            
            # 等待云台扫描结果
            captured_sides = set()
            deadline = time.monotonic() + self.args.culvert_scan_timeout
            while self.running and time.monotonic() < deadline:
                message = self.chassis.read_message()
                if message:
                    message = message.strip()
                    print(f"[CULVERT SERIAL RX] {message}")
                    if message == "Y_START":
                        self._show_oled("云台扫描\n准备识别目标", duration=1500)
                    elif message == "Y_L" and "right" not in captured_sides:
                        captured_sides.add("right")
                        with self._culvert_camera_lock:
                            signal_sequence = self._culvert_camera_sequence
                        scan_frame, _ = self._latest_culvert_frame(
                            after_sequence=signal_sequence, min_new_frames=1, timeout=0.3
                        )
                        scan_results["right"] = self._detect_scan_target("左侧", "Y_L", scan_frame)
                    elif message == "Y_R" and "left" not in captured_sides:
                        captured_sides.add("left")
                        with self._culvert_camera_lock:
                            signal_sequence = self._culvert_camera_sequence
                        scan_frame, _ = self._latest_culvert_frame(
                            after_sequence=signal_sequence, min_new_frames=1, timeout=0.3
                        )
                        scan_results["left"] = self._detect_scan_target("右侧", "Y_R", scan_frame)
                    if message == "Y_END":
                        print(f"[CULVERT] 扫描完成，收到 Y_END")
                        break
                time.sleep(0.02)
            else:
                print("[CULVERT WARN] 等待 Y_END 超时")
            
            self._publish_scan_results(scan_results["left"], scan_results["right"])
        finally:
            with self._culvert_lock:
                self._culvert_detector.reset()
            self._culvert_rearm_at = time.monotonic() + max(0.0, self.args.culvert_rearm_delay)
            self._road_resume_grace_until = time.monotonic() + max(2.0, self.args.road_stale_timeout)
            self._culvert_triggered.clear()
            self._culvert_active.clear()
            self._set_culvert_road(None, None)
    
    def _detect_scan_target(self, side_name: str, trigger_message: str = "", frame=None) -> int:
        """执行YOLO目标识别。"""
        if frame is None:
            with self._culvert_camera_lock:
                seq = self._culvert_camera_sequence
            frame, _ = self._latest_culvert_frame(after_sequence=seq, min_new_frames=1, timeout=0.3)
        if frame is None:
            print(f"[YOLO ERROR] {side_name}拍照失败")
            return -1
        if self.args.culvert_flip in (-1, 0, 1):
            frame = cv2.flip(frame, self.args.culvert_flip)
        if self._yolo is None or self._yolo.rknn is None:
            print(f"[YOLO ERROR] {side_name}目标模型不可用")
            return -1
        with self._npu_lock:
            target_id = int(self._yolo.detect_target(frame))
        text = self._target_text(target_id)
        print(f"[YOLO] {side_name}识别结果: {text}")
        self._show_oled(f"{side_name}识别\n{text}", duration=1500)
        return target_id
    
    @staticmethod
    def _target_text(target_id: int) -> str:
        """目标ID转文字。"""
        if target_id < 1:
            return "未识别"
        if target_id <= 10:
            return f"{target_id}号嫌疑人"
        return f"{target_id - 10}号刀具"
    
    def _publish_scan_results(self, left_id: int, right_id: int) -> None:
        """发布扫描结果（播报）。"""
        self._show_oled(f"右:{self._target_text(left_id)}\n左:{self._target_text(right_id)}", duration=2500)
        detected = [tid for tid in (left_id, right_id) if tid > 0]
        if detected:
            for tid in detected:
                self._broadcast_queue.put(("target", tid))
        else:
            self._broadcast_queue.put(("target", -1))
    
    def _ensure_yolo(self) -> bool:
        """确保YOLO模型可用。"""
        if self._yolo is not None and self._yolo.rknn is not None:
            return True
        if YoloDetector is None:
            print("[YOLO ERROR] 目标识别模块不可用")
            return False
        try:
            with _silence_native_console():
                self._yolo = YoloDetector(preset_classes=tuple(range(1, 21)), show_window=False, save_debug=False)
            ready = self._yolo.rknn is not None
            print(f"[YOLO] 重试加载 {'成功' if ready else '失败'}")
            return ready
        except Exception as exc:
            self._yolo = None
            print(f"[YOLO ERROR] 加载失败: {exc}")
            return False
    
    # ------------------------------------------------------------------------
    # 6.15 路口处理（核心）
    # ------------------------------------------------------------------------
    
    def _handle_crossroad(self) -> bool:
        """
        处理路口事件（核心状态机）。
        包含：障碍物绕行恢复、虚拟节点推进、正常路口转向。
        """
        self.cross_event_count += 1
        cross_id = self.cross_event_count
        print(f"\n[CROSS #{cross_id:03d} DETECT] current={self.current_node} target={self.current_target_node or '-'}")
        
        # ---- 情况1：障碍物绕行恢复 ----
        if self._obstacle_recovering:
            if not self._obstacle_return_from or not self.current_target_node:
                print("[OBSTACLE ERROR] 返回路口时缺少绕行方向")
                self.chassis.stop()
                return False
            
            # 计算绕行转向动作
            action = self.topo_map.turn_action(
                self._obstacle_return_from,
                self.current_node,
                self.current_target_node,
            )
            print(f"[OBSTACLE] 已返回 {self.current_node}，执行绕行转向 action={action}")
            if not self.execute_cross_action(action, cross_id, self.current_node, self.current_target_node):
                return False
            self._obstacle_recovering = False
            self._obstacle_return_from = None
            self.state = "FOLLOW_LINE"
            return True
        
        # ---- 情况2：推进虚拟节点(T1-T4) ----
        if not self._advance_virtual_targets(cross_id):
            return False
        
        # ---- 情况3：正常路口处理 ----
        expected = self.current_target_node
        if not expected:
            print("[NAVIGATION ERROR] 没有待确认的目标节点")
            return False
        
        # 预演下一节点
        preview_reached, preview_path, _ = self.navigator.preview_after_advance()
        if preview_reached != expected:
            print(f"[NAVIGATION ERROR] 位置预演失败: expected={expected} reached={preview_reached}")
            self.chassis.stop()
            return False
        
        next_node = preview_path[1] if preview_path and len(preview_path) > 1 else None
        if not next_node:
            print(f"[NAVIGATION ERROR] 到达 {expected} 后没有后续路线")
            self.chassis.stop()
            return False
        
        # 计算转向动作
        action = self.topo_map.turn_action(self.current_node, expected, next_node)
        print(f"[CROSS #{cross_id:03d} PLAN] {self.current_node} -> {expected} -> {next_node} action={action}")
        
        self._set_culvert_road(None, None)
        if not self.execute_cross_action(action, cross_id, expected, next_node):
            return False
        
        # 确认到达
        reached = self._confirm_next_node(cross_id)
        if reached is None or reached != expected:
            return False
        
        # 检查是否完成任务
        if next_node == "START_AREA":
            return self._run_return_start()
        
        self.state = "FOLLOW_LINE"
        return True
    
    def _confirm_next_node(self, cross_id: int | None = None) -> str | None:
        """确认到达节点并更新导航。"""
        expected = self.current_target_node
        from_node = self.current_node
        reached = self.navigator.advance_path()
        self._sync_navigation_path()
        prefix = f"[CROSS #{cross_id:03d} ARRIVAL]" if cross_id is not None else "[ARRIVAL]"
        if reached:
            print(f"{prefix} from={from_node} expected={expected} reached={reached}")
        else:
            print(f"{prefix} from={from_node} expected={expected} reached=None 状态=FAILED")
        return reached
    
    # ------------------------------------------------------------------------
    # 6.16 出发和返回
    # ------------------------------------------------------------------------
    
    def _run_start_actions(self) -> bool:
        """执行出发区固定动作：前进 → 右转。"""
        print("\n[START_STATE] 全地图测试出发区固定动作...")
        self._show_oled("全地图测试\n正在出发", duration=1200)
        actions = (
            (f"出发区前进 distance={self.args.start_forward_distance}",
             lambda: self.chassis.distance(val=self.args.start_forward_speed, dis=self.args.start_forward_distance)),
            (f"出发区右转 angle={self.args.start_right_turn_angle}",
             lambda: self.chassis.turn_pure(self.args.start_right_turn_angle)),
        )
        for name, cmd in actions:
            if not self._run_action(name, cmd, self.args.start_action_timeout):
                return False
            time.sleep(0.2)
        self.start_line_frames = 0
        self.state = "START_ALIGN"
        print(f"[START_STATE] 进入姿态巡线；稳定帧数={self.args.start_line_required_frames}")
        return True
    
    def _run_start_align(self, road_type: str, offset: int) -> None:
        """出发区姿态巡线：等待稳定后同步到J_TOP。"""
        self.follow_once(offset)
        if road_type not in {"LINE", "CROSS"}:
            return
        self.start_line_frames += 1
        if self.start_line_frames >= self.args.start_line_required_frames:
            reached = self._confirm_next_node()
            if reached != "J_TOP":
                print(f"[START_SYNC ERROR] expected=J_TOP reached={reached}，停车")
                self.chassis.stop()
                self.running = False
                return
            print("[START_SYNC] J_TOP 左转由视觉巡线完成")
            self._cross_event_latched = (road_type == "CROSS")
            print("[START_ALIGN] 姿态巡线稳定，开放后续路口识别")
            self._set_culvert_road("J_TOP", "N1")
            self.state = "FOLLOW_LINE"
    
    def _run_return_start(self) -> bool:
        """执行最终返回出发区。"""
        print(f"[FINISH] 返回出发区: distance={self.args.start_forward_distance}")
        cmd = lambda: self.chassis.distance(val=self.args.start_forward_speed, dis=self.args.start_forward_distance)
        if not self._run_action("最终返回出发区", cmd, self.args.start_action_timeout):
            return False
        reached = self._confirm_next_node()
        if reached != "START_AREA" or not self.navigator.mission_complete:
            print(f"[MISSION ERROR] 返回确认失败: reached={reached}")
            return False
        print("[MISSION] 全地图测试完成")
        self._show_oled("全地图完成\n已返回出发区", duration=4000)
        self.state = "STOP"
        return True
    
    # ------------------------------------------------------------------------
    # 6.17 硬件验证和健康检查
    # ------------------------------------------------------------------------
    
    def validate_hardware(self) -> None:
        """验证硬件是否正常。"""
        if not self.chassis.serial or not self.chassis.serial.is_open:
            raise RuntimeError(f"底盘串口不可用: {self.args.serial_port}")
        if not self.camera.isOpened():
            raise RuntimeError(f"巡线摄像头不可用: {self.args.camera}")
        if self._culvert_camera is None or not self._culvert_camera.isOpened():
            raise RuntimeError(f"高位涵洞摄像头不可用: {self.args.culvert_camera}")
    
    def _wait_until_system_ready(self) -> bool:
        """等待所有模块、线程和帧率就绪。"""
        checks = (
            ("巡线采集线程", self._camera_thread_started, lambda: self._camera_thread.is_alive()),
            ("高位采集线程", self._culvert_camera_thread_started, lambda: self._culvert_camera_thread.is_alive()),
            ("巡线推理线程", self._road_thread_started, lambda: self._road_thread.is_alive()),
            ("涵洞识别线程", self._culvert_thread_started, lambda: self._culvert_thread.is_alive()),
            ("障碍识别线程", self._obstacle_thread_started, lambda: self._obstacle_thread.is_alive()),
            ("RFID线程", self._rfid_thread_started, lambda: self._rfid_thread.is_alive()),
            ("播报线程", self._broadcast_thread_started, lambda: self._broadcast_thread.is_alive()),
            ("巡线摄像头画面", self._camera_ready, lambda: True),
            ("高位摄像头画面", self._culvert_camera_first_frame, lambda: True),
            ("巡线推理帧率", self._road_fps_stable, lambda: True),
        )
        modules = (
            ("RFID", self._rfid_reader is not None and self._rfid_reader.ser is not None and self._rfid_reader.ser.is_open),
            ("播报", self._broadcaster is not None and not getattr(self._broadcaster, "is_mock", True)),
            ("OLED", self._oled is not None and getattr(self._oled, "is_ready", lambda: False)()),
            ("YOLO", self._yolo is not None and self._yolo.rknn is not None),
        )
        deadline = time.monotonic() + self.args.system_ready_timeout
        last_pending = None
        self._show_oled("全地图测试\n等待系统就绪", duration=1200)
        while self.running and time.monotonic() < deadline:
            self.tracker.pump_debug_window()
            self.chassis.stop()
            pending = [name for name, ready in modules if not ready]
            pending.extend(name for name, event, alive in checks if not event.is_set() or not alive())
            if not pending:
                print(f"[STARTUP READY] 全部模块已就绪；帧率={self._startup_road_fps:.1f} FPS")
                self._show_oled("全地图测试\n系统已就绪", duration=1200)
                return True
            pending_text = ", ".join(pending)
            if pending_text != last_pending:
                print(f"[STARTUP WAIT] 等待: {pending_text}")
                last_pending = pending_text
            time.sleep(0.1)
        self.chassis.stop()
        print(f"[STARTUP ERROR] 系统未全部就绪")
        self._show_oled("启动失败\n系统未就绪", duration=4000)
        return False
    
    def _runtime_health_error(self) -> str | None:
        """运行时健康检查。"""
        if not getattr(self.chassis, "serial", None) or not self.chassis.serial.is_open:
            return "底盘串口已断开"
        for name, worker in (("巡线采集", self._camera_thread), ("高位采集", self._culvert_camera_thread),
                             ("巡线推理", self._road_thread), ("涵洞识别", self._culvert_thread),
                             ("障碍识别", self._obstacle_thread), ("RFID", self._rfid_thread),
                             ("播报", self._broadcast_thread)):
            if not worker.is_alive():
                return f"{name}线程已退出"
        if not self._rfid_reader.ser or not self._rfid_reader.ser.is_open:
            return "RFID串口已断开"
        now = time.monotonic()
        if now - self._last_camera_frame_at > self.args.camera_stale_timeout:
            return "巡线摄像头画面超时"
        if now - self._last_culvert_camera_frame_at > self.args.camera_stale_timeout:
            return "高位摄像头画面超时"
        if (not self._culvert_active.is_set() and now >= self._road_resume_grace_until and
            now - self._last_road_result_at > self.args.road_stale_timeout):
            return "巡线推理结果超时"
        return None
    
    # ------------------------------------------------------------------------
    # 6.18 主任务循环
    # ------------------------------------------------------------------------
    
    def run_mission(self) -> int:
        """运行任务主循环。"""
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
            # 健康检查
            self.tracker.pump_debug_window()
            health_error = self._runtime_health_error()
            if health_error is not None:
                print(f"\n[HEALTH ERROR] {health_error}，立即停车")
                self.chassis.stop()
                self._show_oled(f"运行停止\n{health_error}", duration=4000)
                return 5
            
            # 涵洞扫描期间等待
            if self._culvert_active.is_set():
                time.sleep(0.05)
                continue
            
            # 处理障碍物
            if self._obstacle_pending.is_set():
                if not self._handle_pending_obstacle():
                    return 6
                continue
            
            # 读取巡线结果
            road = self.read_road()
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
            
            # 出发区姿态巡线
            if self.state == "START_ALIGN":
                self._run_start_align(road_type, offset)
                continue
            
            # 路口处理
            if road_type == "CROSS":
                if self._cross_event_latched:
                    self.follow_once(offset)
                    time.sleep(0.02)
                    continue
                self._cross_event_latched = True
                self.chassis.drive_steer(0, 0)
                time.sleep(0.2)
                if not self._handle_crossroad():
                    return 4
                continue
            
            # LINE状态：解锁路口
            if road_type == "LINE" and self._cross_event_latched:
                centered = abs(offset) <= self.args.cross_rearm_max_offset
                if time.monotonic() >= self._cross_rearm_at and centered:
                    self._cross_clear_frames += 1
                else:
                    self._cross_clear_frames = 0
                if self._cross_clear_frames >= self.args.cross_rearm_line_frames:
                    self._cross_event_latched = False
                    self._cross_clear_frames = 0
                    print("[CROSS GATE] stable LINE observed; next junction armed")
            
            # 正常巡线
            self.follow_once(offset)
            now = time.monotonic()
            if now - self._last_line_log_time >= 0.25:
                print(f"[LINE] 状态={road_type:<5} 偏差={offset:>4} 目标={self.current_target_node or '-'}",
                      end="\r", flush=True)
                self._last_line_log_time = now
            if self.tracker.last_key in (ord("q"), ord("Q"), 27):
                break
            time.sleep(0.02)
        
        return 0 if self.state == "STOP" and self.navigator.mission_complete else 1
    
    # ------------------------------------------------------------------------
    # 6.19 停止和清理
    # ------------------------------------------------------------------------
    
    def stop(self, *_args: object) -> None:
        """停止任务。"""
        self.stop_requested = True
        self.running = False
        self.chassis.stop()
    
    def _release_yolo(self) -> None:
        """释放YOLO资源。"""
        yolo, self._yolo = self._yolo, None
        if yolo is not None:
            yolo.release()
            print("[YOLO] 目标模型资源已释放")
    
    def close(self) -> None:
        """清理所有资源。"""
        self.running = False
        try:
            self.chassis.stop()
        finally:
            time.sleep(0.1)
            self.camera.release()
            if self._culvert_camera is not None:
                self._culvert_camera.release()
            for thread in (self._camera_thread, self._culvert_camera_thread, self._road_thread,
                           self._culvert_thread, self._obstacle_thread, self._rfid_thread):
                if thread.is_alive():
                    thread.join(timeout=1.0)
            # 清空播报队列
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
            print("[SAFE] 电机已停车，摄像头、识别器和串口已释放")
    
    def run(self) -> int:
        """运行测试。"""
        try:
            self.validate_hardware()
            print(f"[READY] 摄像头={self.args.camera} 串口={self.args.serial_port} 速度={self.args.speed}")
            route, progress = calculate_plan()
            print(f"[FULL MAP TEST] 规划节点数={len(route)}，需覆盖道路={progress['required_edges']}")
            result = self.run_mission()
            if result != 0:
                if self.stop_requested:
                    self._show_oled("全地图测试\n用户已中止", duration=2500)
                else:
                    self._show_oled(f"全地图测试停止\n错误码: {result}", duration=4000)
            return result
        finally:
            self.close()


# ============================================================================
# 第七部分：入口函数
# ============================================================================

def run_live() -> int:
    """运行实时测试。"""
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
        if robot is not None and robot.running:
            robot.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true", help="只打印路线，不运行测试")
    args = parser.parse_args()
    if args.plan_only:
        return 0 if print_plan() else 1
    return run_live()


if __name__ == "__main__":
    raise SystemExit(main())