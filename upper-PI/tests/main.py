import time
import threading
import sys
import cv2
import os

from Serial import ChassisController
from rfid import RFIDReader
from yolo_detect import YoloDetector
from oled import OLEDDisplay      
from broadcast import Broadcaster 
from line_detect import LineTracker
from odometry import Odometry
from topo_map import TopoMap
from navigator import Navigator


class ReconRobot:
    def __init__(self):
        print("[SYS] 正在初始化公共安全侦查机器人...")
        self.is_running = True
        
        # ====== 硬件设备初始化 ======
        self.chassis = ChassisController(port="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A5069RR4-if00-port0")
        self.rfid = RFIDReader(port="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0")
        self.oled = OLEDDisplay()   
        self.broadcaster = Broadcaster()  

        # 两路摄像头相互独立。启动脚本可传入数字编号，或稳定的
        # /dev/v4l/by-id 设备路径。
        self.nav_camera_source = self._parse_camera_source(
            os.environ.get("NAV_CAMERA", "0")
        )
        self.detect_camera_source = self._parse_camera_source(
            os.environ.get("DETECT_CAMERA", "2")
        )
        self.low_camera_only = os.environ.get("LOW_CAMERA_ONLY", "0").lower() \
            in {"1", "true", "yes", "on"}
        self.cameras = {
            "nav": self._create_camera_state(
                "navigation", self.nav_camera_source,
                self._parse_flip_code(os.environ.get("NAV_CAMERA_FLIP", "none")),
            ),
        }
        if not self.low_camera_only:
            self.cameras["detect"] = self._create_camera_state(
                "suspect", self.detect_camera_source,
                self._parse_flip_code(os.environ.get("DETECT_CAMERA_FLIP", "none")),
            )
        self.topo_map = TopoMap()
        # 八个绿色涵洞的位置对机器人未知；这里只建立物理道路图，
        # 通过全路段覆盖在运行中发现任务。
        self.topo_map.build_competition_track(green_count=0)
        print("[MAP] unknown green layout; full-road coverage enabled")
        self.topo_map.pretty_print()

        self.odometry = Odometry()

        self.navigator = Navigator(self.topo_map, self.odometry)
        self.navigator.init_mission(
            "START_AREA", expected_green_count=8, cover_all_edges=True,
        )
        self.mission_plan = list(self.topo_map.fixed_patrol_nodes)
        self.current_node = "START_AREA"
        self.previous_node = None
        self.current_path = None
        self.current_target_node = None
        self.blocked_edges = self.navigator.blocked_edges
        self.obstacle_recovering = False
        self.obstacle_return_from = None
        self._sync_navigation_path()
        
        for camera_name, camera in self.cameras.items():
            status = "ready" if camera["cap"].isOpened() else "NOT OPEN"
            print(f"[CAMERA] {camera_name}: source={camera['source']} {status}")

        # 初始化俯视导航摄像头的巡线识别器。
        print("[DEBUG] 正在初始化巡线识别器...")
        self.tracker = LineTracker()  
        
        if self.low_camera_only:
            self.yolo = None
            print("[DEBUG] 低位摄像头专用模式：不加载高位 YOLO 模型")
        else:
            print("[DEBUG] 正在加载 YOLO 识别模型...")
            # ============== 初始化高位识别摄像头使用的 YOLO 模型 ==============
            self.yolo = YoloDetector(
                preset_classes=[1, 11],
                show_window=False,      # 实车运行时不弹出识别窗口。
                save_debug=False        # 需要保存调试图片时改为 True。
            )
        
        self.hardware_lock = threading.Lock()
        
        # 下位机动作完成和涵洞扫描事件。
        self.event_turn_done = threading.Event()  
        self.event_y_start = threading.Event()
        self.event_y_l = threading.Event()
        self.event_y_r = threading.Event()
        self.event_y_end = threading.Event()
        # 绿色任务是未知涵洞。RFID 仅对应 N1～N12；涵洞改由上位机
        # 视觉识别。识别模块暂不实现，后续注入 callable(frame)->bool。
        self.culvert_detector = None
        self.active_culvert_key = None
        
        self.base_speed = 10
        self.Kp = 0.35 
        self.Kd = 0.1  
        self.last_offset = 0

        # 出发区没有 RFID 或终点地标。车辆在 J_TOP 转入短支路后，
        # 通过编码器定距驶入出发框；参数保留为可现场标定项。
        self.return_start_speed = int(
            os.environ.get("RETURN_START_SPEED", "10")
        )
        self.return_start_distance = int(
            os.environ.get("RETURN_START_DISTANCE", "100")
        )
        self.return_start_timeout = float(
            os.environ.get("RETURN_START_TIMEOUT", "6.0")
        )

        # 路口动作由当前图路径计算，绕障重规划后会立即采用新的分支。
        
        # ========== 状态机从出发区固定动作开始 ==========
        self.state = "START" 
        
        self.last_cross_time = 0
        self.cross_cooldown = 3.0 
        # 出发区固定动作后，先沿普通黑线巡线若干帧以校正抬高摄像头的姿态；
        # 达到稳定帧数后，才允许把下一次真实路口识别为 J_TOP。
        self.start_line_frames = 0
        self.start_line_required_frames = max(
            1, int(os.environ.get("START_LINE_STABLE_FRAMES", "12"))
        )

        # 每路摄像头都有独立读取线程和锁，YOLO 不会释放或阻塞巡线摄像头。
        for camera_name in self.cameras:
            threading.Thread(
                target=self._camera_reader_loop,
                args=(camera_name,),
                daemon=True,
                name=f"camera-{camera_name}",
            ).start()

    @staticmethod
    def _parse_camera_source(value):
        value = str(value).strip()
        try:
            return int(value)
        except ValueError:
            return value

    @staticmethod
    def _parse_flip_code(value):
        value = str(value).strip().lower()
        if value in {"none", "off", "false", "no"}:
            return None
        code = int(value)
        if code not in (-1, 0, 1):
            raise ValueError("camera flip must be -1, 0, 1, or none")
        return code

    @staticmethod
    def _open_capture(source):
        backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        try:
            cap = cv2.VideoCapture(source, backend)
        except Exception:
            cap = cv2.VideoCapture(source)
        if cap.isOpened():
            # 等价于 cv2.VideoWriter_fourcc(*"MJPG")。使用浮点数计算，
            # 避免 OpenCV 不完整类型存根产生检查错误。
            mjpg_fourcc = float(
                ord("M") | (ord("J") << 8) | (ord("P") << 16) | (ord("G") << 24)
            )
            cap.set(cv2.CAP_PROP_FOURCC, mjpg_fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _create_camera_state(self, name, source, flip_code):
        return {
            "name": name,
            "source": source,
            "flip_code": flip_code,
            "cap": self._open_capture(source),
            "capture_lock": threading.Lock(),
            "frame_lock": threading.Lock(),
            "latest_frame": None,
            "latest_ret": False,
            "frame_counter": 0,
            "last_frame_time": 0.0,
        }

    def _camera_reader_loop(self, camera_name):
        camera = self.cameras[camera_name]
        while self.is_running:
            with camera["capture_lock"]:
                cap = camera["cap"]
                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                else:
                    ret, frame = False, None
            if ret and frame is not None:
                if camera["flip_code"] in (-1, 0, 1):
                    frame = cv2.flip(frame, camera["flip_code"])
                with camera["frame_lock"]:
                    camera["latest_ret"] = True
                    camera["latest_frame"] = frame
                    camera["frame_counter"] += 1
                    camera["last_frame_time"] = time.time()
            time.sleep(0.005)

    def read_frame(self, camera_name="nav"):
        camera = self.cameras[camera_name]
        with camera["frame_lock"]:
            if camera["latest_frame"] is not None:
                return camera["latest_ret"], camera["latest_frame"].copy()
            return False, None

    def check_camera_health(self, camera_name):
        camera = self.cameras[camera_name]
        with camera["frame_lock"]:
            elapsed = time.time() - camera["last_frame_time"]
            healthy = camera["latest_frame"] is not None and elapsed <= 1.0
        if not healthy:
            print(f"[CAMERA WARN] {camera_name} frame timeout: {elapsed:.2f}s")
        return healthy

    def start_background_tasks(self):
        """启动串口监听、RFID 监听和摄像头健康检查线程。"""
        threading.Thread(target=self._chassis_listener, daemon=True).start()
        threading.Thread(target=self._rfid_listener, daemon=True).start()
        # 监测摄像头画面超时并自动重连。
        threading.Thread(target=self._camera_health_monitor, daemon=True).start()
        print("[SYS] 后台监听线程已启动。")

    def _camera_health_monitor(self):
        """定期检查两路摄像头是否仍在输出画面。"""
        while self.is_running:
            time.sleep(2.0)
            for camera_name in self.cameras:
                if not self.check_camera_health(camera_name):
                    self._reinit_camera(camera_name)

    def _reinit_camera(self, camera_name):
        camera = self.cameras[camera_name]
        print(f"[CAMERA] reinitializing {camera_name}...")
        with camera["capture_lock"]:
            try:
                if camera["cap"] is not None:
                    camera["cap"].release()
                time.sleep(0.3)
                camera["cap"] = self._open_capture(camera["source"])
                if not camera["cap"].isOpened():
                    print(f"[CAMERA ERROR] cannot open {camera['source']}")
                    return False
                with camera["frame_lock"]:
                    camera["last_frame_time"] = time.time()
                return True
            except Exception as exc:
                print(f"[CAMERA ERROR] {camera_name}: {exc}")
                return False
    def on_obstacle_detected(self):
        """封锁前方路段，并立刻计算绕行路线。"""
        print("[OBSTACLE] obstacle detected; replanning...")
        self.chassis.stop()
        if not self.current_node or not self.current_target_node:
            print("[OBSTACLE] no confirmed graph segment; holding position")
            self.state = "STOP"
            return
        print(f"[OBSTACLE] blocked: {self.current_node} -> "
              f"{self.current_target_node}")
        if self.navigator.on_obstacle(
                self.current_node, self.current_target_node):
            blocked_from = self.current_node
            blocked_to = self.current_target_node
            self._sync_navigation_path()
            print(f"[OBSTACLE] detour: {self.current_path}")
            # B:1 通常在车辆已进入路段后才上报。先掉头返回上一个已确认
            # 节点，再从该节点驶入刚刚规划出的新分支。
            current = self.topo_map.get_node(blocked_from)
            self.obstacle_recovering = bool(
                current and current.type != self.topo_map.NODE_GREEN_TASK
            )
            self.obstacle_return_from = blocked_to
            self.state = "OBSTACLE_RECOVERY"
        else:
            print("[OBSTACLE] no route reaches the remaining mission points")
            self.state = "STOP"

    def _sync_navigation_path(self):
        """将线程安全的规划路径同步到硬件状态机。"""
        self.current_node = self.navigator.current_node
        self.previous_node = self.navigator.previous_node
        path = self.navigator.get_next_segment()
        self.current_path = path
        self.current_target_node = path[1] if path and len(path) > 1 else None
        if self.navigator.mission_complete:
            self.state = "STOP"
        elif not path and self.navigator.sensor_miss:
            print(
                "[MISSION ERROR] all roads covered but fewer than "
                f"{self.navigator.expected_green_count} green culverts "
                "were scanned; check the culvert vision detector"
            )
            self.state = "STOP"

    def _current_culvert_key(self):
        """为当前道路上的未知涵洞生成稳定的去重编号。"""
        if not self.current_node or not self.current_target_node:
            return None
        a, b = self.topo_map.edge_key(
            self.current_node, self.current_target_node,
        )
        return f"CULVERT:{a}<->{b}"

    def on_culvert_detected(self):
        """视觉识别到涵洞后，通知下位机启动侦查扫描。"""
        if self.state == "SCAN_CULVERT":
            return False

        self.active_culvert_key = self._current_culvert_key()
        if self.active_culvert_key:
            print(f"[CULVERT] detected on {self.active_culvert_key}")
        else:
            print("[CULVERT WARN] detected without a known road")

        # 清除上一次扫描残留事件；下位机会依次重新上报四个 Y_* 信号。
        self.event_y_start.clear()
        self.event_y_l.clear()
        self.event_y_r.clear()
        self.event_y_end.clear()
        self.state = "SCAN_CULVERT"
        self.chassis.trigger_culvert_inspection()
        return True

    def _confirm_next_node(self):
        """根据路口或其他到达事件确认已到达路径中的下一个节点。"""
        expected = self.current_target_node
        reached = self.navigator.advance_path()
        self._sync_navigation_path()
        if reached:
            print(f"[NAVIGATION] reached {reached}; next "
                  f"{self.current_target_node}")
        elif expected:
            print(f"[NAVIGATION WARN] could not confirm expected node {expected}")
        return reached

    def _chassis_listener(self):
        while self.is_running:
            try:
                msg = self.chassis.read_message()
                if msg:
                    # ---- 编码器和障碍物上报（无 IMU/陀螺仪） ----
                    if msg.startswith("E:"):
                        parts = msg[2:].split(",")
                        if len(parts) == 2:
                            self.odometry.update_encoders(
                                int(parts[0]), int(parts[1])
                            )
                    elif msg.startswith("B:"):
                        # 下位机检测到前方障碍物。
                        print(f"[CHASSIS] 检测到障碍物！{msg}")
                        self.on_obstacle_detected()
                    # ---- 动作完成与涵洞扫描事件 ----
                    elif "@" in msg:
                        self.event_turn_done.set()
                    elif "Y_START" in msg:
                        self.event_y_start.set()
                    elif "Y_L" in msg:
                        self.event_y_l.set()
                    elif "Y_R" in msg:
                        self.event_y_r.set()
                    elif "Y_END" in msg:
                        self.event_y_end.set()
            except Exception as e:
                print(f"[Chassis Listener Error] {e}")
            time.sleep(0.01)

    def _rfid_listener(self):
        while self.is_running:
            try:
                uid = self.rfid.read_card()
                if uid:
                    print(f"[RFID] 读取到固定点卡片：{uid}")
                    self.safe_show_text(f"固定点卡号：{uid}")
                    self.safe_say_result(f"已到达固定巡逻点 {uid}")

                    # RFID 只对应 12 个固定巡逻点。
                    rfid_node = self.topo_map.get_node_by_rfid(uid)
                    if (rfid_node and
                            rfid_node.type == self.topo_map.NODE_FIXED_PATROL and
                            rfid_node.position):
                        self.odometry.calibrate_by_rfid(
                            uid, rfid_node.position[0], rfid_node.position[1]
                        )
                        if self.navigator.confirm_fixed_patrol(rfid_node.id):
                            self._sync_navigation_path()
                            progress = self.navigator.get_progress()
                            print(f"[NAVIGATION] fixed RFID {uid} -> "
                                  f"{rfid_node.id}; visited "
                                  f"{progress['visited']}/{progress['total']}")
                    else:
                        print(f"[RFID WARN] card {uid} is not mapped to N1-N12")

                    time.sleep(2)
            except Exception as e:
                print(f"[RFID Listener Error] {e}")
            time.sleep(0.05)

    def safe_show_text(self, text):
        """线程安全地在 OLED 上显示文字。"""
        with self.hardware_lock:
            try:
                self.oled.show_text(text)
            except Exception as e:
                print(f"[OLED Error] 显示文字失败：{e}")

    def safe_say_result(self, text):
        """线程安全地播报识别结果。"""
        with self.hardware_lock:
            try:
                self.broadcaster.say_result(text)
            except Exception as e:
                print(f"[Audio Error] 播报结果失败：{e}")

    def safe_play_file(self, path):
        """线程安全地播放指定音频文件。"""
        with self.hardware_lock:
            try:
                self.broadcaster.play_file(path)
            except Exception as e:
                print(f"[Audio Error] 播放音频文件失败：{e}")
    
    def _state_navigate(self):
        """障碍物触发重新规划后，恢复巡线。"""
        if not self.current_target_node:
            self._sync_navigation_path()
        if not self.current_target_node:
            self.state = "STOP" if self.navigator.mission_complete else "FOLLOW_LINE"
            return

        pos = self.odometry.get_position()
        print(f"[NAV] 目标:{self.current_target_node}  "
            f"航向:{pos['heading']:.1f}°  "
            f"里程:{pos['distance']:.2f}m  "
            f"置信度:{pos['confidence']:.2f}  "
            f"RFID:{pos['last_rfid']}", end="\r")

        self.state = "FOLLOW_LINE"

    def _state_obstacle_recovery(self):
        """在被封锁路段掉头，并沿线返回最后确认的节点。"""
        print("[OBSTACLE] U-turn; returning to the last confirmed node")
        self._execute_cross_action("U_TURN")
        self.state = "FOLLOW_LINE"
    def _execute_cross_action(self, action):
        """向下位机发送当前路口的原子动作并等待完成应答。"""
        self.event_turn_done.clear()
        if action == "FORWARD":
            self.chassis.go_cross()
        elif action == "RIGHT":
            self.chassis.cw()
        elif action == "LEFT":
            self.chassis.ccw()
        elif action == "U_TURN":
            self.chassis.turn(180)
        if self.event_turn_done.wait(timeout=10.0):
            print(f"[CROSS] {action} 动作完成")
        else:
            print(f"[CROSS WARN] {action} 动作超时")
        time.sleep(0.3)

    def run_mission(self):
        self._sync_navigation_path()
        progress = self.navigator.get_progress()
        print(f"[MISSION] 12 RFID fixed points + "
              f"{self.navigator.expected_green_count} unknown green culverts; "
              f"covering {progress['required_edges']} road segments; "
              f"first path: {self.current_path}")

        # 按状态机持续执行任务。
        while self.is_running:
            try:
                if self.state == "START":
                    self._state_start()
                elif self.state == "START_ALIGN":
                    self._state_start_align()
                elif self.state == "FOLLOW_LINE":
                    self._state_follow_line()
                elif self.state == "CROSSROAD":
                    self._state_crossroad()
                elif self.state == "SCAN_CULVERT":
                    self._state_scan_culvert()
                elif self.state == "NAVIGATE":
                    self._state_navigate()
                elif self.state == "OBSTACLE_RECOVERY":
                    self._state_obstacle_recovery()
                elif self.state == "RETURN_START":
                    self._state_return_start()
                elif self.state == "STOP":
                    self.chassis.stop()
                    print("[MISSION] 任务已结束")
                    break
            except Exception as e:
                print(f"[ERROR] 主循环发生异常：{e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
            time.sleep(0.02)

    def _state_start(self):
        """执行出发区固定动作，驶入可巡线道路并建立起始拓扑位置。"""
        print("\n[START_STATE] 正在执行出发区固定动作...")
        
        # ================== 第一步：短距离前进 ==================
        self.event_turn_done.clear()  # 清除上一条动作的完成事件。
        print("[START_STATE] 1. 短距离前进...")
        self.chassis.distance(val=12, dis=130)  # 使用编码器闭环前进。
        
        # 等待下位机返回动作完成应答。
        if self.event_turn_done.wait(timeout=5.0):
            print("[START_STATE] 前进动作完成")
        else:
            print("[START_STATE WARN] 前进动作超时")
            
        time.sleep(0.2)  # 留出机械惯性消退时间。
        
        # ================== 第二步：原地转向 ==================
        self.event_turn_done.clear()  # 清除完成事件。
        print("[START_STATE] 2. 原地转向...")
        self.chassis.turn_pure(80)              # 原地旋转到出发方向。
        
        # 等待下位机返回动作完成应答。
        if self.event_turn_done.wait(timeout=5.0):
            print("[START_STATE] 转向动作完成")
        else:
            print("[START_STATE WARN] 转向动作超时")
            
        time.sleep(0.2)  # 留出机械惯性消退时间。
        
        # ================== 第三步：倒车回到巡线路面 ==================
        self.event_turn_done.clear()  # 清除完成事件。
        print("[START_STATE] 3. 倒车回到巡线路面...")
        self.chassis.distance(val=-10, dis=-100)  # 编码器闭环倒车。
        
        # 等待下位机返回动作完成应答。
        if self.event_turn_done.wait(timeout=5.0):
            print("[START_STATE] 倒车动作完成")
        else:
            print("[START_STATE WARN] 倒车动作超时")
            
        # 固定动作只负责将抬高的摄像头调整到可看到一段普通黑线的位置，
        # 并未到达 J_TOP。保持 START_AREA，先完成短暂姿态巡线，再由第一次
        # 真实路口识别确认 J_TOP，避免把出发区动作错误计为通过路口。
        self.start_line_frames = 0
        self.last_cross_time = 0
        print("[START_STATE] 已进入姿态巡线阶段，等待真实 J_TOP 路口。\n")
        self.state = "START_ALIGN"

    def _state_start_align(self):
        """出发后沿普通黑线短暂巡线，稳定姿态后再开放路口识别。"""
        ret, frame = self.read_frame("nav")
        if not ret or frame is None or frame.size == 0:
            return

        road_type, offset = self.tracker.get_road_info(frame)
        turn_val = (self.Kp * offset) + (self.Kd * (offset - self.last_offset))
        self.last_offset = offset
        turn_val = max(min(turn_val, 30), -30)
        self.chassis.drive_steer(speed=self.base_speed, angle=turn_val)

        if road_type == "CROSS":
            # 尚未获得足够的普通黑线时不把该画面当作有效路口事件。
            self.start_line_frames = 0
            return

        self.start_line_frames += 1
        if self.start_line_frames >= self.start_line_required_frames:
            print("[START_ALIGN] 姿态巡线稳定，开始识别 J_TOP 路口")
            self.state = "FOLLOW_LINE"

    def _state_follow_line(self):
        """读取俯视摄像头，进行 PID 巡线并检测路口。"""
        ret, frame = self.read_frame("nav")
        if not ret or frame is None or frame.size == 0:
            # 摄像头暂时无有效画面时，保持本轮不发送新的巡线指令。
            return

        # 涵洞识别算法暂不实现。后续给 culvert_detector 注入
        # callable(frame)->bool 后，这里会自动触发下位机指令 10。
        if (self.culvert_detector is not None and
                self.culvert_detector(frame)):
            if self.on_culvert_detected():
                return

        # 获取道路类型和相对黑线偏移量。
        road_type, offset = self.tracker.get_road_info(frame)
        current_time = time.time()
        
        if road_type == "CROSS":
            if current_time - self.last_cross_time > self.cross_cooldown:
                print("\n[Auto] 检测到路口，进入路口决策状态")
                self.chassis.stop() 
                time.sleep(0.2)
                self.state = "CROSSROAD"
                self.last_cross_time = current_time 
                return
            
        turn_val = (self.Kp * offset) + (self.Kd * (offset - self.last_offset))
        self.last_offset = offset
        turn_val = max(min(turn_val, 30), -30)
        self.chassis.drive_steer(speed=self.base_speed, angle=turn_val)

    def _state_crossroad(self):
        target = self.topo_map.get_node(self.current_target_node)
        if target is None:
            self._sync_navigation_path()
            target = self.topo_map.get_node(self.current_target_node)
        if target is None:
            self.state = "STOP" if self.navigator.mission_complete else "FOLLOW_LINE"
            return

        if self.obstacle_recovering:
            # 车辆已物理返回 current_node，不应再推进图路径；使用被封锁
            # 路段作为来向，选择新的绕行分支。
            next_action = self.topo_map.turn_action(
                self.obstacle_return_from,
                self.current_node,
                self.current_target_node,
            )
            print(f"[OBSTACLE] back at {self.current_node}; detour to "
                  f"{self.current_target_node}: {next_action}")
            self.obstacle_recovering = False
            self.obstacle_return_from = None
            self._execute_cross_action(next_action)
            self.state = "FOLLOW_LINE"
            return

        if target.type == self.topo_map.NODE_GREEN_TASK:
            print(f"[CROSSROAD ERROR] expected RFID point {target.id} was not "
                  "detected; stopping instead of skipping it")
            self.chassis.stop()
            self.state = "STOP"
            return

        previous = self.current_node
        current = self._confirm_next_node()
        next_node = self.current_target_node
        if not current or not next_node:
            self.state = "STOP" if self.navigator.mission_complete else "FOLLOW_LINE"
            return
        next_action = self.topo_map.turn_action(previous, current, next_node)
        print(f"[CROSSROAD] {previous} -> {current} -> {next_node}: "
              f"{next_action}")
        self._execute_cross_action(next_action)
        if next_node == "START_AREA":
            # 转向后车辆已进入出发区短支路。终点没有路口或 RFID，
            # 因此改用专用的编码器定距状态，而不依赖视觉识别。
            self.state = "RETURN_START"
        else:
            self.state = "FOLLOW_LINE"

    def _state_return_start(self):
        """驶入狭小出发框，且仅在收到下位机完成应答后确认到达。"""
        print("[RETURN] entering START_AREA by encoder distance: "
              f"speed={self.return_start_speed}, "
              f"distance={self.return_start_distance}")
        self.event_turn_done.clear()
        self.chassis.distance(
            val=self.return_start_speed,
            dis=self.return_start_distance,
        )
        if not self.event_turn_done.wait(timeout=self.return_start_timeout):
            print("[RETURN ERROR] no completion ACK; START_AREA not confirmed")
            self.chassis.stop()
            self.state = "STOP"
            return

        reached = self._confirm_next_node()
        if reached != "START_AREA":
            print(f"[RETURN ERROR] expected START_AREA, got {reached}")
            self.chassis.stop()
            self.state = "STOP"
            return

        print("[RETURN] START_AREA reached; mission complete")
        self.state = "STOP"

    def _state_scan_culvert(self):
        """视觉识别到涵洞后，等待下位机云台扫描并识别目标。"""
        print("[CULVERT] scanning target with the high camera...")

        left_target = -1
        right_target = -1

        # 指令 10 被下位机接受并停车后，首先上报 Y_START。
        if self.event_y_start.wait(timeout=3.0):
            self.event_y_start.clear()
            print("[DETECT] 下位机已开始侦查扫描")
        else:
            print("[DETECT WARN] 等待 Y_START 超时，取消本次扫描")
            self.active_culvert_key = None
            self.state = "FOLLOW_LINE"
            return

        # 1. 等待云台左扫到位事件，并获取左侧画面。
        if self.event_y_l.wait(timeout=15.0):
            self.event_y_l.clear()
            print("[DETECT] 云台已到左侧，开始识别左侧画面")
            ret, frame = self.read_frame("detect")
            if ret and frame is not None:
                left_target = self.yolo.detect_target(frame)
                print(f"[DETECT] 左侧识别结果：{left_target}")
            else:
                print("[DETECT WARN] 无法获取左侧识别摄像头画面")
        else:
            print("[DETECT WARN] 等待云台左扫事件超时")

        # 2. 等待云台右扫到位事件，并获取右侧画面。
        if self.event_y_r.wait(timeout=15.0):
            self.event_y_r.clear()
            print("[DETECT] 云台已到右侧，开始识别右侧画面")
            ret, frame = self.read_frame("detect")
            if ret and frame is not None:
                right_target = self.yolo.detect_target(frame)
                print(f"[DETECT] 右侧识别结果：{right_target}")
            else:
                print("[DETECT WARN] 无法获取右侧识别摄像头画面")
        else:
            print("[DETECT WARN] 等待云台右扫事件超时")

        # 3. 等待云台归中及扫描结束事件。
        scan_finished = self.event_y_end.wait(timeout=10.0)
        if scan_finished:
            self.event_y_end.clear()
            print("[DETECT] 云台扫描结束")
        else:
            print("[DETECT WARN] 等待云台归中事件超时")
        
        # 4. 优先采用左侧识别结果；未识别时再采用右侧结果。
        final_target = left_target if left_target != -1 else right_target
        
        time.sleep(0.5) 
        if final_target != -1:
            print(f"[DETECT SUCCESS] 识别到目标类别：{final_target}")
            self.safe_show_text(f"目标类别：{final_target}")
            self.safe_say_result(f"发现目标，类别为 {final_target}")
            time.sleep(3.0)
        else:
            print("[DETECT FAIL] 未识别到预设目标")
            self.safe_show_text("未识别到目标")
            
            none_sound_path = os.path.join(self.broadcaster.base_path, "yolo_none.mp3")
            if os.path.exists(none_sound_path):
                print(f"[DETECT FAIL] 播放未发现目标音频：{none_sound_path}")
                self.safe_play_file(none_sound_path)
                time.sleep(3.5)
            else:
                print(f"[DETECT FAIL] 未找到未发现目标音频：{none_sound_path}")
                time.sleep(1.0)
        
        if scan_finished and self.active_culvert_key:
            if self.navigator.mark_culvert_scanned(self.active_culvert_key):
                progress = self.navigator.get_progress()
                print(f"[CULVERT] completed {self.active_culvert_key}; "
                      f"{progress['green_found']}/"
                      f"{progress['green_expected']}")
        elif scan_finished:
            print("[CULVERT WARN] scan completed but could not localize its road")
        self.active_culvert_key = None
        
        # 识别期间导航摄像头持续采集，无需释放、重新打开或额外恢复等待。
        print("[DETECT] suspect-camera scan complete; navigation stream intact")
        time.sleep(0.2)
        
        print("[DETECT] 涵洞目标扫描完成，恢复巡线")
        self.state = "FOLLOW_LINE"

    def shutdown(self):
        print("[SYS] 正在安全关闭系统...")
        self.is_running = False
        time.sleep(0.5)  # 给后台线程留出退出和收尾时间。
        
        # 停止并关闭底盘串口。
        try:
            self.chassis.stop()
            self.chassis.close()
        except:
            pass
        
        # 关闭 RFID 读卡器。
        try:
            self.rfid.disconnect()
        except:
            pass
        
        # 释放 YOLO 识别器资源。
        try:
            if self.yolo is not None:
                self.yolo.release()
        except:
            pass

        # 释放 DeepLabV3 RKNN 巡线识别器资源。
        try:
            if self.tracker is not None:
                self.tracker.release()
        except:
            pass
        
        for camera_name, camera in self.cameras.items():
            with camera["capture_lock"]:
                if camera["cap"] is not None and camera["cap"].isOpened():
                    camera["cap"].release()
            print(f"[CAMERA] released {camera_name}")
            
        cv2.destroyAllWindows()
        print("[SYS] 系统已安全关闭")


if __name__ == "__main__":
    robot = None
    try:
        robot = ReconRobot()
        robot.start_background_tasks()
        robot.run_mission()
    except KeyboardInterrupt:
        print("\n[SYS] 用户中断程序（Ctrl+C）")
    except Exception as e:
        import traceback
        print("\n[SYS ERROR] 程序发生未处理异常：")
        traceback.print_exc()
    finally:
        if robot:
            robot.shutdown()
        sys.exit(0)
