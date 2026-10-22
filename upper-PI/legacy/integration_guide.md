> ⚠️ **本文档已过时，仅作历史参考，不要照着改代码。**
>
> 它写于"把里程计接进 main.py"那个阶段，其中提到的东西与现状不符：
> - 指令 `@|13`（陀螺仪朝向）**已经不存在**——下位机去掉了 IMU，`H:` 遥测不再发送；
> - 遥测间隔指令当时写的是 `16`，实际固件是 `17`；
> - 文中 `main.py` 的改法针对的是旧版集成程序，现行主程序是 `src/full_map_walk_test.py`，
>   里程计已接入但**遥测未开启**（详见 [docs/07-交接记录.md](../../docs/07-交接记录.md) 遗留问题 1）。
>
> 保留它的唯一理由：如果下一届想把里程计真正跑起来，这里的调用时机（收 `E:` 更新、
> RFID 校准、状态机里读位置）是当初的设计意图，仍有参考价值。

# 拓扑地图 & 航迹推算 集成指南

## 文件清单

将以下两个文件复制到 `~/Desktop/src/` 下：
- `odometry.py` - 航迹推算模块
- `topo_map.py` - 拓扑地图模块

## 1. 扩展 Serial.py

在 `ChassisController` 类中添加以下方法（放在 `drive_steer` 后面即可）：

```python
# ================== 新增：遥测请求 ==================

def request_heading(self):
    """请求陀螺仪朝向"""
    self._send_command("@|13|0|0|0#")

def request_encoders(self):
    """请求编码器计数"""
    self._send_command("@|14|0|0|0#")

def enable_telemetry(self, enable=True):
    """开启/关闭下位机自动遥测上报"""
    val = 1 if enable else 0
    self._send_command(f"@|15|{val}|0|0#")

def set_telemetry_interval(self, ms=50):
    """设置遥测上报间隔 (ms)"""
    self._send_command(f"@|16|{int(ms)}|0|0#")
```

同时修改 `read_message()`，在解析 `\n` 行之后，加入遥测模式识别：

```python
# 在 read_message 中，return line 之前，加入解析
if ":" in line:
    # 遥测数据，不返回给上层事件系统，而是直接返回让 chassis_listener 处理
    return line  # 调用方根据前缀区分
```

## 2. 修改 main.py

### 2a. 新增 import

替换原来的 import 块：

```python
from odometry import Odometry
from topo_map import TopoMap
```

### 2b. 在 ReconRobot.__init__ 中添加初始化

```python
# 放在底盘初始化后面
# ====== 拓扑地图 & 航迹推算 ======
self.topo_map = TopoMap()
self.topo_map.build_default_map()
self.topo_map.pretty_print()  # 打印地图结构便于调试

self.odometry = Odometry()

# 任务计划：需要依次到达的节点ID列表
self.mission_plan = ["P1", "P2", "T1", "P3", "E"]
self.current_path = None       # 当前完整路径
self.current_target_node = None  # 当前目标节点
self.blocked_edges = set()     # 被障碍挡住的边
```

### 2c. 修改 _chassis_listener

```python
def _chassis_listener(self):
    while self.is_running:
        try:
            msg = self.chassis.read_message()
            if msg:
                # ---- 遥测数据解析 ----
                if msg.startswith("H:"):
                    yaw = float(msg[2:])
                    self.odometry.update_heading(yaw)
                elif msg.startswith("E:"):
                    parts = msg[2:].split(",")
                    if len(parts) == 2:
                        self.odometry.update_encoders(
                            int(parts[0]), int(parts[1])
                        )
                elif msg.startswith("B:"):
                    # 障碍物触发
                    print(f"[CHASSIS] 障碍物触发! {msg}")
                    self.on_obstacle_detected()
                # ---- 原有信号解析 ----
                elif "@" in msg:
                    self.event_turn_done.set()
                elif "Y_START" in msg:
                    self.state = "DETECT_TUNNEL"
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
```

### 2d. 新增障碍物处理方法

```python
def on_obstacle_detected(self):
    """障碍物触发：标记当前边为阻塞，重新规划路径"""
    print("[OBSTACLE] 检测到障碍物！正在重新规划路径...")
    self.chassis.stop()

    if self.current_target_node and self.current_path:
        # 找到当前所在边并标记为阻塞
        current_node = self.current_path[
            self.current_path.index(self.current_target_node) - 1
        ] if self.current_target_node in self.current_path else None

        if current_node:
            self.blocked_edges.add((current_node, self.current_target_node))
            print(f"[OBSTACLE] 阻塞边: {current_node} -> {self.current_target_node}")

            # 重新规划到终点的路径
            start_node = self.topo_map.get_node_by_rfid(
                self.odometry.last_rfid_id
            )
            start_id = start_node.id if start_node else "S"
            target_id = self.mission_plan[-1]

            new_path = self.topo_map.find_path(
                start_id, target_id, self.blocked_edges
            )
            if new_path:
                print(f"[OBSTACLE] 新路径: {new_path}")
                self.current_path = new_path
                self.state = "NAVIGATE"
            else:
                print("[OBSTACLE] 无可用绕行路径！紧急停止")
                self.chassis.stop()
                self.state = "STOP"
```

### 2e. 修改 _rfid_listener

```python
def _rfid_listener(self):
    while self.is_running:
        try:
            uid = self.rfid.read_card()
            if uid:
                print(f"[RFID 命中] 到达 {uid} 号巡逻点！")
                self.safe_show_text(f"到达巡逻点: {uid}")
                self.safe_say_result(f"到达{uid}号巡逻点，表示播报成功")

                # ---- 新增：校准航迹推算 ----
                rfid_node = self.topo_map.get_node_by_rfid(uid)
                if rfid_node and rfid_node.position:
                    self.odometry.calibrate_by_rfid(
                        uid, rfid_node.position[0], rfid_node.position[1]
                    )
                else:
                    self.odometry.calibrate_by_rfid(uid)

                # ---- 新增：检查任务推进 ----
                if self.current_path:
                    matched_id, next_node = self.topo_map.get_node_by_rfid_in_path(
                        uid, self.current_path
                    )
                    if next_node:
                        self.current_target_node = next_node
                        print(f"[NAVIGATION] 下一目标节点: {next_node}")
                    elif matched_id == self.current_path[-1]:
                        print(f"[NAVIGATION] 到达终点！")
                        self.state = "STOP"

                time.sleep(2)
        except Exception as e:
            print(f"[RFID Listener Error] {e}")
        time.sleep(0.05)
```

### 2f. 修改 run_mission() 状态机

加入 `NAVIGATE` 状态：

```python
def run_mission(self):
    # ... 现有代码保持不变 ...

    # 初始化第一段路径
    self.current_path = self.topo_map.find_path("S", self.mission_plan[0])
    if self.current_path and len(self.current_path) > 1:
        self.current_target_node = self.current_path[1]
    else:
        self.current_path = self.topo_map.find_path("S", self.mission_plan[-1])
        self.current_target_node = self.mission_plan[0]

    # 状态机主循环
    while self.is_running:
        try:
            if self.state == "START":
                self._state_start()
            elif self.state == "FOLLOW_LINE":
                self._state_follow_line()
            elif self.state == "CROSSROAD":
                self._state_crossroad()
            elif self.state == "DETECT_TUNNEL":
                self._state_detect_tunnel()
            elif self.state == "NAVIGATE":
                self._state_navigate()
            elif self.state == "STOP":
                self.chassis.stop()
                print("[MISSION] 任务全部完成！")
                break
        except Exception as e:
            print(f"[ERROR] 状态机执行异常: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)
        time.sleep(0.02)
```

### 2g. 新增 _state_navigate 方法

```python
def _state_navigate(self):
    """导航状态：根据拓扑地图和航迹推算移动到下一节点"""
    if not self.current_target_node:
        self.state = "FOLLOW_LINE"
        return

    pos = self.odometry.get_position()
    print(f"[NAV] 目标:{self.current_target_node}  "
          f"朝向:{pos['heading']:.1f}°  "
          f"距离:{pos['distance']:.2f}m  "
          f"置信度:{pos['confidence']:.2f}  "
          f"RFID:{pos['last_rfid']}", end="\r")

    # 执行对应边的动作
    current_node_id = self.current_path[
        max(0, self.current_path.index(self.current_target_node) - 1)
    ] if self.current_target_node in self.current_path else "S"

    actions = self.topo_map.get_actions_between(
        current_node_id, self.current_target_node
    )
    if actions:
        for action in actions:
            if action == "FOLLOW_LINE":
                # 恢复普通巡线
                self.state = "FOLLOW_LINE"
                return
            elif action == "LEFT":
                self._execute_cross_action("LEFT")
            elif action == "RIGHT":
                self._execute_cross_action("RIGHT")
            elif action == "FORWARD":
                self._execute_cross_action("FORWARD")
            elif action == "FOLLOW_CROSS":
                self.chassis.go_cross()
                time.sleep(2.0)
                self.state = "FOLLOW_LINE"
                return

    self.state = "FOLLOW_LINE"

def _execute_cross_action(self, action):
    """执行路口动作"""
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
        print(f"[CROSS WARN] {action} 超时")
    time.sleep(0.3)
```

## 3. 完整改动流程

1. 将 `odometry.py` 和 `topo_map.py` 复制到 `~/Desktop/src/`
2. 在 `Serial.py` 中加入遥测请求方法
3. 修改 `main.py` 的 import、init、listeners 和状态机

完成后，检查：
- 下位机固件需要支持 `@|13~16` 命令（陀螺仪和编码器上报）
- 下位机需要自动上报 `H:xxx`, `E:l,r` 格式的遥测数据
- 启动时通过 `chassis.enable_telemetry(True)` 开启自动上报
- 起点通过 `odometry.reset_heading(imu_yaw)` 校准方向
