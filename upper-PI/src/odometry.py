"""
纯编码器里程计航迹推算模块

基于下位机上报的左右轮累计编码器数据，推算机器人位置和姿态。
航向由差速轮模型计算，不依赖陀螺仪 yaw。RFID 节点过卡时做绝对
位置校准，消除位置积分漂移。

使用方式：
  1. 在 _chassis_listener 收到 E:l,r 时调用 odometry.update_encoders(l, r)
  2. 在 _rfid_listener 读到卡片时调用 odometry.calibrate_by_rfid(id)
  3. 主循环通过 odometry.get_position() 获取当前姿态
"""

import time
import threading
import math


class Odometry:
    def __init__(self):
        self.lock = threading.Lock()

        # === 姿态 ===
        self.heading = 0.0

        # === 位置 ===
        self.x = 0.0
        self.y = 0.0
        self.distance = 0.0

        # === 编码器 ===
        self.encoder_left = 0
        self.encoder_right = 0
        self.encoder_prev_left = 0
        self.encoder_prev_right = 0
        self.encoders_initialized = False
        self.wheel_base = 0.18
        self.wheel_diameter = 0.065
        self.ticks_per_rev = 500
        self._dist_per_tick = math.pi * self.wheel_diameter / self.ticks_per_rev

        # === RFID 校准 ===
        self.last_rfid_id = None
        self.last_rfid_time = 0
        self.rfid_positions = {}

        # === 时间戳 / 角速度 ===
        self.last_update = time.time()
        self.angular_velocity = 0.0

        # === 置信度 ===
        self.position_confidence = 0.0
        self.distance_since_calib = 0.0
        self.confidence_decay_distance = 10.0
        self.min_confidence = 0.1

    # ======================== 数据更新接口 ========================

    def update_encoders(self, left_ticks, right_ticks):
        """由左右轮累计脉冲更新差速里程计。"""
        with self.lock:
            now = time.time()
            # 首帧只建立累计计数基准，不产生虚假的位移或转角。
            if not self.encoders_initialized:
                self.encoder_left = left_ticks
                self.encoder_right = right_ticks
                self.encoder_prev_left = left_ticks
                self.encoder_prev_right = right_ticks
                self.encoders_initialized = True
                self.last_update = now
                return

            delta_l = left_ticks - self.encoder_left
            delta_r = right_ticks - self.encoder_right
            self.encoder_prev_left = self.encoder_left
            self.encoder_prev_right = self.encoder_right
            self.encoder_left = left_ticks
            self.encoder_right = right_ticks
            dl = delta_l * self._dist_per_tick
            dr = delta_r * self._dist_per_tick
            d_center = (dl + dr) / 2.0

            # 差速轮运动学：左右轮路程差除以轮距得到航向增量（弧度）。
            d_theta = (dr - dl) / self.wheel_base
            heading_before = math.radians(self.heading)
            heading_mid = heading_before + d_theta / 2.0

            self.x += d_center * math.cos(heading_mid)
            self.y += d_center * math.sin(heading_mid)
            self.heading = math.degrees(heading_before + d_theta) % 360.0
            self.distance += abs(d_center)
            self.distance_since_calib += abs(d_center)

            dt = now - self.last_update
            self.angular_velocity = d_theta / dt if dt > 0 else 0.0
            if self.distance_since_calib > self.confidence_decay_distance:
                self.position_confidence = max(
                    self.min_confidence,
                    self.position_confidence * 0.99,
                )
            self.last_update = now

    # ======================== RFID 校准 ========================

    def calibrate_by_rfid(self, rfid_id, known_x=None, known_y=None):
        with self.lock:
            self.last_rfid_id = rfid_id
            self.last_rfid_time = time.time()
            if known_x is not None and known_y is not None:
                self.x = known_x
                self.y = known_y
            self.distance_since_calib = 0.0
            self.position_confidence = 1.0
            self.rfid_positions[rfid_id] = (self.x, self.y)

    def reset_heading(self, yaw_deg):
        """兼容旧接口：将当前编码器航向重置为指定角度。"""
        with self.lock:
            self.heading = float(yaw_deg) % 360.0
            self.position_confidence = 0.5

    # ======================== 查询接口 ========================

    def get_position(self):
        with self.lock:
            return {
                "heading": round(self.heading, 2),
                "x": round(self.x, 4),
                "y": round(self.y, 4),
                "distance": round(self.distance, 4),
                "last_rfid": self.last_rfid_id,
                "confidence": round(self.position_confidence, 3),
                "angular_velocity": round(self.angular_velocity, 4),
                "encoder_left": self.encoder_left,
                "encoder_right": self.encoder_right,
            }

    def is_facing(self, target_heading, tolerance=15):
        diff = (target_heading - self.heading + 180) % 360 - 180
        return abs(diff) < tolerance

    def heading_error(self, target_heading):
        return (target_heading - self.heading + 180) % 360 - 180

    def distance_to(self, target_x, target_y):
        return math.hypot(target_x - self.x, target_y - self.y)

    def get_rfid_position(self, rfid_id):
        return self.rfid_positions.get(rfid_id)
