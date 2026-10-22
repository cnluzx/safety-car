import serial
import time
import threading

class ChassisController:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200):
        """初始化底盘串口"""
        self.port = port
        self.baudrate = baudrate
        self.current_cam_angle = 90
        self.rx_buffer = ""
        self.lock = threading.Lock()
        
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0  # 改为0，非阻塞模式
            )
            print(f"[Serial] 底盘串口 {self.port} 已连接.")
        except Exception as e:
            print(f"[Serial Error] 无法打开底盘串口 {self.port}: {e}")
            self.serial = None

    def turn_pure(self, angle):
        """原地旋转（对应下位机指令码 3）。"""
        cmd = f"@|3|1|{int(angle)}|0#"
        self._send_command(cmd)
    
    def turn_all(self, angle):
        """原地旋转（对应下位机指令码 3）。"""
        cmd = f"@|4|1|{int(angle)}|0#"
        self._send_command(cmd)

    # # def wait_for_completion(self, timeout=10.0):
    #     """阻塞等待下位机返回 '@' 结束信号"""
    #     import time
    #     start_time = time.time()
    #     while time.time() - start_time < timeout:
    #         msg = self.read_message()
    #         if msg == "@":
    #             return True
    #         time.sleep(0.01)
    #     print("[Serial Warning] 等待动作完成超时")
    #     return False


    def _send_command(self, cmd_str):
        """底层发送协议的方法"""
        try:
            if self.serial and self.serial.is_open:
                with self.lock:
                    self.serial.write(cmd_str.encode('utf-8'))
                # print(f"[Serial TX] -> {cmd_str}")  # 保持打印
            else:
                print(f"[Serial Error] 串口未开启，无法发送: {cmd_str}")
        except Exception as e:
            print(f"[Serial FATAL] 发送异常: {e}")

    def read_message(self):
        """非阻塞读取消息（已修复缓存丢包问题）。"""
        if not self.serial or not self.serial.is_open:
            return None
        
        try:
            with self.lock:
                # 1. 尽可能多地读取串口数据，追加存入缓存
                if self.serial.in_waiting > 0:
                    data = self.serial.read(self.serial.in_waiting)
                    if data:
                        try:
                            text = data.decode('UTF-8', errors='ignore')
                            self.rx_buffer += text
                        except Exception as e:
                            print(f"[Serial Decode Error] {e}")
            
            # 2. 精确逐行截取，绝对不丢弃缓存中后面的行
            with self.lock:
                if '\n' in self.rx_buffer:
                    idx = self.rx_buffer.find('\n')
                    line = self.rx_buffer[:idx].strip()
                    # 仅移除已经读取的第一行，保留剩下的数据（如 Y_L\n）供下一次循环读取
                    self.rx_buffer = self.rx_buffer[idx+1:]
                    if line:
                        return line
        except OSError as e:
            if "Inappropriate ioctl" not in str(e):
                print(f"[Serial Error] {e}")
        except Exception as e:
            print(f"[Serial Error] {e}")
        
        return None

    def clear_pending_input(self):
        """丢弃发送新动作前遗留的旧应答和未解析串口数据。"""
        if not self.serial or not self.serial.is_open:
            return
        try:
            with self.lock:
                self.serial.reset_input_buffer()
                self.rx_buffer = ""
        except Exception as e:
            print(f"[Serial Error] 清理接收缓存失败: {e}")

    # ================== 原有运动指令 ==================

    def go(self, val=10, detect_mode=1):
        num = 1 if detect_mode == 1 else 11
        cmd = f"@|{num}|{val}|60|55#"
        self._send_command(cmd)

    def distance(self, val=10, dis=100, detect_mode=1):
        num = 6 if detect_mode == 1 else 16
        cmd = f"@|{num}|{val}|{dis}|55#"
        self._send_command(cmd)

    def cw(self):
        cmd = f"@|2|1|90|0#"
        self._send_command(cmd)

    def ccw(self):
        cmd = f"@|2|1|-90|0#"
        self._send_command(cmd)

    def turn(self, angle):
        cmd = f"@|2|1|{angle}|0#"
        self._send_command(cmd)

    def mini_adjust(self, dis=45):
        cmd = f"@|11|10|{dis}|55#"
        self._send_command(cmd)

    def go_cross(self, speed=9, distance_multiplier=1):
        """Cross an intersection with encoder-closed distance control."""
        cmd = f"@|7|{int(speed)}|{int(distance_multiplier):02d}|55#"
        self._send_command(cmd)
    
    def request_encoders(self):
        """请求编码器计数"""
        self._send_command("@|14|0|0|0#")

    def enable_telemetry(self, enable=True):
        """开启/关闭下位机自动遥测上报"""
        val = 1 if enable else 0
        self._send_command(f"@|15|{val}|0|0#")

    def set_telemetry_interval(self, ms=50):
        """设置遥测上报间隔（毫秒）。"""
        self._send_command(f"@|17|{int(ms)}|0|0#")

    def trigger_culvert_inspection(self):
        """通知下位机：视觉已识别到涵洞，开始侦查扫描。"""
        self._send_command("@|10|0|0|0#")
    
    # def go_start(self):
    #     import time
        
    #     # 1. 前进一段距离 (使用 case 6)
    #     # 例如：速度设为 15，距离系数设为 100（目标 1500 脉冲，约几十厘米，可根据实际赛道微调）
    #     print("[go_start] 正在前进...")
    #     self.distance(val=11, dis=120)  # 发送 @|6|10|90|55#   
        
    #     # 2. 必须等待前进动作彻底完成
    #     self.wait_for_completion(timeout=5.0)
        
    #     # 3. 短暂微休（消解惯性，保护电机）
    #     time.sleep(0.2)
        
    #     # 4. 原地右转 90 度 (使用 case 3)
    #     # 此时角度设为 90，对应的脉冲数为 90 * 14 = 1260 脉冲
    #     print("[go_start] 正在右转...")
    #     self.turn_pure(80)              # 发送 @|3|1|90|0#
        
    #     # 5. 等待右转动作完成
    #     self.wait_for_completion(timeout=4.0)
    #     print("[go_start] 启动并右转全部完成！")

    def stop(self):
        cmd = f"@|5|0|0|0#"
        self._send_command(cmd)

    def drive_steer(self, speed, angle):
        cmd = f"@|8|{int(speed)}|{int(angle)}#"
        self._send_command(cmd)

    # ================== 新增：前后左右微调 ==================
    
    def move_forward(self, speed=20):
        self.drive_steer(speed=speed, angle=0)

    def move_backward(self, speed=20):
        self.drive_steer(speed=-speed, angle=0)

    def spin_left(self, speed=20):
        self.drive_steer(speed=0, angle=-speed)

    def spin_right(self, speed=20):
        self.drive_steer(speed=0, angle=speed)

    # ================== 云台(舵机)控制 ==================

    def set_camera_angle(self, angle):
        """立即转动云台（原有方法）。"""
        cmd = f"@|9|{int(angle)}|0|0#"
        self._send_command(cmd)

    def set_camera_angle_slow(self, target_angle, step=2, delay=0.02):
        """
        【防崩溃神技】：缓慢转动舵机，防电流过载导致重启
        """
        if self.current_cam_angle == target_angle:
            return
            
        step_dir = step if target_angle > self.current_cam_angle else -step
        
        for a in range(self.current_cam_angle, target_angle, step_dir):
            self.set_camera_angle(a)
            time.sleep(delay)
            
        self.set_camera_angle(target_angle)
        self.current_cam_angle = target_angle

    def close(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("[Serial] 底盘串口已关闭.")

if __name__ == "__main__":
    # 测试代码
    controller = ChassisController(port="/dev/ttyUSB0", baudrate=115200)
    
    try:
        controller.turn_pure(angle=90)
    finally:
        controller.close()
    
