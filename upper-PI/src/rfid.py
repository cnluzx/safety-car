import serial
import serial.tools.list_ports
import select
import time
import binascii

class RFIDReader:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.data_buffer = ""
        self.last_digit_23 = None
        self.last_read_time = 0
        self.connect()

    def list_serial_ports(self):
        ports = serial.tools.list_ports.comports()
        available_ports = []
        for port in ports:
            available_ports.append(port.device)
        return available_ports
    
    def connect(self):
        try:
            if not self.port:
                ports = self.list_serial_ports()
                usb_ports = [p for p in ports if "FT232R" not in p]
                self.port = usb_ports[0] if usb_ports else (ports[-1] if ports else None)
                if not self.port: return False
            
            self.ser = serial.Serial(
                port=self.port, baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE, timeout=0.01
            )
            print(f"[RFID] 成功连接到读卡器: {self.port}")
            return True
        except Exception as e:
            print(f"[RFID Error] 连接读卡器失败: {e}")
            return False

    def read_card(self):
        if not self.ser or not self.ser.is_open:
            return None

        if self.ser.in_waiting > 0:
            try:
                raw_data = self.ser.read(self.ser.in_waiting)
                # 过滤并拼接十六进制合法字符
                new_hex = binascii.hexlify(raw_data).decode().upper()
                
                # 💡 只要物理串口接收到任何字节，就必定打印
                # print(f"[RAW RECEIVE] 物理接收到原始数据: {new_hex}")
                
                self.data_buffer += ''.join(c for c in new_hex if c in '0123456789ABCDEF')
                
                while True:
                    # 1. 寻找帧头 7F
                    header_idx = self.data_buffer.find('7F')
                    if header_idx == -1:
                        if len(self.data_buffer) > 0:
                            self.data_buffer = self.data_buffer[-1:]
                        break
                    
                    self.data_buffer = self.data_buffer[header_idx:]
                    
                    # 2. 检查长度是否足够一整帧 (56个十六进制字符 = 28字节)
                    if len(self.data_buffer) < 56:
                        break  
                    
                    # 截取一整帧数据 (56位)
                    frame = self.data_buffer[:56]
                    
                    # 3. 严格校验格式
                    if frame[2:4] != '1A' or frame[6:8] != '91':
                        print(f"[DEBUG] 帧头校验不匹配，滑过此帧头。当前帧片段: {frame[:12]}")
                        self.data_buffer = self.data_buffer[2:]
                        continue
                        
                    # 4. XOR 校验和验证
                    bytes_data = [int(frame[i:i+2], 16) for i in range(2, 56, 2)]
                    checksum = 0
                    for b in bytes_data[:-1]:  
                        checksum ^= b
                    
                    if checksum != bytes_data[-1]:
                        print(f"[RFID Checksum Error] 校验和不匹配！计算值: {hex(checksum).upper()}, 接收值: {hex(bytes_data[-1]).upper()}")
                        self.data_buffer = self.data_buffer[2:]  
                        continue
                    
                    # 5. 提取真正的数据块第一字节 (第 22、23 位字符)
                    point_hex = frame[22:24]
                    raw_val = int(point_hex, 16)
                    
                    # 清除已解析帧
                    self.data_buffer = self.data_buffer[56:]
                    
                    # 6. 空白卡过滤与合法范围过滤
                    if raw_val == 0:
                        print(f"[DEBUG] 检测到空白白卡 (数据块值为 00)，已被安全过滤。")
                        continue
                    
                    # === 智能多方案解析机制 ===
                    high_nibble = raw_val >> 4
                    low_nibble = raw_val & 0x0F
                    
                    # 方案 A: 高位半字节（对应你手上的卡片：0x80 -> 8, 0x90 -> 9）
                    val_scheme_a = high_nibble
                    
                    # 方案 B: 直接16进制转10进制（标准16进制：0x08 -> 8, 0x0A -> 10）
                    val_scheme_b = raw_val
                    
                    # 方案 C: BCD 码格式（BCD格式：0x10 -> 10, 0x12 -> 12）
                    val_scheme_c = high_nibble * 10 + low_nibble
                    
                    # 💡 调试输出：刷卡时会把三种可能解析的数值全部打印出来，方便你比对
                    # print(f"[RFID Decode Debug] 原始16进制字节: {point_hex} (十进制: {raw_val})")
                    # print(f"  └─ 方案 A (高位半字节解析): {val_scheme_a}")
                    # print(f"  └─ 方案 B (直接16进制解析): {val_scheme_b}")
                    # print(f"  └─ 方案 C (BCD 编码解析):   {val_scheme_c}")
                    
                    # 默认判定逻辑：
                    # 如果低位为0且高位在1-12之间（对应 0x80, 0x90 等卡片），优先用方案 A
                    if raw_val >= 0x10 and low_nibble == 0:
                        point_id = val_scheme_a
                    else:
                        # 否则，默认使用直接16进制（如果是两位数，可修改为 val_scheme_c）
                        point_id = val_scheme_b
                        
                    # 7. 合法范围过滤 (精准锁定 1 到 12 个点位)
                    if 1 <= point_id <= 12:
                        current_time = time.time()
                        if point_id == self.last_digit_23 and (current_time - self.last_read_time) < 3.0:
                            return None
                        
                        self.last_digit_23 = point_id
                        self.last_read_time = current_time
                        
                        # print(f"[RFID SUCCESS] 成功识别巡逻点卡片! ID: {point_id} (原始十六进制: {point_hex})")
                        return point_id
                    else:
                        print(f"[RFID Warn] 读取到非比赛范围 ID: {point_id}，已忽略")
                        
            except Exception as e:
                print(f"[RFID Read Error] 解析异常: {e}")
                
        return None

    def wait_for_card(self, timeout=0.2):
        """等待串口可读后解析数据，避免无数据时固定周期空轮询。"""
        if not self.ser or not self.ser.is_open:
            return None
        try:
            readable, _, _ = select.select(
                [self.ser.fileno()], [], [], max(0.0, float(timeout))
            )
        except (OSError, ValueError) as exc:
            print(f"[RFID Wait Error] 串口等待异常: {exc}")
            return None
        if not readable:
            return None
        return self.read_card()

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

if __name__ == "__main__":
    print("=== RFID 独立测试模式 ===")
    import sys
    test_port = "/dev/ttyACM0" if len(sys.argv) < 2 else sys.argv[1]
    reader = RFIDReader(port=test_port)
    if reader.ser and reader.ser.is_open:
        print(f"正在监听 {test_port}，请刷卡测试... (按 Ctrl+C 退出)")
        try:
            while True:
                card_id = reader.wait_for_card(timeout=0.2)
                if card_id:
                    print(f">>>> 成功识别并转换! 比赛巡逻点ID: {card_id}")
        except KeyboardInterrupt:
            print("\n测试结束")
        finally:
            reader.disconnect()
