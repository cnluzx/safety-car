"""高位摄像头移动至识别姿态，再朝车体左侧保持。不会执行完整扫描。"""
import argparse
import os
import time

DEFAULT_PORT = '/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A5069RR4-if00-port0'

def angle(value):
    value = int(value)
    if not 0 <= value <= 180:
        raise argparse.ArgumentTypeError('角度范围必须为 0..180')
    return value

def commands(pitch_start, pitch, pan_start, left):
    yield 5, 0, 0.2
    for code, start, end in ((18, pitch_start, pitch), (9, pan_start, left)):
        step = 2 if end >= start else -2
        for value in range(start, end, step):
            yield code, value, 0.02
        yield code, end, 0.5

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', default=os.environ.get('CHASSIS_PORT', DEFAULT_PORT))
    p.add_argument('--baudrate', type=int, default=115200)
    p.add_argument('--pitch-start', type=angle, default=0, help='当前俯仰角，默认常态 0')
    p.add_argument('--pitch', type=angle, default=90, help='降下后的识别姿态角度')
    p.add_argument('--pan-start', type=angle, default=105, help='当前水平角，默认回中 105')
    p.add_argument('--left-angle', type=angle, default=30, help='车体左侧角度，默认 0')
    p.add_argument('--dry-run', action='store_true', help='只打印指令，不打开串口')
    args = p.parse_args()
    sequence = list(commands(args.pitch_start, args.pitch, args.pan_start, args.left_angle))
    if args.dry_run:
        for code, value, delay in sequence:
            print(f'@|{code}|{value}|0|0#  wait={delay:.2f}s')
        return 0
    import serial
    print('请先退出 full_map_walk_test.py 和其他串口控制脚本。')
    print('假定当前姿态与 --pitch-start / --pan-start 一致；先停车，再移动云台。')
    with serial.Serial(args.port, args.baudrate, timeout=0.1,
                       write_timeout=1, exclusive=True) as link:
        def send(code, value):
            payload=f'@|{code}|{value}|0|0#'.encode('ascii')
            if link.write(payload) != len(payload):
                raise RuntimeError('串口指令未完整发送')
            link.flush()
        try:
            for code, value, delay in sequence:
                if code == 18:
                    # Discard the stop ACK; each pitch command must get its own ACK.
                    link.reset_input_buffer()
                send(code, value)
                if code == 18:
                    deadline=time.monotonic()+2
                    while time.monotonic()<deadline:
                        if link.readline().strip()==b'@':
                            break
                    else:
                        raise RuntimeError('18号俯仰指令无应答，已停止后续动作；检查下位机固件')
                time.sleep(delay)
            print(f'指令发送完成：俯仰 {args.pitch}°，水平 {args.left_angle}°；保持此姿态，不自动回位。')
            print('水平9号指令无到位反馈，请目视确认。')
        except KeyboardInterrupt:
            send(5,0)
            print('已中断，云台保持当前姿态。')
            return 130
        except Exception:
            send(5,0)
            raise
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
