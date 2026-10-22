#!/usr/bin/env python3
"""从 N12 单独测试返回出发区：N12 -> J_TOP -> START_AREA。"""
import argparse
import time
from Serial import ChassisController

class DryChassis:
    def distance(self, val, dis): print(f"[DRY] distance speed={val} distance={dis}")
    def turn_pure(self, angle): print(f"[DRY] turn_pure angle={angle}")
    def stop(self): print("[DRY] stop")

def wait_ack(chassis, timeout):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        msg=chassis.read_message()
        if msg:
            msg=msg.strip()
            print(f"[ACK RX] {msg}")
            if msg=="@": return True
            if msg.startswith("ERR:"): return False
        time.sleep(0.01)
    print("[TIMEOUT] 等待底盘完成")
    return False

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A5069RR4-if00-port0")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--approach-speed", type=int, default=9)
    p.add_argument("--approach-distance", type=int, default=142)
    p.add_argument("--turn-angle", type=int, default=75)
    p.add_argument("--return-speed", type=int, default=9)
    p.add_argument("--return-distance", type=int, default=95)
    p.add_argument("--timeout", type=float, default=8)
    p.add_argument("--dry-run", action="store_true")
    args=p.parse_args()
    chassis=DryChassis() if args.dry_run else ChassisController(args.port,args.baudrate)
    if not args.dry_run and getattr(chassis,"serial",None) is None:
        return 2
    try:
        print("[RETURN TEST] 当前假定位置=N12，路线=N12 -> J_TOP -> START_AREA")
        print(f"[RETURN TEST] 先前进 speed={args.approach_speed} distance={args.approach_distance}")
        chassis.distance(args.approach_speed,args.approach_distance)
        if not args.dry_run and not wait_ack(chassis,args.timeout): return 3
        time.sleep(0.2)
        print(f"[RETURN TEST] J_TOP右转 angle={args.turn_angle}")
        chassis.turn_pure(args.turn_angle)
        if not args.dry_run and not wait_ack(chassis,args.timeout): return 4
        time.sleep(0.2)
        print(f"[RETURN TEST] 再前进 speed={args.return_speed} distance={args.return_distance}")
        chassis.distance(args.return_speed,args.return_distance)
        if not args.dry_run and not wait_ack(chassis,args.timeout): return 5
        print("[RETURN TEST] 返回动作完成")
        return 0
    except KeyboardInterrupt:
        print("[RETURN TEST] 用户中止，立即停车")
        return 130
    finally:
        if not args.dry_run:
            chassis.stop()
            if getattr(chassis,"serial",None) is not None:
                chassis.serial.close()

if __name__=="__main__":
    raise SystemExit(main())
