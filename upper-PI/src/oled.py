import queue
import os
import time
import tkinter as tk
from multiprocessing import get_context


def _display_worker(command_queue, ready_event):
    """Run one persistent Tk window and apply display updates from the parent."""
    root = tk.Tk()
    fullscreen = os.environ.get("OLED_FULLSCREEN", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    root.title("Robot status")
    if fullscreen:
        root.attributes("-fullscreen", True)
    else:
        root.geometry("640x240")
    root.configure(bg="black")
    root.bind("<Escape>", lambda _event: root.withdraw())

    top_label = tk.Label(
        root, text="", font=("Arial", 50, "bold"), fg="white", bg="black"
    )
    bottom_label = tk.Label(
        root, text="", font=("Arial", 40), fg="yellow", bg="black"
    )
    top_label.pack(expand=True)
    hide_at = None

    def show_single(message, duration):
        nonlocal hide_at
        bottom_label.pack_forget()
        top_label.configure(text=message, font=("Arial", 50, "bold"), fg="white")
        if not top_label.winfo_manager():
            top_label.pack(expand=True)
        root.deiconify()
        root.lift()
        hide_at = time.monotonic() + max(0, duration) / 1000.0

    def show_dual(left_text, right_text, duration):
        nonlocal hide_at
        top_label.configure(
            text=f"左侧: {left_text}", font=("Arial", 40), fg="cyan"
        )
        bottom_label.configure(text=f"右侧: {right_text}")
        if not top_label.winfo_manager():
            top_label.pack(expand=True)
        if not bottom_label.winfo_manager():
            bottom_label.pack(expand=True)
        root.deiconify()
        root.lift()
        hide_at = time.monotonic() + max(0, duration) / 1000.0

    def poll_commands():
        nonlocal hide_at
        latest = None
        try:
            while True:
                latest = command_queue.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            command = latest[0]
            if command == "stop":
                root.destroy()
                return
            if command == "text":
                show_single(latest[1], latest[2])
            elif command == "dual":
                show_dual(latest[1], latest[2], latest[3])

        if hide_at is not None and time.monotonic() >= hide_at:
            root.withdraw()
            hide_at = None
        root.after(30, poll_commands)

    root.withdraw()
    ready_event.set()
    root.after(0, poll_commands)
    root.mainloop()


class OLEDDisplay:
    """Non-blocking display backed by one persistent UI process."""

    RESULT_MAP = {
        1: "发现一号人脸", 2: "发现二号人脸", 3: "发现三号人脸", 4: "发现四号人脸", 5: "发现五号人脸",
        6: "发现六号人脸", 7: "发现七号人脸", 8: "发现八号人脸", 9: "发现九号人脸", 10: "发现十号人物",
        11: "发现一号刀具", 12: "发现二号刀具", 13: "发现三号刀具", 14: "发现四号刀具", 15: "发现五号刀具",
        16: "发现六号刀具", 17: "发现七号刀具", 18: "发现八号刀具", 19: "发现九号刀具", 20: "发现十号刀具",
        21: "一号嫌疑人", 22: "二号嫌疑人", 23: "三号嫌疑人", 24: "四号嫌疑人", 25: "五号嫌疑人",
        26: "六号嫌疑人", 27: "七号嫌疑人", 28: "八号嫌疑人", 29: "九号嫌疑人", 30: "十号嫌疑人",
        31: "发现一号可疑刀具", 32: "发现二号可疑刀具", 33: "发现三号可疑刀具", 34: "发现四号可疑刀具", 35: "发现五号可疑刀具",
        36: "发现六号可疑刀具", 37: "发现七号可疑刀具", 38: "发现八号可疑刀具", 39: "发现九号可疑刀具", 40: "发现十号可疑刀具",
    }

    def __init__(self):
        # FullMap starts camera threads before creating this object. Use spawn so
        # the UI process never inherits OpenCV/RKNN locks from the parent.
        context = get_context("spawn")
        self._command_queue = context.Queue(maxsize=8)
        self._ready_event = context.Event()
        self.current_process = context.Process(
            target=_display_worker,
            args=(self._command_queue, self._ready_event),
            name="oled-display",
            daemon=True,
        )
        self.current_process.start()
        if not self._ready_event.wait(timeout=3.0) or not self.current_process.is_alive():
            self.close()
            raise RuntimeError("OLED显示进程启动失败")
        print("[OLED] 常驻显示进程初始化完成.")

    def _send(self, command):
        if not self.is_ready():
            raise RuntimeError("OLED显示进程不可用")
        try:
            self._command_queue.put_nowait(command)
        except queue.Full:
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                pass
            self._command_queue.put_nowait(command)

    def is_ready(self):
        return (
            self.current_process is not None
            and self.current_process.is_alive()
            and self._ready_event.is_set()
        )

    def show_text(self, message, duration=2000):
        self._send(("text", str(message), int(duration)))

    def show_detect_result(self, left_res, right_res, duration=3000):
        def result_text(result):
            if result == -1:
                return "未检测到目标"
            return self.RESULT_MAP.get(int(result), f"未知目标({result})")

        self._send(
            ("dual", result_text(left_res), result_text(right_res), int(duration))
        )

    def close(self):
        process = getattr(self, "current_process", None)
        if process is None:
            return
        if process.is_alive():
            try:
                self._command_queue.put_nowait(("stop",))
            except queue.Full:
                pass
            process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        self.current_process = None
