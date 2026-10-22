"""白色环境中的红色/黄色/蓝色障碍物检测测试模块。

核心接口：
    value = detect_obstacle(frame)

返回值：
    1: 连续两帧检测到红色、黄色或蓝色障碍物
    0: 未检测到障碍物，或尚未连续检测到两帧

同一路视频流应复用检测器，每个新帧仅调用一次；未检出或读取失败时清零。
--image 只检测一帧，候选障碍物不会直接确认为 1。

直接运行：
    python3 obstacle_color_test.py --camera 0
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np


NO_OBSTACLE = 0
OBSTACLE_DETECTED = 1


@dataclass(frozen=True)
class DetectionResult:
    """当前帧的测量值及连续两帧确认结果。"""

    value: int
    obstacle: bool
    colors: tuple[str, ...]
    red_area: float
    yellow_area: float
    combined_area: float
    colored_pixels: int
    area_threshold: float
    roi: tuple[int, int, int, int]
    blue_area: float = 0.0
    gray_area: float = 0.0  # 兼容已有调用方；灰色识别已取消，始终为 0。
    raw_obstacle: bool = False
    consecutive_hits: int = 0


class ColorObstacleDetector:
    """检测白色环境中的红黄蓝障碍物，连续两帧命中才确认。"""

    CONFIRM_FRAMES = 2

    # 白色和灰色的饱和度很低。S>=35 可以排除背景，同时保留阴影中
    # 的暗红、暗黄。红黄范围有意覆盖橙色过渡区。w
    RED_RANGES = (
        (np.array((0, 100, 15), dtype=np.uint8),
         np.array((22, 255, 255), dtype=np.uint8)),
        (np.array((145, 100, 15), dtype=np.uint8),
         np.array((180, 255, 255), dtype=np.uint8)),
    )
    YELLOW_RANGES = (
        (np.array((12, 100, 40), dtype=np.uint8),
         np.array((58, 255, 255), dtype=np.uint8)),
    )

    # OpenCV HSV: H=0..179，蓝色初始范围，可按现场光照调整。
    BLUE_RANGES = (
        (np.array((95, 100, 100), dtype=np.uint8),
         np.array((130, 255, 255), dtype=np.uint8)),
    )
    def __init__(
        self,
        min_area: float = 500.0,
        min_area_ratio: float = 0.01,
        roi_top_ratio: float = 0.1,
        kernel_size: int = 3,
        bridge_kernel_size: int = 13,
        roi_left_ratio: float = 0.45,
        roi_right_ratio: float = 0.55,
        roi_bottom_ratio: float = 1.0,
    ) -> None:
        """
        Args:
            min_area: 合并后锥桶轮廓的最小像素面积。
            min_area_ratio: 最小面积占检测区域面积的比例。
            roi_top_ratio: ROI 上边界比例。
            kernel_size: 单色掩膜去噪核大小，必须为正奇数。
            bridge_kernel_size: 连接红黄条纹的闭运算核大小。
            roi_left_ratio: ROI 左边界比例。
            roi_right_ratio: ROI 右边界比例。
            roi_bottom_ratio: ROI 下边界比例。
        """
        if min_area < 0:
            raise ValueError("min_area 不能小于 0")
        if not 0.0 <= min_area_ratio <= 1.0:
            raise ValueError("min_area_ratio 必须在 0～1 之间")
        if not 0.0 <= roi_left_ratio < roi_right_ratio <= 1.0:
            raise ValueError("ROI 必须满足 0 <= left < right <= 1")
        if not 0.0 <= roi_top_ratio < roi_bottom_ratio <= 1.0:
            raise ValueError("ROI 必须满足 0 <= top < bottom <= 1")
        for name, value in (
            ("kernel_size", kernel_size),
            ("bridge_kernel_size", bridge_kernel_size),
        ):
            if value <= 0 or value % 2 == 0:
                raise ValueError(f"{name} 必须为正奇数")

        self.min_area = float(min_area)
        self._consecutive_hits = 0
        self.min_area_ratio = float(min_area_ratio)
        self.roi_left_ratio = float(roi_left_ratio)
        self.roi_top_ratio = float(roi_top_ratio)
        self.roi_right_ratio = float(roi_right_ratio)
        self.roi_bottom_ratio = float(roi_bottom_ratio)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        self.bridge_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (bridge_kernel_size, bridge_kernel_size)
        )

    def _roi_bounds(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        """将比例 ROI 转为 (x1, y1, x2, y2) 像素坐标。"""
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, int(width * self.roi_left_ratio)))
        y1 = max(0, min(height - 1, int(height * self.roi_top_ratio)))
        x2 = max(x1 + 1, min(width, int(width * self.roi_right_ratio)))
        y2 = max(y1 + 1, min(height, int(height * self.roi_bottom_ratio)))
        return x1, y1, x2, y2

    @staticmethod
    def _build_hsv_mask(hsv: np.ndarray, ranges) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        return mask

    @staticmethod
    def _dark_red_dominance_mask(bgr: np.ndarray) -> np.ndarray:
        """补充检测阴影中接近黑色、但红通道仍占优势的区域。"""
        blue, green, red = cv2.split(bgr)
        blue = blue.astype(np.int16)
        green = green.astype(np.int16)
        red = red.astype(np.int16)
        dominant = (
            (red >= 18)
            & ((red - green) >= 7)
            & ((red - blue) >= 7)
            & (red * 100 >= green * 115)
            & (red * 100 >= blue * 115)
        )
        return dominant.astype(np.uint8) * 255

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

    @staticmethod
    def _largest_contour(mask: np.ndarray):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, 0.0
        contour = max(contours, key=cv2.contourArea)
        return contour, float(cv2.contourArea(contour))

    def _make_masks(self, roi: np.ndarray):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_hsv = self._build_hsv_mask(hsv, self.RED_RANGES)
        red_shadow = self._dark_red_dominance_mask(roi)
        red_mask = self._clean_mask(cv2.bitwise_or(red_hsv, red_shadow))
        yellow_mask = self._clean_mask(
            self._build_hsv_mask(hsv, self.YELLOW_RANGES)
        )

        blue_mask = self._clean_mask(
            self._build_hsv_mask(hsv, self.BLUE_RANGES)
        )
        colored_mask = cv2.bitwise_or(
            cv2.bitwise_or(red_mask, yellow_mask), blue_mask
        )
        # 三角锥桶常有分离的红黄横条。较大的闭运算核把相邻横条连接成
        # 一个锥桶轮廓，但保留 colored_mask 统计真实有色像素，防止空洞
        # 或零散噪声被误判成大障碍物。
        combined_mask = cv2.morphologyEx(
            colored_mask, cv2.MORPH_CLOSE, self.bridge_kernel
        )
        return red_mask, yellow_mask, blue_mask, colored_mask, combined_mask

    def reset(self) -> None:
        """视频流中断或切换时清除连续检测计数。"""
        self._consecutive_hits = 0

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """每个新帧调用一次；连续两帧命中才返回 1，未命中立即清零。"""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            self.reset()
            raise ValueError("frame 必须是有效的 BGR 图像")
        if frame.ndim != 3 or frame.shape[2] != 3:
            self.reset()
            raise ValueError("frame 必须是三通道 BGR 图像")

        roi_x1, roi_y1, roi_x2, roi_y2 = self._roi_bounds(frame)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        red_mask, yellow_mask, blue_mask, colored_mask, combined_mask = self._make_masks(roi)

        _, red_area = self._largest_contour(red_mask)
        _, yellow_area = self._largest_contour(yellow_mask)
        _, blue_area = self._largest_contour(blue_mask)
        combined_contour, combined_area = self._largest_contour(combined_mask)

        area_threshold = max(
            self.min_area,
            float(roi.shape[1] * roi.shape[0]) * self.min_area_ratio,
        )

        colored_pixels = 0
        if combined_contour is not None:
            component = np.zeros_like(combined_mask)
            cv2.drawContours(component, [combined_contour], -1, 255, -1)
            colored_pixels = int(
                cv2.countNonZero(cv2.bitwise_and(colored_mask, component))
            )

        # 合并轮廓需达到面积门槛，且其中至少有一定比例是真实目标颜色像素。
        # 这样既能识别条纹三角锥桶，也不会把几个零散色点桥接成障碍物。
        min_colored_pixels = max(25, int(area_threshold * 0.30))
        raw_obstacle = (
            combined_area >= area_threshold
            and colored_pixels >= min_colored_pixels
        )
        self._consecutive_hits = (
            min(self._consecutive_hits + 1, self.CONFIRM_FRAMES)
            if raw_obstacle else 0
        )
        obstacle = self._consecutive_hits >= self.CONFIRM_FRAMES

        colors = []
        color_area_threshold = max(12.0, area_threshold * 0.12)
        if obstacle and red_area >= color_area_threshold:
            colors.append("red")
        if obstacle and yellow_area >= color_area_threshold:
            colors.append("yellow")
        if obstacle and blue_area >= color_area_threshold:
            colors.append("blue")

        return DetectionResult(
            value=OBSTACLE_DETECTED if obstacle else NO_OBSTACLE,
            obstacle=obstacle,
            raw_obstacle=raw_obstacle,
            consecutive_hits=self._consecutive_hits,
            colors=tuple(colors),
            red_area=red_area,
            yellow_area=yellow_area,
            blue_area=blue_area,
            combined_area=combined_area,
            colored_pixels=colored_pixels,
            area_threshold=area_threshold,
            roi=(roi_x1, roi_y1, roi_x2, roi_y2),
        )

    def draw_result(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """在图像副本上绘制锥桶轮廓和检测状态。"""
        output = frame.copy()
        roi_x1, roi_y1, roi_x2, roi_y2 = self._roi_bounds(output)
        roi = output[roi_y1:roi_y2, roi_x1:roi_x2]
        red_mask, yellow_mask, blue_mask, _, combined_mask = self._make_masks(roi)

        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            if cv2.contourArea(contour) < result.area_threshold:
                continue
            shifted = contour.copy()
            shifted[:, 0, 0] += roi_x1
            shifted[:, 0, 1] += roi_y1
            cv2.drawContours(output, [shifted], -1, (255, 0, 255), 2)
            x, y, w, h = cv2.boundingRect(shifted)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 255), 1)

        label_threshold = max(12.0, result.area_threshold * 0.12)
        for name, mask, color in (
            ("red", red_mask, (0, 0, 255)),
            ("yellow", yellow_mask, (0, 255, 255)),
            ("blue", blue_mask, (255, 0, 0)),
        ):
            contour, area = self._largest_contour(mask)
            if contour is None or area < label_threshold:
                continue
            x, y, _, _ = cv2.boundingRect(contour)
            cv2.putText(output, name,
                        (x + roi_x1, max(18, y + roi_y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

        cv2.rectangle(output, (roi_x1, roi_y1),
                      (roi_x2 - 1, roi_y2 - 1), (255, 255, 0), 2)
        cv2.putText(output, "ROI", (roi_x1 + 5, roi_y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2)

        status = "OBSTACLE=1" if result.obstacle else "CLEAR=0"
        status_color = (0, 0, 255) if result.obstacle else (0, 255, 0)
        if result.raw_obstacle and not result.obstacle:
            status = f"PENDING={result.consecutive_hits}/{self.CONFIRM_FRAMES} VALUE=0"
            status_color = (0, 165, 255)
        colors = ",".join(result.colors) if result.colors else "none"
        cv2.putText(output, f"{status} colors={colors}", (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2)
        cv2.putText(
            output,
            f"red={result.red_area:.0f} yellow={result.yellow_area:.0f} blue={result.blue_area:.0f} "
            f"combined={result.combined_area:.0f} pixels={result.colored_pixels}",
            (8, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1,
        )
        return output


_default_detector = ColorObstacleDetector()


def detect_obstacle(frame: np.ndarray) -> int:
    """同一路视频流逐帧调用：连续两帧检测到障碍物返回 1，否则返回 0。"""
    return _default_detector.detect(frame).value


def parse_flip(value: str):
    value = str(value).strip().lower()
    if value in ("none", "off", "no", "raw"):
        return None
    parsed = int(value)
    if parsed not in (-1, 0, 1):
        raise argparse.ArgumentTypeError("--flip 只能是 none、-1、0 或 1")
    return parsed


def _open_camera(source, width: int, height: int, fps: int):
    camera_source = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(camera_source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {camera_source}")
    return cap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="0", help="摄像头编号或设备路径")
    parser.add_argument("--image", help="检测单张图片，不打开摄像头")
    parser.add_argument("--width", type=int, default=4800)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip", type=parse_flip, default=-1)
    parser.add_argument("--min-area", type=float, default=200.0,
                        help="锥桶合并轮廓最小面积")
    parser.add_argument("--min-area-ratio", type=float, default=0.0004)
    parser.add_argument("--roi-left", type=float, default=0.2,
                        help="ROI 左边界比例，默认 0.20")
    parser.add_argument("--roi-top", type=float, default=0.25,
                        help="ROI 上边界比例，默认 0.25")
    parser.add_argument("--roi-right", type=float, default=0.8,
                        help="ROI 右边界比例，默认 0.80")
    parser.add_argument("--roi-bottom", type=float, default=1.0,
                        help="ROI 下边界比例，默认 1.0")
    parser.add_argument("--no-display", action="store_true",
                        help="无窗口运行，仅在检测值变化时打印")
    args = parser.parse_args()

    detector = ColorObstacleDetector(
        min_area=args.min_area,
        min_area_ratio=args.min_area_ratio,
        roi_left_ratio=args.roi_left,
        roi_top_ratio=args.roi_top,
        roi_right_ratio=args.roi_right,
        roi_bottom_ratio=args.roi_bottom,
    )

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise RuntimeError(f"无法读取图片: {args.image}")
        result = detector.detect(frame)
        print(
            f"OBSTACLE_VALUE={result.value} RAW_OBSTACLE={int(result.raw_obstacle)} "
            f"HITS={result.consecutive_hits}/{detector.CONFIRM_FRAMES} "
            f"COLORS={','.join(result.colors) if result.colors else 'none'} "
            f"RED_AREA={result.red_area:.0f} "
            f"YELLOW_AREA={result.yellow_area:.0f} BLUE_AREA={result.blue_area:.0f} "
            f"COMBINED_AREA={result.combined_area:.0f}"
        )
        if not args.no_display:
            cv2.imshow("Color Obstacle Detection", detector.draw_result(frame, result))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    cap = _open_camera(args.camera, args.width, args.height, args.fps)
    print("[INFO] 连续两帧检测到障碍物返回 1；未检出一帧立即返回 0 并清零")
    print("[INFO] 按 q 或 Esc 退出")
    last_value = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                detector.reset()
                time.sleep(0.03)
                continue
            if args.flip is not None:
                frame = cv2.flip(frame, args.flip)

            result = detector.detect(frame)
            if result.value != last_value:
                print(
                    f"OBSTACLE_VALUE={result.value} RAW_OBSTACLE={int(result.raw_obstacle)} "
                    f"HITS={result.consecutive_hits}/{detector.CONFIRM_FRAMES} "
                    f"COLORS={','.join(result.colors) if result.colors else 'none'} "
                    f"RED_AREA={result.red_area:.0f} "
                    f"YELLOW_AREA={result.yellow_area:.0f} BLUE_AREA={result.blue_area:.0f} "
                    f"COMBINED_AREA={result.combined_area:.0f}"
                )
                last_value = result.value

            if not args.no_display:
                cv2.imshow(
                    "Color Obstacle Detection",
                    detector.draw_result(frame, result),
                )
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
