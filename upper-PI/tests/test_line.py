"""Standalone line-following smoothness comparison and real-vehicle test script (test_line.py).

Modes:
1. Offline image comparison (no RKNN hardware required):
   python3 test_line.py --images not_at_junction.jpg crossroad.jpg left_junction.jpg right_junction.jpg
2. Real-time camera RKNN comparison with live waveform oscilloscope (no chassis commands):
   python3 test_line.py --camera 2 --flip -1
3. Camera + chassis closed-loop line-following test:
   python3 test_line.py --camera 2 --flip -1 --drive --speed 9
"""

from __future__ import annotations

import argparse
from collections import deque
import os
import time
from pathlib import Path
from typing import cast

import cv2
import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError:
    RKNNLite = None

try:
    from Serial import ChassisController
except ImportError:
    ChassisController = None


DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "pp544288.rknn"
)


class SmoothLineTester:
    """Run original global-moment and optimized scanline-envelope algorithms side by side for A/B testing."""

    def __init__(
        self,
        model_path: str | os.PathLike | None = None,
        input_width: int = 544,
        input_height: int = 288,
        threshold: float = 0.6,
        road_class: int = 1,
        load_rknn: bool = True,
        scan_y_start: float = 0.50,
        scan_y_end: float = 0.95,
        num_scan_lines: int = 12,
        max_span_ratio: float = 0.78,
        smooth_alpha: float = 0.28,
        deadband_px: int = 3,
        kp: float = 0.45,
        kd: float = 0.10,
        max_angle: float = 30.0,
    ):
        self.roi_top_limit = 0.20
        self.roi_left_limit = 0.0
        self.roi_right_limit = 1.0

        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.threshold = float(threshold)
        self.road_class = int(road_class)

        self.scan_y_start = float(scan_y_start)
        self.scan_y_end = float(scan_y_end)
        self.num_scan_lines = int(num_scan_lines)
        self.max_span_ratio = float(max_span_ratio)
        self.smooth_alpha = float(smooth_alpha)
        self.deadband_px = int(deadband_px)
        self.kp = float(kp)
        self.kd = float(kd)
        self.max_angle = float(max_angle)

        self.min_component_area = 300
        self.min_bottom_support_ratio = 0.02
        self.junction_top_ratio = 0.68
        self.junction_bottom_ratio = 0.73
        self.cross_width_ratio = 0.85
        self.l_cross_width_ratio = float(os.environ.get("L_CROSS_WIDTH_RATIO", "0.72"))
        self.junction_mode = "NORMAL"
        self.cross_counter = 0
        self.cross_confirm_threshold = 2

        self.orig_last_cx: int | None = None
        self.orig_last_offset: int = 0

        self.opt_last_cx: float | None = None
        self.opt_last_eff_offset: float = 0.0
        self.opt_smooth_d: float = 0.0
        self.opt_smooth_steer: float = 0.0

        self.history_orig = deque(maxlen=120)
        self.history_opt = deque(maxlen=120)

        self.last_inference_ms = 0.0
        self.rknn = None
        if load_rknn:
            configured_model = model_path or os.environ.get("DEEPLAB_MODEL") or DEFAULT_MODEL
            self.model_path = Path(configured_model).expanduser().resolve()
            self.rknn = self._load_runtime(self.model_path)

    @staticmethod
    def _load_runtime(model_path: Path):
        if RKNNLite is None:
            raise RuntimeError("rknnlite is not installed; use --images mode for offline image testing")
        if not model_path.is_file():
            raise FileNotFoundError(f"RKNN model file not found: {model_path}")
        rknn = cast(RKNNLite, RKNNLite(verbose=False))
        if rknn.load_rknn(str(model_path)) != 0:
            rknn.release()
            raise RuntimeError(f"Failed to load RKNN model: {model_path}")
        core_mask = getattr(RKNNLite, "NPU_CORE_0_1_2", RKNNLite.NPU_CORE_AUTO)
        if rknn.init_runtime(core_mask=core_mask) != 0:
            rknn.release()
            raise RuntimeError("Failed to initialize RKNN runtime")
        return rknn

    def reset_states(self) -> None:
        self.orig_last_cx = None
        self.orig_last_offset = 0
        self.opt_last_cx = None
        self.opt_last_eff_offset = 0.0
        self.opt_smooth_d = 0.0
        self.opt_smooth_steer = 0.0
        self.cross_counter = 0

    def _prepare_input(self, frame_bgr):
        resized = cv2.resize(
            frame_bgr,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb[np.newaxis, ...], dtype=np.uint8)

    def _decode_output(self, output):
        array = np.asarray(output)
        if array.ndim == 4:
            array = array[0]
        if array.ndim == 2:
            return array.astype(np.uint8)
        if array.shape[0] <= 64:
            logits = array.astype(np.float32, copy=False)
            if logits.shape[0] == 2:
                delta = np.clip(logits[1] - logits[0], -30.0, 30.0)
                prob = 1.0 / (1.0 + np.exp(-delta))
                return (prob >= self.threshold).astype(np.uint8)
            return np.argmax(logits, axis=0).astype(np.uint8)
        if array.shape[-1] <= 64:
            logits = array.astype(np.float32, copy=False)
            if logits.shape[-1] == 2:
                delta = np.clip(logits[..., 1] - logits[..., 0], -30.0, 30.0)
                prob = 1.0 / (1.0 + np.exp(-delta))
                return (prob >= self.threshold).astype(np.uint8)
            return np.argmax(logits, axis=-1).astype(np.uint8)
        raise RuntimeError(f"Unrecognized output dimension: {array.shape}")

    def infer_roi_mask(self, frame_bgr) -> np.ndarray:
        """Run RKNN inference to obtain a binary road mask for the ROI region."""
        height, width = frame_bgr.shape[:2]
        roi_start_row = int(height * self.roi_top_limit)
        started = time.perf_counter()
        model_input = self._prepare_input(frame_bgr)
        outputs = self.rknn.inference(inputs=[model_input], data_format=["nhwc"])
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0

        class_mask = self._decode_output(outputs[0])
        class_mask = cv2.resize(class_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        roi_classes = class_mask[roi_start_row:height, :]
        return np.where(roi_classes == self.road_class, 255, 0).astype(np.uint8)

    def extract_green_mask_from_debug_image(self, frame_bgr) -> np.ndarray:
        """Extract a binary ROI mask from an image that already has a green mask rendered (offline test)."""
        height = frame_bgr.shape[0]
        roi_start_row = int(height * self.roi_top_limit)
        roi = frame_bgr[roi_start_row:height, :]
        b, g, r = cv2.split(roi.astype(np.int16))
        is_green = ((g - np.maximum(b, r)) > 30) & (g > 95)
        return np.where(is_green, 255, 0).astype(np.uint8)

    def _select_near_road_component(self, road_mask, ref_cx: float | None):
        roi_h, roi_w = road_mask.shape
        count, labels, stats, _ = cv2.connectedComponentsWithStats(road_mask, connectivity=8)
        if count <= 1:
            return None

        bottom_rows = max(4, int(roi_h * 0.10))
        min_bottom_support = max(8, int(roi_w * bottom_rows * self.min_bottom_support_ratio))
        best_label = None
        best_score = -1.0

        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_area:
                continue
            component_bottom = labels[roi_h - bottom_rows :, :] == label
            bottom_support = int(np.count_nonzero(component_bottom))
            if bottom_support < min_bottom_support:
                continue
            component_x = float(stats[label, cv2.CC_STAT_LEFT]) + float(stats[label, cv2.CC_STAT_WIDTH]) * 0.5
            if ref_cx is not None:
                distance_penalty = abs(component_x - ref_cx) * 2.0
            else:
                distance_penalty = abs(component_x - roi_w * 0.5) * 0.5
            score = area + bottom_support * 4.0 - distance_penalty
            if score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            return None
        return np.where(labels == best_label, 255, 0).astype(np.uint8)

    def _is_crossroad(self, component_mask) -> bool:
        roi_h, roi_w = component_mask.shape
        scan_y_start = int(roi_h * self.junction_top_ratio)
        scan_y_end = int(roi_h * self.junction_bottom_ratio)
        width_ratio = self.l_cross_width_ratio if self.junction_mode == "L" else self.cross_width_ratio
        min_wide_rows = 1 if self.junction_mode == "L" else 3

        scan_rows = list(range(scan_y_start, scan_y_end, 3))
        if len(scan_rows) < min_wide_rows:
            scan_rows = np.linspace(scan_y_start, max(scan_y_start, scan_y_end - 1), min_wide_rows, dtype=int)

        wide_rows = 0
        for y in scan_rows:
            is_road = (component_mask[y, :] > 0).astype(np.int8)
            diffs = np.diff(np.r_[0, is_road, 0])
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            if len(starts) and np.max(ends - starts) > int(roi_w * width_ratio):
                wide_rows += 1
        return wide_rows >= min_wide_rows

    def _compute_original(self, raw_roi_mask: np.ndarray, center_x: int):
        """Reproduce the original line_detect.py + full_map_walk_test.py computation exactly."""
        k3 = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(raw_roi_mask, cv2.MORPH_OPEN, k3, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k3, iterations=2)
        comp = self._select_near_road_component(m, self.orig_last_cx)
        if comp is None:
            return center_x, int(raw_roi_mask.shape[0] * 0.7), self.orig_last_offset, 0.0

        roi_h, roi_w = comp.shape
        near_mask = comp.copy()
        near_mask[: int(roi_h * 0.30), :] = 0
        moments = cv2.moments(near_mask, binaryImage=True)
        if moments["m00"] <= 0:
            moments = cv2.moments(comp, binaryImage=True)
        if moments["m00"] <= 0:
            return center_x, int(roi_h * 0.7), self.orig_last_offset, 0.0

        measured_cx = float(moments["m10"] / moments["m00"])
        measured_cy = float(moments["m01"] / moments["m00"])

        if self.orig_last_cx is not None:
            max_jump = max(8, int(roi_w * 0.08))
            measured_cx = float(np.clip(measured_cx, self.orig_last_cx - max_jump, self.orig_last_cx + max_jump))
            measured_cx = 0.65 * self.orig_last_cx + 0.35 * measured_cx

        cx_roi = int(round(measured_cx))
        cy_roi = int(round(measured_cy))
        self.orig_last_cx = cx_roi
        offset = cx_roi - center_x

        steer = self.kp * offset + self.kd * (offset - self.orig_last_offset)
        self.orig_last_offset = offset
        steer = float(np.clip(steer, -self.max_angle, self.max_angle))
        return cx_roi, cy_roi, offset, steer

    def _compute_optimized(self, raw_roi_mask: np.ndarray, center_x: int):
        """Optimized algorithm: 7x7 ellipse close + near-field scanline envelope midpoint + wide-row rejection + float filtering + deadband PD."""
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        m = cv2.morphologyEx(raw_roi_mask, cv2.MORPH_OPEN, k7, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k7, iterations=2)
        comp = self._select_near_road_component(m, self.opt_last_cx)
        if comp is None:
            return center_x, int(raw_roi_mask.shape[0] * 0.75), int(round(self.opt_last_eff_offset)), self.opt_smooth_steer, [], False

        roi_h, roi_w = comp.shape
        y_start = int(roi_h * self.scan_y_start)
        y_end = int(roi_h * self.scan_y_end)
        scan_ys = np.linspace(y_start, y_end - 1, self.num_scan_lines, dtype=int)

        row_centers = []
        row_weights = []
        valid_ys = []
        scan_debug_info = []

        for y in scan_ys:
            xs = np.where(comp[y, :] > 0)[0]
            if xs.size < max(8, int(roi_w * 0.08)):
                continue
            left_edge = float(xs[0])
            right_edge = float(xs[-1])
            span = right_edge - left_edge
            center_pt = 0.5 * (left_edge + right_edge)

            rejected = (span > roi_w * self.max_span_ratio) and (len(row_centers) >= 3)
            scan_debug_info.append((int(y), int(left_edge), int(right_edge), int(round(center_pt)), rejected))

            if not rejected:
                weight = 1.0 + 2.0 * ((y - y_start) / max(1, (y_end - y_start)))
                row_centers.append(center_pt)
                row_weights.append(weight)
                valid_ys.append(float(y))

        if row_centers:
            weights_arr = np.asarray(row_weights, dtype=np.float32)
            centers_arr = np.asarray(row_centers, dtype=np.float32)
            if self.opt_last_cx is not None:
                diffs = np.abs(centers_arr - self.opt_last_cx)
                weights_arr *= np.clip(1.0 - diffs / (roi_w * 0.25), 0.2, 1.0)
            measured_cx = float(np.sum(centers_arr * weights_arr) / np.sum(weights_arr))
            measured_cy = float(np.mean(valid_ys))
        else:
            near_mask = comp.copy()
            near_mask[: int(roi_h * 0.45), :] = 0
            moments = cv2.moments(near_mask, binaryImage=True)
            if moments["m00"] > 0:
                measured_cx = float(moments["m10"] / moments["m00"])
                measured_cy = float(moments["m01"] / moments["m00"])
            else:
                measured_cx = float(center_x)
                measured_cy = float(roi_h * 0.75)

        is_cross = self._is_crossroad(comp)
        if is_cross:
            self.cross_counter += 1
        else:
            self.cross_counter = 0

        if self.opt_last_cx is not None:
            jump_ratio = 0.015 if (is_cross or self.cross_counter > 0) else 0.05
            max_jump = max(3.0, roi_w * jump_ratio)
            measured_cx = float(np.clip(measured_cx, self.opt_last_cx - max_jump, self.opt_last_cx + max_jump))
            measured_cx = (1.0 - self.smooth_alpha) * self.opt_last_cx + self.smooth_alpha * measured_cx

        self.opt_last_cx = measured_cx
        cx_roi = int(round(measured_cx))
        cy_roi = int(round(measured_cy))
        raw_offset = cx_roi - center_x

        if abs(raw_offset) <= self.deadband_px:
            eff_offset = 0.0
        else:
            eff_offset = float(raw_offset - self.deadband_px * (1 if raw_offset > 0 else -1))

        raw_d = eff_offset - self.opt_last_eff_offset
        self.opt_last_eff_offset = eff_offset
        self.opt_smooth_d = 0.6 * self.opt_smooth_d + 0.4 * raw_d

        target_steer = self.kp * eff_offset + self.kd * self.opt_smooth_d
        target_steer = float(np.clip(target_steer, -self.max_angle, self.max_angle))
        max_step = 6.0
        steer = self.opt_smooth_steer + float(np.clip(target_steer - self.opt_smooth_steer, -max_step, max_step))
        self.opt_smooth_steer = steer

        return cx_roi, cy_roi, raw_offset, steer, scan_debug_info, (self.cross_counter >= self.cross_confirm_threshold)

    def evaluate_frame(self, frame_bgr: np.ndarray, from_debug_image: bool = False):
        """Run original and optimized algorithms on the input frame and return a composite visualization and metrics dict."""
        height, width = frame_bgr.shape[:2]
        center_x = width // 2
        roi_start_row = int(height * self.roi_top_limit)

        if from_debug_image:
            raw_roi_mask = self.extract_green_mask_from_debug_image(frame_bgr)
        else:
            raw_roi_mask = self.infer_roi_mask(frame_bgr)

        orig_cx, orig_cy, orig_off, orig_steer = self._compute_original(raw_roi_mask, center_x)
        opt_cx, opt_cy, opt_off, opt_steer, scan_rows, is_cross = self._compute_optimized(raw_roi_mask, center_x)

        self.history_orig.append(orig_off)
        self.history_opt.append(opt_off)

        vis = self._draw_comparison_hud(
            frame_bgr,
            raw_roi_mask,
            roi_start_row,
            center_x,
            (orig_cx, orig_cy, orig_off, orig_steer),
            (opt_cx, opt_cy, opt_off, opt_steer),
            scan_rows,
            is_cross,
            from_debug_image,
        )
        return {
            "orig_offset": orig_off,
            "orig_steer": orig_steer,
            "opt_offset": opt_off,
            "opt_steer": opt_steer,
            "status": "CROSS" if is_cross else "LINE",
            "vis": vis,
        }

    def _draw_comparison_hud(
        self,
        frame,
        raw_roi_mask,
        roi_start_row,
        center_x,
        orig_data,
        opt_data,
        scan_rows,
        is_cross,
        from_debug_image,
    ) -> np.ndarray:
        vis = frame.copy()
        height, width = vis.shape[:2]

        if not from_debug_image:
            overlay = vis.copy()
            roi_view = overlay[roi_start_row:height, :]
            roi_view[raw_roi_mask > 0] = (0, 255, 0)
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0.0, dst=vis)

        cv2.rectangle(vis, (0, roi_start_row), (width - 1, height - 1), (0, 0, 255), 2)
        j_top = roi_start_row + int(raw_roi_mask.shape[0] * self.junction_top_ratio)
        j_bot = roi_start_row + int(raw_roi_mask.shape[0] * self.junction_bottom_ratio)
        cv2.line(vis, (0, j_top), (width - 1, j_top), (0, 165, 255), 1)
        cv2.line(vis, (0, j_bot), (width - 1, j_bot), (0, 165, 255), 1)
        cv2.line(vis, (center_x, roi_start_row), (center_x, height - 1), (0, 255, 255), 1)

        for y, left_x, right_x, mid_x, rejected in scan_rows:
            gy = roi_start_row + y
            if rejected:
                cv2.line(vis, (left_x, gy), (right_x, gy), (0, 140, 255), 1)
            else:
                cv2.line(vis, (left_x, gy), (right_x, gy), (255, 200, 0), 1)
                cv2.circle(vis, (left_x, gy), 3, (255, 120, 0), -1)
                cv2.circle(vis, (right_x, gy), 3, (255, 120, 0), -1)
                cv2.circle(vis, (mid_x, gy), 2, (255, 255, 0), -1)

        orig_cx, orig_cy, orig_off, orig_steer = orig_data
        opt_cx, opt_cy, opt_off, opt_steer = opt_data

        orig_gy = roi_start_row + orig_cy
        opt_gy = roi_start_row + opt_cy

        cv2.circle(vis, (orig_cx, orig_gy), 8, (0, 0, 255), -1)
        cv2.circle(vis, (orig_cx, orig_gy), 9, (255, 255, 255), 1)

        cv2.circle(vis, (opt_cx, opt_gy), 8, (255, 150, 0), -1)
        cv2.circle(vis, (opt_cx, opt_gy), 9, (255, 255, 255), 2)

        cv2.rectangle(vis, (0, 0), (width, 82), (20, 20, 20), -1)
        cv2.putText(
            vis,
            f"[ORIG RED] Offset: {orig_off:+4d} px | Steer: {orig_steer:+5.1f} deg",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (80, 80, 255),
            2,
        )
        cv2.putText(
            vis,
            f"[OPT BLUE] Offset: {opt_off:+4d} px | Steer: {opt_steer:+5.1f} deg",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 200, 50),
            2,
        )
        status_str = "CROSS!" if is_cross else "LINE"
        cv2.putText(
            vis,
            f"Mode: {self.junction_mode} | Status: {status_str} | Infer: {self.last_inference_ms:.1f}ms",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 200, 200),
            1,
        )

        chart_h = 120
        chart = np.full((chart_h, width, 3), 24, dtype=np.uint8)
        mid_y = chart_h // 2
        cv2.line(chart, (0, mid_y), (width, mid_y), (70, 70, 70), 1)
        cv2.putText(chart, "Error Waveform: RED=Original, BLUE=Optimized (0 px center)", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        if len(self.history_orig) >= 2:
            pts_orig = []
            pts_opt = []
            n_pts = len(self.history_orig)
            for i in range(n_pts):
                x = int(10 + (width - 20) * (i / max(1, 119)))
                yo = int(np.clip(mid_y - (self.history_orig[i] / 45.0) * (chart_h * 0.42), 4, chart_h - 4))
                yp = int(np.clip(mid_y - (self.history_opt[i] / 45.0) * (chart_h * 0.42), 4, chart_h - 4))
                pts_orig.append((x, yo))
                pts_opt.append((x, yp))
            cv2.polylines(chart, [np.array(pts_orig, dtype=np.int32)], False, (0, 0, 255), 2)
            cv2.polylines(chart, [np.array(pts_opt, dtype=np.int32)], False, (255, 180, 0), 2)

        return np.vstack([vis, chart])

    def release(self) -> None:
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None


def run_image_test(tester: SmoothLineTester, image_paths: list[str], save_dir: str | None) -> int:
    """Read offline images, print comparison table, and show or save comparison images."""
    print("\n" + "=" * 78)
    print(f"{'Image':<20} | {'Orig Offset':<12} | {'Opt Offset':<12} | {'Orig Steer':<11} | {'Opt Steer'}")
    print("-" * 78)

    out_dir = Path(save_dir) if save_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        tester.reset_states()
        res = tester.evaluate_frame(frame, from_debug_image=True)
        name = Path(img_path).name
        print(
            f"{name:<20} | {res['orig_offset']:>+8d} px  | {res['opt_offset']:>+8d} px  | "
            f"{res['orig_steer']:>+7.1f} deg | {res['opt_steer']:>+7.1f} deg"
        )

        out_file = out_dir / f"compare_{name}"
        cv2.imwrite(str(out_file), res["vis"])

        if has_display:
            cv2.imshow("Test Line A/B Comparison (Press any key for next, Q to quit)", res["vis"])
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    print("=" * 78)
    print(f"[INFO] Comparison images saved to: {out_dir.resolve()}")
    if has_display:
        cv2.destroyAllWindows()
    return 0


def run_live_camera_test(args, tester: SmoothLineTester) -> int:
    """Live camera test with optional chassis closed-loop line following."""
    camera_source: int | str = int(args.camera) if str(args.camera).isdigit() else args.camera
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    cap = cv2.VideoCapture(camera_source, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera: {camera_source}")
        return 1

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    chassis = None
    if args.drive:
        if ChassisController is None:
            print("[ERROR] Serial.ChassisController not found; cannot enable --drive mode")
            cap.release()
            return 1
        chassis = ChassisController(args.port, args.baudrate)
        print(f"[DRIVE] Chassis serial connected: {args.port} | line speed={args.speed}")

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and not args.no_gui
    print("\n>>> Live line-following smoothness comparison started")
    print(">>> Keys: [Q/ESC] quit | [M] toggle NORMAL/L junction mode | [R] reset filter state | [SPACE] emergency stop")

    driving_enabled = bool(args.drive)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if args.flip in (-1, 0, 1):
                frame = cv2.flip(frame, args.flip)

            res = tester.evaluate_frame(frame, from_debug_image=False)

            if chassis is not None:
                if driving_enabled:
                    if res["status"] == "CROSS" and args.stop_on_cross:
                        chassis.stop()
                        driving_enabled = False
                        print("\n[CROSS] Junction detected; stopped automatically (press SPACE to resume)")
                    else:
                        chassis.drive_steer(args.speed, round(res["opt_steer"], 2))
                else:
                    chassis.stop()

            print(
                f"Status={res['status']:<5} | "
                f"ORIG_Off={res['orig_offset']:+4d}px ({res['orig_steer']:+5.1f} deg) | "
                f"OPT_Off={res['opt_offset']:+4d}px ({res['opt_steer']:+5.1f} deg) | "
                f"Diff={res['opt_offset'] - res['orig_offset']:+4d}px | "
                f"Infer={tester.last_inference_ms:5.1f}ms",
                end="\r",
                flush=True,
            )

            if has_display:
                cv2.imshow("Test Line Smoothness (ORIG=Red, OPT=Blue)", res["vis"])
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\n[SYS] User exited test")
                    break
                elif key in (ord("m"), ord("M")):
                    tester.junction_mode = "L" if tester.junction_mode == "NORMAL" else "NORMAL"
                    print(f"\n[MODE] Junction mode switched to: {tester.junction_mode}")
                elif key in (ord("r"), ord("R")):
                    tester.reset_states()
                    print("\n[RESET] Filter history reset")
                elif key == ord(" ") and chassis is not None:
                    driving_enabled = not driving_enabled
                    if not driving_enabled:
                        chassis.stop()
                    print(f"\n[DRIVE] Chassis output: {'RESUMED' if driving_enabled else 'PAUSED'}")

    except KeyboardInterrupt:
        print("\n[SYS] Ctrl+C received; stopping test")
    finally:
        if chassis is not None:
            chassis.stop()
            chassis.close()
        cap.release()
        if has_display:
            cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Line-following error smoothing comparison and real-vehicle test tool (test_line.py)")
    parser.add_argument(
        "--images",
        nargs="+",
        help="Offline test image paths (e.g. --images not_at_junction.jpg crossroad.jpg)",
    )
    parser.add_argument("--save-dir", default=".", help="Directory to save offline comparison images")
    parser.add_argument("--camera", default=os.environ.get("NAV_CAMERA", "2"), help="Camera index or device path")
    parser.add_argument("--flip", type=int, default=int(os.environ.get("NAV_CAMERA_FLIP", "-1")), help="Flip: -1, 0, 1, or 2 (no flip)")
    parser.add_argument("--model", default=os.environ.get("DEEPLAB_MODEL", str(DEFAULT_MODEL)), help="RKNN model path")
    parser.add_argument("--width", type=int, default=544, help="Camera capture width")
    parser.add_argument("--height", type=int, default=288, help="Camera capture height")
    parser.add_argument("--no-gui", action="store_true", help="Disable OpenCV windows (SSH headless mode)")

    parser.add_argument("--y-start", type=float, default=0.50, help="Scanline start height ratio (default: 0.50)")
    parser.add_argument("--span-max", type=float, default=0.78, help="Junction wide-row rejection threshold ratio (default: 0.78)")
    parser.add_argument("--alpha", type=float, default=0.28, help="Center exponential smoothing coefficient (default: 0.28)")
    parser.add_argument("--deadband", type=int, default=3, help="Straight-line deadband pixel threshold (default: 3)")
    parser.add_argument("--kp", type=float, default=0.45, help="Steering proportional gain KP")
    parser.add_argument("--kd", type=float, default=0.10, help="Steering derivative gain KD")

    parser.add_argument("--drive", action="store_true", help="Enable chassis serial control for real-vehicle smooth-steering test")
    parser.add_argument("--stop-on-cross", action="store_true", help="Automatically stop when a junction is detected in drive mode")
    parser.add_argument("--speed", type=int, default=9, help="Real-vehicle line-following speed (default: 9)")
    parser.add_argument(
        "--port",
        default=os.environ.get(
            "CHASSIS_PORT",
            "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A5069RR4-if00-port0",
        ),
        help="Chassis serial port path",
    )
    parser.add_argument("--baudrate", type=int, default=115200, help="Chassis serial baudrate")

    args = parser.parse_args()

    tester = SmoothLineTester(
        model_path=args.model,
        load_rknn=not bool(args.images),
        scan_y_start=args.y_start,
        max_span_ratio=args.span_max,
        smooth_alpha=args.alpha,
        deadband_px=args.deadband,
        kp=args.kp,
        kd=args.kd,
    )

    try:
        if args.images:
            return run_image_test(tester, args.images, args.save_dir)
        return run_live_camera_test(args, tester)
    finally:
        tester.release()


if __name__ == "__main__":
    raise SystemExit(main())