#!/usr/bin/env python3
"""
道路 Mask 前视摄像头分支检测测试（优化版）

核心改动：
1. 不再用“交汇点 -> 最远像素”的直线作为分支线，而是沿 Skeleton 图结构真实追踪路径。
2. 对短毛刺进行剪枝，减少普通道路/Mask 噪声造成的假路口。
3. 用分支“路径长度”而不是欧氏距离判断是否是真分支。
4. 用交汇点附近的初始切线判断 LEFT / STRAIGHT / RIGHT，弯道更稳定。
5. 保留绿色道路 Mask、Skeleton、候选点、交汇点和分支 Debug。
6. 保留连续帧 STABLE 机制。
7. 不连接串口、不控制底盘。

依赖：
    pip install opencv-contrib-python numpy

运行：
    python mask_branch_test_optimized.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

from old.SRC.line_detect import DEFAULT_MODEL, LineTracker


# ============================================================
# 显示配置
# ============================================================

DIRECTION_ORDER = ("BACK", "LEFT", "STRAIGHT", "RIGHT")

# BGR
DIRECTION_COLORS = {
    "BACK": (150, 150, 150),
    "LEFT": (255, 120, 0),
    "STRAIGHT": (0, 255, 255),
    "RIGHT": (0, 120, 255),
}


# ============================================================
# 数据结构
# ============================================================

@dataclass(frozen=True)
class BranchArm:
    direction: str
    endpoint: tuple[int, int]
    length: float
    angle: float
    path: tuple[tuple[int, int], ...]
    start_radius: float = 0.0


@dataclass
class BranchDetection:
    detected: bool
    kind: str
    junction: tuple[int, int] | None
    arms: tuple[BranchArm, ...]
    skeleton: np.ndarray
    candidates: tuple[tuple[int, int], ...]
    thinning_method: str
    processed_mask: np.ndarray
    pruned_skeleton: np.ndarray

    @property
    def directions(self) -> tuple[str, ...]:
        present = {arm.direction for arm in self.arms}
        return tuple(
            name for name in DIRECTION_ORDER
            if name in present
        )

    @property
    def signature(self):
        if not self.detected:
            return None
        return self.kind, self.directions


# ============================================================
# Skeleton + Graph 分支检测
# ============================================================

class MaskBranchDetector:
    """
    针对“前视摄像头 + 道路 Mask”的分支检测器。

    思路：
        Mask
          ↓
        闭运算
          ↓
        Skeleton
          ↓
        删除短毛刺
          ↓
        找 degree >= 3 的候选节点
          ↓
        对每个候选节点建立局部 Graph
          ↓
        沿 Skeleton 真实追踪每一条 arm
          ↓
        按路径长度 + 初始方向判断分支
    """

    def __init__(
        self,
        scale: float = 0.5,
        min_arm_ratio: float = 0.10,
        min_arm_px: float = 12.0,
        junction_radius: int = 7,
        close_kernel: int = 5,
        open_kernel: int = 0,
        spur_length_ratio: float = 0.045,
        max_spur_px: float = 18.0,
        min_start_radius: float = 2.0,
        direction_probe_px: int = 14,
        max_trace_px: int = 500,
    ):
        self.scale = float(scale)
        self.min_arm_ratio = float(min_arm_ratio)
        self.min_arm_px = float(min_arm_px)
        self.junction_radius = int(junction_radius)
        self.close_kernel = int(close_kernel)
        self.open_kernel = int(open_kernel)
        self.spur_length_ratio = float(spur_length_ratio)
        self.max_spur_px = float(max_spur_px)
        self.min_start_radius = float(min_start_radius)
        self.direction_probe_px = int(direction_probe_px)
        self.max_trace_px = int(max_trace_px)

        # 8 邻域
        self.neighbors8 = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1),
        ]

    # --------------------------------------------------------
    # Skeleton
    # --------------------------------------------------------

    @staticmethod
    def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
        image = np.where(mask > 0, 255, 0).astype(np.uint8)
        skeleton = np.zeros_like(image)

        element = cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (3, 3),
        )

        for _ in range(max(image.shape)):
            opened = cv2.morphologyEx(
                image,
                cv2.MORPH_OPEN,
                element,
            )

            residue = cv2.subtract(
                image,
                opened,
            )

            skeleton = cv2.bitwise_or(
                skeleton,
                residue,
            )

            image = cv2.erode(
                image,
                element,
            )

            if cv2.countNonZero(image) == 0:
                break

        return skeleton

    def _thin(
        self,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, str]:

        ximgproc = getattr(
            cv2,
            "ximgproc",
            None,
        )

        thinning = getattr(
            ximgproc,
            "thinning",
            None,
        )

        if callable(thinning):
            skeleton = thinning(
                mask,
                thinningType=cv2.ximgproc.THINNING_ZHANGSUEN,
            )
            return skeleton, "ximgproc"

        return (
            self._morphological_skeleton(mask),
            "morph",
        )

    # --------------------------------------------------------
    # Mask 预处理
    # --------------------------------------------------------

    def _preprocess_mask(
        self,
        component_mask: np.ndarray,
    ) -> np.ndarray:

        binary = np.where(
            component_mask > 0,
            255,
            0,
        ).astype(np.uint8)

        width = max(
            8,
            int(round(binary.shape[1] * self.scale)),
        )

        height = max(
            8,
            int(round(binary.shape[0] * self.scale)),
        )

        mask = cv2.resize(
            binary,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

        # 闭运算：填补 Mask 小裂缝
        if self.close_kernel > 1:
            k = self.close_kernel | 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (k, k),
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=1,
            )

        # 可选开运算：默认关闭，避免把真实窄分支吃掉
        if self.open_kernel > 1:
            k = self.open_kernel | 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (k, k),
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=1,
            )

        return mask

    # --------------------------------------------------------
    # 选择从底部延伸出来的 Skeleton 连通域
    # --------------------------------------------------------

    @staticmethod
    def _select_bottom_component(
        skeleton: np.ndarray,
    ) -> np.ndarray:

        binary = np.where(
            skeleton > 0,
            255,
            0,
        ).astype(np.uint8)

        count, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8,
            )
        )

        if count <= 1:
            return skeleton

        height, width = skeleton.shape

        bottom_rows = max(
            4,
            int(height * 0.15),
        )

        best_label = None
        best_score = -1.0

        for label in range(1, count):

            component = labels == label

            bottom_support = int(
                np.count_nonzero(
                    component[-bottom_rows:, :]
                )
            )

            area = int(
                stats[label, cv2.CC_STAT_AREA]
            )

            center_penalty = abs(
                float(centroids[label][0])
                - width * 0.5
            )

            # 底部连接是非常强的前视小车先验
            score = (
                bottom_support * 1000.0
                + area
                - center_penalty
            )

            if (
                bottom_support > 0
                and score > best_score
            ):
                best_label = label
                best_score = score

        if best_label is None:
            areas = stats[
                1:,
                cv2.CC_STAT_AREA,
            ]

            best_label = (
                int(np.argmax(areas)) + 1
            )

        return np.where(
            labels == best_label,
            255,
            0,
        ).astype(np.uint8)

    # --------------------------------------------------------
    # 邻居计算
    # --------------------------------------------------------

    def _neighbor_count(
        self,
        skeleton: np.ndarray,
    ) -> np.ndarray:

        binary = (
            skeleton > 0
        ).astype(np.uint8)

        kernel = np.ones(
            (3, 3),
            dtype=np.uint8,
        )

        kernel[1, 1] = 0

        return cv2.filter2D(
            binary,
            cv2.CV_16S,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
        )

    # --------------------------------------------------------
    # 删除短毛刺
    # --------------------------------------------------------

    def _prune_short_spurs(
        self,
        skeleton: np.ndarray,
    ) -> np.ndarray:

        result = skeleton.copy()

        h, w = result.shape

        max_spur = min(
            self.max_spur_px,
            h * self.spur_length_ratio,
        )

        max_spur = max(
            4.0,
            max_spur,
        )

        # 反复剪枝，通常 2~4 次足够
        for _ in range(4):

            binary = (
                result > 0
            ).astype(np.uint8)

            neighbors = self._neighbor_count(
                result
            )

            endpoints = np.argwhere(
                (binary > 0)
                & (neighbors == 1)
            )

            junction_mask = (
                (binary > 0)
                & (neighbors >= 3)
            )

            if len(endpoints) == 0:
                break

            remove_pixels = []

            for y, x in endpoints:

                path = []
                current = (
                    int(x),
                    int(y),
                )

                previous = None

                reached_junction = False
                path_length = 0.0

                for _step in range(
                    int(max_spur) + 5
                ):

                    cx, cy = current

                    if not (
                        0 <= cx < w
                        and 0 <= cy < h
                    ):
                        break

                    path.append(
                        (cx, cy)
                    )

                    if (
                        len(path) > 1
                        and junction_mask[cy, cx]
                    ):
                        reached_junction = True
                        break

                    candidates = []

                    for dx, dy in self.neighbors8:

                        nx = cx + dx
                        ny = cy + dy

                        if not (
                            0 <= nx < w
                            and 0 <= ny < h
                        ):
                            continue

                        if result[ny, nx] == 0:
                            continue

                        if previous is not None:
                            if (
                                nx == previous[0]
                                and ny == previous[1]
                            ):
                                continue

                        candidates.append(
                            (nx, ny)
                        )

                    if not candidates:
                        break

                    # 理论上骨架路径上应只有一个下一点
                    # 如果出现多个，则已经到达节点
                    if len(candidates) > 1:
                        if (
                            len(path) <= 2
                            or junction_mask[cy, cx]
                        ):
                            reached_junction = True
                        break

                    nxt = candidates[0]

                    step = math.hypot(
                        nxt[0] - cx,
                        nxt[1] - cy,
                    )

                    path_length += step

                    previous = current
                    current = nxt

                    if path_length > max_spur:
                        break

                if (
                    reached_junction
                    and path_length <= max_spur
                    and len(path) >= 2
                ):
                    remove_pixels.extend(
                        path
                    )

            if not remove_pixels:
                break

            for x, y in remove_pixels:
                result[y, x] = 0

        return result

    # --------------------------------------------------------
    # 分支候选点聚类
    # --------------------------------------------------------

    def _cluster_branch_pixels(
        self,
        skeleton: np.ndarray,
    ) -> list[tuple[int, int]]:

        neighbors = self._neighbor_count(
            skeleton
        )

        branch_pixels = np.where(
            (
                (skeleton > 0)
                & (neighbors >= 3)
            ),
            255,
            0,
        ).astype(np.uint8)

        if cv2.countNonZero(
            branch_pixels
        ) == 0:
            return []

        # 让同一个交汇区域的多个 branch pixels 合并
        merge_size = max(
            3,
            self.junction_radius * 2 + 1,
        )

        merge_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                merge_size,
                merge_size,
            ),
        )

        merged = cv2.dilate(
            branch_pixels,
            merge_kernel,
        )

        count, labels = cv2.connectedComponents(
            merged,
            connectivity=8,
        )

        centers = []

        for label in range(
            1,
            count,
        ):

            ys, xs = np.where(
                (labels == label)
                & (branch_pixels > 0)
            )

            if xs.size == 0:
                continue

            centers.append(
                (
                    int(round(
                        float(np.median(xs))
                    )),
                    int(round(
                        float(np.median(ys))
                    )),
                )
            )

        return centers

    # --------------------------------------------------------
    # Skeleton 图邻居
    # --------------------------------------------------------

    def _skeleton_neighbors(
        self,
        skeleton: np.ndarray,
        point: tuple[int, int],
    ) -> list[tuple[int, int]]:

        x, y = point

        result = []

        for dx, dy in self.neighbors8:

            nx = x + dx
            ny = y + dy

            if (
                0 <= nx < skeleton.shape[1]
                and
                0 <= ny < skeleton.shape[0]
                and
                skeleton[ny, nx] > 0
            ):
                result.append(
                    (nx, ny)
                )

        return result

    # --------------------------------------------------------
    # 从交汇区域向外追踪真实 Skeleton 路径
    # --------------------------------------------------------

    def _trace_component_arm(
        self,
        skeleton: np.ndarray,
        center: tuple[int, int],
        component_mask: np.ndarray,
        distance_map: np.ndarray | None,
    ) -> BranchArm | None:

        center_x, center_y = center

        ys, xs = np.where(
            component_mask > 0
        )

        if xs.size == 0:
            return None

        # 找到距离交汇中心最近的一圈像素作为起点
        distances = np.hypot(
            xs - center_x,
            ys - center_y,
        )

        min_distance = float(
            np.min(distances)
        )

        seed_indices = np.where(
            distances
            <= min_distance + 1.5
        )[0]

        seeds = [
            (
                int(xs[i]),
                int(ys[i]),
            )
            for i in seed_indices
        ]

        if not seeds:
            return None

        # 多源 BFS：
        # 距离表示沿 Skeleton 的真实路径长度
        queue = deque()
        graph_dist = {}
        parent = {}

        for seed in seeds:

            graph_dist[seed] = 0.0
            parent[seed] = None
            queue.append(seed)

        farthest = seeds[0]
        farthest_distance = 0.0

        while queue:

            current = queue.popleft()

            cx, cy = current
            current_distance = graph_dist[
                current
            ]

            if current_distance > farthest_distance:
                farthest = current
                farthest_distance = (
                    current_distance
                )

            if (
                farthest_distance
                >= self.max_trace_px
            ):
                continue

            for nxt in self._skeleton_neighbors(
                skeleton,
                current,
            ):

                if component_mask[
                    nxt[1],
                    nxt[0]
                ] == 0:
                    continue

                step = math.hypot(
                    nxt[0] - cx,
                    nxt[1] - cy,
                )

                new_distance = (
                    current_distance
                    + step
                )

                if new_distance > self.max_trace_px:
                    continue

                old_distance = graph_dist.get(
                    nxt
                )

                if (
                    old_distance is None
                    or new_distance < old_distance
                ):
                    graph_dist[nxt] = (
                        new_distance
                    )
                    parent[nxt] = current
                    queue.append(nxt)

        if farthest_distance <= 0:
            return None

        # 回溯真实 Skeleton 路径
        path = []

        current = farthest

        guard = 0

        while (
            current is not None
            and guard < self.max_trace_px * 2
        ):

            path.append(current)

            current = parent.get(
                current
            )

            guard += 1

        path.reverse()

        if len(path) < 2:
            return None

        # ----------------------------------------------------
        # 方向不要使用“终点 - 交汇点”
        #
        # 因为你的道路是前视透视的，弯道会导致终点方向严重偏移。
        #
        # 使用交汇点附近的一小段真实路径计算初始切线。
        # ----------------------------------------------------

        probe_length = min(
            self.direction_probe_px,
            max(
                2,
                len(path) - 1,
            ),
        )

        probe_index = probe_length

        px, py = path[
            probe_index
        ]

        # 起点使用路径最前端
        sx, sy = path[0]

        dx = float(
            px - sx
        )

        dy = float(
            py - sy
        )

        direction, angle = (
            self._direction(
                dx,
                dy,
            )
        )

        start_radius = 0.0

        if distance_map is not None:

            samples = path[
                :min(
                    len(path),
                    self.direction_probe_px + 1,
                )
            ]

            if samples:

                radii = [
                    float(
                        distance_map[y, x]
                    )
                    for x, y in samples
                ]

                start_radius = float(
                    np.median(radii)
                )

        return BranchArm(
            direction=direction,
            endpoint=farthest,
            length=float(
                farthest_distance
            ),
            angle=float(angle),
            path=tuple(path),
            start_radius=start_radius,
        )

    # --------------------------------------------------------
    # 追踪候选交汇点的所有分支
    # --------------------------------------------------------

    def _trace_arms(
        self,
        skeleton: np.ndarray,
        center: tuple[int, int],
        distance_map: np.ndarray | None,
    ) -> tuple[BranchArm, ...]:

        center_x, center_y = center

        separated = skeleton.copy()

        # 删除交汇点附近的区域
        # 剩下的每一个相邻连通区域就是一个候选 arm
        cv2.circle(
            separated,
            center,
            self.junction_radius,
            0,
            thickness=-1,
        )

        binary = np.where(
            separated > 0,
            255,
            0,
        ).astype(np.uint8)

        count, labels = cv2.connectedComponents(
            binary,
            connectivity=8,
        )

        min_length = max(
            self.min_arm_px,
            skeleton.shape[0]
            * self.min_arm_ratio,
        )

        arms = []

        for label in range(
            1,
            count,
        ):

            component = np.where(
                labels == label,
                255,
                0,
            ).astype(np.uint8)

            ys, xs = np.where(
                component > 0
            )

            if xs.size == 0:
                continue

            distances = np.hypot(
                xs - center_x,
                ys - center_y,
            )

            # 必须真正贴着 junction
            if float(
                np.min(distances)
            ) > self.junction_radius + 2.5:
                continue

            arm = self._trace_component_arm(
                skeleton,
                center,
                component,
                distance_map,
            )

            if arm is None:
                continue

            if arm.length < min_length:
                continue

            # Mask 太窄的位置大概率是噪声/毛刺
            if (
                distance_map is not None
                and arm.start_radius
                < self.min_start_radius
            ):
                continue

            arms.append(arm)

        # 同一方向如果出现多个 arm，只保留最长的一个
        best_by_direction = {}

        for arm in arms:

            previous = best_by_direction.get(
                arm.direction
            )

            if (
                previous is None
                or arm.length > previous.length
            ):
                best_by_direction[
                    arm.direction
                ] = arm

        return tuple(
            best_by_direction[name]
            for name in DIRECTION_ORDER
            if name in best_by_direction
        )

    # --------------------------------------------------------
    # 方向
    # --------------------------------------------------------

    @staticmethod
    def _direction(
        dx: float,
        dy: float,
    ) -> tuple[str, float]:

        # OpenCV:
        #   x 向右
        #   y 向下
        #
        # 转换后：
        #   0°   = RIGHT
        #   90°  = STRAIGHT（图像上方）
        #   180° = LEFT
        #   -90° = BACK（图像下方）

        angle = math.degrees(
            math.atan2(
                -dy,
                dx,
            )
        )

        if -135.0 <= angle < -45.0:
            return "BACK", angle

        if -45.0 <= angle < 45.0:
            return "RIGHT", angle

        if 45.0 <= angle < 135.0:
            return "STRAIGHT", angle

        return "LEFT", angle

    # --------------------------------------------------------
    # 分类
    # --------------------------------------------------------

    @staticmethod
    def _classify(
        arms: tuple[BranchArm, ...],
    ) -> str:

        directions = {
            arm.direction
            for arm in arms
        }

        outgoing = (
            directions
            - {"BACK"}
        )

        if {
            "LEFT",
            "STRAIGHT",
            "RIGHT",
        } <= outgoing:
            return "CROSS"

        if {
            "LEFT",
            "RIGHT",
        } <= outgoing:
            return "T_JUNCTION"

        if {
            "LEFT",
            "STRAIGHT",
        } <= outgoing:
            return "LEFT_BRANCH"

        if {
            "RIGHT",
            "STRAIGHT",
        } <= outgoing:
            return "RIGHT_BRANCH"

        return "BRANCH"

    # --------------------------------------------------------
    # 主检测
    # --------------------------------------------------------

    def detect(
        self,
        component_mask: np.ndarray,
    ) -> BranchDetection:

        if (
            component_mask is None
            or component_mask.ndim != 2
        ):

            empty = np.zeros(
                (1, 1),
                dtype=np.uint8,
            )

            return BranchDetection(
                False,
                "NO_MASK",
                None,
                (),
                empty,
                (),
                "none",
                empty,
                empty,
            )

        mask = self._preprocess_mask(
            component_mask
        )

        skeleton, method = self._thin(
            mask
        )

        skeleton = self._select_bottom_component(
            skeleton
        )

        # 删除短毛刺
        pruned = self._prune_short_spurs(
            skeleton
        )

        candidates = self._cluster_branch_pixels(
            pruned
        )

        # Distance Transform：
        # 让我们知道 Skeleton 附近道路有多宽。
        distance_map = cv2.distanceTransform(
            np.where(
                mask > 0,
                255,
                0,
            ).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )

        best = None
        best_score = None

        for center in candidates:

            arms = self._trace_arms(
                pruned,
                center,
                distance_map,
            )

            directions = {
                arm.direction
                for arm in arms
            }

            outgoing = (
                directions
                - {"BACK"}
            )

            # 前视小车的核心约束：
            #
            # 交汇点必须至少有：
            #   当前道路 BACK
            #   +
            #   至少一个新的前方方向
            #
            # 这样普通直道不会因为奇怪的局部节点被判路口。
            valid = (
                "BACK" in directions
                and len(arms) >= 3
                and len(outgoing) >= 2
            )

            if not valid:
                continue

            # 分支数量
            arm_count_score = len(arms) * 10000.0

            # 真正长分支优先
            length_score = sum(
                arm.length
                for arm in arms
            )

            # 交汇点越靠近画面中下部，
            # 对前视小车越有实际意义。
            y_score = center[1] * 2.0

            # 路口附近道路宽度作为辅助证据
            radius_score = sum(
                arm.start_radius
                for arm in arms
            ) * 10.0

            score = (
                arm_count_score
                + length_score
                + y_score
                + radius_score
            )

            if (
                best_score is None
                or score > best_score
            ):
                best = (
                    center,
                    arms,
                )
                best_score = score

        if best is None:

            return BranchDetection(
                False,
                "LINE",
                None,
                (),
                skeleton,
                tuple(candidates),
                method,
                mask,
                pruned,
            )

        center, arms = best

        return BranchDetection(
            True,
            self._classify(arms),
            center,
            arms,
            skeleton,
            tuple(candidates),
            method,
            mask,
            pruned,
        )


# ============================================================
# 复用原来的 RKNN 道路 Mask
# ============================================================

class BranchMaskTracker(LineTracker):

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        self.latest_component_mask = None

        super().__init__(
            *args,
            **kwargs,
        )

    def get_road_info(
        self,
        frame,
    ):

        self.latest_component_mask = None

        return super().get_road_info(
            frame
        )

    def _is_crossroad(
        self,
        component_mask,
    ):

        self.latest_component_mask = (
            component_mask.copy()
        )

        # 测试脚本完全交给新的 Graph 分支检测器。
        return False


# ============================================================
# 连续帧稳定
# ============================================================

class TemporalBranchConfirmation:

    def __init__(
        self,
        required_frames: int,
    ):

        self.required_frames = max(
            1,
            int(required_frames),
        )

        self.last_signature = None
        self.count = 0

    def update(
        self,
        detection: BranchDetection,
    ) -> tuple[bool, int]:

        signature = detection.signature

        if signature is None:

            self.last_signature = None
            self.count = 0

            return False, 0

        if (
            signature
            == self.last_signature
        ):

            self.count += 1

        else:

            self.last_signature = (
                signature
            )

            self.count = 1

        return (
            self.count
            >= self.required_frames,
            self.count,
        )


# ============================================================
# 摄像头工具
# ============================================================

def parse_camera_source(
    value: object,
) -> int | str:

    text = str(value).strip()

    try:
        return int(text)
    except ValueError:
        return text


def parse_flip_code(
    value: object,
) -> int | None:

    text = str(value).strip().lower()

    if text in {
        "none",
        "off",
        "false",
        "no",
    }:
        return None

    code = int(text)

    if code not in (-1, 0, 1):
        raise ValueError(
            "翻转参数只能是 -1、0、1 或 none"
        )

    return code


def has_graphical_display() -> bool:

    if not sys.platform.startswith(
        "linux"
    ):
        return True

    return bool(
        os.environ.get("DISPLAY")
        or os.environ.get(
            "WAYLAND_DISPLAY"
        )
    )


def open_camera(
    source,
    width,
    height,
    fps,
):

    backend = getattr(
        cv2,
        "CAP_V4L2",
        cv2.CAP_ANY,
    )

    try:
        camera = cv2.VideoCapture(
            source,
            backend,
        )
    except Exception:
        camera = cv2.VideoCapture(
            source
        )

    if (
        not camera.isOpened()
        and backend != cv2.CAP_ANY
    ):

        camera.release()

        camera = cv2.VideoCapture(
            source
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        width,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        height,
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        fps,
    )

    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        ),
    )

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    return camera


def wait_for_camera(
    camera,
    timeout,
) -> bool:

    deadline = (
        time.monotonic()
        + timeout
    )

    while (
        time.monotonic()
        < deadline
    ):

        if camera.isOpened():

            ok, frame = camera.read()

            if (
                ok
                and frame is not None
                and frame.size
            ):
                return True

        time.sleep(0.05)

    return False


# ============================================================
# Debug 绘制
# ============================================================

def draw_detection(
    frame: np.ndarray,
    component_mask: np.ndarray | None,
    detection: BranchDetection,
    confirmed: bool,
    stable_count: int,
    required_frames: int,
    offset: int,
) -> np.ndarray:

    display = frame.copy()

    if component_mask is None:

        cv2.putText(
            display,
            "NO ROAD MASK",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        return display

    mask_h, mask_w = (
        component_mask.shape
    )

    frame_h, frame_w = (
        frame.shape[:2]
    )

    roi_start_y = max(
        0,
        frame_h - mask_h,
    )

    roi_end_y = min(
        frame_h,
        roi_start_y + mask_h,
    )

    roi_end_x = min(
        frame_w,
        mask_w,
    )

    # --------------------------------------------------------
    # 1. 绿色道路 Mask
    # --------------------------------------------------------

    overlay = display.copy()

    roi_overlay = overlay[
        roi_start_y:roi_end_y,
        :roi_end_x,
    ]

    visible_mask = component_mask[
        :roi_end_y - roi_start_y,
        :roi_end_x,
    ]

    roi_overlay[
        visible_mask > 0
    ] = (0, 255, 0)

    cv2.addWeighted(
        overlay,
        0.35,
        display,
        0.65,
        0.0,
        dst=display,
    )

    # --------------------------------------------------------
    # 2. 绿色道路轮廓
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        np.where(
            component_mask > 0,
            255,
            0,
        ).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if contours:

        contour = (
            max(
                contours,
                key=cv2.contourArea,
            )
            + (0, roi_start_y)
        )

        cv2.drawContours(
            display,
            [contour],
            -1,
            (0, 255, 0),
            2,
        )

    # --------------------------------------------------------
    # Skeleton 坐标转换
    # --------------------------------------------------------

    small_h, small_w = (
        detection.skeleton.shape
    )

    scale_x = (
        mask_w
        / max(1, small_w)
    )

    scale_y = (
        mask_h
        / max(1, small_h)
    )

    def to_frame(
        point: tuple[int, int],
    ) -> tuple[int, int]:

        return (
            int(round(
                point[0] * scale_x
            )),
            int(round(
                point[1] * scale_y
                + roi_start_y
            )),
        )

    # --------------------------------------------------------
    # 3. 原始 Skeleton：黄色
    # --------------------------------------------------------

    skeleton_large = cv2.resize(
        detection.skeleton,
        (mask_w, mask_h),
        interpolation=cv2.INTER_NEAREST,
    )

    skeleton_roi = display[
        roi_start_y:roi_end_y,
        :roi_end_x,
    ]

    visible_skeleton = (
        skeleton_large[
            :roi_end_y - roi_start_y,
            :roi_end_x,
        ]
    )

    skeleton_roi[
        visible_skeleton > 0
    ] = (0, 255, 255)

    # --------------------------------------------------------
    # 4. 剪枝后的 Skeleton：白色
    # --------------------------------------------------------

    pruned_large = cv2.resize(
        detection.pruned_skeleton,
        (mask_w, mask_h),
        interpolation=cv2.INTER_NEAREST,
    )

    pruned_roi = display[
        roi_start_y:roi_end_y,
        :roi_end_x,
    ]

    visible_pruned = (
        pruned_large[
            :roi_end_y - roi_start_y,
            :roi_end_x,
        ]
    )

    # 白色只覆盖在道路区域内
    pruned_roi[
        visible_pruned > 0
    ] = (255, 255, 255)

    # --------------------------------------------------------
    # 5. 候选交汇点：橙色空心圆
    # --------------------------------------------------------

    for candidate in (
        detection.candidates
    ):

        cv2.circle(
            display,
            to_frame(candidate),
            5,
            (0, 165, 255),
            1,
        )

    # --------------------------------------------------------
    # 6. 最终交汇点和真实路径
    # --------------------------------------------------------

    if detection.junction is not None:

        junction = to_frame(
            detection.junction
        )

        # 紫色交汇点
        cv2.circle(
            display,
            junction,
            9,
            (255, 0, 255),
            -1,
        )

        # 实际 Skeleton 路径
        for arm in detection.arms:

            color = DIRECTION_COLORS[
                arm.direction
            ]

            path = [
                to_frame(point)
                for point in arm.path
            ]

            if len(path) >= 2:

                pts = np.asarray(
                    path,
                    dtype=np.int32,
                ).reshape(
                    -1,
                    1,
                    2,
                )

                # 这是“真实骨架路径”，
                # 不再画错误的直线。
                cv2.polylines(
                    display,
                    [pts],
                    False,
                    color,
                    3,
                    cv2.LINE_AA,
                )

            endpoint = to_frame(
                arm.endpoint
            )

            cv2.circle(
                display,
                endpoint,
                5,
                color,
                -1,
            )

            # 方向 + 路径长度
            text = (
                f"{arm.direction}"
                f" {arm.length:.0f}"
            )

            cv2.putText(
                display,
                text,
                (
                    endpoint[0] + 5,
                    endpoint[1] - 5,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # 7. 状态信息
    # --------------------------------------------------------

    status_color = (
        (0, 255, 0)
        if confirmed
        else (0, 165, 255)
    )

    directions = (
        ",".join(
            detection.directions
        )
        or "-"
    )

    cv2.putText(
        display,
        f"RAW: {detection.kind} [{directions}]",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        status_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        (
            f"STABLE: "
            f"{stable_count}/"
            f"{required_frames}"
            f" | offset={offset}"
        ),
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        (
            f"skeleton={detection.thinning_method}"
            f" | candidates={len(detection.candidates)}"
        ),
        (10, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if detection.junction is not None:

        cv2.putText(
            display,
            (
                f"junction="
                f"{detection.junction}"
            ),
            (10, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return display


# ============================================================
# 主循环
# ============================================================

def run(args: argparse.Namespace) -> int:

    try:
        source = parse_camera_source(
            args.camera
        )

        flip_code = parse_flip_code(
            args.flip
        )

    except ValueError as exc:

        print(
            f"[配置错误] {exc}"
        )

        return 2

    if (
        not args.no_window
        and not has_graphical_display()
    ):

        print(
            "[显示错误] 当前 Linux 会话没有 "
            "DISPLAY/WAYLAND_DISPLAY；"
            "请在桌面终端运行，或使用 --no-window。"
        )

        return 2

    tracker = None
    camera = None
    branch_events = 0

    try:

        tracker = BranchMaskTracker(
            model_path=args.model,
            debug_mode=False,
        )

        detector = MaskBranchDetector(
            scale=args.skeleton_scale,
            min_arm_ratio=args.min_arm_ratio,
            min_arm_px=args.min_arm_px,
            junction_radius=args.junction_radius,
            close_kernel=args.close_kernel,
            open_kernel=args.open_kernel,
            spur_length_ratio=args.spur_ratio,
            max_spur_px=args.max_spur_px,
            min_start_radius=args.min_start_radius,
            direction_probe_px=args.direction_probe_px,
            max_trace_px=args.max_trace_px,
        )

        confirmation = (
            TemporalBranchConfirmation(
                args.confirm_frames
            )
        )

        camera = open_camera(
            source,
            args.width,
            args.height,
            args.fps,
        )

        if not wait_for_camera(
            camera,
            args.camera_startup_timeout,
        ):

            print(
                f"[摄像头错误] "
                f"{args.camera_startup_timeout:.1f} 秒内"
                f"未从摄像头 {source} 取得有效画面"
            )

            return 1

        print(
            "[分支测试] Graph Skeleton 优化版"
        )

        print(
            "[分支测试] "
            "只使用道路 Mask；"
            "不连接串口、不控制底盘"
        )

        print(
            f"[参数] scale={args.skeleton_scale:.2f} "
            f"min_arm_ratio={args.min_arm_ratio:.3f} "
            f"min_arm_px={args.min_arm_px:.1f} "
            f"junction_radius={args.junction_radius}"
        )

        print(
            f"[参数] spur_ratio={args.spur_ratio:.3f} "
            f"max_spur_px={args.max_spur_px:.1f} "
            f"direction_probe={args.direction_probe_px}"
        )

        if not args.no_window:

            cv2.namedWindow(
                "Mask Branch Test",
                cv2.WINDOW_NORMAL,
            )

            cv2.resizeWindow(
                "Mask Branch Test",
                960,
                720,
            )

            print(
                "[分支测试] q/Esc 退出，"
                "s 保存当前调试画面"
            )

        started_at = time.monotonic()

        misses = 0
        confirmed_active = False
        last_display = None

        while True:

            ok, frame = camera.read()

            if (
                not ok
                or frame is None
                or frame.size == 0
            ):

                misses += 1

                if (
                    misses
                    >= args.max_camera_misses
                ):

                    print(
                        "\n[摄像头错误] "
                        "连续取帧失败次数过多"
                    )

                    return 3

                time.sleep(0.02)

                continue

            misses = 0

            if flip_code in (
                -1,
                0,
                1,
            ):

                frame = cv2.flip(
                    frame,
                    flip_code,
                )

            road_type, offset = (
                tracker.get_road_info(
                    frame
                )
            )

            component_mask = (
                tracker.latest_component_mask
            )

            if component_mask is None:

                empty = np.zeros(
                    (1, 1),
                    dtype=np.uint8,
                )

                detection = (
                    BranchDetection(
                        False,
                        "NO_MASK",
                        None,
                        (),
                        empty,
                        (),
                        "none",
                        empty,
                        empty,
                    )
                )

            else:

                detection = (
                    detector.detect(
                        component_mask
                    )
                )

            confirmed, stable_count = (
                confirmation.update(
                    detection
                )
            )

            if (
                confirmed
                and not confirmed_active
            ):

                branch_events += 1

                print(
                    f"\n[分支确认] "
                    f"#{branch_events} "
                    f"{detection.kind} "
                    f"方向="
                    f"{','.join(detection.directions)}"
                )

                for arm in detection.arms:

                    print(
                        f"    "
                        f"{arm.direction:<8}"
                        f"length={arm.length:6.1f} "
                        f"angle={arm.angle:6.1f}"
                    )

            confirmed_active = confirmed

            print(
                f"[分支] "
                f"road={road_type:<4} "
                f"raw={detection.kind:<12} "
                f"dirs="
                f"{','.join(detection.directions) or '-':<25} "
                f"stable="
                f"{stable_count}/"
                f"{args.confirm_frames} "
                f"events="
                f"{branch_events}",
                end="\r",
                flush=True,
            )

            if not args.no_window:

                last_display = (
                    draw_detection(
                        frame,
                        component_mask,
                        detection,
                        confirmed,
                        stable_count,
                        args.confirm_frames,
                        offset,
                    )
                )

                cv2.imshow(
                    "Mask Branch Test",
                    last_display,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):

                    print(
                        "\n[分支测试] "
                        "收到退出按键"
                    )

                    return 0

                if (
                    key in (
                        ord("s"),
                        ord("S"),
                    )
                    and last_display is not None
                ):

                    filename = (
                        "mask_branch_"
                        f"{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    )

                    if cv2.imwrite(
                        filename,
                        last_display,
                    ):

                        print(
                            f"\n[分支测试] "
                            f"已保存 {filename}"
                        )

            if (
                args.duration > 0
                and (
                    time.monotonic()
                    - started_at
                    >= args.duration
                )
            ):

                print(
                    f"\n[分支测试] "
                    f"已达到设定时长："
                    f"{args.duration:.1f} 秒"
                )

                return 0

    except KeyboardInterrupt:

        print(
            "\n[分支测试] 用户停止测试"
        )

        return 0

    except cv2.error as exc:

        print(
            f"\n[OpenCV错误] {exc}"
        )

        return 4

    finally:

        if camera is not None:
            camera.release()

        if tracker is not None:
            tracker.release()

        cv2.destroyAllWindows()

        print(
            f"[分支测试] 资源已释放；"
            f"确认分支事件={branch_events}"
        )


# ============================================================
# CLI
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--camera",
        default=os.environ.get(
            "NAV_CAMERA",
            "0",
        ),
        help="摄像头编号或 /dev/video* 路径",
    )

    parser.add_argument(
        "--model",
        default=(
            os.environ.get(
                "DEEPLAB_MODEL"
            )
            or str(DEFAULT_MODEL)
        ),
        help="DeepLabV3 RKNN 模型路径",
    )

    parser.add_argument(
        "--flip",
        default=os.environ.get(
            "NAV_CAMERA_FLIP",
            "none",
        ),
        help="画面翻转：-1、0、1 或 none",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--skeleton-scale",
        type=float,
        default=0.5,
        help="Skeleton 计算缩放比例",
    )

    parser.add_argument(
        "--min-arm-ratio",
        type=float,
        default=0.10,
        help="有效分支最小长度 / Skeleton 高度",
    )

    parser.add_argument(
        "--min-arm-px",
        type=float,
        default=12.0,
        help="有效分支的绝对最小长度（缩放后像素）",
    )

    parser.add_argument(
        "--junction-radius",
        type=int,
        default=7,
        help="交汇区域半径",
    )

    parser.add_argument(
        "--close-kernel",
        type=int,
        default=5,
        help="Mask 闭运算核大小；0/1 关闭",
    )

    parser.add_argument(
        "--open-kernel",
        type=int,
        default=0,
        help="Mask 开运算核大小；默认关闭",
    )

    parser.add_argument(
        "--spur-ratio",
        type=float,
        default=0.045,
        help="短毛刺最大长度 / Skeleton 高度",
    )

    parser.add_argument(
        "--max-spur-px",
        type=float,
        default=18.0,
        help="短毛刺最大绝对长度",
    )

    parser.add_argument(
        "--min-start-radius",
        type=float,
        default=2.0,
        help="分支起点最小道路半径",
    )

    parser.add_argument(
        "--direction-probe-px",
        type=int,
        default=14,
        help="使用交汇点附近多少像素估计分支初始方向",
    )

    parser.add_argument(
        "--max-trace-px",
        type=int,
        default=500,
        help="单条 Skeleton 最大追踪长度",
    )

    parser.add_argument(
        "--confirm-frames",
        type=int,
        default=3,
        help="相同分支结构连续确认帧数",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--no-window",
        action="store_true",
    )

    parser.add_argument(
        "--camera-startup-timeout",
        type=float,
        default=float(
            os.environ.get(
                "CAMERA_STARTUP_TIMEOUT",
                "10",
            )
        ),
    )

    parser.add_argument(
        "--max-camera-misses",
        type=int,
        default=int(
            os.environ.get(
                "FULL_MAP_MAX_CAMERA_MISSES",
                "10",
            )
        ),
    )

    args = parser.parse_args()

    if (
        args.width <= 0
        or args.height <= 0
        or args.fps <= 0
    ):
        parser.error(
            "宽度、高度和帧率必须大于 0"
        )

    if not (
        0.1
        <= args.skeleton_scale
        <= 1.0
    ):
        parser.error(
            "--skeleton-scale 必须在 0.1 和 1.0 之间"
        )

    if not (
        0.01
        <= args.min_arm_ratio
        <= 0.5
    ):
        parser.error(
            "--min-arm-ratio 必须在 0.01 和 0.5 之间"
        )

    if args.min_arm_px <= 0:
        parser.error(
            "--min-arm-px 必须大于 0"
        )

    if args.junction_radius < 2:
        parser.error(
            "--junction-radius 至少为 2"
        )

    if args.close_kernel < 0:
        parser.error(
            "--close-kernel 不能为负数"
        )

    if args.open_kernel < 0:
        parser.error(
            "--open-kernel 不能为负数"
        )

    if args.confirm_frames < 1:
        parser.error(
            "--confirm-frames 至少为 1"
        )

    if args.duration < 0:
        parser.error(
            "--duration 不能为负数"
        )

    if args.camera_startup_timeout <= 0:
        parser.error(
            "摄像头启动超时必须大于 0"
        )

    if args.max_camera_misses < 1:
        parser.error(
            "最大连续丢帧次数至少为 1"
        )

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
