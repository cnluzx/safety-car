"""面向未知绿色涵洞布局的动态全路段覆盖导航器。"""

from __future__ import annotations

import threading


class Navigator:
    """巡检所有道路、发现涵洞任务、绕开障碍物并返回出发区。"""

    def __init__(self, topo_map, odometry=None):
        self.map = topo_map
        self.odometry = odometry
        self.lock = threading.RLock()
        self.mission_plan = []
        self.visited_patrols = set()
        self.expected_green_count = 8
        self.discovered_green_ids = set()
        self.required_edges = set()
        # 某些支路只适合在最终返航时覆盖。只要还有其他未覆盖道路，
        # 最近边规划就暂时不选择这些边。
        self.deferred_coverage_edges = set()
        self.covered_edges = set()
        self.blocked_edges = set()
        # 实体障碍使整条边不可穿越，但两端仍需分别靠近检查。这里保留方向，
        # 例如 (A, B) 表示已从 A 端朝 B 端驶近障碍并搜索过这一侧。
        self.inspected_blocked_approaches = set()
        self.current_node = None
        self.previous_node = None
        self.current_path = None
        self.path_index = 0
        self.start_node = None
        self.mission_complete = False
        self.all_patrols_visited = False
        self.coverage_complete = False
        self.sensor_miss = False

    def init_mission(self, start_node="START_AREA", mission_plan=None,
                     expected_green_count=8, cover_all_edges=True):
        with self.lock:
            if start_node not in self.map.nodes:
                raise KeyError(f"unknown start node: {start_node}")
            # 绿色涵洞位置刻意不放入任务列表，而是在巡检物理道路时发现。
            self.mission_plan = list(
                mission_plan if mission_plan is not None
                else self.map.fixed_patrol_nodes
            )
            unknown = set(self.mission_plan).difference(self.map.nodes)
            if unknown:
                raise KeyError(f"unknown mission nodes: {sorted(unknown)}")
            self.expected_green_count = int(expected_green_count)
            self.visited_patrols.clear()
            self.discovered_green_ids.clear()
            self.required_edges = (
                set(self.map.get_coverage_edges()) if cover_all_edges else set()
            )
            self.deferred_coverage_edges.clear()
            self.covered_edges.clear()
            self.blocked_edges.clear()
            self.inspected_blocked_approaches.clear()
            self.current_node = start_node
            self.previous_node = None
            self.current_path = None
            self.path_index = 0
            self.start_node = start_node
            self.mission_complete = False
            self.all_patrols_visited = False
            self.coverage_complete = not self.required_edges
            self.sensor_miss = False

    def defer_coverage_edges(self, edges):
        """将指定覆盖边延后到其他道路全部完成之后。"""
        with self.lock:
            normalized = {self.map.edge_key(*edge) for edge in edges}
            unknown = normalized.difference(self.required_edges)
            if unknown:
                raise ValueError(f"cannot defer non-coverage edges: {sorted(unknown)}")
            self.deferred_coverage_edges = normalized
            self.current_path = None
            self.path_index = 0

    def _resolve_mission_node(self, patrol_id):
        if isinstance(patrol_id, str) and patrol_id in self.map.nodes:
            return patrol_id
        node = self.map.get_node_by_rfid(patrol_id)
        return node.id if node else None

    def mark_culvert_scanned(self, task_id):
        """在视觉触发的涵洞扫描完成后记录一个涵洞任务。"""
        with self.lock:
            normalized = str(task_id).strip()
            if not normalized:
                return False
            before = len(self.discovered_green_ids)
            self.discovered_green_ids.add(normalized)
            self._update_completion_flags()
            return len(self.discovered_green_ids) > before

    def mark_green_found(self, task_id):
        """为仿真和旧调用方保留的兼容别名。"""
        return self.mark_culvert_scanned(task_id)

    def mark_culvert_edge(self, from_node, to_node):
        """将涵洞检测结果绑定到转向完成后进入的实际道路。"""
        with self.lock:
            edge = self.map.mark_culvert_edge(from_node, to_node)
            self.mark_culvert_scanned("edge:" + "->".join(edge))
            return edge

    def confirm_fixed_patrol(self, node_id):
        """将带 RFID 的固定点作为绝对拓扑位置锚点。

        若卡片正好是规划中的下一节点，则正常推进路径以保留道路覆盖记录；
        若读到有效但意外的卡片，则将车辆重定位到该固定点并在线重新规划。
        """
        with self.lock:
            if node_id not in self.mission_plan:
                return False

            expected = None
            if self.current_path and self.path_index < len(self.current_path) - 1:
                expected = self.current_path[self.path_index + 1]
            if expected == node_id:
                return self.advance_path() == node_id

            old_node = self.current_node
            if old_node and self.map.get_edge(old_node, node_id):
                traversed = self.map.edge_key(old_node, node_id)
                if traversed in self.required_edges:
                    self.covered_edges.add(traversed)

            self.previous_node = old_node
            self.current_node = node_id
            self.visited_patrols.add(node_id)
            self.current_path = None
            self.path_index = 0
            self._update_completion_flags()
            return True

    def mark_visited(self, patrol_id):
        """兼容旧接口的固定节点确认方法。"""
        with self.lock:
            node_id = self._resolve_mission_node(patrol_id)
            if node_id is None:
                return False
            self.previous_node = self.current_node
            self.current_node = node_id
            if node_id in self.mission_plan:
                self.visited_patrols.add(node_id)
            self.current_path = None
            self.path_index = 0
            self._update_completion_flags()
            return True

    def mark_position(self, node_id):
        with self.lock:
            if node_id not in self.map.nodes:
                return False
            self.previous_node = self.current_node
            self.current_node = node_id
            if node_id in self.mission_plan:
                self.visited_patrols.add(node_id)
            self._update_completion_flags()
            return True

    def on_obstacle(self, from_node=None, to_node=None):
        with self.lock:
            from_node = from_node or self.current_node
            if to_node is None and self.current_path:
                if self.path_index + 1 < len(self.current_path):
                    to_node = self.current_path[self.path_index + 1]
            if not from_node or not to_node:
                return False
            blocked_edge = self.map.edge_key(from_node, to_node)
            approach = self.map.directed_edge_key(from_node, to_node)
            self.blocked_edges.add(blocked_edge)
            self.inspected_blocked_approaches.add(approach)
            self.current_path = None
            self.path_index = 0
            self._update_completion_flags()
            path = self._replan_to_target()
            if path:
                self.current_path = path
                return True
            return False

    def add_blocked_edges(self, edges):
        with self.lock:
            self.blocked_edges.update(self.map.edge_key(*edge) for edge in edges)
            self.current_path = None
            self.path_index = 0
            self._update_completion_flags()

    def _blocked_edge_fully_inspected(self, edge):
        a, b = edge
        approaches = self.inspected_blocked_approaches
        return (
            self.map.directed_edge_key(a, b) in approaches
            and self.map.directed_edge_key(b, a) in approaches
        )

    def _completed_coverage_edges(self):
        """返回已穿越道路及已从两端检查完毕的障碍道路。"""
        completed = set(self.covered_edges)
        completed.update(
            edge for edge in self.blocked_edges
            if self._blocked_edge_fully_inspected(edge)
        )
        return completed

    def _remaining_coverage_edges(self):
        # 障碍道路不能直接从覆盖任务中删除；只有两端均靠近检查后才算完成，
        # 否则障碍对面可能存在的未知涵洞会被永久漏掉。
        return self.required_edges - self._completed_coverage_edges()

    def _update_completion_flags(self):
        fixed_complete = set(self.mission_plan).issubset(self.visited_patrols)
        self.coverage_complete = not self._remaining_coverage_edges()
        green_complete = (
            len(self.discovered_green_ids) >= self.expected_green_count
        )
        self.all_patrols_visited = (
            fixed_complete and self.coverage_complete and green_complete
        )
        self.mission_complete = (
            self.all_patrols_visited and self.current_node == self.start_node
        )
        self.sensor_miss = self.coverage_complete and not green_complete

    def _path_to_nearest_uncovered_edge(self):
        choices = []
        remaining_edges = self._remaining_coverage_edges()
        regular_edges = remaining_edges.difference(self.deferred_coverage_edges)
        candidate_edges = regular_edges if regular_edges else remaining_edges
        for a, b in sorted(candidate_edges):
            edge = self.map.get_edge(a, b) or self.map.get_edge(b, a)
            if not edge:
                continue
            approaches = ((a, b), (b, a))
            edge_key = self.map.edge_key(a, b)
            if edge_key in self.blocked_edges:
                approaches = tuple(
                    (entry, exit_node)
                    for entry, exit_node in approaches
                    if self.map.directed_edge_key(entry, exit_node)
                    not in self.inspected_blocked_approaches
                )
            for entry, exit_node in approaches:
                path, distance = self.map.shortest_path(
                    self.current_node, entry, self.blocked_edges,
                )
                if path:
                    # 对已知障碍边，仅把车辆规划到未检查的一端并驶向障碍；
                    # 实车会再次触发 on_obstacle，随后掉头回到该入口节点。
                    inspection_cost = (
                        edge.length * 0.5
                        if edge_key in self.blocked_edges else edge.length
                    )
                    choices.append((
                        distance + inspection_cost,
                        edge_key,
                        entry,
                        path + [exit_node],
                    ))
        if not choices:
            return None
        return min(choices, key=lambda item: (item[0], item[1], item[2]))[3]

    def _replan_to_target(self):
        # 全路段覆盖优先，因为未知涵洞可能位于任何尚未巡检的道路上，
        # 包括到达编号固定点时不必经过的道路。
        coverage_path = self._path_to_nearest_uncovered_edge()
        if coverage_path:
            return coverage_path

        remaining_nodes = set(self.mission_plan).difference(self.visited_patrols)
        if remaining_nodes:
            choices = []
            for target in remaining_nodes:
                path, distance = self.map.shortest_path(
                    self.current_node, target, self.blocked_edges,
                )
                if path:
                    choices.append((distance, target, path))
            if choices:
                return min(choices, key=lambda item: (item[0], item[1]))[2]

        if self.current_node != self.start_node:
            return self.map.find_path(
                self.current_node, self.start_node, self.blocked_edges,
            )
        self._update_completion_flags()
        return None

    def get_next_segment(self):
        with self.lock:
            if self.mission_complete:
                return None
            if self.current_path and self.path_index < len(self.current_path) - 1:
                return self.current_path[self.path_index:]
            self.current_path = self._replan_to_target()
            self.path_index = 0
            return list(self.current_path) if self.current_path else None

    def get_next_action(self):
        with self.lock:
            path = self.get_next_segment()
            if not path:
                return ("STOP", None) if self.mission_complete else ("WAIT", None)
            next_node = path[1]
            action = (
                self.map.turn_action(self.previous_node, self.current_node, next_node)
                if self.previous_node else self.map.ACT_FOLLOW_LINE
            )
            return action, next_node

    def advance_path(self):
        with self.lock:
            if not self.current_path or self.path_index >= len(self.current_path) - 1:
                return None
            from_node = self.current_node
            self.previous_node = from_node
            self.path_index += 1
            self.current_node = self.current_path[self.path_index]
            traversed = self.map.edge_key(from_node, self.current_node)
            if traversed in self.required_edges:
                self.covered_edges.add(traversed)
            if self.current_node in self.mission_plan:
                self.visited_patrols.add(self.current_node)
            self._update_completion_flags()
            if self.path_index >= len(self.current_path) - 1:
                self.current_path = None
                self.path_index = 0
            return self.current_node

    def preview_after_advance(self):
        """Preview arrival and replanning without committing navigation state.

        A physical junction action needs the route beyond the node being
        approached, but the logical position must not advance until that
        action has completed.  Temporarily advancing here lets the normal
        replanner choose that next segment while leaving the real navigator
        unchanged for the later completion commit.
        """
        with self.lock:
            snapshot = {
                "current_node": self.current_node,
                "previous_node": self.previous_node,
                "current_path": (
                    list(self.current_path) if self.current_path else None
                ),
                "path_index": self.path_index,
                "visited_patrols": set(self.visited_patrols),
                "covered_edges": set(self.covered_edges),
                "coverage_complete": self.coverage_complete,
                "all_patrols_visited": self.all_patrols_visited,
                "mission_complete": self.mission_complete,
                "sensor_miss": self.sensor_miss,
            }
            try:
                reached = self.advance_path()
                path = self.get_next_segment() if reached else None
                return (
                    reached,
                    list(path) if path else None,
                    self.mission_complete,
                )
            finally:
                for name, value in snapshot.items():
                    setattr(self, name, value)

    def get_progress(self):
        with self.lock:
            covered = self._completed_coverage_edges().intersection(
                self.required_edges
            )
            return {
                "visited": len(self.visited_patrols),
                "total": len(self.mission_plan),
                "missing": sorted(set(self.mission_plan) - self.visited_patrols),
                "green_found": len(self.discovered_green_ids),
                "green_expected": self.expected_green_count,
                "green_ids": sorted(self.discovered_green_ids),
                "covered_edges": len(covered),
                "required_edges": len(self.required_edges),
                "coverage_percent": round(
                    100.0 * len(covered) / len(self.required_edges), 1
                ) if self.required_edges else 100.0,
                "uncovered_edges": sorted(self._remaining_coverage_edges()),
                "current": self.current_node,
                "blocked_edges": len(self.blocked_edges),
                "inspected_blocked_approaches": sorted(
                    self.inspected_blocked_approaches
                ),
                "coverage_complete": self.coverage_complete,
                "sensor_miss": self.sensor_miss,
                "all_points_visited": self.all_patrols_visited,
                "complete": self.mission_complete,
            }
