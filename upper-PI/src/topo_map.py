"""竞赛赛道的拓扑建模与任务规划。

物理赛道独立于八个绿色涵洞的当前位置建模。仿真时，绿色点会插入随机
直线路段，成为真实图节点，从而确保路径规划不会跳过它。
"""

from __future__ import annotations

import heapq
import json
import math
import random


class MapNode:
    def __init__(self, node_id, node_type, rfid_id=None, position=None,
                 requires_culvert_scan=False):
        self.id = node_id
        self.type = node_type
        self.rfid_id = rfid_id
        self.position = tuple(position) if position is not None else None
        self.requires_culvert_scan = bool(requires_culvert_scan)

    def __repr__(self):
        return f"MapNode({self.id}, {self.type})"


class MapEdge:
    def __init__(self, from_id, to_id, actions, expected_heading=0,
                 expected_dist=0, bypass=None, length=None):
        self.from_id = from_id
        self.to_id = to_id
        self.actions = list(actions)
        self.expected_heading = expected_heading
        self.expected_dist = expected_dist
        self.bypass = bypass or []
        self.length = float(length if length is not None else
                            (expected_dist if expected_dist else 1.0))

    def __repr__(self):
        return f"MapEdge({self.from_id} -> {self.to_id}, length={self.length:.2f})"


class TopoMap:
    NODE_START = "START"
    NODE_FIXED_PATROL = "FIXED_PATROL"
    NODE_PATROL = "GREEN_TASK"       # 为旧代码保留的兼容名称
    NODE_GREEN_TASK = "GREEN_TASK"
    NODE_CROSS = "CROSSROAD"
    # Compatibility constant only. T1-T4 are tunnel identifiers attached to
    # physical edges; they are not graph nodes.
    NODE_TUNNEL = "TUNNEL"
    NODE_STOP = "STOP"

    TUNNEL_IDS = frozenset({"T1", "T2", "T3", "T4"})
    TUNNEL_SEGMENTS = {
        frozenset(("I1_1", "I12_1")): "T1",
        frozenset(("I1_2", "I12_2")): "T2",
        frozenset(("I1_3", "I12_3")): "T3",
        frozenset(("N6", "N7")): "T4",
    }

    ACT_FOLLOW_LINE = "FOLLOW_LINE"
    ACT_FORWARD = "FORWARD"
    ACT_LEFT = "LEFT"
    ACT_RIGHT = "RIGHT"
    ACT_U_TURN = "U_TURN"
    ACT_STOP = "STOP"
    ACT_FOLLOW_CROSS = "FOLLOW_CROSS"

    # 根据参考地图读取的八个绿色涵洞位置，仅用于仿真真值；真实涵洞不带
    # RFID，由上位机视觉识别。
    REFERENCE_GREEN_EDGES = [
        ("I1_1", "I1_2"),   # 上部中央左侧竖向道路
        ("N3", "N4"),       # 左侧外圈竖向道路
        ("N4", "I1_3"),     # 第三行左侧横向道路
        ("I1_3", "N6"),     # 下部中央左侧竖向道路
        ("I12_2", "I12_3"), # 中部中央右侧竖向道路
        ("I12_1", "N11"),   # 第一行右侧横向道路
        ("N10", "N9"),      # 右侧外圈竖向道路
        ("N7", "N8"),       # 底部右侧横向道路
    ]

    # 参考地图中的物理道路。在线规划器知道道路连通关系，但不知道其中哪八条
    # 带有绿色涵洞任务。
    BASE_EDGES = [
        ("START_AREA", "J_TOP"),
        ("J_TOP", "N1"), ("J_TOP", "N12"),
        ("N2", "N3"), ("N3", "N4"), ("N4", "N5"),
        ("N1", "I1_1"), ("I1_1", "I1_2"),
        ("I1_2", "I1_3"), ("I1_3", "N6"),
        ("N12", "I12_1"), ("I12_1", "I12_2"),
        ("I12_2", "I12_3"), ("I12_3", "N7"),
        ("N11", "N10"), ("N10", "N9"), ("N9", "N8"),
        ("N2", "I1_1"), ("I1_1", "I12_1"),
        ("I12_1", "N11"),
        ("N3", "I1_2"), ("I1_2", "I12_2"),
        ("I12_2", "N10"),
        ("N4", "I1_3"), ("I1_3", "I12_3"),
        ("I12_3", "N9"),
        ("N5", "N6"), ("N6", "N7"), ("N7", "N8"),
    ]

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.fixed_patrol_nodes = []
        self.green_task_nodes = []
        self.green_placements = []
        self.culvert_edges = set()
        self.map_seed = None

    @staticmethod
    def edge_key(a, b):
        """返回与行驶方向无关的道路边键。"""
        return tuple(sorted((a, b)))

    @staticmethod
    def directed_edge_key(from_id, to_id):
        """返回保留接近方向的边键，用于记录障碍物两侧的巡检状态。"""
        return from_id, to_id

    def is_configured_culvert_edge(self, from_id, to_id):
        """普通道路均可能存在涵洞；T1-T4 隧道路段不触发视觉识别。"""
        if "START_AREA" in (from_id, to_id):
            return False
        return self.tunnel_id_for_segment(from_id, to_id) is None

    def tunnel_id_for_segment(self, from_id, to_id):
        """返回物理道路对应的隧道编号；普通道路返回 ``None``。"""
        return self.TUNNEL_SEGMENTS.get(frozenset((from_id, to_id)))

    def is_tunnel_segment(self, from_id, to_id):
        return self.tunnel_id_for_segment(from_id, to_id) is not None

    def add_node(self, node_id, node_type, rfid_id=None, position=None,
                 requires_culvert_scan=False):
        self.nodes[node_id] = MapNode(
            node_id, node_type, rfid_id, position, requires_culvert_scan
        )
        return self

    def _distance(self, from_id, to_id):
        a = self.nodes[from_id].position
        b = self.nodes[to_id].position
        if a is None or b is None:
            return 1.0
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def add_edge(self, from_id, to_id, actions=None, expected_heading=0,
                 expected_dist=0, bypass=None, length=None):
        if from_id not in self.nodes or to_id not in self.nodes:
            raise KeyError(f"edge contains unknown node: {from_id} -> {to_id}")
        if length is None:
            length = expected_dist or self._distance(from_id, to_id)
        edge = MapEdge(
            from_id, to_id, actions or [self.ACT_FOLLOW_LINE],
            expected_heading, expected_dist, bypass, length,
        )
        self.edges.setdefault(from_id, []).append(edge)
        return self

    def add_bidirectional_edge(self, a, b, actions=None, length=None):
        self.add_edge(a, b, actions, length=length)
        self.add_edge(b, a, actions, length=length)
        return self

    def clear(self):
        self.nodes.clear()
        self.edges.clear()
        self.fixed_patrol_nodes = []
        self.green_task_nodes = []
        self.green_placements = []
        self.culvert_edges = set()

    def mark_culvert_edge(self, from_id, to_id):
        """记录已确认可能存在涵洞的实际道路边。"""
        key = self.edge_key(from_id, to_id)
        self.culvert_edges.add(key)
        return key

    def _blocked_keys(self, blocked_edges):
        return {self.edge_key(*edge) for edge in (blocked_edges or set())}

    def shortest_path(self, start_id, target_id, blocked_edges=None):
        """使用 Dijkstra 求最短路；有实体障碍的道路不可用于穿越。

        ``blocked_edges`` 仍使用无向键，因为实体障碍会同时阻止道路的两个
        方向。是否已从道路某一端靠近并完成搜索，由 ``Navigator`` 单独记录，
        不应混入最短路的可通行性判断。
        """
        if start_id not in self.nodes or target_id not in self.nodes:
            return None, math.inf
        blocked = self._blocked_keys(blocked_edges)
        distances = {start_id: 0.0}
        parents = {}
        queue = [(0.0, start_id)]

        while queue:
            distance, current = heapq.heappop(queue)
            if distance != distances.get(current):
                continue
            if current == target_id:
                path = [current]
                while current in parents:
                    current = parents[current]
                    path.append(current)
                path.reverse()
                return path, distance

            for edge in self.edges.get(current, []):
                if self.edge_key(current, edge.to_id) in blocked:
                    continue
                candidate = distance + edge.length
                if candidate < distances.get(edge.to_id, math.inf):
                    distances[edge.to_id] = candidate
                    parents[edge.to_id] = current
                    heapq.heappush(queue, (candidate, edge.to_id))
        return None, math.inf

    def find_path(self, start_id, target_id, blocked_edges=None):
        return self.shortest_path(start_id, target_id, blocked_edges)[0]

    def get_actions_between(self, from_id, to_id):
        edge = self.get_edge(from_id, to_id)
        return list(edge.actions) if edge else None

    def get_edge(self, from_id, to_id):
        for edge in self.edges.get(from_id, []):
            if edge.to_id == to_id:
                return edge
        return None

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def get_node_by_rfid(self, rfid_id):
        try:
            normalized = int(rfid_id)
        except (TypeError, ValueError):
            normalized = rfid_id
        for node in self.nodes.values():
            if node.rfid_id == normalized:
                return node
        return None

    def get_next_node_in_path(self, current_node_id, path):
        if current_node_id in path:
            index = path.index(current_node_id)
            if index + 1 < len(path):
                return path[index + 1]
        return None

    def get_node_by_rfid_in_path(self, rfid_id, path):
        node = self.get_node_by_rfid(rfid_id)
        if node and node.id in path:
            return node.id, self.get_next_node_in_path(node.id, path)
        return None, None

    def build_default_map(self):
        """供模块级冒烟测试使用的小型地图。"""
        self.clear()
        self.add_node("S", self.NODE_START, position=(0, 0))
        self.add_node("P1", self.NODE_GREEN_TASK, rfid_id=1, position=(1, 0))
        self.add_node("C1", self.NODE_CROSS, position=(2, 0))
        self.add_node("E", self.NODE_STOP, position=(3, 0))
        self.add_bidirectional_edge("S", "P1")
        self.add_bidirectional_edge("P1", "C1")
        self.add_bidirectional_edge("C1", "E")
        self.green_task_nodes = ["P1"]
        return self

    def build_competition_track(self, seed=None, green_count=8,
                                green_edges=None):
        """建立 4×4 竞赛地图，并在仿真中放置绿色任务点。

        当位置来自真实参考地图时，``green_edges`` 可传入节点对列表；若省略，
        则从 ``seed`` 确定性地选取 ``green_count`` 条合法直线路段。
        """
        self.clear()
        self.map_seed = seed
        positions = {
            "START_AREA": (1.5, 5.0),
            "J_TOP": (1.5, 4.0),
            "N1": (1.0, 4.0), "N12": (2.0, 4.0),
            "N2": (0.0, 3.0), "I1_1": (1.0, 3.0),
            "I12_1": (2.0, 3.0), "N11": (3.0, 3.0),
            "N3": (0.0, 2.0), "I1_2": (1.0, 2.0),
            "I12_2": (2.0, 2.0), "N10": (3.0, 2.0),
            "N4": (0.0, 1.0), "I1_3": (1.0, 1.0),
            "I12_3": (2.0, 1.0), "N9": (3.0, 1.0),
            "N5": (0.0, 0.0), "N6": (1.0, 0.0),
            "N7": (2.0, 0.0), "N8": (3.0, 0.0),
        }

        for node_id, position in positions.items():
            if node_id == "START_AREA":
                node_type = self.NODE_START
            elif node_id.startswith("N"):
                node_type = self.NODE_FIXED_PATROL
            else:
                node_type = self.NODE_CROSS
            rfid_id = int(node_id[1:]) if node_type == self.NODE_FIXED_PATROL else None
            self.add_node(
                node_id, node_type, rfid_id=rfid_id, position=position,
            )

        self.fixed_patrol_nodes = [f"N{i}" for i in range(1, 13)]

        base_edges = self.BASE_EDGES

        # 任务点不放在隧道路段或出发区内。
        eligible = [
            edge for edge in base_edges
            if "START_AREA" not in edge and not self.is_tunnel_segment(*edge)
        ]
        eligible_keys = {self.edge_key(*edge): edge for edge in eligible}
        rng = random.Random(seed)
        if green_edges is None:
            if not 0 <= green_count <= len(eligible):
                raise ValueError(f"green_count must be 0..{len(eligible)}")
            selected = rng.sample(eligible, green_count)
        else:
            selected = []
            for edge in green_edges:
                key = self.edge_key(*edge)
                if key not in eligible_keys:
                    raise ValueError(f"invalid green task segment: {edge}")
                selected.append(eligible_keys[key])
            if len({self.edge_key(*edge) for edge in selected}) != len(selected):
                raise ValueError("green task segments must be unique")

        placement_by_edge = {
            self.edge_key(*edge): index
            for index, edge in enumerate(selected, start=1)
        }
        for a, b in base_edges:
            task_index = placement_by_edge.get(self.edge_key(a, b))
            if task_index is None:
                self.add_bidirectional_edge(a, b)
                continue
            task_id = f"P{task_index}"
            pa, pb = self.nodes[a].position, self.nodes[b].position
            position = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
            self.add_node(task_id, self.NODE_GREEN_TASK, position=position)
            self.add_bidirectional_edge(a, task_id)
            self.add_bidirectional_edge(task_id, b)
            self.green_task_nodes.append(task_id)
            self.green_placements.append({
                "task": task_id, "between": (a, b), "position": position,
            })
        self.green_task_nodes.sort(key=lambda n: int(n[1:]))
        self.green_placements.sort(key=lambda item: int(item["task"][1:]))
        return self

    def get_green_candidate_edges(self):
        """允许出现未知绿色涵洞任务的道路集合。"""
        return [
            edge for edge in self.BASE_EDGES
            if "START_AREA" not in edge and not self.is_tunnel_segment(*edge)
        ]

    def make_hidden_green_placements(self, green_edges):
        """描述测试布局，但不将其位置暴露给导航器。"""
        placements = []
        candidates = {self.edge_key(*edge) for edge in self.get_green_candidate_edges()}
        for index, (a, b) in enumerate(green_edges, start=1):
            if self.edge_key(a, b) not in candidates:
                raise ValueError(f"invalid hidden green segment: {(a, b)}")
            pa, pb = self.nodes[a].position, self.nodes[b].position
            placements.append({
                "task": f"P{index}",
                "between": (a, b),
                "position": ((pa[0] + pb[0]) / 2.0,
                             (pa[1] + pb[1]) / 2.0),
            })
        return placements

    def random_hidden_green_placements(self, count=8, seed=None):
        candidates = self.get_green_candidate_edges()
        if not 0 <= count <= len(candidates):
            raise ValueError(f"green count must be 0..{len(candidates)}")
        selected = random.Random(seed).sample(candidates, count)
        return self.make_hidden_green_placements(selected)

    def get_coverage_edges(self):
        """为发现未知任务而必须巡检的全部道路。"""
        return {
            self.edge_key(edge.from_id, edge.to_id)
            for edge_list in self.edges.values() for edge in edge_list
            if "START_AREA" not in (edge.from_id, edge.to_id)
        }

    def get_mission_plan(self):
        """所有任务点：12 个固定编号点加上当前 8 个绿色任务点。"""
        return list(self.fixed_patrol_nodes) + list(self.green_task_nodes)

    def plan_visit_all(self, start_id="START_AREA", required_nodes=None,
                       blocked_edges=None, return_to_start=True):
        """规划一条避开障碍物并访问所有要求节点的较短路线。

        下一个目标选择最近的可达未访问节点。行驶途中经过的要求节点会立即
        计入完成，避免仅因其较晚出现在目标列表中而被重复访问。
        """
        required = set(required_nodes or self.get_mission_plan())
        unknown = required.difference(self.nodes)
        if unknown:
            raise KeyError(f"unknown mission nodes: {sorted(unknown)}")
        route = [start_id]
        visit_order = []
        current = start_id

        while required:
            choices = []
            for target in required:
                path, distance = self.shortest_path(current, target, blocked_edges)
                if path:
                    choices.append((distance, target, path))
            if not choices:
                return None, visit_order
            _, _, path = min(choices, key=lambda item: (item[0], item[1]))
            route.extend(path[1:])
            for node_id in path:
                if node_id in required:
                    required.remove(node_id)
                    visit_order.append(node_id)
            current = route[-1]

        if return_to_start and current != start_id:
            path = self.find_path(current, start_id, blocked_edges)
            if not path:
                return None, visit_order
            route.extend(path[1:])
        return route, visit_order

    def random_obstacles(self, count=3, seed=None, start_id="START_AREA",
                         required_nodes=None, excluded_edges=None):
        """随机选择道路作为障碍物，同时保证所有任务点仍然可达。"""
        required = required_nodes or self.get_mission_plan()
        excluded = {self.edge_key(*edge) for edge in (excluded_edges or [])}
        candidates = sorted({
            self.edge_key(edge.from_id, edge.to_id)
            for edge_list in self.edges.values() for edge in edge_list
            if start_id not in (edge.from_id, edge.to_id)
            and not self.is_tunnel_segment(edge.from_id, edge.to_id)
            and self.edge_key(edge.from_id, edge.to_id) not in excluded
        })
        random.Random(seed).shuffle(candidates)
        blocked = set()
        for edge in candidates:
            trial = blocked | {edge}
            if all(self.find_path(start_id, target, trial) for target in required):
                blocked = trial
                if len(blocked) == count:
                    return blocked
        raise RuntimeError(f"could not place {count} non-disconnecting obstacles")

    def turn_action(self, previous_id, current_id, next_id):
        """根据节点坐标计算车辆在 ``current_id`` 处应执行的转向。"""
        if not previous_id or previous_id == current_id:
            return self.ACT_FORWARD
        p0 = self.nodes[previous_id].position
        p1 = self.nodes[current_id].position
        p2 = self.nodes[next_id].position
        if None in (p0, p1, p2):
            return self.ACT_FORWARD
        incoming = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        outgoing = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        delta = (math.degrees(outgoing - incoming) + 180) % 360 - 180
        if abs(delta) < 35:
            return self.ACT_FORWARD
        if abs(delta) > 145:
            return self.ACT_U_TURN
        return self.ACT_LEFT if delta > 0 else self.ACT_RIGHT

    def export_json(self, path):
        data = {
            "seed": self.map_seed,
            "fixed_patrol_nodes": self.fixed_patrol_nodes,
            "green_task_nodes": self.green_task_nodes,
            "green_placements": self.green_placements,
            "culvert_edges": [list(edge) for edge in sorted(self.culvert_edges)],
            "nodes": {
                node_id: {"type": node.type, "rfid_id": node.rfid_id,
                          "position": node.position,
                          "requires_culvert_scan": node.requires_culvert_scan}
                for node_id, node in self.nodes.items()
            },
            "edges": {
                from_id: [
                    {"to_id": edge.to_id, "actions": edge.actions,
                     "length": edge.length}
                    for edge in edge_list
                ]
                for from_id, edge_list in self.edges.items()
            },
        }
        with open(path, "w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False, indent=2)

    def pretty_print(self):
        print("Competition topology")
        print(f"  fixed patrols ({len(self.fixed_patrol_nodes)}): "
              f"{' '.join(self.fixed_patrol_nodes)}")
        print(f"  green tasks  ({len(self.green_task_nodes)}): "
              f"{' '.join(self.green_task_nodes)}")
        for placement in self.green_placements:
            a, b = placement["between"]
            print(f"    {placement['task']}: {a} -- {b}")


if __name__ == "__main__":
    track = TopoMap().build_competition_track(seed=20260715)
    obstacles = track.random_obstacles(count=3, seed=20260716)
    route, order = track.plan_visit_all(blocked_edges=obstacles)
    track.pretty_print()
    print(f"  obstacles: {sorted(obstacles)}")
    print(f"  visit order: {' -> '.join(order)}")
    print(f"  route nodes: {len(route) if route else 0}")
