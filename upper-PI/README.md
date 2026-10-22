# 上位机代码（upper-PI）

跑在香橙派 5 Pro（RK3588）上的视觉与决策程序。

## 目录结构

```
upper-PI/
├── src/          主程序 + 所有被 import 的模块（平铺，import 直接可用）
├── tests/        单点功能脚本：每个只验证一件事，独立运行
├── tools/        标定/采集工具（分辨率、像素坐标、数据集）
├── models/       RKNN 模型，见 models/README.md
├── legacy/       已被取代的旧文档
├── requirements.txt
├── .env.example  环境变量模板
└── run.sh        一键启动
```

## src/ 文件地图

| 文件 | 行数 | 一句话职责 | 关键出口 |
| --- | --- | --- | --- |
| `full_map_walk_test.py` | 2027 | **主程序**。多线程采集 + 状态机 + 全图覆盖任务 | `python3 src/full_map_walk_test.py` |
| `Serial.py` | 239 | 底盘串口协议封装，所有运动指令都从这里发 | `ChassisController` |
| `line_detect.py` | 704 | 巡线与路口识别（RKNN 道路分割 + 骨架/宽度判断） | `LineTracker.get_road_info(frame) → (状态, 偏差)` |
| `Culvert_detect.py` | 297 | 涵洞检测（亮度一致性 + 边缘纹理），也提供独立自测 | `CulvertBrightnessDetector.inspect(frame) → bool` |
| `yolo_detect.py` | 605 | 侦查视觉：2 类检测模型 + 224 分类模型两级级联 | `YoloDetector.detect_target(frame) → 类别 / -1` |
| `obstacle_color_test.py` | 400+ | 画面下方 ROI 颜色突变 → 判定障碍 | `ColorObstacleDetector` |
| `topo_map.py` | 519 | 赛道拓扑图：节点、边、动作序列、最短路 | `TopoMap.build_competition_track()` |
| `navigator.py` | 399 | 覆盖式任务规划：把每条边走一遍 + 动态涵洞任务 | `Navigator.get_next_segment()` |
| `odometry.py` | 149 | 编码器差速里程计 + RFID 绝对校准 | `Odometry.get_position()` |
| `rfid.py` | 184 | RFID 读卡器帧解析（7F 帧头 + XOR 校验） | `RFIDReader.read_card() → 卡号 / None` |
| `oled.py` | 162 | 小屏状态显示（Tk 全屏窗口 + 子进程） | `OLEDDisplay.show_text()` |
| `broadcast.py` | 136 | 语音播报（pygame 播 mp3，按点号/结果选文件） | `Broadcaster.say_point(id)` |

依赖方向很干净，没有循环引用：

```
full_map_walk_test ─┬─> line_detect ──> (RKNN pp544288)
                    ├─> Culvert_detect ──> Serial
                    ├─> yolo_detect ──> (RKNN 2cls + 224 分类)
                    ├─> obstacle_color_test
                    ├─> Serial ──> 下位机
                    ├─> rfid ──> odometry ──> topo_map
                    ├─> topo_map ──> navigator
                    ├─> oled
                    └─> broadcast
```

## tests/ 文件地图（单点验证，出问题先用它们定位）

| 文件 | 验证什么 | 需要硬件 | 备注 |
| --- | --- | --- | --- |
| `test_line.py` | 巡线效果（离线图片对比 / 实时波形 / 闭环驱动三模式） | 可选 | 调 `kp/kd` 必用 |
| `vision_test.py` | 道路 mask 分支检测算法（只出图，不开车） | 相机 | 2500 行，算法实验场 |
| `yolo_new.py` | 侦查识别（双模型直出 + 多 NPU 核并行 + 数据集评测） | 相机 | 比 `yolo_detect` 新 |
| `test_map.py` | 全图覆盖行走（**旧版**，已被 `src/full_map_walk_test.py` 取代） | 整车 | 留作对照，不建议再改 |
| `main.py` | 整机集成状态机（**旧版**，早期架构） | 整车 | 留作阅读参考 |
| `line_follow_test.py` | 早期巡线闭环 | 整车 | 已被 test_line 取代 |
| `return_start_from_n12_test.py` | 只测"N12 → 出发区"返航 | 整车 | 返航出问题先跑它 |
| `direct_fullframe_models.py` | 全画幅双模型检测封装 | 无 | 被 yolo_new 复用 |
| `test_yolo.py` | 姿态关键点 + 分类两级 | 无 | ⚠️ 缺 `pose_4pt_640x384_fp16.rknn` |
| `test_deeplabv3.py` | 单独跑 DeepLabV3 分割 | 相机 | ⚠️ 缺 `deeplabv3_*_fp16.rknn` |

## tools/ 文件地图

| 文件 | 用途 |
| --- | --- |
| `camera_lower_left.py` | 摄像头画面预览、四角像素测试 |
| `test_pixel.py` | 分辨率与像素坐标标定（选相机型号用） |
| `vision_line_test.py` | 巡线可视化调试（画 mask、骨架、分支） |
| `capture_segmentation_dataset.py` | 采集道路分割数据集（给模型训练用） |
| `camera_resolutions.jpg` | 各分辨率下的实际视场对比图 |

## 怎么运行

```bash
cd upper-PI
pip install -r requirements.txt

# 只算路线，不初始化任何硬件（改完规划器先跑这个）
PYTHONPATH=src python3 src/full_map_walk_test.py --plan-only

# 整车运行（推荐用 run.sh，会先加载 .env）
./run.sh
# 若提示 Permission denied：chmod +x run.sh，或直接 bash run.sh

# 跑单点测试（必须带 PYTHONPATH，因为模块在 src/）
PYTHONPATH=src python3 tests/test_line.py --camera 2 --flip -1
```

> `src/` 平铺的好处是主程序内部 `from Serial import ...` 不用改；
> 代价是外部脚本要 `PYTHONPATH=src`。`run.sh` 已经处理好。

## 调参入口

主程序的**全部**可调参数都在 `src/full_map_walk_test.py` 的 `FullMapConfig` 里，
统一用环境变量覆盖，不需要改代码。最常改的几个：

| 环境变量 | 默认 | 作用 |
| --- | --- | --- |
| `CHASSIS_PORT` / `RFID_PORT` | `/dev/ttyUSB0` / `/dev/ttyACM0` | 串口 |
| `NAV_CAMERA` / `CULVERT_CAMERA` | `2` / `0` | 摄像头编号 |
| `NAV_CAMERA_FLIP` | `-1` | 巡线画面翻转 |
| `FULL_MAP_LINE_SPEED` | `9` | 巡线基础速度 |
| `FULL_MAP_KP` / `FULL_MAP_KD` | `0.5` / `0.10` | 视觉 PID |
| `FULL_MAP_MAX_ANGLE` | `30` | 最大差速限幅 |
| `FULL_MAP_ACTION_TIMEOUT` | `10` | 单个动作等 `@` 的超时（s） |
| `OBSTACLE_DETECT_ENABLED` | `1` | 障碍检测开关 |
| `CULVERT_SAVE_PHOTOS` | `1` | 侦查照片是否落盘 |
| `CULVERT_EXPECTED_COUNT` | `8` | 本场涵洞数量 |
| `LINE_DEBUG` | `0` | 巡线调试图窗 |

完整清单（40+ 项）直接看 `FullMapConfig` 类，每项都有中文注释。
