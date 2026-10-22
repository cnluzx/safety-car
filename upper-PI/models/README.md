# RKNN 模型清单

所有模型是 **RKNN 格式**（Rockchip NPU 专用），只能在 RK3588/RK3568 上用 `rknnlite` 加载。
输入尺寸是按板端实测帧率定的，不要随意改大——NPU 撑不住会掉帧到巡线失控。

## 在用模型

| 文件 | 输入尺寸 | 用途 | 谁在用 | 大小 |
| --- | --- | --- | --- | --- |
| `pp544288.rknn` | 544×288 | **道路分割**（PP-LiteSeg），输出可行驶区域 mask，巡线与路口判断的基础 | `src/line_detect.py`、`tests/test_line.py` | 16.6 MB |
| `2cls_640x384_fp16.rknn` | 640×384 | **两级检测的第 1 级**：在整幅画面里找"人 / 刀"候选框（2 类） | `src/yolo_detect.py`、`tests/yolo_new.py` | 6.7 MB |
| `person_classifier_224_fp16.rknn` | 224×224 | 第 2 级分类：候选框是否真的是人 | `src/yolo_detect.py` 两级模式 | 3.9 MB |
| `knife_classifier_224_fp16.rknn` | 224×224 | 第 2 级分类：候选框是否真的是刀 | `src/yolo_detect.py` 两级模式 | 3.9 MB |
| `person_new1.rknn` | 全画幅（约 640×384） | 端到端人检测（省掉第 2 级，速度快） | `tests/yolo_new.py` | 41.6 MB |
| `knife_new1.rknn` | 全画幅 | 端到端刀检测 | `tests/yolo_new.py` | 41.6 MB |

**两条视觉路线**，按需选：

- **两级级联**（`yolo_detect.py`，主程序用）：第 1 级粗筛框 → 第 2 级 224 分类确认。
  模型小、总延迟低，但依赖第 1 级不漏检。
- **全画幅直出**（`yolo_new.py`）：一个模型同时出框和类别，省一次推理，精度更依赖模型本身。

## 模型路径约定

代码统一按 `<项目根>/models/<文件名>` 解析（例如 `src/line_detect.py` 里
`Path(__file__).parent.parent / "models" / "pp544288.rknn"`），不需要改路径。
也可以用环境变量指向别处：

| 环境变量 | 作用 |
| --- | --- |
| `DEEPLAB_MODEL` | 覆盖道路分割模型路径 |
| `YOLO_PERSON_MODEL` / `YOLO_KNIFE_MODEL` | 覆盖全画幅模型文件名 |
| `YOLO_PERSON_CLASSIFIER` / `YOLO_KNIFE_CLASSIFIER` | 覆盖 224 分类模型路径 |

## archive/ 说明（不在用，仅存档）

| 文件 | 为什么留在这 |
| --- | --- |
| `person_new.rknn` / `knife_new.rknn` | 全画幅模型的上一版，已被 `*_new1` 取代，代码不再引用 |
| `pp_liteseg_80x46_fp16.rknn` | 最早的道路分割模型（80×46 超低分辨率），已被 `pp544288` 取代 |

这三个共约 100 MB，**如果空间紧张可以直接删**，不影响任何代码运行。

## 缺失的模型（代码引用但仓库里没有）

| 文件 | 谁需要 | 说明 |
| --- | --- | --- |
| `pose_4pt_640x384_fp16.rknn` | `tests/test_yolo.py` | 4 点姿态关键点检测，早期方案 |
| `deeplabv3_mobilenet_roboflow_voc_640x480_native_fp16.rknn` | `tests/test_deeplabv3.py` | 早期 DeepLabV3 分割对照 |

这两条是探索性方案，没进最终比赛流程。想复现需要自己用 RKNN-Toolkit2 重新转换
（训练权重没随仓库提供）。跑不起来是正常的，不是环境问题。

## 换模型怎么换（经验）

1. PC 端用 RKNN-Toolkit2 转换：`target_platform='rk3588'`，`quantized_dtype='w8a8'`，
   **`do_quantization=False` 先验证精度**，再量化对比。
2. 板端先跑 `tests/test_line.py`（或 `tests/yolo_new.py`）用录好的图片做离线对比，
   别直接上车。
3. 关注板端实际 FPS：巡线需要 ≥ 25 FPS，低于 20 FPS 会明显画龙。
   降分辨率比降帧率好——巡线是闭环，延迟比精度更致命。
