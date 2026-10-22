import argparse
import signal
import time
import os
import sys
import threading
import numpy as np
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

import cv2


PERSON_CLASS_INDEX_TO_BUSINESS_ID = (1, 10, 2, 3, 4, 5, 6, 7, 8, 9)
PERSON_NUM_CLASSES = len(PERSON_CLASS_INDEX_TO_BUSINESS_ID)
PERSON_NAMES = [f"person_{bid}" for bid in PERSON_CLASS_INDEX_TO_BUSINESS_ID]

# 权重里的类别索引按 data.yaml 的 names 顺序排，而 Roboflow 导出时是字符串排序，
# 所以索引 1 对应的刀号是 10，不是 2。业务号 = 10 + 刀号。
KNIFE_CLASS_INDEX_TO_BUSINESS_ID = (11, 20, 12, 13, 14, 15, 16, 17, 18, 19)
KNIFE_NUM_CLASSES = len(KNIFE_CLASS_INDEX_TO_BUSINESS_ID)
KNIFE_NAMES = [f"knife_{bid - 10:02d}" for bid in KNIFE_CLASS_INDEX_TO_BUSINESS_ID]


class DirectDualModelDetector:
    """两个 10 类模型都在整帧上跑：没有闸门、没有路由、不依赖外部模块。

    与旧版 direct_fullframe_models 的三处差别：
      * 刀具类别按权重里的真实索引顺序还原（索引 1 = 10 号刀 = 业务号 20）
      * 阈值默认 0.6：实测真目标 >= 0.93、越界误报 <= 0.56，中间是一段空白
      * NMS 逐类做，不同类别的框不再互相抑制
    预处理用 letterbox，与训练一致（实测拉伸与 letterbox 结果相同，但
    letterbox 在摄像头分辨率变化时不会静默产生形变）。
    """

    LETTERBOX_PAD_VALUE = 114
    NMS_IOU = 0.45

    def __init__(self, person_path, knife_path, conf=0.6, img_size=(384, 640),
                 parallel=False, verbose=True):
        from rknnlite.api import RKNNLite

        self.conf = float(conf)
        self.img_size = tuple(img_size)
        self.parallel = bool(parallel)
        self.last_detections = []

        self.person = RKNNLite()
        self.knife = RKNNLite()
        self.runtimes = [self.person, self.knife]
        try:
            self._load(self.person, person_path, 0)
            # 并行时两个模型各占一个 NPU 核（RK3588 有 3 个核）
            self._load(self.knife, knife_path, 1 if self.parallel else 0)
        except Exception:
            self.release()
            raise

        if verbose:
            mode = "并行 核0/核1" if self.parallel else "顺序 核0"
            print(f"[Vision] 内置双模型已加载: {Path(person_path).name}, "
                  f"{Path(knife_path).name}  conf>={self.conf:.2f}  {mode}")

    def _load(self, runtime, path, core_index):
        from rknnlite.api import RKNNLite

        if runtime.load_rknn(str(path)) != 0:
            raise RuntimeError(f"load_rknn failed: {path}")
        core = (RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2)[core_index]
        if runtime.init_runtime(core_mask=core) != 0:
            raise RuntimeError(f"init_runtime failed: {path}")

    def _input(self, frame):
        input_h, input_w = self.img_size
        frame_h, frame_w = frame.shape[:2]
        scale = min(input_w / frame_w, input_h / frame_h)
        rw = max(1, min(input_w, int(round(frame_w * scale))))
        rh = max(1, min(input_h, int(round(frame_h * scale))))
        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        pad_left = (input_w - rw) // 2
        pad_top = (input_h - rh) // 2
        canvas = np.full((input_h, input_w, 3), self.LETTERBOX_PAD_VALUE, dtype=np.uint8)
        canvas[pad_top:pad_top + rh, pad_left:pad_left + rw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[np.newaxis, ...]
        return np.ascontiguousarray(rgb, dtype=np.uint8), scale, pad_left, pad_top

    def _decode(self, output, frame_shape, scale, pad_left, pad_top, id_map, tag):
        """模型原始输出 -> 检测列表，逐类 NMS，已换算回原图坐标。"""
        a = np.asarray(output, dtype=np.float32).squeeze()
        if a.ndim != 2:
            return []
        if a.shape[0] < a.shape[1]:
            a = a.T
        num_classes = len(id_map)
        if a.shape[1] < 4 + num_classes:
            return []
        a = a[np.isfinite(a).all(axis=1)]
        if not len(a):
            return []

        boxes = a[:, :4]
        scores = a[:, 4:4 + num_classes]
        # 导出的检测头已过 sigmoid；只有值域明显越界时才补一次，避免二次激活
        if float(scores.min()) < -0.01 or float(scores.max()) > 1.01:
            scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -60, 60)))

        cls_idx = scores.argmax(axis=1)
        cls_val = scores.max(axis=1)
        keep = cls_val >= self.conf
        boxes, cls_idx, cls_val = boxes[keep], cls_idx[keep], cls_val[keep]
        if not len(boxes):
            return []

        frame_h, frame_w = frame_shape[:2]
        results = []
        for c in np.unique(cls_idx):
            idx = np.flatnonzero(cls_idx == c)
            b = boxes[idx]
            xywh = np.column_stack((b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                                    b[:, 2], b[:, 3])).tolist()
            selected = cv2.dnn.NMSBoxes(
                xywh, cls_val[idx].astype(float).tolist(), self.conf, self.NMS_IOU,
            )
            for k in np.asarray(selected).reshape(-1):
                cx, cy, bw, bh = b[k]
                x1 = max(0, min(frame_w, int((cx - bw / 2 - pad_left) / scale)))
                y1 = max(0, min(frame_h, int((cy - bh / 2 - pad_top) / scale)))
                x2 = max(0, min(frame_w, int((cx + bw / 2 - pad_left) / scale)))
                y2 = max(0, min(frame_h, int((cy + bh / 2 - pad_top) / scale)))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                ci = int(c)
                bid = int(id_map[ci])
                results.append({
                    "tag": tag,
                    "business_id": bid,
                    "cls": ci,
                    "name": f"knife_{bid - 10:02d}" if tag == "knife" else f"person_{bid:02d}",
                    "score": float(cls_val[idx][k]),
                    "box": (x1, y1, x2, y2),
                })
        return results

    def detect(self, frame):
        """返回 (business_id, annotated_frame)；无检出时 business_id 为 -1。"""
        if frame is None or frame.size == 0:
            self.last_detections = []
            return -1, frame

        tensor, scale, pad_left, pad_top = self._input(frame)

        outputs = {}
        if self.parallel:
            def run(tag, runtime):
                outputs[tag] = runtime.inference(inputs=[tensor])

            threads = [threading.Thread(target=run, args=(tag, runtime))
                       for tag, runtime in (("person", self.person), ("knife", self.knife))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            outputs["person"] = self.person.inference(inputs=[tensor])
            outputs["knife"] = self.knife.inference(inputs=[tensor])

        detections = []
        for tag, id_map in (("person", PERSON_CLASS_INDEX_TO_BUSINESS_ID),
                            ("knife", KNIFE_CLASS_INDEX_TO_BUSINESS_ID)):
            outs = outputs.get(tag)
            if not outs:
                continue
            detections += self._decode(outs[0], frame.shape, scale,
                                       pad_left, pad_top, id_map, tag)
        detections.sort(key=lambda d: -d["score"])
        self.last_detections = detections

        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            colour = (255, 0, 0) if det["tag"] == "person" else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(annotated, f'{det["name"]} ID={det["business_id"]} {det["score"]:.2f}',
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        if not detections:
            return -1, annotated

        best = detections[0]
        top_summary = ", ".join(f'{d["tag"]}:{d["business_id"]}({d["score"]:.2f})' for d in detections[:3])
        print(f'[Vision] 双模型整帧: person={[round(d["score"], 3) for d in detections if d["tag"] == "person"]} '
              f'knife={[round(d["score"], 3) for d in detections if d["tag"] == "knife"]} '
              f'-> 候选: [{top_summary}]')
        return int(best["business_id"]), annotated

    def release(self):
        for runtime in getattr(self, "runtimes", []):
            try:
                runtime.release()
            except Exception:
                pass


class YoloDetector:
    CLASS_INDEX_TO_BUSINESS_ID = (
        1, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 2, 20, 3, 4, 5, 6, 7, 8, 9,
    )
    LETTERBOX_PAD_VALUE = 114

    def __init__(self, model_path=None, preset_classes=(1, 11), img_size=(384, 640),
                 show_window=True, save_debug=False, letterbox=True, conf=None,
                 parallel=False):
        self._test_backend = None
        self._direct_backend = None
        # 保存本帧全部检出（按置信度降序），供 detect_targets() 与 _save_detection 共用
        self.last_detections = []
        model_dir = Path(__file__).resolve().parent.parent / "models"
        person_name = os.environ.get("YOLO_PERSON_MODEL", "person_new1.rknn")
        knife_name = os.environ.get("YOLO_KNIFE_MODEL", "knife_new1.rknn")

        if os.environ.get("YOLO_DIRECT_PIPELINE", "1").strip().lower() not in {"0", "false", "off", "no"}:
            if conf is None:
                conf = float(os.environ.get("YOLO_DIRECT_CONF", "0.6"))
            if not parallel:
                parallel = os.environ.get(
                    "YOLO_DIRECT_PARALLEL", "0"
                ).strip().lower() in {"1", "true", "on", "yes"}
            try:
                self._direct_backend = DirectDualModelDetector(
                    str(model_dir / person_name),
                    str(model_dir / knife_name),
                    conf=conf,
                    img_size=img_size,
                    parallel=parallel,
                )
                self.preset_classes = list(preset_classes)
                self.show_window = show_window
                self.save_debug = save_debug
                self.last_key = -1
                self.last_annotated_frame = None
                self.rknn = self._direct_backend.person
                self.pt_model = None
                self.two_stage = False
                self.model_path = str(model_dir / person_name)
                self.save_every = os.environ.get(
                    "YOLO_SAVE_EVERY", "1"
                ).strip().lower() not in {"0", "false", "off", "no"}
                self.save_dir = Path(os.environ.get("YOLO_SAVE_DIR", "logs/yolo_detect"))
                self.save_conf_min = float(os.environ.get("YOLO_SAVE_CONF_MIN", "0.0"))
                self.save_count = 0
                if self.save_every:
                    self.save_dir.mkdir(parents=True, exist_ok=True)
                self.save_no_det = False
                self.no_det_count = 0
                self.no_det_dir = Path(os.environ.get(
                    "YOLO_NO_DET_DIR", str(self.save_dir / "no_det")
                ))
                self.debug_count = 0
                self.last_input_geometry = None
                self.stage1_rknn = None
                self.person_classifier = None
                self.knife_classifier = None
                print("[Vision] 内置整图双模型已启用: 人员模型 + 刀具模型")
                return
            except Exception as exc:
                self._direct_backend = None
                print(f"[Vision WARN] 内置双模型不可用，回退到两级级联: {exc}")

        if model_path is None:
            model_path = str(
                Path(__file__).resolve().parent.parent
                / "models"
                / "2cls_640x384_fp16.rknn"
            )
        self.preset_classes = list(preset_classes)
        env_width = os.environ.get("YOLO_INPUT_WIDTH")
        env_height = os.environ.get("YOLO_INPUT_HEIGHT")
        if env_width and env_height:
            self.img_size = (int(env_height), int(env_width))
        else:
            self.img_size = tuple(img_size)
        self.letterbox = os.environ.get(
            "YOLO_LETTERBOX", "1" if letterbox else "0"
        ).strip().lower() not in {"0", "false", "off", "no"}
        self.show_window = show_window
        self.save_debug = save_debug
        self.debug_count = 0
        self.last_key = -1
        self.rknn = None
        self.pt_model = None
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.last_input_geometry = None
        self.last_annotated_frame = None
        self.stage1_rknn = None
        self.person_classifier = None
        self.knife_classifier = None
        self.two_stage = False

        self.save_every = os.environ.get(
            "YOLO_SAVE_EVERY", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}
        self.save_dir = Path(os.environ.get("YOLO_SAVE_DIR", "logs/yolo_detect"))
        self.save_conf_min = float(os.environ.get("YOLO_SAVE_CONF_MIN", "0.0"))
        self.save_count = 0
        if self.save_every:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_no_det = os.environ.get(
            "YOLO_SAVE_NO_DET", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}
        self.no_det_count = 0
        self.no_det_dir = Path(os.environ.get(
            "YOLO_NO_DET_DIR", str(self.save_dir / "no_det")
        ))
        if self.save_every and self.save_no_det:
            self.no_det_dir.mkdir(parents=True, exist_ok=True)

        if self.model_path.lower().endswith(".pt"):
            try:
                from ultralytics import YOLO
                self.pt_model = YOLO(self.model_path)
                print(f"[Vision] Ultralytics model loaded: {self.model_path}")
            except Exception as e:
                print(f"[Vision Error] failed to load PT model: {e}")
            if self.save_debug:
                os.makedirs("logs/yolo_debug", exist_ok=True)
            return

        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            RKNNLite = None

        if RKNNLite is not None and os.path.exists(self.model_path):
            try:
                self.rknn = RKNNLite()
                ret = self.rknn.load_rknn(self.model_path)
                if ret != 0:
                    print(f"[Vision Error] failed to load RKNN model, code: {ret}")
                    self.rknn = None
                else:
                    ret = self.rknn.init_runtime()
                    if ret != 0:
                        print(f"[Vision Error] failed to init NPU runtime, code: {ret}")
                        self.rknn = None
                    else:
                        print(f"[Vision] RKNN model loaded: {self.model_path}")
                        self.stage1_rknn = self.rknn
                        self._load_two_stage_classifiers()
            except Exception as e:
                print(f"[Vision WARN] NPU init exception: {e}, falling back to mock mode.")
                self.rknn = None
        else:
            if RKNNLite is None:
                print("[Vision Info] rknnlite not available, running in [mock mode].")
            else:
                print(f"[Vision Info] RKNN weights not found {self.model_path}, running in [mock mode].")

        if self.save_debug:
            os.makedirs("logs/yolo_debug", exist_ok=True)

    def _load_two_stage_classifiers(self):
        base = Path(self.model_path).parent
        person_path = Path(os.environ.get(
            "YOLO_PERSON_CLASSIFIER",
            str(base / os.environ.get("YOLO_PERSON_MODEL", "person_new1.rknn")),
        ))
        knife_path = Path(os.environ.get(
            "YOLO_KNIFE_CLASSIFIER",
            str(base / os.environ.get("YOLO_KNIFE_MODEL", "knife_new1.rknn")),
        ))
        try:
            from rknnlite.api import RKNNLite
            loaded = []
            for path in (person_path, knife_path):
                runtime = RKNNLite()
                if runtime.load_rknn(str(path)) != 0 or runtime.init_runtime() != 0:
                    raise RuntimeError(f"stage2 model load failed: {path}")
                loaded.append(runtime)
            self.person_classifier, self.knife_classifier = loaded
            self.two_stage = (
                self.img_size == (384, 640)
                and self.model_path.endswith("2cls_640x384_fp16.rknn")
            )
            if self.two_stage:
                print(
                    f"[Vision] two-stage enabled: person={person_path.name}, "
                    f"knife={knife_path.name}"
                )
        except Exception as exc:
            self.person_classifier = None
            self.knife_classifier = None
            self.two_stage = False
            print(f"[Vision WARN] stage2 models not loaded, using single-model postprocess: {exc}")

    def _decode_two_stage(self, outputs, frame_shape, scale, pad_left, pad_top):
        """返回通过第一阶段阈值的候选列表：[((left, top, right, bottom), kind, conf), ...]"""
        if not outputs:
            return []
        output = np.asarray(outputs[0])
        output = np.squeeze(output)
        if output.ndim != 2:
            return []
        if output.shape[0] < output.shape[1]:
            output = output.T
        if output.shape[1] < 6:
            return []
        boxes = np.nan_to_num(output[:, :4].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        scores = np.nan_to_num(output[:, 4:6].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if boxes.size and float(np.max(np.abs(boxes))) <= 2.0:
            input_h, input_w = self.img_size
            boxes[:, (0, 2)] *= input_w
            boxes[:, (1, 3)] *= input_h
        class_ids = np.argmax(scores, axis=1)
        confs = np.max(scores, axis=1)
        threshold = float(os.environ.get("YOLO_STAGE1_CONF_THRESHOLD", "0.25"))
        mask = confs >= threshold
        if not np.any(mask):
            return []
        boxes, confs, class_ids = boxes[mask], confs[mask], class_ids[mask]
        nms_boxes = []
        for x, y, w, h in boxes:
            nms_boxes.append([float(x - w / 2), float(y - h / 2), float(w), float(h)])
        keep = cv2.dnn.NMSBoxes(
            nms_boxes, confs.astype(float).tolist(), threshold,
            float(os.environ.get("YOLO_STAGE1_NMS_THRESHOLD", "0.45")),
        )
        if len(keep) == 0:
            return []

        frame_h, frame_w = frame_shape[:2]
        sx = scale[0] if isinstance(scale, tuple) else scale
        sy = scale[1] if isinstance(scale, tuple) else scale
        gates = []
        for idx in np.asarray(keep).reshape(-1):
            idx = int(idx)
            x, y, w, h = boxes[idx]
            left = int((x - w / 2 - pad_left) / max(sx, 1e-6))
            top = int((y - h / 2 - pad_top) / max(sy, 1e-6))
            right = int((x + w / 2 - pad_left) / max(sx, 1e-6))
            bottom = int((y + h / 2 - pad_top) / max(sy, 1e-6))
            left, top = max(0, left), max(0, top)
            right, bottom = min(frame_w, right), min(frame_h, bottom)
            if right - left < 8 or bottom - top < 8:
                continue
            gates.append(((left, top, right, bottom), int(class_ids[idx]), float(confs[idx])))
        return gates

    def _save_detection(self, annotated, detections, gate_info=None):
        if not self.save_every or annotated is None:
            return
        try:
            self.save_count += 1
            ts = time.strftime("%Y%m%d_%H%M%S")
            ms = int((time.time() % 1) * 1000)
            stem = f"det_{ts}_{ms:03d}_{self.save_count:05d}"
            img_path = self.save_dir / f"{stem}.jpg"
            txt_path = self.save_dir / f"{stem}.txt"

            cv2.imwrite(str(img_path), annotated)

            lines = []
            lines.append(f"# frame_index={self.save_count} time={ts}.{ms:03d}")
            if gate_info is not None:
                tag, gconf, gbox = gate_info
                gx1, gy1, gx2, gy2 = gbox
                lines.append(
                    f"# gate: {tag} conf={gconf:.4f} "
                    f"box=({gx1},{gy1},{gx2},{gy2})"
                )
            lines.append(f"# detections: {len(detections)}")
            lines.append("# name cls score x1 y1 x2 y2")
            kept = 0
            for d in detections:
                score = float(d.get("score", 0.0))
                if score < self.save_conf_min:
                    continue
                x1, y1, x2, y2 = d.get("box", (0, 0, 0, 0))
                name = str(d.get("name", "?"))
                cls = int(d.get("cls", -1))
                lines.append(
                    f"{name} {cls} {score:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}"
                )
                kept += 1
            if self.save_conf_min > 0:
                lines.append(f"# kept_after_conf_min={kept}")
            txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"[Vision WARN] failed to save detection result: {exc}")

    def _save_no_detection(self, gate_info=None, reason="no_detection",
                           stage1_raw=None, stage2_raw=None):
        if not self.save_every or not self.save_no_det:
            return
        try:
            self.no_det_count += 1
            ts = time.strftime("%Y%m%d_%H%M%S")
            ms = int((time.time() % 1) * 1000)
            stem = f"nodet_{ts}_{ms:03d}_{self.no_det_count:05d}"
            txt_path = self.no_det_dir / f"{stem}.txt"

            lines = []
            lines.append(f"# frame_index={self.no_det_count} time={ts}.{ms:03d}")
            lines.append(f"# reason={reason}")
            lines.append("# detector=YOLO26 two-stage")
            lines.append(f"# model={self.model_path}")
            lines.append(
                f"# stage1_conf={os.environ.get('YOLO_STAGE1_CONF_THRESHOLD', '0.25')} "
                f"stage1_nms={os.environ.get('YOLO_STAGE1_NMS_THRESHOLD', '0.45')} "
                f"stage2_conf={os.environ.get('YOLO_STAGE2_CONF_THRESHOLD', '0.6')} "
                f"stage2_nms={os.environ.get('YOLO_STAGE2_NMS_THRESHOLD', '0.45')}"
            )
            lines.append(f"# img_size={self.img_size} letterbox={self.letterbox}")

            if gate_info is not None:
                tag, gconf, gbox = gate_info
                gx1, gy1, gx2, gy2 = gbox
                lines.append(
                    f"# gate: {tag} conf={gconf:.4f} "
                    f"box=({gx1},{gy1},{gx2},{gy2})"
                )
            else:
                lines.append("# gate: none (stage1 below threshold)")

            lines.append("# stage1_raw:")
            if stage1_raw is None:
                lines.append("#   top_class=n/a top_score=n/a")
            else:
                lines.append(
                    f"#   top_class={stage1_raw[0]} top_score={stage1_raw[1]:.4f}"
                )

            lines.append("# stage2_raw:")
            if stage2_raw is None:
                lines.append("#   not_run")
            else:
                lines.append(
                    f"#   num_candidates={stage2_raw.get('num_candidates', 0)} "
                    f"max_score={stage2_raw.get('max_score', 0.0):.4f} "
                    f"max_cls={stage2_raw.get('max_cls', -1)} "
                    f"conf_thr={stage2_raw.get('conf_thr', 0.0)}"
                )
                topk = stage2_raw.get("topk") or []
                lines.append("#   topk: name cls score x1 y1 x2 y2")
                for t in topk:
                    lines.append(
                        f"#     {t['name']} {t['cls']} {t['score']:.4f} "
                        f"{t['x1']} {t['y1']} {t['x2']} {t['y2']}"
                    )

            lines.append("# detections: 0")
            txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"[Vision WARN] failed to save no-detection info: {exc}")

    def _detect_classes_full_frame(self, frame, detector, num_classes, class_map, conf=None):
        empty_raw = {"num_candidates": 0, "max_score": 0.0, "max_cls": -1,
                     "conf_thr": conf or 0.0, "topk": []}
        if detector is None or frame is None or frame.size == 0:
            return [], empty_raw
        input_h, input_w = self.img_size
        frame_h, frame_w = frame.shape[:2]
        scale = min(input_w / frame_w, input_h / frame_h)
        rw = max(1, min(input_w, int(round(frame_w * scale))))
        rh = max(1, min(input_h, int(round(frame_h * scale))))
        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        pad_left = (input_w - rw) // 2
        pad_top = (input_h - rh) // 2
        canvas = np.full((input_h, input_w, 3), self.LETTERBOX_PAD_VALUE, dtype=np.uint8)
        canvas[pad_top:pad_top + rh, pad_left:pad_left + rw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[np.newaxis, ...]
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

        outputs = detector.inference(inputs=[rgb])
        if not outputs:
            return [], empty_raw

        a = np.asarray(outputs[0], dtype=np.float32).squeeze()
        if a.ndim != 2:
            return [], empty_raw
        if a.shape[0] < a.shape[1]:
            a = a.T
        if a.shape[1] < 4 + num_classes:
            return [], empty_raw
        a = a[np.isfinite(a).all(1)]
        if not len(a):
            return [], empty_raw

        scores_all = a[:, 4:4 + num_classes]
        if float(scores_all.min()) < -0.01 or float(scores_all.max()) > 1.01:
            scores_all = 1.0 / (1.0 + np.exp(-np.clip(scores_all, -60, 60)))

        if conf is None:
            conf = float(os.environ.get("YOLO_STAGE2_CONF_THRESHOLD", "0.6"))
        iou = float(os.environ.get("YOLO_STAGE2_NMS_THRESHOLD", "0.45"))

        cls_idx = scores_all.argmax(1)
        cls_val = scores_all.max(1)
        keep = cls_val >= conf
        a2, cls_idx2, cls_val2 = a[keep], cls_idx[keep], cls_val[keep]

        results = []
        if len(a2):
            for c in np.unique(cls_idx2):
                idx = np.flatnonzero(cls_idx2 == c)
                boxes = a2[idx, :4].copy()
                boxes[:, :2] -= boxes[:, 2:] / 2
                selected = cv2.dnn.NMSBoxes(
                    boxes.tolist(), cls_val2[idx].astype(float).tolist(), conf, iou,
                )
                for k in np.asarray(selected).reshape(-1):
                    cx, cy, bw, bh = a2[idx[k], :4]
                    x1 = (cx - bw / 2 - pad_left) / max(scale, 1e-6)
                    y1 = (cy - bh / 2 - pad_top) / max(scale, 1e-6)
                    x2 = (cx + bw / 2 - pad_left) / max(scale, 1e-6)
                    y2 = (cy + bh / 2 - pad_top) / max(scale, 1e-6)
                    x1 = int(max(0, min(frame_w, x1)))
                    y1 = int(max(0, min(frame_h, y1)))
                    x2 = int(max(0, min(frame_w, x2)))
                    y2 = int(max(0, min(frame_h, y2)))
                    if x2 - x1 < 4 or y2 - y1 < 4:
                        continue
                    ci = int(c)
                    bid = int(class_map[ci]) if 0 <= ci < len(class_map) else -1
                    results.append({
                        "business_id": bid,
                        "cls_index": ci,
                        "score": float(cls_val2[idx[k]]),
                        "box": (x1, y1, x2, y2),
                    })
            results.sort(key=lambda r: -r["score"])

        raw_max_score = float(scores_all.max()) if scores_all.size else 0.0
        if scores_all.size:
            fi = int(scores_all.argmax())
            raw_max_cls = int(fi % scores_all.shape[1])
        else:
            raw_max_cls = -1

        topk = []
        try:
            flat = scores_all.reshape(-1)
            order = np.argsort(-flat)[:5]
            for fi in order:
                s = float(flat[fi])
                if s <= 0.0:
                    continue
                r = int(fi // scores_all.shape[1])
                c = int(fi % scores_all.shape[1])
                cx, cy, bw, bh = a[r, :4] if r < a.shape[0] else (0, 0, 0, 0)
                x1 = int(max(0, min(frame_w, (cx - bw / 2 - pad_left) / max(scale, 1e-6))))
                y1 = int(max(0, min(frame_h, (cy - bh / 2 - pad_top) / max(scale, 1e-6))))
                x2 = int(max(0, min(frame_w, (cx + bw / 2 - pad_left) / max(scale, 1e-6))))
                y2 = int(max(0, min(frame_h, (cy + bh / 2 - pad_top) / max(scale, 1e-6))))
                cname = class_map[c] if 0 <= c < len(class_map) else f"cls_{c}"
                topk.append({
                    "name": f"cls_{cname}",
                    "cls": c,
                    "score": s,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
        except Exception:
            pass

        raw_info = {
            "num_candidates": int(scores_all.shape[0]) if scores_all.ndim == 2 else 0,
            "max_score": raw_max_score,
            "max_cls": raw_max_cls,
            "conf_thr": float(conf),
            "topk": topk,
        }
        return results, raw_info

    def _detect_two_stage(self, frame):
        input_data, scale, pad_left, pad_top = self._prepare_rknn_input(frame)
        outputs = self.rknn.inference(inputs=[input_data])
        gates = self._decode_two_stage(outputs, frame.shape, scale, pad_left, pad_top)
        annotated = frame.copy()
        business_id = -1
        detections = []
        gate_info = None
        stage1_raw = None
        stage2_raw = None

        if not gates:
            try:
                s1 = np.asarray(outputs[0], dtype=np.float32).squeeze()
                if s1.ndim == 2:
                    if s1.shape[0] < s1.shape[1]:
                        s1 = s1.T
                    if s1.shape[1] >= 6:
                        sc = s1[:, 4:6]
                        if float(sc.min()) < -0.01 or float(sc.max()) > 1.01:
                            sc = 1.0 / (1.0 + np.exp(-np.clip(sc, -60, 60)))
                        pcm = sc.max(0)
                        tc = int(pcm.argmax())
                        tscore = float(pcm[tc])
                        stage1_raw = (tc, tscore)
            except Exception:
                stage1_raw = None

            cv2.putText(annotated, "No person/knife (gate)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            # 记录最高分的 gate 供日志落盘，同时收集出现过的所有大类（支持同帧触发 person + knife）
            seen_kinds = []
            for idx, ((gl, gt, gr, gb), kind, gate_conf) in enumerate(gates):
                tag = "person" if kind == 0 else "knife"
                if idx == 0:
                    gate_info = (tag, float(gate_conf), (gl, gt, gr, gb))
                cv2.rectangle(annotated, (gl, gt), (gr, gb), (255, 200, 0), 1)
                cv2.putText(annotated, f"gate: {tag} {gate_conf:.2f}",
                            (gl, max(22, gt - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
                if kind not in seen_kinds:
                    seen_kinds.append(kind)

            for kind in seen_kinds:
                if kind == 0:
                    detector = self.person_classifier
                    class_map = PERSON_CLASS_INDEX_TO_BUSINESS_ID
                    num_classes = PERSON_NUM_CLASSES
                    tag = "person"
                    name_map = PERSON_NAMES
                else:
                    detector = self.knife_classifier
                    class_map = KNIFE_CLASS_INDEX_TO_BUSINESS_ID
                    num_classes = KNIFE_NUM_CLASSES
                    tag = "knife"
                    name_map = KNIFE_NAMES

                dets, raw_info = self._detect_classes_full_frame(
                    frame, detector, num_classes, class_map,
                )
                if stage2_raw is None:
                    stage2_raw = raw_info

                for d in dets:
                    x1, y1, x2, y2 = d["box"]
                    bid = d["business_id"]
                    ci = d["cls_index"]
                    score = d["score"]
                    label_name = name_map[ci] if 0 <= ci < len(name_map) else f"{tag}_{ci}"
                    color = (0, 0, 255) if kind == 0 else (0, 255, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"{label_name} id={bid} {score:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                    )
                    detections.append({
                        "tag": tag,
                        "name": label_name,
                        "cls": ci,
                        "score": float(score),
                        "box": (x1, y1, x2, y2),
                        "business_id": bid,
                    })

            detections.sort(key=lambda d: -d["score"])
            if detections:
                business_id = detections[0]["business_id"]
            else:
                cv2.putText(annotated, "gate passed but no det", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        if self.show_window:
            try:
                cv2.imshow("YOLO Two-Stage Detection View", annotated)
                self.last_key = cv2.waitKey(1) & 0xFF
            except Exception as exc:
                print(f"[Vision WARN] 无法显示窗口，改为只落盘: {exc}")
                self.show_window = False

        if detections:
            self._save_detection(annotated, detections, gate_info=gate_info)
        else:
            if not gates:
                reason = "stage1_gate_below_threshold"
            else:
                reason = "stage1_passed_but_stage2_no_det"
            self._save_no_detection(
                gate_info=gate_info,
                reason=reason,
                stage1_raw=stage1_raw,
                stage2_raw=stage2_raw,
            )

        if self.save_debug:
            self.debug_count += 1
            output_dir = Path("logs/yolo_debug")
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = str(
                output_dir / f"two_stage_{int(time.time())}_{self.debug_count}.jpg"
            )
            cv2.imwrite(filename, annotated)

        self.last_annotated_frame = annotated
        self.last_detections = detections
        return business_id

    def _prepare_rknn_input(self, frame):
        input_h, input_w = self.img_size
        frame_h, frame_w = frame.shape[:2]
        if frame_h <= 0 or frame_w <= 0:
            raise ValueError("invalid input image size")
        if not self.letterbox:
            resized = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            scale_x = input_w / frame_w
            scale_y = input_h / frame_h
            input_data = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)[np.newaxis, ...]
            self.last_input_geometry = {
                "source_size": (frame_w, frame_h),
                "model_size": (input_w, input_h),
                "resized_size": (input_w, input_h),
                "pad": (0, 0),
                "scale": (scale_x, scale_y),
                "letterbox": False,
            }
            return np.ascontiguousarray(input_data, dtype=np.uint8), (scale_x, scale_y), 0, 0

        scale = min(input_w / frame_w, input_h / frame_h)
        resized_w = max(1, min(input_w, int(round(frame_w * scale))))
        resized_h = max(1, min(input_h, int(round(frame_h * scale))))
        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        pad_left = (input_w - resized_w) // 2
        pad_top = (input_h - resized_h) // 2
        input_bgr = np.full(
            (input_h, input_w, 3), self.LETTERBOX_PAD_VALUE, dtype=np.uint8
        )
        input_bgr[
            pad_top:pad_top + resized_h,
            pad_left:pad_left + resized_w,
        ] = resized
        input_data = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)[np.newaxis, ...]
        self.last_input_geometry = {
            "source_size": (frame_w, frame_h),
            "model_size": (input_w, input_h),
            "resized_size": (resized_w, resized_h),
            "pad": (pad_left, pad_top),
            "scale": scale,
            "letterbox": True,
        }
        return np.ascontiguousarray(input_data, dtype=np.uint8), scale, pad_left, pad_top

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _decide_targets(self, detections):
        """同侧最多两张图时的判决器：

        * 默认只返回置信度最高的一个；
        * 只有当第二个目标同时满足「置信度也高」且「和第一个不在同一位置」时，
          才认为这一侧真的放了两张图，返回两个；
        * 只有一个高置信度时，说明这侧就一张，按单目标返回。

        阈值：YOLO_PAIR_CONF（默认 0.8，高于基础 conf 0.6），
              YOLO_PAIR_IOU（默认 0.3，大于则视为同一位置）。
        """
        if not detections:
            return []

        if self.preset_classes:
            valid_dets = [d for d in detections if int(d["business_id"]) in self.preset_classes]
        else:
            valid_dets = list(detections)

        if not valid_dets:
            return []

        ordered = sorted(valid_dets, key=lambda d: -float(d["score"]))
        best = ordered[0]
        best_id = int(best["business_id"])
        best_score = float(best["score"])

        pair_conf = float(os.environ.get("YOLO_PAIR_CONF", "0.8"))
        pair_iou = float(os.environ.get("YOLO_PAIR_IOU", "0.3"))

        for det in ordered[1:]:
            other_id = int(det["business_id"])
            if other_id == best_id:
                continue
            other_score = float(det["score"])
            if other_score < pair_conf:
                continue
            overlap = self._iou(best["box"], det["box"])
            if overlap >= pair_iou:
                continue
            print(f"[Vision] 同侧双目标: id={best_id}({best_score:.2f}) + "
                  f"id={other_id}({other_score:.2f}) IoU={overlap:.2f}")
            return [best_id, other_id]

        # 方便现场排查：把第二名也打出来，看清楚为什么没当成双图
        if len(ordered) > 1:
            runner = ordered[1]
            print(f"[Vision] 单目标 id={best_id}({best_score:.2f}); 第二名 "
                  f"id={int(runner['business_id'])}({float(runner['score']):.2f}) "
                  f"IoU={self._iou(best['box'], runner['box']):.2f} 未采纳")
        return [best_id]

    def detect_targets(self, frame):
        """返回本帧检出的全部有效业务号（按分数从高到低，去重）。

        用于一侧放两张靶图的场景，把两个类别全拿出来。detect_target() 仍然只返回
        分数最高的一个，行为与原先完全一致。
        """
        self.last_detections = []
        if frame is None or frame.size == 0:
            return []

        # ---- 内置整图双模型：里面已保存本帧全部检出 ----
        if self._direct_backend is not None:
            try:
                _, annotated = self._direct_backend.detect(frame)
                self.last_annotated_frame = annotated
                self.last_detections = list(self._direct_backend.last_detections)
                if self.show_window and annotated is not None:
                    try:
                        cv2.imshow("YOLO Direct Dual-Model View", annotated)
                        self.last_key = cv2.waitKey(1) & 0xFF
                    except Exception as exc:
                        print(f"[Vision WARN] 无法显示窗口，改为只落盘: {exc}")
                        self.show_window = False
                if self.last_detections:
                    self._save_detection(annotated, self.last_detections)
                else:
                    self._save_no_detection(reason="dual_model_no_detection")
                return self._decide_targets(self.last_detections)
            except Exception as exc:
                print(f"[Vision ERROR] 内置双模型推理失败: {exc}")
                self.last_annotated_frame = frame.copy()
                return []

        # ---- 两级级联：_detect_two_stage 内部会填充 self.last_detections ----
        if self.two_stage and self.rknn is not None:
            try:
                self._detect_two_stage(frame)
                return self._decide_targets(self.last_detections)
            except Exception as exc:
                print(f"[Vision ERROR] two-stage inference failed: {exc}")
                self.last_annotated_frame = frame.copy()
                return []

        # ---- .pt 模型 ----
        if self.pt_model is not None:
            try:
                results = self.pt_model.predict(
                    source=frame,
                    imgsz=self.img_size if not self.letterbox else self.img_size[0],
                    conf=0.5,
                    verbose=False,
                )
                result = results[0]
                detections = []
                if result.boxes is not None:
                    for class_index, conf, xyxy in zip(
                        result.boxes.cls.cpu().tolist(),
                        result.boxes.conf.cpu().tolist(),
                        result.boxes.xyxy.cpu().tolist(),
                    ):
                        class_name = str(result.names[int(class_index)])
                        try:
                            business_id = int(class_name)
                        except ValueError:
                            continue
                        x1, y1, x2, y2 = (int(v) for v in xyxy)
                        detections.append({
                            "business_id": business_id,
                            "cls": int(class_index),
                            "name": class_name,
                            "score": float(conf),
                            "box": (x1, y1, x2, y2),
                        })
                self.last_detections = detections
                display_frame = result.plot()
                self.last_annotated_frame = display_frame.copy()
                if self.show_window:
                    cv2.imshow("YOLO Target Detection", display_frame)
                    self.last_key = cv2.waitKey(1) & 0xFF
                return self._decide_targets(detections)
            except Exception as exc:
                print(f"[Vision ERROR] PT inference error: {exc}")
                return []

        # ---- 旧单模型路径没有多目标解码，退化成单目标 ----
        target = self.detect_target(frame)
        return [target] if target != -1 else []

    def detect_target(self, frame):
        if self._direct_backend is not None:
            try:
                target_id, annotated = self._direct_backend.detect(frame)
                self.last_annotated_frame = annotated
                detections = self._direct_backend.last_detections
                self.last_detections = list(detections)
                # 显示窗口：原来这条分支漏了 imshow，所以 show_window 一直不生效。
                # 单独 try 包住：没有图形界面时 imshow 会抛异常，不能让它连累检测。
                if self.show_window and annotated is not None:
                    try:
                        cv2.imshow("YOLO Direct Dual-Model View", annotated)
                        self.last_key = cv2.waitKey(1) & 0xFF
                    except Exception as exc:
                        print(f"[Vision WARN] 无法显示窗口，改为只落盘: {exc}")
                        self.show_window = False
                if detections:
                    self._save_detection(annotated, detections)
                else:
                    self._save_no_detection(reason="dual_model_no_detection")
                return int(target_id)
            except Exception as exc:
                print(f"[Vision ERROR] 内置双模型推理失败: {exc}")
                self.last_annotated_frame = frame.copy() if frame is not None else None
                return -1

        if self._test_backend is not None:
            try:
                annotated, detections = self._test_backend.infer_frame(frame)
                self.last_annotated_frame = annotated
                self.last_detections = list(detections)
                if detections:
                    return int(detections[0]["business_id"])
                return -1
            except Exception as exc:
                print(f"[Vision ERROR] test_new_yolo inference failed: {exc}")
                self.last_annotated_frame = frame.copy()
                return -1

        if frame is None or frame.size == 0:
            print("[Vision WARN] empty frame, cannot detect")
            return -1

        if self.two_stage and self.rknn is not None:
            try:
                return int(self._detect_two_stage(frame))
            except Exception as exc:
                print(f"[Vision ERROR] two-stage inference failed: {exc}")
                import traceback
                traceback.print_exc()
                self.last_annotated_frame = frame.copy()
                return -1

        if self.pt_model is not None:
            try:
                results = self.pt_model.predict(
                    source=frame,
                    imgsz=self.img_size if not self.letterbox else self.img_size[0],
                    conf=0.5,
                    verbose=False,
                )
                result = results[0]
                detected_ids = []
                if result.boxes is not None:
                    for class_index in result.boxes.cls.cpu().tolist():
                        class_name = str(result.names[int(class_index)])
                        try:
                            detected_ids.append(int(class_name))
                        except ValueError:
                            print(f"[Vision WARN] non-numeric class name: {class_name}")

                display_frame = result.plot()
                self.last_annotated_frame = display_frame.copy()
                if self.show_window:
                    cv2.imshow("YOLO Target Detection", display_frame)
                    self.last_key = cv2.waitKey(1) & 0xFF
                if self.save_debug:
                    self.debug_count += 1
                    output_dir = Path("logs/yolo_debug")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(output_dir / f"pt_{int(time.time())}_{self.debug_count}.jpg"),
                        display_frame,
                    )

                for business_id in detected_ids:
                    if business_id in self.preset_classes:
                        print(f"[Vision] PT hit, business ID: {business_id}")
                        return business_id
                return -1
            except Exception as e:
                print(f"[Vision ERROR] PT inference error: {e}")
                return -1

        if self.rknn is not None:
            try:
                input_data, scale, pad_left, pad_top = self._prepare_rknn_input(frame)
                outputs = self.rknn.inference(inputs=[input_data])
                annotated_frame = frame.copy()
                class_id = self.post_process(
                    outputs,
                    draw_frame=annotated_frame,
                    conf_threshold=0.55,
                    nms_threshold=0.55,
                    scale=scale,
                    pad_left=pad_left,
                    pad_top=pad_top,
                )
                self.last_annotated_frame = annotated_frame

                if self.show_window:
                    cv2.imshow("YOLO NPU Detection View", annotated_frame)
                    self.last_key = cv2.waitKey(1) & 0xFF

                if self.save_debug:
                    self.debug_count += 1
                    timestamp = int(time.time())
                    filename = f"logs/yolo_debug/detect_{timestamp}_{self.debug_count}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    if class_id != -1:
                        cv2.putText(annotated_frame, f"Target: {class_id}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imwrite(filename.replace('.jpg', '_result.jpg'), annotated_frame)

                if class_id in self.preset_classes:
                    print(f"[Vision] NPU hit, class ID: {class_id}")
                    return class_id

                return -1

            except Exception as e:
                print(f"[Vision ERROR] NPU inference error: {e}")
                import traceback
                traceback.print_exc()
                return -1

        else:
            resized_bgr = cv2.resize(frame, self.img_size)
            h, w, _ = resized_bgr.shape
            simulated_result = 1

            if self.show_window:
                cv2.putText(resized_bgr, "Mock NPU Detecting...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(resized_bgr, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 0, 255), 2)
                cv2.putText(resized_bgr, f"Mock Target ID: {simulated_result}", (w // 4, h // 4 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("YOLO Mock Detection View", resized_bgr)
                self.last_key = cv2.waitKey(1) & 0xFF

            if self.save_debug:
                self.debug_count += 1
                timestamp = int(time.time())
                filename = f"logs/yolo_debug/mock_{timestamp}_{self.debug_count}.jpg"
                cv2.imwrite(filename, resized_bgr)

            print(f"[Vision Mock] mock detection done, returned target: {simulated_result}")

            if simulated_result in self.preset_classes:
                return simulated_result
            return -1

    def post_process(
        self, outputs, draw_frame=None, conf_threshold=None,
        nms_threshold=None, scale=1.0, pad_left=0, pad_top=0,
    ):
        if not outputs or len(outputs) == 0:
            return -1
        if conf_threshold is None:
            conf_threshold = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.55"))
        if nms_threshold is None:
            nms_threshold = float(os.environ.get("YOLO_NMS_THRESHOLD", "0.55"))

        try:
            output = outputs[0]
            output = np.squeeze(output)

            if output.shape[0] < output.shape[1]:
                output = np.transpose(output)

            boxes = output[:, :4].astype(np.float32, copy=False)
            scores = output[:, 4:]
            if boxes.size and float(np.nanmax(np.abs(boxes))) <= 2.0:
                input_h, input_w = self.img_size
                boxes = boxes.copy()
                boxes[:, (0, 2)] *= float(input_w)
                boxes[:, (1, 3)] *= float(input_h)

            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)

            mask = confidences > conf_threshold
            if not np.any(mask):
                return -1

            valid_boxes = boxes[mask]
            valid_confs = confidences[mask]
            valid_class_ids = class_ids[mask]

            x_left = valid_boxes[:, 0] - valid_boxes[:, 2] / 2.0
            y_top = valid_boxes[:, 1] - valid_boxes[:, 3] / 2.0
            widths = valid_boxes[:, 2]
            heights = valid_boxes[:, 3]

            nms_boxes = np.column_stack((x_left, y_top, widths, heights)).tolist()
            nms_scores = valid_confs.tolist()

            indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, conf_threshold, nms_threshold)

            if len(indices) > 0:
                indices = np.array(indices).flatten()

                if draw_frame is not None:
                    for idx in indices:
                        class_index = int(valid_class_ids[idx])
                        if class_index < 0 or class_index >= len(self.CLASS_INDEX_TO_BUSINESS_ID):
                            continue
                        cls_id = self.CLASS_INDEX_TO_BUSINESS_ID[class_index]
                        conf = float(valid_confs[idx])

                        if cls_id in self.preset_classes:
                            if isinstance(scale, tuple):
                                scale_x, scale_y = scale
                            else:
                                scale_x = scale_y = scale
                            x = int((x_left[idx] - pad_left) / scale_x)
                            y = int((y_top[idx] - pad_top) / scale_y)
                            w = int(widths[idx] / scale_x)
                            h = int(heights[idx] / scale_y)
                            x = max(0, x)
                            y = max(0, y)
                            w = min(w, draw_frame.shape[1] - x)
                            h = min(h, draw_frame.shape[0] - y)

                            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            label = f"ID: {cls_id} ({conf:.2f})"
                            cv2.putText(draw_frame, label, (x, y - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                for idx in indices:
                    class_index = int(valid_class_ids[idx])
                    if class_index < 0 or class_index >= len(self.CLASS_INDEX_TO_BUSINESS_ID):
                        continue
                    cls_id = self.CLASS_INDEX_TO_BUSINESS_ID[class_index]
                    if cls_id in self.preset_classes:
                        return cls_id

                best_idx = indices[0]
                class_index = int(valid_class_ids[best_idx])
                if class_index < 0 or class_index >= len(self.CLASS_INDEX_TO_BUSINESS_ID):
                    return -1
                detected_class = self.CLASS_INDEX_TO_BUSINESS_ID[class_index]
                return detected_class

        except Exception as e:
            print(f"[Vision PostProcess Error] postprocess parse error: {e}")
            print("hint: dimension mismatch may be related to model export/conversion parameters.")
            import traceback
            traceback.print_exc()

        return -1

    def release(self):
        if self._direct_backend is not None:
            self._direct_backend.release()
            self._direct_backend = None
            cv2.destroyAllWindows()
            return
        if self._test_backend is not None:
            self._test_backend.release()
            self._test_backend = None
            return
        for runtime in (self.person_classifier, self.knife_classifier):
            if runtime is not None and runtime is not self.rknn:
                try:
                    runtime.release()
                except Exception:
                    pass
        if self.rknn is not None:
            try:
                self.rknn.release()
                print("[Vision] RKNN resources released")
            except Exception:
                pass
        cv2.destroyAllWindows()


def _parse_camera(value):
    try:
        return int(value)
    except ValueError:
        return value


def _ensure_pt_runtime(model_path):
    resolved = Path(model_path).expanduser().resolve()
    if resolved.suffix.lower() != ".pt":
        return
    try:
        import ultralytics  # noqa: F401
        return
    except ImportError:
        pass

    venv_python = resolved.parent.parent / "myenv" / "bin" / "python"
    if not venv_python.is_file():
        raise RuntimeError(
            f"current python has no ultralytics and venv not found: {venv_python}"
        )
    print(f"[TEST] switching to project venv: {venv_python}", flush=True)
    os.execv(
        str(venv_python),
        [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
    )


def main():
    parser = argparse.ArgumentParser(description="RKNN YOLO target detection test")
    parser.add_argument("--camera", default="0", help="camera index or device path")
    parser.add_argument("--model", default=None, help="RKNN model path (stage1 gate)")
    parser.add_argument(
        "--frames", type=int, default=0,
        help="max frames to test, 0 means run until Q or Esc",
    )
    parser.add_argument(
        "--classes",
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
        help="business class IDs of interest",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="整图双模型的置信度阈值 (默认取 YOLO_DIRECT_CONF，即 0.6)",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="两个模型各占一个 NPU 核并行推理 (RK3588 有 3 个核)",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="do not show detection window"
    )
    parser.add_argument("--save-debug", action="store_true", help="save debug images")
    parser.add_argument(
        "--save-every", dest="save_every", action="store_true",
        help="save image + txt on every detection (default on)",
    )
    parser.add_argument(
        "--no-save-every", dest="save_every", action="store_false",
        help="disable per-detection saving",
    )
    parser.set_defaults(save_every=True)
    parser.add_argument(
        "--save-dir", default=None, help="output dir (default logs/yolo_detect)",
    )
    parser.add_argument(
        "--save-conf-min", type=float, default=None,
        help="min confidence to write into txt (default 0.0, write all)",
    )
    parser.add_argument(
        "--no-save-no-det", dest="save_no_det", action="store_false",
        help="disable saving txt when nothing is detected",
    )
    parser.set_defaults(save_no_det=True)
    parser.add_argument(
        "--no-det-dir", default=None,
        help="dir for no-detection txt (default <save-dir>/no_det)",
    )
    args = parser.parse_args()

    selected_model = args.model or str(
        Path(__file__).resolve().parent.parent
        / "models"
        / "2cls_640x384_fp16.rknn"
    )
    _ensure_pt_runtime(selected_model)

    os.environ["YOLO_SAVE_EVERY"] = "1" if args.save_every else "0"
    if args.save_dir:
        os.environ["YOLO_SAVE_DIR"] = args.save_dir
    if args.save_conf_min is not None:
        os.environ["YOLO_SAVE_CONF_MIN"] = str(args.save_conf_min)
    os.environ["YOLO_SAVE_NO_DET"] = "1" if args.save_no_det else "0"
    if args.no_det_dir:
        os.environ["YOLO_NO_DET_DIR"] = args.no_det_dir

    preset_classes = tuple(
        int(value.strip()) for value in args.classes.split(",") if value.strip()
    )
    detector = YoloDetector(
        model_path=selected_model,
        preset_classes=preset_classes,
        show_window=not args.no_show,
        save_debug=args.save_debug,
        conf=args.conf,
        parallel=args.parallel,
    )
    if detector.rknn is None and detector.pt_model is None:
        detector.release()
        raise RuntimeError("RKNN model not loaded, refusing to test with mock results")

    source = _parse_camera(args.camera)
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    camera = cv2.VideoCapture(source, backend)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        detector.release()
        raise RuntimeError(f"cannot open camera: {source}")

    # 自己接管 SIGINT：rknnlite 的 inference() 会吞掉 KeyboardInterrupt 并返回
    # None，退出信号传不到主循环。这里只设标志位，连 rknnlite 内部的异常路径
    # 都不会触发，主循环在下一帧开头干净退出。
    stop = {"flag": False}

    def _on_sigint(signum, frame):
        if not stop["flag"]:
            print("\n[TEST] 收到 Ctrl+C，本帧结束后退出...", flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_sigint)

    interrupted = False
    try:
        frame_index = 0
        while args.frames <= 0 or frame_index < args.frames:
            if stop["flag"]:
                interrupted = True
                break
            frame_index += 1
            ok, frame = camera.read()
            if not ok or frame is None or frame.size == 0:
                print("[TEST WARN] camera read failed")
                continue
            results = detector.detect_targets(frame)
            if results or frame_index == 1 or frame_index % 30 == 0:
                print(f"[TEST] frame={frame_index} targets={results}")
            if detector.last_key in (ord("q"), ord("Q"), 27):
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        camera.release()
        detector.release()
    if interrupted:
        print("[TEST] interrupted by user")
    else:
        print("[TEST] test finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())