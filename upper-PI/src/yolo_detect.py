import argparse
import time
import os
import sys
import numpy as np
from pathlib import Path

# OpenCV 自带的 Qt 插件目录不包含字体，显式使用系统字体目录。
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

import cv2

class YoloDetector:
    # last 1_6.pt 的训练类别名称按字符串排序保存在模型中。RKNN输出只
    # 保留类别索引，因此必须使用同一顺序还原业务ID，不能简单 index + 1。
    CLASS_INDEX_TO_BUSINESS_ID = (
        1, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 2, 20, 3, 4, 5, 6, 7, 8, 9,
    )
    LETTERBOX_PAD_VALUE = 114

    def __init__(self, model_path=None, preset_classes=(1, 11), img_size=(384, 640),
                 show_window=False, save_debug=False, letterbox=True):
        """
        初始化 RKNN 侦查视觉系统 (专为香橙派 5 Pro / RKNN3588 设计)
        :param model_path: .rknn 格式的模型路径
        :param preset_classes: 需要关注的类别 ID
        :param img_size: 模型要求的输入图像分辨率 (H, W)，16:9 模型可用 (360, 640)
        :param show_window: 是否显示检测窗口（生产环境建议False）
        :param save_debug: 是否保存调试图片
        """
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

        if self.model_path.lower().endswith(".pt"):
            try:
                from ultralytics import YOLO

                self.pt_model = YOLO(self.model_path)
                print(f"[Vision] 成功加载 Ultralytics 模型: {self.model_path}")
            except Exception as e:
                print(f"[Vision Error] 加载 PT 模型失败: {e}")
            if self.save_debug:
                os.makedirs("logs/yolo_debug", exist_ok=True)
            return

        # ---- 初始化 RKNN NPU 运行时 ----
        try:
            from rknnlite.api import RKNNLite  # type: ignore
        except ImportError:
            RKNNLite = None

        if RKNNLite is not None and os.path.exists(self.model_path):
            try:
                self.rknn = RKNNLite()
                # 1. 加载 RKNN 模型到内存
                ret = self.rknn.load_rknn(self.model_path)
                if ret != 0:
                    print(f"[Vision Error] 加载 RKNN 模型失败，错误码: {ret}")
                    self.rknn = None
                else:
                    # 2. 初始化 NPU 硬件运行时环境
                    ret = self.rknn.init_runtime()
                    if ret != 0:
                        print(f"[Vision Error] 初始化 NPU 运行时失败，错误码: {ret}")
                        self.rknn = None
                    else:
                        print(f"[Vision] 成功加载 RKNN 硬件加速模型: {self.model_path}")
                        self.stage1_rknn = self.rknn
                        self._load_two_stage_classifiers()
            except Exception as e:
                print(f"[Vision WARN] NPU 初始化发生异常: {e}，将自动切换至模拟检测模式。")
                self.rknn = None
        else:
            if RKNNLite is None:
                print("[Vision Info] 未检测到 rknnlite 运行库，当前运行在 [模拟检测模式] 下。")
            else:
                print(f"[Vision Info] 未找到 RKNN 权重文件 {self.model_path}，当前运行在 [模拟检测模式] 下。")
        
        # 创建调试目录
        if self.save_debug:
            os.makedirs("logs/yolo_debug", exist_ok=True)

    def _load_two_stage_classifiers(self):
        """Load the 256x256 person/knife classifiers for the 2-class detector."""
        base = Path(self.model_path).parent
        person_path = Path(os.environ.get(
            "YOLO_PERSON_CLASSIFIER",
            str(base / "person_classifier_224_fp16.rknn"),
        ))
        knife_path = Path(os.environ.get(
            "YOLO_KNIFE_CLASSIFIER",
            str(base / "knife_classifier_224_fp16.rknn"),
        ))
        try:
            from rknnlite.api import RKNNLite
            loaded = []
            for path in (person_path, knife_path):
                runtime = RKNNLite()
                if runtime.load_rknn(str(path)) != 0 or runtime.init_runtime() != 0:
                    raise RuntimeError(f"分类模型加载失败: {path}")
                loaded.append(runtime)
            self.person_classifier, self.knife_classifier = loaded
            self.two_stage = (
                self.img_size == (384, 640)
                and self.model_path.endswith("2cls_640x384_fp16.rknn")
            )
            if self.two_stage:
                print(
                    f"[Vision] 两阶段识别已启用: 人={person_path.name}, "
                    f"刀={knife_path.name}"
                )
        except Exception as exc:
            self.person_classifier = None
            self.knife_classifier = None
            self.two_stage = False
            print(f"[Vision WARN] 两阶段分类模型未加载，将使用单模型后处理: {exc}")

    def _decode_two_stage(self, outputs, frame_shape, scale, pad_left, pad_top):
        """Decode 4+2 YOLO output and return (box, kind, confidence)."""
        if not outputs:
            return None
        output = np.asarray(outputs[0])
        output = np.squeeze(output)
        if output.ndim != 2:
            return None
        if output.shape[0] < output.shape[1]:
            output = output.T
        if output.shape[1] < 6:
            return None
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
            return None
        boxes, confs, class_ids = boxes[mask], confs[mask], class_ids[mask]
        nms_boxes = []
        for x, y, w, h in boxes:
            nms_boxes.append([float(x - w / 2), float(y - h / 2), float(w), float(h)])
        keep = cv2.dnn.NMSBoxes(
            nms_boxes, confs.astype(float).tolist(), threshold,
            float(os.environ.get("YOLO_STAGE1_NMS_THRESHOLD", "0.45")),
        )
        if len(keep) == 0:
            return None
        best = int(np.asarray(keep).reshape(-1)[0])
        x, y, w, h = boxes[best]
        frame_h, frame_w = frame_shape[:2]
        sx = scale[0] if isinstance(scale, tuple) else scale
        sy = scale[1] if isinstance(scale, tuple) else scale
        left = int((x - w / 2 - pad_left) / max(sx, 1e-6))
        top = int((y - h / 2 - pad_top) / max(sy, 1e-6))
        right = int((x + w / 2 - pad_left) / max(sx, 1e-6))
        bottom = int((y + h / 2 - pad_top) / max(sy, 1e-6))
        left, top = max(0, left), max(0, top)
        right, bottom = min(frame_w, right), min(frame_h, bottom)
        if right - left < 8 or bottom - top < 8:
            return None
        return (left, top, right, bottom), int(class_ids[best]), float(confs[best])

    def _classify_crop(self, crop, classifier):
        if classifier is None or crop is None or crop.size == 0:
            return -1, 0.0
        image = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[np.newaxis, ...]
        outputs = classifier.inference(inputs=[np.ascontiguousarray(image, dtype=np.uint8)])
        values = np.asarray(outputs[0]).reshape(-1)
        if values.size == 0:
            return -1, 0.0
        values = np.nan_to_num(values, nan=-1e9, posinf=1e9, neginf=-1e9)
        index = int(np.argmax(values))
        exp = np.exp(values - np.max(values))
        confidence = float(exp[index] / max(float(exp.sum()), 1e-12))
        return index, confidence

    def _detect_two_stage(self, frame):
        input_data, scale, pad_left, pad_top = self._prepare_rknn_input(frame)
        outputs = self.rknn.inference(inputs=[input_data])
        decoded = self._decode_two_stage(outputs, frame.shape, scale, pad_left, pad_top)
        annotated = frame.copy()
        if decoded is None:
            cv2.putText(annotated, "No person/knife", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            self.last_annotated_frame = annotated
            return -1
        (left, top, right, bottom), kind, detector_conf = decoded
        crop = frame[top:bottom, left:right]
        classifier = self.person_classifier if kind == 0 else self.knife_classifier
        cls_index, cls_conf = self._classify_crop(crop, classifier)
        if cls_index < 0:
            self.last_annotated_frame = annotated
            return -1
        business_id = cls_index + 1 if kind == 0 else cls_index + 11
        display_id = cls_index + 1
        label = (
            f"{'person' if kind == 0 else 'knife'} ID={display_id} "
            f"det={detector_conf:.2f} cls={cls_conf:.2f}"
        )
        cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(annotated, label, (left, max(22, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
        self.last_annotated_frame = annotated
        return business_id

    def _prepare_rknn_input(self, frame):
        """Prepare an image for RKNN without cropping its source pixels.

        With ``YOLO_LETTERBOX=0`` (the 16:9 retrained model), the complete
        source is resized directly to the model's 16:9 shape, e.g. 1280x720 to
        640x360. With letterbox enabled, the legacy square model is supported.
        The returned geometry maps NMS boxes back to the original frame.
        """
        input_h, input_w = self.img_size
        frame_h, frame_w = frame.shape[:2]
        if frame_h <= 0 or frame_w <= 0:
            raise ValueError("输入图像尺寸无效")
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

    def detect_target(self, frame):
        if frame is None or frame.size == 0:
            print("[Vision WARN] 传入的画面为空，无法检测")
            return -1

        if self.two_stage and self.rknn is not None:
            try:
                return int(self._detect_two_stage(frame))
            except Exception as exc:
                print(f"[Vision ERROR] 两阶段推理失败: {exc}")
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
                            print(f"[Vision WARN] 非数字类别名称: {class_name}")

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
                        print(f"[Vision] PT 检测命中目标，业务 ID: {business_id}")
                        return business_id
                return -1
            except Exception as e:
                print(f"[Vision ERROR] PT 推理发生错误: {e}")
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
                    cv2.imshow("YOLO NPU Detection View", frame)
                    self.last_key = cv2.waitKey(1) & 0xFF
                
                if self.save_debug:
                    self.debug_count += 1
                    timestamp = int(time.time())
                    filename = f"logs/yolo_debug/detect_{timestamp}_{self.debug_count}.jpg"
                    cv2.imwrite(filename, frame)
                    if class_id != -1:
                        cv2.putText(frame, f"Target: {class_id}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imwrite(filename.replace('.jpg', '_result.jpg'), frame)
                
                if class_id in self.preset_classes:
                    print(f"[Vision] NPU 检测命中目标，类别 ID: {class_id}")
                    return class_id
                
                return -1
                    
            except Exception as e:
                print(f"[Vision ERROR] NPU 推理发生错误: {e}")
                import traceback
                traceback.print_exc()
                
                # 【重要】发生错误时，尝试重置摄像头
                # 但不要在这里重置，应该由上层处理
                return -1

        # ---- 情况 2：未加载 NPU 模型时的模拟模式 ----
        else:
            # 缩放到合适大小展示
            resized_bgr = cv2.resize(frame, self.img_size)
            h, w, _ = resized_bgr.shape
            
            # 在模拟模式下，直接返回预设的第一个类别ID（用于测试流程）
            # 实际使用时，这里应该替换为真实的模拟逻辑
            simulated_result = 1  # 假设检测到类别1
            
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
            
            print(f"[Vision Mock] 模拟检测完成，返回假定目标: {simulated_result}")
            
            if simulated_result in self.preset_classes:
                return simulated_result
            return -1

    def post_process(
        self, outputs, draw_frame=None, conf_threshold=None,
        nms_threshold=None, scale=1.0, pad_left=0, pad_top=0,
    ):
        """
        通用的 NumPy 向量化后处理（并可选在 draw_frame 上直接绘制出检测框）
        """
        if not outputs or len(outputs) == 0:
            return -1
        if conf_threshold is None:
            conf_threshold = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.55"))
        if nms_threshold is None:
            nms_threshold = float(os.environ.get("YOLO_NMS_THRESHOLD", "0.55"))

        try:
            # 提取第一个输出张量
            output = outputs[0]
            output = np.squeeze(output)  # 降维 -> (channels, anchors)
            
            # 保证形状为 (num_anchors, 4 + num_classes)
            if output.shape[0] < output.shape[1]:
                output = np.transpose(output)
            
            # 拆分边界框坐标和分类置信度
            boxes = output[:, :4].astype(np.float32, copy=False)
            scores = output[:, 4:]
            # Support exports that return normalized xywh as well as pixel xywh.
            if boxes.size and float(np.nanmax(np.abs(boxes))) <= 2.0:
                input_h, input_w = self.img_size
                boxes = boxes.copy()
                boxes[:, (0, 2)] *= float(input_w)
                boxes[:, (1, 3)] *= float(input_h)

            # 1. 向量化计算最大类别和得分
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)
            
            # 2. 粗筛低置信度数据
            mask = confidences > conf_threshold
            if not np.any(mask):
                return -1
            
            valid_boxes = boxes[mask]
            valid_confs = confidences[mask]
            valid_class_ids = class_ids[mask]
            
            # 3. 将 (center_x, center_y, w, h) 转换为 (left_x, top_y, w, h)
            x_left = valid_boxes[:, 0] - valid_boxes[:, 2] / 2.0
            y_top = valid_boxes[:, 1] - valid_boxes[:, 3] / 2.0
            widths = valid_boxes[:, 2]
            heights = valid_boxes[:, 3]
            
            nms_boxes = np.column_stack((x_left, y_top, widths, heights)).tolist()
            nms_scores = valid_confs.tolist()
            
            # 4. 调用 OpenCV NMS 过滤重叠框
            indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, conf_threshold, nms_threshold)
            
            if len(indices) > 0:
                indices = np.array(indices).flatten()
                
                # 【画框绘制逻辑】
                if draw_frame is not None:
                    for idx in indices:
                        # YOLO 张量使用 0-based 类别索引；项目其余模块和
                        # RESULT_MAP 使用 1-based 业务 ID。
                        class_index = int(valid_class_ids[idx])
                        if class_index < 0 or class_index >= len(self.CLASS_INDEX_TO_BUSINESS_ID):
                            continue
                        cls_id = self.CLASS_INDEX_TO_BUSINESS_ID[class_index]
                        conf = float(valid_confs[idx])
                        
                        # 只绘制我们在 preset_classes 里注册的目标
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
                            
                            # 绘制绿色的矩形检测框
                            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            # 标注类别 ID 颜色和置信度数值
                            label = f"ID: {cls_id} ({conf:.2f})"
                            cv2.putText(draw_frame, label, (x, y - 8), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 优先返回预设类别中的目标
                for idx in indices:
                    class_index = int(valid_class_ids[idx])
                    if class_index < 0 or class_index >= len(self.CLASS_INDEX_TO_BUSINESS_ID):
                        continue
                    cls_id = self.CLASS_INDEX_TO_BUSINESS_ID[class_index]
                    if cls_id in self.preset_classes:
                        return cls_id
                
                # 如果没有预设类别，返回第一个检测到的
                best_idx = indices[0]
                class_index = int(valid_class_ids[best_idx])
                if class_index < 0 or class_index >= len(self.CLASS_INDEX_TO_BUSINESS_ID):
                    return -1
                detected_class = self.CLASS_INDEX_TO_BUSINESS_ID[class_index]
                return detected_class

        except Exception as e:
            print(f"[Vision PostProcess Error] 后处理解析异常: {e}")
            print("提示：如果遇到维度不匹配错误，可能与您的模型导出/转换参数有关。")
            import traceback
            traceback.print_exc()
        
        return -1
    
    def release(self):
        """释放RKNN资源"""
        for runtime in (self.person_classifier, self.knife_classifier):
            if runtime is not None and runtime is not self.rknn:
                try:
                    runtime.release()
                except Exception:
                    pass
        if self.rknn is not None:
            try:
                self.rknn.release()
                print("[Vision] RKNN资源已释放")
            except:
                pass
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()


def _parse_camera(value):
    try:
        return int(value)
    except ValueError:
        return value


def _ensure_pt_runtime(model_path):
    """直接运行脚本时，为 PT 模型切换到项目虚拟环境。"""
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
            f"当前 Python 未安装 ultralytics，且找不到虚拟环境: {venv_python}"
        )
    print(f"[TEST] 切换到项目虚拟环境: {venv_python}", flush=True)
    os.execv(
        str(venv_python),
        [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
    )


def main():
    parser = argparse.ArgumentParser(description="RKNN YOLO 目标识别测试")
    parser.add_argument("--camera", default="2", help="摄像头编号或设备路径")
    parser.add_argument("--model", default=None, help="RKNN 模型路径")
    parser.add_argument(
        "--frames", type=int, default=0,
        help="最多测试帧数，0 表示持续运行到按 Q 或 Esc",
    )
    parser.add_argument(
        "--classes",
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
        help="关注的业务类别ID",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="不显示检测窗口"
    )
    parser.add_argument("--save-debug", action="store_true", help="保存调试图片")
    args = parser.parse_args()

    selected_model = args.model or str(
        Path(__file__).resolve().parent.parent
        / "models"
        / "2cls_640x384_fp16.rknn"
    )
    _ensure_pt_runtime(selected_model)

    preset_classes = tuple(
        int(value.strip()) for value in args.classes.split(",") if value.strip()
    )
    detector = YoloDetector(
        model_path=selected_model,
        preset_classes=preset_classes,
        show_window=not args.no_show,
        save_debug=args.save_debug,
    )
    if detector.rknn is None and detector.pt_model is None:
        detector.release()
        raise RuntimeError("RKNN 模型未加载，拒绝使用模拟结果进行模型测试")

    source = _parse_camera(args.camera)
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    camera = cv2.VideoCapture(source, backend)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        detector.release()
        raise RuntimeError(f"无法打开摄像头: {source}")

    interrupted = False
    try:
        frame_index = 0
        while args.frames <= 0 or frame_index < args.frames:
            frame_index += 1
            ok, frame = camera.read()
            if not ok or frame is None or frame.size == 0:
                print("[TEST WARN] 摄像头取帧失败")
                continue
            result = detector.detect_target(frame)
            if result != -1 or frame_index == 1 or frame_index % 30 == 0:
                print(f"[TEST] frame={frame_index} result={result}")
            if detector.last_key in (ord("q"), ord("Q"), 27):
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        camera.release()
        detector.release()
    if interrupted:
        print("[TEST] 用户中止")
    else:
        print("[TEST] 测试结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
