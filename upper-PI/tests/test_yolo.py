# pose_visual_test.py
import argparse
import cv2
import numpy as np
from pathlib import Path
from rknnlite.api import RKNNLite

# ===================== 全局常量（和原版保持完全一致，不要修改） =====================
INPUT_H, INPUT_W = 384, 640
PERSON_NAMES = [f"person_{i:02d}" for i in range(1, 11)]
KNIFE_NAMES = [f"knife_{i:02d}" for i in range(1, 21)]

# ===================== 1. Letterbox预处理：原图 -> 模型输入 =====================
def prepare_input(frame):
    h, w = frame.shape[:2]
    scale = min(INPUT_W / w, INPUT_H / h)
    rw, rh = round(w * scale), round(h * scale)
    if min(rw, rh) < 1:
        raise ValueError("image aspect ratio unsupported")
    left, top = (INPUT_W - rw) // 2, (INPUT_H - rh) // 2
    canvas = np.full((INPUT_H, INPUT_W, 3), 114, np.uint8)
    canvas[top:top + rh, left:left + rw] = cv2.resize(frame, (rw, rh))
    # BGR转RGB，NHWC
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]
    return np.ascontiguousarray(canvas_rgb), (rw / w, rh / h), left, top

# ===================== 2. decode_pose：RKNN输出解析 + 类别独立NMS =====================
def decode_pose(output, conf=0.25, iou=0.45):
    a = np.asarray(output, dtype=np.float32)
    if a.ndim != 3 or a.shape[0] != 1:
        raise ValueError(f"expected [1,18,N] or [1,N,18], got {a.shape}")
    a = a[0]
    if a.shape[0] == 18:
        a = a.T
    elif a.shape[1] != 18:
        raise ValueError(f"expected 18 pose channels, got {a.shape}")

    valid = np.isfinite(a[:, :6]).all(axis=1)
    valid &= (a[:, 2] > 0) & (a[:, 3] > 0)
    a = a[valid]
    if not len(a):
        return []

    scores = a[:, 4:6]
    if np.any(scores < -0.01) or np.any(scores > 1.01):
        raise ValueError("class scores are not probabilities; check export/model")
    kinds, confidence = scores.argmax(1), scores.max(1)
    chosen = confidence >= conf
    a, kinds, confidence = a[chosen], kinds[chosen], confidence[chosen]
    if not len(a):
        return []

    xywh = a[:, :4].copy()
    xywh[:, :2] -= xywh[:, 2:] / 2

    indices = []
    for kind in (0, 1):
        group = np.flatnonzero(kinds == kind)
        if not len(group):
            continue
        keep = cv2.dnn.NMSBoxes(xywh[group].tolist(), confidence[group].tolist(), conf, iou)
        if len(keep):
            indices.extend(group[np.asarray(keep).reshape(-1)].tolist())
    return [(a[i, :4].copy(), a[i, 6:18].reshape(4, 3).copy(),
             int(kinds[i]), float(confidence[i]))
            for i in sorted(indices, key=lambda i: -float(confidence[i]))]

# ===================== 3. 坐标反变换：模型图坐标映射回原图 =====================
def map_to_frame(box, keypoints, scales, pad_left, pad_top, frame_shape):
    sx, sy = scales
    cx, cy, bw, bh = map(float, box)
    corners = np.float32([[cx - bw / 2, cy - bh / 2],
                          [cx + bw / 2, cy + bh / 2]])
    corners = (corners - [pad_left, pad_top]) / [sx, sy]
    h, w = frame_shape[:2]
    x1, y1 = np.floor(np.clip(corners[0], [0, 0], [w, h])).astype(int)
    x2, y2 = np.ceil(np.clip(corners[1], [0, 0], [w, h])).astype(int)

    points = np.asarray(keypoints, dtype=np.float32).copy()
    points[:, :2] = (points[:, :2] - [pad_left, pad_top]) / [sx, sy]
    return (int(x1), int(y1), int(x2), int(y2)), points

# ===================== 4. 矫正裁剪（bbox / perspective二选一） =====================
def rectify_crop(frame, box, keypoints, mode="bbox", point_conf=0.5):
    x1, y1, x2, y2 = box
    fallback = frame[y1:y2, x1:x2]
    if mode == "bbox":
        return fallback, "bbox"
    p = np.asarray(keypoints, dtype=np.float32)
    if not np.isfinite(p).all() or np.any(p[:, 2] < point_conf):
        return fallback, "bbox_fallback_low_points"
    q = np.ascontiguousarray(p[:, :2])
    h, w = frame.shape[:2]
    if np.any(q < 0) or np.any(q[:, 0] > w - 1) or np.any(q[:, 1] > h - 1):
        return fallback, "bbox_fallback_outside"
    contour = q.reshape(-1, 1, 2)
    signed_area = float(cv2.contourArea(contour, oriented=True))
    edges = np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)
    box_area = max(1, (x2 - x1) * (y2 - y1))
    if (not cv2.isContourConvex(contour) or signed_area < 64
            or signed_area < box_area * 0.15 or signed_area > box_area * 2.5
            or float(edges.min()) < 8):
        return fallback, "bbox_fallback_bad_quad"
    width = round((np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2)
    height = round((np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2)
    width, height = max(8, min(width, 1600)), max(8, min(height, 1600))
    dst = np.float32([[0, 0], [width - 1, 0],
                      [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(q, dst)
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-9:
        return fallback, "bbox_fallback_bad_transform"
    crop = cv2.warpPerspective(frame, matrix, (width, height), borderValue=(114, 114, 114))
    return crop, "perspective"

# ===================== 5. MobileNetV3 分类器推理 =====================
def classify_crop(crop, classifier_rknn):
    crop_resized = cv2.resize(crop, (224, 224))
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)[None]
    crop_rgb = np.ascontiguousarray(crop_rgb, dtype=np.uint8)
    outputs = classifier_rknn.inference(inputs=[crop_rgb])
    values = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
    prob = np.exp(values - values.max())
    prob /= prob.sum()
    index = int(prob.argmax())
    return index, float(prob[index])

# ===================== 主推理入口 =====================
class PoseTwoStageDetector:
    def __init__(self, model_dir, rectify="bbox", det_conf=0.25, cls_conf=0.0, kpt_conf=0.5):
        self.rectify = rectify
        self.det_conf = det_conf
        self.cls_conf = cls_conf
        self.kpt_conf = kpt_conf
        # 模型路径
        pose_path = Path(model_dir) / "pose_4pt_640x384_fp16.rknn"
        person_cls_path = Path(model_dir) / "person_classifier_224_fp16.rknn"
        knife_cls_path = Path(model_dir) / "knife_classifier_224_fp16.rknn"

        self.rknn_pose = RKNNLite()
        self.rknn_person_cls = RKNNLite()
        self.rknn_knife_cls = RKNNLite()
        self.runtimes = [self.rknn_pose, self.rknn_person_cls, self.rknn_knife_cls]

        print(f"Loading {pose_path.name}")
        self.rknn_pose.load_rknn(str(pose_path))
        self.rknn_pose.init_runtime()
        print(f"Loading {person_cls_path.name}")
        self.rknn_person_cls.load_rknn(str(person_cls_path))
        self.rknn_person_cls.init_runtime()
        print(f"Loading {knife_cls_path.name}")
        self.rknn_knife_cls.load_rknn(str(knife_cls_path))
        self.rknn_knife_cls.init_runtime()
        print("✅ All RKNN models loaded!")

    def infer_frame(self, frame):
        """输入BGR原图，返回标注图 + 检测结果列表"""
        img_pre, scales, pad_left, pad_top = prepare_input(frame)
        pose_out = self.rknn_pose.inference(inputs=[img_pre])
        dets = decode_pose(pose_out[0], conf=self.det_conf)
        draw_img = frame.copy()
        det_result_list = []

        for box_model, kp_model, kind, det_conf in dets:
            box_ori, kp_ori = map_to_frame(box_model, kp_model, scales, pad_left, pad_top, frame.shape)
            x1, y1, x2, y2 = box_ori
            if (x2 - x1) < 8 or (y2 - y1) <8:
                continue
            crop_img, crop_mode = rectify_crop(frame, box_ori, kp_ori, self.rectify, self.kpt_conf)
            # 选择分类器
            if kind == 0:
                cls_idx, cls_conf = classify_crop(crop_img, self.rknn_person_cls)
                business_id = cls_idx + 1
                obj_name = PERSON_NAMES[cls_idx]
            else:
                cls_idx, cls_conf = classify_crop(crop_img, self.rknn_knife_cls)
                business_id = cls_idx + 11
                obj_name = KNIFE_NAMES[cls_idx]
            accepted = cls_conf >= self.cls_conf
            det_result_list.append({
                "business_id": business_id,
                "name": obj_name,
                "box": [x1,y1,x2,y2],
                "keypoints": kp_ori.tolist(),
                "det_conf": det_conf,
                "cls_conf": cls_conf,
                "crop_mode": crop_mode,
                "accepted": accepted
            })
            # 绘图
            color = (0,255,0) if accepted else (0,165,255)
            cv2.rectangle(draw_img, (x1,y1), (x2,y2), color, 2)
            # 绘制4个关键点 +编号
            for kid, (px,py,pc) in enumerate(kp_ori):
                if np.isfinite([px,py,pc]) and 0<=px<frame.shape[1] and 0<=py<frame.shape[0]:
                    cv2.circle(draw_img, (int(px),int(py)), 4, (0,255,255), -1)
                    cv2.putText(draw_img, str(kid), (int(px)+4, int(py)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255),1)
            label = f"{obj_name} ID={business_id} d={det_conf:.2f} c={cls_conf:.2f} {crop_mode}"
            cv2.putText(draw_img, label, (max(0, x1-10), max(20, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color,1)
        return draw_img, det_result_list

    def release(self):
        for rt in self.runtimes:
            rt.release()
        print("RKNN resources released.")

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Pose + MobileNet Visual Test")
    parser.add_argument(
        "--model-dir",
        default=str(Path(__file__).resolve().parent.parent / "models"),
        help="rknn model folder",
    )
    parser.add_argument("--rectify", choices=["bbox","perspective"], default="bbox")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--cls-conf", type=float, default=0.0)
    parser.add_argument("--kpt-conf", type=float, default=0.5)
    parser.add_argument("--image", type=str, default=None, help="single image test path")
    parser.add_argument("--cam-id", type=int, default=2)
    parser.add_argument("--cam-w", type=int, default=1280)
    parser.add_argument("--cam-h", type=int, default=720)
    args = parser.parse_args()

    detector = PoseTwoStageDetector(
        model_dir=args.model_dir,
        rectify=args.rectify,
        det_conf=args.det_conf,
        cls_conf=args.cls_conf,
        kpt_conf=args.kpt_conf
    )
    try:
        if args.image is not None:
            # 图片模式
            frame = cv2.imread(args.image)
            if frame is None:
                print(f"Cannot read image {args.image}")
                return
            draw, res_list = detector.infer_frame(frame)
            print("==== Detection Result ====")
            for item in res_list:
                print(item)
            cv2.imwrite("out_result.jpg", draw)
            print("Saved out_result.jpg")
            cv2.imshow("result", draw)
            cv2.waitKey(0)
        else:
            # 摄像头实时
            cap = cv2.VideoCapture(args.cam_id, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            if not cap.isOpened():
                print("Camera open failed!")
                return
            print(f"Camera {args.cam_id} opened {args.cam_w}x{args.cam_h}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                draw_img, results = detector.infer_frame(frame)
                # 控制台打印
                if len(results)>0:
                    print(f"Det: {results}", end="\r")
                cv2.imshow("Pose Two Stage Visual", draw_img)
                key = cv2.waitKey(1) &0xFF
                if key == ord('q'):
                    break
            cap.release()
    finally:
        detector.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
