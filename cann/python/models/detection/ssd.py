import os
import sys
import numpy as np
from PIL import Image

from ...base import BaseModel

# 输入规格：ssd-10.onnx 真实输入为 image，形状 NCHW 1x3x1200x1200
MODEL_WIDTH = 1200
MODEL_HEIGHT = 1200

# 输出规格（模型内部已完成解码/NMS，输出为后处理结果）：
#   bboxes: float32[1, nbox, 4]
#   labels: int64[1, nbox]
#   scores: float32[1, nbox]
# bbox 坐标约定：
#   1 = 归一化到 [0,1]（默认，映射回原图）
#   0 = 相对模型输入 1200x1200 的绝对像素（直接按比例缩放回原图）
BBOX_NORMALIZED = os.environ.get('SSD_BBOX_NORMALIZED', '1') == '1'

LABELS_COCO = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

NUM_CLASSES = len(LABELS_COCO) - 1  # 不含 background 的真实类别数

CONF_THRESHOLD = 0.30
NMS_IOU_THRESHOLD = 0.45
TOP_K = 10
DEBUG = os.environ.get('SSD_DEBUG', '0') == '1'


class SSDDetect(BaseModel):
    def __init__(self, model_path):
        super().__init__(model_path)
        self._model_width = MODEL_WIDTH
        self._model_height = MODEL_HEIGHT
        self._orig_size = (MODEL_WIDTH, MODEL_HEIGHT)

    def pre_process(self, image_path):
        pil_image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = pil_image.size
        self._orig_size = (orig_w, orig_h)

        pil_resized = pil_image.resize(
            (self._model_width, self._model_height), Image.BILINEAR)
        # 归一化到 [-1,1]？若模型要求 [0,1] 或 mean/std，请按模型文档调整
        img_array = np.array(pil_resized, dtype=np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        if not img_array.flags['C_CONTIGUOUS']:
            img_array = np.ascontiguousarray(img_array)

        if DEBUG:
            print("[DEBUG] preprocessed shape=%s dtype=%s" %
                  (img_array.shape, img_array.dtype))
        return img_array

    def inference(self, input_data):
        if isinstance(input_data, np.ndarray):
            return self._model.execute([input_data, ])
        return self._model.execute(input_data)

    def post_process(self, infer_output):
        if not (isinstance(infer_output, list) and len(infer_output) >= 3):
            return []

        bboxes = np.array(infer_output[0])
        labels = np.array(infer_output[1])
        scores = np.array(infer_output[2])

        # 去掉 batch 维：[1, nbox, ...] -> [nbox, ...]
        if bboxes.ndim == 3:
            bboxes = bboxes[0]
        if labels.ndim >= 2:
            labels = labels[0]
        if scores.ndim >= 2:
            scores = scores[0]

        bboxes = np.asarray(bboxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float32)

        if DEBUG:
            print("[DEBUG] bboxes shape=%s labels shape=%s scores shape=%s" %
                  (bboxes.shape, labels.shape, scores.shape))

        nbox = min(len(bboxes), len(labels), len(scores))
        if nbox <= 0:
            return []

        orig_w, orig_h = self._orig_size
        results = []

        for b in range(nbox):
            score = float(scores[b])
            if score < CONF_THRESHOLD:
                continue

            label = int(labels[b])
            if label < 0 or label >= NUM_CLASSES:
                continue

            x1, y1, x2, y2 = [float(v) for v in bboxes[b]]
            if BBOX_NORMALIZED:
                x1_px = int(x1 * orig_w)
                y1_px = int(y1 * orig_h)
                x2_px = int(x2 * orig_w)
                y2_px = int(y2 * orig_h)
            else:
                # bbox 相对模型输入 1200x1200 的绝对坐标
                scale_x = orig_w / float(MODEL_WIDTH)
                scale_y = orig_h / float(MODEL_HEIGHT)
                x1_px = int(x1 * scale_x)
                y1_px = int(y1 * scale_y)
                x2_px = int(x2 * scale_x)
                y2_px = int(y2 * scale_y)

            if (x2_px - x1_px) < 5 or (y2_px - y1_px) < 5:
                continue

            results.append({
                "class_id": label,
                "label": LABELS_COCO[label],
                "confidence": score,
                "bbox": [x1_px, y1_px, x2_px, y2_px],
            })

        # NMS 去重：合并同一类别的重叠框，避免同一目标输出多个框
        results = self._nms_results(results, iou_thresh=NMS_IOU_THRESHOLD)

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:TOP_K]

    def _nms_results(self, results, iou_thresh=0.45):
        if not results:
            return results
        class_groups = {}
        for r in results:
            class_groups.setdefault(r["class_id"], []).append(r)
        keep = []
        for cid, group in class_groups.items():
            group_sorted = sorted(group, key=lambda x: x["confidence"], reverse=True)
            kept = []
            for r in group_sorted:
                overlap = False
                for k in kept:
                    if self._iou(r["bbox"], k["bbox"]) > iou_thresh:
                        overlap = True
                        break
                if not overlap:
                    kept.append(r)
            keep.extend(kept)
        return keep

    @staticmethod
    def _iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / float(area1 + area2 - inter)