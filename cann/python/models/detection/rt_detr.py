import os
import sys
import numpy as np
from PIL import Image

from ...base import BaseModel

LABELS_COCO = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

MODEL_WIDTH = 640
MODEL_HEIGHT = 640
CONF_THRESHOLD = 0.30
NMS_IOU_THRESHOLD = 0.45
TOP_K = 10
MAX_PER_CLASS = 2
DEBUG = os.environ.get('RTDETR_DEBUG', '0') == '1'


class RTDETRDetect(BaseModel):
    def __init__(self, model_path):
        super().__init__(model_path)
        self._model_width = MODEL_WIDTH
        self._model_height = MODEL_HEIGHT
        self._orig_size = (MODEL_WIDTH, MODEL_HEIGHT)

    def pre_process(self, image_path):
        img = np.array(Image.open(image_path).convert('RGB'))
        if img is None or img.size == 0:
            raise FileNotFoundError("无法读取图像: %s" % image_path)

        orig_h, orig_w = img.shape[:2]
        self._orig_size = (orig_w, orig_h)

        img_resized = np.array(Image.fromarray(img).resize(
            (self._model_width, self._model_height), Image.BILINEAR))

        img_normalized = img_resized.astype(np.float32) / 255.0
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        img_input = np.expand_dims(img_chw, axis=0).astype(np.float32)

        orig_target_sizes = np.array([[orig_w, orig_h]], dtype=np.int64)

        if DEBUG:
            print("[DEBUG] Image preprocessed: shape=%s dtype=%s" % (img_input.shape, img_input.dtype))

        return img_input, orig_target_sizes

    def inference(self, input_data):
        image, target_sizes = input_data
        return self._model.execute([image, target_sizes])

    def post_process(self, infer_output):
        labels = infer_output[0][0]
        boxes = infer_output[1][0]
        scores = infer_output[2][0]

        if DEBUG:
            print("[DEBUG] labels shape=%s" % str(labels.shape))
            print("[DEBUG] boxes shape=%s" % str(boxes.shape))
            print("[DEBUG] scores shape=%s, min=%.3f max=%.3f" % (
                scores.shape, float(scores.min()), float(scores.max())))

        results = []
        for i in range(len(labels)):
            score = float(scores[i])
            if score < CONF_THRESHOLD:
                continue

            label_id = int(labels[i])
            if label_id < 0 or label_id >= len(LABELS_COCO):
                continue

            x1, y1, x2, y2 = float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])
            label_name = LABELS_COCO[label_id]

            results.append({
                "class_id": label_id,
                "label": label_name,
                "confidence": score,
                "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
            })

        if DEBUG:
            print("[DEBUG] 检测数 (score >= %.2f): %d" % (CONF_THRESHOLD, len(results)))

        results = self._nms_results(results, iou_thresh=NMS_IOU_THRESHOLD)
        results = self._keep_best_per_class(results, max_per_class=MAX_PER_CLASS)

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:TOP_K]

    def _keep_best_per_class(self, results, max_per_class=2):
        if not results or max_per_class <= 0:
            return results
        class_groups = {}
        for r in results:
            class_groups.setdefault(r["class_id"], []).append(r)
        keep = []
        for cid, group in class_groups.items():
            group_sorted = sorted(group, key=lambda x: x["confidence"], reverse=True)
            keep.extend(group_sorted[:max_per_class])
        return keep

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
