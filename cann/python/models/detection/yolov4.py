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

MODEL_WIDTH = 416
MODEL_HEIGHT = 416
CONF_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.8
TOP_K = 10
DEBUG = os.environ.get('YOLOV4_DEBUG', '0') == '1'


class YOLOv4Detect(BaseModel):
    def __init__(self, model_path):
        super().__init__(model_path)
        self._model_width = MODEL_WIDTH
        self._model_height = MODEL_HEIGHT
        self._orig_size = (MODEL_WIDTH, MODEL_HEIGHT)

    @staticmethod
    def _rgb_to_nv12(rgb):
        h, w, _ = rgb.shape
        rgb_float = rgb.astype(np.float32)
        r, g, b = rgb_float[:,:,0], rgb_float[:,:,1], rgb_float[:,:,2]

        y = 0.299 * r + 0.587 * g + 0.114 * b

        u = -0.169 * r - 0.331 * g + 0.5 * b + 128
        v = 0.5 * r - 0.419 * g - 0.081 * b + 128

        u_small = u[::2, ::2]
        v_small = v[::2, ::2]

        h_half, w_half = u_small.shape

        uv = np.zeros((h_half, w), dtype=np.uint8)
        uv[:, 0::2] = np.clip(u_small, 0, 255).astype(np.uint8)
        uv[:, 1::2] = np.clip(v_small, 0, 255).astype(np.uint8)

        y_uint8 = np.clip(y, 0, 255).astype(np.uint8)

        nv12 = np.vstack([y_uint8, uv])
        return nv12.flatten()

    def pre_process(self, image_path):
        img = np.array(Image.open(image_path).convert('RGB'))
        if img is None or img.size == 0:
            raise FileNotFoundError("无法读取图像: %s" % image_path)

        orig_h, orig_w = img.shape[:2]
        self._orig_size = (orig_w, orig_h)

        self.x_scale = orig_w / MODEL_WIDTH
        self.y_scale = orig_h / MODEL_HEIGHT

        img_resized = np.array(Image.fromarray(img).resize(
            (MODEL_WIDTH, MODEL_HEIGHT), Image.BILINEAR))

        img_nv12 = self._rgb_to_nv12(img_resized)

        img_input = np.expand_dims(img_nv12, axis=0).astype(np.uint8)

        if DEBUG:
            print("[DEBUG] Image preprocessed: shape=%s dtype=%s, size=%d" % (
                img_input.shape, img_input.dtype, img_input.size))

        return img_input

    def inference(self, input_data):
        return self._model.execute([input_data])

    def post_process(self, infer_output):
        if DEBUG:
            print("[DEBUG] infer_output length: %d" % len(infer_output))
            for idx, out in enumerate(infer_output):
                o = out[0] if isinstance(out, (list, tuple)) else out
                print("[DEBUG] output[%d]: shape=%s, dtype=%s, min=%.4f, max=%.4f" % (
                    idx, o.shape, o.dtype, float(o.min()), float(o.max())))

        results = []

        if len(infer_output) >= 2 and infer_output[1].size > 0:
            out1 = infer_output[1]
            if out1.ndim >= 2 and out1.shape[-1] == 1:
                box_num = int(out1.flatten()[0])
                if DEBUG:
                    print("[DEBUG] YOLOv3 style output: box_num=%d" % box_num)
                if box_num > 0 and len(infer_output) >= 1:
                    box_info = infer_output[0].flatten()
                    if DEBUG:
                        print("[DEBUG] box_info shape=%s, first 20 values: %s" % (box_info.shape, box_info[:20]))
                    scalex = self._orig_size[0] / MODEL_WIDTH
                    scaley = self._orig_size[1] / MODEL_HEIGHT
                    if scalex > scaley:
                        scaley = scalex
                    for n in range(min(box_num, 1000)):
                        try:
                            ids = int(box_info[5 * box_num + n])
                            score = float(box_info[4 * box_num + n])
                            if score < CONF_THRESHOLD:
                                continue
                            x1 = float(box_info[0 * box_num + n]) * scalex
                            y1 = float(box_info[1 * box_num + n]) * scaley
                            x2 = float(box_info[2 * box_num + n]) * scalex
                            y2 = float(box_info[3 * box_num + n]) * scaley
                            x1 = max(0, min(self._orig_size[0], x1))
                            y1 = max(0, min(self._orig_size[1], y1))
                            x2 = max(0, min(self._orig_size[0], x2))
                            y2 = max(0, min(self._orig_size[1], y2))
                            if (x2 - x1) < 5 or (y2 - y1) < 5:
                                continue
                            results.append({
                                "class_id": ids,
                                "label": LABELS_COCO[ids] if ids < len(LABELS_COCO) else str(ids),
                                "confidence": score,
                                "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
                            })
                        except (IndexError, ValueError):
                            break
                    if DEBUG:
                        print("[DEBUG] YOLOv3 style parsed: %d detections" % len(results))
                    results = self._nms_results(results, iou_thresh=NMS_IOU_THRESHOLD)
                    results.sort(key=lambda x: x["confidence"], reverse=True)
                    return results[:TOP_K]

        if len(infer_output) >= 2:
            classes = infer_output[0][0]
            boxes = infer_output[1][0]
            if DEBUG:
                print("[DEBUG] classes shape=%s, boxes shape=%s" % (classes.shape, boxes.shape))

            if classes.ndim == 2 and classes.shape[1] == len(LABELS_COCO):
                if DEBUG:
                    print("[DEBUG] Standard YOLOv4 format detected")
                num_anchors = classes.shape[0]
                for i in range(num_anchors):
                    class_probs = classes[i]
                    max_idx = np.argmax(class_probs)
                    max_score = float(class_probs[max_idx])
                    if max_score < CONF_THRESHOLD:
                        continue
                    cx = float(boxes[i][0])
                    cy = float(boxes[i][1])
                    w = float(boxes[i][2])
                    h = float(boxes[i][3])
                    cx_px = cx * self.x_scale
                    cy_px = cy * self.y_scale
                    w_px = w * self.x_scale
                    h_px = h * self.y_scale
                    x1 = cx_px - w_px / 2
                    y1 = cy_px - h_px / 2
                    x2 = cx_px + w_px / 2
                    y2 = cy_px + h_px / 2
                    x1 = max(0, min(self._orig_size[0], x1))
                    y1 = max(0, min(self._orig_size[1], y1))
                    x2 = max(0, min(self._orig_size[0], x2))
                    y2 = max(0, min(self._orig_size[1], y2))
                    if (x2 - x1) < 10 or (y2 - y1) < 10:
                        continue
                    results.append({
                        "class_id": int(max_idx),
                        "label": LABELS_COCO[max_idx] if max_idx < len(LABELS_COCO) else str(max_idx),
                        "confidence": max_score,
                        "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
                    })
                if DEBUG:
                    print("[DEBUG] Standard format parsed: %d detections" % len(results))

        if len(infer_output) == 1:
            out = infer_output[0]
            if out.ndim == 1 and out.size > 100:
                if DEBUG:
                    print("[DEBUG] Single output, trying flat format")
                total = out.size
                if total % 85 == 0:
                    num_boxes = total // 85
                    for i in range(num_boxes):
                        base = i * 85
                        x1, y1, x2, y2 = float(out[base]), float(out[base+1]), float(out[base+2]), float(out[base+3])
                        score = float(out[base+4])
                        cls_id = int(out[base+5])
                        if score < CONF_THRESHOLD:
                            continue
                        x1 = max(0, min(self._orig_size[0], x1 * self.x_scale))
                        y1 = max(0, min(self._orig_size[1], y1 * self.y_scale))
                        x2 = max(0, min(self._orig_size[0], x2 * self.x_scale))
                        y2 = max(0, min(self._orig_size[1], y2 * self.y_scale))
                        results.append({
                            "class_id": cls_id,
                            "label": LABELS_COCO[cls_id] if cls_id < len(LABELS_COCO) else str(cls_id),
                            "confidence": score,
                            "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
                        })
                    if DEBUG:
                        print("[DEBUG] Flat format parsed: %d detections" % len(results))

        if DEBUG:
            print("[DEBUG] Total raw detections (score >= %.2f): %d" % (CONF_THRESHOLD, len(results)))

        results = self._nms_results(results, iou_thresh=NMS_IOU_THRESHOLD)
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:TOP_K]

    def _nms_results(self, results, iou_thresh=0.8):
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
