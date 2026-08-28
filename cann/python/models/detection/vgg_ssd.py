import os
import sys
import numpy as np
from PIL import Image

from ...base import BaseModel

MODEL_WIDTH = 300
MODEL_HEIGHT = 300

LABELS_VOC = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

CONF_THRESHOLD = 0.05
NMS_IOU_THRESHOLD = 0.30
TOP_K = 5
MAX_PER_CLASS = 1
HIGH_CONF_THRESHOLD = 0.10
DEBUG = os.environ.get('SSD_DEBUG', '0') == '1'


class VGGSSDDetect(BaseModel):
    def __init__(self, model_path):
        super().__init__(model_path)
        self._model_width = MODEL_WIDTH
        self._model_height = MODEL_HEIGHT
        self._orig_size = (MODEL_WIDTH, MODEL_HEIGHT)
        self._is_aipp = None

    def _detect_aipp_mode(self):
        if self._is_aipp is not None:
            return self._is_aipp

        try:
            import acl
            model_desc = self._model._model_desc
            input_size = acl.mdl.get_input_size_by_index(model_desc, 0)
            input_size = int(input_size)
            if input_size < 1000000:
                self._is_aipp = True
                return True
            else:
                self._is_aipp = False
                return False
        except Exception:
            pass

        model_path_lower = self._model_path.lower()
        if 'vgg_ssd' in model_path_lower or 'aipp' in model_path_lower:
            self._is_aipp = True
            return True

        self._is_aipp = False
        return False

    def pre_process(self, image_path):
        pil_image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = pil_image.size
        self._orig_size = (orig_w, orig_h)

        is_aipp = self._detect_aipp_mode()

        if is_aipp:
            aipp_input_w = 304
            aipp_input_h = 300

            pil_resized = pil_image.resize(
                (aipp_input_w, aipp_input_h), Image.BILINEAR)
            img_array = np.array(pil_resized, dtype=np.uint8)
            img_array = np.transpose(img_array, (2, 0, 1))
            img_array = np.expand_dims(img_array, axis=0)

            if not img_array.flags['C_CONTIGUOUS']:
                img_array = np.ascontiguousarray(img_array)
            return img_array
        else:
            pil_image = pil_image.resize(
                (self._model_width, self._model_height), Image.BILINEAR)
            img_array = np.array(pil_image, dtype=np.float32)
            img_array = img_array[:, :, ::-1]
            mean = np.array([104.0, 117.0, 123.0], dtype=np.float32)
            img_array = img_array - mean
            img_array = np.transpose(img_array, (2, 0, 1))
            img_array = np.expand_dims(img_array, axis=0)

            if not img_array.flags['C_CONTIGUOUS']:
                img_array = np.ascontiguousarray(img_array)
            return img_array

    def inference(self, input_data):
        if isinstance(input_data, np.ndarray):
            return self._model.execute([input_data, ])
        return self._model.execute(input_data)

    def post_process(self, infer_output):
        if not (isinstance(infer_output, list) and len(infer_output) >= 2):
            return []

        out0 = np.array(infer_output[0]).flatten()
        out1 = np.array(infer_output[1]).flatten()

        if DEBUG:
            print("[DEBUG] out0 shape:", out0.shape, "values:", out0[:5])
            print("[DEBUG] out1 shape:", out1.shape)

        try:
            box_num = int(out0[0])
        except (ValueError, IndexError):
            return []

        det_count = len(out1) // 8

        orig_w, orig_h = self._orig_size
        results = []

        for b in range(det_count):
            base = b * 8
            if base + 7 >= len(out1):
                break

            label = int(out1[base + 1])
            score = float(out1[base + 2])
            x1 = float(out1[base + 3])
            y1 = float(out1[base + 4])
            x2 = float(out1[base + 5])
            y2 = float(out1[base + 6])

            if score < CONF_THRESHOLD:
                continue

            if label <= 0 or label >= 21:
                continue

            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))

            x1_px = int(x1 * orig_w)
            y1_px = int(y1 * orig_h)
            x2_px = int(x2 * orig_w)
            y2_px = int(y2 * orig_h)

            if (x2_px - x1_px) < 15 or (y2_px - y1_px) < 15:
                continue

            box_area = (x2_px - x1_px) * (y2_px - y1_px)
            img_area = orig_w * orig_h
            if box_area < img_area * 0.005:
                continue

            bottom_threshold = orig_h * 0.70
            if y1_px > bottom_threshold and score < 0.10:
                continue

            if (x1_px < 5 or y1_px < 5 or
                x2_px > orig_w - 5 or y2_px > orig_h - 5):
                if score < 0.10:
                    continue

            if label < len(LABELS_VOC):
                label_name = LABELS_VOC[label]
            else:
                label_name = str(label)

            results.append({
                "class_id": label,
                "label": label_name,
                "confidence": score,
                "bbox": [x1_px, y1_px, x2_px, y2_px],
            })

        results = self._nms_results(results, iou_thresh=NMS_IOU_THRESHOLD)
        results = self._keep_best_per_class(results, max_per_class=MAX_PER_CLASS)

        if results:
            scores = sorted([r["confidence"] for r in results], reverse=True)
            high_conf_count = sum(1 for s in scores if s >= HIGH_CONF_THRESHOLD)
            final_k = min(max(high_conf_count, 1), 5)
        else:
            final_k = TOP_K

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:final_k]

    def _keep_best_per_class(self, results, max_per_class=1):
        if not results or max_per_class <= 0:
            return results

        class_groups = {}
        for r in results:
            cid = r["class_id"]
            class_groups.setdefault(cid, []).append(r)

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
            cid = r["class_id"]
            class_groups.setdefault(cid, []).append(r)

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
