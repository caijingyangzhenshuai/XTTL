import os
import sys
import numpy as np
from PIL import Image

from ...base import BaseModel

LABELS_COCO = [
    "person",
    "bicycle", "car", "motorbike", "aeroplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
    "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

MODEL_WIDTH = 416
MODEL_HEIGHT = 416
CONF_THRESHOLD = 0.4
TOP_K = 10


class YOLOv3Detect(BaseModel):
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

        img_resized = np.array(Image.fromarray(img).resize(
            (MODEL_WIDTH, MODEL_HEIGHT), Image.BILINEAR))

        img_nv12 = self._rgb_to_nv12(img_resized)
        img_input = np.expand_dims(img_nv12, axis=0).astype(np.uint8)

        image_info = np.array(
            [MODEL_WIDTH, MODEL_HEIGHT, MODEL_WIDTH, MODEL_HEIGHT],
            dtype=np.float32)

        return [img_input, image_info]

    def inference(self, input_data):
        return self._model.execute(input_data)

    def post_process(self, infer_output):
        results = []
        box_num = int(infer_output[1][0, 0])
        if box_num > 0:
            box_info = infer_output[0].flatten()
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

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:TOP_K]
