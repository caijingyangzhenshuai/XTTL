import os
import sys
import numpy as np
from PIL import Image

from ...base import BaseModel

MODEL_WIDTH = 513
MODEL_HEIGHT = 513

PALETTE = np.array([
    (0, 0, 0),         (128, 0, 0),     (0, 128, 0),     (128, 128, 0),
    (0, 0, 128),       (128, 0, 128),   (0, 128, 128),   (128, 128, 128),
    (64, 0, 0),        (192, 0, 0),     (64, 128, 0),    (192, 128, 0),
    (64, 0, 128),      (192, 0, 128),   (64, 128, 128),  (192, 128, 128),
    (0, 64, 0),        (128, 64, 0),    (0, 192, 0),     (128, 192, 0),
    (0, 64, 128),
], dtype=np.uint8)

CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "dining table", "dog", "horse",
    "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor",
]

DEBUG = os.environ.get('DEEPLAB_DEBUG', '0') == '1'


class DeepLabV3Segment(BaseModel):
    def __init__(self, model_path):
        super().__init__(model_path)
        self._model_width = MODEL_WIDTH
        self._model_height = MODEL_HEIGHT
        self._orig_size = (MODEL_WIDTH, MODEL_HEIGHT)

    def pre_process(self, image_path):
        pil_img = Image.open(image_path).convert('RGB')
        orig_size = (pil_img.width, pil_img.height)
        self._orig_size = orig_size

        rgb_img = pil_img.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.BILINEAR)
        img = np.array(rgb_img).astype(np.int8)

        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)

        if DEBUG:
            print("[DEBUG] Image preprocessed: shape=%s dtype=%s" % (img.shape, img.dtype))

        return img

    def inference(self, input_data):
        return self._model.execute([input_data])

    def post_process(self, infer_output):
        result_img = infer_output[0].reshape(MODEL_HEIGHT, MODEL_WIDTH)
        seg_map = result_img.astype(np.uint8)
        seg_map = np.clip(seg_map, 0, len(PALETTE) - 1)

        orig_w, orig_h = self._orig_size
        seg_map_full = self._resize_seg_map(seg_map, orig_w, orig_h)
        class_info = self._collect_class_stats(seg_map_full)

        return {
            "image_size": list(self._orig_size),
            "class_info": class_info,
            "seg_map": seg_map_full,
        }

    @staticmethod
    def _resize_seg_map(seg_map, new_w, new_h):
        pil = Image.fromarray(seg_map.astype(np.uint8), mode='P')
        pil = pil.resize((new_w, new_h), Image.NEAREST)
        return np.array(pil)

    @classmethod
    def generate_overlay(cls, image_path, seg_map, alpha=0.6):
        orig_rgb = np.array(Image.open(image_path).convert('RGB'))
        color_map = PALETTE[seg_map]
        overlay = (orig_rgb.astype(np.float32) * (1 - alpha) +
                   color_map.astype(np.float32) * alpha).astype(np.uint8)
        return overlay

    @classmethod
    def save_seg_image(cls, image_path, seg_map, out_path, alpha=0.6):
        overlay = cls.generate_overlay(image_path, seg_map, alpha)
        Image.fromarray(overlay.astype(np.uint8), mode='RGB').save(out_path)
        return out_path

    @staticmethod
    def _collect_class_stats(seg_map, threshold_ratio=0.01):
        total = seg_map.size
        class_info = []
        for c in np.unique(seg_map):
            cnt = int(np.sum(seg_map == c))
            ratio = cnt * 100.0 / total
            if ratio < threshold_ratio:
                continue
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else "class_%d" % c
            class_info.append({
                "id": int(c),
                "name": name,
                "pixels": cnt,
                "ratio": round(ratio, 2),
            })
        class_info.sort(key=lambda x: x["pixels"], reverse=True)
        return class_info
