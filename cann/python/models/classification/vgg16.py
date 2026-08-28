import os
import sys
import numpy as np
from PIL import Image

from ...base import BaseModel

MODEL_WIDTH = 224
MODEL_HEIGHT = 256
TOP_K = 5
CLASSES = ['dog', 'cat']
DEBUG = os.environ.get('VGG16_DEBUG', '0') == '1'


class VGG16Classify(BaseModel):
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

        y = 0.257 * r + 0.504 * g + 0.098 * b + 16
        u = -0.148 * r - 0.291 * g + 0.439 * b + 128
        v = 0.439 * r - 0.368 * g - 0.071 * b + 128

        y = np.clip(y, 0, 255).astype(np.uint8)
        u = np.clip(u, 0, 255).astype(np.uint8)
        v = np.clip(v, 0, 255).astype(np.uint8)

        u_small = u[::2, ::2]
        v_small = v[::2, ::2]

        h_half, w_half = u_small.shape
        uv = np.zeros((h_half, w), dtype=np.uint8)
        uv[:, 0::2] = u_small
        uv[:, 1::2] = v_small

        nv12 = np.vstack([y, uv])
        return nv12.flatten()

    def pre_process(self, image_path):
        img = np.array(Image.open(image_path).convert('RGB'))
        if img is None or img.size == 0:
            raise FileNotFoundError("无法读取图像: %s" % image_path)

        orig_h, orig_w = img.shape[:2]
        self._orig_size = (orig_w, orig_h)

        img_resized = np.array(Image.fromarray(img).resize(
            (self._model_width, self._model_height), Image.BILINEAR))

        img_nv12 = self._rgb_to_nv12(img_resized)
        img_input = np.expand_dims(img_nv12, axis=0).astype(np.uint8)

        if DEBUG:
            print("[DEBUG] Image preprocessed: shape=%s dtype=%s" % (img_input.shape, img_input.dtype))

        return img_input

    def inference(self, input_data):
        return self._model.execute([input_data])

    def post_process(self, infer_output):
        infer_result = infer_output[0]
        vals = infer_result.flatten()
        top_indices = vals.argsort()[-TOP_K:][::-1]

        results = []
        for idx in top_indices:
            idx = int(idx)
            label = CLASSES[idx] if idx < len(CLASSES) else str(idx)
            results.append({
                "class_id": idx,
                "label": label,
                "confidence": int(round(float(vals[idx]) * 100))
            })
        return results
