import os
import sys
import numpy as np
from PIL import Image

from ...base import BaseModel

MODEL_WIDTH = 224
MODEL_HEIGHT = 224
TOP_K = 5
DEBUG = os.environ.get('RESNET_DEBUG', '0') == '1'


class ResNet50Classify(BaseModel):
    def __init__(self, model_path):
        super().__init__(model_path)
        self._model_width = MODEL_WIDTH
        self._model_height = MODEL_HEIGHT

    def pre_process(self, image_path):
        img = np.array(Image.open(image_path).convert('RGB'))
        if img is None or img.size == 0:
            raise FileNotFoundError("无法读取图像: %s" % image_path)

        img_resized = np.array(Image.fromarray(img).resize(
            (self._model_width, self._model_height), Image.BILINEAR))

        # ONNX ResNet50 通常需要 ImageNet 归一化：/255 后再减 mean/std
        # 这里先做 /255 归一化到 [0,1]
        img_float = img_resized.astype(np.float32) / 255.0

        # ImageNet mean/std 归一化（如果模型需要）
        # mean = [0.485, 0.456, 0.406]
        # std  = [0.229, 0.224, 0.225]
        # img_float = (img_float - mean) / std

        img_chw = np.transpose(img_float, (2, 0, 1))
        img_input = np.expand_dims(img_chw, axis=0).astype(np.float32)

        if DEBUG:
            print("[DEBUG] Image preprocessed: shape=%s dtype=%s" % (img_input.shape, img_input.dtype))

        return img_input

    def inference(self, input_data):
        return self._model.execute([input_data])

    def post_process(self, infer_output):
        output = infer_output[0]

        if output.ndim > 1:
            output = output.flatten()

        # 对输出做 softmax 归一化（ONNX 模型输出通常是 logits）
        exp_output = np.exp(output - np.max(output))
        probs = exp_output / np.sum(exp_output)

        if DEBUG:
            print("[DEBUG] output shape=%s, min=%.3f, max=%.3f" % (
                output.shape, float(output.min()), float(output.max())))
            print("[DEBUG] probs top5:", np.sort(probs)[::-1][:5])

        top_indices = np.argsort(probs)[::-1][:TOP_K]

        results = []
        for idx in top_indices:
            idx = int(idx)
            conf = int(round(float(probs[idx]) * 100))
            results.append({
                "class_id": idx,
                "confidence": conf
            })

        return results
