import os
import sys
import numpy as np
from PIL import Image

from ...base import BaseModel

MODEL_WIDTH = 224
MODEL_HEIGHT = 224
MODEL_BATCH = 1
NUM_CLASSES = 1000

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

USE_INT8 = os.environ.get('MB_INT8', '0') == '1'
USE_TF_NORM = os.environ.get('MB_TF_NORM', '0') == '1'
USE_BGR = os.environ.get('MB_BGR', '0') == '1'
USE_NHWC = os.environ.get('MB_NHWC', '0') == '1'
USE_X255_NORM = os.environ.get('MB_X255', '0') == '1'
USE_FP16 = os.environ.get('MB_FP16', '0') == '1'
APPLY_SOFTMAX = True
DEBUG = os.environ.get('MB_DEBUG', '0') == '1'

TOP_K = 5


class MobileNetV1Classify(BaseModel):
    # ACL 数据类型 -> 精度名称（对齐 C++ MobileNetV1 的 in_type_str/out_type_str）
    _DTYPE_NAME = None

    def __init__(self, model_path):
        super().__init__(model_path)
        self._model_width = MODEL_WIDTH
        self._model_height = MODEL_HEIGHT

    def init(self):
        ret = super().init()
        # 打印精度信息（对齐 C++ aclmdlGetInputDataType/GetOutputDataType）
        import acl
        import constants as const
        if self._DTYPE_NAME is None:
            self._DTYPE_NAME = {
                const.ACL_FLOAT:   "FP32",
                const.ACL_FLOAT16: "FP16",
                const.ACL_INT8:    "INT8",
                const.ACL_UINT8:   "UINT8",
            }
        desc = getattr(self._model, "_model_desc", None)
        in_type_str = "?"
        out_type_str = "?"
        if desc is not None:
            try:
                in_type_str = self._DTYPE_NAME.get(
                    acl.mdl.get_input_data_type(desc, 0), "?")
                out_type_str = self._DTYPE_NAME.get(
                    acl.mdl.get_output_data_type(desc, 0), "?")
            except Exception:
                pass
        sys.stderr.write(
            "[INFO][MobileNetV1] 模型精度: 输入=%s, 输出=%s\n"
            % (in_type_str, out_type_str))
        return ret

    def _preprocess_pil(self, pil_image):
        if USE_INT8:
            img_array = np.array(pil_image, dtype=np.int8)
            img_array = np.expand_dims(img_array, axis=0)
            if img_array.shape[0] < MODEL_BATCH:
                pad = np.zeros(
                    (MODEL_BATCH - 1,) + img_array.shape[1:],
                    dtype=img_array.dtype)
                img_array = np.concatenate([img_array, pad], axis=0)
            if not img_array.flags['C_CONTIGUOUS']:
                img_array = np.ascontiguousarray(img_array)
            return img_array

        img_array = np.array(pil_image, dtype=np.float32)

        if USE_TF_NORM:
            if USE_X255_NORM:
                img_array = img_array / 127.5 - 1.0
            else:
                img_array = img_array / 255.0
                img_array = (img_array - 0.5) / 0.5
        else:
            if not USE_X255_NORM:
                img_array = img_array / 255.0
            img_array = (img_array - MEAN) / STD

        if USE_BGR:
            img_array = img_array[..., ::-1]

        if not USE_NHWC:
            img_array = np.transpose(img_array, (2, 0, 1))

        img_array = np.expand_dims(img_array, axis=0)
        if img_array.shape[0] < MODEL_BATCH:
            pad = np.zeros(
                (MODEL_BATCH - 1,) + img_array.shape[1:],
                dtype=img_array.dtype)
            img_array = np.concatenate([img_array, pad], axis=0)
        if not img_array.flags['C_CONTIGUOUS']:
            img_array = np.ascontiguousarray(img_array)

        if USE_FP16:
            img_array = img_array.astype(np.float16)

        return img_array

    def pre_process(self, image_path):
        pil_image = Image.open(image_path).convert('RGB')

        w, h = pil_image.size
        if w < h:
            new_w = 256
            new_h = int(h * 256 / w)
        else:
            new_h = 256
            new_w = int(w * 256 / h)
        pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)

        left = (new_w - self._model_width) // 2
        top = (new_h - self._model_height) // 2
        pil_image = pil_image.crop((left, top,
                                    left + self._model_width,
                                    top + self._model_height))

        return self._preprocess_pil(pil_image)

    def inference(self, input_data):
        if isinstance(input_data, np.ndarray):
            return self._model.execute([input_data, ])
        return self._model.execute(input_data)

    def post_process(self, infer_output):
        infer_result = infer_output[0]
        if infer_result.ndim > 1:
            infer_result = infer_result[0]
        vals = infer_result.flatten().astype(np.float32)

        if DEBUG:
            print("[DEBUG] output shape: %s" % str(infer_result.shape))
            print("[DEBUG] vals min/max/mean: %.3f / %.3f / %.3f" % (
                float(vals.min()), float(vals.max()), float(vals.mean())))

        if APPLY_SOFTMAX and (vals.min() < 0 or vals.max() > 1.0):
            exp_vals = np.exp(vals - np.max(vals))
            probs = exp_vals / np.sum(exp_vals)
        else:
            probs = vals

        probs_1d = probs[:NUM_CLASSES]
        top_indices = probs_1d.argsort()[-TOP_K:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "class_id": int(idx),
                "confidence": int(round(float(probs_1d[idx]) * 100))
            })
        return results
