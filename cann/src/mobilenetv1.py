import sys
import os
import time
import datetime
import numpy as np
from PIL import Image
import acl
import logging
from typing import Dict, List

logging.basicConfig(level=logging.CRITICAL + 1)
logging.getLogger().setLevel(logging.CRITICAL + 1)

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, ".."))
sys.path.append(os.path.join(path, "../examples/samples/python/common/"))
sys.path.append(os.path.join(path, "../examples/samples/python/common/acllite"))

import acllite_logger
acllite_logger.log_info = lambda *args: None
acllite_logger.log_debug = lambda *args: None

from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel
import constants as const
import acllite_utils as utils

currentPath = os.path.join(path, "..")
OUTPUT_DIR = os.path.join(currentPath, 'out')

# MobileNetV2 输入尺寸
MODEL_WIDTH = 224
MODEL_HEIGHT = 224
MODEL_BATCH = 1  # 固定 batch=1
NUM_CLASSES = 1000

# ImageNet 均值与方差（RGB）- PyTorch torchvision 官方标准
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # 在 [0,1] 上
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # 在 [0,1] 上

# 通过环境变量可切换的预处理配置
USE_INT8 = os.environ.get('MB_INT8', '0') == '1'
USE_TF_NORM = os.environ.get('MB_TF_NORM', '0') == '1'
USE_BGR = os.environ.get('MB_BGR', '0') == '1'
USE_NHWC = os.environ.get('MB_NHWC', '0') == '1'
USE_X255_NORM = os.environ.get('MB_X255', '0') == '1'
USE_FP16 = os.environ.get('MB_FP16', '0') == '1'
APPLY_SOFTMAX = True
DEBUG = os.environ.get('MB_DEBUG', '0') == '1'

# 全局共享资源
_acl_resource = None
_classify_instance = None


def _to_numpy_ptr(arr):
    """将 numpy 数组转换为 ACL 设备指针"""
    if "bytes_to_ptr" in dir(acl.util):
        return acl.util.bytes_to_ptr(arr.tobytes())
    return acl.util.numpy_to_ptr(arr)


class Classify(object):
    def __init__(self, model_path, model_width, model_height):
        self._model_path = model_path
        self._model_width = model_width
        self._model_height = model_height
        self._model = None

    def init(self):
        self._model = AclLiteModel(self._model_path)
        return const.SUCCESS

    def _preprocess_pil(self, pil_image):
        """单次预处理：与官方 preprocess.py 完全一致
        INT8 量化模型：直接转 int8，无归一化，NHWC 布局
        """
        if USE_INT8:
            # 官方做法: 直接转 int8，无任何归一化
            img_array = np.array(pil_image, dtype=np.int8)  # (224, 224, 3)
            # 添加 batch 维度 -> (1, 224, 224, 3) NHWC
            img_array = np.expand_dims(img_array, axis=0)
            if img_array.shape[0] < MODEL_BATCH:
                # 用 0 填充（不使用重复，会污染结果）
                pad = np.zeros(
                    (MODEL_BATCH - 1,) + img_array.shape[1:],
                    dtype=img_array.dtype)
                img_array = np.concatenate([img_array, pad], axis=0)
            if not img_array.flags['C_CONTIGUOUS']:
                img_array = np.ascontiguousarray(img_array)
            return img_array
        # 以下为 float32 路径（保留备用）
        img_array = np.array(pil_image, dtype=np.float32)

        # 4. Normalize
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

        # 5. BGR/RGB
        if USE_BGR:
            img_array = img_array[..., ::-1]

        # 6. HWC -> CHW
        if not USE_NHWC:
            img_array = np.transpose(img_array, (2, 0, 1))

        # 7. batch
        img_array = np.expand_dims(img_array, axis=0)
        if img_array.shape[0] < MODEL_BATCH:
            # 用 0 填充（避免重复污染 top-1 结果）
            pad = np.zeros(
                (MODEL_BATCH - 1,) + img_array.shape[1:],
                dtype=img_array.dtype)
            img_array = np.concatenate([img_array, pad], axis=0)
        if not img_array.flags['C_CONTIGUOUS']:
            img_array = np.ascontiguousarray(img_array)

        # 7.5 float16 转换（v1 模型需要）
        if USE_FP16:
            img_array = img_array.astype(np.float16)

        return img_array

    def pre_process(self, image_path):
        """预处理：与 PyTorch torchvision 训练时一致
        流程: Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize
        """
        global USE_TF_NORM, USE_BGR, USE_NHWC, USE_X255_NORM
        pil_image = Image.open(image_path).convert('RGB')

        # 1. Resize 短边到 256（保持比例）
        w, h = pil_image.size
        if w < h:
            new_w = 256
            new_h = int(h * 256 / w)
        else:
            new_h = 256
            new_w = int(w * 256 / h)
        pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)

        # 2. 中心裁剪 224x224
        left = (new_w - self._model_width) // 2
        top = (new_h - self._model_height) // 2
        pil_image = pil_image.crop((left, top,
                                    left + self._model_width,
                                    top + self._model_height))

        return self._preprocess_pil(pil_image)

    def inference(self, input_data):
        # input_data 可以是 np.ndarray 或 AclLiteImage
        if isinstance(input_data, np.ndarray):
            return self._model.execute([input_data, ])
        return self._model.execute(input_data)

    def post_process(self, infer_output):
        """后处理：top-5 (INT8 模型输出可能已是 softmax 概率)"""
        infer_result = infer_output[0]
        # 输出形状: (batch, 1000) -> 只取第一个 batch 的结果
        if infer_result.ndim > 1:
            infer_result = infer_result[0]
        vals = infer_result.flatten().astype(np.float32)

        if DEBUG:
            print("[DEBUG] output shape: %s" % str(infer_result.shape))
            print("[DEBUG] output dtype: %s" % str(infer_result.dtype))
            print("[DEBUG] vals min/max/mean: %.3f / %.3f / %.3f" % (
                float(vals.min()), float(vals.max()), float(vals.mean())))
            print("[DEBUG] top10: %s" % str(
                sorted(vals.tolist(), reverse=True)[:10]))

        # 判断是 logits 还是概率
        if APPLY_SOFTMAX and (vals.min() < 0 or vals.max() > 1.0):
            exp_vals = np.exp(vals - np.max(vals))
            probs = exp_vals / np.sum(exp_vals)
        else:
            # INT8 量化模型输出已经是概率
            probs = vals

        probs_1d = probs[:NUM_CLASSES]
        top_indices = probs_1d.argsort()[-5:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "class_id": int(idx),
                "confidence": int(round(float(probs_1d[idx]) * 100))
            })
        return results


def kzzk_cv(modelfile, imagefile):
    """kzzk_cv 函数接口"""
    global _acl_resource, _classify_instance

    if not os.path.exists(imagefile):
        raise FileNotFoundError("图像文件不存在: %s" % imagefile)
    if not os.path.exists(modelfile):
        raise FileNotFoundError("模型文件不存在: %s" % modelfile)

    if _acl_resource is None:
        _acl_resource = AclLiteResource()
        _acl_resource.init()

    if _classify_instance is None or _classify_instance._model_path != modelfile:
        _classify_instance = Classify(modelfile, MODEL_WIDTH, MODEL_HEIGHT)
        ret = _classify_instance.init()
        utils.check_ret("Classify init", ret)

    start_time = time.perf_counter()

    input_data = _classify_instance.pre_process(imagefile)
    result = _classify_instance.inference(input_data)
    output = _classify_instance.post_process(result)

    infer_cost = int(round((time.perf_counter() - start_time) * 1000))

    model_name = os.path.splitext(os.path.basename(modelfile))[0]

    return {
        "results": output,
        "infer_cost": infer_cost,
        "model_name": model_name
    }


def save_results(results, model_name, image_path):
    """保存结果到 txt 文件（路径与图像同目录）"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.dirname(image_path)
    if not result_dir:
        result_dir = os.getcwd()
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    filename = "%s_%s.txt" % (timestamp, model_name)
    filepath = os.path.join(result_dir, filename)

    with open(filepath, 'w') as f:
        f.write("Total inference cost: %dms\n" % results['infer_cost'])
        f.write("Inference Results:\n")
        f.write("MobileNetV1图像分类推理\n")
        f.write("Top 5 classes:\n")
        f.write("Format: [ID Confidence]\n")
        for i, res in enumerate(results['results'], 1):
            f.write("Result %d: [%d %d%%]\n" % (
                i, res['class_id'], res['confidence']))
    return filepath


def print_help():
    print("MobileNetV1图像分类推理")
    print("Usage: mobilenetv1.py [OPTIONS]")
    print("Options:")
    print("  --modelfile MODEL      指定模型文件路径（.om 文件）")
    print("  --imagefile IMAGE      指定输入图像路径")
    print("  --help                 显示帮助信息")
    print("")
    print("Examples:")
    print("  python mobilenetv1.py --modelfile /home/HwHiAiUser/cann/model/mobilenet-v1_bs1.om --imagefile /home/HwHiAiUser/cann/data/cat.png")


def main():
    args = sys.argv[1:]

    if not args or "--help" in args:
        print_help()
        sys.exit(0)

    modelfile = None
    imagefile = None

    i = 0
    while i < len(args):
        if args[i] == "--modelfile" and i + 1 < len(args):
            modelfile = args[i + 1]
            i += 2
        elif args[i] == "--imagefile" and i + 1 < len(args):
            imagefile = args[i + 1]
            i += 2
        else:
            i += 1

    if not modelfile or not imagefile:
        print_help()
        sys.exit(1)

    if not os.path.exists(modelfile):
        print("错误: 模型文件不存在: %s" % modelfile)
        sys.exit(1)
    if not os.path.exists(imagefile):
        print("错误: 图像文件不存在: %s" % imagefile)
        sys.exit(1)

    try:
        result = kzzk_cv(modelfile, imagefile)
    except Exception as e:
        print("错误: %s" % str(e))
        sys.exit(1)

    print("Total inference cost: %dms" % result['infer_cost'])
    print("Inference Results: ")
    print("MobileNetV1图像分类推理")
    print("Top 5 classes:")
    print("Format: [ID Confidence]")
    for i, res in enumerate(result['results'], 1):
        print("Result %d: [%d %d%%]" % (
            i, res['class_id'], res['confidence']))

    txt_path = save_results(result, result['model_name'], imagefile)
    print("推理结果已保存到: %s" % txt_path)

    os._exit(0)


if __name__ == '__main__':
    main()

