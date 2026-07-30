import os
import sys
import time

from .utils.log_filter import start_filter, stop_filter
from .base import _release_acl_resource

start_filter()

_instances = {}


def _get_model_class(model_name):
    """按文件名匹配模型类，同时返回标准模型名（对齐 C++ 的 GetModelName）。

    返回: (model_class, standard_name, model_type) 或 None
    标准模型名固定为: yolov4 / yolov3 / vgg_ssd / rt_detr /
                     resnet50 / mobilenetv1 / vgg16 / deeplabv3
    """
    name = model_name.lower()

    if 'yolov4' in name:
        from .models.detection.yolov4 import YOLOv4Detect
        return YOLOv4Detect, 'yolov4', 'detection'

    if 'yolov3' in name:
        from .models.detection.yolov3 import YOLOv3Detect
        return YOLOv3Detect, 'yolov3', 'detection'

    if 'vgg' in name and 'ssd' in name:
        from .models.detection.vgg_ssd import VGGSSDDetect
        return VGGSSDDetect, 'vgg_ssd', 'detection'

    if 'rt-detr' in name or 'rt_detr' in name or 'rtdetr' in name:
        from .models.detection.rt_detr import RTDETRDetect
        return RTDETRDetect, 'rt_detr', 'detection'

    if 'resnet50' in name:
        from .models.classification.resnet50 import ResNet50Classify
        return ResNet50Classify, 'resnet50', 'classification'

    if 'mobilenet' in name and 'v1' in name:
        from .models.classification.mobilenetv1 import MobileNetV1Classify
        return MobileNetV1Classify, 'mobilenetv1', 'classification'

    if 'vgg16' in name:
        from .models.classification.vgg16 import VGG16Classify
        return VGG16Classify, 'vgg16', 'classification'

    if 'deeplab' in name:
        from .models.segmentation.deeplabv3 import DeepLabV3Segment
        return DeepLabV3Segment, 'deeplabv3', 'segmentation'

    return None


def kzzk_cv(modelfile, imagefile):
    """
    调用 CANN NPU 进行图像推理（Python 纯实现）

    :param modelfile: 模型文件路径（.om 文件）
    :param imagefile: 输入图像路径
    :return: 推理结果字典
        {
            "model_name": "yolov4",
            "model_type": "detection" | "classification" | "segmentation",
            "infer_cost": 25,   # 毫秒
            "results": [...]     # 具体结果列表
        }

    示例:
        result = kzzk_cv("/home/HwHiAiUser/cann/model/yolov4.om", "dog.jpg")
        result = kzzk_cv("/home/HwHiAiUser/cann/model/resnet50.om", "dog.jpg")
    """
    if not os.path.exists(imagefile):
        raise FileNotFoundError("图像文件不存在: %s" % imagefile)
    if not os.path.exists(modelfile):
        raise FileNotFoundError("模型文件不存在: %s" % modelfile)

    file_name = os.path.splitext(os.path.basename(modelfile))[0].lower()

    if modelfile in _instances:
        instance, standard_name, model_type = _instances[modelfile]
    else:
        matched = _get_model_class(file_name)
        if matched is None:
            raise ValueError("无法识别模型类型: %s" % file_name)
        model_cls, standard_name, model_type = matched
        instance = model_cls(modelfile)
        instance.init()
        _instances[modelfile] = (instance, standard_name, model_type)

    start_time = time.perf_counter()
    results = instance.run(imagefile)
    infer_cost = int(round((time.perf_counter() - start_time) * 1000))

    return {
        "model_name": standard_name,
        "model_type": model_type,
        "infer_cost": infer_cost,
        "results": results
    }


def cleanup():
    """手动释放所有资源"""
    global _instances
    for k, v in _instances.items():
        try:
            # v = (instance, standard_name, model_type)
            v[0].release()
        except Exception:
            pass
    _instances = {}
    _release_acl_resource()
    stop_filter()


import atexit
atexit.register(cleanup)
