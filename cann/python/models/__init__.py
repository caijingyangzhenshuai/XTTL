"""
模型库（按任务类型分类）

延迟导入：只有使用具体模型时才会加载对应模块，避免不必要的依赖。
"""
from .detection import YOLOv4Detect, YOLOv3Detect, YOLOv3YUVDetect, VGGSSDDetect, RTDETRDetect, SSDDetect
from .classification import ResNet50Classify, MobileNetV1Classify, VGG16Classify
from .segmentation import DeepLabV3Segment
