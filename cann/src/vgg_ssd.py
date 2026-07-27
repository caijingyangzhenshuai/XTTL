import os
import sys
import time
import datetime
import numpy as np
from PIL import Image


path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, ".."))
sys.path.append(os.path.join(path, "../examples/samples/python/common/"))
sys.path.append(os.path.join(path, "../examples/samples/python/common/acllite"))

# === 过滤 [INFO] 日志输出 ===
import io

class FilteredStdout:
    """过滤掉 [INFO] 开头的日志行"""
    def __init__(self, original, filter_prefixes=("[INFO]",)):
        self.original = original
        self.filter_prefixes = filter_prefixes
        self.buffer = ""

    def write(self, text):
        # 处理跨行写入
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if not any(line.strip().startswith(p) for p in self.filter_prefixes):
                self.original.write(line + "\n")
                self.original.flush()

    def flush(self):
        if self.buffer:
            if not any(self.buffer.strip().startswith(p) for p in self.filter_prefixes):
                self.original.write(self.buffer)
            self.buffer = ""
        self.original.flush()

# 默认隐藏 [INFO]，可通过 --show-info 启用
SHOW_INFO = '--show-info' in sys.argv

import acl
# acl.mdl 子模块用于模型操作
import constants as const
import acllite_utils as utils
from acllite_imageproc import AclLiteImageProc
from acllite_model import AclLiteModel
from acllite_image import AclLiteImage
from acllite_resource import AclLiteResource

# SSD300 标准参数
MODEL_WIDTH = 300
MODEL_HEIGHT = 300
MODEL_BATCH = 1

# Pascal VOC 类别（21 类含背景）
LABELS_VOC = [
    "background",  # 0
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]  # 1-20

# COCO 80 类（备选）
LABELS_COCO = [
    "person", "bicycle", "car", "motorbike", "aeroplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# SSD 检测参数（基于 MMDetection SSD300 配置）
# 原始模型 score_thr=0.02, nms iou_threshold=0.45, max_per_img=200
# 由于 OM 模型输出异常（18006 框且结果随机），使用较高阈值过滤
CONF_THRESHOLD = 0.05  # 置信度阈值（降低以捕获低分真实目标，如远处的 person）
NMS_IOU_THRESHOLD = 0.30  # NMS IoU 阈值（降低以合并 3x3 网格相邻框）
TOP_K = 5  # 最多保留的检测框数
MAX_PER_CLASS = 1  # 每个类别最多保留的检测数
# VGG-SSD 模型会把 1 个目标检测成多个相邻框（IoU=0，NMS 无法合并）
# 设为 1 可消除重复，但多人场景只显示 1 个
# 多人场景请使用 yolov3_yuv.om
HIGH_CONF_THRESHOLD = 0.10  # 智能 TOP_K 阈值（>= 此值视为真实目标）
DEBUG = os.environ.get('SSD_DEBUG', '0') == '1'  # SSD 调试开关（默认关闭）

OUTPUT_DIR = os.path.join(path, "../out")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局共享资源
_acl_resource = None
_detect_instance = None
_current_labels = LABELS_COCO  # 默认使用 COCO 80 类（无背景）


def _to_numpy_ptr(arr):
    """将 numpy 数组转换为 ACL 设备指针"""
    if "bytes_to_ptr" in dir(acl.util):
        return acl.util.bytes_to_ptr(arr.tobytes())
    return acl.util.numpy_to_ptr(arr)


def print_help():
    """打印帮助信息"""
    print("VGG_SSD目标检测推理")
    print("Usage: vgg_ssd.py [OPTIONS]")
    print("Options:")
    print("  --modelfile MODEL      指定模型文件路径（.om 文件）")
    print("  --imagefile IMAGE      指定输入图像路径")
    print("  --help                 显示帮助信息")
    print("")
    print("Examples:")
    print("  python vgg_ssd.py --modelfile /home/HwHiAiUser/cann/model/vgg_ssd.om --imagefile /home/HwHiAiUser/cann/data/cat.png")


class Detect:
    """VGG_SSD 目标检测类"""

    def __init__(self, model_path, model_width=300, model_height=300):
        self._model_path = model_path
        self._model_width = model_width
        self._model_height = model_height
        self._model = None

    def init(self):
        """初始化 ACL 模型"""
        self._model = AclLiteModel(self._model_path)
        return const.SUCCESS

    def pre_process(self, image_path):
        """预处理：适配有 AIPP 和无 AIPP 两种 OM 模型
        
        AIPP 模型（vgg_ssd.om 转换时带 --insert_op_conf）：
            - 输入: 304x300 UINT8 RGB（不是 300x300！）
            - AIPP 在芯片上完成：裁剪 304->300、减均值、归一化
            - 代码只负责 resize 到 304x300 + 转 UINT8 + HWC->CHW
        
        无 AIPP 模型（ssd300_static_bs1.om）：
            - 输入: 300x300 FP32, NCHW, 已减均值 [104, 117, 123] BGR 顺序
        """
        # 读取原始图像，获取原始尺寸
        pil_image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = pil_image.size
        self._orig_size = (orig_w, orig_h)
        
        # 检测 AIPP 模式
        is_aipp = self._detect_aipp_mode()
        
        if is_aipp:
            # === AIPP 模式：发送 304x300 UINT8 RGB 数据 ===
            # AIPP 配置: src_image_size_w=304, src_image_size_h=300, crop=true
            # 304 * 300 * 3 = 273,600 字节（与警告一致）
            aipp_input_w = 304
            aipp_input_h = 300
            
            if DEBUG:
                print("[DEBUG] AIPP 模式: 输入 %dx%d" % (aipp_input_w, aipp_input_h))
            
            # 调整到 304x300（AIPP 会自动裁剪到 300x300）
            pil_resized = pil_image.resize(
                (aipp_input_w, aipp_input_h), Image.BILINEAR)
            
            # 转 UINT8 RGB（保持 RGB 顺序，AIPP 的 input_format 是 RGB888_U8）
            img_array = np.array(pil_resized, dtype=np.uint8)
            # HWC -> CHW
            img_array = np.transpose(img_array, (2, 0, 1))  # (3, 300, 304)
            img_array = np.expand_dims(img_array, axis=0)  # (1, 3, 300, 304)
            
            if not img_array.flags['C_CONTIGUOUS']:
                img_array = np.ascontiguousarray(img_array)
            
            if DEBUG:
                print("[DEBUG] AIPP 输入 dtype: %s, shape: %s" % (
                    img_array.dtype, img_array.shape))
            return img_array
        else:
            # === 无 AIPP 模式：300x300 FP32 + 减均值 ===
            pil_image = pil_image.resize(
                (self._model_width, self._model_height), Image.BILINEAR)
            
            img_array = np.array(pil_image, dtype=np.float32)
            # RGB -> BGR
            img_array = img_array[:, :, ::-1]
            # 减均值（BGR 顺序）
            mean = np.array([104.0, 117.0, 123.0], dtype=np.float32)
            img_array = img_array - mean
            
            # HWC -> CHW
            img_array = np.transpose(img_array, (2, 0, 1))  # (3, 300, 300)
            img_array = np.expand_dims(img_array, axis=0)  # (1, 3, 300, 300)
            
            if not img_array.flags['C_CONTIGUOUS']:
                img_array = np.ascontiguousarray(img_array)
            
            if DEBUG:
                print("[DEBUG] 无 AIPP 输入 dtype: %s, shape: %s" % (
                    img_array.dtype, img_array.shape))
            return img_array

    def _get_model_input_size(self):
        """获取模型期望的输入大小（字节数）"""
        try:
            # 通过 acl.mdl 子模块查询
            model_desc = self._model._model_desc
            input_size = acl.mdl.get_input_size_by_index(model_desc, 0)
            return int(input_size)
        except Exception as e:
            if DEBUG:
                print("[DEBUG] 获取模型输入大小失败: %s" % str(e))
            return None

    def _detect_aipp_mode(self):
        """检测是否使用 AIPP 模式
        
        Returns:
            bool: True 表示使用 AIPP，False 表示不使用
        """
        # 方法 1: 通过查询模型输入大小判断
        input_size = self._get_model_input_size()
        if input_size is not None:
            # AIPP 模型: UINT8 输入，约 1*3*300*300 = 270000 ~ 1*3*300*304 = 273600
            # 无 AIPP 模型: FP32 输入，1*3*300*300*4 = 1080000
            if input_size < 1000000:
                if DEBUG:
                    print("[DEBUG] 通过输入大小 %d 字节判定为 AIPP 模型" % input_size)
                return True
            else:
                if DEBUG:
                    print("[DEBUG] 通过输入大小 %d 字节判定为无 AIPP 模型" % input_size)
                return False
        
        # 方法 2: 通过模型文件路径回退判断
        model_path_lower = self._model_path.lower()
        if 'vgg_ssd' in model_path_lower or 'aipp' in model_path_lower:
            if DEBUG:
                print("[DEBUG] 通过模型文件名判定为 AIPP 模型: %s" % self._model_path)
            return True
        
        # 默认: 无 AIPP
        if DEBUG:
            print("[DEBUG] 默认判定为无 AIPP 模型")
        return False

    def inference(self, input_data):
        """模型推理"""
        if isinstance(input_data, np.ndarray):
            return self._model.execute([input_data, ])
        return self._model.execute(input_data)

    def post_process(self, infer_output, orig_image_path=None):
        """后处理：解析 vgg_ssd.om 输出（匹配 utils.cpp 格式）

        vgg_ssd.om 输出格式（来自 utils.cpp）：
          output[0]: 1 个数字 (boxNum) - 4 字节
          output[1]: boxNum * 8 个 float，格式为：
            [0]: padding
            [1]: LABEL (0-20, VOC 21 类)
            [2]: SCORE (0-1)
            [3]: TOPLEFTX (归一化 0-1)
            [4]: TOPLEFTY
            [5]: BOTTOMRIGHTX
            [6]: BOTTOMRIGHTY
            [7]: padding
        """
        if DEBUG:
            print("[DEBUG] infer_output type:", type(infer_output))
            if isinstance(infer_output, list):
                for i, o in enumerate(infer_output):
                    arr = np.array(o) if not isinstance(o, np.ndarray) else o
                    print("[DEBUG]  output[%d] shape=%s dtype=%s min=%.3f max=%.3f" % (
                        i, arr.shape, arr.dtype, float(arr.min()), float(arr.max())))

        if not (isinstance(infer_output, list) and len(infer_output) >= 2):
            return []

        out0 = np.array(infer_output[0]).flatten()
        out1 = np.array(infer_output[1]).flatten()

        if DEBUG:
            print("[DEBUG] out0 (boxNum) shape:", out0.shape, "values:", out0[:5])
            print("[DEBUG] out1 (dets) shape:", out1.shape)

        # === 解析 boxNum (output[0]) ===
        # 可能是 FLOAT 或 INT32
        try:
            box_num = int(out0[0])
        except (ValueError, IndexError):
            if DEBUG:
                print("[DEBUG] 无法解析 boxNum")
            return []

        # 8 个 float 为一个检测
        det_count = len(out1) // 8
        if DEBUG:
            print("[DEBUG] out1 长度: %d, 理论检测数: %d" % (len(out1), det_count))
            print("[DEBUG] 前 16 个值: %s" % str(out1[:16]))

        # === 解析每个检测（匹配 C++ 的 BBoxIndex）===
        # BBoxIndex: LABEL=1, SCORE=2, TOPLEFTX=3, TOPLEFTY=4, BOTTOMRIGHTX=5, BOTTOMRIGHTY=6, BOXINFOSIZE=8
        orig_w, orig_h = 1, 1
        if orig_image_path and os.path.exists(orig_image_path):
            with Image.open(orig_image_path) as img:
                orig_w, orig_h = img.size

        results = []
        # vgg_ssd.om 的置信度范围 0.005-1.0+，使用 CONF_THRESHOLD 过滤
        SCORE_THRESHOLD = CONF_THRESHOLD  # 复用全局阈值

        for b in range(det_count):
            base = b * 8
            if base + 7 >= len(out1):
                break

            # vgg_ssd.om 输出格式（按照 utils.cpp 的枚举）：
            # 位置 0: padding, 1: LABEL, 2: SCORE, 3-6: 坐标, 7: 计数/其他
            label = int(out1[base + 1])  # LABEL=1
            score = float(out1[base + 2])  # SCORE=2 (实际置信度，范围 0.01-1.0)
            x1 = float(out1[base + 3])  # TOPLEFTX=3
            y1 = float(out1[base + 4])  # TOPLEFTY=4
            x2 = float(out1[base + 5])  # BOTTOMRIGHTX=5
            y2 = float(out1[base + 6])  # BOTTOMRIGHTY=6

            if DEBUG and b < 5:
                print("[DEBUG]  检测 #%d: label=%d score=%.3f box=[%.3f,%.3f,%.3f,%.3f]" % (
                    b, label, score, x1, y1, x2, y2))

            # 过滤低置信度
            if score < SCORE_THRESHOLD:
                continue

            # 过滤背景类（VOC 中 0=background）
            if label <= 0 or label >= 21:
                continue

            # 坐标裁剪到 [0, 1]
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))

            # 转换为原图像素坐标
            x1_px = int(x1 * orig_w)
            y1_px = int(y1 * orig_h)
            x2_px = int(x2 * orig_w)
            y2_px = int(y2 * orig_h)

            # 过滤过小的框
            if (x2_px - x1_px) < 15 or (y2_px - y1_px) < 15:
                continue

            # 过滤过小的检测（面积 < 图像 0.5%，通常为背景噪声）
            box_area = (x2_px - x1_px) * (y2_px - y1_px)
            img_area = orig_w * orig_h
            if box_area < img_area * 0.005:
                continue

            # 过滤图像底部 30% 区域的低置信度检测（通常是地面/背景误检）
            bottom_threshold = orig_h * 0.70
            if y1_px > bottom_threshold and score < 0.10:
                continue

            # 过滤图像边缘的低置信度检测（边界伪影）
            if (x1_px < 5 or y1_px < 5 or
                x2_px > orig_w - 5 or y2_px > orig_h - 5):
                if score < 0.10:
                    continue

            # VOC 21 类（label 1-20 对应数组索引 1-20）
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

        # === NMS 合并重叠的检测 ===
        # 因为 vgg_ssd.om 输出未做 NMS，会有很多重叠的检测
        results = self._nms_results(results, iou_thresh=NMS_IOU_THRESHOLD)

        # === 按类别保留最佳 ===
        # 解决 vgg_ssd.om 的 3x3 网格重复检测问题
        # MAX_PER_CLASS 控制每个类别保留的检测数（多 person 场景需要 > 1）
        results = self._keep_best_per_class(results, max_per_class=MAX_PER_CLASS)

        # === 智能 TOP_K：按高置信度检测数量自适应 ===
        # 统计 score >= HIGH_CONF_THRESHOLD 的数量（真实目标通常置信度较高）
        # 显示所有高置信度检测（最多 5 个，最少 1 个）
        if results:
            scores = sorted([r["confidence"] for r in results], reverse=True)
            high_conf_count = sum(1 for s in scores if s >= HIGH_CONF_THRESHOLD)
            final_k = min(max(high_conf_count, 1), 5)
        else:
            final_k = TOP_K

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:final_k]

    def _keep_best_per_class(self, results, max_per_class=1):
        """每个类别只保留得分最高的 N 个检测

        解决 vgg_ssd.om 的 3x3 网格重复检测问题（框不重叠，NMS 无法合并）
        """
        if not results or max_per_class <= 0:
            return results

        # 按类别分组
        class_groups = {}
        for r in results:
            cid = r["class_id"]
            class_groups.setdefault(cid, []).append(r)

        # 每组按置信度排序，保留 top-N
        keep = []
        for cid, group in class_groups.items():
            group_sorted = sorted(group, key=lambda x: x["confidence"], reverse=True)
            keep.extend(group_sorted[:max_per_class])

        return keep

    def _nms_results(self, results, iou_thresh=0.45):
        """对 results 列表做 NMS（IoU 大于阈值的保留得分最高的）"""
        if not results:
            return []

        boxes = np.array([r["bbox"] for r in results], dtype=np.float32)
        scores = np.array([r["confidence"] for r in results], dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return [results[i] for i in keep]

    def _generate_default_boxes(self, target_count=None):
        """生成 SSD300 先验框（基于 MMDetection SSD300 配置）
        
        配置参数：
        - basesize_ratio_range=(0.15, 0.9)
        - strides=[8, 16, 32, 64, 100, 300]
        - ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        """
        feature_maps = [38, 19, 10, 5, 3, 1]
        strides = [8, 16, 32, 64, 100, 300]
        ratios_list = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        
        # 根据 basesize_ratio_range=(0.15, 0.9) 计算 scales
        min_scale_ratio = 0.15
        max_scale_ratio = 0.9
        num_levels = len(feature_maps)
        
        # scales = np.linspace(min_scale * input_size, max_scale * input_size, num_levels + 1)
        scales_pixels = np.linspace(
            min_scale_ratio * MODEL_WIDTH,
            max_scale_ratio * MODEL_WIDTH,
            num_levels + 1
        )  # [45, 90, 135, 180, 225, 270, 315]
        
        default_boxes = []
        for k, f in enumerate(feature_maps):
            s_k = scales_pixels[k] / MODEL_WIDTH
            s_k_prime = np.sqrt(scales_pixels[k] * scales_pixels[k + 1]) / MODEL_WIDTH
            
            for i in range(f):
                for j in range(f):
                    cx = (j + 0.5) * strides[k] / MODEL_WIDTH
                    cy = (i + 0.5) * strides[k] / MODEL_HEIGHT
                    
                    # 1) scale s_k, ratio 1
                    default_boxes.append([cx, cy, s_k, s_k])
                    # 2) scale s_k', ratio 1
                    default_boxes.append([cx, cy, s_k_prime, s_k_prime])
                    # 3) 对于每个 ratio r，添加 ratio r 和 ratio 1/r
                    for ar in ratios_list[k]:
                        default_boxes.append([cx, cy, s_k * np.sqrt(ar), s_k / np.sqrt(ar)])
                        default_boxes.append([cx, cy, s_k / np.sqrt(ar), s_k * np.sqrt(ar)])
        
        result = np.array(default_boxes, dtype=np.float32)
        
        if target_count is not None and target_count != result.shape[0]:
            if DEBUG:
                print("[DEBUG] 先验框数量 %d 与模型输出 %d 不匹配" % (
                    result.shape[0], target_count))
        
        return result

    def _decode_boxes(self, loc, default_boxes):
        """解码 SSD 框偏移量到实际坐标
        
        SSD 训练时使用的编码公式：
        cx = (loc_cx * variance[0] * prior_w) + prior_cx
        cy = (loc_cy * variance[1] * prior_h) + prior_cy
        w = prior_w * exp(loc_w * variance[2])
        h = prior_h * exp(loc_h * variance[3])
        """
        variances = [0.1, 0.1, 0.2, 0.2]
        
        cx = loc[:, 0] * variances[0] * default_boxes[:, 2] + default_boxes[:, 0]
        cy = loc[:, 1] * variances[1] * default_boxes[:, 3] + default_boxes[:, 1]
        w = default_boxes[:, 2] * np.exp(loc[:, 2] * variances[2])
        h = default_boxes[:, 3] * np.exp(loc[:, 3] * variances[3])
        
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2
        
        return np.stack([ymin, xmin, ymax, xmax], axis=1)

    def _nms(self, boxes, scores):
        """NMS 实现（CPU 端，向量化优化）"""
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 1]
        y1 = boxes[:, 0]
        x2 = boxes[:, 3]
        y2 = boxes[:, 2]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(iou <= NMS_IOU_THRESHOLD)[0]
            order = order[inds + 1]
        
        return keep

    def _parse_single_output(self, output):
        """单输出格式的备用解析"""
        return []


def save_results(result, model_name, imagefile):
    """保存结果到 txt 文件"""
    image_dir = os.path.dirname(os.path.abspath(imagefile))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(image_dir,
                            "%s_%s.txt" % (timestamp, model_name))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("SSD300目标检测推理\n")
        f.write("Model: %s\n" % model_name)
        f.write("Image: %s\n" % imagefile)
        f.write("Total objects: %d\n" % len(result['results']))
        f.write("Format: [ID] (Class Name) [x1, y1, x2, y2] Confidence\n")
        for i, det in enumerate(result['results'], 1):
            # 限制显示在 0-100%
            display_pct = min(100, max(0, int(round(det['confidence'] * 100))))
            f.write("[%d] %d (%s) [%d,%d,%d,%d] %d%%\n" % (
                i, det['class_id'], det['label'],
                det['bbox'][0], det['bbox'][1],
                det['bbox'][2], det['bbox'][3],
                display_pct))
    return txt_path


def kzzk_cv(modelfile, imagefile):
    """kzzk_cv 函数接口

    Args:
        modelfile: OM 模型文件路径
        imagefile: 输入图像路径
    Returns:
        {
            "results": [...],
            "infer_cost": ms,
            "model_name": str,
        }
    """
    global _acl_resource, _detect_instance

    if not os.path.exists(imagefile):
        raise FileNotFoundError("图像文件不存在: %s" % imagefile)
    if not os.path.exists(modelfile):
        raise FileNotFoundError("模型文件不存在: %s" % modelfile)

    if _acl_resource is None:
        _acl_resource = AclLiteResource()
        _acl_resource.init()

    if (_detect_instance is None
            or _detect_instance._model_path != modelfile):
        _detect_instance = Detect(modelfile, MODEL_WIDTH, MODEL_HEIGHT)
        ret = _detect_instance.init()
        utils.check_ret("Detect init", ret)

    start_time = time.perf_counter()

    input_data = _detect_instance.pre_process(imagefile)
    result = _detect_instance.inference(input_data)
    output = _detect_instance.post_process(result, imagefile)

    infer_cost = int(round((time.perf_counter() - start_time) * 1000))

    model_name = os.path.splitext(os.path.basename(modelfile))[0]

    return {
        "results": output,
        "infer_cost": infer_cost,
        "model_name": model_name,
    }


def main():
    """命令行入口"""
    import sys as _sys
    sys.stdout = _sys.stdout
    # 解析参数
    args = _sys.argv[1:]
    modelfile = None
    imagefile = None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--help", "-h"):
            print_help()
            return
        if a == "--modelfile" and i + 1 < len(args):
            modelfile = args[i + 1]
            i += 2
        elif a == "--imagefile" and i + 1 < len(args):
            imagefile = args[i + 1]
            i += 2
        else:
            i += 1

    if not modelfile or not imagefile:
        print_help()
        return

    # 激活 stdout 过滤器（除非 --show-info）
    _original_stdout = sys.stdout
    if not SHOW_INFO:
        sys.stdout = FilteredStdout(_original_stdout)

    try:
        result = kzzk_cv(modelfile, imagefile)
    except Exception as e:
        sys.stdout = _original_stdout
        print("错误: %s" % str(e))
        return

    # 恢复 stdout 再打印结果
    sys.stdout = _original_stdout
    print("Total inference cost: %dms" % result['infer_cost'])
    print("Inference Results: ")
    print("VGG_SSD目标检测推理")
    print("Total objects: %d" % len(result['results']))
    print("Format: [ID] (Class Name) [x1, y1, x2, y2] Confidence")
    for i, det in enumerate(result['results'], 1):
        # 限制显示在 0-100%
        display_pct = min(100, max(0, int(round(det['confidence'] * 100))))
        print("[%d] %d (%s) [%d,%d,%d,%d] %d%%" % (
            i, det['class_id'], det['label'],
            det['bbox'][0], det['bbox'][1],
            det['bbox'][2], det['bbox'][3],
            display_pct))

    txt_path = save_results(result, result['model_name'], imagefile)
    print("推理结果已保存到: %s" % txt_path)
    os._exit(0)


if __name__ == '__main__':
    main()

