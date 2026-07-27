import os
import sys
import time
import numpy as np
from PIL import Image

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, ".."))
sys.path.append(os.path.join(path, "../examples/samples/python/common/"))
sys.path.append(os.path.join(path, "../examples/samples/python/common/acllite"))

# === 过滤 [INFO] 日志输出 ===
class FilteredStdout:
    """过滤掉 [INFO] 开头的日志行"""
    def __init__(self, original, filter_prefixes=("[INFO]",)):
        self.original = original
        self.filter_prefixes = filter_prefixes
        self.buffer = ""

    def write(self, text):
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

SHOW_INFO = '--show-info' in sys.argv

# === 在导入 ACL 库之前就激活过滤器 ===
# 这样所有 print（包括 acllite 库内部的 [INFO]）都会被过滤
if not SHOW_INFO:
    sys.stdout = FilteredStdout(sys.stdout)
    sys.stderr = FilteredStdout(sys.stderr)

import acl
import constants as const
import acllite_utils as utils
from acllite_imageproc import AclLiteImageProc
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource

# COCO 80 类标签
LABELS_COCO = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# RT-DETR 参数
MODEL_WIDTH = 640
MODEL_HEIGHT = 640
CONF_THRESHOLD = 0.30   # 置信度阈值
NMS_IOU_THRESHOLD = 0.45
TOP_K = 10              # 最多显示 10 个
MAX_PER_CLASS = 2       # 每类最多 2 个
DEBUG = os.environ.get('RTDETR_DEBUG', '0') == '1'

# 全局资源
_acl_resource = None
_detect_instance = None


def print_help():
    print("RT-DETR目标检测推理")
    print("Usage: rtdetr.py [OPTIONS]")
    print("Options:")
    print("  --modelfile MODEL      指定模型文件路径（.om 文件）")
    print("  --imagefile IMAGE      指定输入图像路径")
    print("  --show-info            显示底层库的 [INFO] 日志")
    print("  --help                 显示帮助信息")
    print("")
    print("Examples:")
    print("  python rt_detr.py --modelfile /home/HwHiAiUser/cann/model/rtdetr_model.om --imagefile /home/HwHiAiUser/cann/data/bus.jpg")


class RTDETRDetect:
    def __init__(self, model_path, model_width=640, model_height=640):
        self._model_path = model_path
        self._model_width = model_width
        self._model_height = model_height
        self._model = None
        self._orig_size = (640, 640)  # (w, h)

    def init(self):
        self._model = AclLiteModel(self._model_path)
        if DEBUG:
            print("[DEBUG] RT-DETR model loaded: %s" % self._model_path)
        return 0

    def pre_process(self, image_path):
        """预处理：resize 640x640, normalize [0,1], CHW, add batch dim"""
        img = np.array(Image.open(image_path).convert('RGB'))  # RGB 格式
        if img is None or img.size == 0:
            raise FileNotFoundError("无法读取图像: %s" % image_path)

        orig_h, orig_w = img.shape[:2]
        self._orig_size = (orig_w, orig_h)

        # Resize 到 640x640
        img_resized = np.array(Image.fromarray(img).resize(
            (self._model_width, self._model_height), Image.BILINEAR))
        # PIL 已经是 RGB，无需转换
        # 归一化到 [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        # HWC -> CHW
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        # 添加 batch 维度 [1, 3, 640, 640]
        img_input = np.expand_dims(img_chw, axis=0).astype(np.float32)

        # orig_target_sizes: [1, 2] INT64, 格式 [width, height]
        orig_target_sizes = np.array([[orig_w, orig_h]], dtype=np.int64)

        if DEBUG:
            print("[DEBUG] Image preprocessed: shape=%s dtype=%s" % (img_input.shape, img_input.dtype))
            print("[DEBUG] orig_target_sizes: %s" % orig_target_sizes)

        return img_input, orig_target_sizes

    def inference(self, input_data):
        """推理：2 个输入 (image, orig_target_sizes)"""
        image, target_sizes = input_data
        if DEBUG:
            print("[DEBUG] Running inference...")
        result = self._model.execute([image, target_sizes])
        return result

    def post_process(self, infer_output, orig_image_path=None):
        """后处理：解析 labels/boxes/scores"""
        # RT-DETR 输出: [labels, boxes, scores]
        # labels: [1, 300], boxes: [1, 300, 4], scores: [1, 300]
        labels = infer_output[0][0]   # [300]
        boxes = infer_output[1][0]    # [300, 4] xyxy 像素坐标
        scores = infer_output[2][0]   # [300]

        if DEBUG:
            print("[DEBUG] labels shape=%s" % str(labels.shape))
            print("[DEBUG] boxes shape=%s" % str(boxes.shape))
            print("[DEBUG] scores shape=%s, min=%.3f max=%.3f" % (
                scores.shape, float(scores.min()), float(scores.max())))

        results = []
        for i in range(len(labels)):
            score = float(scores[i])
            if score < CONF_THRESHOLD:
                continue

            label_id = int(labels[i])
            if label_id < 0 or label_id >= len(LABELS_COCO):
                continue

            x1, y1, x2, y2 = float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])
            label_name = LABELS_COCO[label_id]

            results.append({
                "class_id": label_id,
                "label": label_name,
                "confidence": score,
                "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
            })

        if DEBUG:
            print("[DEBUG] 检测数 (score >= %.2f): %d" % (CONF_THRESHOLD, len(results)))

        # NMS 去重
        results = self._nms_results(results, iou_thresh=NMS_IOU_THRESHOLD)
        # 每类保留 top-N
        results = self._keep_best_per_class(results, max_per_class=MAX_PER_CLASS)

        # 按置信度排序
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:TOP_K]

    def _keep_best_per_class(self, results, max_per_class=2):
        if not results or max_per_class <= 0:
            return results
        class_groups = {}
        for r in results:
            class_groups.setdefault(r["class_id"], []).append(r)
        keep = []
        for cid, group in class_groups.items():
            group_sorted = sorted(group, key=lambda x: x["confidence"], reverse=True)
            keep.extend(group_sorted[:max_per_class])
        return keep

    def _nms_results(self, results, iou_thresh=0.45):
        if not results:
            return results
        # 按类别做 NMS
        class_groups = {}
        for r in results:
            class_groups.setdefault(r["class_id"], []).append(r)
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


def save_results(result, model_name, imagefile):
    """保存结果到 txt"""
    out_dir = os.path.dirname(imagefile)
    if not out_dir:
        out_dir = "."
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(imagefile))[0]
    txt_path = os.path.join(out_dir, "%s_%s_%s.txt" % (timestamp, model_name, base))

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Image: %s\n" % imagefile)
        f.write("Model: %s\n" % model_name)
        f.write("Total objects: %d\n" % len(result['results']))
        f.write("Format: [ID] (Class Name) [x1, y1, x2, y2] Confidence\n")
        for i, det in enumerate(result['results'], 1):
            display_pct = min(100, max(0, int(round(det['confidence'] * 100))))
            f.write("[%d] %d (%s) [%d,%d,%d,%d] %d%%\n" % (
                i, det['class_id'], det['label'],
                det['bbox'][0], det['bbox'][1],
                det['bbox'][2], det['bbox'][3],
                display_pct))
    return txt_path


def kzzk_cv(modelfile, imagefile):
    """主推理函数接口"""
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
        _detect_instance = RTDETRDetect(modelfile, MODEL_WIDTH, MODEL_HEIGHT)
        ret = _detect_instance.init()
        utils.check_ret("RTDETR init", ret)

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
    import sys as _sys
    args = _sys.argv[1:]

    if "--help" in args or "-h" in args:
        print_help()
        return

    modelfile = None
    imagefile = None

    i = 0
    while i < len(args):
        a = args[i]
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

    # stdout 过滤器在模块加载时已激活，这里不需要再设置
    try:
        result = kzzk_cv(modelfile, imagefile)
    except Exception as e:
        # 绕过过滤器直接打印错误到原始 stdout
        _orig = sys.stdout
        while isinstance(_orig, FilteredStdout):
            _orig = _orig.original
        print("错误: %s" % str(e))
        return

    print("Total inference cost: %dms" % result['infer_cost'])
    print("Inference Results: ")
    print("RT-DETR目标检测推理")
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

    # === 显式清理资源（在 filter 仍生效时执行，避免末尾 [INFO]） ===
    _explicit_cleanup()


def _explicit_cleanup():
    """显式释放 ACL 资源，避免 Python 析构时打印 [INFO]"""
    global _acl_resource, _detect_instance
    try:
        if _detect_instance is not None and _detect_instance._model is not None:
            try:
                _detect_instance._model.__del__()
            except Exception:
                pass
            _detect_instance._model = None
        _detect_instance = None

        if _acl_resource is not None:
            try:
                # AclLiteResource 通常有 release 或 destroy 方法
                if hasattr(_acl_resource, 'release'):
                    _acl_resource.release()
                elif hasattr(_acl_resource, 'destroy'):
                    _acl_resource.destroy()
            except Exception:
                pass
        _acl_resource = None
    except Exception:
        pass


if __name__ == '__main__':
    try:
        main()
    finally:
        # 兜底清理
        _explicit_cleanup()

