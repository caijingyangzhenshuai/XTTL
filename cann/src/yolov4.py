import os
import sys
import time
import threading
import numpy as np
from PIL import Image

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, ".."))
sys.path.append(os.path.join(path, "../examples/samples/python/common/"))
sys.path.append(os.path.join(path, "../examples/samples/python/common/acllite"))

# === 可靠的日志过滤器（文件描述符级别） ===
class StdoutFilter:
    def __init__(self, filter_func):
        self.filter_func = filter_func
        self._orig_out_fd = os.dup(1)
        self._orig_err_fd = os.dup(2)
        self._read_fd, self._write_fd = os.pipe()
        os.dup2(self._write_fd, 1)
        os.dup2(self._write_fd, 2)
        os.close(self._write_fd)
        
        self._buffer = b""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            try:
                data = os.read(self._read_fd, 4096)
                if not data:
                    break
                self._buffer += data
                while b"\n" in self._buffer:
                    line, self._buffer = self._buffer.split(b"\n", 1)
                    decoded = line.decode('utf-8', errors='replace')
                    if not self.filter_func(decoded):
                        os.write(self._orig_out_fd, line + b"\n")
            except OSError:
                break

    def stop(self):
        self._running = False
        os.close(self._read_fd)
        os.close(self._orig_out_fd)
        os.close(self._orig_err_fd)

# 检查是否显示 INFO
SHOW_INFO = '--show-info' in sys.argv
if not SHOW_INFO:
    stdout_filter = StdoutFilter(lambda line: line.strip().startswith("[INFO]"))

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

# YOLOv4 参数
MODEL_WIDTH = 416
MODEL_HEIGHT = 416
CONF_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.8
TOP_K = 10
DEBUG = os.environ.get('YOLOV4_DEBUG', '0') == '1'

_acl_resource = None
_detect_instance = None


def print_help():
    print("YOLOv4目标检测推理")
    print("Usage: yolov4.py [OPTIONS]")
    print("Options:")
    print("  --modelfile MODEL      指定模型文件路径（.om 文件）")
    print("  --imagefile IMAGE      指定输入图像路径")
#    print("  --show-info            显示底层库的 [INFO] 日志")
    print("  --help                 显示帮助信息")
    print("")
    print("Examples:")
    print("  python yolov4.py --modelfile /home/HwHiAiUser/cann/model/yolov4.om --imagefile /home/HwHiAiUser/cann/data/bus.jpg")


class YOLOv4Detect:
    def __init__(self, model_path):
        self._model_path = model_path
        self._model = None
        self._orig_size = (416, 416)

    def init(self):
        self._model = AclLiteModel(self._model_path)
        if DEBUG:
            print("[DEBUG] YOLOv4 model loaded: %s" % self._model_path)
        return 0

    @staticmethod
    def _rgb_to_nv12(rgb):
        """将 RGB 图像 (H, W, 3) 转换为 NV12 (YUV 420 Semi-Planar)"""
        h, w, _ = rgb.shape
        rgb_float = rgb.astype(np.float32)
        r, g, b = rgb_float[:,:,0], rgb_float[:,:,1], rgb_float[:,:,2]
        
        # Y (BT.601 标准)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        
        # U, V
        u = -0.169 * r - 0.331 * g + 0.5 * b + 128
        v = 0.5 * r - 0.419 * g - 0.081 * b + 128
        
        # 下采样到 (h/2, w/2)
        u_small = u[::2, ::2]
        v_small = v[::2, ::2]
        
        h_half, w_half = u_small.shape
        
        # 构建 UV 交错平面
        uv = np.zeros((h_half, w), dtype=np.uint8)
        uv[:, 0::2] = np.clip(u_small, 0, 255).astype(np.uint8)
        uv[:, 1::2] = np.clip(v_small, 0, 255).astype(np.uint8)
        
        # Y 平面
        y_uint8 = np.clip(y, 0, 255).astype(np.uint8)
        
        # 拼接 [Y_plane; UV_plane]
        nv12 = np.vstack([y_uint8, uv])
        return nv12.flatten()

    def pre_process(self, image_path):
        """预处理：resize 416x416, RGB 转 NV12"""
        img = np.array(Image.open(image_path).convert('RGB'))
        if img is None or img.size == 0:
            raise FileNotFoundError("无法读取图像: %s" % image_path)

        orig_h, orig_w = img.shape[:2]
        self._orig_size = (orig_w, orig_h)

        self.x_scale = orig_w / MODEL_WIDTH
        self.y_scale = orig_h / MODEL_HEIGHT

        # Resize 到 416x416
        img_resized = np.array(Image.fromarray(img).resize(
            (MODEL_WIDTH, MODEL_HEIGHT), Image.BILINEAR))
        
        # 转换为 NV12
        img_nv12 = self._rgb_to_nv12(img_resized)
        
        # 添加 batch 维度，形状: (1, 259584)
        img_input = np.expand_dims(img_nv12, axis=0).astype(np.uint8)

        if DEBUG:
            print("[DEBUG] Image preprocessed: shape=%s dtype=%s, size=%d" % (
                img_input.shape, img_input.dtype, img_input.size))

        return img_input

    def inference(self, input_data):
        if DEBUG:
            print("[DEBUG] Running inference...")
        result = self._model.execute([input_data])
        return result

    def post_process(self, infer_output):
        """后处理：解析 classes 和 boxes"""
        classes = infer_output[0][0]  # [10647, 80]
        boxes = infer_output[1][0]    # [10647, 4]  (cx, cy, w, h)

        if DEBUG:
            print("[DEBUG] classes shape=%s, boxes shape=%s" % (classes.shape, boxes.shape))

        results = []
        num_anchors = classes.shape[0]
        
        for i in range(num_anchors):
            class_probs = classes[i]
            max_idx = np.argmax(class_probs)
            max_score = float(class_probs[max_idx])
            
            if max_score < CONF_THRESHOLD:
                continue
                
            # 获取 (cx, cy, w, h) - 归一化坐标
            cx = float(boxes[i][0])
            cy = float(boxes[i][1])
            w = float(boxes[i][2])
            h = float(boxes[i][3])
            
            # 转换为像素坐标
            cx_px = cx * self.x_scale
            cy_px = cy * self.y_scale
            w_px = w * self.x_scale
            h_px = h * self.y_scale
            
            # 转换为 xyxy 格式
            x1 = cx_px - w_px / 2
            y1 = cy_px - h_px / 2
            x2 = cx_px + w_px / 2
            y2 = cy_px + h_px / 2
            
            # 裁剪到图像范围
            x1 = max(0, min(self._orig_size[0], x1))
            y1 = max(0, min(self._orig_size[1], y1))
            x2 = max(0, min(self._orig_size[0], x2))
            y2 = max(0, min(self._orig_size[1], y2))
            
            # 过滤太小的框
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            results.append({
                "class_id": int(max_idx),
                "label": LABELS_COCO[max_idx] if max_idx < len(LABELS_COCO) else str(max_idx),
                "confidence": max_score,
                "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
            })

        if DEBUG:
            print("[DEBUG] 检测数 (score >= %.2f): %d" % (CONF_THRESHOLD, len(results)))

        # NMS 去重
        results = self._nms_results(results, iou_thresh=NMS_IOU_THRESHOLD)
        
        # 按置信度排序
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:TOP_K]

    def _nms_results(self, results, iou_thresh=0.8):
        if not results:
            return results
            
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
        _detect_instance = YOLOv4Detect(modelfile)
        ret = _detect_instance.init()
        utils.check_ret("YOLOv4 init", ret)

    start_time = time.perf_counter()

    input_data = _detect_instance.pre_process(imagefile)
    result = _detect_instance.inference(input_data)
    output = _detect_instance.post_process(result)

    infer_cost = int(round((time.perf_counter() - start_time) * 1000))
    model_name = os.path.splitext(os.path.basename(modelfile))[0]

    return {
        "results": output,
        "infer_cost": infer_cost,
        "model_name": model_name,
    }


def _explicit_cleanup():
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
                if hasattr(_acl_resource, 'release'):
                    _acl_resource.release()
                elif hasattr(_acl_resource, 'destroy'):
                    _acl_resource.destroy()
            except Exception:
                pass
        _acl_resource = None
    except Exception:
        pass


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

    try:
        result = kzzk_cv(modelfile, imagefile)
    except Exception as e:
        print("错误: %s" % str(e))
        return
    finally:
        _explicit_cleanup()
        
    print("Total inference cost: %dms" % result['infer_cost'])
    print("Inference Results: ")
    print("YOLOv4目标检测推理")
    print("Total objects: %d" % len(result['results']))
    print("Format: [ID] (Class Name) [x1, y1, x2, y2] Confidence")
    for i, det in enumerate(result['results'], 1):
        display_pct = min(100, max(0, int(round(det['confidence'] * 100))))
        print("[%d] %d (%s) [%d,%d,%d,%d] %d%%" % (
            i, det['class_id'], det['label'],
            det['bbox'][0], det['bbox'][1],
            det['bbox'][2], det['bbox'][3],
            display_pct))

    txt_path = save_results(result, result['model_name'], imagefile)
    print("推理结果已保存到: %s" % txt_path)
    
    # 停止日志过滤器
    if 'stdout_filter' in globals():
        stdout_filter.stop()


if __name__ == '__main__':
    try:
        main()
    finally:
        _explicit_cleanup()
