import os
import sys
import numpy as np
from PIL import Image

from ...base import BaseModel

LABELS_COCO = [
    "person", "bicycle", "car", "motorbike", "aeroplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
    "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

MODEL_WIDTH = 416
MODEL_HEIGHT = 416

# YOLOv3 三个尺度（由小到大）: (特征图尺寸, anchors)
# anchors 为 (w, h) 对，归一化到网格尺寸
SCALES = [
    (52,  [[10, 13], [16, 30], [33, 23]]),   # 52x52
    (26,  [[30, 61], [62, 45], [59, 119]]),  # 26x26
    (13,  [[116, 90], [156, 198], [373, 326]])  # 13x13
]

CONF_THRESHOLD = 0.30
NMS_IOU_THRESHOLD = 0.45
TOP_K = 10
DEBUG = os.environ.get('YOLOV3_DEBUG', '0') == '1'


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _decode_scale(feat, grid_size, anchors):
    """把单个尺度的输出特征图解码成检测框。
    支持输入 [1,255,H,W] 或 [255,H,W]。
    返回: x1,y1,x2,y2(归一化), scores, class_ids
    """
    feat = np.asarray(feat, dtype=np.float32)
    # 去掉可能的 batch 维
    if feat.ndim == 4:
        feat = feat[0]          # [255,H,W]
    # feat 现在是 [255,H,W]
    feat = np.transpose(feat, (1, 2, 0))  # [H,W,255]
    num_anchors = len(anchors)
    feat = feat.reshape(grid_size, grid_size, num_anchors, 85)

    box_xy = feat[..., 0:2]
    box_wh = feat[..., 2:4]
    box_conf = feat[..., 4]
    box_cls = feat[..., 5:]

    # 网格坐标：x 对应列(W)，y 对应行(H)
    xv = np.arange(grid_size, dtype=np.float32)[None, :, None]  # [1,W,1]
    yv = np.arange(grid_size, dtype=np.float32)[:, None, None]  # [H,1,1]

    # box_xy 原始是 [x, y]
    raw_x = box_xy[..., 0]  # [H,W,num_anchors]
    raw_y = box_xy[..., 1]
    cx = (_sigmoid(raw_x) + xv) / grid_size   # 归一化中心 x
    cy = (_sigmoid(raw_y) + yv) / grid_size   # 归一化中心 y

    # anchors 归一化到输入尺寸
    anchors_np = np.array(anchors, dtype=np.float32) / MODEL_WIDTH  # [w,h] 归一化
    aw = np.exp(box_wh[..., 0]) * anchors_np[None, None, :, 0]  # 归一化宽
    ah = np.exp(box_wh[..., 1]) * anchors_np[None, None, :, 1]  # 归一化高

    # 检测框（归一化 [0,1]）
    x1 = cx - aw / 2
    y1 = cy - ah / 2
    x2 = cx + aw / 2
    y2 = cy + ah / 2

    # 目标分数 = sigmoid(conf) * max(class)
    obj_conf = _sigmoid(box_conf)
    class_scores = _sigmoid(box_cls)  # [H,W,num_anchors,80]
    cls_max = np.max(class_scores, axis=-1)
    cls_id = np.argmax(class_scores, axis=-1)
    scores = obj_conf * cls_max

    return (x1.flatten(), y1.flatten(),
            x2.flatten(), y2.flatten(),
            scores.flatten(), cls_id.flatten())


class YOLOv3Detect(BaseModel):
    def __init__(self, model_path):
        super().__init__(model_path)
        self._model_width = MODEL_WIDTH
        self._model_height = MODEL_HEIGHT
        self._orig_size = (MODEL_WIDTH, MODEL_HEIGHT)

    def pre_process(self, image_path):
        """
        预处理：对齐 yolov3_backbone.onnx 的输入
          input_1: float32[1,3,416,416] RGB /255, NCHW
          image_shape: float32[1,2] [orig_w, orig_h]
        """
        pil_image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = pil_image.size
        self._orig_size = (orig_w, orig_h)

        pil_resized = pil_image.resize(
            (self._model_width, self._model_height), Image.BILINEAR)
        img_array = np.array(pil_resized, dtype=np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        if not img_array.flags['C_CONTIGUOUS']:
            img_array = np.ascontiguousarray(img_array)

        image_shape = np.array([[orig_w, orig_h]], dtype=np.float32)

        if DEBUG:
            print("[DEBUG] input_1 shape=%s" % (img_array.shape,))
            print("[DEBUG] image_shape=%s" % (image_shape,))

        return [img_array, image_shape]

    def inference(self, input_data):
        return self._model.execute(input_data)

    def post_process(self, infer_output):
        """
        三个输出：convolution_output2(13x13), convolution_output1(26x26), convolution_output(52x52)
        每个 [1,255,H,W]，做 anchor 解码 + 跨尺度合并 + NMS
        """
        if not (isinstance(infer_output, list) and len(infer_output) >= 3):
            return []

        # 三个输出顺序对应：13x13, 26x26, 52x52
        # 注意：AT 的输出顺序需要实测确认，先按 13/26/52 顺序
        feat_13 = np.array(infer_output[0])
        feat_26 = np.array(infer_output[1])
        feat_52 = np.array(infer_output[2])

        if DEBUG:
            print("[DEBUG] infer_output count:", len(infer_output))
            for i, o in enumerate(infer_output):
                print("[DEBUG] output%d shape=%s" % (i, o.shape))
            print("[DEBUG] feat_13 shape=%s feat_26 shape=%s feat_52 shape=%s" % (
                feat_13.shape, feat_26.shape, feat_52.shape))

        # 输出顺序：以实际 shape 推断。若 infer_output[i] 的元素数能整除 255 且为 H*W，
        # 用元素总数判断空间尺寸，保证 feat_13/feat_26/feat_52 正确对应。
        def _grid_of(arr):
            n = np.asarray(arr).size // 255
            return int(round(n ** 0.5))

        # 收集三个输出，按空间尺寸重排（13/26/52）
        outs = [infer_output[0], infer_output[1], infer_output[2]]
        size_map = {}
        for o in outs:
            size_map[_grid_of(o)] = o
        if 13 in size_map and 26 in size_map and 52 in size_map:
            feat_13 = np.array(size_map[13])
            feat_26 = np.array(size_map[26])
            feat_52 = np.array(size_map[52])
        else:
            # 无法从尺寸推断，退回顺序假设（可能导致维度错误）
            if DEBUG:
                print("[DEBUG] 未能从输出尺寸推断尺度顺序，回退到[0]=13,[1]=26,[2]=52")
            feat_13 = np.array(outs[0] if _grid_of(outs[0]) == 13 else outs[0])
            feat_26 = np.array(outs[1])
            feat_52 = np.array(outs[2])

        if DEBUG:
            print("[DEBUG] 重排后 feat_13=%s feat_26=%s feat_52=%s" % (
                feat_13.shape, feat_26.shape, feat_52.shape))

        all_x1, all_y1, all_x2, all_y2, all_scores, all_cls = [], [], [], [], [], []
        # 按空间尺度解码：13x13, 26x26, 52x52
        # SCALES 是 [(52,anchors),(26,anchors),(13,anchors)]，建立 grid->(feat,anchors) 映射
        scale_map = {g: a for g, a in SCALES}
        feat_map = {13: feat_13, 26: feat_26, 52: feat_52}
        for grid_size in [13, 26, 52]:
            anchors = scale_map[grid_size]
            feat = feat_map[grid_size]
            x1, y1, x2, y2, scores, cls = _decode_scale(feat, grid_size, anchors)
            all_x1.extend(x1)
            all_y1.extend(y1)
            all_x2.extend(x2)
            all_y2.extend(y2)
            all_scores.extend(scores)
            all_cls.extend(cls)

        all_x1 = np.array(all_x1)
        all_y1 = np.array(all_y1)
        all_x2 = np.array(all_x2)
        all_y2 = np.array(all_y2)
        all_scores = np.array(all_scores)
        all_cls = np.array(all_cls)

        orig_w, orig_h = self._orig_size
        results = []

        # 置信度过滤
        for i in range(len(all_scores)):
            score = float(all_scores[i])
            if score < CONF_THRESHOLD:
                continue
            label = int(all_cls[i])
            if label < 0 or label >= len(LABELS_COCO):
                continue

            x1_px = int(round(all_x1[i] * orig_w))
            y1_px = int(round(all_y1[i] * orig_h))
            x2_px = int(round(all_x2[i] * orig_w))
            y2_px = int(round(all_y2[i] * orig_h))

            # 钳制
            x1_px = max(0, min(orig_w, x1_px))
            y1_px = max(0, min(orig_h, y1_px))
            x2_px = max(0, min(orig_w, x2_px))
            y2_px = max(0, min(orig_h, y2_px))

            if (x2_px - x1_px) < 5 or (y2_px - y1_px) < 5:
                continue

            results.append({
                "class_id": label,
                "label": LABELS_COCO[label] if label < len(LABELS_COCO) else str(label),
                "confidence": score,
                "bbox": [x1_px, y1_px, x2_px, y2_px],
            })

        # NMS（按类）
        results = self._nms_results(results, iou_thresh=NMS_IOU_THRESHOLD)
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:TOP_K]

    def _nms_results(self, results, iou_thresh=0.45):
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