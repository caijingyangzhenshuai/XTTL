## 目标
用 `yolov3-10.onnx` 替换现有的 YOLOv3 模型实现（NV12 版 → ONNX 版），保持工厂关键字 `yolov3` 不变，修改 Python 和 C++ 的模型类。

## 模型签名
**输入**（2 个）：
- `input_1`：`float32[1,3,416,416]`（NCHW，RGB /255）
- `image_shape`：`float32[1,2]`（[orig_w, orig_h]）

**输出**（3 个）：
- `yolonms_layer_1/ExpandDims_1:0`：`float32[1,nbox,4]` bboxes（原图像素坐标，x1y1x2y2）
- `yolonms_layer_1/ExpandDims_3:0`：`float32[1,nbox,1]` scores（最高分）
- `yolonms_layer_1/concat_2:0`：`int32[1,nbox]` class_id（标准 COCO 索引，0=person）

## 改动文件
### Python
1. **重写** `cann/python/models/detection/yolov3.py` — 现有 `YOLOv3Detect` 替换为新的 ONNX 版
   - `pre_process`：读图 RGB → resize 416×416 → /255 → CHW → 返回 [img_input, image_shape]
   - `inference`：`self._model.execute([img, image_info])`（2 输入）
   - `post_process`：解析 3 输出 → 按 scores 阈值过滤 → 读出 bbox/labels → 直接返回（bboxes 已是原图像素坐标）→ TopK

2. **重写** `cann/python/models/__init__.py` 不变（yolov3 关键字不变，类名 YOLOv3Detect 不变）

### C++
3. **重写** `cann/cpp/inc/models/detection/yolov3.h` — 保持类名 `YOLOv3`，但接口改为 ONNX 版
4. **重写** `cann/cpp/src/models/detection/yolov3.cpp` — 完全替换
5. **CMakeLists.txt** 不变（源文件名 yolov3.cpp 不变）
6. **kzzk_cv.cpp** 工厂不变（yolov3 关键字不变）

### 文档
7. **追加** `cann/model/model-conversion.txt` 写入 ATC 命令

## ATC 转换命令
### 310P3 版
```
atc --model=./yolov3-10.onnx --framework=5 \
    --input_shape="input_1:1,3,416,416;image_shape:1,2" \
    --input_format=NCHW --output_type=FP32 \
    --output=./yolov3 \
    --soc_version=Ascend310P3 --log=debug
```

### 910B4 版
```
atc --model=./yolov3-10.onnx --framework=5 \
    --input_shape="input_1:1,3,416,416;image_shape:1,2" \
    --input_format=NCHW --output_type=FP32 \
    --output=./yolov3 \
    --soc_version=Ascend910B4 --log=debug
```

## 关键设计
- `image_shape` 输入告诉模型原图尺寸，模型输出的 bbox 直接是**原图像素坐标**（不需要手动缩放）
- 预处理：RGB /255（与 `ssd.py` 预处理一致，但 resize 到 416×416）
- 后处理：针对 3 个输出，scores 和 labels 按 `[1, nbox]` 读取，bboxes 按 `[1, nbox, 4]` 读取（直接作为像素坐标）
- 类别：标准 COCO 80 类（person=0, bus=5, dog=16 等），无需 offset
- 由于 `ExpandDims_3:0` 可能是 `[1,nbox,1]`，读作 `float32[1,nbox,1]` → squeeze 成 `[nbox]`