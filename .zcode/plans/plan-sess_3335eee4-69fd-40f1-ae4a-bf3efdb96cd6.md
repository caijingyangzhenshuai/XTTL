# 修复 yolov3 / rt_detr / vgg_ssd 三个检测模型

## 根因诊断（对照 Python 参考实现）

三个模型的失败都源于同一个根因：**C++ 实现没有正确复刻 Python 的多输入构造逻辑**。acelite 的 `model.execute([img, info])` 会自动把列表里的每个元素构造为独立的 `aclDataBuffer` 并 `aclmdlAddDatasetBuffer` 进 dataset；而 C++ 实现只构造了 1 个 buffer，导致输入数量与模型定义不匹配 → `aclmdlExecute` 失败或段错误。

### 具体差异

| 模型 | Python 喂的输入 | C++ 现状 | 后果 |
|------|----------------|---------|------|
| **yolov3_yuv** | `[NV12 uint8, image_info(4个float)]` = 2 输入 | 只喂 1 个 NV12 | execute 失败，耗时 0 |
| **rtdetr_model** | `[FP32 CHW 图像, int64 target_sizes]` = 2 输入 | 只喂 1 个图像 | execute 失败，耗时 0 |
| **vgg_ssd** | AIPP模式: 单 uint8 304×300 NCHW；非AIPP: float 300×300 减mean | 固定300×300 float，不减mean、不检测AIPP | 格式/数值全错 → 段错误 |

此外 RT-DETR 的后处理也有错（Python 输出是 3 个 buffer：labels/boxes/scores，C++ 假设 2 个），VGG-SSD 标签用错了（Python 用 VOC 20 类，C++ 用 COCO 80 类）。

---

## 实施方案

### 改动 1：新增 `cann/cpp/utils/voc_labels.h`（新文件）
仿照 `coco_labels.h` 的结构，定义 VGG-SSD 需要的 VOC 20 类标签（含 background 共 21 个）：
```cpp
inline const std::vector<std::string>& GetVocLabels() {
    static const std::vector<std::string> labels = {
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    };
    return labels;
}
inline const std::string& GetVocLabel(int class_id) { ... }
```

### 改动 2：重写 `yolov3.cpp` + `yolov3.h` — 补 image_info 第二输入
- **预处理**：完全对齐 Python `_rgb_to_nv12`（手算 YUV 而非依赖 DVPP，避免硬件坑）+ resize 到 416×416，生成 NV12 uint8。
- **关键：构造 2 个输入 buffer**
  - input[0]：NV12 图像数据（uint8）
  - input[1]：`image_info` = `[416, 416, orig_w, orig_h]`（4 个 float32）—— 注意 Python 用的是 `[MODEL_W, MODEL_H, MODEL_W, MODEL_H]`，但后处理用 `self._orig_size` 还原坐标，所以 image_info 传原图尺寸用于坐标还原。**此处严格按 Python**：image_info = `[416.0, 416.0, 416.0, 416.0]`，坐标还原用 Python 后处理逻辑（`scalex=orig_w/416`，`scaley` 取 `max(scalex,scaley)`）。
- **后处理**：对齐 Python `post_process`（output[1] 是 box_num，output[0] 是 box_info，box 按 `box_info[n + k*box_num]` 列布局读取 x1/y1/x2/y2/score/id）。
- `CreateModelInput` 改为接收 buffer 列表，循环 `aclmdlAddDatasetBuffer` 添加每个输入。

### 改动 3：重写 `rt_detr.cpp` + `rt_detr.h` — 补 target_sizes 第二输入
- **预处理**：对齐 Python —— resize 到 640×640，RGB，/255 归一化，CHW float32（不做 ImageNet mean/std，Python 没做）。
- **关键：构造 2 个输入 buffer**
  - input[0]：FP32 CHW 图像（1×3×640×640）
  - input[1]：`orig_target_sizes` = `[[orig_w, orig_h]]`，**int64 类型**，1×2（注意是 int64 不是 float！需用 `aclmdlGetInputSizeByIndex` 查每个输入的真实 size 来分配）
- **后处理**：对齐 Python —— 3 个输出 buffer（output[0]=labels, output[1]=boxes, output[2]=scores），boxes 坐标已是原图尺寸（因为喂了 target_sizes），直接取 `score>=0.30` 的，NMS + 每类保留最多 2 个。
- 阈值对齐 Python：`CONF_THRESHOLD=0.30`, `NMS_IOU=0.45`, `MAX_PER_CLASS=2`, `TOP_K=10`。

### 改动 4：重写 `vgg_ssd.cpp` + `vgg_ssd.h` — 加 AIPP 检测 + 修正预处理
- **预处理**：对齐 Python `_detect_aipp_mode` + `pre_process`
  - 加 AIPP 模式探测：查 `aclmdlGetInputSizeByIndex(model_desc_, 0)`，若 < 1000000 字节 → AIPP 模式。
  - AIPP 模式：resize 到 **304×300**（注意是 304 宽！），转 NCHW uint8（`transpose(2,0,1)`），无归一化。
  - 非 AIPP 模式：resize 到 300×300，BGR→RGB，减 `mean=[104,117,123]`，转 NCHW float32。
- **后处理**：对齐 Python —— output[0] 首元素是 box_num，output[1] 是检测结果数组（每 8 个 float 一组：`[image_id, label, score, x1, y1, x2, y2, ...]`），坐标是归一化 [0,1]，乘原图尺寸还原。
- **标签**：从 COCO 改为 VOC（`#include "voc_labels.h"`，`kClassNum=21`）。
- 阈值对齐 Python：`CONF_THRESHOLD=0.05`, `NMS_IOU=0.30`, `MAX_PER_CLASS=1`, `HIGH_CONF_THRESHOLD=0.10`。

### 改动 5：`main.cpp` 的提示文案（可选，不影响功能）
当前 `PrintClassificationResults` 里有「可能不在 cat/dog 范围内」的误导文案（MobileNetV1 是 1000 类 ImageNet）。这一项**默认不动**（你之前要求保持），除非你要求改。

---

## 不改动的部分
- `yolov4.cpp`（已正常工作）
- `mobilenetv1.cpp`（上一轮已修好）
- `kzzk_cv.cpp`、`CMakeLists.txt`（GLOB_RECURSE 会自动纳入新增文件，无需改构建配置；`create_model` 的文件名匹配逻辑已覆盖 `yolov3`/`vgg_ssd`/`rt_detr`）

## 验证方式
重新编译（`cd cann/cpp && cmake . && make`）后，分别跑：
```
./kzzk_cv --modelfile .../yolov3_yuv.om --imagefile .../dog.jpg
./kzzk_cv --modelfile .../rtdetr_model.om --imagefile .../dog.jpg
./kzzk_cv --modelfile .../vgg_ssd.om --imagefile .../dog.jpg
```
预期：三个都能输出 `[ID] (dog) [x1,y1,x2,y2] Confidence` 检测框。

## 风险与备注
- RT-DETR 和 VGG-SSD 的输出 buffer 布局是基于 Python 实现推断的。如果实际 OM 模型的输出顺序/格式与 Python 假设不同，后处理可能解析出错。我会在后处理中加入 size 校验和防御性边界检查（避免段错误），如果第一次跑结果异常，可以临时打开调试日志看真实输出。
- yolov3 的 image_info 具体传什么值，不同 OM 导出方式可能不同。我会严格按 Python 传 `[416,416,416,416]`；若坐标还原不对，再调整为 `[orig_w, orig_h]`。
- 三个模型都会保持「无中间调试日志」的约定（对齐你之前清理 mobilenetv1 的要求），只在出错时打印 `[ERROR]`。