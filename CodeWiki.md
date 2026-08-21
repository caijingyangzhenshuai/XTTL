# 智能加速卡协同推理软件 - Code Wiki

> 本项目是一个面向 Ascend NPU（华为昇腾智能加速卡）的协同推理框架，包含两大子系统：
> 1. **CANN CV**：计算机视觉推理（目标检测、图像分类、语义分割），提供 Python 与 C++ 双语言实现
> 2. **MindIE LLM**：大语言模型推理客户端，提供 RESTful API 封装

---

## 目录

1. [项目整体架构](#1-项目整体架构)
2. [CANN CV 子系统](#2-cann-cv-子系统)
   - 2.1 [Python 实现](#21-python-实现)
   - 2.2 [C++ 实现](#22-c-实现)
   - 2.3 [支持的视觉模型](#23-支持的视觉模型)
3. [MindIE LLM 子系统](#3-mindie-llm-子系统)
   - 3.1 [Python 客户端](#31-python-客户端)
   - 3.2 [C++ 客户端](#32-c-客户端)
4. [关键数据结构](#4-关键数据结构)
5. [模型输入/输出规范](#5-模型输入输出规范)
6. [依赖关系](#6-依赖关系)
7. [构建与部署](#7-构建与部署)
8. [运行方法与示例](#8-运行方法与示例)
9. [设计要点与注意事项](#9-设计要点与注意事项)

---

## 1. 项目整体架构

### 1.1 目录结构

```
code/
├── cann/                              # CANN CV 子系统（计算机视觉推理）
│   ├── cpp/                           # C++ 实现
│   │   ├── inc/                       # 头文件目录
│   │   │   ├── kzzk_cv.h              # 统一对外接口
│   │   │   ├── base_model.h           # 抽象基类
│   │   │   ├── types.h                # 数据结构定义
│   │   │   └── models/                # 各模型头文件
│   │   │       ├── detection/         # 目标检测
│   │   │       │   ├── yolov4.h
│   │   │       │   ├── yolov3.h
│   │   │       │   ├── vgg_ssd.h
│   │   │       │   └── rt_detr.h
│   │   │       ├── classification/    # 图像分类
│   │   │       │   ├── resnet50.h
│   │   │       │   ├── mobilenetv1.h
│   │   │       │   └── vgg16.h
│   │   │       └── segmentation/      # 语义分割
│   │   │           └── deeplabv3.h
│   │   ├── src/                       # 源文件目录（与 inc/ 结构对应）
│   │   │   ├── kzzk_cv.cpp            # 统一接口实现（工厂模式）
│   │   │   ├── base_model.cpp         # 抽象基类实现
│   │   │   ├── main.cpp               # 命令行入口
│   │   │   └── models/                # 各模型实现
│   │   ├── utils/                     # 工具头文件
│   │   │   ├── coco_labels.h          # COCO 80类标签
│   │   │   └── voc_labels.h           # VOC 20类标签
│   │   └── CMakeLists.txt             # CMake 构建脚本
│   ├── python/                        # Python 实现
│   │   ├── kzzk_cv.py                 # 命令行入口 & 对外 API
│   │   ├── __init__.py                # Python 包入口（kzzk_cv 函数）
│   │   ├── base.py                    # 抽象基类 BaseModel
│   │   ├── models/                    # 各模型实现
│   │   │   ├── detection/
│   │   │   ├── classification/
│   │   │   └── segmentation/
│   │   └── utils/
│   │       └── log_filter.py          # [INFO] 日志过滤器（FD 级重定向）
│   ├── model/                         # .om 模型文件（CANN 离线模型）
│   ├── data/                          # 测试图像
│   ├── third_party/                   # 第三方库（OpenCV）
│   ├── examples/                      # CANN 官方样例 & 测试脚本
│   │   └── samples/                   # 含 acllite 公共库（Python/C++）
│   └── install_cann.sh                # CANN 一键安装/部署脚本
│
├── mindie/                            # MindIE LLM 子系统（大模型推理）
│   ├── inc/
│   │   └── kzzk_llm.h                 # C++ 头文件
│   ├── src/
│   │   ├── kzzk_llm.cpp               # C++ 实现（基于 libcurl）
│   │   ├── kzzk_py_llm.py             # Python 实现（基于 requests）
│   │   ├── main.cpp                   # C++ 命令行入口
│   │   └── CMakeLists.txt             # CMake 构建脚本
│   ├── examples/
│   │   ├── test_llm.cpp
│   │   └── test_py_llm.py
│   ├── third_party/
│   │   ├── cpr/                       # C++ HTTP 库（备选）
│   │   └── json/                      # nlohmann/json 头文件库
│   └── install_mindie.sh              # MindIE 一键部署脚本
│
└── .gitignore
```

### 1.2 系统架构图

```
┌──────────────────────────────────────────────────────────┐
│                     用户调用层                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Python CLI  │  │  C++ CLI     │  │  Library API │   │
│  │  kzzk_cv.py  │  │  ./kzzk_cv   │  │  kzzk_cv()   │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
└─────────┼─────────────────┼─────────────────┼───────────┘
          │                 │                 │
┌─────────▼─────────────────▼─────────────────▼───────────┐
│                  工厂模式模型分发层                        │
│  extract_model_name() → create_model() → 具体子类        │
│  (按文件名关键字匹配模型类型)                             │
└──────┬──────────────┬──────────────────┬─────────────────┘
       │              │                  │
┌──────▼──────┐ ┌─────▼────────┐ ┌───────▼─────────┐       │
│ 目标检测     │ │ 图像分类      │ │ 语义分割         │       │
│ YOLOv4/3    │ │ ResNet50     │ │ DeepLabV3       │       │
│ VGG-SSD     │ │ MobileNetV1  │ │                 │       │
│ RT-DETR     │ │ VGG16        │ │                 │       │
└──────┬──────┘ └─────┬────────┘ └───────┬─────────┘       │
       │              │                  │                  │
└──────▼──────────────▼──────────────────▼──────────────────┘
                  ┌───────────────┐
                  │  Ascend NPU   │
                  │  (CANN ACL)   │
                  └───────────────┘

── MindIE LLM 架构 ──────────────────────────────────────

┌──────────────┐   HTTP/REST    ┌──────────────────────┐
│ Python/C++   │ ─────────────► │ MindIE Service       │
│ Client       │                │ (mindieservice_daemon)│
│ kzzk_llm()   │ ◄───────────── │ (端口 1025 / 8000)   │
└──────────────┘                └──────────┬───────────┘
                                            │
                                     ┌──────▼──────┐
                                     │ Ascend NPU  │
                                     │ 大模型推理   │
                                     └─────────────┘
```

---

## 2. CANN CV 子系统

### 2.1 Python 实现

#### 2.1.1 包结构

Python 代码位于 [cann/python/](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/python)，包名为 `python`。

#### 2.1.2 对外接口

主入口函数定义在 [cann/python/__init__.py](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/python/__init__.py#L51-L103)：

```python
def kzzk_cv(modelfile, imagefile) -> dict
```

**返回值格式：**
```python
{
    "model_name": "yolov4",
    "model_type": "detection" | "classification" | "segmentation",
    "infer_cost": 25,              # 毫秒
    "results": [...]               # 结果列表（见各模型）
}
```

命令行入口在 [cann/python/kzzk_cv.py](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/python/kzzk_cv.py)。

#### 2.1.3 抽象基类 `BaseModel`

定义在 [cann/python/base.py](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/python/base.py#L38-L74)：

| 方法 | 作用 |
|------|------|
| `__init__(model_path)` | 构造，传入 .om 模型路径 |
| `init()` | 初始化 ACL 资源、加载模型（懒加载全局单例） |
| `release()` | 释放模型资源 |
| `pre_process(image_path)` | **抽象方法**：图像预处理，返回模型输入数据 |
| `inference(input_data)` | **抽象方法**：调用 NPU 推理，返回原始输出 |
| `post_process(infer_output)` | **抽象方法**：后处理，返回结构化结果 |
| `run(image_path)` | 模板方法：pre_process → inference → post_process |

**全局 ACL 资源管理：**
- `_get_acl_resource()`：懒初始化 `AclLiteResource` 单例
- `_release_acl_resource()`：退出时释放

#### 2.1.4 日志过滤器 `StdoutFilter`

定义在 [cann/python/utils/log_filter.py](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/python/utils/log_filter.py)。

**实现原理：** 文件描述符（FD）级重定向
1. `os.dup(1)` 备份原始 stdout/stderr FD
2. `os.pipe()` 创建管道，`os.dup2(pipe_write, 1/2)` 将 stdout/stderr 重定向到管道写入端
3. 后台线程从管道读取端消费数据，按行过滤 `[INFO]` 开头的行
4. 未过滤的行 `os.write` 回原始 stdout FD

**注意：** 必须在 `_release_acl_resource()` 之后调用 `stop()`，否则资源释放产生的 [INFO] 日志会泄漏。

---

### 2.2 C++ 实现

#### 2.2.1 目录结构

C++ 代码位于 [cann/cpp/](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp)。

#### 2.2.2 统一对外接口

声明在 [cann/cpp/inc/kzzk_cv.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/inc/kzzk_cv.h)：

```cpp
namespace kzzk {
    InferenceResult kzzk_cv(const std::string& modelfile, const std::string& imagefile);
}
```

实现位于 [cann/cpp/src/kzzk_cv.cpp](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/src/kzzk_cv.cpp#L84-L153)，执行流程：

1. **文件存在性检查**：`file_exists()` 验证模型和图像路径
2. **模型名解析**：`extract_model_name()` 从路径中提取 basename（去扩展名、小写）
3. **工厂创建**：`create_model()`  按关键字匹配 → 返回 `unique_ptr<BaseModel>`
   - 关键字 `yolov4` → `YOLOv4`
   - 关键字 `yolov3` → `YOLOv3`
   - 关键字 `vgg_ssd` / `vgg-ssd` → `VGG_SSD`
   - 关键字 `rt_detr` / `rt-detr` / `rtdetr` → `RT_DETR`
   - 关键字 `resnet50` → `ResNet50`
   - 关键字 `mobilenet` → `MobileNetV1`
   - 关键字 `vgg16` → `VGG16`
   - 关键字 `deeplab` / `deeplabv3` → `DeepLabV3`
4. **模型初始化**：`model->Initialize(model_path, 0)`
5. **执行推理**：`model->Infer(image_path)` → 填充 `InferenceResult`
6. **资源清理**：`model->Finalize()`

**静默模式开关：** 在 `kzzk_cv.cpp` L22 定义 `KZZK_CV_SILENT_MODE` 宏，注释掉即可恢复 [INFO]/[DEBUG]/[WARN] 日志输出。

#### 2.2.3 抽象基类 `BaseModel`

声明在 [cann/cpp/inc/base_model.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/inc/base_model.h)：

```cpp
class BaseModel {
public:
    virtual bool Initialize(const std::string& model_path, int device_id = 0) = 0;
    virtual void Finalize() = 0;
    virtual InferenceResult Infer(const std::string& image_path) = 0;
    virtual ModelType GetModelType() const = 0;  // DETECTION / CLASSIFICATION / SEGMENTATION
    virtual std::string GetModelName() const = 0;
protected:
    int device_id_;
    bool initialized_;
    std::string model_path_;
};
```

每个具体模型子类内部都有完整的 ACL 资源生命周期管理：
- `InitAclResource()` / `DestroyAclResource()`：Context、Stream
- `LoadModel()` / `UnloadModel()`：`aclmdlLoadFromFile` → `model_id_`
- `CreateModelInput()` / `DestroyModelInput()`：`aclmdlDataset` 输入数据集
- `CreateModelOutput()` / `DestroyModelOutput()`：`aclmdlDataset` 输出数据集
- `ExecuteModel()`：`aclmdlExecute`

---

### 2.3 支持的视觉模型

#### 2.3.1 目标检测（Detection）

| 模型 | 输入格式 | 输入尺寸 | 类别数 | 置信度阈值 | NMS IOU | 输出锚点数 |
|------|----------|----------|--------|-----------|---------|-----------|
| YOLOv4 | NV12 (uint8) | 416×416 | 80 (COCO) | 0.4 | 0.8 | 10647 |
| YOLOv3 | NV12 (uint8) | 416×416 | 80 (COCO) | - | - | - |
| VGG-SSD | - | 300×300 | 20 (VOC) | - | - | - |
| RT-DETR | - | - | 80 (COCO) | - | - | - |

**YOLOv4 输出解析（关键逻辑）：**
- 输出分两个 buffer：`output[0]` = 分类概率 `[10647 × 80]`，`output[1]` = 坐标 `[10647 × 4]`
- 坐标格式：`(cx, cy, w, h)` 归一化值，需乘以 `x_scale / y_scale` 映射回原图
- 锚点数 `10647 = (13² + 26² + 52²) × 3`（3 种尺度 × 3 个 anchor）
- **重要：** NPU 输出 buffer 在 Device 内存，必须 `aclrtMemcpy(DEVICE_TO_HOST)` 后才能 CPU 解析（否则段错误）

对应文件：
- Python: [cann/python/models/detection/yolov4.py](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/python/models/detection/yolov4.py)
- C++ 头: [cann/cpp/inc/models/detection/yolov4.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/inc/models/detection/yolov4.h)
- 标签: [cann/cpp/utils/coco_labels.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/utils/coco_labels.h)

#### 2.3.2 图像分类（Classification）

| 模型 | 输入格式 | 输入尺寸 | 类别数 | Top-K |
|------|----------|----------|--------|-------|
| ResNet50 | NV12 or RGB FP32 | 224×224 | 1000 (ImageNet) | 5 |
| MobileNetV1 | - | 224×224 | 1000 (ImageNet) | 5 |
| VGG16 (cat/dog) | NV12 (uint8) | **224×256** | **2** (cat/dog) | 5 |

**VGG16 特殊设计（关键！）：**
1. **输入尺寸非正方形**：224(宽) × 256(高)，86016 字节 = 224×256×1.5 (NV12 = Y + UV 4:2:0)
2. **输入类型为 uint8**：不是 float32，直接给 AIPP 处理
3. **不加 Softmax**：模型输出已是概率值（已含 Softmax 层），再做 Softmax 会破坏置信度
4. **OOD 双保险过滤**（见 [cann/cpp/inc/models/classification/vgg16.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/inc/models/classification/vgg16.h#L65-L72)）：
   - `kMinConfidencePercent = 60`：Top1 置信度 < 60% → 拒绝
   - `kMaxTopGapPercent = 98`：Top1-Top2 差值 > 98% → 拒绝（防止二分类模型对 OOD 输入强制 100%/0%）
   - 过滤后输出空结果，main.cpp 显示：`"未识别到模型训练范围内的类别"`

**NV12 预处理精确对齐（Python/C++ 一致）：**
- RGB → YUV 系数：Y=0.257R+0.504G+0.098B+16, U=-0.148R-0.291G+0.439B+128, V=0.439R-0.368G-0.071B+128
- resize：使用 INTER_AREA（PIL Image.BILINEAR / OpenCV）
- UV 下采样：2×2 逐点平均（即取偶数索引 `[::2,::2]`）
- NV12 字节序：U在前、V在后（偶数字节=U，奇数字节=V），**不是 NV21（V前U后）**

对应文件：
- Python: [cann/python/models/classification/vgg16.py](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/python/models/classification/vgg16.py)
- C++ 头: [cann/cpp/inc/models/classification/vgg16.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/inc/models/classification/vgg16.h)

#### 2.3.3 语义分割（Segmentation）

| 模型 | 输入格式 | 输入尺寸 | 类别数 |
|------|----------|----------|--------|
| DeepLabV3+ | RGB (int8) | 513×513 | 21 (VOC) |

**后处理步骤：**
1. 输出 `[513×513]` 分割图，每个像素值 ∈ [0, 20]（类别 ID）
2. **NEAREST 邻域插值** resize 回原图尺寸（不能用双线性，否则产生新类别值）
3. 类别统计：`np.unique` 统计每类像素数与比例（<1% 的类别过滤）
4. 输出：类别信息（ID、名称、像素数、比例%）+ 原始分割图
5. 保存可视化：原图 × 0.6 + 着色分割图 × 0.4 → overlay 保存为 `{base}_seg.png`

对应文件：
- Python: [cann/python/models/segmentation/deeplabv3.py](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/python/models/segmentation/deeplabv3.py)
- C++ 头: [cann/cpp/inc/models/segmentation/deeplabv3.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/inc/models/segmentation/deeplabv3.h)

---

## 3. MindIE LLM 子系统

MindIE 是大语言模型推理服务，客户端通过 HTTP REST API 调用。服务端由 `mindieservice_daemon` 启动，默认监听 **1025** 端口。

### 3.1 Python 客户端

定义在 [mindie/src/kzzk_py_llm.py](file:///c:/Users/迟昕玥/Desktop/协同推理/code/mindie/src/kzzk_py_llm.py)。

#### 核心 API

```python
def kzzk_llm(modelfile: str, prompt: str,
             server_url=None, max_tokens=1024, temperature=0.7) -> str

def list_models(server_url=None) -> list[str]
```

**多端点自动容错：** 按顺序尝试 4 种常见 API 格式，任一成功即返回：

| 顺序 | 端点 | 请求格式 |
|------|------|----------|
| 1 | `/v1/chat/completions` | OpenAI ChatML：`{model, messages:[{role,content}], ...}` |
| 2 | `/v1/completions` | 传统 Completion：`{model, prompt, ...}` |
| 3 | `/api/text-generation` | HF Text-Generation：`{model, input, ...}` |
| 4 | `/generate` | 自定义：`{model, prompt, max_new_tokens, ...}` |

**响应解析自动适配多种字段：**
- `choices[0].text` / `choices[0].message.content`
- `response` / `text` / `output.text` / `output`

**重试机制：** 每个端点最多重试 `MAX_RETRIES=3` 次，间隔 `RETRY_DELAY=2` 秒。

**响应清理：** `_clean_response()` 移除 `<think_>...</think_>` 思维链标记、清理前导标点。

命令行入口：`python3 kzzk_py_llm.py --listModels / --modelfile X --prompt Y`

---

### 3.2 C++ 客户端

头文件：[mindie/inc/kzzk_llm.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/mindie/inc/kzzk_llm.h)
实现：[mindie/src/kzzk_llm.cpp](file:///c:/Users/迟昕玥/Desktop/协同推理/code/mindie/src/kzzk_llm.cpp)

#### 核心类 `LLMClient`

```cpp
namespace kzzk {
    // 推理参数
    struct InferenceOptions {
        float temperature;    // default 0.7
        int max_tokens;       // default 1024
        float top_p, top_k;
        float repetition_penalty;
        int num_beams;
        bool do_sample;
    };

    struct ChatMessage { std::string role, content; };
    struct ModelInfo { std::string id, object, owned_by; int64_t created; };

    class LLMClient {
    public:
        LLMClient(const std::string& baseUrl = "http://127.0.0.1:8000",
                  long timeoutSeconds = 300, bool enableDebug = false);

        // 统一聊天接口
        std::string chat(const std::string& modelfile,
                        const std::vector<ChatMessage>& messages,
                        bool stream = false,
                        const InferenceOptions& params = {},
                        const std::map<std::string, nlohmann::json>& options = {});

        // JSON格式响应接口
        std::string chatWithJson(const std::string& modelfile,
                        const std::vector<ChatMessage>& messages,
                        bool stream = false,
                        const InferenceOptions& params = {});

        // 直接 POST 原始 JSON
        std::string postJson(const std::string& endpoint, const std::string& body);

        std::vector<ModelInfo> listModels();
        ModelInfo getModelInfo(const std::string& modelName);
    };

    // 全局便捷函数（单轮对话）
    std::string kzzk_llm(const std::string& modelfile, const std::string& prompt);
    std::string kzzk_llm(const std::string& modelfile, const std::string& prompt,
                         const InferenceOptions& options);
}
```

**实现特点：**
- 使用 **PImpl 惯用法**（`class Impl; unique_ptr<Impl> pImpl`）隐藏 libcurl 细节
- 基于 **libcurl** 发送 HTTP POST，自动重试 3 次（间隔 5 秒）
- JSON 序列化/反序列化使用 **nlohmann/json**
- 禁止拷贝（`= delete`），允许移动（`= default`）

---

## 4. 关键数据结构

### 4.1 CV 推理结果（C++）

定义在 [cann/cpp/inc/types.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/inc/types.h)：

```cpp
enum class ModelType { DETECTION, CLASSIFICATION, SEGMENTATION };

struct BBox { int x1, y1, x2, y2; };  // 左上+右下（像素坐标）

struct DetectionResult {
    int class_id;
    std::string label;
    float confidence;          // [0.0, 1.0]
    BBox bbox;
};

struct ClassificationResult {
    int class_id;
    int confidence;            // 百分比整数 [0, 100]
};

struct ClassInfo {
    int id;
    std::string name;
    int pixels;
    float ratio;               // 百分比
};

struct SegmentationResult {
    int width, height;
    std::vector<ClassInfo> class_info;
    std::vector<unsigned char> seg_map;  // W×H 单通道类别图
};

struct InferenceResult {
    std::string model_name;
    ModelType model_type;
    int infer_cost_ms;
    // 三种结果只有对应类型的字段被填充
    std::vector<DetectionResult> detections;
    std::vector<ClassificationResult> classifications;
    SegmentationResult segmentation;
};
```

### 4.2 CV 推理结果（Python）

```python
# Detection results[i]:
{
    "class_id": 1,
    "label": "bicycle",
    "confidence": 0.92,         # float [0,1]
    "bbox": [x1, y1, x2, y2]    # ints 像素
}

# Classification results[i]:
{
    "class_id": 207,
    "label": "golden retriever",
    "confidence": 87            # int 百分比
}

# Segmentation results:
{
    "image_size": [W, H],
    "class_info": [
        {"id": 15, "name": "person", "pixels": 12345, "ratio": 23.45},
        ...
    ],
    "seg_map": np.ndarray(W, H, dtype=uint8)
}
```

---

## 5. 模型输入/输出规范

### 5.1 模型文件格式

所有 CV 模型使用 **CANN 离线模型格式（.om）**，由 CANN 工具链（ATC 或 torch_npu.onnx_export + atc）从 PyTorch/ONNX 转换而来。

预训练模型存放于 [cann/model/](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/model)：

| 文件名 | 对应模型 |
|--------|----------|
| `yolov4.om` | YOLOv4 目标检测 |
| `yolov3_yuv.om` | YOLOv3 目标检测 |
| `vgg_ssd.om` | VGG-SSD 目标检测 |
| `rtdetr_model.om` | RT-DETR 目标检测 |
| `resnet50.om` | ResNet50 分类 |
| `vgg16_cat_dog.om` | VGG16 猫/狗二分类 |
| `mobilenet-v1_bs1.om` / `mobilenet-v1_bs32.om` | MobileNetV1 分类 |
| `deeplabv3_plus.om` | DeepLabV3+ 语义分割 |

### 5.2 输入图像格式速查

| 模型 | 格式 | 数据类型 | 尺寸 (W×H) | 总字节数 |
|------|------|----------|-----------|---------|
| YOLOv4 | NV12 | uint8 | 416×416 | 416×416×1.5 = 259,584 |
| YOLOv3 | NV12 | uint8 | 416×416 | 259,584 |
| ResNet50 | RGB | FP32 / uint8 | 224×224 | 224×224×3×4=602,112 |
| VGG16 | NV12 | uint8 | 224×256 | 224×256×1.5 = 86,016 |
| MobileNetV1 | NV12 | uint8 | 224×224 | 224×224×1.5=75,264 |
| DeepLabV3 | RGB | int8 | 513×513 | 513×513×3=789,507 |

> **NV12 说明：** YUV 4:2:0 Semi-Planar，布局 = `W×H` Y 平面 + `W×H/2` UV 交错平面。UV 平面中 U 在偶数位、V 在奇数位。

### 5.3 输出格式速查

| 任务类型 | CLI 输出格式 | 结果保存 |
|---------|-------------|---------|
| 目标检测 | `[序号] ClassID (名称) [x1,y1,x2,y2] 置信度%` | `{timestamp}_{model}.txt` |
| 图像分类 | `Top 5 → Result N: [ID 置信度%]` | `{timestamp}_{model}.txt` |
| 语义分割 | `Class N: [ID] (名称) [像素数] [比例%]` + 分割图路径 | txt + `{base}_seg.png` |

---

## 6. 依赖关系

### 6.1 CANN CV 依赖

#### C++ 依赖

| 依赖 | 版本要求 | 用途 | 来源 |
|------|---------|------|------|
| **CANN ACL (ascendcl)** | 匹配 NPU 驱动 | NPU 推理核心 API | `/usr/local/Ascend/ascend-toolkit/latest` |
| **acl_dvpp** | 同上 | 图像预处理（JPEG解码、Resize、格式转换） | 同 ACL |
| **OpenCV** | 4.x（推荐 4.5+） | 图像读写、Resize、着色、保存 | 优先系统 apt/yum，其次 third_party/opencv |
| **libjpeg / libtiff / libpng / zlib** | 系统版 | OpenCV imgcodecs 依赖 | 系统包管理器 |
| **C++ 标准** | ≥ C++14 | `std::make_unique` 等 | GCC 5+ |
| **CMake** | ≥ 3.10 | 构建系统 | - |
| **线程/动态链接** | pthread / dl / stdc++ | 运行时 | 系统 |

OpenCV 搜索优先级（见 CMakeLists.txt L140-L235）：
1. `find_package(OpenCV)` 系统包
2. 标准系统路径手动搜索 `/usr/lib/x86_64-linux-gnu` 等
3. 项目自带 `third_party/opencv`（`lib/` 或 `lib64/`，含版本号 .so）

#### Python 依赖

| 依赖 | 用途 |
|------|------|
| **Python** | ≥ 3.7 |
| **numpy** | 图像/张量处理 |
| **Pillow (PIL)** | 图像读写、resize |
| **acllite** (Python 版) | `AclLiteResource`, `AclLiteModel` 封装 |
| 系统库：**os, sys, threading, atexit, time** | 标准库 |

`acllite` 来源：`cann/examples/samples/python/common/acllite/`（代码中通过 sys.path 注入）

### 6.2 MindIE LLM 依赖

#### C++ 依赖

| 依赖 | 用途 | 来源 |
|------|------|------|
| **C++17** | `std::optional` 等现代语法 | GCC 8+ |
| **libcurl** | HTTP 客户端 | 系统: `find_package(CURL REQUIRED)` |
| **Threads (pthread)** | 线程 | 系统 |
| **nlohmann/json** ≥ 3.0 | JSON 序列化 | `mindie/third_party/json/include`（头文件-only） |
| **CMake** ≥ 3.16 | 构建 | - |

#### Python 依赖

| 依赖 | 用途 |
|------|------|
| **requests** | HTTP 客户端 |
| **re, sys, time, logging** | 标准库 |

---

## 7. 构建与部署

### 7.1 CANN CV 构建

#### 一键脚本（推荐 aarch64 NPU 服务器）

```bash
sudo bash cann/install_cann.sh
```

脚本执行流程（详见 [install_cann.sh](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/install_cann.sh)）：
1. 检查 root、aarch64 架构、NPU 驱动 (`/dev/davinci0`)、CANN 路径
2. 检查 cmake/gcc/g++、OpenCV
3. `cann/cpp/` 下 `rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)`
4. `cpp/build/bin/kzzk_cv` → 拷贝到 `/usr/local/bin/kzzk_cv`

#### 手动构建

```bash
# 1. 准备 CANN 环境（如果未设置）
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 2. 环境变量（可选，CMake 会自动搜索以下路径）
export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_TOOLKIT_PATH=/usr/local/Ascend/ascend-toolkit/latest

# 3. 构建
cd cann/cpp
rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 4. 输出位置
./bin/kzzk_cv --help
```

CMake Ascend 自动搜索顺序（[CMakeLists.txt](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/CMakeLists.txt) L16-L139）：
- 环境变量：`$ASCEND_HOME` → `$INSTALL_DIR/runtime` → `$ASCEND_TOOLKIT_PATH`
- 标准路径：`/usr/local/Ascend/ascend-toolkit/latest` → `/usr/local/Ascend/nnrt/latest` → `/home/HwHiAiUser/Ascend/...` → `/opt/Ascend/...`
- 架构子目录：`aarch64-linux` / `x86_64-linux` / `arm64-linux` / 空
- `acllib/lib64` → `runtime/lib64` → `lib64` → `driver/lib64`（含 stub 回退）

运行时 RPATH 已设置（`$ORIGIN` + Ascend + OpenCV + 系统库路径）。

### 7.2 MindIE LLM 构建与部署

#### 一键脚本

```bash
sudo bash mindie/install_mindie.sh
```

脚本执行流程（详见 [install_mindie.sh](file:///c:/Users/迟昕玥/Desktop/协同推理/code/mindie/install_mindie.sh)）：
1. root、NPU 检测（`/dev/davinci*`，自动枚举 ID）
2. MindIE 安装检查（`/usr/local/Ascend/mindie/latest`）
3. 扫描模型目录 `/home/HwHiAiUser/mindie/model/*/` → 用户选择模型
4. `mindie/src/` CMake 构建 → `kzzk_llm` 拷到 `/usr/local/bin/`
5. 更新 `config.json`：modelName、modelWeightPath、npuDeviceIds、worldSize、禁用 HTTPS/TLS、超时 600s
6. 停旧服务（pkill mindieservice/python，清理共享内存）
7. 设环境变量，启动 daemon（nohup `mindieservice_daemon`）
8. curl 轮询 `http://127.0.0.1:1025/v1/models` 等 300s 内就绪

#### 手动构建客户端

```bash
cd mindie/src
rm -rf build && mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 产物：
# ./kzzk_llm              命令行工具
# ./test_llm              示例程序
# ./libkzzk_llm_static.a  静态库
```

---

## 8. 运行方法与示例

### 8.1 CANN CV 命令行（C++）

```bash
# 帮助
./kzzk_cv --help
kzzk_cv --help        # 若已安装到 /usr/local/bin

# 目标检测 - YOLOv4
kzzk_cv --modelfile /home/HwHiAiUser/cann/model/yolov4.om \
        --imagefile /home/HwHiAiUser/cann/data/dog.jpg

# 图像分类 - VGG16 (cat/dog)
kzzk_cv -m /home/HwHiAiUser/cann/model/vgg16_cat_dog.om \
        -i /home/HwHiAiUser/cann/data/cat.jpg

# 图像分类 - 不在训练范围的 OOD 测试（bus.jpg）
kzzk_cv -m vgg16_cat_dog.om -i bus.jpg
# 期望输出："未识别到模型训练范围内的类别..."

# 语义分割 - DeepLabV3
kzzk_cv -m deeplabv3_plus.om -i test.jpg
# 输出：类别统计 + Segmentation image: test_seg.png
```

### 8.2 CANN CV Python API 调用

```python
import sys
sys.path.insert(0, '/home/HwHiAiUser/cann')
from python import kzzk_cv

# YOLOv4 检测
result = kzzk_cv("model/yolov4.om", "data/dog.jpg")
print("耗时:", result["infer_cost"], "ms")
for det in result["results"]:
    print(det["label"], det["confidence"], det["bbox"])

# VGG16 分类
result = kzzk_cv("model/vgg16_cat_dog.om", "data/cat.jpg")
for r in result["results"]:
    print(r["class_id"], r["confidence"], "%")

# DeepLabV3 分割 + 保存
from python.models.segmentation.deeplabv3 import DeepLabV3Segment
result = kzzk_cv("model/deeplabv3_plus.om", "data/test.jpg")
seg_map = result["results"]["seg_map"]
DeepLabV3Segment.save_seg_image("data/test.jpg", seg_map, "output_seg.png")
```

命令行：`python3 cann/python/kzzk_cv.py --modelfile ... --imagefile ...`

### 8.3 MindIE LLM C++

```bash
# 列出可用模型（需服务已启动）
kzzk_llm --listModels

# 使用模型名调用（默认连 127.0.0.1:1025）
kzzk_llm --modelfile Qwen3-14B --prompt "写一个快速排序，要求注释"

# 使用远程 URL 调用（URL 解析 host:port 为服务端，最后一段为模型名）
kzzk_llm --modelfile http://192.168.1.69:8000/Qwen3-14B \
         --prompt "介绍一下昇腾 NPU"

# 运行示例
./test_llm
```

### 8.4 MindIE LLM Python

```python
import sys
sys.path.insert(0, '/home/HwHiAiUser/mindie/src')
from kzzk_py_llm import kzzk_llm, list_models

# 列模型
models = list_models()
print("可用模型:", models)

# 推理
reply = kzzk_llm(models[0], "请写一个冒泡排序，Python 版本")
print(reply)

# 自定义参数
reply = kzzk_llm("DeepSeek-14B", "你好",
                 server_url="http://127.0.0.1:1025",
                 max_tokens=2048, temperature=0.9)
```

命令行：`python3 mindie/src/kzzk_py_llm.py --listModels / --modelfile X --prompt Y`

运行示例：`python3 mindie/examples/test_py_llm.py`

---

## 9. 设计要点与注意事项

### 9.1 双语言一致性

Python 与 C++ 实现必须保持行为一致，特别是以下易出错点：

| 问题 | 正确做法 | 常见错误 |
|------|---------|---------|
| **NV12 UV 顺序** | U 前 V 后（字节序） | 误写为 NV21（V前U后），导致颜色全反 |
| **预处理尺寸** | 严格按模型要求（VGG16: 224×256） | 自动从 OM 解析不可靠，推荐硬编码 |
| **输入类型** | 模型声明 uint8 就用 uint8 | 误传 float32 导致 size mismatch（如 YOLOv4 FP32 会有 17× 大小不匹配） |
| **Device→Host 拷贝** | 输出 buffer 必须 `aclrtMemcpy` | 直接解引用 Device 指针 → **段错误** |
| **Softmax 时机** | 模型已含 Softmax 层则不加 | 重复 Softmax 扭曲概率分布 |
| **分割图 resize** | 必须 NEAREST 邻域 | 双线性插值会产生非法类别值 |

### 9.2 日志过滤

- Python 使用 **FD 级重定向**（log_filter.py），而非 `sys.stdout` 包装（后者对 C 扩展输出无效）
- C++ 使用 **宏级开关**（`KZZK_CV_SILENT_MODE`），编译期决定
- 过滤器 `stop()` 必须在资源释放之后调用（否则释放过程中的 INFO 泄漏）

### 9.3 OOD 检测（分布外样本）

二分类/少类别模型对 OOD 输入会因 Softmax 饱和强制输出 100%/0%（而非均匀分布）。
**解决方案：**
- 置信度阈值（< 60% 拒绝）
- Top 间隙阈值（Gap > 98% 拒绝）
- 两者任一触发即过滤，返回空分类结果

### 9.4 Ascend 环境变量

运行时必需的典型环境变量（脚本已处理）：

```bash
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_TOOLKIT_PATH=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=$ASCEND_TOOLKIT_PATH/aarch64-linux/lib64:\
$ASCEND_TOOLKIT_PATH/aarch64-linux/acllib/lib64:\
$LD_LIBRARY_PATH
```

### 9.5 CMake 构建调试

- 找不到 Ascend：设置 `ASCEND_HOME` 环境变量或修改 CMakeLists.txt L17 搜索路径
- OpenCV 链接错误（未定义 jpeg/tiff 符号）：优先安装系统 OpenCV (`apt install libopencv-dev`)，其次检查系统图像库路径
- 链接时 `--unresolved-symbols=ignore-in-shared-libs`：用于兼容第三方 OpenCV 共享库内部依赖缺失

### 9.6 模型添加新类流程

1. 在对应目录下新增头文件（`inc/models/{task}/xxx.h`）继承 `BaseModel`
2. 实现源文件（`src/models/{task}/xxx.cpp`），提供 `Initialize/Infer/Finalize`
3. 在 `src/kzzk_cv.cpp` 的 `create_model()` 中添加关键字匹配分支
4. Python 端对应：继承 `base.BaseModel`，在 `python/__init__.py` 的 `_get_model_class()` 加映射

---

## 附录：关键文件索引

### CANN CV Python

| 文件 | 作用 |
|------|------|
| [python/__init__.py](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/python/__init__.py) | 对外入口、工厂、模型缓存 |
| [python/kzzk_cv.py](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/python/kzzk_cv.py) | 命令行 CLI、结果打印、文件保存 |
| [python/base.py](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/python/base.py) | BaseModel 抽象类、ACL 资源单例 |
| [python/utils/log_filter.py](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/python/utils/log_filter.py) | FD 级 [INFO] 过滤 |
| [python/models/detection/yolov4.py](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/python/models/detection/yolov4.py) | YOLOv4 检测实现 |
| [python/models/classification/vgg16.py](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/python/models/classification/vgg16.py) | VGG16 分类实现 |
| [python/models/segmentation/deeplabv3.py](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/python/models/segmentation/deeplabv3.py) | DeepLabV3 分割实现 |

### CANN CV C++

| 文件 | 作用 |
|------|------|
| [inc/kzzk_cv.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/inc/kzzk_cv.h) | 对外统一接口声明 |
| [inc/base_model.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/inc/base_model.h) | BaseModel 抽象类 |
| [inc/types.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/inc/types.h) | 所有结果数据结构 |
| [src/kzzk_cv.cpp](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/src/kzzk_cv.cpp) | 工厂实现、主流程 |
| [src/main.cpp](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/src/main.cpp) | CLI 入口、结果打印/保存 |
| [inc/models/detection/yolov4.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/inc/models/detection/yolov4.h) | YOLOv4 声明 |
| [inc/models/classification/vgg16.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/inc/models/classification/vgg16.h) | VGG16 声明（含 OOD 阈值） |
| [inc/models/segmentation/deeplabv3.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/inc/models/segmentation/deeplabv3.h) | DeepLabV3 声明 |
| [utils/coco_labels.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/utils/coco_labels.h) | COCO 80 类标签 |
| [CMakeLists.txt](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/cpp/CMakeLists.txt) | CMake 构建脚本（Ascend/OpenCV 自动探测） |

### MindIE LLM

| 文件 | 作用 |
|------|------|
| [inc/kzzk_llm.h](file:///c:/Users/迟昕玥/Desktop/协同推理/code/mindie/inc/kzzk_llm.h) | C++ LLMClient 头文件、InferenceOptions、全局函数 |
| [src/kzzk_llm.cpp](file:///c:/Users/迟昕玥/Desktop/协同推理/code/mindie/src/kzzk_llm.cpp) | C++ 实现（libcurl + nlohmann/json，PImpl） |
| [src/kzzk_py_llm.py](file:///c:/Users/迟昕玥/Desktop/协同推理/code/mindie/src/kzzk_py_llm.py) | Python 实现（requests，多端点容错） |
| [src/main.cpp](file:///c:/Users/迟昕玥/Desktop/协同推理/code/mindie/src/main.cpp) | C++ CLI 入口 |
| [src/CMakeLists.txt](file:///c:/Users/迟昕玥/Desktop/协同推理/code/mindie/src/CMakeLists.txt) | MindIE 构建脚本 |
| [install_mindie.sh](file:///c:/Users/迟昕玥/Desktop/协同推理/code/mindie/install_mindie.sh) | 部署脚本（构建 + 配置 + 启动服务） |

### 一键安装脚本

| 文件 | 作用 |
|------|------|
| [cann/install_cann.sh](file:///c:/Users/迟昕玥/Desktop/协同推理/code/cann/install_cann.sh) | CANN CV 检查环境 + 编译 + 安装到 /usr/local/bin |
| [mindie/install_mindie.sh](file:///c:/Users/迟昕玥/Desktop/协同推理/code/mindie/install_mindie.sh) | MindIE 选模型 + 编译 + 配 config + 启服务 + 就绪检查 |

---
