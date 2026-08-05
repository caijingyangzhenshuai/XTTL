# 智能加速卡协同推理软件

[![Platform](https://img.shields.io/badge/platform-Ascend%20NPU-brightgreen)](https://www.huawei.com/ascend)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

> 面向 **华为昇腾智能加速卡（Ascend NPU）** 的协同推理框架，包含两大子系统：
> 1. **CANN CV** — 计算机视觉推理（目标检测、图像分类、语义分割），提供 **Python** 与 **C++** 双语言实现
> 2. **MindIE LLM** — 大语言模型推理客户端，封装 RESTful API

---

## 目录

- [系统架构](#系统架构)
- [CANN CV 子系统](#cann-cv-子系统)
- [MindIE LLM 子系统](#mindie-llm-子系统)
- [快速开始](#快速开始)
- [构建与安装](#构建与安装)
- [运行示例](#运行示例)
- [目录结构](#目录结构)
- [支持的模型](#支持的模型)
- [开发扩展](#开发扩展)
- [注意事项](#注意事项)
- [详细文档](#详细文档)

---

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                     用户调用层                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Python CLI  │  │  C++ CLI     │  │  Library API │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
└─────────┼─────────────────┼─────────────────┼───────────┘
          │                 │                 │
┌─────────▼─────────────────▼─────────────────▼───────────┐
│                 工厂模式模型分发层                         │
│   extract_model_name() → create_model() → 具体子类       │
└──────┬────────────┬──────────────────┬──────────────────┘
       │            │                  │
┌──────▼──────┐ ┌───▼────────┐ ┌───────▼─────────┐        │
│  目标检测    │ │  图像分类    │ │   语义分割       │        │
│ YOLOv4/3   │ │ ResNet50   │ │   DeepLabV3+    │        │
│ VGG-SSD    │ │ MobileNetV1│ │                 │        │
│ RT-DETR    │ │ VGG16      │ │                 │        │
└──────┬──────┘ └────┬───────┘ └───────┬─────────┘        │
       │             │                  │                   │
└──────▼─────────────▼──────────────────▼──────────────────┘
              ┌─────────────────────────────┐
              │   Ascend NPU (CANN ACL)     │
              └─────────────────────────────┘
```

```
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

## CANN CV 子系统

### 功能概述

基于 CANN（Compute Architecture for Neural Networks）ACL 接口，实现三类计算机视觉任务的 NPU 加速推理：

| 任务类型 | 支持模型 |
|---------|---------|
| **目标检测** | YOLOv4、YOLOv3、VGG-SSD、RT-DETR |
| **图像分类** | ResNet50、MobileNetV1、VGG16 (cat/dog) |
| **语义分割** | DeepLabV3+ |

### 接口

#### Python

```python
from python import kzzk_cv

# 推理
result = kzzk_cv("model/yolov4.om", "data/dog.jpg")
print(result)
# {
#   "model_name": "yolov4",
#   "model_type": "detection",
#   "infer_cost": 25,
#   "results": [
#     {"class_id": 1, "label": "bicycle", "confidence": 0.92, "bbox": [x1,y1,x2,y2]}
#   ]
# }
```

#### C++

```cpp
#include "kzzk_cv.h"

InferenceResult result = kzzk::kzzk_cv(model_path, image_path);
```

### 关键特性

- **工厂模式自动分发**：按 `.om` 文件名关键字自动匹配模型类型
- **双语言行为一致**：Python 与 C++ 预处理/后处理逻辑完全对齐
- **FD 级日志过滤**：Python 使用文件描述符重定向，屏蔽 CANN 原生 `[INFO]` 日志
- **OOD 检测**：VGG16 猫狗分类支持分布外样本过滤（置信度 + Top 间隙双保险）
- **NV12 精确预处理**：Python 与 C++ 的 YUV 转换、resize、字节序完全一致

---

## MindIE LLM 子系统

### 功能概述

基于 MindIE（Model Inference for Deep Neural Network）的大语言模型推理客户端，通过 HTTP REST API 调用 MindIE 服务端。

### 接口

#### Python

```python
from kzzk_py_llm import kzzk_llm, list_models

# 列出可用模型
models = list_models()
print("可用模型:", models)

# 推理
reply = kzzk_llm(models[0], "请写一个冒泡排序")
print(reply)
```

#### C++

```cpp
#include "kzzk_llm.h"

std::string reply = kzzk::kzzk_llm("Qwen3-14B", "写一个快速排序");
```

### 关键特性

- **多端点自动容错**：依次尝试 `/v1/chat/completions`、`/v1/completions`、`/api/text-generation`、`/generate`
- **响应自动适配**：兼容 OpenAI、HF、自定义等多种响应格式
- **自动重试**：每端点最多重试 3 次
- **PImpl 模式**（C++）：隐藏 libcurl 实现细节

---

## 快速开始

### 环境要求

- **硬件**：华为昇腾 NPU（Ascend 910B / 910 / 310P 等）
- **驱动**：Ascend NPU 驱动已安装（`/dev/davinci*` 存在）
- **CANN Toolkit**：已安装至 `/usr/local/Ascend/ascend-toolkit/latest`
- **架构**：推荐 aarch64（兼容 x86_64）
- **系统库**：OpenCV ≥ 4.5、CMake ≥ 3.10、GCC ≥ 5

### 一键部署（推荐）

```bash
# CANN CV
sudo bash cann/install_cann.sh

# MindIE LLM
sudo bash mindie/install_mindie.sh
```

---

## 构建与安装

### CANN CV — 手动构建

```bash
# 1. 准备 CANN 环境
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 2. 构建
cd cann/cpp
rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 3. 输出
./bin/kzzk_cv --help
```

### MindIE LLM — 手动构建

```bash
cd mindie/src
rm -rf build && mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 产物
# ./kzzk_llm              命令行工具
# ./test_llm              示例程序
# ./libkzzk_llm_static.a  静态库
```

---

## 运行示例

### CANN CV

```bash
# 目标检测 - YOLOv4
./kzzk_cv --modelfile ../model/yolov4.om --imagefile ../data/dog.jpg

# 图像分类 - VGG16 cat/dog
./kzzk_cv --modelfile ../model/vgg16_cat_dog.om --imagefile ../data/cat.jpg

# 语义分割 - DeepLabV3+
./kzzk_cv --modelfile ../model/deeplabv3_plus.om --imagefile ../data/test.jpg
```

### Python API

```python
import sys
sys.path.insert(0, '/home/HwHiAiUser/code/cann')
from python import kzzk_cv

# YOLOv4 检测
result = kzzk_cv("model/yolov4.om", "data/dog.jpg")
for det in result["results"]:
    print(det["label"], det["confidence"], det["bbox"])
```

### MindIE LLM

```bash
# 列模型
kzzk_llm --listModels

# 推理（默认连接 127.0.0.1:1025）
kzzk_llm --modelfile Qwen3-14B --prompt "介绍一下昇腾 NPU"

# 推理（指定远程 URL）
kzzk_llm --modelfile http://192.168.1.69:8000/Qwen3-14B --prompt "你好"
```

---

## 目录结构

```
code/
├── cann/                              # CANN CV 子系统
│   ├── cpp/                           # C++ 实现
│   │   ├── inc/                       # 头文件
│   │   │   ├── kzzk_cv.h              # 统一对外接口
│   │   │   ├── base_model.h           # 抽象基类
│   │   │   ├── types.h                # 数据结构定义
│   │   │   └── models/                # 各模型头文件
│   │   ├── src/                       # 源文件
│   │   ├── utils/                     # 工具（COCO/VOC 标签）
│   │   └── CMakeLists.txt             # 构建脚本
│   ├── python/                        # Python 实现
│   │   ├── __init__.py                # 包入口 & kzzk_cv 函数
│   │   ├── kzzk_cv.py                 # 命令行入口
│   │   ├── base.py                    # BaseModel 抽象基类
│   │   ├── models/                    # 各模型实现
│   │   └── utils/                     # 工具（日志过滤等）
│   ├── model/                         # .om 离线模型文件
│   ├── data/                          # 测试图像
│   ├── third_party/                   # 第三方库（OpenCV）
│   ├── examples/                      # CANN 官方样例
│   └── install_cann.sh                # 一键安装脚本
│
├── mindie/                            # MindIE LLM 子系统
│   ├── inc/
│   │   └── kzzk_llm.h                 # C++ 头文件
│   ├── src/
│   │   ├── kzzk_llm.cpp               # C++ 实现（libcurl）
│   │   ├── kzzk_py_llm.py             # Python 实现（requests）
│   │   ├── main.cpp                   # C++ 命令行入口
│   │   └── CMakeLists.txt             # 构建脚本
│   ├── examples/
│   │   ├── test_llm.cpp
│   │   └── test_py_llm.py
│   ├── third_party/
│   │   ├── cpr/                       # C++ HTTP 库（备选）
│   │   └── json/                      # nlohmann/json
│   └── install_mindie.sh              # 一键部署脚本
│
├── CodeWiki.md                        # 完整技术文档（含 API 参考）
└── .gitignore
```

---

## 支持的模型

### 计算机视觉

| 模型 | 任务 | 输入尺寸 | 输入格式 | 类别数 |
|------|------|---------|---------|--------|
| YOLOv4 | 目标检测 | 416×416 | NV12 (uint8) | 80 (COCO) |
| YOLOv3 | 目标检测 | 416×416 | NV12 (uint8) | 80 (COCO) |
| VGG-SSD | 目标检测 | 300×300 | - | 20 (VOC) |
| RT-DETR | 目标检测 | - | - | 80 (COCO) |
| ResNet50 | 图像分类 | 224×224 | RGB FP32 / uint8 | 1000 (ImageNet) |
| MobileNetV1 | 图像分类 | 224×224 | NV12 (uint8) | 1000 (ImageNet) |
| VGG16 | 图像分类 | 224×256 | NV12 (uint8) | 2 (cat/dog) |
| DeepLabV3+ | 语义分割 | 513×513 | RGB (int8) | 21 (VOC) |

### 大语言模型

由 MindIE 服务端提供，常见模型包括：`Qwen3-14B`、`DeepSeek-14B`、`ChatGLM` 等（具体以服务端部署为准）。

---

## 开发扩展

### 添加新 CV 模型

**C++：**

1. 在 `cann/cpp/inc/models/{task}/xxx.h` 声明，继承 `BaseModel`
2. 在 `cann/cpp/src/models/{task}/xxx.cpp` 实现 `Initialize`/`Infer`/`Finalize`
3. 在 `cann/cpp/src/kzzk_cv.cpp` 的 `create_model()` 中添加关键字匹配分支

**Python：**

1. 在 `cann/python/models/{task}/xxx.py` 实现，继承 `base.BaseModel`
2. 在 `cann/python/__init__.py` 的 `_get_model_class()` 中添加映射

### 添加新 LLM 端点

在 `mindie/src/kzzk_py_llm.py` 的端点列表中追加，按优先级排列即可。

---

## 注意事项

| 事项 | 说明 |
|------|------|
| **NV12 字节序** | U 在前、V 在后（偶数字节=U），**不是** NV21 |
| **输入类型** | 模型声明 uint8 就传 uint8，不要误传 float32（否则 size mismatch） |
| **Device→Host** | NPU 输出 buffer 在 Device 内存，必须 `aclrtMemcpy` 后才能 CPU 解析，否则段错误 |
| **Softmax 时机** | VGG16 等已含 Softmax 层的模型，推理后**不要**重复做 Softmax |
| **分割图 resize** | 必须使用 NEAREST 邻域插值，双线性会产生非法类别值 |
| **VGG16 尺寸** | 224(宽) × 256(高)，**非正方形**，NV12 总字节 = 86,016 |
| **日志过滤** | Python `stop()` 必须在 ACL 资源释放**之后**调用 |

---

## 详细文档

本仓库包含一份完整的 **CodeWiki.md** 技术文档，涵盖：

- 各模型的输入/输出规格与预处理细节
- 完整的 Python / C++ API 参考
- 数据结构定义
- 依赖关系与编译选项
- 构建调试指南
- 关键文件索引

请参阅：[CodeWiki.md](./CodeWiki.md)

---

## 许可

MIT License
