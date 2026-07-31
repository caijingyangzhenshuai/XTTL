
# 重构 mindie 目录结构（对齐 cann 标准布局）

## 当前 vs 目标结构

```
当前 mindie/                        目标 mindie/
├── 1.md                            ├── cpp/
├── 2.md                            │   ├── CMakeLists.txt
├── install_mindie.sh                │   ├── inc/
├── model/                          │   │   └── kzzk_llm.h        ← 从 inc/ 移入
├── src/                            │   └── src/
│   ├── CMakeLists.txt               │       ├── main.cpp          ← 从 src/ 移入
│   ├── kzzk_llm.cpp                 │       └── kzzk_llm.cpp      ← 从 src/ 移入
│   ├── kzzk_py_llm.py              ├── python/
│   └── main.cpp                    │   ├── __init__.py            ← 新建：re-export kzzk_llm 和 list_models
├── inc/                            │   ├── client.py              ← 从 kzzk_py_llm.py 拆出：HTTP 客户端实现
│   └── kzzk_llm.h                  │   └── kzzk_llm.py            ← 从 kzzk_py_llm.py 拆出：CLI 入口
├── examples/                        ├── examples/
│   ├── test_llm.cpp                 │   ├── test_llm.cpp          （不动）
│   └── test_py_llm.py               │   └── test_py_llm.py        （更新 import 路径）
└── third_party/                     ├── 1.md
    ├── cpr/                         ├── 2.md
    └── json/                        ├── install_mindie.sh        （更新 BUILD_DIR 路径）
                                     ├── model/
                                     └── third_party/
                                         ├── cpr/
                                         └── json/
```

## 具体改动

### 1. 新建 `cpp/` 目录，移入 C++ 文件

| 文件 | 操作 |
|------|------|
| `cpp/CMakeLists.txt` | 从 `src/CMakeLists.txt` 移入，修改 2 处路径 |
| `cpp/inc/kzzk_llm.h` | 从 `inc/kzzk_llm.h` 移入 |
| `cpp/src/main.cpp` | 从 `src/main.cpp` 移入 |
| `cpp/src/kzzk_llm.cpp` | 从 `src/kzzk_llm.cpp` 移入 |

**CMakeLists.txt 路径修改：**
- `include_directories(../inc)` → `include_directories(inc)`（inc 现在在 cpp/ 下）
- 其余路径（`../third_party/json/include`、`../examples/test_llm.cpp`）不变

### 2. 新建 `python/` 目录，拆分 Python 文件

`kzzk_py_llm.py`（248 行）当前混合了 API 实现和 CLI，拆分为 3 个文件：

| 文件 | 内容 | 对标 cann |
|------|------|-----------|
| `python/__init__.py` | re-export `kzzk_llm` 和 `list_models` | 对标 cann/python/\_\_init\_\_.py（导出 kzzk_cv 函数） |
| `python/client.py` | HTTP 客户端核心逻辑：`kzzk_llm()`、`list_models()`、`_clean_response()`、常量定义 | 对标 cann/python/base.py（核心实现） |
| `python/kzzk_llm.py` | CLI 入口：`main()`、`print_help()`、`if __name__ == "__main__"` | 对标 cann/python/kzzk_cv.py（CLI） |

**`__init__.py` 内容：**
```python
from .client import kzzk_llm, list_models

__all__ = ['kzzk_llm', 'list_models']
```

### 3. 更新 `examples/test_py_llm.py`

```python
# 旧
from src.kzzk_py_llm import kzzk_llm, list_models
# 新
from python import kzzk_llm, list_models
```

### 4. 更新 `install_mindie.sh`

```bash
# 旧
BUILD_DIR="/home/HwHiAiUser/mindie/src/build"
# 新
BUILD_DIR="/home/HwHiAiUser/mindie/cpp/build"
```

### 5. 删除旧目录

- 删除 `mindie/src/`（所有文件已移到 `cpp/`）
- 删除 `mindie/inc/`（已移到 `cpp/inc/`）

## 不改动的部分

- `examples/test_llm.cpp`（C++ example，CMakeLists.txt 的 `../examples/` 相对路径不变）
- `third_party/cpr/` 和 `third_party/json/`（不移动）
- `model/`（保持原位）
- `1.md`、`2.md`（保持原位）
- 所有源码逻辑零修改，仅调整文件位置和 import 路径
