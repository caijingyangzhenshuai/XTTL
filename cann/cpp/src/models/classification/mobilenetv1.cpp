#include "models/classification/mobilenetv1.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace kzzk {

namespace {

// ============================================================
// 【关键：不依赖头文件！用匿名 namespace 静态全局变量存模型参数】
// 这样用户服务器即使头文件是旧版（没有 model_input_width_/height_/size_ 成员），
// 只同步本 cpp 也能 100% 编过！
// ============================================================
static int   g_mb1_width  = 224;   // 兜底 Python 默认 MODEL_WIDTH=224
static int   g_mb1_height = 224;   // 兜底 Python 默认 MODEL_HEIGHT=224
static size_t g_mb1_size  = 0;     // 从 OM 实时解析（LoadModel 里赋值）

// MobileNetV1 常量（不依赖类 static const，直接字面量 + 本文件内 static const）
static const int MB1_TOP_K        = 5;
static const int MB1_NUM_CLASSES  = 1000;  // ImageNet
static const int MB1_RESIZE_SHORT = 256;   // Python resize_short=256
}

MobileNetV1::MobileNetV1()
    : context_(nullptr), stream_(nullptr),
      model_id_(0), model_desc_(nullptr), input_(nullptr), output_(nullptr) {
    // 不依赖头文件成员变量，宽高size用匿名命名空间全局变量 g_mb1_*
    initialized_ = false;
}

MobileNetV1::~MobileNetV1() { Finalize(); }

bool MobileNetV1::Initialize(const std::string& model_path, int device_id) {
    if (initialized_) return true;
    device_id_ = device_id;
    model_path_ = model_path;
    if (!InitAclResource()) return false;
    if (!LoadModel(model_path.c_str())) return false;
    initialized_ = true;
    return true;
}

void MobileNetV1::Finalize() {
    if (!initialized_) return;
    UnloadModel();
    DestroyAclResource();
    initialized_ = false;
}

bool MobileNetV1::InitAclResource() {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_ERROR_NONE) return false;
    ret = aclrtSetDevice(device_id_);
    if (ret != ACL_ERROR_NONE) return false;
    ret = aclrtCreateContext(&context_, device_id_);
    if (ret != ACL_ERROR_NONE) return false;
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_ERROR_NONE) return false;
    return true;
}

void MobileNetV1::DestroyAclResource() {
    if (stream_) { aclrtDestroyStream(stream_); stream_ = nullptr; }
    if (context_) { aclrtDestroyContext(context_); context_ = nullptr; }
    aclrtResetDevice(device_id_);
    aclFinalize();
}

// ============================================================
// LoadModel：从 OM 自动解析输入尺寸和大小（解析结果存匿名 namespace 全局静态变量，不依赖头文件）
// ============================================================
bool MobileNetV1::LoadModel(const char* model_path) {
    aclError ret = aclmdlLoadFromFile(model_path, &model_id_);
    if (ret != ACL_ERROR_NONE) return false;

    model_desc_ = aclmdlCreateDesc();
    if (!model_desc_) return false;

    ret = aclmdlGetDesc(model_desc_, model_id_);
    if (ret != ACL_ERROR_NONE) return false;

    size_t num_inputs = aclmdlGetNumInputs(model_desc_);
    if (num_inputs < 1) return false;

    aclmdlIODims input_dims;
    ret = aclmdlGetInputDims(model_desc_, 0, &input_dims);
    g_mb1_width = 0;
    g_mb1_height = 0;

    if (ret == ACL_ERROR_NONE && input_dims.dimCount >= 4) {
        for (size_t i = 1; i < input_dims.dimCount; ++i) {
            int d = static_cast<int>(input_dims.dims[i]);
            if (d == 1 || d == 3 || d == 4) continue;
            if (g_mb1_height == 0) g_mb1_height = d;
            else if (g_mb1_width == 0) g_mb1_width = d;
        }
    }

    if (g_mb1_width <= 0)  g_mb1_width = 224;
    if (g_mb1_height <= 0) g_mb1_height = 224;

    // OM 解析到的真实 size_bytes（关键！）
    g_mb1_size = aclmdlGetInputSizeByIndex(model_desc_, 0);

    return true;
}

void MobileNetV1::UnloadModel() {
    if (model_id_ != 0) { aclmdlUnload(model_id_); model_id_ = 0; }
    if (model_desc_) { aclmdlDestroyDesc(model_desc_); model_desc_ = nullptr; }
    g_mb1_width = 0;
    g_mb1_height = 0;
    g_mb1_size = 0;
}

// ============================================================
// PreProcess：完全对齐 Python MobileNetV1 默认模式（所有环境变量=0）
// ============================================================
bool MobileNetV1::PreProcess(const std::string& image_path, std::vector<float>& output) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) return false;

    int orig_w = img.cols;
    int orig_h = img.rows;

    // ============================================================
    // Step1: 按比例 resize 短边 = MB1_RESIZE_SHORT (256)，保持长宽比！
    // Python: w<h → new_w=256, new_h=h*256/w；h<=w → new_h=256, new_w=w*256/h
    // ============================================================
    int new_w, new_h;
    if (orig_w < orig_h) {
        new_w = MB1_RESIZE_SHORT;
        new_h = static_cast<int>(std::round(static_cast<double>(orig_h) * MB1_RESIZE_SHORT / orig_w));
    } else {
        new_h = MB1_RESIZE_SHORT;
        new_w = static_cast<int>(std::round(static_cast<double>(orig_w) * MB1_RESIZE_SHORT / orig_h));
    }
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // ============================================================
    // Step2: 中心 crop 到模型要求尺寸（默认 224x224）
    // Python: left=(new_w-model_w)//2, top=(new_h-model_h)//2
    //   注意：Python 的 // 是 floor division（整数向下取整），
    //         C++ 的 / 对正整数也是向零取整（=floor），和 Python 一致。
    // ============================================================
    int mw = g_mb1_width > 0 ? g_mb1_width : 224;
    int mh = g_mb1_height > 0 ? g_mb1_height : 224;

    int crop_x = std::max(0, (new_w - mw) / 2);
    int crop_y = std::max(0, (new_h - mh) / 2);
    // 防止极端情况下（如 new_w < mw）越界
    crop_x = std::min(crop_x, std::max(0, new_w - mw));
    crop_y = std::min(crop_y, std::max(0, new_h - mh));
    cv::Rect roi(crop_x, crop_y,
                 std::min(mw, new_w - crop_x),
                 std::min(mh, new_h - crop_y));
    cv::Mat cropped = resized(roi).clone();

    // ============================================================
    // Step3: BGR → RGB（PIL 默认读 RGB）
    // ============================================================
    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);

    // ============================================================
    // Step4: /255.0 → 到 [0,1] 区间，再减 MEAN / 除 STD
    // Python 默认模式：MEAN=[0.485,0.456,0.406], STD=[0.229,0.224,0.225]
    // 注意：MEAN/STD 是 [0,1] 区间的！不是 255 区间的！
    // ============================================================
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);  // → [0,1]

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_inv[3] = {
        1.0f / 0.229f,
        1.0f / 0.224f,
        1.0f / 0.225f
    };

    int actual_mw = cropped.cols;
    int actual_mh = cropped.rows;

    output.resize((size_t)3 * (size_t)mh * (size_t)mw, 0.0f);
    float* out_ptr = output.data();

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < actual_mh; ++h) {
            for (int w = 0; w < actual_mw; ++w) {
                float val = rgb.at<cv::Vec3f>(h, w)[c];  // [0,1]
                val = (val - mean[c]) * std_inv[c];
                size_t out_idx = (size_t)c * mh * mw + (size_t)h * mw + w;
                if (out_idx < output.size()) out_ptr[out_idx] = val;
            }
        }
    }

    return true;
}

bool MobileNetV1::CreateModelInput(void* input_data_buffer, size_t buffer_size) {
    input_ = aclmdlCreateDataset();
    if (!input_) return false;
    aclDataBuffer* data_buffer = aclCreateDataBuffer(input_data_buffer, buffer_size);
    if (!data_buffer) return false;
    return aclmdlAddDatasetBuffer(input_, data_buffer) == ACL_ERROR_NONE;
}

void MobileNetV1::DestroyModelInput() {
    if (input_) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
            aclDestroyDataBuffer(aclmdlGetDatasetBuffer(input_, i));
        }
        aclmdlDestroyDataset(input_);
        input_ = nullptr;
    }
}

bool MobileNetV1::CreateModelOutput() {
    output_ = aclmdlCreateDataset();
    if (!output_) return false;
    size_t output_size = aclmdlGetNumOutputs(model_desc_);
    for (size_t i = 0; i < output_size; ++i) {
        size_t buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);
        void* output_buffer = nullptr;
        if (aclrtMalloc(&output_buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_ERROR_NONE) return false;
        aclDataBuffer* data_buffer = aclCreateDataBuffer(output_buffer, buffer_size);
        if (!data_buffer) return false;
        if (aclmdlAddDatasetBuffer(output_, data_buffer) != ACL_ERROR_NONE) return false;
    }
    return true;
}

void MobileNetV1::DestroyModelOutput() {
    if (output_) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
            aclDataBuffer* buffer = aclmdlGetDatasetBuffer(output_, i);
            void* data = aclGetDataBufferAddr(buffer);
            if (data) aclrtFree(data);
            aclDestroyDataBuffer(buffer);
        }
        aclmdlDestroyDataset(output_);
        output_ = nullptr;
    }
}

bool MobileNetV1::ExecuteModel() {
    return aclmdlExecute(model_id_, input_, output_) == ACL_ERROR_NONE;
}

// ============================================================
// Softmax 工具（用于 MobileNetV1 智能判断）
// ============================================================
static void MobileNetV1_Softmax(const float* input, int n, float* output) {
    if (n <= 0) return;
    float max_val = input[0];
    for (int i = 1; i < n; ++i)
        if (input[i] > max_val) max_val = input[i];
    float sum = 0;
    for (int i = 0; i < n; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    if (sum <= 1e-12f) sum = 1e-12f;
    for (int i = 0; i < n; ++i) output[i] /= sum;
}

// ============================================================
// PostProcess：完全对齐 Python MobileNetV1
//   - 智能 Softmax：vals 范围不在 [0,1] → logits，加 Softmax；否则已是概率直接用
//   - NUM_CLASSES = 1000（默认取前 kNumClasses）
//   - 最后乘 100 取整为百分比
// ============================================================
void MobileNetV1::PostProcess(const float* host_out_data, size_t num_classes,
                               std::vector<ClassificationResult>& results) {
    results.clear();
    if (!host_out_data || num_classes == 0) return;

    // ---- Python: infer_result[0] → [0]（取 batch 0）→ flatten → vals ----
    // 我们 dev_out_size / sizeof(float) 已经是扁平化后的长度，正常情况下 == 1000
    size_t n = std::min<size_t>(num_classes, MB1_NUM_CLASSES);

    // ---- 1. 先找 min/max，用于智能判断是否需要 Softmax ----
    float vmin = host_out_data[0];
    float vmax = host_out_data[0];
    for (size_t i = 1; i < n; ++i) {
        float v = host_out_data[i];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    // ---- 2. 智能 Softmax：与 Python L116-L120 等价 ----
    //    APPLY_SOFTMAX=True and (vals.min()<0 or vals.max()>1) → 加 Softmax
    std::vector<float> probs(n);
    bool need_softmax = (vmin < -1e-6f || vmax > 1.0f + 1e-6f);

    if (need_softmax) {
        MobileNetV1_Softmax(host_out_data, static_cast<int>(n), probs.data());
    } else {
        // 已是概率，直接用，并钳制防异常
        for (size_t i = 0; i < n; ++i) {
            float p = host_out_data[i];
            if (p < 0.0f) p = 0.0f;
            if (p > 1.0f) p = 1.0f;
            probs[i] = p;
        }
    }

    // ---- 3. TopK ----
    std::vector<std::pair<float, int>> scores;
    scores.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        scores.emplace_back(probs[i], static_cast<int>(i));
    }

    int top_k = std::min(MB1_TOP_K, static_cast<int>(n));
    std::partial_sort(scores.begin(), scores.begin() + top_k, scores.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first > b.first;
                      });

    // ---- 4. 直接返回 Top-K（对齐 Python mobilenetv1.py，不做任何 OOD 过滤） ----
    for (int i = 0; i < top_k; ++i) {
        ClassificationResult r;
        r.class_id = scores[i].second;
        r.confidence = static_cast<int>(std::round(scores[i].first * 100.0f));
        results.push_back(r);
    }
}

// ============================================================
// Infer：主流程
// ============================================================
InferenceResult MobileNetV1::Infer(const std::string& image_path) {
    InferenceResult result;
    result.model_name = GetModelName();
    result.model_type = GetModelType();
    result.infer_cost_ms = 0;

    if (!initialized_ || g_mb1_size == 0) return result;

    std::vector<float> input_data;
    if (!PreProcess(image_path, input_data)) return result;

    size_t input_size = input_data.size() * sizeof(float);

    void* dev_input = nullptr;
    aclError ret = aclrtMalloc(&dev_input, input_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) return result;

    ret = aclrtMemcpy(dev_input, input_size, input_data.data(), input_size,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) { aclrtFree(dev_input); return result; }

    if (!CreateModelInput(dev_input, input_size)) { aclrtFree(dev_input); return result; }
    if (!CreateModelOutput()) { DestroyModelInput(); aclrtFree(dev_input); return result; }

    auto t1 = std::chrono::high_resolution_clock::now();
    if (!ExecuteModel()) {
        DestroyModelOutput(); DestroyModelInput(); aclrtFree(dev_input); return result;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    result.infer_cost_ms = static_cast<int>(std::max<long long>(1LL,
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()));

    // ---- Device → Host 拷贝后处理（杜绝段错误） ----
    {
        aclDataBuffer* out_buffer = aclmdlGetDatasetBuffer(output_, 0);
        void* dev_out_addr = aclGetDataBufferAddr(out_buffer);
        size_t dev_out_size = aclGetDataBufferSizeV2(out_buffer);
        if (!dev_out_addr || dev_out_size == 0) {
            DestroyModelOutput(); DestroyModelInput(); aclrtFree(dev_input); return result;
        }

        void* host_out_ptr = nullptr;
        ret = aclrtMallocHost(&host_out_ptr, dev_out_size);
        if (ret != ACL_ERROR_NONE || !host_out_ptr) {
            DestroyModelOutput(); DestroyModelInput(); aclrtFree(dev_input); return result;
        }
        ret = aclrtMemcpy(host_out_ptr, dev_out_size, dev_out_addr, dev_out_size,
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_ERROR_NONE) {
            aclrtFreeHost(host_out_ptr);
            DestroyModelOutput(); DestroyModelInput(); aclrtFree(dev_input); return result;
        }

        const float* host_out_f32 = static_cast<const float*>(host_out_ptr);
        size_t num_classes = dev_out_size / sizeof(float);
        PostProcess(host_out_f32, num_classes, result.classifications);

        aclrtFreeHost(host_out_ptr);
    }

    DestroyModelOutput();
    DestroyModelInput();
    aclrtFree(dev_input);

    return result;
}

} // namespace kzzk
