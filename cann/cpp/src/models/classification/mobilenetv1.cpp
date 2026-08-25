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

// 精度类型：从 .om 自动探测
static aclDataType g_mb1_input_type  = ACL_FLOAT;   // 0=FP32, 1=FP16, 2=INT8, 4=UINT8
static aclDataType g_mb1_output_type = ACL_FLOAT;

// MobileNetV1 常量（不依赖类 static const，直接字面量 + 本文件内 static const）
static const int MB1_TOP_K        = 5;
static const int MB1_NUM_CLASSES  = 1000;  // ImageNet
static const int MB1_RESIZE_SHORT = 256;   // Python resize_short=256

// ============================================================
// float16 ↔ float32 转换工具
// ============================================================
static uint16_t float_to_half(float f) {
    uint32_t i;
    memcpy(&i, &f, sizeof(i));
    uint32_t sign = (i >> 31) & 1;
    int32_t exp  = ((i >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = i & 0x007FFFFF;
    if (exp <= 0) return (uint16_t)(sign << 15);
    if (exp >= 31) { frac = frac ? 0x2000 : 0; exp = 31; }
    uint32_t round = 0x00001000 + (frac >> 1);
    int32_t bit = 0x00000800;
    while (round >= 0x00400000 && exp < 31) { round >>= 1; bit >>= 1; exp++; }
    if (round & 0x00200000 && (bit & 0x00400000)) { bit += bit; }
    frac = bit | (round >> 13);
    if (frac >= 0x40000000) { exp += 1; frac = 0; }
    return (uint16_t)((sign << 15) | (exp << 10) | (frac >> 13));
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    int32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    if (exp == 0) { exp = -14; }
    else { exp += -15 + 127; frac |= 0x400; }
    uint32_t i = (sign << 31) | (exp << 23) | (frac << 13);
    float f;
    memcpy(&f, &i, sizeof(f));
    return f;
}

} // anonymous namespace

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
// LoadModel：从 OM 自动解析输入尺寸、大小、精度类型
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

    // 精度类型探测：输入/输出数据类型
    g_mb1_input_type  = aclmdlGetInputDataType(model_desc_, 0);
    g_mb1_output_type = aclmdlGetOutputDataType(model_desc_, 0);

    // 打印精度信息（调试用）
    const char* in_type_str = "?";
    switch(g_mb1_input_type) {
        case ACL_FLOAT:   in_type_str = "FP32";  break;
        case ACL_FLOAT16: in_type_str = "FP16";  break;
        case ACL_INT8:    in_type_str = "INT8";  break;
        case ACL_UINT8:   in_type_str = "UINT8"; break;
    }
    const char* out_type_str = "?";
    switch(g_mb1_output_type) {
        case ACL_FLOAT:   out_type_str = "FP32";  break;
        case ACL_FLOAT16: out_type_str = "FP16";  break;
        case ACL_INT8:    out_type_str = "INT8";  break;
        case ACL_UINT8:   out_type_str = "UINT8"; break;
    }
    std::cerr << "[INFO][MobileNetV1] 模型精度: 输入=" << in_type_str
              << ", 输出=" << out_type_str << std::endl;

    return true;
}

void MobileNetV1::UnloadModel() {
    if (model_id_ != 0) { aclmdlUnload(model_id_); model_id_ = 0; }
    if (model_desc_) { aclmdlDestroyDesc(model_desc_); model_desc_ = nullptr; }
    g_mb1_width = 0;
    g_mb1_height = 0;
    g_mb1_size = 0;
    g_mb1_input_type  = ACL_FLOAT;
    g_mb1_output_type = ACL_FLOAT;
}

// ============================================================
// PreProcess：精度自适应预处理
//   - FP32/FP16: /255 + mean/std 归一化，输出 float
//   - INT8/UINT8: raw uint8 输出，不做归一化
// ============================================================
bool MobileNetV1::PreProcess(const std::string& image_path, std::vector<uint8_t>& output) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) return false;

    int orig_w = img.cols;
    int orig_h = img.rows;

    // ============================================================
    // Step1: 按比例 resize 短边 = MB1_RESIZE_SHORT (256)，保持长宽比！
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
    // Step2: 中心 crop 到模型要求尺寸
    // ============================================================
    int mw = g_mb1_width > 0 ? g_mb1_width : 224;
    int mh = g_mb1_height > 0 ? g_mb1_height : 224;

    int crop_x = std::max(0, (new_w - mw) / 2);
    int crop_y = std::max(0, (new_h - mh) / 2);
    crop_x = std::min(crop_x, std::max(0, new_w - mw));
    crop_y = std::min(crop_y, std::max(0, new_h - mh));
    cv::Rect roi(crop_x, crop_y,
                 std::min(mw, new_w - crop_x),
                 std::min(mh, new_h - crop_y));
    cv::Mat cropped = resized(roi).clone();

    int actual_w = cropped.cols;
    int actual_h = cropped.rows;

    // ============================================================
    // Step3: BGR → RGB
    // ============================================================
    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);

    size_t num_elements = (size_t)3 * (size_t)mh * (size_t)mw;

    if (g_mb1_input_type == ACL_INT8 || g_mb1_input_type == ACL_UINT8) {
        // ====== INT8/UINT8 模式：raw uint8，不做归一化 ======
        output.resize(num_elements);
        uint8_t* out_ptr = output.data();
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < actual_h; ++h) {
                for (int w = 0; w < actual_w; ++w) {
                    size_t idx = (size_t)c * mh * mw + (size_t)h * mw + w;
                    out_ptr[idx] = rgb.at<cv::Vec3b>(h, w)[c];
                }
            }
        }
    } else {
        // ====== FP32/FP16 模式：归一化后转为相应精度字节 ======
        rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);  // → [0,1]

        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std_inv[3] = {
            1.0f / 0.229f,
            1.0f / 0.224f,
            1.0f / 0.225f
        };

        // 先归一化到 float 数组
        std::vector<float> floats(num_elements, 0.0f);
        float* fp = floats.data();
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < actual_h; ++h) {
                for (int w = 0; w < actual_w; ++w) {
                    float val = rgb.at<cv::Vec3f>(h, w)[c];
                    val = (val - mean[c]) * std_inv[c];
                    size_t idx = (size_t)c * mh * mw + (size_t)h * mw + w;
                    fp[idx] = val;
                }
            }
        }

        if (g_mb1_input_type == ACL_FLOAT16) {
            // FP16：float → half
            output.resize(num_elements * sizeof(uint16_t));
            for (size_t i = 0; i < num_elements; ++i) {
                uint16_t h = float_to_half(fp[i]);
                output[i * 2]     = static_cast<uint8_t>(h & 0xFF);
                output[i * 2 + 1] = static_cast<uint8_t>((h >> 8) & 0xFF);
            }
        } else {
            // FP32：直接用 float 字节
            output.resize(num_elements * sizeof(float));
            if (!output.empty()) {
                std::memcpy(output.data(), floats.data(), output.size());
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
// ============================================================
void MobileNetV1::PostProcess(const float* host_out_data, size_t num_classes,
                               std::vector<ClassificationResult>& results) {
    results.clear();
    if (!host_out_data || num_classes == 0) return;

    size_t n = std::min<size_t>(num_classes, MB1_NUM_CLASSES);

    float vmin = host_out_data[0];
    float vmax = host_out_data[0];
    for (size_t i = 1; i < n; ++i) {
        float v = host_out_data[i];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    std::vector<float> probs(n);
    bool need_softmax = (vmin < -1e-6f || vmax > 1.0f + 1e-6f);

    if (need_softmax) {
        MobileNetV1_Softmax(host_out_data, static_cast<int>(n), probs.data());
    } else {
        for (size_t i = 0; i < n; ++i) {
            float p = host_out_data[i];
            if (p < 0.0f) p = 0.0f;
            if (p > 1.0f) p = 1.0f;
            probs[i] = p;
        }
    }

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

    for (int i = 0; i < top_k; ++i) {
        ClassificationResult r;
        r.class_id = scores[i].second;
        r.confidence = static_cast<int>(std::round(scores[i].first * 100.0f));
        results.push_back(r);
    }
}

// ============================================================
// Infer：主流程（精度自适应）
// ============================================================
InferenceResult MobileNetV1::Infer(const std::string& image_path) {
    InferenceResult result;
    result.model_name = GetModelName();
    result.model_type = GetModelType();
    result.infer_cost_ms = 0;

    if (!initialized_ || g_mb1_size == 0) return result;

    std::vector<uint8_t> input_data;
    if (!PreProcess(image_path, input_data)) return result;

    // 使用 .om 报告的真实输入字节大小（精度自适应）
    size_t input_size = g_mb1_size;

    void* dev_input = nullptr;
    aclError ret = aclrtMalloc(&dev_input, input_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) return result;

    ret = aclrtMemcpy(dev_input, input_size, input_data.data(), input_data.size(),
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

    // ---- Device → Host 拷贝后处理（精度自适应） ----
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

        // 根据输出精度类型，统一转为 float 后处理
        std::vector<float> out_floats;
        if (g_mb1_output_type == ACL_FLOAT16) {
            size_t num_half = dev_out_size / sizeof(uint16_t);
            out_floats.resize(num_half);
            const uint8_t* bytes = static_cast<const uint8_t*>(host_out_ptr);
            for (size_t i = 0; i < num_half; ++i) {
                uint16_t h = bytes[i * 2] | (static_cast<uint16_t>(bytes[i * 2 + 1]) << 8);
                out_floats[i] = half_to_float(h);
            }
        } else if (g_mb1_output_type == ACL_INT8 || g_mb1_output_type == ACL_UINT8) {
            // INT8/UINT8：每个元素是 1 字节，转为 float 再 softmax
            size_t num_int8 = dev_out_size;
            out_floats.resize(num_int8);
            const int8_t* int8_ptr = static_cast<const int8_t*>(host_out_ptr);
            for (size_t i = 0; i < num_int8; ++i) {
                out_floats[i] = static_cast<float>(int8_ptr[i]);
            }
        } else {
            // FP32
            size_t num_float = dev_out_size / sizeof(float);
            out_floats.resize(num_float);
            if (!out_floats.empty()) {
                std::memcpy(out_floats.data(), host_out_ptr, dev_out_size);
            }
        }

        PostProcess(out_floats.data(), out_floats.size(), result.classifications);

        aclrtFreeHost(host_out_ptr);
    }

    DestroyModelOutput();
    DestroyModelInput();
    aclrtFree(dev_input);

    return result;
}

} // namespace kzzk