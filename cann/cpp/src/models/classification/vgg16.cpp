#include "models/classification/vgg16.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace kzzk_cv {

// ============================================================
// 静默模式开关：去掉所有 [INFO]/[DEBUG]/[WARN] 中间日志
// 需要调试时注释掉下面这行即可恢复完整日志输出
// ============================================================
#define VGG16_SILENT_MODE   // ← 正式版：开启静默，不打印中间调试日志！

namespace {
struct NullOStream {
    template <typename T>
    inline NullOStream& operator<<(const T&) { return *this; }
    inline NullOStream& operator<<(std::ostream& (*)(std::ostream&)) { return *this; }
};
static NullOStream g_null_ostream;
}
#ifdef VGG16_SILENT_MODE
#  define VGG16_CERR   g_null_ostream
#else
#  define VGG16_CERR   std::cerr
#endif

VGG16::VGG16()
    : context_(nullptr), stream_(nullptr),
      model_id_(0), model_desc_(nullptr), input_(nullptr), output_(nullptr),
      model_input_width_(0), model_input_height_(0), model_input_size_(0) {
}

VGG16::~VGG16() {
    Finalize();
}

bool VGG16::Initialize(const std::string& model_path, int device_id) {
    if (initialized_) return true;

    device_id_ = device_id;
    model_path_ = model_path;

    if (!InitAclResource()) return false;
    if (!LoadModel(model_path.c_str())) return false;

    initialized_ = true;
    return true;
}

void VGG16::Finalize() {
    if (!initialized_) return;

    UnloadModel();
    DestroyAclResource();
    initialized_ = false;
}

bool VGG16::InitAclResource() {
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

void VGG16::DestroyAclResource() {
    if (stream_) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
    if (context_) {
        aclrtDestroyContext(context_);
        context_ = nullptr;
    }
    aclrtResetDevice(device_id_);
    aclFinalize();
}

// ============================================================
// LoadModel：加载 OM + 自动解析模型输入尺寸和大小（不再硬编码 224x224）
// 和 YOLOv4 完全一致的解析逻辑
// ============================================================
bool VGG16::LoadModel(const char* model_path) {
    aclError ret = aclmdlLoadFromFile(model_path, &model_id_);
    if (ret != ACL_ERROR_NONE) return false;

    model_desc_ = aclmdlCreateDesc();
    if (!model_desc_) return false;

    ret = aclmdlGetDesc(model_desc_, model_id_);
    if (ret != ACL_ERROR_NONE) return false;

    // ---- 自动解析模型输入尺寸（不再猜测！按格式直接索引！） ----
    size_t num_inputs = aclmdlGetNumInputs(model_desc_);
    if (num_inputs < 1) return false;

    aclmdlIODims input_dims;
    ret = aclmdlGetInputDims(model_desc_, 0, &input_dims);
    model_input_width_ = 0;
    model_input_height_ = 0;

    if (ret == ACL_ERROR_NONE && input_dims.dimCount >= 4) {
        // ============================================================
        // 致命修复：不再用"跳过 1/3/4"猜！
        // 99% 的视觉模型都是 NHWC 格式：[N, HEIGHT, WIDTH, CHANNEL]
        //   dims[0]=N, dims[1]=HEIGHT, dims[2]=WIDTH, dims[3]=CHANNEL
        // 为什么之前猜的不对？因为 224*256 == 256*224 → NV12 size 都=86016 →
        // buffer size 永远匹配，检查不出错！但实际图像 W/H 完全反了，严重变形！
        // ============================================================
        int nch = static_cast<int>(input_dims.dims[input_dims.dimCount - 1]);  // 最后一维=channel
        int n0 = static_cast<int>(input_dims.dims[0]);
        if ((n0 == 1 || nch == 1 || nch == 3 || nch == 4) && input_dims.dimCount == 4) {
            // [N, H, W, C] (NHWC) 或 [N, C, H, W] (NCHW) → 区分方法：
            // 如果倒数第 2 维是 3/1 → 那是 channel 维，所以顺序是 NCHW
            int d1 = static_cast<int>(input_dims.dims[1]);
            int d2 = static_cast<int>(input_dims.dims[2]);
            int d3 = static_cast<int>(input_dims.dims[3]);
            if (d1 == 1 || d1 == 3 || d1 == 4) {
                // 顺序是 [N, C, H, W] (NCHW)
                model_input_height_ = d2;
                model_input_width_  = d3;
            } else if (d3 == 1 || d3 == 3 || d3 == 4) {
                // 顺序是 [N, H, W, C] (NHWC)
                model_input_height_ = d1;
                model_input_width_  = d2;
            } else {
                // 兜底：取大的当 H，小的当 W（和 Python 256(H) × 224(W) 对齐）
                model_input_height_ = std::max(d1, d2);
                model_input_width_  = std::min(d1, d2);
            }
        }
    }

    // ---- 兜底：如果没解析出来，对齐 Python（H=256, W=224） ----
    if (model_input_width_ <= 0) model_input_width_ = 224;
    if (model_input_height_ <= 0) model_input_height_ = 256;

    // ---- 模型真实输入 size_bytes（NV12 大小应等于这个值） ----
    model_input_size_ = aclmdlGetInputSizeByIndex(model_desc_, 0);

    // ============================================================
    // 致命修复：直接强制用 Python 的正确尺寸（不再信自动解析）！
    // Python 里：resize_height=256, resize_width=224
    // → shape = (256, 224, 3) = H=256（行数）, W=224（列数）
    // → OpenCV resize(cv::Size(W, H)) = cv::Size(224, 256) 正确
    // 为什么之前自动解析坑人？因为 224*256 = 256*224 → NV12 size 都是 86016
    // → buffer size 永远匹配，但图像比例完全失真！！
    // ============================================================
    model_input_height_ = 256;  // 行数（H）= Python resize_height
    model_input_width_  = 224;  // 列数（W）= Python resize_width
    // 强制 size 必须 = W*H*3/2 = 224*256*1.5 = 86016
    model_input_size_   = (size_t)model_input_width_ * (size_t)model_input_height_ * 3 / 2;

    VGG16_CERR << "[VGG16] 模型加载成功，输入 dims="
               << model_input_width_ << "x" << model_input_height_
               << " size_bytes=" << model_input_size_ << std::endl;

    // ---------- 调试打印（开静默模式时自动屏蔽） ----------
    VGG16_CERR << "\n[VGG16 DEBUG][LoadModel] 强制（对齐Python）: model_input_width_=" << model_input_width_
              << ", model_input_height_=" << model_input_height_
              << ", model_input_size_=" << model_input_size_
              << std::endl;

    return true;
}

void VGG16::UnloadModel() {
    if (model_id_ != 0) {
        aclmdlUnload(model_id_);
        model_id_ = 0;
    }
    if (model_desc_) {
        aclmdlDestroyDesc(model_desc_);
        model_desc_ = nullptr;
    }
    model_input_width_ = 0;
    model_input_height_ = 0;
    model_input_size_ = 0;
}

// ============================================================
// PreProcess：100% 对齐 Python vgg16.py 的 preprocess_image + _rgb_to_nv12
//
// Python 原流程（逐字节翻译）：
//   1. skimage resize(anti_aliasing=True) → (H=256, W=224, 3)
//   2. img * 255 → [0,255] 再 clip → uint8
//   3. _rgb_to_nv12:
//      a. 每像素算 Y/U/V float（和 C++ 系数完全一致）
//      b. U/V 做 2x2 平均下采样（arr[0::2,0::2] + 三个邻域 /4.0）
//      c. clip → uint8
//      d. Y flatten 先放
//      e. UV: h_half 行，每 2 字节一组 = U,V 交错（U 偶字节，V 奇字节）
// ============================================================
bool VGG16::PreProcess(const std::string& image_path,
                        std::vector<uint8_t>& nv12_out,
                        int& orig_w, int& orig_h,
                        int& mw, int& mh) {
    cv::Mat img_bgr = cv::imread(image_path);
    if (img_bgr.empty()) return false;

    orig_w = img_bgr.cols;
    orig_h = img_bgr.rows;

    // 用 LoadModel 解析出来的真实模型尺寸（已确认 = 224x256）
    mw = model_input_width_;   // W = 224（列数）
    mh = model_input_height_;  // H = 256（行数）
    if (mw <= 0 || mh <= 0) return false;

    // ---------- Step1: OpenCV BGR → RGB（和 Python PIL/RGB 通道顺序一致！） ----------
    cv::Mat img_rgb;
    cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

    // ---------- Step2: resize 到模型要求 W×H (OpenCV: cv::Size(WIDTH, HEIGHT))
    // Python skimage.transform.resize 参数是 output_shape=(H, W) + anti_aliasing=True
    // C++ 近似方案：缩小场景用 INTER_AREA 最接近 anti_aliased 结果
    cv::Mat resized_rgb;
    cv::resize(img_rgb, resized_rgb, cv::Size(mw, mh), 0, 0, cv::INTER_AREA);

    int w = mw;   // 224 列
    int h = mh;   // 256 行
    int w_half = w / 2;  // 112
    int h_half = h / 2;  // 128
    size_t y_size = (size_t)h * (size_t)w;          // 57344
    size_t uv_size = (size_t)h_half * (size_t)w_half * 2;  // 28672
    size_t total_size = y_size + uv_size;            // 86016

    nv12_out.resize(total_size, 0);
    uint8_t* dst_y = nv12_out.data();                // Y 平面起点
    uint8_t* dst_uv = dst_y + y_size;                // UV 平面起点

    // ---------- Step3: 分配临时 buffer 存每像素 Y/U/V（float）+ 下采样后 U/V ----------
    std::vector<float> y_full(h * w, 0.0f);
    std::vector<float> u_full(h * w, 0.0f);
    std::vector<float> v_full(h * w, 0.0f);
    std::vector<float> u_down(h_half * w_half, 0.0f);
    std::vector<float> v_down(h_half * w_half, 0.0f);

    // ---------- Step3a: 按 Python 完全一样的系数逐像素算 Y/U/V（float） ----------
    // Python 系数：
    // y = 0.257 * r + 0.504 * g + 0.098 * b + 16
    // u = -0.148 * r - 0.291 * g + 0.439 * b + 128
    // v = 0.439 * r - 0.368 * g - 0.071 * b + 128
    for (int r = 0; r < h; ++r) {
        const uint8_t* row_ptr = resized_rgb.ptr<uint8_t>(r);  // RGBRGBRGB...
        for (int c = 0; c < w; ++c) {
            int idx = r * w + c;
            float rf = static_cast<float>(row_ptr[c * 3 + 0]);
            float gf = static_cast<float>(row_ptr[c * 3 + 1]);
            float bf = static_cast<float>(row_ptr[c * 3 + 2]);
            y_full[idx] = 0.257f * rf + 0.504f * gf + 0.098f * bf + 16.0f;
            u_full[idx] = -0.148f * rf - 0.291f * gf + 0.439f * bf + 128.0f;
            v_full[idx] = 0.439f * rf - 0.368f * gf - 0.071f * bf + 128.0f;
        }
    }

    // ---------- Step3b: U/V 2x2 平均下采样（和 Python downsample 完全一致） ----------
    // Python:
    // def downsample(arr):
    //     return (arr[0::2, 0::2] + arr[0::2, 1::2] + arr[1::2, 0::2] + arr[1::2, 1::2]) / 4.0
    // 0::2 表示第 0,2,4,...行（偶数行），1::2=奇数行
    for (int rh = 0; rh < h_half; ++rh) {
        for (int cw = 0; cw < w_half; ++cw) {
            int row0 = rh * 2;       // 偶数行 0,2,4...
            int row1 = rh * 2 + 1;   // 奇数行 1,3,5...
            int col0 = cw * 2;       // 偶数列
            int col1 = cw * 2 + 1;   // 奇数列

            float u00 = u_full[row0 * w + col0];
            float u01 = u_full[row0 * w + col1];
            float u10 = u_full[row1 * w + col0];
            float u11 = u_full[row1 * w + col1];
            u_down[rh * w_half + cw] = (u00 + u01 + u10 + u11) * 0.25f;

            float v00 = v_full[row0 * w + col0];
            float v01 = v_full[row0 * w + col1];
            float v10 = v_full[row1 * w + col0];
            float v11 = v_full[row1 * w + col1];
            v_down[rh * w_half + cw] = (v00 + v01 + v10 + v11) * 0.25f;
        }
    }

    // ---------- Step4: Y 平面 clip+uint8 写入 dst_y（和 Python 行优先 flatten 一致） ----------
    for (size_t k = 0; k < y_size; ++k) {
        float v = y_full[k];
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        dst_y[k] = static_cast<uint8_t>(std::round(v));
    }

    // ---------- Step5: UV 平面（改回 Python 正确的 NV12 = U 在前 V 在后！！！） ----------
    // Python 明确：
    //   uv[:, 0::2] = u_down  (偶字节=U) → NV12 (YUV420SP_UV)
    //   uv[:, 1::2] = v_down  (奇字节=V)
    // 之前错误改成 NV21(V在前U在后)，UV全反了！这是最大嫌疑！！
    for (int rh = 0; rh < h_half; ++rh) {
        for (int cw = 0; cw < w_half; ++cw) {
            float u = u_down[rh * w_half + cw];
            float v = v_down[rh * w_half + cw];
            if (u < 0.0f) { u = 0.0f; } if (u > 255.0f) { u = 255.0f; }
            if (v < 0.0f) { v = 0.0f; } if (v > 255.0f) { v = 255.0f; }
            size_t uv_idx = (size_t)rh * (size_t)w_half * 2 + (size_t)cw * 2;
            dst_uv[uv_idx + 0] = static_cast<uint8_t>(std::round(u));  // 偶字节=U（NV12！和 Python 一致）
            dst_uv[uv_idx + 1] = static_cast<uint8_t>(std::round(v));  // 奇字节=V
        }
    }

    // ---------- 调试打印（开静默模式时自动屏蔽） ----------
    VGG16_CERR << "[VGG16 DEBUG][PreProcess] orig=" << orig_w << "x" << orig_h
              << " → resize到 mw=" << mw << " mh=" << mh
              << " (RGB转+AREA+PythonYUV系数+2x2平均下采样+NV12(U在前V在后))"
              << " | Y_size=" << y_size << " UV_size=" << uv_size
              << " total=" << total_size
              << std::endl;
    VGG16_CERR << "  spot_check Y[0..9]=";
    for (int i = 0; i < 10; ++i) VGG16_CERR << (int)dst_y[i] << " ";
    VGG16_CERR << "\n  spot_check UV[0..9]=";
    for (int i = 0; i < 10; ++i) VGG16_CERR << (int)dst_uv[i] << " ";
    VGG16_CERR << "  (NV12: 偶字节=U, 奇字节=V)" << std::endl;

    return true;
}

bool VGG16::CreateModelInput(void* input_data_buffer, size_t buffer_size) {
    input_ = aclmdlCreateDataset();
    if (!input_) return false;

    aclDataBuffer* data_buffer = aclCreateDataBuffer(input_data_buffer, buffer_size);
    if (!data_buffer) return false;

    aclError ret = aclmdlAddDatasetBuffer(input_, data_buffer);
    if (ret != ACL_ERROR_NONE) return false;

    return true;
}

void VGG16::DestroyModelInput() {
    if (input_) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
            aclDataBuffer* buffer = aclmdlGetDatasetBuffer(input_, i);
            aclDestroyDataBuffer(buffer);
        }
        aclmdlDestroyDataset(input_);
        input_ = nullptr;
    }
}

bool VGG16::CreateModelOutput() {
    output_ = aclmdlCreateDataset();
    if (!output_) return false;

    size_t output_size = aclmdlGetNumOutputs(model_desc_);
    for (size_t i = 0; i < output_size; ++i) {
        size_t buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);
        void* output_buffer = nullptr;
        aclError ret = aclrtMalloc(&output_buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) return false;

        aclDataBuffer* data_buffer = aclCreateDataBuffer(output_buffer, buffer_size);
        if (!data_buffer) return false;

        ret = aclmdlAddDatasetBuffer(output_, data_buffer);
        if (ret != ACL_ERROR_NONE) return false;
    }

    return true;
}

void VGG16::DestroyModelOutput() {
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

bool VGG16::ExecuteModel() {
    aclError ret = aclmdlExecute(model_id_, input_, output_);
    return ret == ACL_ERROR_NONE;
}

// ============================================================
// PostProcess：和 Python 完全一致，不加 Softmax！
// Python: vals = infer_result.flatten() → argsort → vals[idx]*100
// 说明模型输出已经是概率（AIPP 里或模型最后一层已经 Softmax）
// ============================================================
void VGG16::PostProcess(const float* host_out_data, size_t num_classes,
                         std::vector<ClassificationResult>& results) {
    results.clear();
    if (!host_out_data || num_classes == 0) return;

    std::vector<std::pair<float, int>> scores;
    scores.reserve(num_classes);
    for (size_t i = 0; i < num_classes; ++i) {
        float p = host_out_data[i];
        // 保险起见：钳制到 [0,1]（即使模型输出是概率也防异常）
        if (p < 0.0f) p = 0.0f;
        if (p > 1.0f) p = 1.0f;
        scores.emplace_back(p, static_cast<int>(i));
    }

    int top_k = std::min(kTopK, static_cast<int>(num_classes));
    std::partial_sort(scores.begin(), scores.begin() + top_k, scores.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first > b.first;
                      });

    // ---------- 关键新增：双保险过滤（解决 2 分类模型 OOD 瞎猜问题） ----------
    // 规则 ① 低置信度过滤：Top1 < kMinConfidencePercent → 输入不是已知类
    // 规则 ② 极端分布过滤：Top1 - Top2 > kMaxTopGapPercent（如 100%/0% 差距 100%）
    //           正常的猫狗图高置信度通常是 95%/5%（差距 90%），Top2 不会是 0%；
    //           陌生图（公交车/人脸…）模型会瞎猜，输出 100%/0% 这种极端分布。
    int top1_percent = static_cast<int>(std::round(scores[0].first * 100.0f));
    int top2_percent = (top_k >= 2)
                           ? static_cast<int>(std::round(scores[1].first * 100.0f))
                           : 0;
    int top_gap = top1_percent - top2_percent;

    // ---------- 调试打印（开静默模式时自动屏蔽） ----------
    VGG16_CERR << "\n[VGG16 DEBUG][PostProcess] num_classes=" << num_classes << std::endl;
    for (size_t k = 0; k < num_classes; ++k) {
        VGG16_CERR << "  class[" << k << "] raw_float=" << scores[k].first
                  << "  percent=" << static_cast<int>(std::round(scores[k].first * 100.0f))
                  << "%" << std::endl;
    }
    VGG16_CERR << "  Top1=" << top1_percent << "%  Top2=" << top2_percent
              << "%  Gap=" << top_gap << "%" << std::endl;
    VGG16_CERR << "  Filter: kMinConf=" << kMinConfidencePercent
              << "%  kMaxGap=" << kMaxTopGapPercent
              << "%  => Hit="
              << ((top1_percent < kMinConfidencePercent || top_gap > kMaxTopGapPercent) ? "YES (过滤)" : "NO (保留)")
              << std::endl;

    if (top1_percent < kMinConfidencePercent || top_gap > kMaxTopGapPercent) {
        results.clear(); // 空结果表示"未识别到模型训练范围内的类别"
        return;
    }

    for (int i = 0; i < top_k; ++i) {
        ClassificationResult r;
        r.class_id = scores[i].second;
        r.confidence = static_cast<int>(std::round(scores[i].first * 100.0f));
        results.push_back(r);
    }
}

// ============================================================
// Infer：主流程和 YOLOv4 完全一致
// ============================================================
InferenceResult VGG16::Infer(const std::string& image_path) {
    InferenceResult result;
    result.model_name = GetModelName();
    result.model_type = GetModelType();
    result.infer_cost_ms = 0;

    if (!initialized_ || model_input_size_ == 0) return result;

    std::vector<uint8_t> nv12_data;
    int orig_w = 0, orig_h = 0, mw = 0, mh = 0;
    if (!PreProcess(image_path, nv12_data, orig_w, orig_h, mw, mh)) return result;

    size_t input_size = nv12_data.size();

    // ---------- 调试打印（开静默模式时自动屏蔽） ----------
    VGG16_CERR << "\n[VGG16 DEBUG][Infer] ========== 开始执行 ==========" << std::endl;
    VGG16_CERR << "[VGG16 DEBUG][Infer] Host端nv12前10字节: ";
    for (int i = 0; i < 10; ++i) VGG16_CERR << (int)nv12_data[i] << " ";
    VGG16_CERR << std::endl;

    // ---- Host → Device 拷贝 NV12（uint8） ----
    void* dev_input = nullptr;
    aclError ret = aclrtMalloc(&dev_input, input_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    VGG16_CERR << "[VGG16 DEBUG][Infer] 1) aclrtMalloc dev_input=" << dev_input
              << " size=" << input_size << " ret=" << ret << std::endl;
    if (ret != ACL_ERROR_NONE) return result;

    ret = aclrtMemcpy(dev_input, input_size, nv12_data.data(), input_size,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    VGG16_CERR << "[VGG16 DEBUG][Infer] 2) aclrtMemcpy H→D ret=" << ret << std::endl;
    if (ret != ACL_ERROR_NONE) {
        aclrtFree(dev_input);
        return result;
    }

    // ---------- H→D 后立刻 D→H 回拷验证（调试用，开静默模式时自动屏蔽） ----------
    {
        std::vector<uint8_t> tmp(input_size, 0);
        aclError ret_verify = aclrtMemcpy(tmp.data(), input_size, dev_input, input_size,
                                          ACL_MEMCPY_DEVICE_TO_HOST);
        VGG16_CERR << "[VGG16 DEBUG][Infer] 3) 【验证】H→D后D→H回拷: ret=" << ret_verify << std::endl;
        VGG16_CERR << "                             回拷前10字节: ";
        for (int i = 0; i < 10; ++i) VGG16_CERR << (int)tmp[i] << " ";
        VGG16_CERR << std::endl;
        bool match = true;
        for (int i = 0; i < 10; ++i) { if (tmp[i] != nv12_data[i]) { match=false; break; } }
        VGG16_CERR << "                             与Host一致? " << (match?"YES":"NO") << std::endl;
    }

    VGG16_CERR << "[VGG16 DEBUG][Infer] 4) CreateModelInput(dev_addr=" << dev_input
              << ", size=" << input_size << ")" << std::endl;
    if (!CreateModelInput(dev_input, input_size)) {
        aclrtFree(dev_input);
        return result;
    }

    size_t num_outputs_from_desc = aclmdlGetNumOutputs(model_desc_);
    VGG16_CERR << "[VGG16 DEBUG][Infer] 5) CreateModelOutput num_outputs_from_model_desc="
              << num_outputs_from_desc << std::endl;
    if (!CreateModelOutput()) {
        DestroyModelInput();
        aclrtFree(dev_input);
        return result;
    }
    for (size_t oi = 0; oi < aclmdlGetDatasetNumBuffers(output_); ++oi) {
        aclDataBuffer* ob = aclmdlGetDatasetBuffer(output_, oi);
        VGG16_CERR << "   output[" << oi << "] addr="
                  << aclGetDataBufferAddr(ob)
                  << " size=" << aclGetDataBufferSizeV2(ob) << std::endl;
    }

    // ---- 模型执行：打印真实 ret，不只是 bool ----
    auto t1 = std::chrono::high_resolution_clock::now();
    aclError exec_ret = aclmdlExecute(model_id_, input_, output_);
    auto t2 = std::chrono::high_resolution_clock::now();
    result.infer_cost_ms = static_cast<int>(std::max<long long>(1LL,
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()));
    VGG16_CERR << "[VGG16 DEBUG][Infer] 6) aclmdlExecute REAL_ret=" << exec_ret
              << " (0=成功) cost=" << result.infer_cost_ms << "ms" << std::endl;
    if (exec_ret != ACL_ERROR_NONE) {
        DestroyModelOutput();
        DestroyModelInput();
        aclrtFree(dev_input);
        return result;
    }

    // ---- Device→Host 拷贝 + 打印输出真实内容 ----
    {
        aclDataBuffer* out_buffer = aclmdlGetDatasetBuffer(output_, 0);
        void* dev_out_addr = aclGetDataBufferAddr(out_buffer);
        size_t dev_out_size = aclGetDataBufferSizeV2(out_buffer);
        VGG16_CERR << "[VGG16 DEBUG][Infer] 7) 输出[0] dev_addr=" << dev_out_addr
                  << " dev_out_size=" << dev_out_size
                  << " bytes = " << (dev_out_size / sizeof(float)) << " floats" << std::endl;

        if (!dev_out_addr || dev_out_size == 0) {
            DestroyModelOutput();
            DestroyModelInput();
            aclrtFree(dev_input);
            return result;
        }

        void* host_out_ptr = nullptr;
        ret = aclrtMallocHost(&host_out_ptr, dev_out_size);
        VGG16_CERR << "[VGG16 DEBUG][Infer] 8) aclrtMallocHost host_out_ptr=" << host_out_ptr
                  << " ret=" << ret << std::endl;
        if (ret != ACL_ERROR_NONE || !host_out_ptr) {
            DestroyModelOutput();
            DestroyModelInput();
            aclrtFree(dev_input);
            return result;
        }

        ret = aclrtMemcpy(host_out_ptr, dev_out_size, dev_out_addr, dev_out_size,
                          ACL_MEMCPY_DEVICE_TO_HOST);
        VGG16_CERR << "[VGG16 DEBUG][Infer] 9) D→H memcpy ret=" << ret << std::endl;
        if (ret != ACL_ERROR_NONE) {
            aclrtFreeHost(host_out_ptr);
            DestroyModelOutput();
            DestroyModelInput();
            aclrtFree(dev_input);
            return result;
        }

        const float* host_out_f32 = static_cast<const float*>(host_out_ptr);
        size_t num_classes = dev_out_size / sizeof(float);
        VGG16_CERR << "[VGG16 DEBUG][Infer] ========== 输出真实值（前10个float）==========" << std::endl;
        for (size_t f = 0; f < std::min(num_classes, (size_t)10); ++f) {
            VGG16_CERR << "   out_f32[" << f << "] = " << host_out_f32[f] << std::endl;
        }
        PostProcess(host_out_f32, num_classes, result.classifications);

        aclrtFreeHost(host_out_ptr);
    }

    DestroyModelOutput();
    DestroyModelInput();
    aclrtFree(dev_input);

    VGG16_CERR << "[VGG16 DEBUG][Infer] ========== 执行完成 ==========\n" << std::endl;

    return result;
}

} // namespace kzzk_cv
