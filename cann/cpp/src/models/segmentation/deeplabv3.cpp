#include "models/segmentation/deeplabv3.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace kzzk_cv {

DeepLabV3::DeepLabV3()
    : context_(nullptr), stream_(nullptr),
      model_id_(0), model_desc_(nullptr), input_(nullptr), output_(nullptr),
      is_aipp_(false) {
    initialized_ = false;
}

DeepLabV3::~DeepLabV3() { Finalize(); }

bool DeepLabV3::Initialize(const std::string& model_path, int device_id) {
    if (initialized_) return true;
    device_id_ = device_id;
    model_path_ = model_path;
    if (!InitAclResource()) return false;
    if (!LoadModel(model_path.c_str())) return false;

    // 探测 AIPP 模式：input_size < 1M = AIPP（uint8），否则非 AIPP（float32）
    DetectAippMode();

    initialized_ = true;
    return true;
}

void DeepLabV3::Finalize() {
    if (!initialized_) return;
    UnloadModel();
    DestroyAclResource();
    initialized_ = false;
}

bool DeepLabV3::InitAclResource() {
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

void DeepLabV3::DestroyAclResource() {
    if (stream_) { aclrtDestroyStream(stream_); stream_ = nullptr; }
    if (context_) { aclrtDestroyContext(context_); context_ = nullptr; }
    aclrtResetDevice(device_id_);
    aclFinalize();
}

bool DeepLabV3::LoadModel(const char* model_path) {
    aclError ret = aclmdlLoadFromFile(model_path, &model_id_);
    if (ret != ACL_ERROR_NONE) return false;
    model_desc_ = aclmdlCreateDesc();
    if (!model_desc_) return false;
    ret = aclmdlGetDesc(model_desc_, model_id_);
    if (ret != ACL_ERROR_NONE) return false;
    return true;
}

void DeepLabV3::UnloadModel() {
    if (model_id_ != 0) { aclmdlUnload(model_id_); model_id_ = 0; }
    if (model_desc_) { aclmdlDestroyDesc(model_desc_); model_desc_ = nullptr; }
}

void DeepLabV3::DetectAippMode() {
    if (!model_desc_) { is_aipp_ = false; return; }
    size_t input_size = aclmdlGetInputSizeByIndex(model_desc_, 0);
    // AIPP 模式输入是 uint8: 513*513*3 = 789,507 bytes
    // 非 AIPP 模式输入是 float32: 513*513*3*4 = 3,158,028 bytes
    is_aipp_ = (input_size < 1000000);
}

// ============================================================
// 预处理：完全对齐 cann/python/models/segmentation/deeplabv3.py
//   PIL读图 → RGB → resize 513x513 (BILINEAR) → int8 HWC
//   注意 Python 用 np.int8（无归一化，无均值减，无 transpose！）
// ============================================================
bool DeepLabV3::PreProcess(const std::string& image_path,
                           std::vector<int8_t>& output,
                           int& orig_w, int& orig_h) {
    cv::Mat bgr = cv::imread(image_path);
    if (bgr.empty()) return false;
    orig_w = bgr.cols;
    orig_h = bgr.rows;

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, cv::Size(kInputWidth, kInputHeight), 0, 0, cv::INTER_LINEAR);

    // 对齐 Python: np.array(rgb_img).astype(np.int8) → HWC int8
    size_t total = static_cast<size_t>(kInputWidth) * kInputHeight * 3;
    output.resize(total);
    // rgb.data 是 uint8，直接 reinterpret_cast 为 int8（字节相同）
    std::memcpy(output.data(), rgb.data, total);
    return true;
}

bool DeepLabV3::CreateModelInput(const std::vector<void*>& dev_buffers,
                                 const std::vector<size_t>& buffer_sizes) {
    input_ = aclmdlCreateDataset();
    if (!input_) return false;
    size_t num_inputs = aclmdlGetNumInputs(model_desc_);
    for (size_t i = 0; i < num_inputs; ++i) {
        if (i >= dev_buffers.size()) return false;
        aclDataBuffer* db = aclCreateDataBuffer(dev_buffers[i], buffer_sizes[i]);
        if (!db) return false;
        if (aclmdlAddDatasetBuffer(input_, db) != ACL_ERROR_NONE) return false;
    }
    return true;
}

void DeepLabV3::DestroyModelInput() {
    if (input_) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
            aclDestroyDataBuffer(aclmdlGetDatasetBuffer(input_, i));
        }
        aclmdlDestroyDataset(input_);
        input_ = nullptr;
    }
}

bool DeepLabV3::CreateModelOutput() {
    output_ = aclmdlCreateDataset();
    if (!output_) return false;
    size_t output_size = aclmdlGetNumOutputs(model_desc_);
    for (size_t i = 0; i < output_size; ++i) {
        size_t buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);
        void* output_buffer = nullptr;
        if (aclrtMalloc(&output_buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_ERROR_NONE)
            return false;
        aclDataBuffer* db = aclCreateDataBuffer(output_buffer, buffer_size);
        if (!db) return false;
        if (aclmdlAddDatasetBuffer(output_, db) != ACL_ERROR_NONE) return false;
    }
    return true;
}

void DeepLabV3::DestroyModelOutput() {
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

bool DeepLabV3::ExecuteModel() {
    return aclmdlExecute(model_id_, input_, output_) == ACL_ERROR_NONE;
}

std::string DeepLabV3::GetClassName(int class_id) {
    static const char* names[] = {
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "dining table",
        "dog", "horse", "motorbike", "person", "potted plant",
        "sheep", "sofa", "train", "tv/monitor"
    };
    if (class_id >= 0 && class_id < kNumClasses) {
        return names[class_id];
    }
    return "unknown";
}

// ============================================================
// 后处理：对齐 Python post_process
//   输出格式：513x513 整数标签（每像素一个 class_id）
//   Python: reshape(513,513) → astype(uint8) → clip(0,20) → resize(NEAREST)
//   注意：必须先 Device→Host 拷贝，不能直接读 device 指针！
// ============================================================
void DeepLabV3::PostProcess(const aclmdlDataset* output, int orig_w, int orig_h,
                             SegmentationResult& result) {
    result.width = 0;
    result.height = 0;
    result.seg_map.clear();
    result.class_info.clear();

    size_t output_num = aclmdlGetDatasetNumBuffers(output);
    if (output_num < 1) return;

    aclDataBuffer* buf = aclmdlGetDatasetBuffer(output, 0);
    if (!buf) return;
    size_t buf_size = aclGetDataBufferSizeV2(buf);
    void* dev_addr = aclGetDataBufferAddr(buf);
    if (!dev_addr || buf_size == 0) return;

    // ---- Device→Host 拷贝（关键！） ----
    void* host_buf = nullptr;
    if (aclrtMallocHost(&host_buf, buf_size) != ACL_ERROR_NONE || !host_buf) return;
    if (aclrtMemcpy(host_buf, buf_size, dev_addr, buf_size,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
        aclrtFreeHost(host_buf);
        return;
    }

    size_t num_pixels = static_cast<size_t>(kInputWidth) * kInputHeight;

    // 从模型描述中查询实际输出数据类型（比 buffer size 猜测更准确）
    aclDataType output_dtype = ACL_FLOAT;
    if (model_desc_) {
        output_dtype = aclmdlGetOutputDataType(model_desc_, 0);
    }

    // 根据数据类型处理输出
    if (output_dtype == ACL_INT64) {
        // int64 标签
        if (buf_size < num_pixels * sizeof(int64_t)) {
            aclrtFreeHost(host_buf);
            return;
        }
        const int64_t* labels = static_cast<const int64_t*>(host_buf);
        for (size_t i = 0; i < num_pixels; ++i) {
            int v = static_cast<int>(labels[i]);
            seg_map_small[i] = static_cast<unsigned char>(std::max(0, std::min(v, kNumClasses - 1)));
        }
    } else if (output_dtype == ACL_INT32) {
        // int32 标签
        if (buf_size < num_pixels * sizeof(int32_t)) {
            aclrtFreeHost(host_buf);
            return;
        }
        const int32_t* data = static_cast<const int32_t*>(host_buf);
        for (size_t i = 0; i < num_pixels; ++i) {
            int v = static_cast<int>(data[i]);
            seg_map_small[i] = static_cast<unsigned char>(std::max(0, std::min(v, kNumClasses - 1)));
        }
    } else if (output_dtype == ACL_FLOAT && buf_size >= num_pixels * 21 * sizeof(float)) {
        // float32 21+ 类概率 → argmax
        size_t num_classes = buf_size / (num_pixels * sizeof(float));
        if (num_classes > 21) num_classes = 21;
        const float* probs = static_cast<const float*>(host_buf);
        for (size_t i = 0; i < num_pixels; ++i) {
            float max_val = probs[i];
            int max_class = 0;
            for (size_t c = 1; c < num_classes; ++c) {
                float val = probs[c * num_pixels + i];
                if (val > max_val) { max_val = val; max_class = static_cast<int>(c); }
            }
            seg_map_small[i] = static_cast<unsigned char>(std::max(0, std::min(max_class, kNumClasses - 1)));
        }
    } else {
        // float32 单通道（或未知类型）：每个元素是 class ID 的浮点值
        size_t n = std::min(num_pixels, buf_size / sizeof(float));
        const float* data = static_cast<const float*>(host_buf);
        for (size_t i = 0; i < n; ++i) {
            int v = static_cast<int>(std::round(data[i]));
            seg_map_small[i] = static_cast<unsigned char>(std::max(0, std::min(v, kNumClasses - 1)));
        }
    }

    // ---- resize 到原图尺寸（NEAREST） ----
    cv::Mat seg_mat(kInputHeight, kInputWidth, CV_8UC1, seg_map_small.data());
    cv::Mat seg_full;
    cv::resize(seg_mat, seg_full, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_NEAREST);

    result.width = orig_w;
    result.height = orig_h;
    result.seg_map.assign(seg_full.data, seg_full.data + static_cast<size_t>(orig_w) * orig_h);

    // ---- 统计各类像素数（对齐 Python _collect_class_stats） ----
    std::unordered_map<int, int> class_counts;
    for (int i = 0; i < orig_w * orig_h; ++i) {
        class_counts[static_cast<int>(result.seg_map[i])]++;
    }

    int total_pixels = orig_w * orig_h;
    const float kThresholdRatio = 0.01f;
    result.class_info.clear();
    for (const auto& pair : class_counts) {
        float ratio = static_cast<float>(pair.second) / total_pixels * 100.0f;
        if (ratio < kThresholdRatio) continue;
        ClassInfo info;
        info.id = pair.first;
        info.name = GetClassName(pair.first);
        info.pixels = pair.second;
        info.ratio = ratio;
        result.class_info.push_back(info);
    }

    std::sort(result.class_info.begin(), result.class_info.end(),
              [](const ClassInfo& a, const ClassInfo& b) {
                  return a.pixels > b.pixels;
              });

    aclrtFreeHost(host_buf);
}

InferenceResult DeepLabV3::Infer(const std::string& image_path) {
    InferenceResult result;
    result.model_name = GetModelName();
    result.model_type = GetModelType();
    result.infer_cost_ms = 0;
    if (!initialized_) return result;

    // ---- 预处理（对齐 Python：int8 HWC，无归一化） ----
    std::vector<int8_t> input_data;
    int orig_w = 0, orig_h = 0;
    if (!PreProcess(image_path, input_data, orig_w, orig_h)) return result;

    const void* host_ptr = static_cast<const void*>(input_data.data());
    size_t host_size = input_data.size() * sizeof(int8_t);

    // ---- 多输入处理 ----
    size_t num_inputs = aclmdlGetNumInputs(model_desc_);
    std::vector<void*> dev_bufs;
    std::vector<size_t> dev_sizes;
    dev_bufs.reserve(num_inputs);
    dev_sizes.reserve(num_inputs);

    bool ok = true;
    for (size_t i = 0; i < num_inputs; ++i) {
        size_t om_size = aclmdlGetInputSizeByIndex(model_desc_, i);
        void* dev = nullptr;
        if (i == 0) {
            // 分配并零填充整个 buffer（防止 om_size > host_size 时未初始化内存被模型读到）
            if (aclrtMalloc(&dev, om_size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_ERROR_NONE) { ok = false; break; }
            if (aclrtMemset(dev, om_size, 0, om_size) != ACL_ERROR_NONE) { aclrtFree(dev); ok = false; break; }
            size_t cpy = std::min(om_size, host_size);
            if (aclrtMemcpy(dev, om_size, host_ptr, cpy, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
                aclrtFree(dev); ok = false; break;
            }
        } else {
            size_t alloc_size = om_size > 0 ? om_size : 1;
            if (aclrtMalloc(&dev, alloc_size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_ERROR_NONE) { ok = false; break; }
            if (om_size > 0) {
                if (aclrtMemset(dev, alloc_size, 0, alloc_size) != ACL_ERROR_NONE) { aclrtFree(dev); ok = false; break; }
            }
        }
        dev_bufs.push_back(dev);
        dev_sizes.push_back(om_size > 0 ? om_size : 1);
    }
    if (!ok) {
        for (void* p : dev_bufs) aclrtFree(p);
        return result;
    }

    if (!CreateModelInput(dev_bufs, dev_sizes)) {
        for (void* p : dev_bufs) aclrtFree(p);
        return result;
    }
    if (!CreateModelOutput()) {
        DestroyModelInput();
        for (void* p : dev_bufs) aclrtFree(p);
        return result;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    if (!ExecuteModel()) {
        DestroyModelOutput(); DestroyModelInput();
        for (void* p : dev_bufs) aclrtFree(p);
        return result;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    result.infer_cost_ms = static_cast<int>(std::max<long long>(1LL,
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()));

    PostProcess(output_, orig_w, orig_h, result.segmentation);

    DestroyModelOutput();
    DestroyModelInput();
    for (void* p : dev_bufs) aclrtFree(p);

    return result;
}

} // namespace kzzk_cv