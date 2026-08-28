#include "models/detection/yolov3.h"
#include "coco_labels.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace kzzk {

const int   YOLOv3::kInputWidth     = 416;
const int   YOLOv3::kInputHeight    = 416;
const float YOLOv3::kConfThreshold  = 0.30f;  // 对齐 Python CONF_THRESHOLD
const float YOLOv3::kNmsThreshold   = 0.45f;  // 对齐 Python NMS_IOU_THRESHOLD
const int   YOLOv3::kTopK           = 10;     // 对齐 Python TOP_K
const int   YOLOv3::kNumClasses     = 80;     // COCO 80 类

YOLOv3::YOLOv3()
    : context_(nullptr), stream_(nullptr),
      model_id_(0), model_desc_(nullptr), input_(nullptr), output_(nullptr) {
    initialized_ = false;
}

YOLOv3::~YOLOv3() { Finalize(); }

bool YOLOv3::Initialize(const std::string& model_path, int device_id) {
    if (initialized_) return true;
    device_id_ = device_id;
    model_path_ = model_path;
    if (!InitAclResource()) return false;
    if (!LoadModel(model_path.c_str())) return false;
    initialized_ = true;
    return true;
}

void YOLOv3::Finalize() {
    if (!initialized_) return;
    UnloadModel();
    DestroyAclResource();
    initialized_ = false;
}

bool YOLOv3::InitAclResource() {
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

void YOLOv3::DestroyAclResource() {
    if (stream_) { aclrtDestroyStream(stream_); stream_ = nullptr; }
    if (context_) { aclrtDestroyContext(context_); context_ = nullptr; }
    aclrtResetDevice(device_id_);
    aclFinalize();
}

bool YOLOv3::LoadModel(const char* model_path) {
    aclError ret = aclmdlLoadFromFile(model_path, &model_id_);
    if (ret != ACL_ERROR_NONE) return false;
    model_desc_ = aclmdlCreateDesc();
    if (!model_desc_) return false;
    ret = aclmdlGetDesc(model_desc_, model_id_);
    if (ret != ACL_ERROR_NONE) return false;
    return true;
}

void YOLOv3::UnloadModel() {
    if (model_id_ != 0) { aclmdlUnload(model_id_); model_id_ = 0; }
    if (model_desc_) { aclmdlDestroyDesc(model_desc_); model_desc_ = nullptr; }
}

// ============================================================
// 预处理：对齐 cann/python/models/detection/yolov3.py pre_process
//   input_1: RGB → resize 416x416 → /255 → CHW float32
//   image_shape: [orig_w, orig_h] float32
// ============================================================
bool YOLOv3::PreProcess(const std::string& image_path,
                        std::vector<float>& img_host,
                        std::vector<float>& shape_host,
                        int& orig_w, int& orig_h) {
    cv::Mat bgr = cv::imread(image_path);
    if (bgr.empty()) return false;
    orig_w = bgr.cols;
    orig_h = bgr.rows;

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, cv::Size(kInputWidth, kInputHeight), 0, 0, cv::INTER_LINEAR);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);  // → [0,1]

    // HWC → CHW
    img_host.resize(static_cast<size_t>(3) * kInputWidth * kInputHeight);
    float* dst = img_host.data();
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < kInputHeight; ++h) {
            for (int w = 0; w < kInputWidth; ++w) {
                dst[c * kInputHeight * kInputWidth + h * kInputWidth + w] =
                    rgb.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    // image_shape = [orig_w, orig_h]
    shape_host = { static_cast<float>(orig_w), static_cast<float>(orig_h) };
    return true;
}

bool YOLOv3::CreateModelInput(const std::vector<void*>& dev_buffers,
                              const std::vector<size_t>& buffer_sizes) {
    input_ = aclmdlCreateDataset();
    if (!input_) return false;
    size_t num_inputs = aclmdlGetNumInputs(model_desc_);
    for (size_t i = 0; i < num_inputs; ++i) {
        if (i >= dev_buffers.size()) {
            std::cerr << "[ERROR][YOLOv3] 模型需要 " << num_inputs
                      << " 个输入，但只提供了 " << dev_buffers.size() << " 个" << std::endl;
            return false;
        }
        aclDataBuffer* db = aclCreateDataBuffer(dev_buffers[i], buffer_sizes[i]);
        if (!db) return false;
        if (aclmdlAddDatasetBuffer(input_, db) != ACL_ERROR_NONE) return false;
    }
    return true;
}

void YOLOv3::DestroyModelInput() {
    if (input_) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
            aclDestroyDataBuffer(aclmdlGetDatasetBuffer(input_, i));
        }
        aclmdlDestroyDataset(input_);
        input_ = nullptr;
    }
}

bool YOLOv3::CreateModelOutput() {
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

void YOLOv3::DestroyModelOutput() {
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

bool YOLOv3::ExecuteModel() {
    return aclmdlExecute(model_id_, input_, output_) == ACL_ERROR_NONE;
}

float YOLOv3::IOU(const BBoxRaw& b1, const BBoxRaw& b2) {
    float x1 = std::max(b1.x1, b2.x1);
    float y1 = std::max(b1.y1, b2.y1);
    float x2 = std::min(b1.x2, b2.x2);
    float y2 = std::min(b1.y2, b2.y2);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    if (inter == 0) return 0.0f;
    float area1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1);
    float area2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1);
    return inter / (area1 + area2 - inter);
}

// ============================================================
// 后处理：对齐 cann/python/models/detection/yolov3.py post_process
//   3 个输出（裁剪后的主干特征图，NCHW）：
//     output[0] = convolution_output2, float32[1,255,13,13]
//     output[1] = convolution_output1, float32[1,255,26,26]
//     output[2] = convolution_output,  float32[1,255,52,52]
//   在代码内做 YOLOv3 anchor 解码 + NMS。
// ============================================================
void YOLOv3::ProcessModelOutput(const aclmdlDataset* output,
                                int orig_w, int orig_h,
                                std::vector<DetectionResult>& results) {
    results.clear();
    size_t output_num = aclmdlGetDatasetNumBuffers(output);
    if (output_num < 3) {
        std::cerr << "[ERROR][YOLOv3] 期望 3 个输出，实际 " << output_num << " 个" << std::endl;
        return;
    }

    // ---- Device → Host 拷贝 3 个输出特征图 ----
    void* host[3] = { nullptr, nullptr, nullptr };
    size_t sizes[3] = { 0, 0, 0 };
    for (int i = 0; i < 3; ++i) {
        aclDataBuffer* buf = aclmdlGetDatasetBuffer(output, i);
        if (!buf) {
            for (int j = 0; j < i; ++j) { if (host[j]) aclrtFreeHost(host[j]); }
            return;
        }
        sizes[i] = aclGetDataBufferSizeV2(buf);
        if (aclrtMallocHost(&host[i], sizes[i]) != ACL_ERROR_NONE || !host[i]) {
            for (int j = 0; j < i; ++j) { if (host[j]) aclrtFreeHost(host[j]); }
            return;
        }
        if (aclrtMemcpy(host[i], sizes[i], aclGetDataBufferAddr(buf), sizes[i],
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
            for (int j = 0; j <= i; ++j) { if (host[j]) aclrtFreeHost(host[j]); }
            return;
        }
    }

    // 三个尺度 anchors（w,h），归一化到输入尺寸
    // 13x13（大目标）
    static const float anchors_13[3][2] = { {116.f, 90.f}, {156.f, 198.f}, {373.f, 326.f} };
    // 26x26（中目标）
    static const float anchors_26[3][2] = { {30.f, 61.f}, {62.f, 45.f}, {59.f, 119.f} };
    // 52x52（小目标）
    static const float anchors_52[3][2] = { {10.f, 13.f}, {16.f, 30.f}, {33.f, 23.f} };

    // ---- 解码三个尺度 ----
    std::vector<BBoxRaw> all;
    all.reserve(10647);  // 13^2+26^2+52^2 个网格 × 3 anchor = 10647
    DecodeScale(static_cast<const float*>(host[0]), 13, anchors_13, all);
    DecodeScale(static_cast<const float*>(host[1]), 26, anchors_26, all);
    DecodeScale(static_cast<const float*>(host[2]), 52, anchors_52, all);

    // ---- 置信度过滤已经在 DecodeScale 内完成 ----

    // ---- 按类 NMS ----
    std::vector<bool> removed(all.size(), false);
    std::vector<BBoxRaw> nms_out;
    // 先按分数降序
    std::vector<BBoxRaw> sorted = all;
    std::sort(sorted.begin(), sorted.end(),
              [](const BBoxRaw& a, const BBoxRaw& b) { return a.score > b.score; });
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (removed[i]) continue;
        nms_out.push_back(sorted[i]);
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (removed[j]) continue;
            if (sorted[i].class_index == sorted[j].class_index &&
                IOU(sorted[i], sorted[j]) > kNmsThreshold) {
                removed[j] = true;
            }
        }
    }

    // ---- 按分数降序，取 TopK ----
    std::sort(nms_out.begin(), nms_out.end(),
              [](const BBoxRaw& a, const BBoxRaw& b) { return a.score > b.score; });
    if (nms_out.size() > static_cast<size_t>(kTopK)) nms_out.resize(kTopK);

    // ---- 归一化坐标 → 原图像素 ----
    for (const auto& b : nms_out) {
        DetectionResult det;
        det.class_id = static_cast<int>(b.class_index);
        det.label = GetCocoLabel(det.class_id);
        det.confidence = b.score;
        int x1 = static_cast<int>(std::round(b.x1 * orig_w));
        int y1 = static_cast<int>(std::round(b.y1 * orig_h));
        int x2 = static_cast<int>(std::round(b.x2 * orig_w));
        int y2 = static_cast<int>(std::round(b.y2 * orig_h));
        det.bbox.x1 = std::max(0, std::min(orig_w, x1));
        det.bbox.y1 = std::max(0, std::min(orig_h, y1));
        det.bbox.x2 = std::max(0, std::min(orig_w, x2));
        det.bbox.y2 = std::max(0, std::min(orig_h, y2));
        results.push_back(det);
    }

    for (int i = 0; i < 3; ++i) { if (host[i]) aclrtFreeHost(host[i]); }
}

// ============================================================
// 解码单个尺度特征图 feat[1,255,grid,grid]（NCHW float32）
// 对齐 Python _decode_scale：
//   每个格子 3 个 anchor，每个 anchor 85 维 [x,y,w,h,obj,80cls]
//   cx = (sigmoid(tx) + grid_x) / grid
//   cy = (sigmoid(ty) + grid_y) / grid
//   w  = exp(tw) * anchor_w / input
//   h  = exp(th) * anchor_h / input
//   score = sigmoid(obj) * max(sigmoid(cls))
// ============================================================
void YOLOv3::DecodeScale(const float* feat, int grid,
                         const float anchors[3][2],
                         std::vector<BBoxRaw>& all) {
    const float kInvInputW = 1.0f / kInputWidth;
    const float kInvInputH = 1.0f / kInputHeight;

    // K&T indices: NCHW, G=grid, anchor a, offset c
    // feat 布局：[1, 255, grid, grid]，255 = 3 anchors × 85
    for (int a = 0; a < 3; ++a) {
        for (int gy = 0; gy < grid; ++gy) {
            for (int gx = 0; gx < grid; ++gx) {
                // 找到该位置起始偏移：AL = a*85, 通道索引 = a*85 + c
                const float* base = feat + (a * 85) * grid * grid + gy * grid + gx;

                float tx = base[0];
                float ty = base[1 * grid * grid];
                float tw = base[2 * grid * grid];
                float th = base[3 * grid * grid];
                float obj = base[4 * grid * grid];

                // sigmoid
                float obj_conf = 1.0f / (1.0f + std::exp(-obj));

                // 类别分数
                int best_cls = 0;
                float best_cls_score = 0.0f;
                for (int c = 0; c < kNumClasses; ++c) {
                    float cls_raw = base[(5 + c) * grid * grid];
                    float cls_s = 1.0f / (1.0f + std::exp(-cls_raw));
                    if (cls_s > best_cls_score) { best_cls_score = cls_s; best_cls = c; }
                }

                float score = obj_conf * best_cls_score;
                if (score < kConfThreshold) continue;

                // 解码
                float cx = (1.0f / (1.0f + std::exp(-tx)) + gx) / grid;
                float cy = (1.0f / (1.0f + std::exp(-ty)) + gy) / grid;
                float w = std::exp(tw) * anchors[a][0] * kInvInputW;
                float h = std::exp(th) * anchors[a][1] * kInvInputH;

                BBoxRaw b;
                b.x1 = cx - w / 2;
                b.y1 = cy - h / 2;
                b.x2 = cx + w / 2;
                b.y2 = cy + h / 2;
                // 过滤无效/越界框
                if (b.x2 <= b.x1 + 1e-6f || b.y2 <= b.y1 + 1e-6f) continue;
                b.score = score;
                b.class_index = static_cast<size_t>(best_cls);
                all.push_back(b);
            }
        }
    }
}

InferenceResult YOLOv3::Infer(const std::string& image_path) {
    InferenceResult result;
    result.model_name = GetModelName();
    result.model_type = GetModelType();
    result.infer_cost_ms = 0;
    if (!initialized_) return result;

    // ---- 预处理 ----
    std::vector<float> img_host;
    std::vector<float> shape_host;
    int orig_w = 0, orig_h = 0;
    if (!PreProcess(image_path, img_host, shape_host, orig_w, orig_h)) return result;

    // ---- 构造 2 个输入的 host 视图 ----
    struct HostView { const void* ptr; size_t size; };
    HostView host_views[2] = {
        { img_host.data(), img_host.size() * sizeof(float) },
        { shape_host.data(), shape_host.size() * sizeof(float) }
    };
    const int n_user_inputs = 2;

    size_t num_inputs = aclmdlGetNumInputs(model_desc_);
    std::vector<void*> dev_bufs;
    std::vector<size_t> dev_sizes;
    dev_bufs.reserve(num_inputs);
    dev_sizes.reserve(num_inputs);

    bool ok = true;
    for (size_t i = 0; i < num_inputs; ++i) {
        size_t om_size = aclmdlGetInputSizeByIndex(model_desc_, i);
        void* dev = nullptr;
        if (i < static_cast<size_t>(n_user_inputs)) {
            if (aclrtMalloc(&dev, om_size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_ERROR_NONE) { ok = false; break; }
            size_t cpy = std::min(om_size, host_views[i].size);
            if (aclrtMemcpy(dev, om_size, host_views[i].ptr, cpy, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
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

    ProcessModelOutput(output_, orig_w, orig_h, result.detections);

    DestroyModelOutput();
    DestroyModelInput();
    for (void* p : dev_bufs) aclrtFree(p);

    return result;
}

} // namespace kzzk