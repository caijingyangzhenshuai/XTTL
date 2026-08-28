#include "models/detection/yolov3_yuv.h"
#include "coco_labels.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace kzzk {

const int   YOLOv3YUV::kInputWidth     = 416;
const int   YOLOv3YUV::kInputHeight    = 416;
const float YOLOv3YUV::kConfThreshold  = 0.4f;  // 对齐 Python CONF_THRESHOLD
const int   YOLOv3YUV::kTopK           = 10;    // 对齐 Python TOP_K

YOLOv3YUV::YOLOv3YUV()
    : context_(nullptr), stream_(nullptr),
      model_id_(0), model_desc_(nullptr), input_(nullptr), output_(nullptr) {
    initialized_ = false;
}

YOLOv3YUV::~YOLOv3YUV() { Finalize(); }

bool YOLOv3YUV::Initialize(const std::string& model_path, int device_id) {
    if (initialized_) return true;
    device_id_ = device_id;
    model_path_ = model_path;
    if (!InitAclResource()) return false;
    if (!LoadModel(model_path.c_str())) return false;
    initialized_ = true;
    return true;
}

void YOLOv3YUV::Finalize() {
    if (!initialized_) return;
    UnloadModel();
    DestroyAclResource();
    initialized_ = false;
}

bool YOLOv3YUV::InitAclResource() {
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

void YOLOv3YUV::DestroyAclResource() {
    if (stream_) { aclrtDestroyStream(stream_); stream_ = nullptr; }
    if (context_) { aclrtDestroyContext(context_); context_ = nullptr; }
    aclrtResetDevice(device_id_);
    aclFinalize();
}

bool YOLOv3YUV::LoadModel(const char* model_path) {
    aclError ret = aclmdlLoadFromFile(model_path, &model_id_);
    if (ret != ACL_ERROR_NONE) return false;
    model_desc_ = aclmdlCreateDesc();
    if (!model_desc_) return false;
    ret = aclmdlGetDesc(model_desc_, model_id_);
    if (ret != ACL_ERROR_NONE) return false;
    return true;
}

void YOLOv3YUV::UnloadModel() {
    if (model_id_ != 0) { aclmdlUnload(model_id_); model_id_ = 0; }
    if (model_desc_) { aclmdlDestroyDesc(model_desc_); model_desc_ = nullptr; }
}

// ============================================================
// 预处理：对齐 cann/python/models/detection/yolov3_yuv.py
//   1. RGB → resize 到 416x416
//   2. _rgb_to_nv12：手算 YUV，下采样 U/V 拼 NV12（uint8）
//   3. image_info = [416.0, 416.0, 416.0, 416.0]
// ============================================================
bool YOLOv3YUV::PreProcess(const std::string& image_path,
                           std::vector<uint8_t>& nv12_host,
                           std::vector<float>& info_host,
                           int& orig_w, int& orig_h) {
    cv::Mat bgr = cv::imread(image_path);
    if (bgr.empty()) return false;
    orig_w = bgr.cols;
    orig_h = bgr.rows;

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, cv::Size(kInputWidth, kInputHeight), 0, 0, cv::INTER_LINEAR);
    rgb.convertTo(rgb, CV_32FC3);

    // Y = 0.299R + 0.587G + 0.114B
    // U = -0.169R - 0.331G + 0.5B + 128
    // V = 0.5R - 0.419G - 0.081B + 128
    const int W = kInputWidth;
    const int H = kInputHeight;
    cv::Mat y_plane(H, W, CV_8UC1);
    cv::Mat u_small(H / 2, W / 2, CV_8UC1);
    cv::Mat v_small(H / 2, W / 2, CV_8UC1);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            cv::Vec3f p = rgb.at<cv::Vec3f>(y, x);
            float r = p[0], g = p[1], b = p[2];
            float yv = 0.299f * r + 0.587f * g + 0.114f * b;
            y_plane.at<uint8_t>(y, x) = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, yv)));
        }
    }
    for (int y = 0; y < H / 2; ++y) {
        for (int x = 0; x < W / 2; ++x) {
            cv::Vec3f p = rgb.at<cv::Vec3f>(y * 2, x * 2);
            float r = p[0], g = p[1], b = p[2];
            float uu = -0.169f * r - 0.331f * g + 0.5f * b + 128.0f;
            float vv = 0.5f * r - 0.419f * g - 0.081f * b + 128.0f;
            u_small.at<uint8_t>(y, x) = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, uu)));
            v_small.at<uint8_t>(y, x) = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, vv)));
        }
    }

    // 拼 NV12：[Y: W*H][UV 交错: W*H/2]
    const size_t y_size = static_cast<size_t>(W) * H;
    nv12_host.resize(y_size * 3 / 2);
    std::memcpy(nv12_host.data(), y_plane.data, y_size);

    uint8_t* uv_dst = nv12_host.data() + y_size;
    const int h_half = H / 2;
    const int w_half = W / 2;
    for (int y = 0; y < h_half; ++y) {
        for (int x = 0; x < w_half; ++x) {
            uv_dst[y * W + x * 2 + 0] = u_small.at<uint8_t>(y, x);
            uv_dst[y * W + x * 2 + 1] = v_small.at<uint8_t>(y, x);
        }
    }

    info_host = { static_cast<float>(kInputWidth),
                  static_cast<float>(kInputHeight),
                  static_cast<float>(kInputWidth),
                  static_cast<float>(kInputHeight) };
    return true;
}

bool YOLOv3YUV::CreateModelInput(const std::vector<void*>& dev_buffers,
                                 const std::vector<size_t>& buffer_sizes) {
    input_ = aclmdlCreateDataset();
    if (!input_) return false;
    size_t num_inputs = aclmdlGetNumInputs(model_desc_);
    for (size_t i = 0; i < num_inputs; ++i) {
        if (i >= dev_buffers.size()) {
            std::cerr << "[ERROR][YOLOv3YUV] 模型需要 " << num_inputs
                      << " 个输入，但只提供了 " << dev_buffers.size() << " 个" << std::endl;
            return false;
        }
        aclDataBuffer* db = aclCreateDataBuffer(dev_buffers[i], buffer_sizes[i]);
        if (!db) return false;
        if (aclmdlAddDatasetBuffer(input_, db) != ACL_ERROR_NONE) return false;
    }
    return true;
}

void YOLOv3YUV::DestroyModelInput() {
    if (input_) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
            aclDestroyDataBuffer(aclmdlGetDatasetBuffer(input_, i));
        }
        aclmdlDestroyDataset(input_);
        input_ = nullptr;
    }
}

bool YOLOv3YUV::CreateModelOutput() {
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

void YOLOv3YUV::DestroyModelOutput() {
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

bool YOLOv3YUV::ExecuteModel() {
    return aclmdlExecute(model_id_, input_, output_) == ACL_ERROR_NONE;
}

// ============================================================
// 后处理：对齐 cann/python/models/detection/yolov3_yuv.py post_process
//   output[1] = box_num（int32），output[0] = box_info（float，列布局）
//   box_info 布局：[x1_0..x1_{n-1}, y1_0..y1_{n-1}, x2_0.., y2_0.., score_0.., id_0..]
//   即 box_info[k * box_num + n]，k=0..5 对应 x1,y1,x2,y2,score,id
// ============================================================
void YOLOv3YUV::ProcessModelOutput(const aclmdlDataset* output,
                                   int orig_w, int orig_h,
                                   std::vector<DetectionResult>& results) {
    results.clear();

    size_t output_num = aclmdlGetDatasetNumBuffers(output);
    if (output_num < 2) return;

    aclDataBuffer* bn_buf = aclmdlGetDatasetBuffer(output, 1);
    aclDataBuffer* bi_buf = aclmdlGetDatasetBuffer(output, 0);
    if (!bn_buf || !bi_buf) return;

    size_t bn_size = aclGetDataBufferSizeV2(bn_buf);
    size_t bi_size = aclGetDataBufferSizeV2(bi_buf);
    void* bn_host = nullptr;
    void* bi_host = nullptr;
    if (aclrtMallocHost(&bn_host, bn_size) != ACL_ERROR_NONE || !bn_host) return;
    if (aclrtMallocHost(&bi_host, bi_size) != ACL_ERROR_NONE || !bi_host) {
        aclrtFreeHost(bn_host); return;
    }
    if (aclrtMemcpy(bn_host, bn_size, aclGetDataBufferAddr(bn_buf), bn_size,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
        aclrtFreeHost(bn_host); aclrtFreeHost(bi_host); return;
    }
    if (aclrtMemcpy(bi_host, bi_size, aclGetDataBufferAddr(bi_buf), bi_size,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
        aclrtFreeHost(bn_host); aclrtFreeHost(bi_host); return;
    }

    int32_t* box_num_ptr = static_cast<int32_t*>(bn_host);
    float* box_info = static_cast<float*>(bi_host);
    int box_num = box_num_ptr[0];

    if (box_num <= 0) {
        aclrtFreeHost(bn_host); aclrtFreeHost(bi_host);
        return;
    }

    float scalex = static_cast<float>(orig_w) / kInputWidth;
    float scaley = static_cast<float>(orig_h) / kInputHeight;
    if (scalex > scaley) scaley = scalex;

    if (box_num > 1000) box_num = 1000;

    for (int n = 0; n < box_num; ++n) {
        size_t base_id    = static_cast<size_t>(5) * box_num + n;
        size_t base_score = static_cast<size_t>(4) * box_num + n;
        if (base_id >= bi_size / sizeof(float)) break;

        int ids = static_cast<int>(box_info[base_id]);
        float score = box_info[base_score];
        if (score < kConfThreshold) continue;

        float x1 = box_info[0 * box_num + n] * scalex;
        float y1 = box_info[1 * box_num + n] * scaley;
        float x2 = box_info[2 * box_num + n] * scalex;
        float y2 = box_info[3 * box_num + n] * scaley;

        x1 = std::max(0.0f, std::min(static_cast<float>(orig_w), x1));
        y1 = std::max(0.0f, std::min(static_cast<float>(orig_h), y1));
        x2 = std::max(0.0f, std::min(static_cast<float>(orig_w), x2));
        y2 = std::max(0.0f, std::min(static_cast<float>(orig_h), y2));

        if ((x2 - x1) < 5 || (y2 - y1) < 5) continue;

        DetectionResult det;
        det.class_id = ids;
        det.label = GetCocoLabel(ids);
        det.confidence = score;
        det.bbox.x1 = static_cast<int>(std::round(x1));
        det.bbox.y1 = static_cast<int>(std::round(y1));
        det.bbox.x2 = static_cast<int>(std::round(x2));
        det.bbox.y2 = static_cast<int>(std::round(y2));
        results.push_back(det);
    }

    std::sort(results.begin(), results.end(),
              [](const DetectionResult& a, const DetectionResult& b) {
                  return a.confidence > b.confidence;
              });
    if (results.size() > static_cast<size_t>(kTopK)) results.resize(kTopK);

    aclrtFreeHost(bn_host);
    aclrtFreeHost(bi_host);
}

InferenceResult YOLOv3YUV::Infer(const std::string& image_path) {
    InferenceResult result;
    result.model_name = GetModelName();
    result.model_type = GetModelType();
    result.infer_cost_ms = 0;
    if (!initialized_) return result;

    std::vector<uint8_t> nv12_host;
    std::vector<float> info_host;
    int orig_w = 0, orig_h = 0;
    if (!PreProcess(image_path, nv12_host, info_host, orig_w, orig_h)) return result;

    struct HostView { const void* ptr; size_t size; };
    HostView host_views[2] = {
        { nv12_host.data(), nv12_host.size() * sizeof(uint8_t) },
        { info_host.data(), info_host.size() * sizeof(float) }
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