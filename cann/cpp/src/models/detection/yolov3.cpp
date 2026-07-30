#include "models/detection/yolov3.h"
#include "coco_labels.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace kzzk_cv {

const int   YOLOv3::kInputWidth     = 416;
const int   YOLOv3::kInputHeight    = 416;
const float YOLOv3::kConfThreshold  = 0.4f;   // 对齐 Python CONF_THRESHOLD
const int   YOLOv3::kTopK           = 10;      // 对齐 Python TOP_K

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
// 预处理：完全对齐 cann/python/models/detection/yolov3.py
//   1. PIL 读图 → RGB → resize 到 416x416（BILINEAR）
//   2. _rgb_to_nv12：手算 YUV，下采样 U/V 拼 NV12（uint8）
//   3. image_info = [416.0, 416.0, 416.0, 416.0]（4 个 float32）
// ============================================================
bool YOLOv3::PreProcess(const std::string& image_path,
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
    rgb.convertTo(rgb, CV_32FC3);  // 转 float 便于手算 YUV

    // ---- 对齐 Python _rgb_to_nv12 ----
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
            // Python 用 u[::2, ::2] —— 取偶数行偶数列的下采样
            cv::Vec3f p = rgb.at<cv::Vec3f>(y * 2, x * 2);
            float r = p[0], g = p[1], b = p[2];
            float uv = -0.169f * r - 0.331f * g + 0.5f * b + 128.0f;
            float vv = 0.5f * r - 0.419f * g - 0.081f * b + 128.0f;
            u_small.at<uint8_t>(y, x) = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, uv)));
            v_small.at<uint8_t>(y, x) = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, vv)));
        }
    }

    // 拼 NV12：[Y: W*H][UV 交错: W*H/2]，uv[h_half][w] 中偶数列放 U、奇数列放 V
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

    // image_info（对齐 Python：[MODEL_W, MODEL_H, MODEL_W, MODEL_H]）
    info_host = { static_cast<float>(kInputWidth),
                  static_cast<float>(kInputHeight),
                  static_cast<float>(kInputWidth),
                  static_cast<float>(kInputHeight) };
    return true;
}

// ============================================================
// 多输入 dataset 构造：对齐 acllite _gen_input_dataset
//   遍历每个输入，按 OM 要求的 size 校验后 add 进 dataset
// ============================================================
bool YOLOv3::CreateModelInput(const std::vector<void*>& dev_buffers,
                              const std::vector<size_t>& buffer_sizes) {
    input_ = aclmdlCreateDataset();
    if (!input_) return false;

    size_t num_inputs = aclmdlGetNumInputs(model_desc_);
    // 动态 batch 输入（ascend_mbatch_shape_data）会让 input_num 比 user input 多 1，
    // acllite 会自动 append 一个。这里按 num_inputs 为准，缺少则报错。
    if (dev_buffers.size() < num_inputs && dev_buffers.size() != num_inputs) {
        // 多数情况：user 给的 buffer 数应等于模型实际需要的数据输入数
        // 如果模型有动态 batch 输入，这里不做处理（本模型不是动态 batch）
    }

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

// ============================================================
// 后处理：完全对齐 cann/python/models/detection/yolov3.py post_process
//   output[1] = box_num（int32），output[0] = box_info（float，列布局）
//   box_info 布局：[x1_0..x1_{n-1}, y1_0..y1_{n-1}, x2_0.., y2_0.., score_0.., id_0..]
//   即 box_info[k * box_num + n]，k=0..5 对应 x1,y1,x2,y2,score,id
// ============================================================
void YOLOv3::ProcessModelOutput(const aclmdlDataset* output,
                                int orig_w, int orig_h,
                                std::vector<DetectionResult>& results) {
    results.clear();

    size_t output_num = aclmdlGetDatasetNumBuffers(output);
    if (output_num < 2) return;

    // ---- Device → Host 拷贝两个输出（杜绝段错误） ----
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

    // ---- 对齐 Python 坐标还原：scalex=orig_w/416, scaley 先同 scalex 再覆盖 ----
    float scalex = static_cast<float>(orig_w) / kInputWidth;
    float scaley = static_cast<float>(orig_h) / kInputHeight;
    if (scalex > scaley) scaley = scalex;

    if (box_num > 1000) box_num = 1000;

    for (int n = 0; n < box_num; ++n) {
        // Python: ids = box_info[5*box_num + n]; score = box_info[4*box_num + n]
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

        // 钳制到原图范围
        x1 = std::max(0.0f, std::min(static_cast<float>(orig_w), x1));
        y1 = std::max(0.0f, std::min(static_cast<float>(orig_h), y1));
        x2 = std::max(0.0f, std::min(static_cast<float>(orig_w), x2));
        y2 = std::max(0.0f, std::min(static_cast<float>(orig_h), y2));

        // 对齐 Python：过小的框跳过
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

    // 对齐 Python：按置信度降序，取 Top-K
    std::sort(results.begin(), results.end(),
              [](const DetectionResult& a, const DetectionResult& b) {
                  return a.confidence > b.confidence;
              });
    if (results.size() > static_cast<size_t>(kTopK)) results.resize(kTopK);

    aclrtFreeHost(bn_host);
    aclrtFreeHost(bi_host);
}

InferenceResult YOLOv3::Infer(const std::string& image_path) {
    InferenceResult result;
    result.model_name = GetModelName();
    result.model_type = GetModelType();
    result.infer_cost_ms = 0;
    if (!initialized_) return result;

    // ---- 预处理（host 端） ----
    std::vector<uint8_t> nv12_host;
    std::vector<float> info_host;
    int orig_w = 0, orig_h = 0;
    if (!PreProcess(image_path, nv12_host, info_host, orig_w, orig_h)) return result;

// ---- 查 OM 各输入真实 size，分配 device 内存并 H2D 拷贝 ----
	    size_t num_inputs = aclmdlGetNumInputs(model_desc_);
	    std::vector<void*> dev_bufs;
	    std::vector<size_t> dev_sizes;
	    dev_bufs.reserve(num_inputs);
	    dev_sizes.reserve(num_inputs);

	    // host 端待拷数据（按输入顺序：input[0]=NV12, input[1]=image_info）
	    // 注意：先按 OM 要求的 size 分配，host 数据拷过去时取 min(host_size, om_size)
	    struct HostView { const void* ptr; size_t size; };
	    HostView host_views[2] = {
	        { nv12_host.data(), nv12_host.size() * sizeof(uint8_t) },
	        { info_host.data(), info_host.size() * sizeof(float) }
	    };
	    const int n_user_inputs = 2;

	    bool ok = true;
	    for (size_t i = 0; i < num_inputs; ++i) {
	        size_t om_size = aclmdlGetInputSizeByIndex(model_desc_, i);
	        void* dev = nullptr;
	        if (i < static_cast<size_t>(n_user_inputs)) {
	            // 用户提供的输入：按 OM 要求分配，H2D 拷贝用户数据
	            if (aclrtMalloc(&dev, om_size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_ERROR_NONE) { ok = false; break; }
	            size_t cpy = std::min(om_size, host_views[i].size);
	            if (aclrtMemcpy(dev, om_size, host_views[i].ptr, cpy, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
	                aclrtFree(dev); ok = false; break;
	            }
	        } else {
	            // 模型额外输入（如动态 batch shape）：分配一个零缓冲区
	            size_t alloc_size = om_size > 0 ? om_size : 1;
	            if (aclrtMalloc(&dev, alloc_size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_ERROR_NONE) { ok = false; break; }
	            if (om_size > 0) {
	                // 用 0 填充
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

} // namespace kzzk_cv
