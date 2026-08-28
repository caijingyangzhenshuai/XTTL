#include "models/detection/ssd.h"
#include "coco_labels.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

// 取消注释以下宏即可启用调试输出（打印原始 buffer 的 size 和前 N 个值）
// #define SSD_DEBUG

namespace kzzk {

const int   SSD::kInputWidth    = 1200;
const int   SSD::kInputHeight   = 1200;
const float SSD::kConfThreshold = 0.30f;   // 对齐 Python CONF_THRESHOLD
const float SSD::kNmsThreshold  = 0.45f;   // 对齐 Python NMS_IOU_THRESHOLD
const int   SSD::kTopK          = 10;       // 对齐 Python TOP_K

SSD::SSD()
    : context_(nullptr), stream_(nullptr),
      model_id_(0), model_desc_(nullptr), input_(nullptr), output_(nullptr) {
    initialized_ = false;
}

SSD::~SSD() { Finalize(); }

bool SSD::Initialize(const std::string& model_path, int device_id) {
    if (initialized_) return true;
    device_id_ = device_id;
    model_path_ = model_path;
    if (!InitAclResource()) return false;
    if (!LoadModel(model_path.c_str())) return false;
    initialized_ = true;
    return true;
}

void SSD::Finalize() {
    if (!initialized_) return;
    UnloadModel();
    DestroyAclResource();
    initialized_ = false;
}

bool SSD::InitAclResource() {
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

void SSD::DestroyAclResource() {
    if (stream_) { aclrtDestroyStream(stream_); stream_ = nullptr; }
    if (context_) { aclrtDestroyContext(context_); context_ = nullptr; }
    aclrtResetDevice(device_id_);
    aclFinalize();
}

bool SSD::LoadModel(const char* model_path) {
    aclError ret = aclmdlLoadFromFile(model_path, &model_id_);
    if (ret != ACL_ERROR_NONE) return false;
    model_desc_ = aclmdlCreateDesc();
    if (!model_desc_) return false;
    ret = aclmdlGetDesc(model_desc_, model_id_);
    if (ret != ACL_ERROR_NONE) return false;
    return true;
}

void SSD::UnloadModel() {
    if (model_id_ != 0) { aclmdlUnload(model_id_); model_id_ = 0; }
    if (model_desc_) { aclmdlDestroyDesc(model_desc_); model_desc_ = nullptr; }
}

// ============================================================
// 预处理：对齐 cann/python/models/detection/ssd.py pre_process
//   PIL 读图 → RGB → resize 1200x1200 → /255 → transpose(2,0,1) → FP32 CHW
//   注意：OpenCV 读到的是 BGR，需转 RGB 再 /255（与 Python 一致）
// ============================================================
bool SSD::PreProcess(const std::string& image_path,
                     std::vector<float>& img_host,
                     int& orig_w, int& orig_h) {
    cv::Mat bgr = cv::imread(image_path);
    if (bgr.empty()) return false;
    orig_w = bgr.cols;
    orig_h = bgr.rows;

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, cv::Size(kInputWidth, kInputHeight), 0, 0, cv::INTER_LINEAR);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);  // → [0,1]

    // HWC → CHW（对齐 np.transpose((2,0,1))）
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
    return true;
}

bool SSD::CreateModelInput(const std::vector<void*>& dev_buffers,
                           const std::vector<size_t>& buffer_sizes) {
    input_ = aclmdlCreateDataset();
    if (!input_) return false;
    size_t num_inputs = aclmdlGetNumInputs(model_desc_);
    for (size_t i = 0; i < num_inputs; ++i) {
        if (i >= dev_buffers.size()) {
            std::cerr << "[ERROR][SSD] 模型需要 " << num_inputs
                      << " 个输入，但只提供了 " << dev_buffers.size() << " 个" << std::endl;
            return false;
        }
        aclDataBuffer* db = aclCreateDataBuffer(dev_buffers[i], buffer_sizes[i]);
        if (!db) return false;
        if (aclmdlAddDatasetBuffer(input_, db) != ACL_ERROR_NONE) return false;
    }
    return true;
}

void SSD::DestroyModelInput() {
    if (input_) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
            aclDestroyDataBuffer(aclmdlGetDatasetBuffer(input_, i));
        }
        aclmdlDestroyDataset(input_);
        input_ = nullptr;
    }
}

bool SSD::CreateModelOutput() {
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

void SSD::DestroyModelOutput() {
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

bool SSD::ExecuteModel() {
    return aclmdlExecute(model_id_, input_, output_) == ACL_ERROR_NONE;
}

float SSD::IOU(const BBoxRaw& b1, const BBoxRaw& b2) {
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
// 后处理：对齐 cann/python/models/detection/ssd.py post_process
//   3 个输出（注意顺序与 RT-DETR 不同）：
//     output[0] = bboxes (float32[1,nbox,4])，坐标归一化 [0,1]
//     output[1] = labels (int64[1,nbox])，class_id = 标准COCO索引 + 1
//     output[2] = scores (float32[1,nbox])
//   过程：score>=CONF_THRESHOLD → 减1换算 COCO 索引 → NMS(按类) → TopK
//   坐标：bboxes 归一化，乘原图尺寸还原为像素
// ============================================================
void SSD::ProcessModelOutput(const aclmdlDataset* output,
                             int orig_w, int orig_h,
                             std::vector<DetectionResult>& results) {
    results.clear();
    size_t output_num = aclmdlGetDatasetNumBuffers(output);
    if (output_num < 3) {
        std::cerr << "[ERROR][SSD] 期望 3 个输出，实际 " << output_num << " 个" << std::endl;
        return;
    }

    // ---- Device → Host 拷贝 3 个输出：bboxes / labels / scores ----
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

    // 调试：打印 3 个输出 buffer 的前 5 个值（取消注释 #define SSD_DEBUG 即可启用）
#ifdef SSD_DEBUG
    {
        std::cerr << "[DEBUG][SSD] output sizes: " << sizes[0] << " " << sizes[1] << " " << sizes[2] << std::endl;
        std::cerr << "[DEBUG][SSD] nbox by sizes: float=" << (sizes[0]/sizeof(float)/4)
                  << " int64=" << (sizes[1]/sizeof(int64_t))
                  << " float=" << (sizes[2]/sizeof(float)) << std::endl;
        std::cerr << "[DEBUG][SSD] bboxes[0..4]: ";
        const float* db = static_cast<const float*>(host[0]);
        for (int k = 0; k < std::min(static_cast<int>(sizes[0]/sizeof(float)), 20); ++k)
            std::cerr << db[k] << " ";
        std::cerr << std::endl;
        std::cerr << "[DEBUG][SSD] labels(float)[0..4]: ";
        const float* dl = static_cast<const float*>(host[1]);
        for (int k = 0; k < std::min(static_cast<int>(sizes[1]/sizeof(float)), 10); ++k)
            std::cerr << dl[k] << " ";
        std::cerr << std::endl;
        std::cerr << "[DEBUG][SSD] scores[0..4]: ";
        const float* ds = static_cast<const float*>(host[2]);
        for (int k = 0; k < std::min(static_cast<int>(sizes[2]/sizeof(float)), 10); ++k)
            std::cerr << ds[k] << " ";
        std::cerr << std::endl;
    }
#endif
    // 注意：ATC 的 --output_type=FP32 会把 int64 的 labels 也转成 float32！
    // 所以 labels 实际是 float32[1, nbox]，不是 int64。
    // 读作 float，再 static_cast<int64_t>。
    const float*   bboxes = static_cast<const float*>  (host[0]);
    const float*   labels_raw = static_cast<const float*>(host[1]);
    const float*   scores = static_cast<const float*>  (host[2]);

    size_t n_scores = sizes[2] / sizeof(float);
    size_t n_labels = sizes[1] / sizeof(float);  // labels 实际是 float32！
    size_t num_boxes = std::min(n_scores, n_labels);
    // bboxes 至少要有 num_boxes*4 个 float
    if (sizes[0] / sizeof(float) < num_boxes * 4) num_boxes = std::min(num_boxes, sizes[0] / sizeof(float) / 4);

    // ---- Step1: score 过阈值，换算 COCO 索引 ----
    std::vector<BBoxRaw> all;
    all.reserve(num_boxes);
    for (size_t i = 0; i < num_boxes; ++i) {
        float s = scores[i];
        if (s < kConfThreshold) continue;
        // labels_raw 是 float32，转成 int64
        int64_t raw_label = static_cast<int64_t>(std::round(static_cast<double>(labels_raw[i])));
        // 模型 label = 标准 COCO 索引 + 1（含 background），-1 还原
        int64_t coco_idx = raw_label - 1;
        if (coco_idx < 0 || coco_idx >= 80) continue;  // COCO 80 类

        BBoxRaw b;
        b.x1 = bboxes[i * 4 + 0];
        b.y1 = bboxes[i * 4 + 1];
        b.x2 = bboxes[i * 4 + 2];
        b.y2 = bboxes[i * 4 + 3];
        // 过滤无效框（全 0 坐标或面积过小）
        if (b.x2 <= b.x1 + 1e-6f || b.y2 <= b.y1 + 1e-6f) continue;
        if (b.x1 < 0.0f || b.y1 < 0.0f || b.x2 > 1.0f || b.y2 > 1.0f) continue;
        b.score = s;
        b.class_index = static_cast<size_t>(coco_idx);
        all.push_back(b);
    }

    // ---- Step2: 按分数降序 ----
    std::sort(all.begin(), all.end(),
              [](const BBoxRaw& a, const BBoxRaw& b) { return a.score > b.score; });

    // ---- Step3: 按类 NMS（对齐 Python _nms_results：每类内做 NMS） ----
    std::vector<BBoxRaw> nms_out;
    {
        std::vector<bool> removed(all.size(), false);
        for (size_t i = 0; i < all.size(); ++i) {
            if (removed[i]) continue;
            nms_out.push_back(all[i]);
            for (size_t j = i + 1; j < all.size(); ++j) {
                if (removed[j]) continue;
                if (all[i].class_index == all[j].class_index &&
                    IOU(all[i], all[j]) > kNmsThreshold) {
                    removed[j] = true;
                }
            }
        }
    }

    // ---- Step4: 再次按分数降序，取 TopK ----
    std::sort(nms_out.begin(), nms_out.end(),
              [](const BBoxRaw& a, const BBoxRaw& b) { return a.score > b.score; });
    if (nms_out.size() > static_cast<size_t>(kTopK)) nms_out.resize(kTopK);

    // ---- 还原坐标到原图像素（bboxes 归一化 → x*orig_w / y*orig_h） ----
    for (const auto& b : nms_out) {
        DetectionResult det;
        det.class_id = static_cast<int>(b.class_index);
        det.label = GetCocoLabel(det.class_id);
        det.confidence = b.score;
        det.bbox.x1 = static_cast<int>(std::round(b.x1 * orig_w));
        det.bbox.y1 = static_cast<int>(std::round(b.y1 * orig_h));
        det.bbox.x2 = static_cast<int>(std::round(b.x2 * orig_w));
        det.bbox.y2 = static_cast<int>(std::round(b.y2 * orig_h));
        results.push_back(det);
    }

    for (int i = 0; i < 3; ++i) { if (host[i]) aclrtFreeHost(host[i]); }
}

InferenceResult SSD::Infer(const std::string& image_path) {
    InferenceResult result;
    result.model_name = GetModelName();
    result.model_type = GetModelType();
    result.infer_cost_ms = 0;
    if (!initialized_) return result;

    // ---- 预处理 ----
    std::vector<float> img_host;
    int orig_w = 0, orig_h = 0;
    if (!PreProcess(image_path, img_host, orig_w, orig_h)) return result;

    // ---- 构造 1 个输入的 host 视图 ----
    const void* host_ptr = static_cast<const void*>(img_host.data());
    size_t host_size = img_host.size() * sizeof(float);

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
            // 第一个输入：用户预处理的数据
            if (aclrtMalloc(&dev, om_size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_ERROR_NONE) { ok = false; break; }
            size_t cpy = std::min(om_size, host_size);
            if (aclrtMemcpy(dev, om_size, host_ptr, cpy, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
                aclrtFree(dev); ok = false; break;
            }
        } else {
            // 模型额外输入（如动态 batch shape）：分配零填充缓冲区
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