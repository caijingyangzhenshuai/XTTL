#include "models/detection/rt_detr.h"
#include "coco_labels.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace kzzk_cv {

const int   RT_DETR::kInputWidth    = 640;
const int   RT_DETR::kInputHeight   = 640;
const float RT_DETR::kConfThreshold = 0.30f;   // 对齐 Python CONF_THRESHOLD
const float RT_DETR::kNmsThreshold  = 0.45f;   // 对齐 Python NMS_IOU_THRESHOLD
const int   RT_DETR::kMaxPerClass   = 2;        // 对齐 Python MAX_PER_CLASS
const int   RT_DETR::kTopK          = 10;       // 对齐 Python TOP_K

RT_DETR::RT_DETR()
    : context_(nullptr), stream_(nullptr),
      model_id_(0), model_desc_(nullptr), input_(nullptr), output_(nullptr) {
    initialized_ = false;
}

RT_DETR::~RT_DETR() { Finalize(); }

bool RT_DETR::Initialize(const std::string& model_path, int device_id) {
    if (initialized_) return true;
    device_id_ = device_id;
    model_path_ = model_path;
    if (!InitAclResource()) return false;
    if (!LoadModel(model_path.c_str())) return false;
    initialized_ = true;
    return true;
}

void RT_DETR::Finalize() {
    if (!initialized_) return;
    UnloadModel();
    DestroyAclResource();
    initialized_ = false;
}

bool RT_DETR::InitAclResource() {
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

void RT_DETR::DestroyAclResource() {
    if (stream_) { aclrtDestroyStream(stream_); stream_ = nullptr; }
    if (context_) { aclrtDestroyContext(context_); context_ = nullptr; }
    aclrtResetDevice(device_id_);
    aclFinalize();
}

bool RT_DETR::LoadModel(const char* model_path) {
    aclError ret = aclmdlLoadFromFile(model_path, &model_id_);
    if (ret != ACL_ERROR_NONE) return false;
    model_desc_ = aclmdlCreateDesc();
    if (!model_desc_) return false;
    ret = aclmdlGetDesc(model_desc_, model_id_);
    if (ret != ACL_ERROR_NONE) return false;
    return true;
}

void RT_DETR::UnloadModel() {
    if (model_id_ != 0) { aclmdlUnload(model_id_); model_id_ = 0; }
    if (model_desc_) { aclmdlDestroyDesc(model_desc_); model_desc_ = nullptr; }
}

// ============================================================
// 预处理：对齐 cann/python/models/detection/rt_detr.py pre_process
//   PIL 读图 → RGB → resize 640x640 → /255 → transpose(2,0,1) → FP32 CHW
//   target_sizes = [[orig_w, orig_h]] int64
//   注意：Python 没做 ImageNet mean/std，仅 /255
// ============================================================
bool RT_DETR::PreProcess(const std::string& image_path,
                         std::vector<float>& img_host,
                         int64_t target_sizes[2],
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

    target_sizes[0] = static_cast<int64_t>(orig_w);
    target_sizes[1] = static_cast<int64_t>(orig_h);
    return true;
}

bool RT_DETR::CreateModelInput(const std::vector<void*>& dev_buffers,
                               const std::vector<size_t>& buffer_sizes) {
    input_ = aclmdlCreateDataset();
    if (!input_) return false;
    size_t num_inputs = aclmdlGetNumInputs(model_desc_);
    for (size_t i = 0; i < num_inputs; ++i) {
        if (i >= dev_buffers.size()) {
            std::cerr << "[ERROR][RT-DETR] 模型需要 " << num_inputs
                      << " 个输入，但只提供了 " << dev_buffers.size() << " 个" << std::endl;
            return false;
        }
        aclDataBuffer* db = aclCreateDataBuffer(dev_buffers[i], buffer_sizes[i]);
        if (!db) return false;
        if (aclmdlAddDatasetBuffer(input_, db) != ACL_ERROR_NONE) return false;
    }
    return true;
}

void RT_DETR::DestroyModelInput() {
    if (input_) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
            aclDestroyDataBuffer(aclmdlGetDatasetBuffer(input_, i));
        }
        aclmdlDestroyDataset(input_);
        input_ = nullptr;
    }
}

bool RT_DETR::CreateModelOutput() {
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

void RT_DETR::DestroyModelOutput() {
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

bool RT_DETR::ExecuteModel() {
    return aclmdlExecute(model_id_, input_, output_) == ACL_ERROR_NONE;
}

float RT_DETR::IOU(const BBoxRaw& b1, const BBoxRaw& b2) {
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
// 后处理：对齐 cann/python/models/detection/rt_detr.py post_process
//   3 个输出：output[0]=labels, output[1]=boxes, output[2]=scores
//   labels[i]/scores[i] 是每个 query 的最终类别/分数
//   boxes[i] = [x1,y1,x2,y2]（已是原图尺寸，因为喂了 target_sizes）
//   过程：score>=CONF_THRESHOLD → NMS(按类) → 每类最多 MAX_PER_CLASS → TopK
// ============================================================
void RT_DETR::ProcessModelOutput(const aclmdlDataset* output,
                                 std::vector<DetectionResult>& results) {
    results.clear();
    size_t output_num = aclmdlGetDatasetNumBuffers(output);
    if (output_num < 3) {
        std::cerr << "[ERROR][RT-DETR] 期望 3 个输出，实际 " << output_num << " 个" << std::endl;
        return;
    }

    // ---- Device → Host 拷贝 3 个输出 ----
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

    // Python: labels = infer_output[0][0]; boxes = infer_output[1][0]; scores = infer_output[2][0]
    // output[0] 形状 [1, N] int64（labels）；output[1] 形状 [1, N, 4] float（boxes）
    // output[2] 形状 [1, N] float（scores）。N 通常为 300（query 数）
    const int64_t* labels = static_cast<const int64_t*>(host[0]);
    const float*   boxes  = static_cast<const float*>  (host[1]);
    const float*   scores = static_cast<const float*>  (host[2]);

    // 用 labels 的元素数作为 query 数 N（最稳健，labels 与 scores 长度应一致）
    size_t n_labels = sizes[0] / sizeof(int64_t);
    size_t n_scores = sizes[2] / sizeof(float);
    size_t num_queries = std::min(n_labels, n_scores);
    // boxes 至少要有 num_queries*4 个 float
    if (sizes[1] / sizeof(float) < num_queries * 4) num_queries = std::min(num_queries, sizes[1] / sizeof(float) / 4);

    // ---- Step1: score 过阈值 ----
    std::vector<BBoxRaw> all;
    all.reserve(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        float s = scores[i];
        if (s < kConfThreshold) continue;
        int64_t lid = labels[i];
        if (lid < 0 || lid >= 80) continue;  // COCO 80 类

        BBoxRaw b;
        b.x1 = boxes[i * 4 + 0];
        b.y1 = boxes[i * 4 + 1];
        b.x2 = boxes[i * 4 + 2];
        b.y2 = boxes[i * 4 + 3];
        b.score = s;
        b.class_index = static_cast<size_t>(lid);
        all.push_back(b);
    }

    // ---- Step2: 按分数降序 ----
    std::sort(all.begin(), all.end(),
              [](const BBoxRaw& a, const BBoxRaw& b) { return a.score > b.score; });

    // ---- Step3: 按类 NMS（对齐 Python _nms_results：每类内做 NMS） ----
    std::vector<BBoxRaw> nms_out;
    {
        // 分组
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

    // ---- Step4: 每类最多保留 kMaxPerClass（对齐 Python _keep_best_per_class） ----
    {
        // 按类分组，nms_out 已是分数降序
        std::vector<std::vector<size_t>> by_class(80);
        for (size_t i = 0; i < nms_out.size(); ++i) by_class[nms_out[i].class_index].push_back(i);
        std::vector<BBoxRaw> kept;
        for (size_t c = 0; c < 80; ++c) {
            for (size_t k = 0; k < by_class[c].size() && k < static_cast<size_t>(kMaxPerClass); ++k) {
                kept.push_back(nms_out[by_class[c][k]]);
            }
        }
        nms_out.swap(kept);
    }

    // ---- Step5: 再次按分数降序，取 TopK ----
    std::sort(nms_out.begin(), nms_out.end(),
              [](const BBoxRaw& a, const BBoxRaw& b) { return a.score > b.score; });
    if (nms_out.size() > static_cast<size_t>(kTopK)) nms_out.resize(kTopK);

    for (const auto& b : nms_out) {
        DetectionResult det;
        det.class_id = static_cast<int>(b.class_index);
        det.label = GetCocoLabel(det.class_id);
        det.confidence = b.score;
        det.bbox.x1 = static_cast<int>(std::round(b.x1));
        det.bbox.y1 = static_cast<int>(std::round(b.y1));
        det.bbox.x2 = static_cast<int>(std::round(b.x2));
        det.bbox.y2 = static_cast<int>(std::round(b.y2));
        results.push_back(det);
    }

    for (int i = 0; i < 3; ++i) { if (host[i]) aclrtFreeHost(host[i]); }
}

InferenceResult RT_DETR::Infer(const std::string& image_path) {
    InferenceResult result;
    result.model_name = GetModelName();
    result.model_type = GetModelType();
    result.infer_cost_ms = 0;
    if (!initialized_) return result;

    // ---- 预处理 ----
    std::vector<float> img_host;
    int64_t target_sizes[2] = { 0, 0 };
    int orig_w = 0, orig_h = 0;
    if (!PreProcess(image_path, img_host, target_sizes, orig_w, orig_h)) return result;

    // ---- 构造 2 个输入的 host 视图 ----
    // input[0] = FP32 CHW 图像；input[1] = int64 [orig_w, orig_h]
    struct HostView { const void* ptr; size_t size; };
    HostView host_views[2] = {
        { img_host.data(), img_host.size() * sizeof(float) },
        { target_sizes, 2 * sizeof(int64_t) }
    };

size_t num_inputs = aclmdlGetNumInputs(model_desc_);
	    std::vector<void*> dev_bufs;
	    std::vector<size_t> dev_sizes;
	    dev_bufs.reserve(num_inputs);
	    dev_sizes.reserve(num_inputs);

	    const int n_user_inputs = 2;
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

    ProcessModelOutput(output_, result.detections);

    DestroyModelOutput();
    DestroyModelInput();
    for (void* p : dev_bufs) aclrtFree(p);

    return result;
}

} // namespace kzzk_cv
