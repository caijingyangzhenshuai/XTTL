#include "models/detection/vgg_ssd.h"
#include "voc_labels.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace kzzk {
const int   VGG_SSD::kModelWidth       = 300;
const int   VGG_SSD::kModelHeight      = 300;
const int   VGG_SSD::kAippWidth        = 304;    // 对齐 Python aipp_input_w=304
const int   VGG_SSD::kAippHeight       = 300;    // 对齐 Python aipp_input_h=300
const float VGG_SSD::kConfThreshold    = 0.05f;  // 对齐 Python CONF_THRESHOLD
const float VGG_SSD::kNmsThreshold     = 0.30f;  // 对齐 Python NMS_IOU_THRESHOLD
const int   VGG_SSD::kMaxPerClass      = 1;       // 对齐 Python MAX_PER_CLASS
const float VGG_SSD::kHighConfThreshold = 0.10f;  // 对齐 Python HIGH_CONF_THRESHOLD
const int   VGG_SSD::kTopK             = 5;       // 对齐 Python TOP_K

VGG_SSD::VGG_SSD()
    : context_(nullptr), stream_(nullptr),
      model_id_(0), model_desc_(nullptr), input_(nullptr), output_(nullptr),
      is_aipp_(false) {
    initialized_ = false;
}

VGG_SSD::~VGG_SSD() { Finalize(); }

bool VGG_SSD::Initialize(const std::string& model_path, int device_id) {
    if (initialized_) return true;
    device_id_ = device_id;
    model_path_ = model_path;
    if (!InitAclResource()) return false;
    if (!LoadModel(model_path.c_str())) return false;

    // 加载后立即探测 AIPP 模式（对齐 Python _detect_aipp_mode）
    if (!DetectAippMode(is_aipp_)) {
        // 探测失败按非 AIPP 处理（保守）
        is_aipp_ = false;
    }
    initialized_ = true;
    return true;
}

void VGG_SSD::Finalize() {
    if (!initialized_) return;
    UnloadModel();
    DestroyAclResource();
    initialized_ = false;
}

bool VGG_SSD::InitAclResource() {
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

void VGG_SSD::DestroyAclResource() {
    if (stream_) { aclrtDestroyStream(stream_); stream_ = nullptr; }
    if (context_) { aclrtDestroyContext(context_); context_ = nullptr; }
    aclrtResetDevice(device_id_);
    aclFinalize();
}

bool VGG_SSD::LoadModel(const char* model_path) {
    aclError ret = aclmdlLoadFromFile(model_path, &model_id_);
    if (ret != ACL_ERROR_NONE) return false;
    model_desc_ = aclmdlCreateDesc();
    if (!model_desc_) return false;
    ret = aclmdlGetDesc(model_desc_, model_id_);
    if (ret != ACL_ERROR_NONE) return false;
    return true;
}

void VGG_SSD::UnloadModel() {
    if (model_id_ != 0) { aclmdlUnload(model_id_); model_id_ = 0; }
    if (model_desc_) { aclmdlDestroyDesc(model_desc_); model_desc_ = nullptr; }
}

// 对齐 Python _detect_aipp_mode：
//   input_size < 1000000 → AIPP（因为 AIPP 模式输入是 uint8 ~27万字节）
//   否则非 AIPP（float32 ~108万字节）
bool VGG_SSD::DetectAippMode(bool& is_aipp) {
    if (!model_desc_) return false;
    size_t input_size = aclmdlGetInputSizeByIndex(model_desc_, 0);
    if (input_size < 1000000) {
        is_aipp = true;
    } else {
        is_aipp = false;
    }
    return true;
}

// ============================================================
// 预处理：对齐 cann/python/models/detection/vgg_ssd.py pre_process
//   AIPP: resize (304,300) → uint8 → transpose(2,0,1) NCHW
//   非AIPP: resize (300,300) → RGB → 减 mean[104,117,123] → transpose(2,0,1) NCHW float32
//   注意 Python：img_array = img_array[:, :, ::-1]（RGB→BGR）再减 mean，
//   mean=[104,117,123]。即对 BGR 通道顺序减 BGR 的 mean。
//   PIL 读出来是 RGB，[::-1] 后变 BGR，mean 也是按 [B,G,R]=[104,117,123]。
//   OpenCV 读出来本就是 BGR，所以直接减 [104,117,123]，无需转 RGB。
// ============================================================
bool VGG_SSD::PreProcess(const std::string& image_path,
                         std::vector<uint8_t>& u8_host,
                         std::vector<float>& fp32_host,
                         bool& use_u8,
                         int& orig_w, int& orig_h) {
    cv::Mat bgr = cv::imread(image_path);
    if (bgr.empty()) return false;
    orig_w = bgr.cols;
    orig_h = bgr.rows;

    if (is_aipp_) {
        // ---- AIPP 模式：304x300 uint8 NCHW（无归一化） ----
        use_u8 = true;
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(kAippWidth, kAippHeight), 0, 0, cv::INTER_LINEAR);

        // HWC uint8 → CHW uint8（对齐 np.transpose((2,0,1))）
        const int W = kAippWidth, H = kAippHeight;
        u8_host.resize(static_cast<size_t>(3) * W * H);
        uint8_t* dst = u8_host.data();
        for (int c = 0; c < 3; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    dst[c * H * W + h * W + w] = resized.at<cv::Vec3b>(h, w)[c];
        fp32_host.clear();
    } else {
        // ---- 非 AIPP 模式：300x300 → 减 mean → NCHW float32 ----
        use_u8 = false;
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(kModelWidth, kModelHeight), 0, 0, cv::INTER_LINEAR);

        // OpenCV 是 BGR，Python 对 PIL(RGB) 做 [:, :, ::-1] 得到 BGR，再减 mean[104,117,123]
        // 所以这里 BGR 直接减 [104,117,123]
        resized.convertTo(resized, CV_32FC3);
        const float mean[3] = { 104.0f, 117.0f, 123.0f };

        const int W = kModelWidth, H = kModelHeight;
        fp32_host.resize(static_cast<size_t>(3) * W * H);
        float* dst = fp32_host.data();
        for (int c = 0; c < 3; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    dst[c * H * W + h * W + w] = resized.at<cv::Vec3f>(h, w)[c] - mean[c];
        u8_host.clear();
    }
    return true;
}

bool VGG_SSD::CreateModelInput(const std::vector<void*>& dev_buffers,
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

void VGG_SSD::DestroyModelInput() {
    if (input_) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
            aclDestroyDataBuffer(aclmdlGetDatasetBuffer(input_, i));
        }
        aclmdlDestroyDataset(input_);
        input_ = nullptr;
    }
}

bool VGG_SSD::CreateModelOutput() {
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

void VGG_SSD::DestroyModelOutput() {
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

bool VGG_SSD::ExecuteModel() {
    return aclmdlExecute(model_id_, input_, output_) == ACL_ERROR_NONE;
}

float VGG_SSD::IOU(const BBoxRaw& b1, const BBoxRaw& b2) {
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
// 后处理：对齐 cann/python/models/detection/vgg_ssd.py post_process
//   output[0]: 首元素是 box_num（int32 / float，Python 用 out0[0]）
//   output[1]: 检测结果数组，每 8 个 float 一组：
//      [image_id, label, score, x1, y1, x2, y2, (extra)]
//   坐标是归一化 [0,1]，乘原图尺寸还原
// ============================================================
void VGG_SSD::ProcessModelOutput(const aclmdlDataset* output,
                                 int orig_w, int orig_h,
                                 std::vector<DetectionResult>& results) {
    results.clear();
    size_t output_num = aclmdlGetDatasetNumBuffers(output);
    if (output_num < 2) return;

    // ---- Device → Host 拷贝 2 个输出 ----
    aclDataBuffer* out0_buf = aclmdlGetDatasetBuffer(output, 0);
    aclDataBuffer* out1_buf = aclmdlGetDatasetBuffer(output, 1);
    if (!out0_buf || !out1_buf) return;

    size_t s0 = aclGetDataBufferSizeV2(out0_buf);
    size_t s1 = aclGetDataBufferSizeV2(out1_buf);
    void* h0 = nullptr;
    void* h1 = nullptr;
    if (aclrtMallocHost(&h0, s0) != ACL_ERROR_NONE || !h0) return;
    if (aclrtMallocHost(&h1, s1) != ACL_ERROR_NONE || !h1) { aclrtFreeHost(h0); return; }
    if (aclrtMemcpy(h0, s0, aclGetDataBufferAddr(out0_buf), s0, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
        aclrtFreeHost(h0); aclrtFreeHost(h1); return;
    }
    if (aclrtMemcpy(h1, s1, aclGetDataBufferAddr(out1_buf), s1, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
        aclrtFreeHost(h0); aclrtFreeHost(h1); return;
    }

    // Python: box_num = int(out0[0])
    const float* out0_f = static_cast<const float*>(h0);
    const float* out1 = static_cast<const float*>(h1);
    size_t out1_floats = s1 / sizeof(float);
    int box_num = static_cast<int>(out0_f[0]);

    // Python: det_count = len(out1) // 8
    size_t det_count = out1_floats / 8;
    // 用 box_num 兜底（某些实现 box_num 才是准的）
    if (box_num > 0 && static_cast<size_t>(box_num) < det_count) det_count = static_cast<size_t>(box_num);

    // ---- Step1: 收集满足条件的框（对齐 Python 的多重过滤） ----
    std::vector<BBoxRaw> all;
    for (size_t b = 0; b < det_count; ++b) {
        size_t base = b * 8;
        if (base + 7 >= out1_floats) break;

        int label = static_cast<int>(out1[base + 1]);
        float score = out1[base + 2];
        float x1 = out1[base + 3];
        float y1 = out1[base + 4];
        float x2 = out1[base + 5];
        float y2 = out1[base + 6];

        if (score < kConfThreshold) continue;
        if (label <= 0 || label >= 21) continue;  // VOC 1..20，0 是背景

        // 钳制坐标到 [0,1]（Python: max(0,min(1,x))）
        x1 = std::max(0.0f, std::min(1.0f, x1));
        y1 = std::max(0.0f, std::min(1.0f, y1));
        x2 = std::max(0.0f, std::min(1.0f, x2));
        y2 = std::max(0.0f, std::min(1.0f, y2));

        BBoxRaw box;
        box.x1 = x1; box.y1 = y1; box.x2 = x2; box.y2 = y2;
        box.score = score;
        box.class_index = label;
        all.push_back(box);
    }

    // ---- Step2: 按分数降序 ----
    std::sort(all.begin(), all.end(),
              [](const BBoxRaw& a, const BBoxRaw& b) { return a.score > b.score; });

    // ---- Step3: 按类 NMS（对齐 Python _nms_results，IoU=0.30） ----
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

    // ---- Step4: 每类最多保留 kMaxPerClass（对齐 Python _keep_best_per_class=1） ----
    {
        std::vector<std::vector<size_t>> by_class(21);
        for (size_t i = 0; i < nms_out.size(); ++i) by_class[nms_out[i].class_index].push_back(i);
        std::vector<BBoxRaw> kept;
        for (int c = 1; c <= 20; ++c) {
            for (size_t k = 0; k < by_class[c].size() && k < static_cast<size_t>(kMaxPerClass); ++k) {
                kept.push_back(nms_out[by_class[c][k]]);
            }
        }
        nms_out.swap(kept);
    }

    // ---- Step5: 决定最终保留数量（对齐 Python: high_conf_count 决定 final_k） ----
    int final_k;
    if (!nms_out.empty()) {
        int high_conf_count = 0;
        for (const auto& b : nms_out) if (b.score >= kHighConfThreshold) ++high_conf_count;
        final_k = std::min(std::max(high_conf_count, 1), kTopK);
    } else {
        final_k = kTopK;
    }

    // ---- Step6: 按分数降序，取 final_k ----
    std::sort(nms_out.begin(), nms_out.end(),
              [](const BBoxRaw& a, const BBoxRaw& b) { return a.score > b.score; });
    if (static_cast<int>(nms_out.size()) > final_k) nms_out.resize(final_k);

    // ---- 还原坐标到原图像素 + 生成结果（坐标乘 orig_w/orig_h） ----
    for (const auto& b : nms_out) {
        DetectionResult det;
        det.class_id = b.class_index;
        det.label = GetVocLabel(b.class_index);  // VOC 标签
        det.confidence = b.score;
        det.bbox.x1 = static_cast<int>(std::round(b.x1 * orig_w));
        det.bbox.y1 = static_cast<int>(std::round(b.y1 * orig_h));
        det.bbox.x2 = static_cast<int>(std::round(b.x2 * orig_w));
        det.bbox.y2 = static_cast<int>(std::round(b.y2 * orig_h));
        results.push_back(det);
    }

    aclrtFreeHost(h0);
    aclrtFreeHost(h1);
}

InferenceResult VGG_SSD::Infer(const std::string& image_path) {
    InferenceResult result;
    result.model_name = GetModelName();
    result.model_type = GetModelType();
    result.infer_cost_ms = 0;
    if (!initialized_) return result;

    // ---- 预处理 ----
    std::vector<uint8_t> u8_host;
    std::vector<float> fp32_host;
    bool use_u8 = false;
    int orig_w = 0, orig_h = 0;
    if (!PreProcess(image_path, u8_host, fp32_host, use_u8, orig_w, orig_h)) return result;

// ---- 选择 host 视图 + 分配 device 内存 ----
	    const void* host_ptr = use_u8 ? static_cast<const void*>(u8_host.data())
	                                  : static_cast<const void*>(fp32_host.data());
	    size_t host_size = use_u8 ? u8_host.size() * sizeof(uint8_t)
	                              : fp32_host.size() * sizeof(float);

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
