#include "models/classification/resnet50.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>

namespace kzzk_cv {

ResNet50::ResNet50()
    : context_(nullptr), stream_(nullptr),
      model_id_(0), model_desc_(nullptr), input_(nullptr), output_(nullptr) {
    initialized_ = false;
}

ResNet50::~ResNet50() {
    Finalize();
}

bool ResNet50::Initialize(const std::string& model_path, int device_id) {
    if (initialized_) return true;

    device_id_ = device_id;
    model_path_ = model_path;

    if (!InitAclResource()) return false;
    if (!LoadModel(model_path.c_str())) return false;

    initialized_ = true;
    return true;
}

void ResNet50::Finalize() {
    if (!initialized_) return;

    UnloadModel();
    DestroyAclResource();
    initialized_ = false;
}

bool ResNet50::InitAclResource() {
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

void ResNet50::DestroyAclResource() {
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

bool ResNet50::LoadModel(const char* model_path) {
    aclError ret = aclmdlLoadFromFile(model_path, &model_id_);
    if (ret != ACL_ERROR_NONE) return false;

    model_desc_ = aclmdlCreateDesc();
    if (!model_desc_) return false;

    ret = aclmdlGetDesc(model_desc_, model_id_);
    if (ret != ACL_ERROR_NONE) return false;

    return true;
}

void ResNet50::UnloadModel() {
    if (model_id_ != 0) {
        aclmdlUnload(model_id_);
        model_id_ = 0;
    }
    if (model_desc_) {
        aclmdlDestroyDesc(model_desc_);
        model_desc_ = nullptr;
    }
}

bool ResNet50::PreProcess(const std::string& image_path, std::vector<float>& output,
                           int& orig_w, int& orig_h) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) return false;

    orig_w = img.cols;
    orig_h = img.rows;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(kInputWidth, kInputHeight));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    rgb.convertTo(rgb, CV_32FC3);

    output.resize(kInputWidth * kInputHeight * 3);
    float* out_ptr = output.data();

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < kInputHeight; ++h) {
            for (int w = 0; w < kInputWidth; ++w) {
                out_ptr[c * kInputHeight * kInputWidth + h * kInputWidth + w] =
                    rgb.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    return true;
}

bool ResNet50::CreateModelInput(void* input_data_buffer, size_t buffer_size) {
    input_ = aclmdlCreateDataset();
    if (!input_) return false;

    aclDataBuffer* data_buffer = aclCreateDataBuffer(input_data_buffer, buffer_size);
    if (!data_buffer) return false;

    aclError ret = aclmdlAddDatasetBuffer(input_, data_buffer);
    if (ret != ACL_ERROR_NONE) return false;

    return true;
}

void ResNet50::DestroyModelInput() {
    if (input_) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
            aclDataBuffer* buffer = aclmdlGetDatasetBuffer(input_, i);
            aclDestroyDataBuffer(buffer);
        }
        aclmdlDestroyDataset(input_);
        input_ = nullptr;
    }
}

bool ResNet50::CreateModelOutput() {
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

void ResNet50::DestroyModelOutput() {
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

bool ResNet50::ExecuteModel() {
    aclError ret = aclmdlExecute(model_id_, input_, output_);
    return ret == ACL_ERROR_NONE;
}

void ResNet50::PostProcess(const float* host_out_data, size_t num_classes,
                            std::vector<ClassificationResult>& results) {
    results.clear();
    if (!host_out_data || num_classes == 0) return;

    std::vector<std::pair<float, int>> scores;
    scores.reserve(num_classes);
    for (size_t i = 0; i < num_classes; ++i) {
        scores.emplace_back(host_out_data[i], static_cast<int>(i));
    }

    int top_k = std::min(kTopK, static_cast<int>(num_classes));
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

InferenceResult ResNet50::Infer(const std::string& image_path) {
    InferenceResult result;
    result.model_name = GetModelName();
    result.model_type = GetModelType();
    result.infer_cost_ms = 0;

    if (!initialized_) return result;

    std::vector<float> input_data;
    int orig_w, orig_h;
    if (!PreProcess(image_path, input_data, orig_w, orig_h)) return result;

    void* dev_input = nullptr;
    size_t input_size = input_data.size() * sizeof(float);
    aclError ret = aclrtMalloc(&dev_input, input_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) return result;

    ret = aclrtMemcpy(dev_input, input_size, input_data.data(), input_size,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        aclrtFree(dev_input);
        return result;
    }

    if (!CreateModelInput(dev_input, input_size)) {
        aclrtFree(dev_input);
        return result;
    }

    if (!CreateModelOutput()) {
        DestroyModelInput();
        aclrtFree(dev_input);
        return result;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    if (!ExecuteModel()) {
        DestroyModelOutput();
        DestroyModelInput();
        aclrtFree(dev_input);
        return result;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    result.infer_cost_ms = static_cast<int>(std::max<long long>(1LL, std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()));

    // ============================================================
    // 致命修复：和 YOLOv4 一样，必须先 Device→Host 拷贝，再做后处理！
    // 直接读 aclGetDataBufferAddr() 返回的 Device 指针会 100% 段错误
    // ============================================================
    {
        aclDataBuffer* out_buffer = aclmdlGetDatasetBuffer(output_, 0);
        void* dev_out_addr = aclGetDataBufferAddr(out_buffer);
        size_t dev_out_size = aclGetDataBufferSizeV2(out_buffer);

        if (!dev_out_addr || dev_out_size == 0) {
            DestroyModelOutput();
            DestroyModelInput();
            aclrtFree(dev_input);
            return result;
        }

        // 1. 分配 Host 端 pinned memory
        void* host_out_ptr = nullptr;
        ret = aclrtMallocHost(&host_out_ptr, dev_out_size);
        if (ret != ACL_ERROR_NONE || !host_out_ptr) {
            DestroyModelOutput();
            DestroyModelInput();
            aclrtFree(dev_input);
            return result;
        }

        // 2. Device → Host 拷贝
        ret = aclrtMemcpy(host_out_ptr, dev_out_size, dev_out_addr, dev_out_size,
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_ERROR_NONE) {
            aclrtFreeHost(host_out_ptr);
            DestroyModelOutput();
            DestroyModelInput();
            aclrtFree(dev_input);
            return result;
        }

        // 3. 后处理：只读 Host 内存，绝对不碰 Device 指针！
        const float* host_out_f32 = static_cast<const float*>(host_out_ptr);
        size_t num_classes = dev_out_size / sizeof(float);
        PostProcess(host_out_f32, num_classes, result.classifications);

        // 4. 释放 Host 内存
        aclrtFreeHost(host_out_ptr);
    }

    DestroyModelOutput();
    DestroyModelInput();
    aclrtFree(dev_input);

    return result;
}

} // namespace kzzk_cv
