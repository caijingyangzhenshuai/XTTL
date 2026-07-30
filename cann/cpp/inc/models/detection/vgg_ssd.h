#ifndef KZZK_CV_VGG_SSD_H
#define KZZK_CV_VGG_SSD_H

#include "base_model.h"
#include "types.h"
#include <string>
#include <vector>

#include "acl/acl.h"
#include "opencv2/opencv.hpp"

namespace kzzk_cv {

// VGG-SSD：VOC 20 类目标检测，单输入模型
//   AIPP 模式：输入 304x300 NCHW uint8（无归一化，AIPP 配置在 OM 内）
//   非 AIPP 模式：输入 300x300 NCHW float32（减 mean[104,117,123]）
// 输出为 2 个 buffer：output[0] 首元素是 box_num，output[1] 是检测结果（每 8 float 一组）
// 完全对齐 cann/python/models/detection/vgg_ssd.py
class VGG_SSD : public BaseModel {
public:
    VGG_SSD();
    ~VGG_SSD() override;

    bool Initialize(const std::string& model_path, int device_id = 0) override;
    void Finalize() override;
    InferenceResult Infer(const std::string& image_path) override;
    ModelType GetModelType() const override { return ModelType::DETECTION; }
    std::string GetModelName() const override { return "vgg_ssd"; }

private:
    struct BBoxRaw {
        float x1, y1, x2, y2;  // 归一化 [0,1] 坐标（还原前）
        float score;
        int class_index;  // VOC 标签 id（1..20，0 是背景）
    };

    bool InitAclResource();
    void DestroyAclResource();
    bool LoadModel(const char* model_path);
    void UnloadModel();

    // 对齐 Python _detect_aipp_mode：input size < 1000000 字节 → AIPP
    bool DetectAippMode(bool& is_aipp);

    // 预处理：对齐 Python pre_process
    //   AIPP: resize 304x300 → NCHW uint8
    //   非AIPP: resize 300x300 → BGR→RGB → 减 mean → NCHW float32
    bool PreProcess(const std::string& image_path,
                    std::vector<uint8_t>& u8_host,    // AIPP 模式填充
                    std::vector<float>& fp32_host,    // 非 AIPP 模式填充
                    bool& use_u8,
                    int& orig_w, int& orig_h);

    bool CreateModelInput(const std::vector<void*>& dev_buffers,
                          const std::vector<size_t>& buffer_sizes);
    void DestroyModelInput();
    bool CreateModelOutput();
    void DestroyModelOutput();
    bool ExecuteModel();

    // 后处理：对齐 Python post_process
    void ProcessModelOutput(const aclmdlDataset* output,
                            int orig_w, int orig_h,
                            std::vector<DetectionResult>& results);
    static float IOU(const BBoxRaw& b1, const BBoxRaw& b2);

    aclrtContext context_;
    aclrtStream stream_;
    uint32_t model_id_;
    aclmdlDesc* model_desc_;
    aclmdlDataset* input_;
    aclmdlDataset* output_;
    bool is_aipp_;  // 缓存 AIPP 探测结果

    // 对齐 Python 常量
    static const int   kModelWidth;       // 300
    static const int   kModelHeight;      // 300
    static const int   kAippWidth;        // 304（AIPP 模式输入宽）
    static const int   kAippHeight;       // 300
    static const float kConfThreshold;    // 0.05
    static const float kNmsThreshold;     // 0.30
    static const int   kMaxPerClass;      // 1
    static const float kHighConfThreshold;// 0.10
    static const int   kTopK;             // 5
};

} // namespace kzzk_cv

#endif // KZZK_CV_VGG_SSD_H
