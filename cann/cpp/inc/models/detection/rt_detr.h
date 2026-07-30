#ifndef KZZK_CV_RT_DETR_H
#define KZZK_CV_RT_DETR_H

#include "base_model.h"
#include "types.h"
#include <string>
#include <vector>

#include "acl/acl.h"
#include "opencv2/opencv.hpp"

namespace kzzk_cv {

// RT-DETR：COCO 80 类目标检测，模型为 2 输入：
//   input[0] = FP32 CHW 图像（1x3x640x640，RGB /255 归一化）
//   input[1] = int64 target_sizes（1x2，[orig_w, orig_h]）
// 输出为 3 个 buffer：labels / boxes / scores
// 完全对齐 cann/python/models/detection/rt_detr.py
class RT_DETR : public BaseModel {
public:
    RT_DETR();
    ~RT_DETR() override;

    bool Initialize(const std::string& model_path, int device_id = 0) override;
    void Finalize() override;
    InferenceResult Infer(const std::string& image_path) override;
    ModelType GetModelType() const override { return ModelType::DETECTION; }
    std::string GetModelName() const override { return "rt_detr"; }

private:
    struct BBoxRaw {
        float x1, y1, x2, y2;
        float score;
        size_t class_index;
    };

    bool InitAclResource();
    void DestroyAclResource();
    bool LoadModel(const char* model_path);
    void UnloadModel();

    // 预处理：对齐 Python —— resize 640x640，RGB /255，CHW FP32
    bool PreProcess(const std::string& image_path,
                    std::vector<float>& img_host,
                    int64_t target_sizes[2],
                    int& orig_w, int& orig_h);

    // 多输入 dataset 构造
    bool CreateModelInput(const std::vector<void*>& dev_buffers,
                          const std::vector<size_t>& buffer_sizes);
    void DestroyModelInput();
    bool CreateModelOutput();
    void DestroyModelOutput();
    bool ExecuteModel();

    // 后处理：对齐 Python —— 3 输出 labels/boxes/scores，坐标已是原图尺寸
    void ProcessModelOutput(const aclmdlDataset* output,
                            std::vector<DetectionResult>& results);
    static float IOU(const BBoxRaw& b1, const BBoxRaw& b2);

    aclrtContext context_;
    aclrtStream stream_;
    uint32_t model_id_;
    aclmdlDesc* model_desc_;
    aclmdlDataset* input_;
    aclmdlDataset* output_;

    static const int   kInputWidth;       // 640
    static const int   kInputHeight;      // 640
    static const float kConfThreshold;    // 0.30（对齐 Python CONF_THRESHOLD）
    static const float kNmsThreshold;     // 0.45（对齐 Python NMS_IOU_THRESHOLD）
    static const int   kMaxPerClass;      // 2（对齐 Python MAX_PER_CLASS）
    static const int   kTopK;             // 10（对齐 Python TOP_K）
};

} // namespace kzzk_cv

#endif // KZZK_CV_RT_DETR_H
