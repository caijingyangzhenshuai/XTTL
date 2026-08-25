#ifndef KZZK_CV_SSD_H
#define KZZK_CV_SSD_H

#include "base_model.h"
#include "types.h"
#include <string>
#include <vector>

#include "acl/acl.h"
#include "opencv2/opencv.hpp"

namespace kzzk {

// SSD (ssd-10.onnx)：COCO 目标检测，单输入模型
//   输入：image，float32[1,3,1200,1200]（RGB /255，CHW）
// 输出为 3 个 buffer：bboxes / labels / scores
//   索引 0 = bboxes (float32[1,nbox,4])，坐标归一化 [0,1]
//   索引 1 = labels (int64[1,nbox])，class_id = 标准COCO索引 + 1（多了 background）
//   索引 2 = scores (float32[1,nbox])
// 完全对齐 cann/python/models/detection/ssd.py
class SSD : public BaseModel {
public:
    SSD();
    ~SSD() override;

    bool Initialize(const std::string& model_path, int device_id = 0) override;
    void Finalize() override;
    InferenceResult Infer(const std::string& image_path) override;
    ModelType GetModelType() const override { return ModelType::DETECTION; }
    std::string GetModelName() const override { return "ssd"; }

private:
    struct BBoxRaw {
        float x1, y1, x2, y2;  // 归一化 [0,1] 坐标（还原前）
        float score;
        size_t class_index;    // 标准 COCO 索引（已减 1 去掉 background）
    };

    bool InitAclResource();
    void DestroyAclResource();
    bool LoadModel(const char* model_path);
    void UnloadModel();

    // 预处理：对齐 Python —— resize 1200x1200，RGB /255，CHW FP32
    bool PreProcess(const std::string& image_path,
                    std::vector<float>& img_host,
                    int& orig_w, int& orig_h);

    bool CreateModelInput(const std::vector<void*>& dev_buffers,
                          const std::vector<size_t>& buffer_sizes);
    void DestroyModelInput();
    bool CreateModelOutput();
    void DestroyModelOutput();
    bool ExecuteModel();

    // 后处理：对齐 Python —— 3 输出 bboxes/labels/scores
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

    static const int   kInputWidth;      // 1200
    static const int   kInputHeight;     // 1200
    static const float kConfThreshold;   // 0.30（对齐 Python CONF_THRESHOLD）
    static const float kNmsThreshold;    // 0.45（对齐 Python NMS_IOU_THRESHOLD）
    static const int   kTopK;            // 10（对齐 Python TOP_K）
};

} // namespace kzzk

#endif // KZZK_CV_SSD_H