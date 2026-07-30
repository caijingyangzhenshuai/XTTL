#ifndef KZZK_CV_YOLOV3_H
#define KZZK_CV_YOLOV3_H

#include "base_model.h"
#include "types.h"
#include <string>
#include <vector>
#include <cstdint>

#include "acl/acl.h"
#include "opencv2/opencv.hpp"

namespace kzzk_cv {

// YOLOv3：COCO 80 类目标检测，模型输入为 YUV NV12（416x416）+ image_info（4 个 float）
// 完全对齐 cann/python/models/detection/yolov3.py 的预处理与后处理
class YOLOv3 : public BaseModel {
public:
    YOLOv3();
    ~YOLOv3() override;

    bool Initialize(const std::string& model_path, int device_id = 0) override;
    void Finalize() override;
    InferenceResult Infer(const std::string& image_path) override;
    ModelType GetModelType() const override { return ModelType::DETECTION; }
    std::string GetModelName() const override { return "yolov3"; }

private:
    struct BBoxRaw {
        float x1, y1, x2, y2;  // 像素坐标（已还原到原图尺度）
        float score;
        size_t class_index;
    };

    bool InitAclResource();
    void DestroyAclResource();
    bool LoadModel(const char* model_path);
    void UnloadModel();

    // 预处理：对齐 Python pre_process —— RGB→YUV NV12（416x416）+ image_info[416,416,416,416]
    // 返回：nv12_host(NV12 字节)、info_host(4 float)、orig_w/orig_h
    bool PreProcess(const std::string& image_path,
                    std::vector<uint8_t>& nv12_host,
                    std::vector<float>& info_host,
                    int& orig_w, int& orig_h);

    // 用 buffer 列表构造多输入 dataset（input[0]=NV12 dev ptr, input[1]=image_info dev ptr）
    bool CreateModelInput(const std::vector<void*>& dev_buffers,
                          const std::vector<size_t>& buffer_sizes);
    void DestroyModelInput();
    bool CreateModelOutput();
    void DestroyModelOutput();
    bool ExecuteModel();

    // 后处理：对齐 Python post_process
    // output[1] = box_num（int32，shape [1,1]），output[0] = box_info（float，列布局）
    void ProcessModelOutput(const aclmdlDataset* output,
                            int orig_w, int orig_h,
                            std::vector<DetectionResult>& results);

    aclrtContext context_;
    aclrtStream stream_;

    uint32_t model_id_;
    aclmdlDesc* model_desc_;
    aclmdlDataset* input_;
    aclmdlDataset* output_;

    static const int   kInputWidth;      // 416
    static const int   kInputHeight;     // 416
    static const float kConfThreshold;   // 0.4（对齐 Python CONF_THRESHOLD）
    static const int   kTopK;            // 10（对齐 Python TOP_K）
};

} // namespace kzzk_cv

#endif // KZZK_CV_YOLOV3_H
