#ifndef KZZK_CV_YOLOV3_H
#define KZZK_CV_YOLOV3_H

#include "base_model.h"
#include "types.h"
#include <string>
#include <vector>
#include <cstdint>

#include "acl/acl.h"
#include "opencv2/opencv.hpp"

namespace kzzk {

// YOLOv3 (yolov3-10.onnx 裁剪版)：COCO 80 类目标检测
//   输入1: input_1, float32[1,3,416,416]（RGB /255, NCHW）
//   输入2: image_shape, float32[1,2]（[orig_w, orig_h]）
//   输出（3 个，NCHW 特征图）：
//     output[0] = convolution_output2, float32[1,255,13,13]
//     output[1] = convolution_output1, float32[1,255,26,26]
//     output[2] = convolution_output,  float32[1,255,52,52]
// 后处理在代码内完成 YOLOv3 anchor 解码 + NMS。
// 完全对齐 cann/python/models/detection/yolov3.py
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
        float x1, y1, x2, y2;  // 归一化 [0,1] 坐标
        float score;
        size_t class_index;
    };

    bool InitAclResource();
    void DestroyAclResource();
    bool LoadModel(const char* model_path);
    void UnloadModel();

    // 预处理：对齐 Python —— RGB→resize 416x416→/255→CHW float32
    //   input[0] = CHW float32 图像
    //   input[1] = [orig_w, orig_h] float32
    bool PreProcess(const std::string& image_path,
                    std::vector<float>& img_host,
                    std::vector<float>& shape_host,
                    int& orig_w, int& orig_h);

    bool CreateModelInput(const std::vector<void*>& dev_buffers,
                          const std::vector<size_t>& buffer_sizes);
    void DestroyModelInput();
    bool CreateModelOutput();
    void DestroyModelOutput();
    bool ExecuteModel();

    // 后处理：对齐 Python —— 3 输出特征图，YOLOv3 解码 + NMS
    void ProcessModelOutput(const aclmdlDataset* output,
                            int orig_w, int orig_h,
                            std::vector<DetectionResult>& results);

    // 解码单个尺度特征图 [1,255,H,W]（NCHW float32），并把检测框填入 all
    // grid: 13/26/52；anchors 是 3 个 (w,h)，归一化到输入尺寸
    static void DecodeScale(const float* feat, int grid, const float anchors[3][2],
                            std::vector<BBoxRaw>& all);
    static float IOU(const BBoxRaw& b1, const BBoxRaw& b2);

    aclrtContext context_;
    aclrtStream stream_;

    uint32_t model_id_;
    aclmdlDesc* model_desc_;
    aclmdlDataset* input_;
    aclmdlDataset* output_;

    static const int   kInputWidth;      // 416
    static const int   kInputHeight;     // 416
    static const float kConfThreshold;   // 0.30（对齐 Python CONF_THRESHOLD）
    static const float kNmsThreshold;    // 0.45（对齐 Python NMS_IOU_THRESHOLD）
    static const int   kTopK;            // 10（对齐 Python TOP_K）
    static const int   kNumClasses;      // 80
};

} // namespace kzzk

#endif // KZZK_CV_YOLOV3_H