#ifndef KZZK_CV_YOLOV4_H
#define KZZK_CV_YOLOV4_H

#include "base_model.h"
#include "types.h"
#include <string>
#include <vector>
#include <cstdint>

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "opencv2/opencv.hpp"

namespace kzzk {

class YOLOv4 : public BaseModel {
public:
    YOLOv4();
    ~YOLOv4() override;

    bool Initialize(const std::string& model_path, int device_id = 0) override;
    void Finalize() override;
    InferenceResult Infer(const std::string& image_path) override;
    ModelType GetModelType() const override { return ModelType::DETECTION; }
    std::string GetModelName() const override { return "yolov4"; }

private:
    struct BBoxRaw {
        float x;
        float y;
        float w;
        float h;
        float score;
        size_t class_index;
    };

    static uint32_t AlignSize(uint32_t orig_size, uint32_t alignment);
    bool InitAclResource();
    void DestroyAclResource();
    bool LoadModel(const char* model_path);
    void UnloadModel();
    bool InitDvppResource();
    void DestroyDvppResource();
    bool InitDvppOutputPara(int model_input_width, int model_input_height);
    void DestroyDvppOutputPara();
    bool GetModelInputWH(int& width, int& height);
    bool ProcessDvpp(const std::string& image_path, void*& output_buffer, int& output_size,
                     float& x_scale, float& y_scale, int& orig_w, int& orig_h);
    bool CreateModelInput(void* input_data_buffer, size_t buffer_size);
    void DestroyModelInput();
    bool CreateModelOutput();
    void DestroyModelOutput();
    bool ExecuteModel();
    void ProcessModelOutput(const aclmdlDataset* output, float x_scale, float y_scale,
                            int orig_w, int orig_h,
                            std::vector<DetectionResult>& results);
    static bool SortScore(const BBoxRaw& b1, const BBoxRaw& b2);
    static float IOU(const BBoxRaw& b1, const BBoxRaw& b2);
    void NMS(std::vector<BBoxRaw>& boxes, std::vector<BBoxRaw>& result);

    aclrtContext context_;
    aclrtStream stream_;

    uint32_t model_id_;
    aclmdlDesc* model_desc_;
    aclmdlDataset* input_;
    aclmdlDataset* output_;

    acldvppChannelDesc* dvpp_channel_desc_;
    acldvppResizeConfig* resize_config_;
    void* decode_out_dev_buffer_;
    acldvppPicDesc* decode_output_desc_;
    acldvppPicDesc* resize_input_desc_;
    acldvppPicDesc* resize_output_desc_;
    void* in_dev_buffer_;
    uint32_t in_dev_buffer_size_;
    uint32_t jpeg_decode_output_size_;
    uint32_t decode_output_width_;
    uint32_t decode_output_width_stride_;
    uint32_t decode_output_height_;
    void* resize_out_buffer_dev_;
    uint32_t resize_out_buffer_size_;
    uint32_t model_input_width_;
    uint32_t model_input_height_;
    uint32_t resize_out_width_stride_;
    uint32_t resize_out_height_stride_;

    static const size_t kClassNum;
    static const size_t kModelOutputBoxNum;  // 多尺度 anchors 总数 = (13²+26²+52²)*3 = 10647
    static const float kNmsThreshold;
    static const float kScoreThreshold;

    // 按 anchor index 从 output[1] 坐标 buffer 里取 (cx, cy, w, h)，乘 x/y scale
    void SetBoxInfo(size_t anchor_index, BBoxRaw& box,
                    const float* box_host_buf, float x_scale, float y_scale);
};

} // namespace kzzk

#endif // KZZK_CV_YOLOV4_H
