#ifndef KZZK_CV_DEEPLABV3_H
#define KZZK_CV_DEEPLABV3_H

#include "base_model.h"
#include "types.h"
#include <string>
#include <vector>
#include <cstdint>

#include "acl/acl.h"
#include "opencv2/opencv.hpp"

namespace kzzk {

class DeepLabV3 : public BaseModel {
public:
    DeepLabV3();
    ~DeepLabV3() override;

    bool Initialize(const std::string& model_path, int device_id = 0) override;
    void Finalize() override;
    InferenceResult Infer(const std::string& image_path) override;
    ModelType GetModelType() const override { return ModelType::SEGMENTATION; }
    std::string GetModelName() const override { return "deeplabv3"; }

private:
    bool InitAclResource();
    void DestroyAclResource();
    bool LoadModel(const char* model_path);
    void UnloadModel();
    void DetectAippMode();

    bool PreProcess(const std::string& image_path,
                    std::vector<int8_t>& output,
                    int& orig_w, int& orig_h);
    bool CreateModelInput(const std::vector<void*>& dev_buffers,
                          const std::vector<size_t>& buffer_sizes);
    void DestroyModelInput();
    bool CreateModelOutput();
    void DestroyModelOutput();
    bool ExecuteModel();
    void PostProcess(const aclmdlDataset* output, int orig_w, int orig_h,
                     SegmentationResult& result);
    static std::string GetClassName(int class_id);

    aclrtContext context_;
    aclrtStream stream_;

    uint32_t model_id_;
    aclmdlDesc* model_desc_;
    aclmdlDataset* input_;
    aclmdlDataset* output_;
    bool is_aipp_;

    static const int kInputWidth = 513;
    static const int kInputHeight = 513;
    static const int kNumClasses = 21;
};

} // namespace kzzk

#endif // KZZK_CV_DEEPLABV3_H