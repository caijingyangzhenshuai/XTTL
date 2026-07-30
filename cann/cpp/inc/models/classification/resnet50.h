#ifndef KZZK_CV_RESNET50_H
#define KZZK_CV_RESNET50_H

#include "base_model.h"
#include "types.h"
#include <string>
#include <vector>

#include "acl/acl.h"
#include "opencv2/opencv.hpp"

namespace kzzk_cv {

class ResNet50 : public BaseModel {
public:
    ResNet50();
    ~ResNet50() override;

    bool Initialize(const std::string& model_path, int device_id = 0) override;
    void Finalize() override;
    InferenceResult Infer(const std::string& image_path) override;
    ModelType GetModelType() const override { return ModelType::CLASSIFICATION; }
    std::string GetModelName() const override { return "resnet50"; }

private:
    bool InitAclResource();
    void DestroyAclResource();
    bool LoadModel(const char* model_path);
    void UnloadModel();
    bool PreProcess(const std::string& image_path, std::vector<float>& output,
                    int& orig_w, int& orig_h);
    bool CreateModelInput(void* input_data_buffer, size_t buffer_size);
    void DestroyModelInput();
    bool CreateModelOutput();
    void DestroyModelOutput();
    bool ExecuteModel();
    void PostProcess(const float* host_out_data, size_t num_classes,
                     std::vector<ClassificationResult>& results);

    aclrtContext context_;
    aclrtStream stream_;

    uint32_t model_id_;
    aclmdlDesc* model_desc_;
    aclmdlDataset* input_;
    aclmdlDataset* output_;

    static const int kTopK = 5;
    static const int kInputWidth = 224;
    static const int kInputHeight = 224;
};

} // namespace kzzk_cv

#endif // KZZK_CV_RESNET50_H
