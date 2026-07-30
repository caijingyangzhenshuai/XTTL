#ifndef KZZK_CV_VGG16_H
#define KZZK_CV_VGG16_H

#include "base_model.h"
#include "types.h"
#include <string>
#include <vector>
#include <cstdint>

#include "acl/acl.h"
#include "opencv2/opencv.hpp"

namespace kzzk_cv {

class VGG16 : public BaseModel {
public:
    VGG16();
    ~VGG16() override;

    bool Initialize(const std::string& model_path, int device_id = 0) override;
    void Finalize() override;
    InferenceResult Infer(const std::string& image_path) override;
    ModelType GetModelType() const override { return ModelType::CLASSIFICATION; }
    std::string GetModelName() const override { return "vgg16"; }

private:
    bool InitAclResource();
    void DestroyAclResource();
    bool LoadModel(const char* model_path);
    void UnloadModel();

    // ---------- 预处理：和 Python/NV12 + AIPP 路径完全对齐 ----------
    // Python: PIL(RGB) -> resize(224x256) -> RGB2NV12(uint8) -> AIPP 内部完成后续
    // C++:    OpenCV(BGR) -> resize(模型要求 W/H) -> BGR2YUV_I420 -> 手动拼 NV12(uint8)
    // 返回：true=成功，nv12_out 输出 NV12 字节流，mw/mh 模型 W/H
    bool PreProcess(const std::string& image_path,
                    std::vector<uint8_t>& nv12_out,
                    int& orig_w, int& orig_h,
                    int& mw, int& mh);

    bool CreateModelInput(void* input_data_buffer, size_t buffer_size);
    void DestroyModelInput();
    bool CreateModelOutput();
    void DestroyModelOutput();
    bool ExecuteModel();

    // ---------- 后处理：对齐 Python（不加 Softmax，模型输出已是概率） ----------
    void PostProcess(const float* host_out_data, size_t num_classes,
                     std::vector<ClassificationResult>& results);

    aclrtContext context_;
    aclrtStream stream_;

    uint32_t model_id_;
    aclmdlDesc* model_desc_;
    aclmdlDataset* input_;
    aclmdlDataset* output_;

    // ---------- 从模型描述符自动读取的输入参数（不再硬编码） ----------
    int model_input_width_;   // 从 om 模型 dims 解析
    int model_input_height_;  // 从 om 模型 dims 解析
    size_t model_input_size_; // 从 om 模型 size_bytes 解析（就是 NV12 实际需要的 size）

    static const int kTopK = 5;
    // ---------- 正式版本阈值（清理调试后恢复） ----------
    // 二分类模型说明：
    //   - kMinConfidencePercent=60：过滤低置信度样本（Top1<60% → 不是已知类）
    //   - kMaxTopGapPercent=101：二分类 Softmax 对清晰样本会输出 100%/0%（Gap 100%）
    //     这是正常表现（logits绝对值大时Softmax会饱和），所以此处暂不过滤Gap；
    //     如需过滤OOD（bus/bird等），后续可加置信度校准或logits绝对值判断
    static const int kMinConfidencePercent = 60;
    static const int kMaxTopGapPercent   = 101;
};

} // namespace kzzk_cv

#endif // KZZK_CV_VGG16_H
