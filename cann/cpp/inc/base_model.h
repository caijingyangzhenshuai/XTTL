#ifndef KZZK_CV_BASE_MODEL_H
#define KZZK_CV_BASE_MODEL_H

#include <string>
#include "types.h"

namespace kzzk {

class BaseModel {
public:
    BaseModel();
    virtual ~BaseModel();

    virtual bool Initialize(const std::string& model_path, int device_id = 0) = 0;
    virtual void Finalize() = 0;

    virtual InferenceResult Infer(const std::string& image_path) = 0;

    virtual ModelType GetModelType() const = 0;
    virtual std::string GetModelName() const = 0;

protected:
    int device_id_;
    bool initialized_;
    std::string model_path_;
};

} // namespace kzzk

#endif // KZZK_CV_BASE_MODEL_H
