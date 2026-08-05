#ifndef KZZK_CV_TYPES_H
#define KZZK_CV_TYPES_H

#include <string>
#include <vector>

namespace kzzk {

enum class ModelType {
    DETECTION,
    CLASSIFICATION,
    SEGMENTATION
};

struct BBox {
    int x1;
    int y1;
    int x2;
    int y2;
};

struct DetectionResult {
    int class_id;
    std::string label;
    float confidence;
    BBox bbox;
};

struct ClassificationResult {
    int class_id;
    int confidence;
};

struct ClassInfo {
    int id;
    std::string name;
    int pixels;
    float ratio;
};

struct SegmentationResult {
    int width;
    int height;
    std::vector<ClassInfo> class_info;
    std::vector<unsigned char> seg_map;
};

struct InferenceResult {
    std::string model_name;
    ModelType model_type;
    int infer_cost_ms;
    std::vector<DetectionResult> detections;
    std::vector<ClassificationResult> classifications;
    SegmentationResult segmentation;
};

} // namespace kzzk

#endif // KZZK_CV_TYPES_H
