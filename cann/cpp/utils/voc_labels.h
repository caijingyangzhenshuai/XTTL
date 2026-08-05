#ifndef KZZK_CV_VOC_LABELS_H
#define KZZK_CV_VOC_LABELS_H

#include <vector>
#include <string>

namespace kzzk {

// VOC 2012 数据集 20 类（含 background 共 21 个）
// 对齐 cann/python/models/detection/vgg_ssd.py 中的 LABELS_VOC
inline const std::vector<std::string>& GetVocLabels() {
    static const std::vector<std::string> labels = {
        "background",
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    };
    return labels;
}

inline const std::string& GetVocLabel(int class_id) {
    static const std::string unknown = "unknown";
    const auto& labels = GetVocLabels();
    if (class_id >= 0 && class_id < static_cast<int>(labels.size())) {
        return labels[class_id];
    }
    return unknown;
}

} // namespace kzzk

#endif // KZZK_CV_VOC_LABELS_H
