#ifndef KZZK_CV_COCO_LABELS_H
#define KZZK_CV_COCO_LABELS_H

#include <vector>
#include <string>

namespace kzzk_cv {

inline const std::vector<std::string>& GetCocoLabels() {
    static const std::vector<std::string> labels = {
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "sofa", "potted plant", "bed", "dining table", "toilet", "TV monitor",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
        "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };
    return labels;
}

inline const std::string& GetCocoLabel(int class_id) {
    static const std::string unknown = "unknown";
    const auto& labels = GetCocoLabels();
    if (class_id >= 0 && class_id < static_cast<int>(labels.size())) {
        return labels[class_id];
    }
    return unknown;
}

} // namespace kzzk_cv

#endif // KZZK_CV_COCO_LABELS_H
