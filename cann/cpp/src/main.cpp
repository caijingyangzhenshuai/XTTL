#include <iostream>
#include <vector>
#include <string>
#include <getopt.h>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>

#include "opencv2/opencv.hpp"

#include "kzzk_cv.h"
#include "types.h"
#include "coco_labels.h"

void ShowHelp(const std::string& programName) {
    std::cout << "智能加速卡协同推理软件" << std::endl;
    std::cout << "C++ 版命令行接口" << std::endl;
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help          显示帮助信息" << std::endl;
    std::cout << "  --modelfile     指定模型文件路径（.om 文件）" << std::endl;
    std::cout << "  --imagefile     指定输入图像路径" << std::endl;
    std::cout << std::endl;
    std::cout << "支持的模型:" << std::endl;
    std::cout << "  YOLOv4, YOLOv3, VGG-SSD, RT-DETR  (目标检测)" << std::endl;
    std::cout << "  ResNet50, MobileNetV1, VGG16       (图像分类)" << std::endl;
    std::cout << "  DeepLabV3                          (图像分割)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << programName << " --modelfile /home/HwHiAiUser/cann/model/yolov4.om --imagefile /home/HwHiAiUser/cann/data/dog.jpg" << std::endl;
    std::cout << "  " << programName << " --modelfile /home/HwHiAiUser/cann/model/resnet50.om --imagefile /home/HwHiAiUser/cann/data/cat.jpg" << std::endl;
}

std::string GetOutputDir(const std::string& imageFile) {
    size_t lastSlash = imageFile.find_last_of("/\\");
    if (lastSlash == std::string::npos) {
        return ".";
    }
    return imageFile.substr(0, lastSlash);
}

std::string GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
    return ss.str();
}

void SaveDetectionResults(const kzzk_cv::InferenceResult& result, const std::string& modelName,
                          const std::string& imageFile) {
    std::string out_dir = GetOutputDir(imageFile);
    std::string timestamp = GetTimestamp();
    std::string txt_path = out_dir + "/" + timestamp + "_" + modelName + ".txt";

    std::ofstream outFile(txt_path);
    if (!outFile.is_open()) {
        std::cerr << "警告: 无法打开文件保存结果: " << txt_path << std::endl;
        return;
    }

    outFile << "Total inference cost: " << result.infer_cost_ms << "ms" << std::endl;
    outFile << "Inference Results:" << std::endl;
    outFile << modelName << "目标检测推理" << std::endl;
    outFile << "Total objects: " << result.detections.size() << std::endl;
    outFile << "Format: [ID] (Class Name) [x1, y1, x2, y2] Confidence" << std::endl;

    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& det = result.detections[i];
        outFile << "[" << i + 1 << "] " << det.class_id << " (" << det.label << ") ["
                << det.bbox.x1 << "," << det.bbox.y1 << ","
                << det.bbox.x2 << "," << det.bbox.y2 << "] "
                << static_cast<int>(std::round(det.confidence * 100)) << "%" << std::endl;
    }

    outFile.close();
    std::cout << "推理结果已保存到: " << txt_path << std::endl;
}

void SaveClassificationResults(const kzzk_cv::InferenceResult& result, const std::string& modelName,
                                const std::string& imageFile) {
    std::string out_dir = GetOutputDir(imageFile);
    std::string timestamp = GetTimestamp();
    std::string txt_path = out_dir + "/" + timestamp + "_" + modelName + ".txt";

    std::ofstream outFile(txt_path);
    if (!outFile.is_open()) {
        std::cerr << "警告: 无法打开文件保存结果: " << txt_path << std::endl;
        return;
    }

    outFile << "Total inference cost: " << result.infer_cost_ms << "ms" << std::endl;
    outFile << "Inference Results:" << std::endl;
    outFile << modelName << "图像分类推理" << std::endl;
    outFile << "Top 5 classes:" << std::endl;
    outFile << "Format: [ID Confidence]" << std::endl;

    if (result.classifications.empty()) {
        outFile << "未识别到模型训练范围内的类别" << std::endl;
    }

    for (size_t i = 0; i < result.classifications.size(); ++i) {
        const auto& cls = result.classifications[i];
        outFile << "Result " << i + 1 << ": [" << cls.class_id << " " << cls.confidence << "%]" << std::endl;
    }

    outFile.close();
    std::cout << "推理结果已保存到: " << txt_path << std::endl;
}

void SaveSegmentationResults(const kzzk_cv::InferenceResult& result, const std::string& modelName,
                              const std::string& imageFile) {
    std::string out_dir = GetOutputDir(imageFile);
    std::string timestamp = GetTimestamp();
    std::string txt_path = out_dir + "/" + timestamp + "_" + modelName + ".txt";

    std::ofstream outFile(txt_path);
    if (!outFile.is_open()) {
        std::cerr << "警告: 无法打开文件保存结果: " << txt_path << std::endl;
        return;
    }

    const auto& seg = result.segmentation;

    outFile << "Total inference cost: " << result.infer_cost_ms << "ms" << std::endl;
    outFile << "Inference Results:" << std::endl;
    outFile << modelName << "语义分割推理" << std::endl;
    outFile << "Output classes count: " << seg.class_info.size() << std::endl;
    outFile << "Format: [ID] (Class Name) [Pixels] [Ratio]" << std::endl;

    for (size_t i = 0; i < seg.class_info.size(); ++i) {
        const auto& cls = seg.class_info[i];
        outFile << "Class " << i + 1 << ": [" << cls.id << "] " << cls.id << " (" << cls.name << ") ["
                << cls.pixels << "] [" << std::fixed << std::setprecision(2) << cls.ratio << "%]" << std::endl;
    }

    outFile.close();

    std::string base_name = imageFile;
    size_t lastSlash = base_name.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        base_name = base_name.substr(lastSlash + 1);
    }
    size_t lastDot = base_name.find_last_of('.');
    if (lastDot != std::string::npos) {
        base_name = base_name.substr(0, lastDot);
    }
    std::string seg_img_path = out_dir + "/" + base_name + "_seg.png";

    cv::Mat orig = cv::imread(imageFile);
    if (!orig.empty()) {
        cv::Mat seg_img(seg.height, seg.width, CV_8UC1, (void*)seg.seg_map.data());
        cv::Mat color_seg;
        cv::applyColorMap(seg_img, color_seg, cv::COLORMAP_JET);
        cv::Mat overlay;
        cv::addWeighted(orig, 0.6, color_seg, 0.4, 0, overlay);
        cv::imwrite(seg_img_path, overlay);
        std::cout << "Segmentation image: " << seg_img_path << std::endl;
    }

    std::cout << "推理结果已保存到: " << txt_path << std::endl;
}

void PrintDetectionResults(const kzzk_cv::InferenceResult& result, const std::string& modelName) {
    std::cout << "Inference Results:" << std::endl;
    std::cout << modelName << "目标检测推理" << std::endl;
    std::cout << "Total objects: " << result.detections.size() << std::endl;
    std::cout << "Format: [ID] (Class Name) [x1, y1, x2, y2] Confidence" << std::endl;

    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& det = result.detections[i];
        std::cout << "[" << i + 1 << "] " << det.class_id << " (" << det.label << ") ["
                  << det.bbox.x1 << "," << det.bbox.y1 << ","
                  << det.bbox.x2 << "," << det.bbox.y2 << "] "
                  << static_cast<int>(std::round(det.confidence * 100)) << "%" << std::endl;
    }
}

void PrintClassificationResults(const kzzk_cv::InferenceResult& result, const std::string& modelName) {
    std::cout << "Inference Results:" << std::endl;
    std::cout << modelName << "图像分类推理" << std::endl;
    std::cout << "Top 5 classes:" << std::endl;
    std::cout << "Format: [ID Confidence]" << std::endl;

    if (result.classifications.empty()) {
        std::cout << "未识别到模型训练范围内的类别（输入图像可能不在 cat/dog 范围内，已自动过滤 OOD 瞎猜结果）" << std::endl;
    }

    for (size_t i = 0; i < result.classifications.size(); ++i) {
        const auto& cls = result.classifications[i];
        std::cout << "Result " << i + 1 << ": [" << cls.class_id << " " << cls.confidence << "%]" << std::endl;
    }
}

void PrintSegmentationResults(const kzzk_cv::InferenceResult& result, const std::string& modelName) {
    const auto& seg = result.segmentation;

    std::cout << "Inference Results:" << std::endl;
    std::cout << modelName << "语义分割推理" << std::endl;
    std::cout << "Output classes count: " << seg.class_info.size() << std::endl;
    std::cout << "Format: [ID] (Class Name) [Pixels] [Ratio]" << std::endl;

    for (size_t i = 0; i < seg.class_info.size(); ++i) {
        const auto& cls = seg.class_info[i];
        std::cout << "Class " << i + 1 << ": [" << cls.id << "] " << cls.id << " (" << cls.name << ") ["
                  << cls.pixels << "] [" << std::fixed << std::setprecision(2) << cls.ratio << "%]" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::string modelFile = "";
    std::string imageFile = "";

    const char* shortOpts = "hm:i:";
    const struct option longOpts[] = {
        {"help", no_argument, nullptr, 'h'},
        {"modelfile", required_argument, nullptr, 'm'},
        {"imagefile", required_argument, nullptr, 'i'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, shortOpts, longOpts, nullptr)) != -1) {
        switch (opt) {
            case 'h':
                ShowHelp(argv[0]);
                return 0;
            case 'm':
                modelFile = optarg;
                break;
            case 'i':
                imageFile = optarg;
                break;
            default:
                ShowHelp(argv[0]);
                return -1;
        }
    }

    if (modelFile.empty() || imageFile.empty()) {
        std::cerr << "错误: 必须指定 --modelfile 和 --imagefile 参数" << std::endl;
        ShowHelp(argv[0]);
        return -1;
    }

    kzzk_cv::InferenceResult result = kzzk_cv::kzzk_cv(modelFile, imageFile);

    if (result.infer_cost_ms == 0 && result.detections.empty() &&
        result.classifications.empty() && result.segmentation.class_info.empty()) {
        std::cerr << "错误: 模型推理失败，请检查模型文件和图像路径" << std::endl;
        return -1;
    }

    std::cout << "Total inference cost: " << result.infer_cost_ms << "ms" << std::endl;

    kzzk_cv::ModelType type = result.model_type;
    std::string displayName = result.model_name;

    switch (type) {
        case kzzk_cv::ModelType::DETECTION:
            PrintDetectionResults(result, displayName);
            SaveDetectionResults(result, displayName, imageFile);
            break;
        case kzzk_cv::ModelType::CLASSIFICATION:
            PrintClassificationResults(result, displayName);
            SaveClassificationResults(result, displayName, imageFile);
            break;
        case kzzk_cv::ModelType::SEGMENTATION:
            PrintSegmentationResults(result, displayName);
            SaveSegmentationResults(result, displayName, imageFile);
            break;
    }

    return 0;
}
