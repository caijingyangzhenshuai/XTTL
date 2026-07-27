#include <iostream>
#include <vector>
#include <string>
#include <getopt.h>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include "yolov4.h"
#include "resnet50.h"

// 显示帮助信息
void ShowHelp(const std::string& programName) {
    std::cout << "智能加速卡协同推理工具" << std::endl;
    std::cout << "命令行接口" << std::endl;
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help          Show this help message" << std::endl;
    std::cout << "  --modelfile     Path to the model file (.om)" << std::endl;
    std::cout << "  --imagefile     Path to the input image file" << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << programName << " -m model/yolov4.om -i test.jpg" << std::endl;
    std::cout << "  " << programName << " --modelfile model/resnet50.om --imagefile image.png" << std::endl;
}


// 保存推理结果到文件
void SaveResultsToFile(
    const std::string& modelType,
    const std::string& modelFile,
    const std::string& imageFile,
    const std::vector<std::vector<int>>& result
) {
    // 生成时间戳
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");

    // 提取模型名称
    std::string modelName = modelFile;
    size_t lastSlash = modelName.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        modelName = modelName.substr(lastSlash + 1);
    }
    size_t lastDot = modelName.find_last_of('.');
    if (lastDot != std::string::npos) {
        modelName = modelName.substr(0, lastDot);
    }

    // 构建文件名
    std::stringstream fileName;
    fileName << "/home/HwHiAiUser/cann/data/" << timestamp.str() << "_" << modelName << ".txt";

    // 打开文件
    std::ofstream outFile(fileName.str());
    if (!outFile.is_open()) {
        std::cerr << "警告: 无法打开文件保存结果: " << fileName.str() << std::endl;
        return;
    }

    // 写入文件头信息
    outFile << "==================================================" << std::endl;
    outFile << "CANN 模型推理结果" << std::endl;
    outFile << "==================================================" << std::endl;
    outFile << "模型类型: " << modelType << std::endl;
    outFile << "模型文件: " << modelFile << std::endl;
    outFile << "图像文件: " << imageFile << std::endl;
    outFile << "时间戳: " << timestamp.str() << std::endl;
    outFile << "结果数量: " << result.size() << std::endl;
    outFile << "==================================================" << std::endl;

    // 写入结果数据
    if (result.empty()) {
        outFile << "未检测到任何结果" << std::endl;
    } else {
        if (modelType == "yolov4") {
            outFile << "【目标检测结果 - YOLOv4】" << std::endl;
            outFile << "格式说明: [序号] 类别ID (类别名称) [x1, y1, x2, y2] 置信度" << std::endl;
            outFile << "--------------------------------------------------" << std::endl;

            for (size_t i = 0; i < result.size(); ++i) {
                if (result[i].size() >= 6) {
                    int classId = result[i][0];
                    int x1 = result[i][1];
                    int y1 = result[i][2];
                    int x2 = result[i][3];
                    int y2 = result[i][4];
                    int confidence = result[i][5];

                    std::string className = (classId >= 0 && classId < YOLOV4_LABELS.size())
                                            ? YOLOV4_LABELS[classId] : "unknown";

                    outFile << "[" << i + 1 << "] "
                           << classId << " (" << className << ") ["
                           << x1 << ", " << y1 << ", "
                           << x2 << ", " << y2 << "] "
                           << std::fixed << std::setprecision(3)
                           << confidence << "%" << std::endl;
                }
            }
        } else {
            outFile << "【图像分类结果 - ResNet50】" << std::endl;
            outFile << "格式说明: [序号] 类别ID 置信度" << std::endl;
            outFile << "--------------------------------------------------" << std::endl;

            for (size_t i = 0; i < result.size(); ++i) {
                if (result[i].size() >= 2) {
                    int classId = result[i][0];
                    int confidence = result[i][1];

                    outFile << "[" << i + 1 << "] "
                           << classId << " "
                           << confidence << std::endl;
                }
            }
        }
    }

    outFile << "==================================================" << std::endl;
    outFile.close();

    std::cout << "推理结果已保存到: " << fileName.str() << std::endl;
}

// 主函数
int main(int argc, char* argv[]) {
    std::string modelFile = "";
    std::string imageFile = "";

    // 解析命令行参数
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


    // 检查必要参数
    if (modelFile.empty() || imageFile.empty()) {
        std::cerr << "Error: Model file and image file are required" << std::endl;
        ShowHelp(argv[0]);
        return -1;
    }

    // 通过模型文件名识别模型类型
    std::string modelType = "yolov4"; // 默认使用yolov4
    if (modelFile.find("resnet50") != std::string::npos) {
        modelType = "resnet50";
    } else if (modelFile.find("yolov4") == std::string::npos) {
        std::cerr << "Warning: Cannot determine model type from filename, defaulting to yolov4" << std::endl;
    }

    std::vector<std::vector<int>> result;

    if (modelType == "yolov4") {
        // 使用YOLOV4模型
        YOLOv4 yolov4;
        result = yolov4.kzzk_cv(modelFile, imageFile);
    } else {
        // 使用RESNET50模型
        RESNET50 resnet50;
        if (!resnet50.Initialize()) {
            std::cerr << "Error: Failed to initialize RESNET50" << std::endl;
            return -1;
        }

        result = resnet50.kzzk_cv(modelFile, imageFile);

        resnet50.Finalize();
    }

    // 输出结果
    std::cout << "Inference Results:" << std::endl;
    if (result.empty()) {
        std::cout << "No results found" << std::endl;
    } else {
        if (modelType == "yolov4") {
            // YOLOv4 输出格式
	    std::cout << "YOLOv4图像检测推理" << std::endl;
            std::cout << "Total objects: " << result.size() << std::endl;
            std::cout << "Format: [ID] (Class Name) [x1, y1, x2, y2] Confidence" << std::endl;

            for (size_t i = 0; i < result.size(); ++i) {
                if (result[i].size() >= 6) {
                    int classId = result[i][0];
                    int x1 = result[i][1];
                    int y1 = result[i][2];
                    int x2 = result[i][3];
                    int y2 = result[i][4];
                    int confidence = result[i][5];

                    // 获取类别名称
                    std::string className = (classId >= 0 && classId < YOLOV4_LABELS.size()) ? YOLOV4_LABELS[classId] : "unknown";

                    // 格式化输出
                    std::cout << "[" << i + 1 << "] " << classId << " (" << className << ") ["
                              << x1 << ", " << y1 << ", "
                              << x2 << ", " << y2 << "] "
                              << std::fixed << std::setprecision(3) << confidence << "%" << std::endl;
                }
            }
        } else {
            // ResNet50 输出格式
	    std::cout << "ResNet50图像识别推理" << std::endl;
	    std::cout << "Top 5 classes:" << std::endl;
            std::cout << "Format: [ID Confidence]" << std::endl;

            for (size_t i = 0; i < result.size(); ++i) {
                if (result[i].size() >= 2) {
                    int classId = result[i][0];
                    int confidence = result[i][1];

                    // 格式化输出
                    std::cout << "Result " << i + 1 << ": [" << classId << ", " << confidence << "]" << std::endl;
                }
            }
        }
    }

    // 保存结果到文件
    SaveResultsToFile(modelType, modelFile, imageFile, result);

    return 0;
}

