/**
 * sample_kzzk_cv.cpp
 *
 * 示例：调用 kzzk_cv::kzzk_cv(modelfile, imagefile) 接口进行 CANN NPU 推理。
 *
 * 编译方式（与现有项目相同）：
 *   g++ -std=c++14 sample_kzzk_cv.cpp -I inc -I /usr/local/Ascend/ascend-toolkit/latest/acllib/include \
 *       -L /usr/local/Ascend/ascend-toolkit/latest/acllib/lib64 -lascendcl -lacldvpp \
 *       `pkg-config --cflags --libs opencv4` \
 *       -o sample_kzzk_cv
 *
 * 或直接加入现有 CMakeLists.txt 的 add_executable 中。
 *
 * 运行方式：
 *   ./sample_kzzk_cv --modelfile /path/to/model.om --imagefile /path/to/image.jpg
 */

#include <iostream>
#include <string>
#include <cstring>

#include "kzzk_cv.h"     // 统一接口头文件
#include "types.h"        // 结果类型定义

static void print_usage(const char* prog) {
    std::cout << "用法: " << prog << " --modelfile <模型.om> --imagefile <图片>\n"
              << "示例: " << prog << " --modelfile /home/HwHiAiUser/cann/model/yolov4.om "
              << "--imagefile /home/HwHiAiUser/cann/data/dog.jpg\n";
}

int main(int argc, char* argv[]) {
    // ---------- 解析命令行参数 ----------
    std::string model_file, image_file;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--modelfile") == 0 && i + 1 < argc) {
            model_file = argv[++i];
        } else if (strcmp(argv[i], "--imagefile") == 0 && i + 1 < argc) {
            image_file = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (model_file.empty() || image_file.empty()) {
        std::cerr << "[ERROR] 必须指定 --modelfile 和 --imagefile 参数\n";
        print_usage(argv[0]);
        return -1;
    }

    // ---------- 调用 kzzk_cv 接口 ----------
    kzzk_cv::InferenceResult result = kzzk_cv::kzzk_cv(model_file, image_file);

    // ---------- 检查推理结果 ----------
    if (result.infer_cost_ms == 0 &&
        result.detections.empty() &&
        result.classifications.empty() &&
        result.segmentation.class_info.empty()) {
        std::cerr << "[ERROR] 推理失败，请检查模型文件和图片路径\n";
        return -1;
    }

    std::cout << "推理耗时: " << result.infer_cost_ms << " ms\n";
    std::cout << "模型名称: " << result.model_name << "\n\n";

    // ---------- 按类型输出结果 ----------
    switch (result.model_type) {
        case kzzk_cv::ModelType::DETECTION: {
            std::cout << "=== 目标检测结果 ===\n"
                      << "检测到 " << result.detections.size() << " 个目标\n";
            for (size_t i = 0; i < result.detections.size(); ++i) {
                const auto& d = result.detections[i];
                std::cout << "  [" << (i + 1) << "] "
                          << "ClassID=" << d.class_id
                          << " (" << d.label << ") "
                          << "[" << d.bbox.x1 << "," << d.bbox.y1
                          << "," << d.bbox.x2 << "," << d.bbox.y2 << "] "
                          << static_cast<int>(d.confidence * 100) << "%\n";
            }
            break;
        }
        case kzzk_cv::ModelType::CLASSIFICATION: {
            std::cout << "=== 图像分类结果 ===\n"
                      << "Top " << result.classifications.size() << " 类别:\n";
            for (size_t i = 0; i < result.classifications.size(); ++i) {
                const auto& c = result.classifications[i];
                std::cout << "  Result " << (i + 1) << ": ["
                          << c.class_id << " " << c.confidence << "%]\n";
            }
            break;
        }
        case kzzk_cv::ModelType::SEGMENTATION: {
            const auto& seg = result.segmentation;
            std::cout << "=== 语义分割结果 ===\n"
                      << "分割尺寸: " << seg.width << "x" << seg.height << "\n"
                      << "类别数: " << seg.class_info.size() << "\n";
            for (size_t i = 0; i < seg.class_info.size(); ++i) {
                const auto& ci = seg.class_info[i];
                std::cout << "  Class " << (i + 1) << ": [ID=" << ci.id
                          << " (" << ci.name << ")] "
                          << "像素=" << ci.pixels
                          << " 占比=" << ci.ratio << "%\n";
            }
            break;
        }
    }

    return 0;
}