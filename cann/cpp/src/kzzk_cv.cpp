#include "kzzk_cv.h"
#include "base_model.h"
#include "models/detection/yolov4.h"
#include "models/detection/yolov3.h"
#include "models/detection/vgg_ssd.h"
#include "models/detection/rt_detr.h"
#include "models/classification/resnet50.h"
#include "models/classification/mobilenetv1.h"
#include "models/classification/vgg16.h"
#include "models/segmentation/deeplabv3.h"

#include <memory>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <sys/stat.h>

// ============================================================
// 静默模式开关：去掉所有 [INFO]/[DEBUG]/[WARN] 中间日志
// 需要调试时注释掉下面这行即可恢复完整日志输出
// ============================================================
#define KZZK_CV_SILENT_MODE

#ifdef KZZK_CV_SILENT_MODE
#  define KZZK_CERR   (void)0
#else
#  define KZZK_CERR   std::cerr
#endif

namespace kzzk_cv {

static std::string to_lower(const std::string& s) {
    std::string res = s;
    std::transform(res.begin(), res.end(), res.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return res;
}

static bool file_exists(const std::string& path) {
    struct stat st;
    return (stat(path.c_str(), &st) == 0) && S_ISREG(st.st_mode);
}

static std::string extract_model_name(const std::string& model_path) {
    std::string name = model_path;
    size_t last_slash = name.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        name = name.substr(last_slash + 1);
    }
    size_t last_dot = name.find_last_of('.');
    if (last_dot != std::string::npos) {
        name = name.substr(0, last_dot);
    }
    return to_lower(name);
}

static std::unique_ptr<BaseModel> create_model(const std::string& model_name) {
    std::string lower = to_lower(model_name);

    if (lower.find("yolov4") != std::string::npos) {
        return std::make_unique<YOLOv4>();
    } else if (lower.find("yolov3") != std::string::npos) {
        return std::make_unique<YOLOv3>();
    } else if (lower.find("vgg_ssd") != std::string::npos || lower.find("vgg-ssd") != std::string::npos) {
        return std::make_unique<VGG_SSD>();
    } else if (lower.find("rtdetr") != std::string::npos ||
               lower.find("rt_detr") != std::string::npos ||
               lower.find("rt-detr") != std::string::npos) {
        return std::make_unique<RT_DETR>();
    } else if (lower.find("resnet50") != std::string::npos) {
        return std::make_unique<ResNet50>();
    } else if (lower.find("mobilenet") != std::string::npos) {
        return std::make_unique<MobileNetV1>();
    } else if (lower.find("vgg16") != std::string::npos) {
        return std::make_unique<VGG16>();
    } else if (lower.find("deeplabv3") != std::string::npos ||
               lower.find("deeplab") != std::string::npos) {
        return std::make_unique<DeepLabV3>();
    }

    return nullptr;
}

InferenceResult kzzk_cv(const std::string& modelfile, const std::string& imagefile) {
    InferenceResult result;
    result.infer_cost_ms = 0;

    // ---------- 第一步：文件存在性预检查 ----------
    if (!file_exists(modelfile)) {
        std::cerr << "[ERROR] 模型文件不存在: " << modelfile << std::endl;
        result.model_name = extract_model_name(modelfile);
        result.model_type = ModelType::DETECTION;
        return result;
    }
    if (!file_exists(imagefile)) {
        std::cerr << "[ERROR] 图像文件不存在: " << imagefile << std::endl;
        result.model_name = extract_model_name(modelfile);
        result.model_type = ModelType::DETECTION;
        return result;
    }

    // ---------- 第二步：按文件名匹配模型类 ----------
    std::string model_name = extract_model_name(modelfile);
    auto model = create_model(model_name);

    if (!model) {
        std::cerr << "[ERROR] 无法识别的模型类型: '" << model_name
                  << "' (从文件名 " << modelfile << " 解析)" << std::endl;
        std::cerr << "        支持的关键字: yolov4 / yolov3 / vgg_ssd / rt_detr / resnet50 / mobilenet / vgg16 / deeplabv3" << std::endl;
        result.model_name = model_name;
        result.model_type = ModelType::DETECTION;
        return result;
    }

    result.model_name = model->GetModelName();
    result.model_type = model->GetModelType();
    // ----- 静默模式：去掉 [INFO] 中间日志（取消注释调试时，需同步注释掉顶部 #define KZZK_CV_SILENT_MODE）-----
    // KZZK_CERR << "[INFO] 已识别模型: " << result.model_name << " (类型: "
    //           << static_cast<int>(result.model_type) << ")" << std::endl;
    // ------------------------------------------

    // ---------- 第三步：模型初始化 ----------
    if (!model->Initialize(modelfile, 0)) {
        std::cerr << "[ERROR] 模型初始化失败，请检查: " << std::endl;
        std::cerr << "        1) Ascend 驱动/环境是否就绪 (npu-smi info)" << std::endl;
        std::cerr << "        2) .om 模型文件是否损坏或与当前芯片架构匹配" << std::endl;
        std::cerr << "        3) 设备权限 (是否需要 root)" << std::endl;
        return result;
    }
    // ----- 静默模式：去掉 [INFO] 中间日志 -----
    // KZZK_CERR << "[INFO] 模型初始化成功" << std::endl;
    // ------------------------------------------

    // ---------- 第四步：执行推理 ----------
    result = model->Infer(imagefile);
    result.model_name = model->GetModelName();
    result.model_type = model->GetModelType();

    size_t total = result.detections.size() + result.classifications.size() + result.segmentation.class_info.size();
    // 真正失败的条件：耗时为 0 并且 **没有任何输出结果**（detection/classification/segmentation 全为空）
    // 如果有任一结果，哪怕耗时显示为 0ms 也只是因为 <1ms 被整数截断了，推理实际成功！
    if (result.infer_cost_ms == 0 && total == 0) {
        std::cerr << "[ERROR] 推理执行失败 (耗时为 0)，可能是图像预处理或模型执行阶段出错" << std::endl;
    } else {
        // ----- 静默模式：去掉 [INFO] 中间日志 -----
        // KZZK_CERR << "[INFO] 推理完成，耗时: " << result.infer_cost_ms << "ms" << std::endl;
        // KZZK_CERR << "[INFO] 输出结果条数: " << total << std::endl;
        // ------------------------------------------
    }

    model->Finalize();
    return result;
}

} // namespace kzzk_cv
