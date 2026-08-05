#include "models/detection/yolov4.h"
#include "coco_labels.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <sys/stat.h>
#include <chrono>
#include <cstdio>
#include <cstring>

// ============================================================================
// 静默模式开关（用户要求去掉所有 [INFO]/[DEBUG]/[WARN] 中间日志）
// 如果需要恢复调试日志，把下面这行注释掉即可。
#define YOLOV4_SILENT_MODE
// ============================================================================
namespace {
// 空输出流：吞掉所有 operator<< 的内容，什么都不输出
struct NullOStream {
    template <typename T>
    inline NullOStream& operator<<(const T&) { return *this; }
    inline NullOStream& operator<<(std::ostream& (*)(std::ostream&)) { return *this; }
};
static NullOStream g_null_ostream;
}  // namespace

#ifdef YOLOV4_SILENT_MODE
#  define YOLOV4_CERR   g_null_ostream   // 静默模式 → 丢进空流（完全不打印）
#else
#  define YOLOV4_CERR   YOLOV4_CERR        // 调试模式 → 正常打印到 stderr
#endif

namespace kzzk {

const size_t YOLOv4::kClassNum = 80;
// 多尺度 anchors 总数: (13^2 + 26^2 + 52^2) * 3 = (169 + 676 + 2704) * 3 = 3549 * 3 = 10647
const size_t YOLOv4::kModelOutputBoxNum = 10647;
const float YOLOv4::kNmsThreshold = 0.8f;
const float YOLOv4::kScoreThreshold = 0.4f;

namespace {
// sigmoid: 处理模型输出为 raw logit 的情况 (-∞ ~ +∞) → (0, 1)
inline float Sigmoid(float x) {
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = std::exp(x);
        return z / (1.0f + z);
    }
}

enum class ClsOrder {
    AnchorClass = 0,  // [N_anchor, N_class] → index = i*N_class + j
    ClassAnchor = 1,  // [N_class, N_anchor] → index = j*N_anchor + i
};
enum class CoordOrder {
    Anchor4 = 0,      // [N_anchor, 4] → index = i*4 + (0..3)
    FourAnchor = 1,   // [4, N_anchor] → cx=i, cy=i+N, w=i+2N, h=i+3N
};
}  // namespace

YOLOv4::YOLOv4()
    : context_(nullptr), stream_(nullptr),
      model_id_(0), model_desc_(nullptr), input_(nullptr), output_(nullptr),
      dvpp_channel_desc_(nullptr), resize_config_(nullptr),
      decode_out_dev_buffer_(nullptr), decode_output_desc_(nullptr),
      resize_input_desc_(nullptr), resize_output_desc_(nullptr),
      in_dev_buffer_(nullptr), in_dev_buffer_size_(0),
      jpeg_decode_output_size_(0), decode_output_width_(0),
      decode_output_width_stride_(0), decode_output_height_(0),
      resize_out_buffer_dev_(nullptr), resize_out_buffer_size_(0),
      model_input_width_(0), model_input_height_(0),
      resize_out_width_stride_(0), resize_out_height_stride_(0) {
    initialized_ = false;
}

YOLOv4::~YOLOv4() {
    Finalize();
}

uint32_t YOLOv4::AlignSize(uint32_t orig_size, uint32_t alignment) {
    return (orig_size + alignment - 1) / alignment * alignment;
}

bool YOLOv4::Initialize(const std::string& model_path, int device_id) {
    if (initialized_) return true;

    device_id_ = device_id;
    model_path_ = model_path;

    if (!InitAclResource()) {
        std::cerr << "[ERROR][YOLOv4] ACL 资源初始化失败 (aclInit / setDevice / createContext / createStream)" << std::endl;
        return false;
    }
    if (!LoadModel(model_path.c_str())) {
        std::cerr << "[ERROR][YOLOv4] 模型加载失败: " << model_path << std::endl;
        return false;
    }
    if (!InitDvppResource()) {
        std::cerr << "[ERROR][YOLOv4] DVPP 资源初始化失败 (createChannel / createResizeConfig)" << std::endl;
        return false;
    }

    // 拿到模型输入尺寸，赋值成员变量 + 初始化 DVPP 输出 buffer（参考实现每次 Detect 都 InitDvppOutputPara）
    int w = 0, h = 0;
    if (GetModelInputWH(w, h)) {
        YOLOV4_CERR << "[INFO][YOLOv4] 模型输入尺寸: " << w << " x " << h << std::endl;
        // ★ 关键修复 1：初始化成员宽高（之前只打印没赋值，导致 Infer 里 mw=mh=0 除 0 inf + OpenCV 崩溃）
        model_input_width_  = (uint32_t)w;
        model_input_height_ = (uint32_t)h;
        // ★ 关键修复 2：提前分配好 DVPP resize 的输出 NV12 buffer（259584 bytes）
        if (!InitDvppOutputPara(w, h)) {
            std::cerr << "[ERROR][YOLOv4] InitDvppOutputPara(" << w << "," << h << ") 初始化 DVPP 输出 buffer 失败" << std::endl;
            return false;
        }
    } else {
        YOLOV4_CERR << "[WARN][YOLOv4] 无法从模型描述中读取输入尺寸，将在预处理阶段再次尝试" << std::endl;
    }

    initialized_ = true;
    return true;
}

void YOLOv4::Finalize() {
    if (!initialized_) return;

    DestroyDvppOutputPara();
    DestroyDvppResource();
    UnloadModel();
    DestroyAclResource();

    initialized_ = false;
}

bool YOLOv4::InitAclResource() {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR][YOLOv4] aclInit 失败, ret=" << ret << std::endl;
        return false;
    }

    ret = aclrtSetDevice(device_id_);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR][YOLOv4] aclrtSetDevice(" << device_id_ << ") 失败, ret=" << ret << std::endl;
        return false;
    }

    ret = aclrtCreateContext(&context_, device_id_);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR][YOLOv4] aclrtCreateContext 失败, ret=" << ret << std::endl;
        return false;
    }

    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR][YOLOv4] aclrtCreateStream 失败, ret=" << ret << std::endl;
        return false;
    }

    return true;
}

void YOLOv4::DestroyAclResource() {
    if (stream_) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
    if (context_) {
        aclrtDestroyContext(context_);
        context_ = nullptr;
    }
    aclrtResetDevice(device_id_);
    aclFinalize();
}

bool YOLOv4::LoadModel(const char* model_path) {
    aclError ret = aclmdlLoadFromFile(model_path, &model_id_);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR][YOLOv4] aclmdlLoadFromFile 失败, path=" << model_path << ", ret=" << ret << std::endl;
        return false;
    }
    YOLOV4_CERR << "[INFO][YOLOv4] aclmdlLoadFromFile 成功, model_id=" << model_id_ << std::endl;

    model_desc_ = aclmdlCreateDesc();
    if (model_desc_ == nullptr) {
        std::cerr << "[ERROR][YOLOv4] aclmdlCreateDesc 返回空指针" << std::endl;
        return false;
    }

    ret = aclmdlGetDesc(model_desc_, model_id_);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR][YOLOv4] aclmdlGetDesc 失败, ret=" << ret << std::endl;
        return false;
    }

    // 打印输入/输出数量，方便排查模型结构
    size_t num_inputs = aclmdlGetNumInputs(model_desc_);
    size_t num_outputs = aclmdlGetNumOutputs(model_desc_);
    YOLOV4_CERR << "[INFO][YOLOv4] 模型输入数: " << num_inputs << ", 输出数: " << num_outputs << std::endl;

    // 详细打印每个输入/输出：数据类型、格式、真实字节大小
    // aclDataType (Ascend CANN 正确枚举):
    //   0=ACL_FLOAT(FP32), 1=ACL_FLOAT16(FP16), 2=ACL_INT8, 3=ACL_INT32, 4=ACL_UINT8
    //   6=ACL_INT16, 7=ACL_UINT16, 9=ACL_INT64, 10=ACL_UINT64, 11=ACL_DOUBLE, 12=ACL_BOOL
    // aclFormat (Ascend CANN 正确枚举):
    //   0=ACL_FORMAT_NCHW, 1=ACL_FORMAT_NHWC, 2=ACL_FORMAT_ND, 3=ACL_FORMAT_NC1HWC0
    for (size_t i = 0; i < num_inputs; ++i) {
        aclDataType dt = aclmdlGetInputDataType(model_desc_, i);
        aclFormat fmt   = aclmdlGetInputFormat(model_desc_, i);
        size_t sz       = aclmdlGetInputSizeByIndex(model_desc_, i);
        aclmdlIODims dims;
        aclError r_dim = aclmdlGetInputDims(model_desc_, i, &dims);
        YOLOV4_CERR << "[INFO][YOLOv4]   输入[" << i << "]:";
        if (r_dim == ACL_ERROR_NONE) {
            YOLOV4_CERR << " dims=[";
            for (int d = 0; d < (int)dims.dimCount; ++d) {
                if (d) YOLOV4_CERR << ",";
                YOLOV4_CERR << dims.dims[d];
            }
            YOLOV4_CERR << "]";
        }
        YOLOV4_CERR << " type=" << dt
                  << " format=" << fmt
                  << " size_bytes=" << sz;
        // 给几个关键枚举值加可读性标签（按 Ascend 文档正确枚举值）
        const char* dts = "?";
        switch(dt) {
            case 0:  dts = "FP32";    break;
            case 1:  dts = "FP16";    break;
            case 2:  dts = "INT8";    break;
            case 3:  dts = "INT32";   break;
            case 4:  dts = "UINT8";   break;   // ✓ 用户模型 type=4 就是 UINT8
            case 6:  dts = "INT16";   break;
            case 7:  dts = "UINT16";  break;
            case 9:  dts = "INT64";   break;
            case 10: dts = "UINT64";  break;
            case 11: dts = "DOUBLE";  break;
            case 12: dts = "BOOL";    break;
            default: dts = "?";
        }
        const char* fmts = "?";
        switch(fmt) {
            case 0:  fmts = "NCHW";     break;
            case 1:  fmts = "NHWC";     break;  // ✓ 用户模型 format=1 就是 NHWC
            case 2:  fmts = "ND";       break;
            case 3:  fmts = "NC1HWC0";  break;
            default: fmts = "?";
        }
        YOLOV4_CERR << " (" << dts << " / " << fmts << ")" << std::endl;
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        aclDataType dt = aclmdlGetOutputDataType(model_desc_, i);
        aclFormat fmt   = aclmdlGetOutputFormat(model_desc_, i);
        size_t sz       = aclmdlGetOutputSizeByIndex(model_desc_, i);
        const char* dts = "?";
        switch(dt) {
            case 0:  dts = "FP32";    break;
            case 1:  dts = "FP16";    break;
            case 2:  dts = "INT8";    break;
            case 3:  dts = "INT32";   break;
            case 4:  dts = "UINT8";   break;
            case 11: dts = "DOUBLE";  break;
            default: dts = "?";
        }
        const char* fmts = "?";
        switch(fmt) {
            case 0:  fmts = "NCHW";     break;
            case 1:  fmts = "NHWC";     break;
            case 2:  fmts = "ND";       break;
            case 3:  fmts = "NC1HWC0";  break;
            default: fmts = "?";
        }
        YOLOV4_CERR << "[INFO][YOLOv4]   输出[" << i << "]:"
                  << " type=" << dt
                  << " format=" << fmt
                  << " size_bytes=" << sz
                  << " (" << dts << " / " << fmts << ")" << std::endl;
    }

    return true;
}

void YOLOv4::UnloadModel() {
    if (model_id_ != 0) {
        aclmdlUnload(model_id_);
        model_id_ = 0;
    }
    if (model_desc_) {
        aclmdlDestroyDesc(model_desc_);
        model_desc_ = nullptr;
    }
}

bool YOLOv4::InitDvppResource() {
    dvpp_channel_desc_ = acldvppCreateChannelDesc();
    if (dvpp_channel_desc_ == nullptr) {
        std::cerr << "[ERROR][YOLOv4] acldvppCreateChannelDesc 返回空指针" << std::endl;
        return false;
    }

    aclError ret = acldvppCreateChannel(dvpp_channel_desc_);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR][YOLOv4] acldvppCreateChannel 失败, ret=" << ret << std::endl;
        return false;
    }

    resize_config_ = acldvppCreateResizeConfig();
    if (resize_config_ == nullptr) {
        std::cerr << "[ERROR][YOLOv4] acldvppCreateResizeConfig 返回空指针" << std::endl;
        return false;
    }

    return true;
}

void YOLOv4::DestroyDvppResource() {
    if (resize_config_) {
        acldvppDestroyResizeConfig(resize_config_);
        resize_config_ = nullptr;
    }
    if (dvpp_channel_desc_) {
        acldvppDestroyChannel(dvpp_channel_desc_);
        acldvppDestroyChannelDesc(dvpp_channel_desc_);
        dvpp_channel_desc_ = nullptr;
    }
}

bool YOLOv4::InitDvppOutputPara(int model_input_width, int model_input_height) {
    model_input_width_ = model_input_width;
    model_input_height_ = model_input_height;

    resize_out_width_stride_ = AlignSize(model_input_width_, 16);
    resize_out_height_stride_ = AlignSize(model_input_height_, 2);

    resize_out_buffer_size_ = resize_out_width_stride_ * resize_out_height_stride_ * 3 / 2;

    aclError ret = acldvppMalloc(&resize_out_buffer_dev_, resize_out_buffer_size_);
    if (ret != ACL_ERROR_NONE) return false;

    return true;
}

void YOLOv4::DestroyDvppOutputPara() {
    if (resize_out_buffer_dev_) {
        acldvppFree(resize_out_buffer_dev_);
        resize_out_buffer_dev_ = nullptr;
    }
}

bool YOLOv4::GetModelInputWH(int& width, int& height) {
    if (!model_desc_) return false;

    aclmdlIODims dims;
    aclError ret = aclmdlGetInputDims(model_desc_, 0, &dims);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR][YOLOv4] aclmdlGetInputDims 失败, ret=" << ret << std::endl;
        return false;
    }

    if (dims.dimCount < 4) {
        std::cerr << "[ERROR][YOLOv4] 输入维度数 < 4, dimCount=" << dims.dimCount << std::endl;
        return false;
    }

    // dims.dims[] 可能是 NCHW [N,C,H,W]，也可能是 NHWC [N,H,W,C]
    // YOLOv4 通道数 C=3（RGB/YUV），所以判断哪个位置等于 3
    int w = 0, h = 0;
    if (dims.dims[1] == 3) {
        // NCHW: [0]=N, [1]=C=3, [2]=H, [3]=W
        h = static_cast<int>(dims.dims[2]);
        w = static_cast<int>(dims.dims[3]);
        YOLOV4_CERR << "[INFO][YOLOv4] 输入维度顺序: NCHW [" << dims.dims[0] << "," << dims.dims[1]
                  << "," << dims.dims[2] << "," << dims.dims[3] << "]" << std::endl;
    } else if (dims.dims[3] == 3) {
        // NHWC: [0]=N, [1]=H, [2]=W, [3]=C=3
        h = static_cast<int>(dims.dims[1]);
        w = static_cast<int>(dims.dims[2]);
        YOLOV4_CERR << "[INFO][YOLOv4] 输入维度顺序: NHWC [" << dims.dims[0] << "," << dims.dims[1]
                  << "," << dims.dims[2] << "," << dims.dims[3] << "]" << std::endl;
    } else {
        // 退化成旧逻辑（假设 NCHW），但打个警告
        YOLOV4_CERR << "[WARN][YOLOv4] 无法确定 NCHW/NHWC，dim1=" << dims.dims[1]
                  << " dim3=" << dims.dims[3] << "，按 NCHW 处理" << std::endl;
        h = static_cast<int>(dims.dims[2]);
        w = static_cast<int>(dims.dims[3]);
    }

    width = w;
    height = h;
    return true;
}

// CPU 预处理回退：用 OpenCV 读图 + resize + 转 YUV NV12（YUV420 Semi-Planar）
// 适合 DVPP 无法处理的图片（非标准 JPG、PNG、BMP 等）
static bool PreprocessCpuFallback(const std::string& image_path,
                                    int model_w, int model_h,
                                    std::vector<uint8_t>& out_nv12,
                                    int& orig_w, int& orig_h,
                                    float& x_scale, float& y_scale) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR][YOLOv4][CPU] cv::imread 读取图片失败: " << image_path << std::endl;
        return false;
    }
    orig_w = img.cols;
    orig_h = img.rows;
    YOLOV4_CERR << "[INFO][YOLOv4][CPU] 原图尺寸: " << orig_w << " x " << orig_h << std::endl;

    // 1. 缩放到模型输入尺寸
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);

    // 2. BGR -> YUV I420 (YUV420 Planar: Y plane + U plane + V plane)
    cv::Mat yuv_i420;
    cv::cvtColor(resized, yuv_i420, cv::COLOR_BGR2YUV_I420);
    // I420 布局：W*H (Y) + (W/2)*(H/2) (U) + (W/2)*(H/2) (V)
    int y_size = model_w * model_h;
    int uv_half = (model_w / 2) * (model_h / 2);
    if (yuv_i420.total() != static_cast<size_t>(y_size + 2 * uv_half)) {
        std::cerr << "[ERROR][YOLOv4][CPU] YUV I420 尺寸不匹配: total=" << yuv_i420.total()
                  << " expect=" << (y_size + 2 * uv_half) << std::endl;
        return false;
    }
    uint8_t* yuv_ptr = yuv_i420.data;

    // 3. I420 -> NV12 (Y plane + UV interleaved plane)
    out_nv12.resize(y_size * 3 / 2);
    // Y 平面直接拷贝
    memcpy(out_nv12.data(), yuv_ptr, y_size);
    // UV 平面：把 U 和 V 平面交替合并 (U0 V0 U1 V1 ...)
    uint8_t* uv_dst = out_nv12.data() + y_size;
    uint8_t* u_src = yuv_ptr + y_size;
    uint8_t* v_src = yuv_ptr + y_size + uv_half;
    for (int i = 0; i < uv_half; ++i) {
        uv_dst[i * 2 + 0] = u_src[i];
        uv_dst[i * 2 + 1] = v_src[i];
    }

    x_scale = static_cast<float>(orig_w) / model_w;
    y_scale = static_cast<float>(orig_h) / model_h;
    YOLOV4_CERR << "[INFO][YOLOv4][CPU] CPU 预处理完成: NV12 " << model_w << "x" << model_h
              << " (" << out_nv12.size() << " bytes), scale(x=" << x_scale << ", y=" << y_scale << ")" << std::endl;
    return true;
}

bool YOLOv4::ProcessDvpp(const std::string& image_path, void*& output_buffer,
                          int& output_size, float& x_scale, float& y_scale,
                          int& orig_w, int& orig_h) {
    // ---------- 前置：OpenCV 先读图拿尺寸（同时确认图片是好的） ----------
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR][YOLOv4] cv::imread 读取图片失败: " << image_path << std::endl;
        return false;
    }
    orig_w = img.cols;
    orig_h = img.rows;
    img.release(); // 释放掉，后续 DVPP 自己从文件读原始字节

    int input_w = 0, input_h = 0;
    if (!GetModelInputWH(input_w, input_h)) return false;
    YOLOV4_CERR << "[INFO][YOLOv4] 预处理: 目标模型输入尺寸 " << input_w << "x" << input_h
              << ", 原图 " << orig_w << "x" << orig_h << std::endl;

    // 关键修复：如果已经在 Initialize 中初始化过且尺寸匹配，就不再重复 InitDvppOutputPara（避免旧 buffer 泄漏）
    bool need_init_dvpp_out = (resize_out_buffer_dev_ == nullptr) ||
                                (static_cast<int>(model_input_width_)  != input_w) ||
                                (static_cast<int>(model_input_height_) != input_h);
    if (need_init_dvpp_out) {
        if (!InitDvppOutputPara(input_w, input_h)) {
            std::cerr << "[ERROR][YOLOv4] InitDvppOutputPara 失败 (可能是 acldvppMalloc 出错)" << std::endl;
            return false;
        }
    }

    // ---------- 读取原始文件字节给 DVPP JPEG 解码 ----------
    std::ifstream file(image_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[ERROR][YOLOv4] 无法打开图片文件 (ifstream): " << image_path << std::endl;
        return false;
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(file_size);
    if (!file.read(buffer.data(), file_size)) {
        std::cerr << "[ERROR][YOLOv4] 读取图片文件内容失败, size=" << file_size << std::endl;
        file.close();
        return false;
    }
    file.close();

    aclError ret = acldvppMalloc(&in_dev_buffer_, file_size);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR][YOLOv4] acldvppMalloc 输入缓冲失败, size=" << file_size << ", ret=" << ret << std::endl;
        goto FALLBACK_CPU;
    }
    in_dev_buffer_size_ = static_cast<int>(file_size);

    ret = aclrtMemcpy(in_dev_buffer_, file_size, buffer.data(), file_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR][YOLOv4] aclrtMemcpy H2D 失败, ret=" << ret << std::endl;
        goto FALLBACK_CPU;
    }

    decode_output_width_ = AlignSize(orig_w, 128);
    decode_output_height_ = AlignSize(orig_h, 16);
    decode_output_width_stride_ = decode_output_width_;
    jpeg_decode_output_size_ = decode_output_width_ * decode_output_height_ * 3 / 2;

    ret = acldvppMalloc(&decode_out_dev_buffer_, jpeg_decode_output_size_);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR][YOLOv4] acldvppMalloc 解码输出缓冲失败, ret=" << ret << std::endl;
        goto FALLBACK_CPU;
    }

    decode_output_desc_ = acldvppCreatePicDesc();
    if (!decode_output_desc_) {
        std::cerr << "[ERROR][YOLOv4] acldvppCreatePicDesc (decode output) 返回空" << std::endl;
        goto FALLBACK_CPU;
    }
    acldvppSetPicDescData(decode_output_desc_, decode_out_dev_buffer_);
    acldvppSetPicDescSize(decode_output_desc_, jpeg_decode_output_size_);
    acldvppSetPicDescFormat(decode_output_desc_, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    acldvppSetPicDescWidth(decode_output_desc_, orig_w);
    acldvppSetPicDescHeight(decode_output_desc_, orig_h);
    acldvppSetPicDescWidthStride(decode_output_desc_, decode_output_width_stride_);
    acldvppSetPicDescHeightStride(decode_output_desc_, decode_output_height_);

    ret = acldvppJpegDecodeAsync(dvpp_channel_desc_, in_dev_buffer_, file_size,
                                  decode_output_desc_, stream_);
    if (ret != ACL_ERROR_NONE) {
        YOLOV4_CERR << "[WARN][YOLOv4] acldvppJpegDecodeAsync 失败, ret=" << ret
                  << " (非标准 JPG/PNG/BMP 格式常见问题)，将切换到 CPU 预处理" << std::endl;
        goto FALLBACK_CPU;
    }
    ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_ERROR_NONE) {
        YOLOV4_CERR << "[WARN][YOLOv4] JpegDecode 同步失败, ret=" << ret << "，将切换到 CPU 预处理" << std::endl;
        goto FALLBACK_CPU;
    }

    resize_input_desc_ = acldvppCreatePicDesc();
    if (!resize_input_desc_) {
        YOLOV4_CERR << "[WARN][YOLOv4] acldvppCreatePicDesc (resize input) 空，切换 CPU" << std::endl;
        goto FALLBACK_CPU;
    }
    acldvppSetPicDescData(resize_input_desc_, decode_out_dev_buffer_);
    acldvppSetPicDescSize(resize_input_desc_, jpeg_decode_output_size_);
    acldvppSetPicDescFormat(resize_input_desc_, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    acldvppSetPicDescWidth(resize_input_desc_, orig_w);
    acldvppSetPicDescHeight(resize_input_desc_, orig_h);
    acldvppSetPicDescWidthStride(resize_input_desc_, decode_output_width_stride_);
    acldvppSetPicDescHeightStride(resize_input_desc_, decode_output_height_);

    resize_output_desc_ = acldvppCreatePicDesc();
    if (!resize_output_desc_) {
        YOLOV4_CERR << "[WARN][YOLOv4] acldvppCreatePicDesc (resize output) 空，切换 CPU" << std::endl;
        goto FALLBACK_CPU;
    }
    acldvppSetPicDescData(resize_output_desc_, resize_out_buffer_dev_);
    acldvppSetPicDescSize(resize_output_desc_, resize_out_buffer_size_);
    acldvppSetPicDescFormat(resize_output_desc_, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    acldvppSetPicDescWidth(resize_output_desc_, input_w);
    acldvppSetPicDescHeight(resize_output_desc_, input_h);
    acldvppSetPicDescWidthStride(resize_output_desc_, resize_out_width_stride_);
    acldvppSetPicDescHeightStride(resize_output_desc_, resize_out_height_stride_);

    ret = acldvppVpcResizeAsync(dvpp_channel_desc_, resize_input_desc_,
                                 resize_output_desc_, resize_config_, stream_);
    if (ret != ACL_ERROR_NONE) {
        YOLOV4_CERR << "[WARN][YOLOv4] acldvppVpcResizeAsync 失败, ret=" << ret << "，切换 CPU" << std::endl;
        goto FALLBACK_CPU;
    }
    ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_ERROR_NONE) {
        YOLOV4_CERR << "[WARN][YOLOv4] Resize 同步失败, ret=" << ret << "，切换 CPU" << std::endl;
        goto FALLBACK_CPU;
    }

    // ---------- DVPP 路径成功 ----------
    output_buffer = resize_out_buffer_dev_;
    output_size = resize_out_buffer_size_;
    x_scale = static_cast<float>(orig_w) / input_w;
    y_scale = static_cast<float>(orig_h) / input_h;
    YOLOV4_CERR << "[INFO][YOLOv4] DVPP 预处理完成: " << input_w << "x" << input_h
              << " (" << output_size << " bytes), scale(x=" << x_scale << ", y=" << y_scale << ")" << std::endl;

    // 清理 DVPP 中间资源
    if (in_dev_buffer_)        { acldvppFree(in_dev_buffer_);           in_dev_buffer_ = nullptr; }
    if (decode_out_dev_buffer_){ acldvppFree(decode_out_dev_buffer_);   decode_out_dev_buffer_ = nullptr; }
    if (decode_output_desc_)   { acldvppDestroyPicDesc(decode_output_desc_);  decode_output_desc_ = nullptr; }
    if (resize_input_desc_)    { acldvppDestroyPicDesc(resize_input_desc_);   resize_input_desc_ = nullptr; }
    if (resize_output_desc_)   { acldvppDestroyPicDesc(resize_output_desc_);  resize_output_desc_ = nullptr; }
    return true;

    // ---------- DVPP 失败 → CPU 预处理回退 ----------
FALLBACK_CPU:
    // 先把 DVPP 中间资源清理干净
    if (in_dev_buffer_)        { acldvppFree(in_dev_buffer_);           in_dev_buffer_ = nullptr; }
    if (decode_out_dev_buffer_){ acldvppFree(decode_out_dev_buffer_);   decode_out_dev_buffer_ = nullptr; }
    if (decode_output_desc_)   { acldvppDestroyPicDesc(decode_output_desc_);  decode_output_desc_ = nullptr; }
    if (resize_input_desc_)    { acldvppDestroyPicDesc(resize_input_desc_);   resize_input_desc_ = nullptr; }
    if (resize_output_desc_)   { acldvppDestroyPicDesc(resize_output_desc_);  resize_output_desc_ = nullptr; }
    // ★ 关键修复：不要无条件 free resize_out_buffer_dev_（Initialize 已为该模型尺寸预分配好，下次 Infer 还要用）
    //   只有当当前分配的尺寸与新需求不一致时，才先释放再重新分配
    {
        uint32_t new_stride_w = AlignSize(input_w, 16);
        uint32_t new_stride_h = AlignSize(input_h, 2);
        size_t expected_sz = (size_t)new_stride_w * (size_t)new_stride_h * 3 / 2;
        if ((resize_out_buffer_dev_ != nullptr) && (resize_out_buffer_size_ != expected_sz)) {
            acldvppFree(resize_out_buffer_dev_);
            resize_out_buffer_dev_ = nullptr;
            resize_out_buffer_size_ = 0;
        }
    }

    std::vector<uint8_t> nv12_host;
    if (!PreprocessCpuFallback(image_path, input_w, input_h, nv12_host, orig_w, orig_h, x_scale, y_scale)) {
        return false;
    }
    int nv12_size = static_cast<int>(nv12_host.size());

    // 重新申请 DVPP 设备缓冲（复用 resize_out_buffer_dev_ 字段，保持 Infer 里释放逻辑一致）
    ret = acldvppMalloc(&resize_out_buffer_dev_, nv12_size);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR][YOLOv4][CPU] acldvppMalloc NV12 输出缓冲失败, ret=" << ret << std::endl;
        return false;
    }
    resize_out_buffer_size_ = nv12_size;

    ret = aclrtMemcpy(resize_out_buffer_dev_, nv12_size, nv12_host.data(), nv12_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR][YOLOv4][CPU] aclrtMemcpy NV12 H2D 失败, ret=" << ret << std::endl;
        return false;
    }

    output_buffer = resize_out_buffer_dev_;
    output_size = nv12_size;
    return true;
}

bool YOLOv4::CreateModelInput(void* input_data_buffer, size_t buffer_size) {
    input_ = aclmdlCreateDataset();
    if (!input_) {
        std::cerr << "[ERROR][YOLOv4] aclmdlCreateDataset (input) 返回空" << std::endl;
        return false;
    }

    aclDataBuffer* data_buffer = aclCreateDataBuffer(input_data_buffer, buffer_size);
    if (!data_buffer) {
        std::cerr << "[ERROR][YOLOv4] aclCreateDataBuffer (input) 返回空, size=" << buffer_size << std::endl;
        return false;
    }

    aclError ret = aclmdlAddDatasetBuffer(input_, data_buffer);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR][YOLOv4] aclmdlAddDatasetBuffer (input) 失败, ret=" << ret << std::endl;
        return false;
    }
    YOLOV4_CERR << "[INFO][YOLOv4] 模型输入 Dataset 创建成功, buffer_size=" << buffer_size << std::endl;
    return true;
}

void YOLOv4::DestroyModelInput() {
    if (input_) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
            aclDataBuffer* buffer = aclmdlGetDatasetBuffer(input_, i);
            aclDestroyDataBuffer(buffer);
        }
        aclmdlDestroyDataset(input_);
        input_ = nullptr;
    }
}

bool YOLOv4::CreateModelOutput() {
    output_ = aclmdlCreateDataset();
    if (!output_) {
        std::cerr << "[ERROR][YOLOv4] aclmdlCreateDataset (output) 返回空" << std::endl;
        return false;
    }

    size_t output_num = aclmdlGetNumOutputs(model_desc_);
    YOLOV4_CERR << "[INFO][YOLOv4] 准备创建 " << output_num << " 个输出 buffer" << std::endl;
    for (size_t i = 0; i < output_num; ++i) {
        size_t buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);
        void* output_buffer = nullptr;
        aclError ret = aclrtMalloc(&output_buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            std::cerr << "[ERROR][YOLOv4] aclrtMalloc 输出[" << i << "] 失败, size=" << buffer_size << ", ret=" << ret << std::endl;
            return false;
        }

        aclDataBuffer* data_buffer = aclCreateDataBuffer(output_buffer, buffer_size);
        if (!data_buffer) {
            std::cerr << "[ERROR][YOLOv4] aclCreateDataBuffer (output[" << i << "]) 空" << std::endl;
            return false;
        }

        ret = aclmdlAddDatasetBuffer(output_, data_buffer);
        if (ret != ACL_ERROR_NONE) {
            std::cerr << "[ERROR][YOLOv4] aclmdlAddDatasetBuffer (output[" << i << "]) 失败, ret=" << ret << std::endl;
            return false;
        }
        YOLOV4_CERR << "[INFO][YOLOv4]   output[" << i << "] buffer_size=" << buffer_size << std::endl;
    }
    return true;
}

void YOLOv4::DestroyModelOutput() {
    if (output_) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
            aclDataBuffer* buffer = aclmdlGetDatasetBuffer(output_, i);
            void* data = aclGetDataBufferAddr(buffer);
            if (data) aclrtFree(data);
            aclDestroyDataBuffer(buffer);
        }
        aclmdlDestroyDataset(output_);
        output_ = nullptr;
    }
}

bool YOLOv4::ExecuteModel() {
    aclError ret = aclmdlExecute(model_id_, input_, output_);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[ERROR][YOLOv4] aclmdlExecute 模型推理执行失败, ret=" << ret << std::endl;
        return false;
    }
    YOLOV4_CERR << "[INFO][YOLOv4] aclmdlExecute 执行成功" << std::endl;
    return true;
}

// 从 output[1] (坐标 buffer) 里按 anchor 索引取 cx, cy, w, h，乘缩放因子
void YOLOv4::SetBoxInfo(size_t anchor_index, BBoxRaw& box,
                        const float* box_host_buf, float x_scale, float y_scale) {
    const float* p = box_host_buf + anchor_index * 4;
    box.x = p[0] * x_scale;  // cx
    box.y = p[1] * y_scale;  // cy
    box.w = p[2] * x_scale;  // w
    box.h = p[3] * y_scale;  // h
}

bool YOLOv4::SortScore(const BBoxRaw& b1, const BBoxRaw& b2) {
    return b1.score > b2.score;
}

float YOLOv4::IOU(const BBoxRaw& b1, const BBoxRaw& b2) {
    float x1 = std::max(b1.x - b1.w / 2, b2.x - b2.w / 2);
    float y1 = std::max(b1.y - b1.h / 2, b2.y - b2.h / 2);
    float x2 = std::min(b1.x + b1.w / 2, b2.x + b2.w / 2);
    float y2 = std::min(b1.y + b1.h / 2, b2.y + b2.h / 2);

    float overlap_w = std::max(0.0f, x2 - x1);
    float overlap_h = std::max(0.0f, y2 - y1);
    float overlap = overlap_w * overlap_h;

    float area1 = b1.w * b1.h;
    float area2 = b2.w * b2.h;

    if (area1 + area2 - overlap <= 0) return 0;
    return overlap / (area1 + area2 - overlap);
}

void YOLOv4::NMS(std::vector<BBoxRaw>& boxes, std::vector<BBoxRaw>& result) {
    std::sort(boxes.begin(), boxes.end(), SortScore);

    std::vector<bool> removed(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (removed[i]) continue;
        result.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (removed[j]) continue;
            if (boxes[i].class_index == boxes[j].class_index) {
                if (IOU(boxes[i], boxes[j]) > kNmsThreshold) {
                    removed[j] = true;
                }
            }
        }
    }
}

void YOLOv4::ProcessModelOutput(const aclmdlDataset* output, float x_scale, float y_scale,
                                 int orig_w, int orig_h,
                                 std::vector<DetectionResult>& results) {
    results.clear();

    size_t output_num = aclmdlGetDatasetNumBuffers(output);
    YOLOV4_CERR << "[INFO][YOLOv4] 后处理: 模型输出个数 = " << output_num << std::endl;

    // ================ 第一步：收集所有 Device 端输出信息 ================
    struct OutputInfo {
        void* dev_addr;
        size_t size_bytes;
        void* host_addr;
    };
    std::vector<OutputInfo> outputs(output_num);
    for (size_t i = 0; i < output_num; ++i) {
        aclDataBuffer* b = aclmdlGetDatasetBuffer(output, i);
        outputs[i].dev_addr  = aclGetDataBufferAddr(b);
        outputs[i].size_bytes = aclGetDataBufferSizeV2(b);
        outputs[i].host_addr = nullptr;
        YOLOV4_CERR << "[INFO][YOLOv4]   output[" << i << "] addr=" << outputs[i].dev_addr
                  << " size=" << outputs[i].size_bytes << " bytes ("
                  << (outputs[i].size_bytes / 4) << " floats)" << std::endl;
    }

    // ================ 第二步：Device → Host 拷贝 ================
    // 注意：aclrtMalloc 分配的是 Device 内存，CPU 直接读会段错误！
    bool copy_ok = true;
    for (size_t i = 0; i < output_num; ++i) {
        aclError ret = aclrtMallocHost(&outputs[i].host_addr, outputs[i].size_bytes);
        if (ret != ACL_ERROR_NONE) {
            std::cerr << "[ERROR][YOLOv4] aclrtMallocHost output[" << i << "] 失败, size="
                      << outputs[i].size_bytes << ", ret=" << ret << std::endl;
            copy_ok = false;
            break;
        }
        ret = aclrtMemcpy(outputs[i].host_addr, outputs[i].size_bytes,
                          outputs[i].dev_addr, outputs[i].size_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_ERROR_NONE) {
            std::cerr << "[ERROR][YOLOv4] aclrtMemcpy D→H output[" << i << "] 失败, ret=" << ret << std::endl;
            copy_ok = false;
            break;
        }
    }
    if (!copy_ok) {
        for (auto& o : outputs) {
            if (o.host_addr) aclrtFreeHost(o.host_addr);
        }
        return;
    }
    YOLOV4_CERR << "[INFO][YOLOv4] 所有 Device→Host 拷贝完成" << std::endl;

    // ================ 第三步：判断输出格式并解析 ================
    // 格式 A（原生多尺度 anchors）：
    //   output[0] = 10647 × 80 floats → 分类概率
    //   output[1] = 10647 × 4  floats → 坐标 (cx, cy, w, h)
    // 格式 B（带 NMS/box_num）：
    //   一个输出是单个 int32 box_num，另一个是 box_num × N 个 box 数据
    bool parsed = false;

    size_t expect_cls_bytes = kModelOutputBoxNum * kClassNum * sizeof(float);  // 10647*80*4 = 3,407,040
    size_t expect_box_bytes = kModelOutputBoxNum * 4          * sizeof(float);  // 10647* 4*4 = 170,352
    if (output_num >= 2 &&
        ((outputs[0].size_bytes == expect_cls_bytes && outputs[1].size_bytes == expect_box_bytes) ||
         (outputs[1].size_bytes == expect_cls_bytes && outputs[0].size_bytes == expect_box_bytes))) {
        // --- 格式 A：原生 anchors 输出 ---
        const float* cls_host = nullptr;
        const float* box_host = nullptr;
        if (outputs[0].size_bytes == expect_cls_bytes) {
            cls_host = static_cast<const float*>(outputs[0].host_addr);
            box_host = static_cast<const float*>(outputs[1].host_addr);
            YOLOV4_CERR << "[INFO][YOLOv4] 检测到格式 A (原生 anchors): out[0]=分类(80), out[1]=坐标(4), 共 "
                      << kModelOutputBoxNum << " 个 anchors" << std::endl;
        } else {
            cls_host = static_cast<const float*>(outputs[1].host_addr);
            box_host = static_cast<const float*>(outputs[0].host_addr);
            YOLOV4_CERR << "[INFO][YOLOv4] 检测到格式 A (原生 anchors, 反向): out[1]=分类(80), out[0]=坐标(4)" << std::endl;
        }

        // ============= 【自动检测分类排列 & sigmoid】 =============
        // 尝试 2 种排列 × 2 种值处理 = 4 种组合，取 Top1_score 最高的组合：
        //   排列 AC (AnchorClass): v = cls_host[i*N_class + j]
        //   排列 CA (ClassAnchor): v = cls_host[j*N_anchor + i]
        //   值处理 raw: 直接用 v（假设已经是概率 [0,1]）
        //   值处理 sigmoid: 用 Sigmoid(v)（假设是 raw logit [-∞,+∞]）
        struct ClsConfig {
            ClsOrder order;
            bool use_sigmoid;
            float top1_score;
        };
        ClsConfig best_cls{ClsOrder::AnchorClass, false, -1.0f};
        const ClsOrder orders_arr[2] = {ClsOrder::AnchorClass, ClsOrder::ClassAnchor};
        const bool sig_arr[2] = {false, true};
        for (int oi = 0; oi < 2; ++oi) {
            for (int si = 0; si < 2; ++si) {
                ClsOrder ord = orders_arr[oi];
                bool use_sig = sig_arr[si];
                float cur_top1 = -1.0f;
                for (size_t i = 0; i < kModelOutputBoxNum; ++i) {
                    float ms = -1e9f;  // 接受负数 logit
                    for (size_t j = 0; j < kClassNum; ++j) {
                        float raw_v;
                        if (ord == ClsOrder::AnchorClass) raw_v = cls_host[i * kClassNum + j];
                        else                               raw_v = cls_host[j * kModelOutputBoxNum + i];
                        float v = use_sig ? Sigmoid(raw_v) : raw_v;
                        if (v > ms) ms = v;
                    }
                    if (ms > cur_top1) cur_top1 = ms;
                }
                if (cur_top1 > best_cls.top1_score) {
                    best_cls = ClsConfig{ord, use_sig, cur_top1};
                }
            }
        }
        YOLOV4_CERR << "[INFO][YOLOv4] 分类排列自动检测: AC(raw/ sigmoid) vs CA(raw/ sigmoid), 胜出 = "
                  << (best_cls.order == ClsOrder::AnchorClass ? "[Anchor×Class]" : "[Class×Anchor]")
                  << (best_cls.use_sigmoid ? " + sigmoid" : " + raw")
                  << ", Top1_score = " << best_cls.top1_score << std::endl;

        // ============= 【调试】用最佳配置收集所有 anchor 的 max 分数分布 =============
        std::vector<float> all_max_scores;
        std::vector<size_t> all_max_idx;
        all_max_scores.reserve(kModelOutputBoxNum);
        all_max_idx.reserve(kModelOutputBoxNum);
        for (size_t i = 0; i < kModelOutputBoxNum; ++i) {
            float ms = -1e9f;
            size_t midx = 0;
            for (size_t j = 0; j < kClassNum; ++j) {
                float raw_v;
                if (best_cls.order == ClsOrder::AnchorClass) raw_v = cls_host[i * kClassNum + j];
                else                                         raw_v = cls_host[j * kModelOutputBoxNum + i];
                float v = best_cls.use_sigmoid ? Sigmoid(raw_v) : raw_v;
                if (v > ms) { ms = v; midx = j; }
            }
            all_max_scores.push_back(ms);
            all_max_idx.push_back(midx);
        }
        std::vector<float> sorted_scores = all_max_scores;
        std::sort(sorted_scores.rbegin(), sorted_scores.rend());
        YOLOV4_CERR << "[DEBUG][YOLOv4] 最佳配置下 anchors 分类置信度 Top20: ";
        for (int i = 0; i < 20 && i < (int)sorted_scores.size(); ++i) {
            char b[32]; std::snprintf(b, sizeof(b), "%.3f ", sorted_scores[i]);
            YOLOV4_CERR << b;
        }
        YOLOV4_CERR << std::endl;
        // 分数分档
        int lt001 = 0, lt001_01 = 0, lt01_03 = 0, lt03_05 = 0, ge05 = 0;
        for (float s : all_max_scores) {
            if (s < 0.001f) lt001++;
            else if (s < 0.01f) lt001_01++;
            else if (s < 0.3f) lt01_03++;
            else if (s < 0.5f) lt03_05++;
            else ge05++;
        }
        YOLOV4_CERR << "[DEBUG][YOLOv4] 分数分档: <0.001=" << lt001
                  << ", 0.001~0.01=" << lt001_01
                  << ", 0.01~0.3="   << lt01_03
                  << ", 0.3~0.5="    << lt03_05
                  << ", >=0.5="      << ge05 << std::endl;
        // 两个输出 buffer 前 40 个 float 预览（判断是不是接近 0 或异常值）
        YOLOV4_CERR << "[DEBUG][YOLOv4] 分类buffer前40 float: ";
        for (int i = 0; i < 40; ++i) {
            char b[32]; std::snprintf(b, sizeof(b), "%.3f ", cls_host[i]);
            YOLOV4_CERR << b;
        }
        YOLOV4_CERR << std::endl;
        YOLOV4_CERR << "[DEBUG][YOLOv4] 坐标buffer前40 float: ";
        for (int i = 0; i < 40; ++i) {
            char b[32]; std::snprintf(b, sizeof(b), "%.1f ", box_host[i]);
            YOLOV4_CERR << b;
        }
        YOLOV4_CERR << std::endl;
        // ============= 【调试结束】 =============

        // ★ 高分门槛过滤：4 种组合里 Top1 都 < 0.7，说明分类值都在 0.5 附近（sigmoid(0)=0.5）
        // 或 raw 值全 <0.7，属于垃圾排列，不生成检测结果，避免一堆 50% 假框。
        const float kMinValidTop1 = 0.7f;
        if (best_cls.top1_score < kMinValidTop1) {
            YOLOV4_CERR << "[WARN][YOLOv4] 所有排列组合的分类 Top1_score = " << best_cls.top1_score
                      << " < 有效阈值 " << kMinValidTop1
                      << "，判定为未检测到有效目标（可能模型输入数据异常）。跳过检测。" << std::endl;
            parsed = true;
        } else {
            // 封装 lambda：按指定坐标排列执行检测
            auto RunDetectionByCoord = [&](CoordOrder coord_order) -> std::vector<DetectionResult> {
                std::vector<DetectionResult> res;
                std::vector<BBoxRaw> all_boxes;
                int pass_threshold = 0;
                for (size_t i = 0; i < kModelOutputBoxNum; ++i) {
                    float max_score = all_max_scores[i];
                    size_t max_idx = all_max_idx[i];
                    if (max_score < kScoreThreshold) continue;
                    pass_threshold++;

                    BBoxRaw box;
                    box.score = max_score;
                    box.class_index = max_idx;
                    if (coord_order == CoordOrder::Anchor4) {
                        SetBoxInfo(i, box, box_host, x_scale, y_scale);  // (i*4+0..3)
                    } else {
                        const size_t N = kModelOutputBoxNum;
                        float cx = box_host[i + 0 * N] * x_scale;
                        float cy = box_host[i + 1 * N] * y_scale;
                        float bw = box_host[i + 2 * N] * x_scale;
                        float bh = box_host[i + 3 * N] * y_scale;
                        box.x = cx; box.y = cy; box.w = bw; box.h = bh;
                    }
                    all_boxes.push_back(box);
                }
                YOLOV4_CERR << "[INFO][YOLOv4] 坐标排列 "
                          << (coord_order == CoordOrder::Anchor4 ? "Anchor4 (i*4+0..3): "
                                                                 : "FourAnchor (j*N+i):   ")
                          << "过阈值(" << kScoreThreshold << ") 的 anchors 数: " << pass_threshold
                          << std::endl;

                std::vector<BBoxRaw> nms_result;
                NMS(all_boxes, nms_result);
                YOLOV4_CERR << "[INFO][YOLOv4] NMS(IoU=" << kNmsThreshold << ") 后 box 数: "
                          << nms_result.size() << std::endl;

                for (const auto& box : nms_result) {
                    DetectionResult det;
                    det.class_id = static_cast<int>(box.class_index);
                    det.label = GetCocoLabel(det.class_id);
                    det.confidence = box.score;
                    det.bbox.x1 = static_cast<int>(std::max(0.0f,            box.x - box.w / 2.0f));
                    det.bbox.y1 = static_cast<int>(std::max(0.0f,            box.y - box.h / 2.0f));
                    det.bbox.x2 = static_cast<int>(std::min(static_cast<float>(orig_w), box.x + box.w / 2.0f));
                    det.bbox.y2 = static_cast<int>(std::min(static_cast<float>(orig_h), box.y + box.h / 2.0f));
                    res.push_back(det);
                }
                return res;
            };

            // 先试 Anchor4 (与 can/src SetBoxInfo 一致)，无结果再回退 FourAnchor
            results = RunDetectionByCoord(CoordOrder::Anchor4);
            if (results.empty()) {
                YOLOV4_CERR << "[INFO][YOLOv4] Anchor4 无检测结果，尝试 FourAnchor 坐标排列" << std::endl;
                auto r2 = RunDetectionByCoord(CoordOrder::FourAnchor);
                results.swap(r2);
            }
            parsed = true;
        }
    } else if (output_num >= 2) {
        // --- 格式 B：带 box_num 的检测输出（兼容旧模型） ---
        YOLOV4_CERR << "[WARN][YOLOv4] 未匹配格式 A，尝试按格式 B (box_num + box_data) 解析" << std::endl;

        // 找到 size 比较小的那个，当 box_num int32 数组
        int idx_box_num = (outputs[0].size_bytes <= outputs[1].size_bytes) ? 0 : 1;
        int idx_box_data = 1 - idx_box_num;
        const int32_t* bn_host = static_cast<const int32_t*>(outputs[idx_box_num].host_addr);
        const float*   bd_host = static_cast<const float*>  (outputs[idx_box_data].host_addr);
        size_t bd_floats = outputs[idx_box_data].size_bytes / sizeof(float);
        int box_num = bn_host[0];

        // 调试预览
        int preview = std::min<int>(10, static_cast<int>(outputs[idx_box_num].size_bytes / sizeof(int32_t)));
        YOLOV4_CERR << "[DEBUG][YOLOv4] output[" << idx_box_num << "] (box_num候选) 前" << preview << "个int32: ";
        for (int i = 0; i < preview; ++i) YOLOV4_CERR << bn_host[i] << ",";
        YOLOV4_CERR << std::endl;
        preview = std::min<int>(30, static_cast<int>(bd_floats));
        YOLOV4_CERR << "[DEBUG][YOLOv4] output[" << idx_box_data << "] (box_data候选) 前" << preview << "个float: ";
        for (int i = 0; i < preview; ++i) {
            char buf[32]; snprintf(buf, sizeof(buf), "%.3f,", bd_host[i]);
            YOLOV4_CERR << buf;
        }
        YOLOV4_CERR << std::endl;

        if (box_num <= 0) {
            YOLOV4_CERR << "[WARN][YOLOv4] box_num = " << box_num << "，没有检测到物体" << std::endl;
        } else {
            YOLOV4_CERR << "[INFO][YOLOv4] 格式 B 解析: box_num=" << box_num
                      << ", box_data total floats=" << bd_floats
                      << ", expect per-box=6 floats (x1,y1,x2,y2,score,cls)" << std::endl;
            std::vector<BBoxRaw> all_boxes;
            for (int i = 0; i < box_num && i < static_cast<int>(kModelOutputBoxNum); ++i) {
                size_t base = static_cast<size_t>(i) * 6;
                if (base + 6 > bd_floats) {
                    YOLOV4_CERR << "[WARN][YOLOv4] box[" << i << "] 越界，终止" << std::endl;
                    break;
                }
                float x1 = bd_host[base + 0];
                float y1 = bd_host[base + 1];
                float x2 = bd_host[base + 2];
                float y2 = bd_host[base + 3];
                float sc = bd_host[base + 4];
                size_t ci = static_cast<size_t>(bd_host[base + 5]);
                if (sc < kScoreThreshold) continue;

                BBoxRaw box;
                box.x = (x1 + x2) / 2.0f * x_scale;    // cx
                box.y = (y1 + y2) / 2.0f * y_scale;    // cy
                box.w = (x2 - x1)        * x_scale;    // w
                box.h = (y2 - y1)        * y_scale;    // h
                box.score = sc;
                box.class_index = ci;
                all_boxes.push_back(box);
            }
            std::vector<BBoxRaw> nms_result;
            NMS(all_boxes, nms_result);
            for (const auto& box : nms_result) {
                DetectionResult det;
                det.class_id = static_cast<int>(box.class_index);
                det.label = GetCocoLabel(det.class_id);
                det.confidence = box.score;
                det.bbox.x1 = static_cast<int>(std::max(0.0f,            box.x - box.w / 2.0f));
                det.bbox.y1 = static_cast<int>(std::max(0.0f,            box.y - box.h / 2.0f));
                det.bbox.x2 = static_cast<int>(std::min(static_cast<float>(orig_w), box.x + box.w / 2.0f));
                det.bbox.y2 = static_cast<int>(std::min(static_cast<float>(orig_h), box.y + box.h / 2.0f));
                results.push_back(det);
            }
        }
        parsed = true;
    } else {
        std::cerr << "[ERROR][YOLOv4] 无法识别的输出格式: output_num=" << output_num << std::endl;
    }

    // ================ 第四步：释放 Host 缓冲 ================
    for (auto& o : outputs) {
        if (o.host_addr) {
            aclrtFreeHost(o.host_addr);
            o.host_addr = nullptr;
        }
    }

    if (parsed) {
        YOLOV4_CERR << "[INFO][YOLOv4] 后处理完成, 最终检测结果: " << results.size() << " 个框" << std::endl;
    }
}

InferenceResult YOLOv4::Infer(const std::string& image_path) {
    InferenceResult result;
    result.model_name = GetModelName();
    result.model_type = GetModelType();
    result.infer_cost_ms = 0;

    if (!initialized_) {
        std::cerr << "[ERROR][YOLOv4] Infer 调用时模型未初始化" << std::endl;
        return result;
    }
    YOLOV4_CERR << "[INFO][YOLOv4] ====== 开始推理, 图片: " << image_path << " ======" << std::endl;

    // --- 先查模型真实输入属性，决定走 DVPP(YUV) 还是 CPU(BGR FP32/UINT8) 预处理路径 ---
    size_t  real_input_size   = aclmdlGetInputSizeByIndex(model_desc_, 0);
    aclDataType real_input_dt = aclmdlGetInputDataType(model_desc_, 0);
    int mw = (int)model_input_width_;
    int mh = (int)model_input_height_;
    size_t yuv_nv12_size  = (size_t)AlignSize(mw,16) * (size_t)AlignSize(mh,2) * 3 / 2;   // DVPP 对齐后大小
    size_t expect_fp32_sz = (size_t)mw * (size_t)mh * 3 * sizeof(float);                  // NHWC FP32
    size_t expect_u8_sz   = (size_t)mw * (size_t)mh * 3 * sizeof(uint8_t);                // NHWC UINT8
    YOLOV4_CERR << "[INFO][YOLOv4] 模型真实输入 size=" << real_input_size
              << " bytes (DVPP YUV=" << yuv_nv12_size
              << ", FP32 NHWC=" << expect_fp32_sz
              << ", UINT8 NHWC=" << expect_u8_sz << ")" << std::endl;

    void* dev_input_buf  = nullptr;   // 最终喂给模型的 Device input buffer
    size_t dev_input_sz  = 0;
    bool  input_from_cpu = false;     // true=需要推理完 aclrtFree 这个 dev 内存
    float x_scale = 1.0f, y_scale = 1.0f;
    int orig_w = 0, orig_h = 0;

    // 【自动判断路径】真实 size 与 DVPP YUV 差异 > 1% 就走 CPU 预处理
    bool size_match_dvpp = (real_input_size > 0 && std::abs((long long)real_input_size - (long long)yuv_nv12_size)
                                                  <= (long long)(yuv_nv12_size / 100));
    // ★ 临时强制走 CPU 预处理：先绕开 DVPP 内存域相关的坑，保证先能出检测结果
    //    CPU 路径会在下面根据 real_input_size 判断是否需要手动转 NV12（AIPP 要求的格式）
    size_match_dvpp = false;
    if (size_match_dvpp) {
        // ====== 路径 A：DVPP (YUV NV12) ======
        YOLOV4_CERR << "[INFO][YOLOv4] 输入格式匹配 YUV NV12，使用 DVPP 预处理" << std::endl;
        void* dvpp_output = nullptr;
        int dvpp_output_size = 0;
        if (!ProcessDvpp(image_path, dvpp_output, dvpp_output_size, x_scale, y_scale, orig_w, orig_h)) {
            std::cerr << "[ERROR][YOLOv4] ProcessDvpp 预处理失败" << std::endl;
            return result;
        }
        // ★ 致命修复：acldvppMalloc 分配的是 DVPP 专用内存，不允许直接作为模型计算域的输入！
        // 必须先 aclrtMalloc 分配一块 Device 计算域内存，再通过 D→D 拷贝把 DVPP 结果拷过去。
        dev_input_sz = std::max<size_t>(real_input_size, (size_t)dvpp_output_size);
        aclError ret = aclrtMalloc(&dev_input_buf, dev_input_sz, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            std::cerr << "[ERROR][YOLOv4][DVPP] aclrtMalloc 输入 buffer 失败, size=" << dev_input_sz
                      << ", ret=" << ret << std::endl;
            return result;
        }
        size_t copy_sz = std::min<size_t>((size_t)dvpp_output_size, dev_input_sz);
        ret = aclrtMemcpy(dev_input_buf, copy_sz, dvpp_output, copy_sz, ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) {
            std::cerr << "[ERROR][YOLOv4][DVPP] aclrtMemcpy D→D (DVPP→ModelInput) 失败, ret=" << ret << std::endl;
            aclrtFree(dev_input_buf);
            dev_input_buf = nullptr;
            return result;
        }
        input_from_cpu = true;  // 因为是用 aclrtMalloc 新分配的，推理结束后需要 aclrtFree
        YOLOV4_CERR << "[INFO][YOLOv4][DVPP] DVPP→Model D→D 拷贝完成, copy_size=" << copy_sz
                  << " / dev_input_buf_size=" << dev_input_sz << std::endl;
    } else {
        // ====== 路径 B：CPU 预处理 (OpenCV BGR → 模型期望 dtype/format) ======
        YOLOV4_CERR << "[WARN][YOLOv4] 真实输入 size 与 DVPP YUV 不匹配，使用 CPU 预处理回退" << std::endl;
        input_from_cpu = true;

        // 1) OpenCV 读图（BGR UINT8），记录原图 w/h
        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            std::cerr << "[ERROR][YOLOv4][CPU] imread 读取图片失败: " << image_path << std::endl;
            return result;
        }
        orig_w = img.cols;
        orig_h = img.rows;
        x_scale = (float)orig_w / (float)mw;
        y_scale = (float)orig_h / (float)mh;
        YOLOV4_CERR << "[INFO][YOLOv4][CPU] 原图 " << orig_w << "x" << orig_h
                  << " → resize 到模型 " << mw << "x" << mh
                  << ", scale(x=" << x_scale << ", y=" << y_scale << ")" << std::endl;

        // 2) Resize 到模型输入尺寸
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(mw, mh), 0, 0, cv::INTER_LINEAR);

        // 3) 根据真实模型输入尺寸/格式生成 host buffer
        //    优先级 1：如果 real_input_size == yuv_nv12_size（模型带 AIPP 输入为 NV12）
        //              → CPU 手动做 BGR→YUV_I420→交错 U/V 拼 NV12，保证 AIPP 吃到正确格式
        //    优先级 2：否则按 real_input_dt（FP32/UINT8 NHWC）生成 BGR 数据
        std::vector<uint8_t> host_u8;
        std::vector<float>   host_fp32;
        void*  host_ptr = nullptr;
        size_t host_sz  = 0;
        if (real_input_size == yuv_nv12_size) {
            // ========== 模型 AIPP 要求 NV12 输入：CPU 手动拼 NV12 ==========
            YOLOV4_CERR << "[INFO][YOLOv4][CPU] 模型输入 size=NV12(" << yuv_nv12_size
                      << " bytes)，CPU 将 BGR 转为 NV12" << std::endl;
            host_u8.resize(yuv_nv12_size);
            uint8_t* dst_y  = host_u8.data();                           // Y 平面
            uint8_t* dst_uv = host_u8.data() + (size_t)mw * (size_t)mh; // UV 半平面（交错）

            // Step 1: BGR → YUV I420 (Y + U + V 三平面)
            cv::Mat yuv_i420;
            cv::cvtColor(resized, yuv_i420, cv::COLOR_BGR2YUV_I420);
            // I420 memory layout: [Y: w*h][U: w*h/4][V: w*h/4]
            const uint8_t* src_y = yuv_i420.ptr<uint8_t>(0);
            const uint8_t* src_u = src_y + (size_t)mw * (size_t)mh;
            const uint8_t* src_v = src_u + ((size_t)mw * (size_t)mh) / 4;

            // Step 2: copy Y plane (1:1)
            size_t y_sz = (size_t)mw * (size_t)mh;
            std::memcpy(dst_y, src_y, y_sz);

            // Step 3: interleave U/V → UV semi-planar (NV12)
            size_t uv_half = ((size_t)mw / 2) * ((size_t)mh / 2);
            for (size_t k = 0; k < uv_half; ++k) {
                dst_uv[k * 2 + 0] = src_u[k];
                dst_uv[k * 2 + 1] = src_v[k];
            }

            host_ptr = host_u8.data();
            host_sz  = host_u8.size() * sizeof(uint8_t);
            YOLOV4_CERR << "[INFO][YOLOv4][CPU] BGR→NV12 转换完成, host_sz=" << host_sz << std::endl;
        } else if (real_input_dt == ACL_FLOAT /* 0 */) {
            // FP32 NHWC：BGR U8 → BGR FP32；255 归一化 → [0,1]
            host_fp32.resize((size_t)mw * (size_t)mh * 3);
            float* dst = host_fp32.data();
            for (int y = 0; y < mh; ++y) {
                const uint8_t* row = resized.ptr<uint8_t>(y);
                for (int x = 0; x < mw; ++x) {
                    size_t base = (size_t)(y * mw + x) * 3;
                    dst[base + 0] = (float)row[x * 3 + 0] / 255.0f;  // B
                    dst[base + 1] = (float)row[x * 3 + 1] / 255.0f;  // G
                    dst[base + 2] = (float)row[x * 3 + 2] / 255.0f;  // R
                }
            }
            host_ptr = host_fp32.data();
            host_sz  = host_fp32.size() * sizeof(float);
        } else {
            // 默认 UINT8：直接 BGR U8（多数不带 AIPP / 要求 RAW BGR 输入的 om）
            host_u8.assign(resized.data, resized.data + resized.total() * 3);
            host_ptr = host_u8.data();
            host_sz  = host_u8.size() * sizeof(uint8_t);
        }
        // 预览：host buffer 前 30 字节
        YOLOV4_CERR << "[DEBUG][YOLOv4][CPU] 预处理后 host buffer 前 30 字节: ";
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(host_ptr);
        for (int i = 0; i < 30 && i < (int)host_sz; ++i) {
            char b[8]; std::snprintf(b, sizeof(b), "%02X ", pb[i]);
            YOLOV4_CERR << b;
        }
        YOLOV4_CERR << std::endl;

        // 4) 分配 Device 内存 + H2D 拷贝；模型要求的 size 与实际准备的 size 取 max
        dev_input_sz = std::max<size_t>(real_input_size, host_sz);
        aclError ret = aclrtMalloc(&dev_input_buf, dev_input_sz, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            std::cerr << "[ERROR][YOLOv4][CPU] aclrtMalloc 输入 buffer 失败, size=" << dev_input_sz
                      << ", ret=" << ret << std::endl;
            return result;
        }
        ret = aclrtMemcpy(dev_input_buf, host_sz, host_ptr, host_sz, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) {
            std::cerr << "[ERROR][YOLOv4][CPU] aclrtMemcpy H→D 失败, ret=" << ret << std::endl;
            aclrtFree(dev_input_buf);
            return result;
        }
        YOLOV4_CERR << "[INFO][YOLOv4][CPU] CPU 预处理完成, 输入 size=" << dev_input_sz << " bytes, 已 H→D 拷贝" << std::endl;
    }

    if (!CreateModelInput(dev_input_buf, dev_input_sz)) {
        std::cerr << "[ERROR][YOLOv4] CreateModelInput 失败" << std::endl;
        if (input_from_cpu) aclrtFree(dev_input_buf);
        return result;
    }

    if (!CreateModelOutput()) {
        std::cerr << "[ERROR][YOLOv4] CreateModelOutput 失败" << std::endl;
        DestroyModelInput();
        if (input_from_cpu) aclrtFree(dev_input_buf);
        return result;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    if (!ExecuteModel()) {
        std::cerr << "[ERROR][YOLOv4] ExecuteModel 执行失败" << std::endl;
        DestroyModelOutput();
        DestroyModelInput();
        if (input_from_cpu) aclrtFree(dev_input_buf);
        return result;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    result.infer_cost_ms = static_cast<int>(std::max<long long>(1LL, std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()));
    YOLOV4_CERR << "[INFO][YOLOv4] 纯模型执行耗时: " << result.infer_cost_ms << "ms" << std::endl;

    ProcessModelOutput(output_, x_scale, y_scale, orig_w, orig_h, result.detections);
    YOLOV4_CERR << "[INFO][YOLOv4] 后处理完成, 最终检测结果: " << result.detections.size() << " 个框" << std::endl;

    DestroyModelOutput();
    DestroyModelInput();
    if (input_from_cpu) aclrtFree(dev_input_buf);
    // ★ 修复：DVPP 的 resize_out_buffer_dev_ 在 Finalize 中统一释放，这里不释放（避免重复释放/下一次推理找不到 buffer）

    return result;
}

} // namespace kzzk
