#include "yolov4.h"
#include <algorithm>
#include <memory>
#include <sys/time.h>
#include <chrono>
#include <iomanip>

using namespace std;

// 静态成员初始化
const std::vector <std::string> YOLOv4::g_labels = YOLOV4_LABELS;

const uint32_t YOLOv4::g_lineSolid = 2;
const double YOLOv4::g_fontScale = 0.5;
const cv::Scalar YOLOv4::g_fontColor(0, 0, 255);
const uint32_t YOLOv4::g_labelOffset = 11;
const size_t YOLOv4::g_classNum = 80;
const size_t YOLOv4::g_modelOutputBoxNum = 10647;
const float YOLOv4::g_nmsThreshold = 0.8;
const float YOLOv4::g_scoreThreshold = 0.4;
const std::vector <cv::Scalar> YOLOv4::g_colors = {
        cv::Scalar(237, 149, 100), cv::Scalar(0, 215, 255), cv::Scalar(50, 205, 50),
        cv::Scalar(139, 85, 26)};

// 智能指针缓冲区删除器
void BufferDeleter(void *p) {
    if (p != nullptr) {
        (void) aclrtFreeHost(p);
    }
}

YOLOv4::YOLOv4()
    : isDevice_(false), isInitialized_(false), isModelLoaded_(false),
      deviceId_(-1), context_(nullptr), stream_(nullptr),
      modelId_(0), modelDesc_(nullptr), input_(nullptr), output_(nullptr),
      dvppChannelDesc_(nullptr), resizeConfig_(nullptr), decodeOutDevBuffer_(nullptr),
      decodeOutputDesc_(nullptr), resizeInputDesc_(nullptr), resizeOutputDesc_(nullptr),
      inDevBuffer_(nullptr), inDevBufferSize_(0), jpegDecodeOutputSize_(0),
      decodeOutputWidth_(0), decodeOutputWidthStride_(0), decodeOutputHeight_(0),
      resizeOutBufferDev_(nullptr), resizeOutBufferSize_(0), modelInputWidth_(0),
      modelInputHeight_(0), resizeOutWidthStride_(0), resizeOutHeightStride_(0),
      outputClass_(nullptr), outputBox_(nullptr), xScale_(1.0f), yScale_(1.0f) {
}

YOLOv4::~YOLOv4() {
    Finalize();
}

uint32_t YOLOv4::AlignSize(uint32_t origSize, uint32_t alignment) {
    if (alignment == 0) {
        return 0;
    }
    uint32_t alignmentH = alignment - 1;
    return (origSize + alignmentH) / alignment * alignment;
}

Result YOLOv4::ReadBinFile(PicDesc &picDesc, void *&inputBuff, uint32_t &fileSize) {
    std::string fileName = picDesc.picName;
    if (CheckPathIsFile(fileName) == FAILED) {
        ERROR_LOG("%s is not a file", fileName.c_str());
        return FAILED;
    }

    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        ERROR_LOG("open file %s failed", fileName.c_str());
        return FAILED;
    }

    binFile.seekg(0, binFile.end);
    uint32_t binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        ERROR_LOG("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return FAILED;
    }
    binFile.seekg(0, binFile.beg);

    aclError ret;
    if (!isDevice_) { // 应用运行在主机侧
        ret = aclrtMallocHost(&inputBuff, binFileBufferLen);
        if (inputBuff == nullptr) {
            ERROR_LOG("malloc binFileBufferData failed. binFileBufferLen is %u, errorCode is %d",
                      binFileBufferLen, static_cast<int32_t>(ret));
            binFile.close();
            return FAILED;
        }
    } else { // 应用运行在设备侧
        ret = acldvppMalloc(&inputBuff, binFileBufferLen);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("malloc device buffer failed. size is %u, errorCode is %d",
                      binFileBufferLen, static_cast<int32_t>(ret));
            binFile.close();
            return FAILED;
        }
    }
    binFile.read(static_cast<char *>(inputBuff), binFileBufferLen);
    binFile.close();
    fileSize = binFileBufferLen;

    int32_t components = 0;
    ret = acldvppJpegGetImageInfo(inputBuff, binFileBufferLen, &picDesc.width, &picDesc.height, &components);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acldvppJpegGetImageInfo failed, errorCode is %d", static_cast<int32_t>(ret));
        return FAILED;
    }

    return SUCCESS;
}

Result YOLOv4::CheckPathIsFile(const std::string &fileName) {
#if defined(_MSC_VER)
    DWORD bRet = GetFileAttributes((LPCSTR)fileName.c_str());
    if (bRet == FILE_ATTRIBUTE_DIRECTORY) {
        ERROR_LOG("%s is not a file, please enter a file", fileName.c_str());
        return FAILED;
    }
#else
    struct stat sBuf;
    int fileStatus = stat(fileName.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file");
        return FAILED;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", fileName.c_str());
        return FAILED;
    }
#endif
    return SUCCESS;
}

Result YOLOv4::GetDeviceBufferOfPicture(PicDesc &picDesc, void *&picDevBuffer, uint32_t &devPicBufferSize) {
    if (picDesc.picName.empty()) {
        ERROR_LOG("picture file name is empty");
        return FAILED;
    }

    uint32_t inputBuffSize = 0;
    void *inputBuff = nullptr;
    Result ret = ReadBinFile(picDesc, inputBuff, inputBuffSize);
    if (ret != SUCCESS) {
        ERROR_LOG("read bin file failed, file name is %s", picDesc.picName.c_str());
        return FAILED;
    }
    aclError aclRet = acldvppJpegGetImageInfo(inputBuff, inputBuffSize, &picDesc.width, &picDesc.height, nullptr);
    if (aclRet != ACL_SUCCESS) {
        ERROR_LOG("get jpeg image info failed, errorCode is %d", static_cast<int32_t>(aclRet));
        if (!isDevice_) {
            (void) aclrtFreeHost(inputBuff);
        } else {
            (void) acldvppFree(inputBuff);
        }
        return FAILED;
    }
    aclRet = acldvppJpegPredictDecSize(inputBuff, inputBuffSize, PIXEL_FORMAT_YUV_SEMIPLANAR_420,
                                       &picDesc.jpegDecodeSize);
    if (aclRet != ACL_SUCCESS) {
        ERROR_LOG("get jpeg decode size failed, errorCode is %d", static_cast<int32_t>(aclRet));
        if (!isDevice_) {
            (void) aclrtFreeHost(inputBuff);
        } else {
            (void) acldvppFree(inputBuff);
        }
        return FAILED;
    }

    if (!isDevice_) { // 应用运行在主机侧
        aclRet = acldvppMalloc(&picDevBuffer, inputBuffSize);
        if (aclRet != ACL_SUCCESS) {
            ERROR_LOG("malloc device buffer failed. size is %u, errorCode is %d",
                      inputBuffSize, static_cast<int32_t>(aclRet));
            (void) aclrtFreeHost(inputBuff);
            return FAILED;
        }

        // 如果应用运行在主机侧，需要将数据从主机复制到设备
        aclRet = aclrtMemcpy(picDevBuffer, inputBuffSize, inputBuff, inputBuffSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            ERROR_LOG("memcpy failed. device buffer size is %u, input host buffer size is %u, errorCode is %d",
                      inputBuffSize, inputBuffSize, static_cast<int32_t>(aclRet));
            (void) acldvppFree(picDevBuffer);
            (void) aclrtFreeHost(inputBuff);
            return FAILED;
        }
        (void) aclrtFreeHost(inputBuff);
    } else { // 应用运行在设备侧
        picDevBuffer = inputBuff;
    }
    devPicBufferSize = inputBuffSize;

    return SUCCESS;
}

Result YOLOv4::InitDvppResource() {
    dvppChannelDesc_ = acldvppCreateChannelDesc();
    if (dvppChannelDesc_ == nullptr) {
        ERROR_LOG("acldvppCreateChannelDesc failed");
        return FAILED;
    }

    aclError ret = acldvppCreateChannel(dvppChannelDesc_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acldvppCreateChannelAsync failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }

    resizeConfig_ = acldvppCreateResizeConfig();
    if (resizeConfig_ == nullptr) {
        ERROR_LOG("acldvppCreateResizeConfig failed");
        return FAILED;
    }

    return SUCCESS;
}

Result YOLOv4::InitDvppOutputPara(int modelInputWidth, int modelInputHeight) {
    if ((modelInputWidth <= 0) || (modelInputHeight <= 0)) {
        ERROR_LOG("InitInput para invalid, modelInputWidth %d, modelInputHeight %d",
                  modelInputWidth, modelInputHeight);
        return FAILED;
    }
    int byte1 = 16;
    int byte2 = 2;
    modelInputWidth_ = modelInputWidth;
    modelInputHeight_ = modelInputHeight;
    resizeOutWidthStride_ = AlignSize(modelInputWidth, byte1); // 16字节对齐
    resizeOutHeightStride_ = AlignSize(modelInputHeight, byte2); // 2字节对齐

    // 输出缓冲区，根据实际模型调整
    resizeOutBufferSize_ = resizeOutWidthStride_ * resizeOutHeightStride_ * 3 / 2; // yuv格式大小
    aclError ret = acldvppMalloc(&resizeOutBufferDev_, resizeOutBufferSize_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acldvppMalloc resizeOutBuffer failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }

    return SUCCESS;
}

void YOLOv4::SetDvppInput(void *inDevBuffer, uint32_t inDevBufferSize, const PicDesc &picDesc) {
    inDevBuffer_ = inDevBuffer;
    inDevBufferSize_ = inDevBufferSize;
    jpegDecodeOutputSize_ = picDesc.jpegDecodeSize;
}

void YOLOv4::GetDvppOutput(void **outputBuffer, int &outputSize) {
    if (outputBuffer == nullptr) {
        ERROR_LOG("outputBuffer is nullptr");
        return;
    }
    *outputBuffer = resizeOutBufferDev_;
    outputSize = resizeOutBufferSize_;
    resizeOutBufferDev_ = nullptr;
    resizeOutBufferSize_ = 0;
}

Result YOLOv4::InitDecodeOutputDesc() {
    aclError ret = acldvppMalloc(&decodeOutDevBuffer_, jpegDecodeOutputSize_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acldvppMalloc jpegOutBufferDev failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }

    decodeOutputDesc_ = acldvppCreatePicDesc();
    if (decodeOutputDesc_ == nullptr) {
        ERROR_LOG("acldvppCreatePicDesc decodeOutputDesc failed");
        return FAILED;
    }

    (void) acldvppSetPicDescData(decodeOutputDesc_, decodeOutDevBuffer_);
    (void) acldvppSetPicDescFormat(decodeOutputDesc_, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    (void) acldvppSetPicDescSize(decodeOutputDesc_, jpegDecodeOutputSize_);
    return SUCCESS;
}

Result YOLOv4::ProcessDecode() {
    // 解码为yuv格式
    aclError ret = acldvppJpegDecodeAsync(dvppChannelDesc_, inDevBuffer_, inDevBufferSize_,
                                          decodeOutputDesc_, stream_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acldvppJpegDecodeAsync failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }

    ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtSynchronizeStream failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }
    // 获取yuv图像宽度和高度
    decodeOutputWidth_ = acldvppGetPicDescWidth(decodeOutputDesc_);
    decodeOutputHeight_ = acldvppGetPicDescHeight(decodeOutputDesc_);
    decodeOutputWidthStride_ = acldvppGetPicDescWidthStride(decodeOutputDesc_);
    return SUCCESS;
}

void YOLOv4::DestroyDecodeResource() {
    if (decodeOutputDesc_ != nullptr) {
        (void) acldvppDestroyPicDesc(decodeOutputDesc_);
        decodeOutputDesc_ = nullptr;
    }
}

Result YOLOv4::InitResizeInputDesc() {
    uint32_t jpegOutWidthStride = decodeOutputWidthStride_; // 310上128字节对齐，310P上64字节对齐
    uint32_t jpegOutHeightStride = AlignSize(decodeOutputHeight_, 16); // 16字节对齐
    uint32_t jpegOutBufferSize = jpegOutWidthStride * jpegOutHeightStride * 3 / 2; // yuv格式大小
    resizeInputDesc_ = acldvppCreatePicDesc();
    if (resizeInputDesc_ == nullptr) {
        ERROR_LOG("InitResizeInputDesc failed");
        return FAILED;
    }

    (void) acldvppSetPicDescData(resizeInputDesc_, decodeOutDevBuffer_);
    (void) acldvppSetPicDescFormat(resizeInputDesc_, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    (void) acldvppSetPicDescWidth(resizeInputDesc_, decodeOutputWidth_);
    (void) acldvppSetPicDescHeight(resizeInputDesc_, decodeOutputHeight_);
    (void) acldvppSetPicDescWidthStride(resizeInputDesc_, jpegOutWidthStride);
    (void) acldvppSetPicDescHeightStride(resizeInputDesc_, jpegOutHeightStride);
    (void) acldvppSetPicDescSize(resizeInputDesc_, jpegOutBufferSize);
    return SUCCESS;
}

Result YOLOv4::InitResizeOutputDesc() {
    resizeOutputDesc_ = acldvppCreatePicDesc();
    if (resizeOutputDesc_ == nullptr) {
        ERROR_LOG("acldvppCreatePicDesc failed");
        return FAILED;
    }

    (void) acldvppSetPicDescData(resizeOutputDesc_, resizeOutBufferDev_);
    (void) acldvppSetPicDescFormat(resizeOutputDesc_, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    (void) acldvppSetPicDescWidth(resizeOutputDesc_, modelInputWidth_);
    (void) acldvppSetPicDescHeight(resizeOutputDesc_, modelInputHeight_);
    (void) acldvppSetPicDescWidthStride(resizeOutputDesc_, resizeOutWidthStride_);
    (void) acldvppSetPicDescHeightStride(resizeOutputDesc_, resizeOutHeightStride_);
    (void) acldvppSetPicDescSize(resizeOutputDesc_, resizeOutBufferSize_);
    return SUCCESS;
}

Result YOLOv4::ProcessResize() {
    // 调整图像尺寸
    aclError ret = acldvppVpcResizeAsync(dvppChannelDesc_, resizeInputDesc_,
                                         resizeOutputDesc_, resizeConfig_, stream_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acldvppVpcResizeAsync failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }

    ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtSynchronizeStream failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }

    return SUCCESS;
}

void YOLOv4::DestroyResizeResource() {
    if (decodeOutDevBuffer_ != nullptr) {
        (void) acldvppFree(decodeOutDevBuffer_);
        decodeOutDevBuffer_ = nullptr;
    }

    if (resizeInputDesc_ != nullptr) {
        acldvppDestroyPicDesc(resizeInputDesc_);
        resizeInputDesc_ = nullptr;
    }

    if (resizeOutputDesc_ != nullptr) {
        acldvppDestroyPicDesc(resizeOutputDesc_);
        resizeOutputDesc_ = nullptr;
    }
}

void YOLOv4::DestroyDvppResource() {
    // resizeConfig_在initResource中创建
    if (resizeConfig_ != nullptr) {
        acldvppDestroyResizeConfig(resizeConfig_);
        resizeConfig_ = nullptr;
    }

    if (dvppChannelDesc_ != nullptr) {
        aclError ret = acldvppDestroyChannel(dvppChannelDesc_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("acldvppDestroyChannel failed, errorCode = %d", static_cast<int32_t>(ret));
        }

        (void) acldvppDestroyChannelDesc(dvppChannelDesc_);
        dvppChannelDesc_ = nullptr;
    }
}

void YOLOv4::DestroyDvppOutputPara() {
    if (resizeOutBufferDev_ != nullptr) {
        (void) acldvppFree(resizeOutBufferDev_);
        resizeOutBufferDev_ = nullptr;
    }
}

Result YOLOv4::ProcessDvpp() {
    // 图片解码
    Result ret = InitDecodeOutputDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("InitDecodeOutputDesc failed");
        DestroyDecodeResource();
        return FAILED;
    }

    ret = ProcessDecode();
    if (ret != SUCCESS) {
        ERROR_LOG("ProcessDecode failed");
        DestroyDecodeResource();
        return FAILED;
    }

    DestroyDecodeResource();

    // 图片缩放
    ret = InitResizeInputDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("InitResizeInputDesc failed");
        DestroyResizeResource();
        return FAILED;
    }

    ret = InitResizeOutputDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("InitResizeOutputDesc failed");
        DestroyResizeResource();
        return FAILED;
    }

    ret = ProcessResize();
    if (ret != SUCCESS) {
        ERROR_LOG("ProcessResize failed");
        DestroyResizeResource();
        return FAILED;
    }

    DestroyResizeResource();

    return SUCCESS;
}

Result YOLOv4::Initialize(const char* modelPath, int deviceId) {
    // ACL初始化
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclInit failed, errorCode is %d", static_cast<int32_t>(ret));
        return FAILED;
    }
    isInitialized_ = true;

    // 打开设备
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtSetDevice failed, errorCode is %d", static_cast<int32_t>(ret));
        return FAILED;
    }

    deviceId_ = deviceId;

    // 创建上下文
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtCreateContext failed, errorCode is %d", static_cast<int32_t>(ret));
        return FAILED;
    }

    // 创建流
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtCreateStream failed, errorCode is %d", static_cast<int32_t>(ret));
        return FAILED;
    }

    // 检查运行模式
    aclrtRunMode runMode;
    ret = aclrtGetRunMode(&runMode);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtGetRunMode failed, errorCode is %d", static_cast<int32_t>(ret));
        return FAILED;
    }
    isDevice_ = (runMode == ACL_DEVICE);

    // 加载模型
    ret = LoadModel(modelPath);
    if (ret != SUCCESS) {
        ERROR_LOG("LoadModel failed");
        return FAILED;
    }

    return SUCCESS;
}

void YOLOv4::Finalize() {
    // 销毁DVPP资源
    DestroyDvppResource();
    DestroyDvppOutputPara();

    // 销毁模型资源
    UnloadModel();
    DestroyModelInput();
    DestroyModelOutput();

    // 销毁流和上下文
    if (stream_ != nullptr) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }

    if (context_ != nullptr) {
        aclrtDestroyContext(context_);
        context_ = nullptr;
    }

    // 重置设备
    if (deviceId_ >= 0) {
        (void) aclrtResetDevice(deviceId_);
        deviceId_ = -1;
    }

    // 清理ACL
    if (isInitialized_) {
        (void) aclFinalize();
        isInitialized_ = false;
    }

}

Result YOLOv4::LoadModel(const char *modelPath) {
    if (isModelLoaded_) {
        ERROR_LOG("model has already been loaded");
        return FAILED;
    }

    aclError ret = aclmdlLoadFromFile(modelPath, &modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("load model from file failed, model file is %s, errorCode is %d",
                  modelPath, static_cast<int32_t>(ret));
        return FAILED;
    }

    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        (void) aclmdlUnload(modelId_);
        return FAILED;
    }

    ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("get model description failed");
        (void) aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
        (void) aclmdlUnload(modelId_);
        return FAILED;
    }

    isModelLoaded_ = true;

    return SUCCESS;
}

void YOLOv4::UnloadModel() {
    if (!isModelLoaded_) {
        return;
    }

    aclError ret = aclmdlUnload(modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("unload model failed, modelId is %u, errorCode is %d", modelId_, static_cast<int32_t>(ret));
    }

    if (modelDesc_ != nullptr) {
        (void) aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }

    isModelLoaded_ = false;

}

Result YOLOv4::CreateModelInput(void *inputDataBuffer, size_t bufferSize) {
    // 本示例中的OM模型只有一个输入
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        (void) acldvppFree(inputDataBuffer);
        return FAILED;
    }
    size_t modelInputSize = aclmdlGetInputSizeByIndex(modelDesc_, 0);
    if (bufferSize != modelInputSize) {
        ERROR_LOG("input image size[%zu] is not equal to model input size[%zu]", bufferSize, modelInputSize);
        (void) acldvppFree(inputDataBuffer);
        return FAILED;
    }

    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        (void) acldvppFree(inputDataBuffer);
        return FAILED;
    }

    aclDataBuffer *inputData = aclCreateDataBuffer(inputDataBuffer, bufferSize);
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        (void) acldvppFree(inputDataBuffer);
        return FAILED;
    }

    aclError ret = aclmdlAddDatasetBuffer(input_, inputData);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("add input dataset buffer failed, errorCode is %d", static_cast<int32_t>(ret));
        (void) aclDestroyDataBuffer(inputData);
        inputData = nullptr;
        (void) acldvppFree(inputDataBuffer);
        return FAILED;
    }

    return SUCCESS;
}

void YOLOv4::DestroyModelInput() {
    if (input_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(input_, i);
        void *data = aclGetDataBufferAddr(dataBuffer);
        (void) acldvppFree(data);
        (void) aclDestroyDataBuffer(dataBuffer);
    }
    (void) aclmdlDestroyDataset(input_);
    input_ = nullptr;

}

Result YOLOv4::CreateModelOutput() {
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create output failed");
        return FAILED;
    }

    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }

    size_t outputSize = aclmdlGetNumOutputs(modelDesc_);
    for (size_t i = 0; i < outputSize; ++i) {
        size_t modelOutputSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);
        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, modelOutputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("can't malloc buffer, create output failed, size is %zu, errorCode is %d",
                      modelOutputSize, static_cast<int32_t>(ret));
            return FAILED;
        }

        aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, modelOutputSize);
        if (outputData == nullptr) {
            ERROR_LOG("can't create data buffer, create output failed");
            (void) aclrtFree(outputBuffer);
            return FAILED;
        }

        ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("can't add data buffer, create output failed, errorCode is %d", static_cast<int32_t>(ret));
            (void) aclrtFree(outputBuffer);
            (void) aclDestroyDataBuffer(outputData);
            return FAILED;
        }
    }

    return SUCCESS;
}

void YOLOv4::DestroyModelOutput() {
    if (output_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void *data = aclGetDataBufferAddr(dataBuffer);
        (void) aclrtFree(data);
        (void) aclDestroyDataBuffer(dataBuffer);
    }

    (void) aclmdlDestroyDataset(output_);
    output_ = nullptr;
    //INFO_LOG("destroy model output success");
}

Result YOLOv4::ExecuteModel() {
    aclError ret = aclmdlExecute(modelId_, input_, output_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("execute model failed, modelId is %u, errorCode is %d", modelId_, static_cast<int32_t>(ret));
        return FAILED;
    }

    return SUCCESS;
}

Result YOLOv4::GetModelInputWH(int &width, int &height) {
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, get input hw failed");
        return FAILED;
    }
    // 本应用使用的OM格式为NHWC，dimsCount为4
    // dims[0]为N，dims[1]为H，dims[2]为W，dims[3]为C
    aclmdlIODims dims;
    // 本应用中的OM模型只有一个输入
    aclError ret = aclmdlGetInputDims(modelDesc_, 0, &dims);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("get model input dims failed, errorCode is %d", static_cast<int32_t>(ret));
        return FAILED;
    }
    if (dims.dimCount != 4) {
        ERROR_LOG("invalid dimsCount %zu, get input hw failed", dims.dimCount);
        return FAILED;
    }
    width = dims.dims[2];
    height = dims.dims[1];

    return SUCCESS;
}

void YOLOv4::ProcessModelOutput(const aclmdlDataset *output, const char *originImage, float xScale, float yScale) {
    FILE *outputFile = nullptr;
    shared_ptr<void> dataBuff[2] = {nullptr, nullptr};
    size_t index = 0;
    for (size_t i = 0; (i < aclmdlGetDatasetNumBuffers(output)) && (i < 2); ++i) {
        //std::string name = "../out/output/output_" + to_string(i) + ".bin";
        std::string name = "/home/HwHiAiUser/cann/data/out_ " + to_string(i) + ".bin";
        if (outputFile != nullptr) {
            fclose(outputFile);
        }
        outputFile = fopen(name.c_str(), "wb");
        // 获取模型输出数据
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(output, i);
        void *data = aclGetDataBufferAddr(dataBuffer);
        //size_t len = aclGetDataBufferSize(dataBuffer);
	size_t len = static_cast<size_t>(aclGetDataBufferSizeV2(dataBuffer));

        if (isDevice_) {
            fwrite(data, len, sizeof(char), outputFile);
            shared_ptr<void> ptr(data, BufferDeleter);
            dataBuff[i] = ptr;
        } else {
            void *outHostData = nullptr;
            aclError ret = aclrtMallocHost(&outHostData, len);
            if (ret != ACL_SUCCESS) {
                cout << "aclrtMallocHost failed, result code is " << ret << endl;
                break;
            }
            ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                cout << "aclrtMemcpy failed, result code is " << ret << endl;
                (void) aclrtFreeHost(outHostData);
                break;
            }
            fwrite(outHostData, len, sizeof(char), outputFile);
            shared_ptr<void> ptr(outHostData, BufferDeleter);
            dataBuff[i] = ptr;
        }
    }
    if (outputFile != nullptr) {
        fclose(outputFile);
    }

    // 后处理
    outputClass_ = dataBuff[0].get();
    outputBox_ = dataBuff[1].get();
    xScale_ = xScale;
    yScale_ = yScale;

    vector <BBox> boxes;
    float *classBuff = static_cast<float *>(outputClass_);
    for (size_t i = 0; i < g_modelOutputBoxNum; ++i) {
        float maxValue = 0;
        float maxIndex = 0;
        for (size_t j = 0; j < g_classNum; ++j) {
            float value = classBuff[i * g_classNum + j];
            if (value > maxValue) {
                maxIndex = j;
                maxValue = value;
            }
        }
        if (maxValue >= g_scoreThreshold) {
            BBox b;
            SetBoxInfo(i, b);
            b.score = maxValue;
            b.classIndex = maxIndex;
            b.index = i;
            if (maxIndex < g_classNum) {
                boxes.push_back(b);
            }
        }
    }

    vector <BBox> result;
    NMS(boxes, result);
    
    // 填充原始检测结果
    rawResults_.clear();
    for (const auto& box : result) {
        std::vector<float> detectionResult;
        detectionResult.push_back(static_cast<float>(box.classIndex));
        detectionResult.push_back(box.score);
        detectionResult.push_back(box.x);
        detectionResult.push_back(box.y);
        detectionResult.push_back(box.w);
        detectionResult.push_back(box.h);
        rawResults_.push_back(detectionResult);
    }
    
    DrawBoundBoxToImage(result, originImage);
}

void YOLOv4::SetBoxInfo(size_t index, BBox &box) {
    float *boxBuff = static_cast<float *>(outputBox_);
    boxBuff += (index * sizeof(float));
    box.x = boxBuff[0] * xScale_;
    box.y = boxBuff[1] * yScale_;
    box.w = boxBuff[2] * xScale_;
    box.h = boxBuff[3] * yScale_;
}

bool YOLOv4::SortScore(BBox box1, BBox box2) {
    return box1.score > box2.score;
}

float YOLOv4::IOU(const BBox &b1, const BBox &b2) {
    float x1 = max(b1.x, b2.x);
    float y1 = max(b1.y, b2.y);
    float x2 = min(b1.x + b1.w, b2.x + b2.w);
    float y2 = min(b1.y + b1.h, b2.y + b2.h);
    float w = max(0.0f, x2 - x1 + 1);
    float h = max(0.0f, y2 - y1 + 1);
    float area = w * h;
    return area / (b1.w * b1.h + b2.w * b2.h - area);
}

void YOLOv4::NMS(std::vector <BBox> &boxes, std::vector <BBox> &result) {
    result.clear();
    std::sort(boxes.begin(), boxes.end(), SortScore);

    while (boxes.size() != 0) {
        result.push_back(boxes[0]);
        size_t index = 1;
        while (boxes.size() > index) {
            float iou = IOU(boxes[0], boxes[index]);
            if (iou > g_nmsThreshold) {
                boxes.erase(boxes.begin() + index);
                continue;
            }
            ++index;
        }
        boxes.erase(boxes.begin());
    }
}

void YOLOv4::DrawBoundBoxToImage(const std::vector <BBox> &result, const std::string &originImage) {
    cv::Mat image = cv::imread(originImage, cv::IMREAD_UNCHANGED);
    int half = 2;
    for (size_t i = 0; i < result.size(); ++i) {
        cv::Point p1, p2;
        p1.x = result[i].x - result[i].w / half;
        p1.y = result[i].y - result[i].h / half;
        p2.x = result[i].x + result[i].w / half;
        p2.y = result[i].y + result[i].h / half;
        cv::rectangle(image, p1, p2, g_colors[i % g_colors.size()], g_lineSolid);
        string className = g_labels[result[i].classIndex];
        cv::putText(image, className, cv::Point(p1.x, p1.y + g_labelOffset),
                    cv::FONT_HERSHEY_COMPLEX, g_fontScale, g_fontColor);
    }

    size_t pos = originImage.find_last_of("/");
    string fileName(originImage.substr(pos + 1));
    stringstream sstream;
    sstream.str("");
    sstream << "/home/HwHiAiUser/cann/data/out_" << fileName;
    cv::imwrite(sstream.str(), image);
}

std::vector<std::vector<float>> YOLOv4::GetRawDetectionResults() {
    return rawResults_;
}

void YOLOv4::SaveResultsToFile(const std::string& imagePath) {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");

    std::string fileName = "/home/HwHiAiUser/cann/data/" + timestamp.str() + "_yolov4.txt";
    
    std::ofstream outFile(fileName);
    if (outFile.is_open()) {
        outFile << "==================================================" << std::endl;
        outFile << "CANN 模型推理结果" << std::endl;
        outFile << "==================================================" << std::endl;
        outFile << "模型类型: yolov4" << std::endl;
        outFile << "图像文件: " << imagePath << std::endl;
        outFile << "时间戳: " << timestamp.str() << std::endl;
        outFile << "结果数量: " << rawResults_.size() << std::endl;
        outFile << "==================================================" << std::endl;
        
        if (rawResults_.empty()) {
            outFile << "未检测到任何结果" << std::endl;
        } else {
            for (size_t i = 0; i < rawResults_.size(); ++i) {
                outFile << "检测结果 " << (i + 1) << ":" << std::endl;
                outFile << "  类别索引: " << rawResults_[i][0] << std::endl;
                outFile << "  置信度: " << rawResults_[i][1] << std::endl;
                outFile << "  中心点X: " << rawResults_[i][2] << std::endl;
                outFile << "  中心点Y: " << rawResults_[i][3] << std::endl;
                outFile << "  宽度: " << rawResults_[i][4] << std::endl;
                outFile << "  高度: " << rawResults_[i][5] << std::endl;
                
                size_t classIndex = static_cast<size_t>(rawResults_[i][0]);
                if (classIndex < g_labels.size()) {
                    outFile << "  类别名称: " << g_labels[classIndex] << std::endl;
                }
                
                outFile << std::endl;
            }
        }
        
        outFile << "==================================================" << std::endl;
        outFile.close();
        
    } else {
        ERROR_LOG("无法打开文件保存结果: %s", fileName.c_str());
    }
}

Result YOLOv4::Detect(const char* imagePath) {
    if (imagePath == nullptr) {
        ERROR_LOG("imagePath is nullptr");
        return FAILED;
    }

    // 获取模型输入宽度和高度
    int width = 0;
    int height = 0;
    Result ret = GetModelInputWH(width, height);
    if (ret != SUCCESS) {
        ERROR_LOG("GetModelInputWH failed");
        return FAILED;
    }

    // 获取图像缓冲区
    uint32_t devBufferSize = 0;
    void *devBuffer = nullptr;
    PicDesc testPic = {imagePath, 0, 0, 0};
    if (SUCCESS != GetDeviceBufferOfPicture(testPic, devBuffer, devBufferSize)) {
        ERROR_LOG("get image device buffer failed.");
        return FAILED;
    }
    //INFO_LOG("get device buffer for picture success.");

    // 计算缩放因子
    float xScale = static_cast<float>(testPic.width) / static_cast<float>(width);
    float yScale = static_cast<float>(testPic.height) / static_cast<float>(height);

    // 使用DVPP处理图像
    ret = InitDvppResource();
    if (ret != SUCCESS) {
        ERROR_LOG("dvpp process init failed.");
        (void) acldvppFree(devBuffer);
        return FAILED;
    }

    SetDvppInput(devBuffer, devBufferSize, testPic);
    if (SUCCESS != InitDvppOutputPara(width, height)) {
        ERROR_LOG("dvpp init output parameters failed.");
        (void) acldvppFree(devBuffer);
        return FAILED;
    }

    if (SUCCESS != ProcessDvpp()) {
        ERROR_LOG("dvpp process failed.");
        (void) acldvppFree(devBuffer);
        return FAILED;
    }

    //INFO_LOG("dvpp decode and resize success.");

    // 获取DVPP输出
    void *inputBuffer = nullptr;
    size_t inputLen = 0;
    int len = 0;
    GetDvppOutput(&inputBuffer, len);
    inputLen = static_cast<size_t>(len);

    (void) acldvppFree(devBuffer);
    devBuffer = nullptr;

    // 创建模型输入
    ret = CreateModelInput(inputBuffer, inputLen);
    if (ret != SUCCESS) {
        ERROR_LOG("CreateModelInput failed");
        return FAILED;
    }

    // 创建模型输出
    ret = CreateModelOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("CreateModelOutput failed");
        return FAILED;
    }

    // 执行模型
    ret = ExecuteModel();
    if (ret != SUCCESS) {
        ERROR_LOG("ExecuteModel failed");
        return FAILED;
    }

    // 处理模型输出
    ProcessModelOutput(output_, imagePath, xScale, yScale);

    // 保存检测结果到文件
    SaveResultsToFile(imagePath);

    // 销毁资源
    DestroyModelOutput();
    DestroyModelInput();

    //INFO_LOG("Detection completed successfully");
    return SUCCESS;
}


std::vector<std::vector<int>> YOLOv4::kzzk_cv(
    const std::string& modelFile,
    const std::string& imageFile
) {
    try {
        std::vector<std::vector<int>> result_kzzk_cv;


        // 初始化YOLOv4检测器
        Result ret = Initialize(modelFile.c_str());
        if (ret != SUCCESS) {
            ERROR_LOG("Failed to initialize YOLOv4 detector");
            return result_kzzk_cv;
        }
        struct timeval start, end; // 测量推理时间
        gettimeofday(&start, NULL);

        // 检测图像中的对象
        ret = Detect(imageFile.c_str());
        if (ret != SUCCESS) {
            ERROR_LOG("Failed to detect objects in the image");
            Finalize();
            return result_kzzk_cv;
        }
        gettimeofday(&end, NULL);
        long totalInferenceTime = ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) / 1000;
        cout << "Total inference cost: " << totalInferenceTime << "ms" << endl;

        // 获取原始检测结果
        std::vector<std::vector<float>> rawResults = GetRawDetectionResults();

        // 清理YOLOv4检测器
        Finalize();

        // 构建结果数组：[class_id, x1, y1, x2, y2, score*100]
        // 注意：rawResults 内部是 (cx, cy, w, h) 格式，需要转换为 xyxy
        for (const auto& result : rawResults) {
            if (result.size() >= 6) {
                std::vector<int> obj;
                float cx = result[2];
                float cy = result[3];
                float w = result[4];
                float h = result[5];
                obj.push_back(static_cast<int>(result[0]));          // class_id
                obj.push_back(static_cast<int>(std::round(cx - w / 2.0f)));  // x1 = cx - w/2
                obj.push_back(static_cast<int>(std::round(cy - h / 2.0f)));  // y1 = cy - h/2
                obj.push_back(static_cast<int>(std::round(cx + w / 2.0f)));  // x2 = cx + w/2
                obj.push_back(static_cast<int>(std::round(cy + h / 2.0f)));  // y2 = cy + h/2
                obj.push_back(static_cast<int>(std::round(result[1] * 100)));  // score*100
                result_kzzk_cv.push_back(obj);
            }
        }


        return result_kzzk_cv;
    } catch (const std::exception& e) {
        ERROR_LOG("[kzzk_cv] error: %s", e.what());
        return {};
    }
}
