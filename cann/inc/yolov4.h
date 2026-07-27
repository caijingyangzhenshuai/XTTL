#ifndef YOLOV4_H
#define YOLOV4_H

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <fstream>
#include <cstring>
#include <map>

#if defined(_MSC_VER)
#include <windows.h>
#else
#include <sys/stat.h>
#endif

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/types_c.h"

#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...) fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) fprintf(stderr, "[ERROR]  " fmt "\n", ##__VA_ARGS__)

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

typedef struct PicDesc {
    std::string picName;
    uint32_t width;
    uint32_t height;
    uint32_t jpegDecodeSize;
} PicDesc;

typedef struct BBox {
    float x;
    float y;
    float w;
    float h;
    float score;
    
    size_t classIndex;
    size_t index;
} BBox;

// YOLOv4类别名称列表（COCO数据集80个类别）
const std::vector<std::string> YOLOV4_LABELS = {
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

class YOLOv4 {
public:
    YOLOv4();
    virtual ~YOLOv4();
    Result Initialize(const char* modelPath, int deviceId = 0);
    void Finalize();
    Result Detect(const char* imagePath);
    std::vector<std::vector<float>> GetRawDetectionResults();
    std::vector<std::vector<int>> kzzk_cv(
        const std::string& modelFile,
        const std::string& imageFile
    );
    std::vector<std::vector<int>> kzzk_npu_little(
        const std::string& modelFile,
        const std::string& imageFile
    );
    void SaveResultsToFile(const std::string& imagePath);

private:
    static uint32_t AlignSize(uint32_t origSize, uint32_t alignment);
    Result ReadBinFile(PicDesc &picDesc, void *&inputBuff, uint32_t &fileSize);
    Result CheckPathIsFile(const std::string &fileName);
    Result GetDeviceBufferOfPicture(PicDesc &picDesc, void *&picDevBuffer, uint32_t &devPicBufferSize);
    Result InitDvppResource();
    Result InitDvppOutputPara(int modelInputWidth, int modelInputHeight);
    void SetDvppInput(void *inDevBuffer, uint32_t inDevBufferSize, const PicDesc &picDesc);
    void GetDvppOutput(void **outputBuffer, int &outputSize);
    Result ProcessDvpp();
    Result InitDecodeOutputDesc();
    Result ProcessDecode();
    void DestroyDecodeResource();
    Result InitResizeInputDesc();
    Result InitResizeOutputDesc();
    Result ProcessResize();
    void DestroyResizeResource();
    void DestroyDvppResource();
    void DestroyDvppOutputPara();
    Result LoadModel(const char *modelPath);
    void UnloadModel();
    Result CreateModelInput(void *inputDataBuffer, size_t bufferSize);
    void DestroyModelInput();
    Result CreateModelOutput();
    void DestroyModelOutput();
    Result ExecuteModel();
    Result GetModelInputWH(int &width, int &height);
    void ProcessModelOutput(const aclmdlDataset *output, const char *originImage, float xScale, float yScale);
    void SetBoxInfo(size_t index, BBox &box);
    static bool SortScore(BBox box1, BBox box2);
    static float IOU(const BBox &b1, const BBox &b2);
    void NMS(std::vector <BBox> &boxes, std::vector <BBox> &result);
    void DrawBoundBoxToImage(const std::vector <BBox> &result, const std::string &originImage);

private:
    // 原始检测结果
    std::vector<std::vector<float>> rawResults_;

    // 全局变量
    bool isDevice_;
    bool isInitialized_;
    bool isModelLoaded_;
    int deviceId_;
    aclrtContext context_;
    aclrtStream stream_;

    // 模型处理变量
    uint32_t modelId_;
    aclmdlDesc *modelDesc_;
    aclmdlDataset *input_;
    aclmdlDataset *output_;

    // DVPP处理变量
    acldvppChannelDesc *dvppChannelDesc_;
    acldvppResizeConfig *resizeConfig_;
    void *decodeOutDevBuffer_;
    acldvppPicDesc *decodeOutputDesc_;
    acldvppPicDesc *resizeInputDesc_;
    acldvppPicDesc *resizeOutputDesc_;
    void *inDevBuffer_;
    uint32_t inDevBufferSize_;
    uint32_t jpegDecodeOutputSize_;
    uint32_t decodeOutputWidth_;
    uint32_t decodeOutputWidthStride_;
    uint32_t decodeOutputHeight_;
    void *resizeOutBufferDev_;
    uint32_t resizeOutBufferSize_;
    uint32_t modelInputWidth_;
    uint32_t modelInputHeight_;
    uint32_t resizeOutWidthStride_;
    uint32_t resizeOutHeightStride_;

    // 后处理变量
    void *outputClass_;
    void *outputBox_;
    float xScale_;
    float yScale_;

    static const std::vector <std::string> g_labels;
    static const uint32_t g_lineSolid;
    static const double g_fontScale;
    static const cv::Scalar g_fontColor;
    static const uint32_t g_labelOffset;
    static const size_t g_classNum;
    static const size_t g_modelOutputBoxNum;
    static const float g_nmsThreshold;
    static const float g_scoreThreshold;
    static const std::vector <cv::Scalar> g_colors;
};

#endif // YOLOV4_H

