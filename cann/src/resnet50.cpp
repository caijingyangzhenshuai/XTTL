#include "resnet50.h"
#include "acl/acl.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <map>
#include <algorithm>
#include <memory>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

namespace {

struct AclHostDeleter {
    void operator()(void* p) const {
        if (p) aclrtFreeHost(p);
    }
};

struct AclDeviceDeleter {
    void operator()(void* p) const {
        if (p) aclrtFree(p);
    }
};

struct AclDatasetDeleter {
    void operator()(aclmdlDataset* p) const {
        if (p) aclmdlDestroyDataset(p);
    }
};

struct AclDataBufferDeleter {
    void operator()(aclDataBuffer* p) const {
        if (p) aclDestroyDataBuffer(p);
    }
};

struct AclModelDescDeleter {
    void operator()(aclmdlDesc* p) const {
        if (p) aclmdlDestroyDesc(p);
    }
};

using HostPtr = std::unique_ptr<void, AclHostDeleter>;
using DevicePtr = std::unique_ptr<void, AclDeviceDeleter>;
using DatasetPtr = std::unique_ptr<aclmdlDataset, AclDatasetDeleter>;
using DataBufferPtr = std::unique_ptr<aclDataBuffer, AclDataBufferDeleter>;
using ModelDescPtr = std::unique_ptr<aclmdlDesc, AclModelDescDeleter>;

}

RESNET50::RESNET50()
    : isInitialized_(false), isModelLoaded_(false) {
}

RESNET50::~RESNET50() {
    Finalize();
}

bool RESNET50::Initialize() {
    if (isInitialized_) {
        return true;
    }

    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclInit failed, error code is " << static_cast<int32_t>(ret) << std::endl;
        return false;
    }

    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed, error code is " << static_cast<int32_t>(ret) << std::endl;
        aclFinalize();
        return false;
    }

    isInitialized_ = true;
    return true;
}

void RESNET50::Finalize() {
    if (isModelLoaded_) {
        aclmdlUnload(modelId_);
        isModelLoaded_ = false;
    }

    if (isInitialized_) {
        aclrtResetDevice(0);
        aclFinalize();
        isInitialized_ = false;
    }
}

bool RESNET50::LoadModel(const std::string& modelPath) {
    if (!isInitialized_) {
        std::cerr << "Not initialized. Call Initialize() first" << std::endl;
        return false;
    }

    if (isModelLoaded_) {
        aclmdlUnload(modelId_);
        isModelLoaded_ = false;
    }

    aclError ret = aclmdlLoadFromFile(modelPath.c_str(), &modelId_);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclmdlLoadFromFile failed, error code is " << static_cast<int32_t>(ret) << std::endl;
        return false;
    }

    modelPath_ = modelPath;
    isModelLoaded_ = true;
    return true;
}

std::vector<std::vector<int>> RESNET50::kzzk_cv(
    const std::string& modelFile,
    const std::string& imageFile
) {
    try {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        
        if (!isInitialized_) {
            if (!Initialize()) {
                std::cerr << "Failed to initialize RESNET50" << std::endl;
                return {};
            }
        }
        
        if (!isModelLoaded_ || modelPath_ != modelFile) {
            if (!LoadModel(modelFile)) {
                std::cerr << "Failed to load ResNet50 model: " << modelFile << std::endl;
                return {};
            }
        }

        auto getFileExtension = [](const std::string& filename) -> std::string {
            size_t dotPos = filename.find_last_of(".");
            if (dotPos == std::string::npos) {
                return "";
            }
            std::string extension = filename.substr(dotPos + 1);
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            return extension;
        };

        std::string extension = getFileExtension(imageFile);
        bool isBinFile = (extension == "bin");
        
        HostPtr pictureHostData;
        size_t pictureDataSize = 0;

        if (isBinFile) {
            std::ifstream binFile(imageFile, std::ios::binary);
            if (!binFile.is_open()) {
                std::cerr << "Failed to open image file: " << imageFile << std::endl;
                return {};
            }

            binFile.seekg(0, std::ios::end);
            pictureDataSize = binFile.tellg();
            binFile.seekg(0, std::ios::beg);

            void* hostPtr = nullptr;
            aclError ret = aclrtMallocHost(&hostPtr, pictureDataSize);
            if (ret != ACL_SUCCESS) {
                std::cerr << "aclrtMallocHost failed, error code is " << static_cast<int32_t>(ret) << std::endl;
                return {};
            }
            pictureHostData.reset(hostPtr);

            binFile.read(reinterpret_cast<char*>(pictureHostData.get()), pictureDataSize);
            binFile.close();
        } else {
            cv::Mat image = cv::imread(imageFile);
            if (image.empty()) {
                std::cerr << "Failed to read image file: " << imageFile << std::endl;
                return {};
            }

            ModelDescPtr modelDesc(aclmdlCreateDesc());
            if (!modelDesc) {
                std::cerr << "aclmdlCreateDesc failed" << std::endl;
                return {};
            }
            
            aclError ret = aclmdlGetDesc(modelDesc.get(), modelId_);
            if (ret != ACL_SUCCESS) {
                std::cerr << "aclmdlGetDesc failed, error code is " << static_cast<int32_t>(ret) << std::endl;
                return {};
            }
            
            aclmdlIODims inputDims;
            aclError ret2 = aclmdlGetInputDims(modelDesc.get(), 0, &inputDims);
            int inputWidth = 224;
            int inputHeight = 224;
            int inputChannels = 3; 
            
            if (ret2 == ACL_SUCCESS && inputDims.dimCount >= 4) {
                bool foundChannel = false;
                for (size_t i = 1; i < inputDims.dimCount; ++i) {
                    if (inputDims.dims[i] == 1 || inputDims.dims[i] == 3 || inputDims.dims[i] == 4) {
                        inputChannels = inputDims.dims[i];
                        foundChannel = true;
            
                        std::vector<int32_t> otherDims;
                        for (size_t j = 1; j < inputDims.dimCount; ++j) {
                            if (j != i) {
                                otherDims.push_back(inputDims.dims[j]);
                            }
                        }
                    
                        if (otherDims.size() >= 2) {
                            inputHeight = otherDims[0];
                            inputWidth = otherDims[1];
                        }
                        break;
                    }
                }
                
                if (!foundChannel) {
                    std::vector<int32_t> dims(inputDims.dims + 1, inputDims.dims + inputDims.dimCount);
                    std::sort(dims.begin(), dims.end());
                    
                    inputChannels = dims[0]; 
                    inputHeight = dims[1]; 
                    inputWidth = dims[2];   
                  
                    if (inputChannels > 4) {
                        inputChannels = 3;
                        inputHeight = 224;
                        inputWidth = 224;
                    }
                }
            }

            cv::Mat resizedImage;
            cv::resize(image, resizedImage, cv::Size(inputWidth, inputHeight), cv::INTER_LINEAR);

            cv::Mat adjustedImage;
            if (resizedImage.channels() != inputChannels) {
                if (inputChannels == 3) {
                    cv::cvtColor(resizedImage, adjustedImage, cv::COLOR_GRAY2BGR);
                } else if (inputChannels == 1) {
                    cv::cvtColor(resizedImage, adjustedImage, cv::COLOR_BGR2GRAY);
                } else {
                    std::cerr << "Unsupported number of channels: " << inputChannels << std::endl;
                    return {};
                }
            } else {
                adjustedImage = resizedImage.clone();
            }

            cv::Mat rgbImage;
            if (inputChannels == 3) {
                cv::cvtColor(adjustedImage, rgbImage, cv::COLOR_BGR2RGB);
            } else {
                rgbImage = adjustedImage.clone();
            }

            cv::Mat floatImage;
            int cvType = (inputChannels == 3) ? CV_32FC3 : CV_32FC1;
            rgbImage.convertTo(floatImage, cvType);

            pictureDataSize = inputWidth * inputHeight * inputChannels * sizeof(float);
            
            void* hostPtr = nullptr;
            ret = aclrtMallocHost(&hostPtr, pictureDataSize);
            if (ret != ACL_SUCCESS) {
                std::cerr << "aclrtMallocHost failed, error code is " << static_cast<int32_t>(ret) << std::endl;
                return {};
            }
            pictureHostData.reset(hostPtr);

            float* hostData = reinterpret_cast<float*>(pictureHostData.get());
            size_t channelSize = inputWidth * inputHeight;
            
            if (inputChannels == 3) {
                std::vector<cv::Mat> channels;
                cv::split(floatImage, channels);
                
                memcpy(hostData + 0 * channelSize, channels[0].data, channelSize * sizeof(float));
                memcpy(hostData + 1 * channelSize, channels[1].data, channelSize * sizeof(float));
                memcpy(hostData + 2 * channelSize, channels[2].data, channelSize * sizeof(float));
            } else {
                memcpy(hostData, floatImage.data, channelSize * sizeof(float));
            }
        }

        DevicePtr pictureDeviceData;
        {
            void* devPtr = nullptr;
            aclError ret = aclrtMalloc(&devPtr, pictureDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                std::cerr << "aclrtMalloc failed, error code is " << static_cast<int32_t>(ret) << std::endl;
                return {};
            }
            pictureDeviceData.reset(devPtr);
        }

        aclError ret = aclrtMemcpy(pictureDeviceData.get(), pictureDataSize, 
                                   pictureHostData.get(), pictureDataSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtMemcpy failed, error code is " << static_cast<int32_t>(ret) << std::endl;
            return {};
        }

        DatasetPtr inputDataSet(aclmdlCreateDataset());
        if (!inputDataSet) {
            std::cerr << "aclmdlCreateDataset failed" << std::endl;
            return {};
        }

        DataBufferPtr inputDataBuffer(aclCreateDataBuffer(pictureDeviceData.get(), pictureDataSize));
        if (!inputDataBuffer) {
            std::cerr << "aclCreateDataBuffer failed" << std::endl;
            return {};
        }

        ret = aclmdlAddDatasetBuffer(inputDataSet.get(), inputDataBuffer.get());
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclmdlAddDatasetBuffer failed, error code is " << static_cast<int32_t>(ret) << std::endl;
            return {};
        }

        DatasetPtr outputDataSet(aclmdlCreateDataset());
        if (!outputDataSet) {
            std::cerr << "aclmdlCreateDataset failed" << std::endl;
            return {};
        }

        ModelDescPtr modelDesc(aclmdlCreateDesc());
        if (!modelDesc) {
            std::cerr << "aclmdlCreateDesc failed" << std::endl;
            return {};
        }

        ret = aclmdlGetDesc(modelDesc.get(), modelId_);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclmdlGetDesc failed, error code is " << static_cast<int32_t>(ret) << std::endl;
            return {};
        }

        size_t outputDataSize = aclmdlGetOutputSizeByIndex(modelDesc.get(), 0);
        
        DevicePtr outputDeviceData;
        {
            void* devPtr = nullptr;
            ret = aclrtMalloc(&devPtr, outputDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                std::cerr << "aclrtMalloc failed, error code is " << static_cast<int32_t>(ret) << std::endl;
                return {};
            }
            outputDeviceData.reset(devPtr);
        }

        DataBufferPtr outputDataBuffer(aclCreateDataBuffer(outputDeviceData.get(), outputDataSize));
        if (!outputDataBuffer) {
            std::cerr << "aclCreateDataBuffer failed" << std::endl;
            return {};
        }

        ret = aclmdlAddDatasetBuffer(outputDataSet.get(), outputDataBuffer.get());
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclmdlAddDatasetBuffer failed, error code is " << static_cast<int32_t>(ret) << std::endl;
            return {};
        }

        ret = aclmdlExecute(modelId_, inputDataSet.get(), outputDataSet.get());
        
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclmdlExecute failed, error code is " << static_cast<int32_t>(ret) << std::endl;
            return {};
        }

        HostPtr outputHostData;
        {
            void* hostPtr = nullptr;
            ret = aclrtMallocHost(&hostPtr, outputDataSize);
            if (ret != ACL_SUCCESS) {
                std::cerr << "aclrtMallocHost failed, error code is " << static_cast<int32_t>(ret) << std::endl;
                return {};
            }
            outputHostData.reset(hostPtr);
        }

        ret = aclrtMemcpy(outputHostData.get(), outputDataSize, 
                          outputDeviceData.get(), outputDataSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtMemcpy failed, error code is " << static_cast<int32_t>(ret) << std::endl;
            return {};
        }

        float* outFloatData = reinterpret_cast<float*>(outputHostData.get());
        size_t outputCount = outputDataSize / sizeof(float);

        std::map<float, unsigned int, std::greater<float>> resultMap;
        for (unsigned int i = 0; i < outputCount; ++i) {
            resultMap[outFloatData[i]] = i;
        }

        std::vector<std::vector<int>> result_kzzk_cv;
        int cnt = 0;
        for (auto it = resultMap.begin(); it != resultMap.end() && cnt < TOP_K; ++it) {
            result_kzzk_cv.push_back({
                static_cast<int>(it->second),
                static_cast<int>(std::round(it->first * 100.0f))
            });
            cnt++;
        }

        gettimeofday(&end, NULL);
        long totalInferenceTime = ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) / 1000;
        std::cout << "Total inference cost: " << totalInferenceTime << "ms" << std::endl;

        return result_kzzk_cv;

    } catch (const std::exception& e) {
        std::cerr << "[RESNET50::kzzk_cv] error: " << e.what() << std::endl;
        return {};
    }
}

