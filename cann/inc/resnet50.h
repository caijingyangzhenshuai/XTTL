#ifndef RESNET50_H
#define RESNET50_H

#include <vector>
#include <string>
#include <memory>
#include <acl/acl.h>

// ResNet50类定义
class RESNET50 {
private:
    bool isInitialized_;
    bool isModelLoaded_;
    std::string modelPath_;
    uint32_t modelId_; // 模型ID
    
    // ResNet50参数配置
    const int TOP_K = 5;
    
public:
    RESNET50();
    ~RESNET50();
    
    // 初始化和释放资源
    bool Initialize();
    void Finalize();
    
    // 加载模型
    bool LoadModel(const std::string& modelPath);
    
    // 执行推理
    std::vector<std::vector<int>> kzzk_cv(
        const std::string& modelFile,
        const std::string& imageFile
    );
};

#endif // RESNET50_H

