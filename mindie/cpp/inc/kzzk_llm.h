#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>
#include <nlohmann/json.hpp>

namespace kzzk {

// 推理参数配置
struct InferenceOptions {
    float temperature = 0.7f;
    int max_tokens = 1024;
    float top_p = 1.0f;
    float top_k = 50;
    float repetition_penalty = 1.0f;
    int num_beams = 1;
    bool do_sample = true;

    nlohmann::json toJson() const {
        nlohmann::json j;
        j["temperature"] = temperature;
        j["max_tokens"] = max_tokens;
        j["top_p"] = top_p;
        j["top_k"] = top_k;
        j["repetition_penalty"] = repetition_penalty;
        j["num_beams"] = num_beams;
        j["do_sample"] = do_sample;
        return j;
    }
};

// 聊天消息结构
struct ChatMessage {
    std::string role;
    std::string content;

    ChatMessage(const std::string& r, const std::string& c) : role(r), content(c) {}
};

// 模型信息结构
struct ModelInfo {
    std::string id;
    std::string object;
    int64_t created;
    std::string owned_by;
};

// LLM客户端类
class LLMClient {
public:
    // 构造函数
    LLMClient(
        const std::string& baseUrl = "http://127.0.0.1:8000",
        long timeoutSeconds = 300,
        bool enableDebug = false
    );

    // 析构函数
    ~LLMClient();

    // 禁止拷贝
    LLMClient(const LLMClient&) = delete;
    LLMClient& operator=(const LLMClient&) = delete;

    // 允许移动
    LLMClient(LLMClient&&) = default;
    LLMClient& operator=(LLMClient&&) = default;

    // 通用聊天接口
    std::string chat(
        const std::string& model,
        const std::vector<ChatMessage>& messages,
        bool stream = false,
        const InferenceOptions& params = {},
        const std::map<std::string, nlohmann::json>& options = {}
    );

    // 简化接口 - 单轮对话
    std::string kzzk_llm(const std::string& modelfile, const std::string& prompt);
    std::string kzzk_llm(const std::string& modelfile, const std::string& prompt, const InferenceOptions& options);

    // JSON格式响应接口
    std::string chatWithJson(
        const std::string& model,
        const std::vector<ChatMessage>& messages,
        bool stream = false,
        const InferenceOptions& params = {}
    );

    // 直接发送 POST 请求并获取原始 JSON 响应
    std::string postJson(
        const std::string& endpoint,
        const std::string& body
    );

    // 获取模型列表
    std::vector<ModelInfo> listModels();

    // 获取模型信息
    ModelInfo getModelInfo(const std::string& modelName);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// 全局便捷函数
std::string kzzk_llm(const std::string& modelfile, const std::string& prompt);
std::string kzzk_llm(const std::string& modelfile, const std::string& prompt, const InferenceOptions& options);

} // namespace kzzk