#include "kzzk_llm.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cstring>
#include <map>

#include <unistd.h>

using json = nlohmann::json;

namespace kzzk {

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    static_cast<std::string*>(userp)->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

class LLMClient::Impl {
public:
    std::string baseUrl_;
    long timeoutSeconds_;
    bool debugMode_;
    int maxRetries_;

    Impl(const std::string& baseUrl, long timeoutSeconds, bool enableDebug)
        : baseUrl_(baseUrl), timeoutSeconds_(timeoutSeconds), debugMode_(enableDebug), maxRetries_(3) {
        curl_global_init(CURL_GLOBAL_ALL);
        std::cout << "MindIE Client 初始化成功" << std::endl;
        std::cout << "服务地址: " << baseUrl_ << std::endl;
        std::cout << "超时设置: " << timeoutSeconds_ << " 秒" << std::endl;
        std::cout << "重试次数: " << maxRetries_ << std::endl;
        if (timeoutSeconds_ >= 300) {
            std::cout << "注意: 超时时间较长，建议等待模型预热完成" << std::endl;
        }
    }

    ~Impl() {
        curl_global_cleanup();
    }

    void setMaxRetries(int retries) {
        maxRetries_ = retries > 0 ? retries : 3;
    }

    std::string httpPost(const std::string& endpoint, const std::string& body) {
        std::string lastError;

        for (int attempt = 1; attempt <= maxRetries_; ++attempt) {
            try {
                return doHttpPost(endpoint, body);
            } catch (const std::runtime_error& e) {
                lastError = e.what();
                if (attempt < maxRetries_) {
                    std::cerr << "请求失败 (尝试 " << attempt << "/" << maxRetries_ << "): " << e.what() << std::endl;
                    std::cerr << "等待 5 秒后重试..." << std::endl;
                    sleep(5);
                }
            }
        }

        throw std::runtime_error("请求失败，已重试 " + std::to_string(maxRetries_) + " 次: " + lastError);
    }

    std::string doHttpPost(const std::string& endpoint, const std::string& body) {
        CURL* curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("失败: CURL 初始化失败");
        }

        std::string url = baseUrl_ + endpoint;
        std::string response_string;

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeoutSeconds_);

        if (debugMode_) {
            std::cout << "[POST] " << url << std::endl;
            std::cout << "请求体: " << body << std::endl;
        }

        CURLcode res = curl_easy_perform(curl);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (debugMode_) {
            std::cout << "HTTP状态码: " << http_code << std::endl;
            std::cout << "响应体: " << response_string << std::endl;
        }

        if (res != CURLE_OK) {
            std::string errorMsg = "请求失败: " + std::string(curl_easy_strerror(res));
            if (res == CURLE_OPERATION_TIMEDOUT) {
                errorMsg += "\n可能原因: 服务未启动、网络不通、端口错误";
                errorMsg += "\n请检查: " + url;
            }
            throw std::runtime_error(errorMsg);
        }

        if (http_code != 200) {
            throw std::runtime_error(
                "HTTP 错误码: " + std::to_string(http_code) +
                "\n响应内容: " + response_string
            );
        }

        return response_string;
    }

    std::string httpGet(const std::string& endpoint) {
        std::string lastError;

        for (int attempt = 1; attempt <= maxRetries_; ++attempt) {
            try {
                return doHttpGet(endpoint);
            } catch (const std::runtime_error& e) {
                lastError = e.what();
                if (attempt < maxRetries_) {
                    std::cerr << "请求失败 (尝试 " << attempt << "/" << maxRetries_ << "): " << e.what() << std::endl;
                    std::cerr << "等待 5 秒后重试..." << std::endl;
                    sleep(5);
                }
            }
        }

        throw std::runtime_error("请求失败，已重试 " + std::to_string(maxRetries_) + " 次: " + lastError);
    }

    std::string doHttpGet(const std::string& endpoint) {
        CURL* curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("失败: CURL 初始化失败");
        }

        std::string url = baseUrl_ + endpoint;
        std::string response_string;

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeoutSeconds_);

        if (debugMode_) {
            std::cout << "[GET] " << url << std::endl;
        }

        CURLcode res = curl_easy_perform(curl);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        curl_easy_cleanup(curl);

        if (debugMode_) {
            std::cout << "HTTP状态码: " << http_code << std::endl;
            std::cout << "响应体: " << response_string << std::endl;
        }

        if (res != CURLE_OK) {
            std::string errorMsg = "请求失败: " + std::string(curl_easy_strerror(res));
            if (res == CURLE_OPERATION_TIMEDOUT) {
                errorMsg += "\n可能原因: 服务未启动、网络不通、端口错误";
                errorMsg += "\n请检查: " + url;
            }
            throw std::runtime_error(errorMsg);
        }

        if (http_code != 200) {
            throw std::runtime_error(
                "HTTP 错误码: " + std::to_string(http_code) +
                "\n响应内容: " + response_string
            );
        }

        return response_string;
    }

    bool checkRepetitiveResponse(const std::string& response) const {
        if (response.length() < 100) return false;

        std::istringstream iss(response);
        std::string line;
        std::map<std::string, int> line_counts;
        int total_lines = 0;

        while (std::getline(iss, line)) {
            if (line.empty()) continue;
            line_counts[line]++;
            total_lines++;
        }

        for (const auto& [line, count] : line_counts) {
            if (count > 10 && total_lines > 20) {
                return true;
            }
        }

        return false;
    }

    std::string filterDebugJson(const std::string& rawResponse) const {
        try {
            json j = json::parse(rawResponse);

            const std::vector<std::string> long_array_fields = {
                "decode_time_arr", "prefill_time", "usage"
            };

            for (const auto& field : long_array_fields) {
                if (j.contains(field)) {
                    if (j[field].is_array()) {
                        size_t size = j[field].size();
                        j[field] = "[Array with " + std::to_string(size) + " items]";
                    } else {
                        j[field] = "[Metadata hidden]";
                    }
                }
            }

            return j.dump(2);
        } catch (...) {
            return rawResponse.substr(0, 500) + "...";
        }
    }
};

LLMClient::LLMClient(const std::string& baseUrl, long timeoutSeconds, bool enableDebug)
    : pImpl(std::make_unique<Impl>(baseUrl, timeoutSeconds, enableDebug)) {}

LLMClient::~LLMClient() = default;

std::string LLMClient::chat(
    const std::string& model,
    const std::vector<ChatMessage>& messages,
    bool stream,
    const InferenceOptions& params,
    const std::map<std::string, nlohmann::json>& options
) {
    json body;
    body["model"] = model;

    json jMessages = json::array();
    for (const auto& msg : messages) {
        jMessages.push_back({
            {"role", msg.role},
            {"content", msg.content}
        });
    }
    body["messages"] = jMessages;
    body["stream"] = stream;

    body.update(params.toJson());

    for (const auto& [key, value] : options) {
        body[key] = value;
    }

    if (pImpl->debugMode_) {
        std::cout << "发送对话请求，模型: " << model << std::endl;
        std::cout << "推理参数: " << params.toJson().dump() << std::endl;
    }

    std::string response = pImpl->httpPost("/v1/chat/completions", body.dump());

    try {
        auto j = json::parse(response);

        if (pImpl->debugMode_) {
            std::cout << "原始响应: " << pImpl->filterDebugJson(response) << std::endl;
        }

        if (stream) {
            std::string output;
            std::istringstream ss(response);
            std::string line;
            while (std::getline(ss, line)) {
                if (line.empty()) continue;
                auto line_json = json::parse(line);
                if (line_json.contains("choices") && !line_json["choices"].empty()) {
                    auto delta = line_json["choices"][0]["delta"];
                    if (delta.contains("content")) {
                        output += delta["content"].get<std::string>();
                    }
                }
            }

            if (pImpl->checkRepetitiveResponse(output)) {
                std::cerr << "警告: 检测到高度重复内容，请检查模型状态或调整temperature参数" << std::endl;
            }

            return output;
        } else {
            if (!j.contains("choices") || !j["choices"].is_array() || j["choices"].empty()) {
                throw std::runtime_error("响应格式错误: 无效的 choices 字段");
            }

            auto first_choice = j["choices"][0];

            std::string finish_reason = first_choice.value("finish_reason", "unknown");
            if (pImpl->debugMode_) {
                std::cout << "Finish reason: " << finish_reason << std::endl;
            }

            if (finish_reason == "length") {
                std::cerr << "警告: 响应因达到max_tokens长度而终止，可能存在问题" << std::endl;
            }

            if (!first_choice.contains("message")) {
                throw std::runtime_error("响应格式错误: 未找到 message 字段");
            }

            auto message = first_choice["message"];

            if (!message.contains("content")) {
                throw std::runtime_error("响应格式错误: 未找到 content 字段");
            }

            std::string content = message["content"].get<std::string>();

            if (pImpl->checkRepetitiveResponse(content)) {
                std::cerr << "警告: 检测到高度重复内容，建议：" << std::endl;
                std::cerr << "1. 检查 MindIE 服务端模型是否完全加载" << std::endl;
                std::cerr << "2. 提高 temperature 参数（当前: " << params.temperature << ")" << std::endl;
                std::cerr << "3. 重启 MindIE 服务重新加载模型" << std::endl;
            }

            return content;
        }
    } catch (const json::exception& e) {
        throw std::runtime_error(
            "JSON解析失败: " + std::string(e.what()) +
            "\n原始响应: " + response
        );
    }

    return "";
}

std::string LLMClient::kzzk_llm(const std::string& modelpath, const std::string& prompt) {
    InferenceOptions options;
    return kzzk_llm(modelpath, prompt, options);
}

std::string LLMClient::kzzk_llm(const std::string& modelpath, const std::string& prompt, const InferenceOptions& options) {
    if (modelpath.empty() || prompt.empty()) {
        return "";
    }
    std::vector<ChatMessage> messages = {{ "user", prompt }};
    return chat(modelpath, messages, false, options);
}

std::string LLMClient::chatWithJson(
    const std::string& model,
    const std::vector<ChatMessage>& messages,
    bool stream,
    const InferenceOptions& params
) {
    json body;
    body["model"] = model;

    json jMessages = json::array();
    for (const auto& msg : messages) {
        jMessages.push_back({
            {"role", msg.role},
            {"content", msg.content}
        });
    }
    body["messages"] = jMessages;
    body["stream"] = stream;

    body.update(params.toJson());

    std::string response = pImpl->httpPost("/v1/chat/completions", body.dump());

    try {
        auto j = json::parse(response);

        if (stream) {
            std::string output;
            std::istringstream ss(response);
            std::string line;
            while (std::getline(ss, line)) {
                if (line.empty()) continue;
                auto line_json = json::parse(line);
                if (line_json.contains("choices") && !line_json["choices"].empty()) {
                    auto delta = line_json["choices"][0]["delta"];
                    if (delta.contains("content")) {
                        output += delta["content"].get<std::string>();
                    }
                }
            }
            return output;
        } else {
            if (!j.contains("choices") || !j["choices"].is_array() || j["choices"].empty()) {
                throw std::runtime_error("响应格式错误: 无效的 choices 字段");
            }

            auto first_choice = j["choices"][0];

            if (!first_choice.contains("message")) {
                throw std::runtime_error("响应格式错误: 未找到 message 字段");
            }

            auto message = first_choice["message"];

            if (!message.contains("content")) {
                throw std::runtime_error("响应格式错误: 未找到 content 字段");
            }

            std::string content = message["content"].get<std::string>();

            if (pImpl->checkRepetitiveResponse(content)) {
                std::cerr << "警告: 检测到高度重复内容，建议：" << std::endl;
                std::cerr << "1. 检查 MindIE 服务端模型是否完全加载" << std::endl;
                std::cerr << "2. 提高 temperature 参数（当前: " << params.temperature << ")" << std::endl;
                std::cerr << "3. 重启 MindIE 服务重新加载模型" << std::endl;
            }

            return content;
        }
    } catch (const json::exception& e) {
        throw std::runtime_error(
            "JSON解析失败: " + std::string(e.what()) +
            "\n原始响应: " + response
        );
    }

    return "";
}

std::string LLMClient::postJson(const std::string& endpoint, const std::string& body) {
    return pImpl->httpPost(endpoint, body);
}

std::vector<ModelInfo> LLMClient::listModels() {
    if (pImpl->debugMode_) {
        std::cout << "查询模型列表..." << std::endl;
    }

    std::string response = pImpl->httpGet("/v1/models");

    auto j = json::parse(response);
    std::vector<ModelInfo> models;

    if (j.contains("data")) {
        for (auto& item : j["data"]) {
            ModelInfo info;
            info.id = item.value("id", "");
            info.object = item.value("object", "");
            info.created = item.value("created", 0);
            info.owned_by = item.value("owned_by", "");
            models.push_back(info);
        }
    } else {
        throw std::runtime_error("响应格式错误: 未找到 data 字段");
    }

    if (pImpl->debugMode_) {
        std::cout << "成功获取 " << models.size() << " 个模型" << std::endl;
    }

    return models;
}

ModelInfo LLMClient::getModelInfo(const std::string& modelName) {
    std::string endpoint = "/v1/models/" + modelName;

    if (pImpl->debugMode_) {
        std::cout << "查询模型信息: " << modelName << std::endl;
    }

    std::string response = pImpl->httpGet(endpoint);
    auto j = json::parse(response);

    ModelInfo info;
    info.id = j.value("id", "");
    info.object = j.value("object", "");
    info.created = j.value("created", 0);
    info.owned_by = j.value("owned_by", "");

    return info;
}

std::string kzzk_llm(const std::string& modelpath, const std::string& prompt) {
    LLMClient client;
    return client.kzzk_llm(modelpath, prompt);
}

std::string kzzk_llm(const std::string& modelpath, const std::string& prompt, const InferenceOptions& options) {
    LLMClient client;
    return client.kzzk_llm(modelpath, prompt, options);
}

} // namespace kzzk