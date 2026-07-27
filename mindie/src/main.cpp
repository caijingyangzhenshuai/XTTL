#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>
#include <nlohmann/json.hpp>
#include "kzzk_llm.h"

using json = nlohmann::json;

void printHelp() {
    std::cout << "自研加速卡协同推理软件" << std::endl;
    std::cout << "命令行接口" << std::endl;
    std::cout << "Usage: kzzk_llm [OPTIONS]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help              Show this help message" << std::endl;
    std::cout << "  --listModels        List all available models from server" << std::endl;
    std::cout << "  --modelfile <path>  Model path or remote URL" << std::endl;
    std::cout << "                      - Local: model name (e.g., Qwen3-14B)" << std::endl;
    std::cout << "                      - Remote: http://host:port/model" << std::endl;
    std::cout << "  --prompt <text>     Input prompt for inference" << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  kzzk_llm --listModels" << std::endl;
    std::cout << "  kzzk_llm --modelfile Qwen3-14B --prompt \"Hello!\"" << std::endl;
    std::cout << "  kzzk_llm --modelfile http://192.168.1.69:8000/Qwen3-14B --prompt \"写一个快速排序\"" << std::endl;
}

std::string cleanPrompt(const std::string& input) {
    std::string output;
    for (unsigned char c : input) {
        if (c >= 0x20 || c == '\n' || c == '\t') {
            output += c;
        }
    }
    return output;
}

int main(int argc, char* argv[]) {
    std::string modelFile;
    std::string prompt;
    bool listModels = false;

    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];

        if (key == "--help") {
            printHelp();
            return 0;
        }

        if (key == "--listModels") {
            listModels = true;
            continue;
        }

        if (key == "--modelfile" && i + 1 < argc) {
            modelFile = argv[++i];
            continue;
        }

        if (key == "--prompt" && i + 1 < argc) {
            prompt.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                if (!prompt.empty()) prompt += " ";
                prompt += argv[++i];
            }
            continue;
        }

        std::cerr << "Unknown option: " << key << std::endl;
        printHelp();
        return 1;
    }

    try {
        std::string serverUrl = "http://127.0.0.1:1025";
        std::string modelName = modelFile;

        if (!modelFile.empty() && (modelFile.find("http://") == 0 || modelFile.find("https://") == 0)) {
            size_t protocolEnd = modelFile.find("://");
            if (protocolEnd != std::string::npos) {
                size_t pathStart = modelFile.find('/', protocolEnd + 3);
                if (pathStart != std::string::npos) {
                    serverUrl = modelFile.substr(0, pathStart);
                    modelName = modelFile.substr(pathStart + 1);
                }
            }
        }

        kzzk::LLMClient client(serverUrl, 600L, false);

        if (listModels) {
            std::vector<kzzk::ModelInfo> models = client.listModels();
            std::cout << "Available Models:\n";
            std::cout << "-------------------\n";
            if (models.empty()) {
                std::cout << "  No models found on server\n";
            } else {
                for (const auto& model : models) {
                    std::cout << "• " << model.id << "\n";
                }
            }
            std::cout << "-------------------\n";
            return 0;
        }

        if (modelFile.empty()) {
            std::cerr << "Error: --modelfile is required\n";
            printHelp();
            return 1;
        }

        if (prompt.empty()) {
            std::cerr << "Error: --prompt is required\n";
            printHelp();
            return 1;
        }

        prompt = cleanPrompt(prompt);

        std::cout << "Using model: " << modelName << "\n";
        std::cout << "Server: " << serverUrl << "\n";
        std::cout << "Prompt: " << prompt << "\n";
        std::cout << "Generating response...\n\n";

        json requestBody;
        requestBody["model"] = modelName;
        requestBody["messages"] = json::array({
            {{"role", "user"}, {"content", prompt}}
        });
        requestBody["stream"] = false;

        std::string rawJson = client.postJson("/v1/chat/completions", requestBody.dump());

        json responseJson = json::parse(rawJson);
        std::string reply;

        if (responseJson.contains("choices") && !responseJson["choices"].empty()) {
            reply = responseJson["choices"][0]["message"]["content"].get<std::string>();
        } else {
            reply = "Failed to parse response";
        }

        std::cout << "Response:\n";
        std::cout << "-------------------\n";
        std::cout << reply << "\n";
        std::cout << "-------------------\n";

        std::time_t now = std::time(nullptr);
        char timestamp[64];
        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&now));

        std::string fileName = "/home/HwHiAiUser/mindie/data/" + std::string(timestamp) + "_" + modelName + ".json";

        std::ofstream outFile(fileName);
        if (outFile.is_open()) {
            outFile << rawJson << "\n";
            outFile.close();
            std::cout << "JSON 响应已保存到: " << fileName << std::endl;
        } else {
            std::cerr << "无法保存 JSON 到文件: " << fileName << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}