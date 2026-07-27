import requests
import json
import re
import sys
import time
import logging

DEFAULT_SERVER_URL = "http://127.0.0.1:1025"
DEFAULT_TIMEOUT = 300
MAX_RETRIES = 3
RETRY_DELAY = 2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def kzzk_llm(modelfile, prompt, server_url=None, max_tokens=1024, temperature=0.7):
    """
    调用 MindIE LLM 进行文本生成
    
    :param modelfile: 模型名称或路径（必填）
    :param prompt: 提示词（必填）
    :param server_url: 服务地址，默认 http://127.0.0.1:1025
    :param max_tokens: 最大生成 token 数
    :param temperature: 温度参数
    :return: 生成的文本字符串
    
    示例:
        reply = kzzk_llm("Qwen3-14B", "请写一个冒泡排序")
        reply = kzzk_llm("DeepSeek-14B", "你好")
    """
    url = server_url or DEFAULT_SERVER_URL
    
    if modelfile is None:
        models = list_models(server_url=url)
        if not models:
            raise RuntimeError("无可用模型")
        modelfile = models[0]
    
    headers = {"Content-Type": "application/json"}
    
    api_endpoints = [
        {
            "url": f"{url}/v1/chat/completions",
            "data": {
                "model": modelfile,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
        },
        {
            "url": f"{url}/v1/completions",
            "data": {
                "model": modelfile,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
        },
        {
            "url": f"{url}/api/text-generation",
            "data": {
                "model": modelfile,
                "input": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        },
        {
            "url": f"{url}/generate",
            "data": {
                "model": modelfile,
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "temperature": temperature
            }
        }
    ]
    
    errors = []
    
    for api in api_endpoints:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    api["url"],
                    headers=headers,
                    json=api["data"],
                    timeout=DEFAULT_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    response_text = ""
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        if "text" in result["choices"][0]:
                            response_text = result["choices"][0]["text"]
                        elif "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                            response_text = result["choices"][0]["message"]["content"]
                    elif "response" in result:
                        response_text = str(result["response"])
                    elif "text" in result:
                        response_text = str(result["text"])
                    elif "output" in result:
                        if isinstance(result["output"], dict) and "text" in result["output"]:
                            response_text = str(result["output"]["text"])
                        else:
                            response_text = str(result["output"])
                    
                    response_text = _clean_response(response_text)
                    
                    return response_text.strip()
                else:
                    errors.append(f"{api['url']}: HTTP {response.status_code}")
                    
            except requests.exceptions.Timeout:
                errors.append(f"{api['url']}: 请求超时")
                if attempt < MAX_RETRIES:
                    logger.warning(f"请求超时，{RETRY_DELAY}秒后重试 (尝试 {attempt}/{MAX_RETRIES})")
                    time.sleep(RETRY_DELAY)
            except requests.exceptions.ConnectionError as e:
                errors.append(f"{api['url']}: 连接失败 - {e}")
                if attempt < MAX_RETRIES:
                    logger.warning(f"连接失败，{RETRY_DELAY}秒后重试 (尝试 {attempt}/{MAX_RETRIES})")
                    time.sleep(RETRY_DELAY)
            except Exception as e:
                errors.append(f"{api['url']}: {type(e).__name__} - {e}")
                break
    
    error_msg = "所有 API 端点都无法调用成功:\n" + "\n".join(f"  - {e}" for e in errors)
    raise RuntimeError(error_msg)


def _clean_response(text):
    if not text:
        return ""
    
    text = re.sub(r'<think_>[\s\S]*?</think_>', '', text)
    text = text.lstrip("。.、,，")
    text = text.strip()
    
    return text


def list_models(server_url=None):
    """
    获取服务器上所有可用模型
    
    :param server_url: 服务地址
    :return: 模型名称列表
    
    示例:
        models = list_models()
        print(models)
    """
    url = server_url or DEFAULT_SERVER_URL
    
    endpoints = ["/v1/models", "/api/models", "/models", "/api/list-models"]
    
    for endpoint in endpoints:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(f"{url}{endpoint}", timeout=DEFAULT_TIMEOUT)
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data:
                        return [item["id"] for item in data["data"]]
                    elif isinstance(data, list):
                        return [item.get("id", str(item)) for item in data]
                    elif isinstance(data, dict) and "models" in data:
                        return data["models"]
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES:
                    logger.debug(f"获取模型列表失败: {endpoint} - {e}")
            except Exception as e:
                logger.debug(f"解析响应失败: {endpoint} - {e}")
                break
    
    return []


def print_help():
    print("自研加速卡协同推理软件")
    print("命令行接口")
    print("Usage: kzzk_py_llm.py [OPTIONS]")
    print("Options:")
    print("  --listModels           列出可用模型")
    print("  --modelfile MODEL      指定模型名称")
    print("  --prompt PROMPT        指定提示词")
    print("  --help                 显示帮助信息")
    print("")
    print("Examples:")
    print("  python3 kzzk_py_llm.py --listModels")
    print("  python3 kzzk_py_llm.py --modelfile Qwen3-14B --prompt \"你好\"")


def main():
    args = sys.argv[1:]
    
    if not args or "--help" in args:
        print_help()
        sys.exit(0)
    
    if "--listModels" in args:
        models = list_models()
        if models:
            print("可用模型:")
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
        else:
            print("未找到可用模型")
        sys.exit(0)
    
    modelfile = None
    prompt = None
    
    i = 0
    while i < len(args):
        if args[i] == "--modelfile" and i + 1 < len(args):
            modelfile = args[i + 1]
            i += 2
        elif args[i] == "--prompt" and i + 1 < len(args):
            prompt = args[i + 1]
            i += 2
        else:
            i += 1
    
    if not modelfile or not prompt:
        print("错误: 缺少必要参数")
        print_help()
        sys.exit(1)
    
    try:
        reply = kzzk_llm(modelfile, prompt)
        print(reply)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
