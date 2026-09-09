import sys
import os
from urllib.parse import urlparse

path = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(path)
sys.path.insert(0, parent)

from python import kzzk_llm, list_models


def print_help():
    print("自研加速卡协同推理软件")
    print("命令行接口")
    print("Usage: kzzk_llm.py [OPTIONS]")
    print("Options:")
    print("  --listModels           列出可用模型")
    print("  --modelfile MODEL      指定模型名称")
    print("  --prompt PROMPT        指定提示词")
    print("  --help                 显示帮助信息")
    print("")
    print("Examples:")
    print("  python3 kzzk_llm.py --listModels")
    print("  python3 kzzk_llm.py --modelfile Qwen3-14B --prompt \"你好\"")


def main():
    args = sys.argv[1:]

    if not args or "--help" in args:
        print_help()
        sys.exit(0)

    if "--listModels" in args:
        print("模型列表显示 Python版")
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

    # 兼容 --modelfile 传入完整 URL 的情况，如 http://127.0.0.1:1025/Baichuan2-7B-Chat
    server_url = None
    if modelfile and "://" in modelfile:
        parsed = urlparse(modelfile)
        if parsed.path.strip("/"):
            server_url = f"{parsed.scheme}://{parsed.netloc}"
            modelfile = parsed.path.strip("/")

    try:
        print(f"{modelfile}模型交互 Python版")
        reply = kzzk_llm(modelfile, prompt, server_url=server_url)
        print(reply)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
