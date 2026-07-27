import subprocess
import os
import re
import sys
import logging
from typing import List, Dict, Union, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

KZZK_CV_BIN = os.environ.get('KZZK_CV_BIN', '')
KZZK_MODEL_PATH = os.environ.get('KZZK_MODEL_PATH', '/home/HwHiAiUser/cann/model')

DEFAULT_BIN_PATHS = [
    KZZK_CV_BIN,
    "/home/HwHiAiUser/cann/src/build/kzzk_cv",
    "/root/cann/src/build/kzzk_cv",
    "./src/build/kzzk_cv",
    "./build/kzzk_cv",
    "kzzk_cv"
]

DEFAULT_BIN_PATHS = [p for p in DEFAULT_BIN_PATHS if p]


def _find_executable() -> Optional[str]:
    for path in DEFAULT_BIN_PATHS:
        if os.path.exists(path) and os.access(path, os.X_OK):
            logger.debug(f"找到可执行文件: {path}")
            return path
    return None


def kzzk_cv(modelfile: str, imagefile: str) -> Union[List[Dict], Dict]:
    """
    调用 CANN NPU 进行图像识别
    
    :param modelfile: 模型文件路径（.om 文件）（必填）
    :param imagefile: 输入图像路径（必填）
    :return: 识别结果
        - yolov4: 返回检测列表 [{"class": "...", "confidence": 0.9, "bbox": [x1, y1, x2, y2]}, ...]
        - resnet50: 返回分类结果 {"class": "...", "confidence": 0.9}
    
    示例:
        result = kzzk_cv("/home/HwHiAiUser/cann/model/yolov4.om", "dog.jpg")
        result = kzzk_cv("/home/HwHiAiUser/cann/model/resnet50.om", "dog.jpg")
    """
    bin_path = _find_executable()
    if bin_path is None:
        raise RuntimeError(
            f"未找到可执行文件。请设置环境变量 KZZK_CV_BIN 或检查以下路径:\n" +
            "\n".join(f"  - {p}" for p in DEFAULT_BIN_PATHS if p)
        )
    
    if not os.path.exists(imagefile):
        raise FileNotFoundError(f"图像文件不存在: {imagefile}")
    if not os.path.exists(modelfile):
        raise FileNotFoundError(f"模型文件不存在: {modelfile}")
    
    
    if "yolo" in modelfile.lower():
        return _detect_objects(imagefile, modelfile, bin_path)
    elif "resnet" in modelfile.lower():
        return _classify_image(imagefile, modelfile, bin_path)
    else:
        logger.warning(f"无法从模型名称判断类型: {modelfile}，尝试 YOLOv4")
        try:
            return _detect_objects(imagefile, modelfile, bin_path)
        except Exception as e:
            logger.debug(f"YOLOv4 检测失败: {e}，尝试 ResNet50")
            return _classify_image(imagefile, modelfile, bin_path)


def _detect_objects(imagefile: str, modelfile: str, bin_path: str) -> List[Dict]:
    try:
        cmd = [bin_path, "-m", modelfile, "-i", imagefile]
        logger.debug(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() or f"返回码: {result.returncode}"
            raise RuntimeError(f"检测失败: {error_msg}")
        
        lines = result.stdout.strip().split('\n')
        detections = []
        
        for line in lines:
            if '[' in line and ']' in line and '(' in line and ')' in line:
                try:
                    class_match = re.search(r'\(([^)]+)\)', line)
                    if not class_match:
                        continue
                    class_name = class_match.group(1)
                    
                    all_bbox_matches = re.findall(r'\[([^\]]+)\]', line)
                    bbox_parts = None
                    for bbox_str in all_bbox_matches:
                        parts = [x.strip() for x in bbox_str.split(',')]
                        if len(parts) == 4:
                            try:
                                bbox_parts = [int(x) for x in parts]
                                break
                            except ValueError:
                                continue
                    
                    if bbox_parts is None:
                        continue
                    
                    confidence_match = re.search(r'(\d+)%', line)
                    confidence = int(confidence_match.group(1)) / 100.0 if confidence_match else 0.0
                    
                    detections.append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": bbox_parts
                    })
                except (ValueError, IndexError) as e:
                    logger.debug(f"解析行失败: {line} - {e}")
                    continue
        
        return detections
        
    except Exception as e:
        raise RuntimeError(f"检测失败: {e}")


def _classify_image(imagefile: str, modelfile: str, bin_path: str) -> Dict:
    try:
        cmd = [bin_path, "-m", modelfile, "-i", imagefile]
        logger.debug(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() or f"返回码: {result.returncode}"
            raise RuntimeError(f"分类失败: {error_msg}")
        
        lines = result.stdout.strip().split('\n')
        
        for line in lines:
            if 'Result' in line and '[' in line and ']' in line:
                match = re.search(r'\[(\d+),\s*(\d+)\]', line)
                if match:
                    class_id = int(match.group(1))
                    confidence = int(match.group(2)) / 100.0
                    
                    logger.info(f"分类结果: class_id={class_id}, confidence={confidence:.2f}")
                    return {
                        "class": str(class_id),
                        "confidence": confidence
                    }
        
        logger.warning("未能解析分类结果")
        return {"class": "unknown", "confidence": 0.0}
        
    except Exception as e:
        raise RuntimeError(f"分类失败: {e}")


def print_help():
    print("自研加速卡协同推理软件")
    print("命令行接口")
    print("Usage: kzzk_py_cv.py [OPTIONS]")
    print("Options:")
    print("  --modelfile MODEL      指定模型文件路径（.om 文件）")
    print("  --imagefile IMAGE      指定输入图像路径")
    print("  --help                 显示帮助信息")
    print("")
    print("Examples:")
    print("  python kzzk_py_cv.py --modelfile /home/HwHiAiUser/cann/model/yolov4.om --imagefile /home/HwHiAiUser/cann/data/dog.jpg")
    print("  python kzzk_py_cv.py --modelfile /home/HwHiAiUser/cann/model/resnet50.om --imagefile /home/HwHiAiUser/cann/data/dog.jpg")


def main():
    args = sys.argv[1:]
    
    if not args or "--help" in args:
        print_help()
        sys.exit(0)
    
    modelfile = None
    imagefile = None
    
    i = 0
    while i < len(args):
        if args[i] == "--modelfile" and i + 1 < len(args):
            modelfile = args[i + 1]
            i += 2
        elif args[i] == "--imagefile" and i + 1 < len(args):
            imagefile = args[i + 1]
            i += 2
        else:
            i += 1
    
    if not modelfile or not imagefile:
        print("错误: 缺少必要参数", file=sys.stderr)
        print_help()
        sys.exit(1)
    
    try:
        result = kzzk_cv(modelfile, imagefile)
        
        if isinstance(result, list):
            print(f"检测到 {len(result)} 个目标:")
            for det in result:
                print(f"  - {det['class']} (置信度: {det['confidence']:.2f}, 位置: {det['bbox']})")
        elif isinstance(result, dict):
            print(f"分类结果: {result['class']} (置信度: {result['confidence']:.2f})")
        else:
            print(result)
            
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

