import os
import sys
import time
import numpy as np
from PIL import Image

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, ".."))
sys.path.append(os.path.join(path, "../examples/samples/python/common/"))
sys.path.append(os.path.join(path, "../examples/samples/python/common/acllite"))

# === 过滤 [INFO] 日志输出 ===
class FilteredStdout:
    """过滤掉 [INFO] 开头的日志行"""
    def __init__(self, original, filter_prefixes=("[INFO]",)):
        self.original = original
        self.filter_prefixes = filter_prefixes
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if not any(line.strip().startswith(p) for p in self.filter_prefixes):
                self.original.write(line + "\n")
                self.original.flush()

    def flush(self):
        if self.buffer:
            if not any(self.buffer.strip().startswith(p) for p in self.filter_prefixes):
                self.original.write(self.buffer)
            self.buffer = ""
        self.original.flush()

SHOW_INFO = '--show-info' in sys.argv

if not SHOW_INFO:
    sys.stdout = FilteredStdout(sys.stdout)
    sys.stderr = FilteredStdout(sys.stderr)

import acl
import constants as const
import acllite_utils as utils
from acllite_imageproc import AclLiteImageProc
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource

# ImageNet 部分类别 (简化，完整列表可替换)
LABELS_IMAGENET = [
    "category_" + str(i) for i in range(1000)
]

# ResNet50 参数
MODEL_WIDTH = 224
MODEL_HEIGHT = 224
TOP_K = 5
DEBUG = os.environ.get('RESNET_DEBUG', '0') == '1'

_acl_resource = None
_detect_instance = None


def print_help():
    print("ResNet50图像分类推理")
    print("Usage: resnet50.py [OPTIONS]")
    print("Options:")
    print("  --modelfile MODEL      指定模型文件路径（.om 文件）")
    print("  --imagefile IMAGE      指定输入图像路径")
    print("  --show-info            显示底层库的 [INFO] 日志")
    print("  --help                 显示帮助信息")
    print("")
    print("Examples:")
    print("  python resnet50.py --modelfile /home/HwHiAiUser/cann/model/resnet50.om --imagefile test.jpg")


class ResNet50Classify:
    def __init__(self, model_path, model_width=224, model_height=224):
        self._model_path = model_path
        self._model_width = model_width
        self._model_height = model_height
        self._model = None

    def init(self):
        self._model = AclLiteModel(self._model_path)
        if DEBUG:
            print("[DEBUG] ResNet50 model loaded: %s" % self._model_path)
        return 0

    def pre_process(self, image_path):
        """预处理：resize 224x224, normalize [0,1], CHW, add batch dim"""
        img = np.array(Image.open(image_path).convert('RGB'))
        if img is None or img.size == 0:
            raise FileNotFoundError("无法读取图像: %s" % image_path)

        # Resize 到 224x224
        img_resized = np.array(Image.fromarray(img).resize(
            (self._model_width, self._model_height), Image.BILINEAR))
        
        # 归一化到 [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # HWC -> CHW
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        # 添加 batch 维度 [1, 3, 224, 224]
        img_input = np.expand_dims(img_chw, axis=0).astype(np.float32)

        if DEBUG:
            print("[DEBUG] Image preprocessed: shape=%s dtype=%s" % (img_input.shape, img_input.dtype))

        return img_input

    def inference(self, input_data):
        if DEBUG:
            print("[DEBUG] Running inference...")
        result = self._model.execute([input_data])
        return result

    def post_process(self, infer_output):
        """后处理：解析 logits -> softmax -> top-k"""
        logits = infer_output[0]
        
        if logits.ndim > 1:
            logits = logits.flatten()

        if DEBUG:
            print("[DEBUG] logits shape=%s, min=%.3f, max=%.3f" % (
                logits.shape, float(logits.min()), float(logits.max())))

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        # Top-K
        top_indices = np.argsort(probabilities)[::-1][:TOP_K]
        
        results = []
        for idx in top_indices:
            idx = int(idx)
            results.append({
                "class_id": idx,
                "label": LABELS_IMAGENET[idx] if idx < len(LABELS_IMAGENET) else str(idx),
                "confidence": float(probabilities[idx])
            })

        if DEBUG:
            print("[DEBUG] Top-%d predictions:" % TOP_K)
            for r in results:
                print("  [%d] %s: %.4f" % (r["class_id"], r["label"], r["confidence"]))

        return results


def save_results(result, model_name, imagefile):
    out_dir = os.path.dirname(imagefile)
    if not out_dir:
        out_dir = "."
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(imagefile))[0]
    txt_path = os.path.join(out_dir, "%s_%s_%s.txt" % (timestamp, model_name, base))

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Image: %s\n" % imagefile)
        f.write("Model: %s\n" % model_name)
        f.write("Top-%d predictions:\n" % TOP_K)
        f.write("Format: [ID] (Class Name) Confidence\n")
        for i, det in enumerate(result['results'], 1):
            display_pct = min(100, max(0, int(round(det['confidence'] * 100))))
            f.write("[%d] %d (%s) %d%%\n" % (
                i, det['class_id'], det['label'], display_pct))
    return txt_path


def kzzk_cv(modelfile, imagefile):
    global _acl_resource, _detect_instance

    if not os.path.exists(imagefile):
        raise FileNotFoundError("图像文件不存在: %s" % imagefile)
    if not os.path.exists(modelfile):
        raise FileNotFoundError("模型文件不存在: %s" % modelfile)

    if _acl_resource is None:
        _acl_resource = AclLiteResource()
        _acl_resource.init()

    if (_detect_instance is None
            or _detect_instance._model_path != modelfile):
        _detect_instance = ResNet50Classify(modelfile, MODEL_WIDTH, MODEL_HEIGHT)
        ret = _detect_instance.init()
        utils.check_ret("ResNet50 init", ret)

    start_time = time.perf_counter()

    input_data = _detect_instance.pre_process(imagefile)
    result = _detect_instance.inference(input_data)
    output = _detect_instance.post_process(result)

    infer_cost = int(round((time.perf_counter() - start_time) * 1000))
    model_name = os.path.splitext(os.path.basename(modelfile))[0]

    return {
        "results": output,
        "infer_cost": infer_cost,
        "model_name": model_name,
    }


def _explicit_cleanup():
    global _acl_resource, _detect_instance
    try:
        if _detect_instance is not None and _detect_instance._model is not None:
            try:
                _detect_instance._model.__del__()
            except Exception:
                pass
            _detect_instance._model = None
        _detect_instance = None

        if _acl_resource is not None:
            try:
                if hasattr(_acl_resource, 'release'):
                    _acl_resource.release()
                elif hasattr(_acl_resource, 'destroy'):
                    _acl_resource.destroy()
            except Exception:
                pass
        _acl_resource = None
    except Exception:
        pass


def main():
    import sys as _sys
    args = _sys.argv[1:]

    if "--help" in args or "-h" in args:
        print_help()
        return

    modelfile = None
    imagefile = None

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--modelfile" and i + 1 < len(args):
            modelfile = args[i + 1]
            i += 2
        elif a == "--imagefile" and i + 1 < len(args):
            imagefile = args[i + 1]
            i += 2
        else:
            i += 1

    if not modelfile or not imagefile:
        print_help()
        return

    try:
        result = kzzk_cv(modelfile, imagefile)
    except Exception as e:
        _orig = sys.stdout
        while isinstance(_orig, FilteredStdout):
            _orig = _orig.original
        print("错误: %s" % str(e))
        return

    print("Total inference cost: %dms" % result['infer_cost'])
    print("Inference Results: ")
    print("ResNet50图像分类推理")
    print("Top-%d predictions:" % TOP_K)
    print("Format: [ID] (Class Name) Confidence")
    for i, det in enumerate(result['results'], 1):
        display_pct = min(100, max(0, int(round(det['confidence'] * 100))))
        print("[%d] %d (%s) %d%%" % (
            i, det['class_id'], det['label'], display_pct))

    txt_path = save_results(result, result['model_name'], imagefile)
    print("推理结果已保存到: %s" % txt_path)


if __name__ == '__main__':
    try:
        main()
    finally:
        _explicit_cleanup()
