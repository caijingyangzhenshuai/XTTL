import sys
import os
import time
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import acl
import logging
from typing import Dict, List

logging.basicConfig(level=logging.CRITICAL + 1)
logging.getLogger().setLevel(logging.CRITICAL + 1)

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, ".."))
sys.path.append(os.path.join(path, "../examples/samples/python/common/"))
sys.path.append(os.path.join(path, "../examples/samples/python/common/acllite"))

import acllite_logger
acllite_logger.log_info = lambda *args: None
acllite_logger.log_debug = lambda *args: None

from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel
from acllite_image import AclLiteImage
from acllite_imageproc import AclLiteImageProc
import constants as const
import acllite_utils as utils

currentPath = os.path.join(path, "..")
OUTPUT_DIR = os.path.join(currentPath, 'out')
MODEL_WIDTH = 224
MODEL_HEIGHT = 256
CLS = ['dog', 'cat']

_acl_resource = None
_classify_instance = None

class Classify(object):
    def __init__(self, model_path, model_width, model_height):
        self._model_path = model_path
        self._model_width = model_width
        self._model_height = model_height
        self._model = None
        self._dvpp = None

    def init(self):
        self._dvpp = AclLiteImageProc()
        self._model = AclLiteModel(self._model_path)
        return const.SUCCESS

    def pre_process(self, image):
        is_png = hasattr(image, '_encode_format') and image._encode_format == 2
        
        if is_png:
            yuv_image = self._convert_non_jpeg_to_yuv(image)
        else:
            image_dvpp = image.copy_to_dvpp()
            yuv_image = self._dvpp.jpegd(image_dvpp)
            if yuv_image is None:
                yuv_image = self._convert_non_jpeg_to_yuv(image)
        
        if yuv_image is None:
            raise RuntimeError("图像预处理失败")
        
        resized_image = self._dvpp.resize(yuv_image,
                                          self._model_width, self._model_height)
        return resized_image  

    def _convert_non_jpeg_to_yuv(self, image):
        import numpy as np
        import acl
        from PIL import Image
        
        if image._image_path is None:
            return None
        
        stride_width = utils.align_up128(image.width)
        stride_height = utils.align_up16(image.height)
        stride_size = utils.yuv420sp_size(stride_width, stride_height)
        
        out_buffer, ret = acl.media.dvpp_malloc(stride_size)
        if ret != const.ACL_SUCCESS:
            return None
        
        pil_image = Image.open(image._image_path).convert('RGB')
        rgb_array = np.array(pil_image).astype(np.uint8)
        
        R = rgb_array[:, :, 0].astype(np.float32)
        G = rgb_array[:, :, 1].astype(np.float32)
        B = rgb_array[:, :, 2].astype(np.float32)
        
        Y = 0.257 * R + 0.504 * G + 0.098 * B + 16
        U = -0.148 * R - 0.291 * G + 0.439 * B + 128
        V = 0.439 * R - 0.368 * G - 0.071 * B + 128
        
        Y = np.clip(Y, 0, 255).astype(np.uint8)
        U = np.clip(U, 0, 255).astype(np.uint8)
        V = np.clip(V, 0, 255).astype(np.uint8)
        
        y_plane_size = stride_width * stride_height
        
        yuv_buffer = np.zeros(stride_size, dtype=np.uint8)
        
        y_dst = yuv_buffer[:y_plane_size].reshape(stride_height, stride_width)
        for row in range(image.height):
            y_dst[row, :image.width] = Y[row]
        
        u_sub = U[::2, ::2]
        v_sub = V[::2, ::2]
        
        uv_start = y_plane_size
        uv_h = image.height // 2
        uv_w = image.width // 2
        
        if uv_w > 0 and uv_h > 0:
            uv_dst = yuv_buffer[uv_start:uv_start + uv_h * stride_width].reshape(uv_h, stride_width)
            for row in range(uv_h):
                uv_dst[row, :uv_w * 2:2] = u_sub[row, :uv_w]
                uv_dst[row, 1:uv_w * 2:2] = v_sub[row, :uv_w]
        
        if "bytes_to_ptr" in dir(acl.util):
            data_ptr = acl.util.bytes_to_ptr(yuv_buffer.tobytes())
        else:
            data_ptr = acl.util.numpy_to_ptr(yuv_buffer)
        
        acl.rt.memcpy(out_buffer, stride_size, data_ptr, stride_size, const.ACL_MEMCPY_HOST_TO_DEVICE)
        
        return AclLiteImage(out_buffer, image.width, image.height, stride_width,
                           stride_height, stride_size, const.MEMORY_DVPP)

    def inference(self, input_data):
        return self._model.execute(input_data)

    def post_process(self, infer_output):
        infer_result = infer_output[0]
        vals = infer_result.flatten()
        top_indices = vals.argsort()[-5:][::-1]
        results = []
        for idx in top_indices:
            label = CLS[idx] if idx < len(CLS) else str(idx)
            results.append({
                "class_id": int(idx),
                "label": label,
                "confidence": int(round(float(vals[idx]) * 100))
            })
        return results


def kzzk_cv(modelfile, imagefile):
    global _acl_resource, _classify_instance
    
    if not os.path.exists(imagefile):
        raise FileNotFoundError(f"图像文件不存在: {imagefile}")
    if not os.path.exists(modelfile):
        raise FileNotFoundError(f"模型文件不存在: {modelfile}")
    
    if _acl_resource is None:
        _acl_resource = AclLiteResource()
        _acl_resource.init()
    
    if _classify_instance is None or _classify_instance._model_path != modelfile:
        _classify_instance = Classify(modelfile, MODEL_WIDTH, MODEL_HEIGHT)
        ret = _classify_instance.init()
        utils.check_ret("Classify init", ret)
    
    start_time = time.perf_counter()
    
    image = AclLiteImage(imagefile)
    
    resized_image = _classify_instance.pre_process(image)
    result = _classify_instance.inference([resized_image, ])
    output = _classify_instance.post_process(result)
    
    infer_cost = int(round((time.perf_counter() - start_time) * 1000))
    
    model_name = os.path.splitext(os.path.basename(modelfile))[0]
    
    return {
        "results": output,
        "infer_cost": infer_cost,
        "model_name": model_name
    }


def save_results(results, model_name, image_path):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.dirname(image_path)
    if not result_dir:
        result_dir = os.getcwd()
    filename = f"{timestamp}_{model_name}.txt"
    filepath = os.path.join(result_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"Total inference cost: {results['infer_cost']}ms\n")
        f.write(f"Inference Results:\n")
        f.write(f"{model_name}图像识别推理\n")
        f.write(f"Top 5 classes:\n")
        f.write(f"Format: [ID Label Confidence]\n")
        for i, res in enumerate(results['results'], 1):
            label = res.get('label', str(res['class_id']))
            f.write(f"Result {i}: [{res['class_id']} {label} {res['confidence']}%]\n")
    
    return filepath


def print_help():
    print("VGG16图像分类推理")
    print("Usage: vgg16.py [OPTIONS]")
    print("Options:")
    print("  --modelfile MODEL      指定模型文件路径（.om 文件）")
    print("  --imagefile IMAGE      指定输入图像路径")
    print("  --help                 显示帮助信息")
    print("")
    print("Examples:")
    print("  python vgg16.py --modelfile /home/HwHiAiUser/cann/model/vgg16_cat_dog.om --imagefile /home/HwHiAiUser/cann/data/cat.jpg")


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
        
        print(f"Total inference cost: {result['infer_cost']}ms")
        print(f"Inference Results:")
        print(f"VGG16图像分类推理")
        print(f"Top 5 classes:")
        print(f"Format: [ID Label Confidence]")
        for i, res in enumerate(result['results'], 1):
            label = res.get('label', str(res['class_id']))
            print(f"Result {i}: [{res['class_id']} {label} {res['confidence']}%]")
        
        save_path = save_results(result, result['model_name'], imagefile)
        print(f"推理结果已保存到: {save_path}")
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
 

