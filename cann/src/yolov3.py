import sys
import os
import time
import datetime
import numpy as np
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

import acllite_utils as utils
from PIL import Image, ImageDraw, ImageFont
from acllite_imageproc import AclLiteImageProc
import constants as const
from acllite_model import AclLiteModel
from acllite_image import AclLiteImage
from acllite_resource import AclLiteResource

labels = ["person",
        "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
        "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]

MODEL_WIDTH = 416
MODEL_HEIGHT = 416

_acl_resource = None
_model_instance = None

class YOLOv3(object):
    def __init__(self, model_path):
        self._model_path = model_path
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
            image_input = image.copy_to_dvpp()
            yuv_image = self._dvpp.jpegd(image_input)
            if yuv_image is None:
                yuv_image = self._convert_non_jpeg_to_yuv(image)
        
        if yuv_image is None:
            raise RuntimeError("图像预处理失败")
        
        resized_image = self._dvpp.crop_and_paste(yuv_image, image.width, image.height, MODEL_WIDTH, MODEL_HEIGHT)
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

    def construct_image_info(self):
        image_info = np.array([MODEL_WIDTH, MODEL_HEIGHT, 
                               MODEL_WIDTH, MODEL_HEIGHT], 
                               dtype = np.float32) 
        return image_info

    def inference(self, input_data):
        return self._model.execute(input_data)

    def post_process(self, infer_output, origin_img):
        results = []
        box_num = int(infer_output[1][0, 0])
        if box_num > 0:
            box_info = infer_output[0].flatten()
            scalex = origin_img.width / MODEL_WIDTH
            scaley = origin_img.height / MODEL_HEIGHT
            if scalex > scaley:
                scaley = scalex
            
            for n in range(box_num):
                ids = int(box_info[5 * box_num + n])
                label = labels[ids] if ids < len(labels) else str(ids)
                score = int(round(float(box_info[4 * box_num + n]) * 100))
                top_left_x = int(box_info[0 * box_num + n] * scalex)
                top_left_y = int(box_info[1 * box_num + n] * scaley)
                bottom_right_x = int(box_info[2 * box_num + n] * scalex)
                bottom_right_y = int(box_info[3 * box_num + n] * scaley)
                
                results.append({
                    "class": label,
                    "class_id": ids,
                    "confidence": score,
                    "bbox": [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
                })
        return results


def kzzk_cv(modelfile, imagefile):
    global _acl_resource, _model_instance
    
    if not os.path.exists(imagefile):
        raise FileNotFoundError(f"图像文件不存在: {imagefile}")
    if not os.path.exists(modelfile):
        raise FileNotFoundError(f"模型文件不存在: {modelfile}")
    
    if _acl_resource is None:
        _acl_resource = AclLiteResource()
        _acl_resource.init()
    
    if _model_instance is None or _model_instance._model_path != modelfile:
        _model_instance = YOLOv3(modelfile)
        ret = _model_instance.init()
        utils.check_ret("YOLOv3 init", ret)
    
    start_time = time.perf_counter()
    
    image = AclLiteImage(imagefile)
    resized_image = _model_instance.pre_process(image)
    image_info = _model_instance.construct_image_info()
    result = _model_instance.inference([resized_image, image_info])
    output = _model_instance.post_process(result, image)
    
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
        f.write(f"YOLOv3图像检测推理\n")
        f.write(f"Total objects: {len(results['results'])}\n")
        f.write(f"Format: [ID] (Class Name) [x1, y1, x2, y2] Confidence\n")
        for i, res in enumerate(results['results'], 1):
            bbox_str = ",".join(map(str, res['bbox']))
            f.write(f"[{i}] {res['class_id']} ({res['class']}) [{bbox_str}] {res['confidence']}%\n")
    
    return filepath


def print_help():
    print("YOLOv3图像检测推理")
    print("Usage: yolov3.py [OPTIONS]")
    print("Options:")
    print("  --modelfile MODEL      指定模型文件路径（.om 文件）")
    print("  --imagefile IMAGE      指定输入图像路径")
    print("  --help                 显示帮助信息")
    print("")
    print("Examples:")
    print("  python yolov3.py --modelfile /home/HwHiAiUser/cann/model/yolov3_yuv.om --imagefile /home/HwHiAiUser/cann/data/dog.jpg")


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
        print(f"{result['model_name']}图像检测推理")
        print(f"Total objects: {len(result['results'])}")
        print(f"Format: [ID] (Class Name) [x1, y1, x2, y2] Confidence")
        for i, res in enumerate(result['results'], 1):
            bbox_str = ",".join(map(str, res['bbox']))
            print(f"[{i}] {res['class_id']} ({res['class']}) [{bbox_str}] {res['confidence']}%")
        
        save_path = save_results(result, result['model_name'], imagefile)
        print(f"推理结果已保存到: {save_path}")
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

