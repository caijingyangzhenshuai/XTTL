import sys
import os
import time
import argparse
import datetime

# 自动切换到脚本所在目录，避免在不同目录执行找不到模块/文件
SRC_PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(SRC_PATH)
sys.path.insert(0, SRC_PATH)
sys.path.append(os.path.join(SRC_PATH, ".."))
sys.path.append(os.path.join(SRC_PATH, "../examples/samples/python/common/"))
sys.path.append(os.path.join(SRC_PATH, "../examples/samples/python/common/acllite"))

# 数据与输出统一目录
DEFAULT_DATA_DIR = "/home/HwHiAiUser/cann/data"
if not os.path.isdir(DEFAULT_DATA_DIR):
    # 兼容其他部署位置
    DEFAULT_DATA_DIR = os.path.join(SRC_PATH, "../data")
    if not os.path.isdir(DEFAULT_DATA_DIR):
        DEFAULT_DATA_DIR = SRC_PATH

import numpy as np
import acl
import acllite_logger as acl_log
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource

# 延迟加载 OpenCV（避免 libGL 缺失问题）
_cv2 = None


def _get_cv2():
    """延迟加载 cv2，若不可用返回 None"""
    global _cv2
    if _cv2 is None:
        try:
            import cv2
            _cv2 = cv2
        except ImportError:
            _cv2 = False
    return _cv2 if _cv2 else None


MODEL_WIDTH = 513
MODEL_HEIGHT = 513

# Pascal VOC 21 类调色板 (RGB)
PALETTE = np.array([
    (0, 0, 0),         (128, 0, 0),     (0, 128, 0),     (128, 128, 0),
    (0, 0, 128),       (128, 0, 128),   (0, 128, 128),   (128, 128, 128),
    (64, 0, 0),        (192, 0, 0),     (64, 128, 0),    (192, 128, 0),
    (64, 0, 128),      (192, 0, 128),   (64, 128, 128),  (192, 128, 128),
    (0, 64, 0),        (128, 64, 0),    (0, 192, 0),     (128, 192, 0),
    (0, 64, 128),
], dtype=np.uint8)

# VOC 类别名（Pascal VOC 2012）
CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "dining table", "dog", "horse",
    "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor",
]


def print_help():
    print("Deeplabv3语义分割推理")
    print("Usage: deeplabv3.py [OPTIONS]")
    print("Options:")
    print("  --modelfile MODEL      指定模型文件路径（.om 文件）")
    print("  --imagefile IMAGE      指定输入图像路径")
    print("  --output_dir DIR       指定输出目录（默认: /home/HwHiAiUser/cann/data）")
    print("  --help                 显示帮助信息")
    print("")
    print("Examples:")
    print("  python deeplabv3.py --modelfile /home/HwHiAiUser/cann/model/deeplabv3_plus.om --imagefile /home/HwHiAiUser/cann/data/cat.png")


def parse_args():
    if len(sys.argv) == 1:
        print_help()
        sys.exit(0)

    parser = argparse.ArgumentParser(description='Deeplabv3 语义分割推理')
    parser.add_argument('--modelfile', type=str,
                        default=os.path.join(SRC_PATH, "../model/deeplabv3_plus.om"),
                        help='指定模型文件路径（.om 文件）')
    parser.add_argument('--imagefile', type=str, default=None,
                        help='指定输入图像路径')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_DATA_DIR,
                        help='指定输出目录')
    return parser.parse_args()


def preprocess(pic_path):
    """PIL 读取 + resize 到 513x513"""
    from PIL import Image
    pil_img = Image.open(pic_path).convert('RGB')
    orig_size = (pil_img.width, pil_img.height)  # (W, H)
    rgb_img = pil_img.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.BILINEAR)
    img = np.array(rgb_img).astype(np.int8)
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    return orig_size, img


def _save_png(out_path, rgb_array):
    """PIL 保存 PNG"""
    from PIL import Image
    Image.fromarray(rgb_array.astype(np.uint8), mode='RGB').save(out_path)


def _resize_nearest_pil(arr, new_w, new_h):
    from PIL import Image
    pil = Image.fromarray(arr.astype(np.uint8), mode='RGB')
    pil = pil.resize((new_w, new_h), Image.NEAREST)
    return np.array(pil)


def _imread_rgb(pic_path):
    from PIL import Image
    return np.array(Image.open(pic_path).convert('RGB'))


def _to_bgr_via_cv2(rgb):
    cv2 = _get_cv2()
    if cv2 is not None:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb[:, :, ::-1].copy()


def _imwrite_via_cv2(out_path, bgr):
    cv2 = _get_cv2()
    if cv2 is not None:
        cv2.imwrite(out_path, bgr)
    else:
        rgb = bgr[:, :, ::-1]
        _save_png(out_path, rgb)


def _resize_nearest(rgb, w, h):
    cv2 = _get_cv2()
    if cv2 is not None:
        bgr = rgb[:, :, ::-1].copy()
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_NEAREST)
        return bgr[:, :, ::-1]
    return _resize_nearest_pil(rgb, w, h)


def _add_overlay(orig_rgb, color_rgb, alpha=0.6):
    """原图 + 彩色 mask 半透明叠加"""
    cv2 = _get_cv2()
    if cv2 is not None:
        orig_bgr = orig_rgb[:, :, ::-1].copy()
        color_bgr = color_rgb[:, :, ::-1].copy()
        overlay = cv2.addWeighted(orig_bgr, 1 - alpha, color_bgr, alpha, 0)
        return overlay[:, :, ::-1]
    return (orig_rgb.astype(np.float32) * (1 - alpha) +
            color_rgb.astype(np.float32) * alpha).astype(np.uint8)


def postprocess(result_list, orig_size, pic_path, output_dir):
    """生成彩色分割图（叠加到原图），返回保存路径"""
    result_img = result_list[0].reshape(MODEL_HEIGHT, MODEL_WIDTH)
    seg_map = result_img.astype(np.uint8)
    seg_map = np.clip(seg_map, 0, len(PALETTE) - 1)

    # 调色板上色
    color_map = PALETTE[seg_map]

    # resize 回原图
    orig_w, orig_h = orig_size
    color_map_full = _resize_nearest(color_map, orig_w, orig_h)
    orig_rgb = _imread_rgb(pic_path)

    # 叠加
    overlay_rgb = _add_overlay(orig_rgb, color_map_full, alpha=0.6)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pic_path))[0]
    out_pic = os.path.join(output_dir, base + "_seg.png")
    _save_png(out_pic, overlay_rgb)
    return out_pic


def collect_class_stats(seg_map, threshold_ratio=0.01):
    """统计每类像素占比，过滤掉占比 < threshold_ratio 的噪声类"""
    total = seg_map.size
    class_info = []
    for c in np.unique(seg_map):
        cnt = int(np.sum(seg_map == c))
        ratio = cnt * 100.0 / total
        if ratio < threshold_ratio:
            continue
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        class_info.append({
            "id": int(c),
            "name": name,
            "pixels": cnt,
            "ratio": round(ratio, 2),
        })
    # 按像素数降序
    class_info.sort(key=lambda x: x["pixels"], reverse=True)
    return class_info


def save_results(results, model_name, image_path, output_pic):
    """将结果保存为 txt 文件，存到与图像同目录（即 cann/data/）"""
    result_dir = os.path.dirname(image_path)
    if not result_dir:
        result_dir = os.getcwd()
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "%s_%s.txt" % (timestamp, model_name)
    filepath = os.path.join(result_dir, filename)

    lines = []
    lines.append("Deeplabv3语义分割推理")
    #lines.append("Input image: %s" % image_path)
    #lines.append("Image size: %dx%d" % (results["image_size"][0], results["image_size"][1]))
    lines.append("Segmentation image: %s" % output_pic)
    lines.append("Format: [ID] (Class Name) Pixels Ratio")
    for item in results["class_info"]:
        lines.append("[%d] %s (%s) %d %.2f%%" % (
            item["id"], item["id"], item["name"], item["pixels"], item["ratio"]))

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    return filepath


# 用于在多个脚本间共享 ACL 资源与已加载的模型
_acl_resource = None
_model = None
_model_path_cache = None


def kzzk_cv(modelfile, imagefile, output_dir=None):
    """kzzk_cv 函数接口（与 vgg16.py/yolov3.py 保持一致）

    Args:
        modelfile:  .om 模型文件路径
        imagefile: 输入图像路径（支持 PNG/JPG/BMP/TIFF）
        output_dir: 可选，分割结果保存目录，默认 DEFAULT_DATA_DIR

    Returns:
        dict: {
            "results": {...},      # 分割统计信息
            "infer_cost": int,     # 推理耗时（毫秒）
            "model_name": str,     # 模型名
            "output_pic": str,     # 分割可视化图片路径
            "result_txt": str,     # 分割结果 txt 路径
        }
    """
    global _acl_resource, _model, _model_path_cache

    if not os.path.exists(imagefile):
        raise FileNotFoundError("图像文件不存在: %s" % imagefile)
    if not os.path.exists(modelfile):
        raise FileNotFoundError("模型文件不存在: %s" % modelfile)

    if _acl_resource is None:
        _acl_resource = AclLiteResource()
        _acl_resource.init()

    if _model is None or _model_path_cache != modelfile:
        _model = AclLiteModel(modelfile)
        _model_path_cache = modelfile

    if output_dir is None:
        output_dir = DEFAULT_DATA_DIR

    orig_size, l_data = preprocess(imagefile)

    start = time.time()
    result_list = _model.execute([l_data])
    infer_cost = int(round((time.time() - start) * 1000))

    out_pic = postprocess(result_list, orig_size, imagefile, output_dir)

    seg_map = result_list[0].reshape(MODEL_HEIGHT, MODEL_WIDTH).astype(np.uint8)
    class_info = collect_class_stats(seg_map)

    model_name = os.path.splitext(os.path.basename(modelfile))[0]
    result_txt = save_results(
        {"image_size": list(orig_size), "class_info": class_info},
        model_name, imagefile, out_pic)

    return {
        "results": {
            "image_size": list(orig_size),
            "class_info": class_info,
        },
        "infer_cost": infer_cost,
        "model_name": model_name,
        "output_pic": out_pic,
        "result_txt": result_txt,
    }


def main():
    args = parse_args()

    if not args.imagefile:
        print_help()
        sys.exit(1)
    if not os.path.isfile(args.imagefile):
        acl_log.log_error("图像文件不存在: %s" % args.imagefile)
        sys.exit(1)
    if not os.path.isfile(args.modelfile):
        acl_log.log_error("模型文件不存在: %s" % args.modelfile)
        sys.exit(1)

    # 静默 ACL 库内部 [INFO] 日志（重定向 stdout 到 devnull）
    import io
    saved_stdout = sys.stdout
    devnull = io.StringIO()
    sys.stdout = devnull
    # 让所有底层 print 都被静默
    print = lambda *a, **k: None  # noqa: E731

    try:
        # ACL 初始化
        acl_resource = AclLiteResource()
        acl_resource.init()
        model = AclLiteModel(args.modelfile)

        orig_size, l_data = preprocess(args.imagefile)
        start = time.time()
        result_list = model.execute([l_data])
        cost_ms = int((time.time() - start) * 1000)

        out_pic = postprocess(result_list, orig_size, args.imagefile, args.output_dir)

        seg_map = result_list[0].reshape(MODEL_HEIGHT, MODEL_WIDTH).astype(np.uint8)
        class_info = collect_class_stats(seg_map)

        model_name = os.path.splitext(os.path.basename(args.modelfile))[0]
        result_txt = save_results(
            {"image_size": list(orig_size), "class_info": class_info},
            model_name, args.imagefile, out_pic)
    finally:
        # 恢复 stdout 与 print
        sys.stdout = saved_stdout
        import builtins
        builtins.print = _builtin_print

    # 输出
    _builtin_print("Total inference cost: %dms" % cost_ms)
    _builtin_print("Inference Results: ")
    _builtin_print("Deeplabv3语义分割推理")
    #_builtin_print("Input image: %s" % args.imagefile)
    #_builtin_print("Image size: %dx%d" % (orig_size[0], orig_size[1]))
    _builtin_print("Output classes count: %d" % len(class_info))
    #_builtin_print("Top classes: ")
    _builtin_print("Format: [ID] (Class Name) [Pixels] [Ratio]")
    for i, item in enumerate(class_info, 1):
        _builtin_print("Class %d: [%d] %s (%s) [%d] [%.2f%%]" % (
            i, item["id"], item["id"], item["name"],
            item["pixels"], item["ratio"]))
    _builtin_print("Segmentation image: %s" % out_pic)
    _builtin_print("推理结果已保存到: %s" % result_txt)

    # 用 os._exit() 直接退出，跳过 Python 退出时的 atexit 清理，
    # 避免 ACL 库析构/atexit 打印 [INFO] 资源释放日志
    os._exit(0)


_builtin_print = print  # 在模块顶部保存原始 print


if __name__ == '__main__':
    main()

