import sys
import os

path = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(path)
sys.path.insert(0, parent)

from python import kzzk_cv


def print_help():
    print("智能加速卡协同推理软件")
    print("Python 版命令行接口")
    print("Usage: kzzk_cv.py [OPTIONS]")
    print("Options:")
    print("  --modelfile MODEL      指定模型文件路径（.om 文件）")
    print("  --imagefile IMAGE      指定输入图像路径")
    print("  --help                 显示帮助信息")
    print("")
    print("支持的模型:")
    print("  YOLOv4, YOLOv3, VGG-SSD, RT-DETR  (目标检测)")
    print("  ResNet50, MobileNetV1, VGG16       (图像分类)")
    print("  DeepLabV3                          (图像分割)")
    print("")
    print("Examples:")
    print("  python kzzk_cv.py --modelfile /home/HwHiAiUser/cann/model/yolov4.om --imagefile /home/HwHiAiUser/cann/data/dog.jpg")
    print("  python kzzk_cv.py --modelfile /home/HwHiAiUser/cann/model/resnet50.om --imagefile /home/HwHiAiUser/cann/data/dog.jpg")


def main():
    args = sys.argv[1:]

    if not args or "--help" in args or "-h" in args:
        print_help()
        return

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
    except Exception as e:
        print("错误: %s" % str(e))
        sys.exit(1)

    print("Total inference cost: %dms" % result['infer_cost'])
    print("Inference Results: ")

    model_type = result['model_type']
    model_name = result['model_name']

    if model_type == 'detection':
        print("%s目标检测推理" % model_name)
        print("Total objects: %d" % len(result['results']))
        print("Format: [ID] (Class Name) [x1, y1, x2, y2] Confidence")
        for i, det in enumerate(result['results'], 1):
            bbox = det['bbox']
            print("[%d] %d (%s) [%d,%d,%d,%d] %d%%" % (
                i, det['class_id'], det['label'],
                bbox[0], bbox[1], bbox[2], bbox[3],
                int(round(det['confidence'] * 100))))

    elif model_type == 'classification':
        print("%s图像分类推理" % model_name)
        print("Top 5 classes:")
        print("Format: [ID Confidence]")
        for i, res in enumerate(result['results'], 1):
            print("Result %d: [%d %d%%]" % (
                i, res['class_id'], res['confidence']))

    elif model_type == 'segmentation':
        print("%s语义分割推理" % model_name)
        class_info = result['results']['class_info']
        print("Output classes count: %d" % len(class_info))
        print("Format: [ID] (Class Name) [Pixels] [Ratio]")
        for i, cls in enumerate(class_info, 1):
            print("Class %d: [%d] %d (%s) [%d] [%.2f%%]" % (
                i, cls['id'], cls['id'], cls['name'],
                cls['pixels'], cls['ratio']))

        seg_img_path = save_segmentation_image(result, imagefile)
        print("Segmentation image: %s" % seg_img_path)

    save_result_to_file(result, imagefile)


def save_segmentation_image(result, imagefile):
    from python.models.segmentation.deeplabv3 import DeepLabV3Segment
    out_dir = os.path.dirname(imagefile)
    if not out_dir:
        out_dir = "."
    base = os.path.splitext(os.path.basename(imagefile))[0]
    out_path = os.path.join(out_dir, base + "_seg.png")
    seg_map = result['results']['seg_map']
    DeepLabV3Segment.save_seg_image(imagefile, seg_map, out_path)
    return out_path


def save_result_to_file(result, imagefile):
    import datetime
    out_dir = os.path.dirname(imagefile)
    if not out_dir:
        out_dir = "."
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = result['model_name']
    model_type = result['model_type']
    txt_path = os.path.join(out_dir, "%s_%s.txt" % (timestamp, model_name))

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Total inference cost: %dms\n" % result['infer_cost'])
        f.write("Inference Results:\n")

        if model_type == 'detection':
            f.write("%s目标检测推理\n" % model_name)
            f.write("Total objects: %d\n" % len(result['results']))
            f.write("Format: [ID] (Class Name) [x1, y1, x2, y2] Confidence\n")
            for i, det in enumerate(result['results'], 1):
                bbox = det['bbox']
                f.write("[%d] %d (%s) [%d,%d,%d,%d] %d%%\n" % (
                    i, det['class_id'], det['label'],
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    int(round(det['confidence'] * 100))))

        elif model_type == 'classification':
            f.write("%s图像分类推理\n" % model_name)
            f.write("Top 5 classes:\n")
            f.write("Format: [ID Confidence]\n")
            for i, res in enumerate(result['results'], 1):
                f.write("Result %d: [%d %d%%]\n" % (
                    i, res['class_id'], res['confidence']))

        elif model_type == 'segmentation':
            seg_data = result['results']
            class_info = seg_data['class_info']
            f.write("%s语义分割推理\n" % model_name)
            f.write("Output classes count: %d\n" % len(class_info))
            f.write("Format: [ID] (Class Name) [Pixels] [Ratio]\n")
            for i, cls in enumerate(class_info, 1):
                f.write("Class %d: [%d] %d (%s) [%d] [%.2f%%]\n" % (
                    i, cls['id'], cls['id'], cls['name'],
                    cls['pixels'], cls['ratio']))

    print("推理结果已保存到: %s" % txt_path)


if __name__ == '__main__':
    main()

