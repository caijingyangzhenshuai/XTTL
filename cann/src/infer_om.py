import os
import random
import numpy as np
import cv2
import argparse
from pathlib import Path

try:
    from ais_bench.infer.interface import InferSession
except ImportError:
    print("Error: ais_bench not installed. Install with: pip install ais_bench")
    exit(1)

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def preprocess_image(image_path, input_size=640):
    """Preprocess image for inference"""
    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]

    # Resize
    img_resized = cv2.resize(img, (input_size, input_size))

    # Convert to RGB and normalize
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0

    # Transpose to CHW format
    img_input = np.transpose(img_normalized, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)

    return img_input, img, (orig_h, orig_w)

def draw_boxes(image, labels, boxes, scores, threshold=0.5):
    """Draw bounding boxes on image"""
    h, w = image.shape[:2]

    for label, box, score in zip(labels, boxes, scores):
        if score < threshold:
            continue

        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label_text = f"{COCO_CLASSES[int(label)]}: {score:.2f}"
        cv2.putText(image, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def main():
    parser = argparse.ArgumentParser(description='RT-DETR OM inference on COCO2017')
    parser.add_argument('-m', '--model', required=True, help='OM model file')
    parser.add_argument('-i', '--image', default=None, help='Image path (random if not specified)')
    parser.add_argument('-d', '--data-path', default='./dataset/coco', help='COCO dataset path')
    parser.add_argument('-o', '--output', default='result.jpg', help='Output image path')
    parser.add_argument('-s', '--input-size', type=int, default=640, help='Input size')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=int, default=0, help='NPU device ID')
    args = parser.parse_args()

    # Select image
    if args.image:
        image_path = args.image
    else:
        val_dir = Path(args.data_path) / 'val2017'
        images = list(val_dir.glob('*.jpg'))
        if not images:
            print(f"Error: No images found in {val_dir}")
            exit(1)
        image_path = str(random.choice(images))

    print(f"Image: {image_path}")

    # Load model
    print(f"Loading OM model: {args.model}")
    session = InferSession(device_id=args.device, model_path=args.model)

    # Preprocess
    img_input, orig_img, (orig_h, orig_w) = preprocess_image(image_path, args.input_size)
    orig_target_sizes = np.array([[orig_w, orig_h]], dtype=np.int64)  # [width, height] format

    print(f"Original size: {orig_h}x{orig_w}")
    print(f"Input shape: {img_input.shape}")

    # Inference
    print("Running inference...")
    outputs = session.infer([img_input, orig_target_sizes])

    labels = outputs[0][0]
    boxes = outputs[1][0]
    scores = outputs[2][0]

    print(f"\nDetections: {len(labels)}")
    print(f"Labels shape: {labels.shape}")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Scores shape: {scores.shape}")

    # Filter by threshold
    valid_idx = scores >= args.threshold
    labels = labels[valid_idx]
    boxes = boxes[valid_idx]
    scores = scores[valid_idx]

    print(f"\nDetections above threshold {args.threshold}: {len(labels)}")
    for i, (label, box, score) in enumerate(zip(labels, boxes, scores)):
        print(f"  {i+1}. {COCO_CLASSES[int(label)]}: {score:.3f} at {box}")

    # Draw results
    result_img = draw_boxes(orig_img.copy(), labels, boxes, scores, args.threshold)
    cv2.imwrite(args.output, result_img)
    print(f"\nResult saved to: {args.output}")

if __name__ == '__main__':
    main()

