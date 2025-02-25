import argparse
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def get_args():
    parser = argparse.ArgumentParser(description='Object Detection with PyTorch and GPU')
    parser.add_argument('--input', type=str, required=True, help='入力画像のパス')
    parser.add_argument('--output', type=str, required=True, help='出力画像の保存先パス')
    parser.add_argument('--threshold', type=float, default=0.5, help='検出閾値')
    return parser.parse_args()

def load_model():
    # GPU使用可能かどうかを確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        raise RuntimeError("GPU が利用できません。環境を確認してください。")
    
    print(f"Device: {device}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 事前学習済みモデルの読み込み
    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    model.eval()
    model.to(device)
    
    return model, device

def detect_objects(model, image_path, device, threshold=0.5):
    # 画像の読み込み
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # PyTorch形式に変換
    tensor_image = F.to_tensor(image)
    
    # GPUに転送
    tensor_image = tensor_image.to(device)
    
    # 推論
    start_time = time.time()
    with torch.no_grad():
        predictions = model([tensor_image])
    end_time = time.time()
    
    # 結果を取得
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    
    # 閾値以上のスコアを持つ検出結果をフィルタリング
    mask = pred_scores >= threshold
    boxes = pred_boxes[mask]
    scores = pred_scores[mask]
    labels = pred_labels[mask]
    
    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time:.4f} seconds")
    
    return image, boxes, scores, labels

def visualize_results(image, boxes, scores, labels, output_path):
    # COCO データセットのクラス名
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # 結果の可視化
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()
    
    # 各検出結果を描画
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        
        # バウンディングボックスを描画
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        
        # クラス名とスコアのテキストを描画
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        text = f"{class_name}: {score:.2f}"
        plt.text(x1, y1, text, fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.axis('off')
    plt.tight_layout()
    
    # 結果を保存
    plt.savefig(output_path, bbox_inches='tight')
    print(f"検出結果を {output_path} に保存しました")
    
    # 表示（オプション）
    # plt.show()

def main():
    args = get_args()
    
    try:
        # モデルのロード
        model, device = load_model()
        
        # 物体検出の実行
        image, boxes, scores, labels = detect_objects(model, args.input, device, args.threshold)
        
        # 結果の可視化と保存
        visualize_results(image, boxes, scores, labels, args.output)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()