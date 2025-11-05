import cv2
import os
from pathlib import Path
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # 若有数字专属权重请替换

train_dir = 'train'

for num_dir in os.listdir(train_dir):
    num_path = os.path.join(train_dir, num_dir)
    if not os.path.isdir(num_path):
        continue
    output_dir = f"{num_dir}_new"
    os.makedirs(output_dir, exist_ok=True)
    for img_file in os.listdir(num_path):
        img_path = os.path.join(num_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        results = model(img)
        boxes = results.xyxy[0].cpu().numpy()

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2, conf, cls = box
            crop_img = img[int(y1):int(y2), int(x1):int(x2)]
            crop_name = img_file if len(boxes) == 1 else f"{Path(img_file).stem}_{idx}.jpg"
            save_path = os.path.join(output_dir, crop_name)
            cv2.imwrite(save_path, crop_img)

print("裁剪完成，每类数字的新图片已存入相应 *_new 文件夹！")
