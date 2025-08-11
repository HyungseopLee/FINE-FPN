from ultralytics import YOLO
from ultralytics import RTDETR

import torch
import torch.distributed as dist
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model-config", type=str, default='yolov8s.yaml', help="Model configuration file")
parser.add_argument("--project", type=str, default='runs/detect/tinyperson-v5/baseline-yolov8s_100e', help="Project directory")
args = parser.parse_args()
print(f"args.model_config: {args.model_config}")

# Load a model
# model = YOLO('yolov5n_tile.yaml')  # if i want to train from scratch, i should use '*.yaml' instead of '.pt'
model = YOLO(args.model_config)

# # resume
# model = YOLO('/home/hslee/FINE-FPN/ultralytics/runs/detect/coco/*yolov10m_300e_baseline/train/weights/last.pt')
# weights = torch.load('/home/hslee/FINE-FPN/ultralytics/runs/detect/coco/*yolov10m_300e_baseline/train/weights/last.pt')
# model.model.load_state_dict(weights, strict=True)


model.info()

results = model.train(
    
    data="coco.yaml",
    
    epochs=500,
    batch=128, # 4 GPUs
    nbs=256,
    
    imgsz=640,
    project=args.project,
    
    device='0,1,2,3', 
    
    save_period=100,
    patience=100,
    
    optimizer='SGD',
    
    # yolov10-M settings
    scale=0.9,
    mixup=0.1,
    copy_paste=0.1,
    
    # # yolov10-L settings
    # scale=0.9,
    # mixup=0.15,
    # copy_paste=0.3,
    # optimizer='SGD',
)


'''

/home2/hslee/FINE-FPN/FINE_to_YOLOv10/runs/detect/coco/yolo10m_500e_ours/train/weights/last.pt

yolo detect train data=coco.yaml \
    model=./ultralytics/cfg/models/v10/yolov10m_FINE.yaml \
    project=./runs/detect/coco/yolo10m_500e_ours \
    epochs=500 batch=128 nbs=256 \
    imgsz=640 device=0,1,2,3 \
    optimizer='SGD' save_period=100 patience=100 \
    scale=0.9 mixup=0.1 copy_paste=0.1 \
    2>&1 | tee ./runs/detect/coco/yolo10m_500e_ours/train_posEmbed_nhead16-8_noCosFor.log
    
    
yolo detect train data=coco.yaml \
    model=/home2/hslee/FINE-FPN/FINE_to_YOLOv10/runs/detect/coco/yolo10m_500e_ours/train/weights/last.pt \
    project=./runs/detect/coco/yolo10m_500e_ours \
    epochs=500 batch=128 nbs=256 \
    imgsz=640 device=0,1,2,3 \
    optimizer='SGD' save_period=100 patience=100 \
    scale=0.9 mixup=0.1 copy_paste=0.1 \
    resume=True \
    2>&1 | tee ./runs/detect/coco/yolo10m_500e_ours/train_posEmbed_nhead16-8_noCosFor.log
    
'''