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
model = YOLO(args.model_config)
model.info()

model.export(format="onnx")


'''


python export.py \
    --model-config /home/hslee/Desktop/Embedded_AI/CLASS-FPN/new_ultralytics/ultralytics/cfg/models/v5/yolov5s.yaml \
    --project runs/export/yolov5s_baseline \
    2>&1 | tee ./runs/export/yolov5s_baseline/export_onnx.log


    
'''