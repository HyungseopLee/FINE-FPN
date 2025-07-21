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

tensorrt_model = model.export(
    format="engine",
    half=True,

)

# trtexec


'''

# export
python export_trt.py \
    --model-config /home/hslee/CLASS-FPN/new_ultralytics/ultralytics/cfg/models/v5/yolov5s.yaml \
    --project runs/export/yolov5s_baseline \
    2>&1 | tee ./runs/export/yolov5s_baseline/export_trt.log

# benchmarking with trtexec
trtexec \
    --loadEngine=/home/hslee/CLASS-FPN/new_ultralytics/yolov5s.engine \
    --fp16 \
    --warmUp=200 \
    --iterations=1200 \
    --avgRuns=50 \
    --verbose \
    --useCudaGraph \
    --useSpinWait \
    2>&1 | tee ./runs/export/yolov5s_baseline/benchmark_fp16.log

'''