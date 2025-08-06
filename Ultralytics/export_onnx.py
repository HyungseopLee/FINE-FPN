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

trt_model = model.export(
    format="onnx",
    half=True,
    device=0,
    imgsz=640,
    batch=1,
    
)


'''

# export onnx
python export_onnx.py \
    --model-config /home/hslee/CLASS-FPN/new_ultralytics/ultralytics/cfg/models/v8/yolov8s_FINE.yaml \
    --project runs/export/yolov8s_ours \
    2>&1 | tee ./runs/export/yolov8s_ours/export_onnx.log

# onnx to trt
trtexec \
    --onnx=/home/hslee/CLASS-FPN/new_ultralytics/yolov8s_FINE.onnx \
    --saveEngine=/home/hslee/CLASS-FPN/new_ultralytics/yolov8s_FINE.engine \
    --fp16 \
    2>&1 | tee ./runs/export/yolov8s_ours/onnx_trt.log


    
--memPoolSize=workspace:512 \    

    
# benchmarking with trtexec
trtexec \
    --loadEngine=/home/hslee/CLASS-FPN/new_ultralytics/yolov8s.engine \
    --fp16 \
    --warmUp=300 \
    --iterations=1300 \
    --avgRuns=10 \
    --verbose \
    --useCudaGraph \
    --useSpinWait \
    2>&1 | tee ./runs/export/yolov8s_baseline/benchmark_fp16.log
trtexec \
    --loadEngine=/home/hslee/CLASS-FPN/new_ultralytics/yolov8s_FINE.engine \
    --fp16 \
    --warmUp=300 \
    --iterations=1300 \
    --avgRuns=10 \
    --verbose \
    --useCudaGraph \
    --useSpinWait \
    2>&1 | tee ./runs/export/yolov8s_ours/benchmark_fp16.log

'''