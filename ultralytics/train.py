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

# Train the model with 2 GPUs
results = model.train(
    # data="tinyperson-5.yaml",
    # data="VisDrone.yaml", 
    # data="coco128-seg.yaml", 
    # data="VOC.yaml",
    data="coco.yaml",
    
    epochs=300,
    
    # batch=32,   # 1 GPU
    # batch=64, # 2 GPUs
    batch=128, # 4 GPUs
    imgsz=640, 
    
    pretrained=False,
    project=args.project,
    
    # device='0',  # single GPU
    # device='0,1',
    device='0,1,2,3',
    
    save_period=50,
    patience=50,
    resume=False,
)



'''

python -m torch.distributed.run --nproc_per_node 2 train.py \
    2>&1 | tee ./runs/detect/VisDrone/train.log


export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_DEBUG=INFO
python -m torch.distributed.run --nproc_per_node 2 train.py \
    --model-config /home/hslee/FINE-FPN/ultralytics/ultralytics/cfg/models/v10/yolov10m.yaml \
    --project runs/detect/coco/*yolov10m_300e_baseline \
    2>&1 | tee ./runs/detect/coco/*yolov10m_300e_baseline/train.log
    
python -m torch.distributed.run --nproc_per_node 4 train.py \
    --model-config /home/hslee/FINE-FPN/ultralytics/ultralytics/cfg/models/v8/yolov8m.yaml \
    --project runs/detect/coco/yolov8m_300e_baseline \
    2>&1 | tee ./runs/detect/coco/yolov8m_300e_baseline/train.log
    
    

python -m torch.distributed.run --nproc_per_node 2 train.py \
    --model-config /home/hslee/FINE-FPN/ultralytics/ultralytics/cfg/models/v8/yolov8m.yaml \
    --project runs/detect/coco/*yolov8m_300e_TransNormer \
    2>&1 | tee ./test.log
    
python -m torch.distributed.run --nproc_per_node 2 train.py \
    --weights=''
        
'''