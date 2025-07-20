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
# weights = torch.load('/home/hslee/FINE-FPN/yolov10/runs/detect/coco/*yolo10l_500e_NormAttention/train_4GPU_nbs256_h16_posEmbed/weights/last.pt')
# model.model.load_state_dict(weights, strict=True)

model.info()

results = model.train(
    
    data="coco.yaml",
    
    epochs=500,
    batch=64, # 4 GPUs
    nbs=256,
    
    imgsz=640, 
    
    project=args.project,
    
    device='0,1', # idle GPUs
    
    save_period=50,
    patience=100,
    resume=False,
    workers=16,
    amp=True,
    
    # yolov10-L settings
    scale=0.9,
    mixup=0.15,
    copy_paste=0.3,
    optimizer='SGD',
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
    
python train.py \
    --model-config /home/hslee/FINE-FPN/ultralytics/ultralytics/cfg/models/v10/yolov10m_SAFusion.yaml \
    --project runs/detect/coco/*yolov10m_300e_SemanticReLULinear \
    2>&1 | tee ./runs/detect/coco/*yolov10m_300e_SemanticReLULinear/train_noPosEmbed.log
    
    

python -m torch.distributed.run --nproc_per_node 2 train.py \
    --model-config /home/hslee/SONeck/yolov10/ultralytics/cfg/models/v10/yolov10l_FINE.yaml \
    --project runs/detect/coco/yolo10l_500e_ours \
    2>&1 | tee ./runs/detect/coco/yolo10l_500e_ours/train_2GPU_PosEmbed_relu.log
    
python -m torch.distributed.run --nproc_per_node 2 train.py \
    --weights=''
   
   
python -m torch.distributed.run --master_port=1235 --nproc_per_node 1 train.py \
    --model-config /home/hslee/FINE-FPN/new_ultralytics/ultralytics/cfg/models/v10/yolov10s_FINE.yaml \
    2>&1 | tee ./runs/detect/coco/test.log
    
yolo detect train data=coco.yaml model=yolov10l.yaml epochs=500 batch=128 nbs=256 imgsz=640 device=0,1,2,3 \
    2>&1 | tee ./test.log
    
    
'''