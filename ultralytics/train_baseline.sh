#!/bin/bash

#SBATCH --job-name=yolov8m_300e_baseline
#SBATCH --output=runs/detect/coco/yolov8m_300e_baseline/%j.out
#SBATCH --error=runs/detect/coco/yolov8m_300e_baseline/%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:4 # 2 gpus
#SBATCH --ntasks=4 # 1 task per gpu

#SBATCH --cpus-per-task=16  # 64 cpus, or hyper threads, total
#SBATCH --mem=256G # 8G is reserved for swap

# virtual environment activation
source ~/anaconda3/bin/activate py313

# train command
CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.run --nproc_per_node 4 train.py \
    --model-config /home/hslee/FINE-FPN/ultralytics/ultralytics/cfg/models/v8/yolov8m.yaml \
    --project runs/detect/coco/yolov8m_300e_baseline \
    2>&1 | tee ./runs/detect/coco/yolov8m_300e_baseline/train.log