#!/bin/bash

#SBATCH --job-name=v10m-ours

#SBATCH --partition=gpu
#SBATCH --gres=gpu:4 # 2 gpus
#SBATCH --ntasks=8 # 1 task per gpu

#SBATCH --cpus-per-task=4  # 32 cpus, or hyper threads, total
#SBATCH --mem=150G # 8G is reserved for swap


# baseline
mkdir -p ./runs/detect/coco/yolo10m_500e_ours
python -m torch.distributed.run --nproc_per_node 4 train.py \
    --model-config ultralytics/cfg/models/v10/yolov10m_FINE.yaml \
    --project runs/detect/coco/yolo10m_500e_ours \
    2>&1 | tee ./runs/detect/coco/yolo10m_500e_ours/train_noPosEmbed_noCosFor.log

# # ours
# mkdir -p ./runs/detect/coco/yolo10l_500e_ours
# python -m torch.distributed.run --nproc_per_node 4 train.py \
#     --model-config ultralytics/cfg/models/v10/yolov10l_FINE.yaml \
#     --project runs/detect/coco/yolo10l_500e_ours \
#     2>&1 | tee ./runs/detect/coco/yolo10l_500e_ours/train_noPosEmbed_nhead32-16_noCosFor.log