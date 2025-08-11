#!/bin/bash

#SBATCH --job-name=v10m-ours

#SBATCH --partition=gpu
#SBATCH --gres=gpu:4 # 2 gpus
#SBATCH --ntasks=8 # 1 task per gpu

#SBATCH --cpus-per-task=4  # 32 cpus, or hyper threads, total
#SBATCH --mem=150G # 8G is reserved for swap

# virtual environment activation
# source ~/myenv/bin/activate

conda activate yolov10

# train command
# mkdir -p ./runs/detect/coco/yolo10m_500e_ours
# python -m torch.distributed.run --nproc_per_node 4 train.py \
#     --model-config ultralytics/cfg/models/v10/yolov10m_FINE.yaml \
#     --project runs/detect/coco/yolo10m_500e_ours \
#     2>&1 | tee ./runs/detect/coco/yolo10m_500e_ours/train_posEmbed_nhead16-8_noCosFor.log

yolo detect train data=coco.yaml \
    model=./ultralytics/cfg/models/v10/yolov10m_FINE.yaml \
    project=./runs/detect/coco/yolo10m_500e_ours \
    epochs=500 batch=128 nbs=256 \
    imgsz=640 device=0,1,2,3 \
    optimizer='SGD' save_period=100 patience=100 \
    scale=0.9 mixup=0.1 copy_paste=0.1 \
    2>&1 | tee ./runs/detect/coco/yolo10m_500e_ours/train_posEmbed_nhead16-8_noCosFor_nearest.log