#!/bin/bash

#SBATCH --job-name=v10s_ours

#SBATCH --partition=gpu
#SBATCH --gres=gpu:4 # 2 gpus
#SBATCH --ntasks=4 # 1 task per gpu

#SBATCH --cpus-per-task=16  # 64 cpus, or hyper threads, total
#SBATCH --mem=500G # 8G is reserved for swap

# virtual environment activation
# source ~/myenv/bin/activate

# train command

mkdir -p ./runs/detect/coco/yolo10l_500e_ours
python -m torch.distributed.run --master_port=1234 --nproc_per_node 4 train.py \
    --model-config /home/hslee/FINE-FPN/yolov10/ultralytics/cfg/models/v10/yolov10l_FINE.yaml \
    --project runs/detect/coco/yolo10l_500e_ours \
    2>&1 | tee ./runs/detect/coco/yolo10l_500e_ours/train_4GPU_nbs256_h3216_noPosEmbed.log