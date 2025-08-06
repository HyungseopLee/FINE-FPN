#!/bin/bash

#SBATCH --job-name=FRCN-R50-SNI

#SBATCH --partition=gpu
#SBATCH --gres=gpu:4 # 2 gpus
#SBATCH --ntasks=4 # 1 task per gpu

#SBATCH --cpus-per-task=16  # 64 cpus, or hyper threads, total
#SBATCH --mem=300G # 8G is reserved for swap

# virtual environment activation
# source ~/myenv/bin/activate

# train command
torchrun --nproc_per_node=4 train.py \
    --dataset coco --data-path /home/hslee/FINE-FPN/data/coco \
    --model fasterrcnn_resnet50_fpn \
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 \
    --weights-backbone ResNet50_Weights.IMAGENET1K_V1 \
    --batch-size 2 --lr 0.01 \
    --output-dir ./outputs/fasterrcnn-r50/SNI \
    --print-freq 100 \
    2>&1 | tee ./outputs/fasterrcnn-r50/SNI/train_4GPUs.log