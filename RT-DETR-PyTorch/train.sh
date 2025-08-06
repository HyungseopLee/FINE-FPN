#!/bin/bash

#SBATCH --job-name=rt50-PAN-ours

#SBATCH --partition=gpu
#SBATCH --gres=gpu:4 # 2 gpus
#SBATCH --ntasks=4 # 1 task per gpu

#SBATCH --cpus-per-task=16  # 64 cpus, or hyper threads, total
#SBATCH --mem=300G # 8G is reserved for swap

# virtual environment activation
# source ~/myenv/bin/activate

# train command


export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 \
    tools/train.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --output-dir=./output/*v1_r50vd_72e_coco_ours \
    --use-amp --seed=0 2>&1 | tee ./output/*v1_r50vd_72e_coco_ours/train_noPosEmbed_nhead-16_L2H.log