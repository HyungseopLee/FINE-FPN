#!/bin/bash

#SBATCH --job-name=rtdetrv1_pr50_ours_posEmbed_LN

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2 # 2 gpus
#SBATCH --ntasks=2 # 1 task per gpu

#SBATCH --cpus-per-task=16  # 64 cpus, or hyper threads, total
#SBATCH --mem=128G # 8G is reserved for swap

# virtual environment activation
# source ~/myenv/bin/activate

# train command
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 tools/train.py \
    -c /home/hslee/FINE-FPN/rtdetrv2_pytorch2/configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
    2>&1 | tee ./output/*v1_r18vd_72e_coco_SemanticAlign-ReLUCosFormer/train_posEmbed_LN.log