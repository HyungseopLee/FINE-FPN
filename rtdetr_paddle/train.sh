#!/bin/bash

#SBATCH --job-name=p-RTDETR50

#SBATCH --partition=gpu
#SBATCH --gres=gpu:4 # 2 gpus
#SBATCH --ntasks=4 # 1 task per gpu

#SBATCH --cpus-per-task=8  # 32 cpus, or hyper threads, total
#SBATCH --mem=190G # 8G is reserved for swap

# virtual environment activation
# source ~/myenv/bin/activate

# train command

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --fleet --eval \
    --vdl_log_dir ./output/rtdetr_r50vd_6x_coco_ours \
    2>&1 | tee ./output/rtdetr_r50vd_6x_coco_ours/train_noPosEmbed.log