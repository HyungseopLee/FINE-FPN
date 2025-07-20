#!/bin/bash

#SBATCH --job-name=yolov8m_300e_ReLUCosFormer
#SBATCH --output=runs/detect/coco/yolov8m_300e_ReLUCosFormer/%j.out
#SBATCH --error=runs/detect/coco/yolov8m_300e_ReLUCosFormer/%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2 # 2 gpus
#SBATCH --ntasks=2 # 1 task per gpu

#SBATCH --cpus-per-task=16  # 64 cpus, or hyper threads, total
#SBATCH --mem=128G # 8G is reserved for swap

# virtual environment activation
# source ~/myenv/bin/activate

# train command
python -m torch.distributed.run --nproc_per_node 2 train.py \
    --model-config /home/hslee/FINE-FPN/ultralytics/ultralytics/cfg/models/v8/yolov8m_SAFusion.yaml \
    --project runs/detect/coco/*yolov8m_300e_SemanticCosFormer \
    2>&1 | tee runs/detect/coco/*yolov8m_300e_SemanticCosFormer/train.log