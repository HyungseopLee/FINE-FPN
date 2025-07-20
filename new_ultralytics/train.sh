#!/bin/bash

#SBATCH --job-name=v10l_PAN-ours

#SBATCH --partition=gpu
#SBATCH --gres=gpu:4 # 2 gpus
#SBATCH --ntasks=4 # 1 task per gpu

#SBATCH --cpus-per-task=16  # 64 cpus, or hyper threads, total
#SBATCH --mem=500G # 8G is reserved for swap

# virtual environment activation
# source ~/myenv/bin/activate

# train command

# python -m torch.distributed.run --nproc_per_node 2 train.py \
#     --model-config /home/hslee/FINE-FPN/new_ultralytics/ultralytics/cfg/models/v10/yolov8m_FINE.yaml
#     --project runs/detect/coco/yolov8m_500e_baseline \
#     2>&1 | tee ./runs/detect/coco/yolov8m_500e_baseline/train.log

mkdir -p ./runs/detect/coco/yolo10l_500e_ours
python -m torch.distributed.run --master_port=1234 --nproc_per_node 4 train.py \
    --model-config /home/hslee/FINE-FPN/yolov10/ultralytics/cfg/models/v10/yolov10l_FINE.yaml \
    --project runs/detect/coco/yolo10l_500e_ours \
    2>&1 | tee ./runs/detect/coco/yolo10l_500e_ours/train_PosEmbed_relu_FFN2.log

yolo detect train data=coco.yaml model=/home/hslee/FINE-FPN/yolov10/ultralytics/cfg/models/v10/yolov10l_FINE.yaml epochs=500 batch=128 nbs=256 imgsz=640 device=0,1,2,3