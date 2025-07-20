# cur="/home/hslee/SONeck/vision/references/detection/"
# torchrun --nproc_per_node=2 ${cur}/train.py \
#     --dataset coco --data-path /media/data/coco \
#     --model fcos_resnet50_fpn \
#     --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 \
#     --amp --weights-backbone ResNet50_Weights.IMAGENET1K_V1 \
#     --batch-size 2 --lr 0.005 \
#     --output-dir ${cur}/outputs/fcos-r50/SemanticAlign_ReLU_CosFormer \
#     --print-freq 100 \
#     2>&1 | tee ${cur}/outputs/fcos-r50/SemanticAlign_ReLU_CosFormer/train_noPosEmbed_norm.log


# rtdetr 18 (pos embed)
cur="/home/hslee/SONeck/rtdetrv2_pytorch2"
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 ${cur}/tools/train.py \
    -c ${cur}/configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
    2>&1 | tee ${cur}/output/*v1_r18vd_72e_coco_SemanticAlign-ReLUCosFormer/train_posEmbed.log

# rtdetr 50 (no pos embed)
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 ${cur}/tools/train.py \
    -c ${cur}/configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    2>&1 | tee ${cur}/output/*v1_r50vd_72e_coco_SemanticAlign-ReLUCosFormer/train_posEmbed.log