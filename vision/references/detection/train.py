"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time

import presets
import torch
import torch.utils.data

import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
print(f"print(torchvision.__file__): {print(torchvision.__file__)}") # /home/hslee/anaconda3/lib/python3.11/site-packages/torchvision/__init__.py
# FPN: /home/hslee/anaconda3/lib/python3.11/site-packages/torchvision/ops/feature_pyramid_network.py

import utils
print(f"print(utils.__file__): {print(utils.__file__)}") # /home/hslee/SONeck/vision/references/detection/utils.py
from coco_utils import get_coco
from engine import evaluate, train_one_epoch, visualize_coco_samples, \
    visualize_cosine_similarity_of_hierarchical_features, \
    get_multi_scale_logits, \
    get_similarity_with_multi_scale_logits, \
    coco_val_fp_fn
        
from group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from torchvision.transforms import InterpolationMode
from transforms import SimpleCopyPaste


import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import seaborn as sns
import random
from torchvision.datasets import CocoDetection


def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*utils.collate_fn(batch))


def get_dataset(is_train, args):
    image_set = "train" if is_train else "val"
    num_classes, mode = {"coco": (91, "instances"), "coco_kp": (2, "person_keypoints")}[args.dataset]
    with_masks = "mask" in args.model
    ds = get_coco(
        root=args.data_path,
        image_set=image_set,
        transforms=get_transform(is_train, args),
        mode=mode,
        use_v2=args.use_v2,
        with_masks=with_masks,
    )
    return ds, num_classes


def get_transform(is_train, args):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=args.data_augmentation, backend=args.backend, use_v2=args.use_v2
        )
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval(backend=args.backend, use_v2=args.use_v2)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument(
        "--dataset",
        default="coco",
        type=str,
        help="dataset name. Use coco for object detection and instance segmentation and coco_kp for Keypoint detection",
    )
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 8)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument(
        "--use-copypaste",
        action="store_true",
        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
    )

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    
    # 2025.05.14 @HyungseopLee
    parser.add_argument(
        "--visualize",
        default=False,
        type=str,
        help="Path to model weights for visualization. If set, --image is required.",
    )
    # 2025.06.27 @HyungseopLee
    parser.add_argument(
        "--semantic-gap",
        default=False,
        type=str,
        help="to check semantic gap, set path to model weights.",
    )
    parser.add_argument("--image", default=False, type=str, help="Path to image for visualization")

    return parser


import torch
def gram_linear(x):
    return x @ x.T

def linear_cka(x, y):
    # 입력: (N, C)
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    
    gram_x = gram_linear(x)
    gram_y = gram_linear(y)

    numerator = (gram_x * gram_y).sum()
    denominator = torch.norm(gram_x) * torch.norm(gram_y)
    return numerator / denominator
    
def main(args):
    if args.backend.lower() == "tv_tensor" and not args.use_v2:
        raise ValueError("Use --use-v2 if you want to use the tv_tensor backend.")
    if args.dataset not in ("coco", "coco_kp"):
        raise ValueError(f"Dataset should be coco or coco_kp, got {args.dataset}")
    if "keypoint" in args.model and args.dataset != "coco_kp":
        raise ValueError("Oops, if you want Keypoint detection, set --dataset coco_kp")
    if args.dataset == "coco_kp" and args.use_v2:
        raise ValueError("KeyPoint detection doesn't support V2 transforms yet")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
  
    if args.test_only and args.visualize:
        torch.backends.cudnn.deterministic = True
        is_baseline = args.visualize == "baseline"

        if is_baseline:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        else:
            # 
            model = torchvision.models.get_model(
                args.model,
                weights=None,
                weights_backbone=None,
            )
            print(f"model: {model}")
            checkpoint = torch.load(args.visualize, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"])

        model.to(device)
        model.eval()

        # COCO 이미지 루트 및 annotation 파일 경로
        coco_ann_file = "/media/data/coco/annotations/instances_val2017.json"
        image_root = "/media/data/coco/images/val2017"

        visualize_coco_samples(model, coco_ann_file, image_root, device, is_baseline=is_baseline)
        return
    
    
    
    if args.test_only and args.semantic_gap:
        torch.backends.cudnn.deterministic = True
        is_baseline = args.semantic_gap == "baseline"
        
        if is_baseline:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        else:
            model = torchvision.models.get_model(
                args.model,
                weights=None,
                weights_backbone=None,
            )
            # print(f"model: {model}")
            checkpoint = torch.load(args.semantic_gap, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"])

        model.eval()
        device = torch.device(args.device)
        model.to(device)

        # # false positive, false negative
        # coco_val_fp_fn(model, device)
        # return

        coco_ann_file = "/media/data/coco/annotations/instances_val2017.json"
        coco_img_dir = "/media/data/coco/images/val2017"
        coco_dataset = CocoDetection(coco_img_dir, coco_ann_file)
        
        sample_id = "000000466835"
        sample_id = "000000099024"
        # 1. 이미지 불러오기 및 전처리
        import torchvision.transforms as transforms
        image = Image.open(f"{coco_img_dir}/{sample_id}.jpg").convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, H, W)
        
        print(f"input_tensor.shape: {input_tensor.shape}")  # (1, 3, 640, 640)
        
        # inference for get multi-scale hooks
        with torch.no_grad():
            model.eval()
            outputs = model(input_tensor)
            
        # get multi-scale feature using hooks
        # FPN feature hooks
        fpn_feats = model.backbone.fpn.feature_hooks  # e.g., {'0': {...}, '1': {...}, '2': {...}}
        
        # remove '0' level if it exists
        if '0' in fpn_feats:
            del fpn_feats['0']
        
        levels = list(fpn_feats.keys())  # 예: ['0', '1', '2']

        low_feats = []
        fused_feats = []

        # '3' : high-level
        # ...
        # '0' : low-level

        for i, lvl in enumerate(levels):
            fused = fpn_feats[lvl]['fused']  # shape: (B, C, H, W)
            low_level = fpn_feats[lvl]['low']  # shape: (B, C, H, W)
            print(f"level: {lvl}, fused.shape: {fused.shape}, low_level.shape: {low_level.shape}")  # e.g., (1, 256, 80, 80)
            low_feats.append(low_level.mean(dim=[0, 2, 3]).numpy())  # (C,)
            fused_feats.append(fused.mean(dim=[0, 2, 3]).numpy())
                        
        # label 정의 (역순이면 'P5', 'P4', ...)
        level_labels = [f"P{i+2}" for i in range(len(fused_feats))]

        # Pearson correlation matrix ---------------------------------------------------------

        # visualize correlation matrix (P2, P3, P4, P5)
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        # numpy array로 변환 (N, C)
        low_features = np.stack(low_feats, axis=0)  # shape: (num_levels, C)
        features = np.stack(fused_feats, axis=0)  # shape: (num_levels, C)
        low_corr = np.corrcoef(low_features)
        fused_corr = np.corrcoef(features)

        # 시각화
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(
            fused_corr,
            annot=True,
            xticklabels=level_labels,
            yticklabels=level_labels,
            cmap="coolwarm",
            vmin=-1, vmax=1,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8}
        )

        # x축 라벨을 위로 이동
        ax.xaxis.set_ticks_position('top')        # x축 눈금을 위로
        ax.xaxis.set_label_position('top')        # x축 라벨 위치도 위로
        ax.tick_params(top=True, labeltop=True)   # 위쪽 눈금과 라벨 표시

        plt.title("Correlation Matrix of FPN Inputs (Fused Feature)", pad=20)
        plt.xlabel("")  # 아래쪽 x축 라벨 제거
        plt.ylabel("Feature Level (Row)")
        plt.tight_layout()
        plt.show()
        
        # save
        if is_baseline:
            plt.savefig(f"./features_corr_baseline_{sample_id}.png")
        
        else: 
            plt.savefig(f"./features_corr_ours_{sample_id}.png")
        
        
        
        # CKA  ---------------------------------------------------------
        
        
        return

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(is_train=True, args=args)
    dataset_test, _ = get_dataset(is_train=False, args=args)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

        train_collate_fn = copypaste_collate_fn

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=train_collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    print("Creating model")
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    
    model = torchvision.models.get_model(
        args.model, weights=args.weights, weights_backbone=args.weights_backbone, num_classes=num_classes, **kwargs
    )
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    if args.resume:
        # checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        # args.start_epoch = checkpoint["epoch"] + 1
        # if args.amp:
            # scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        
        torch.backends.cudnn.deterministic = True
        evaluate(model, data_loader_test, device=device)
        return
        
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
        from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
        from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
        
        # # faster-rcnn
        # model = fasterrcnn_resnet50_fpn(
        #     weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if args.weights is None else args.weights,
        #     weights_backbone=args.weights_backbone,
        #     num_classes=num_classes,
        # )
        
        # # retinanet
        # model = retinanet_resnet50_fpn(
        #     weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT if args.weights is None else args.weights,
        #     weights_backbone=args.weights_backbone,
        #     num_classes=num_classes,
        # )
        
        # # fcos
        # model = fcos_resnet50_fpn(
        #     weights=FCOS_ResNet50_FPN_Weights.DEFAULT if args.weights is None else args.weights,
        #     weights_backbone=args.weights_backbone,
        #     num_classes=num_classes,
        # )
        
        # mask-rcnn
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if args.weights is None else args.weights,
            weights_backbone=args.weights_backbone,
            num_classes=num_classes,
        )
        
        model.to(device)
        model.eval()
        torch.backends.cudnn.deterministic = True
        evaluate(model, data_loader_test, device=device)
        return
    
    if args.visualize:
        print(f"Visualizing model predictions on {args.image}")
        visualize(model, args.image, device=device)
        return

    from fvcore.nn import FlopCountAnalysis, flop_count_table
    
    print(f"model: {model}")
    
    # # flops
    # model.eval()
    # with torch.no_grad():
    #     input = torch.randn(1, 3, 640, 640).to(device)
    #     flops = FlopCountAnalysis(model, input)
    #     print(flop_count_table(flops))
    #     gflops = flops.total() / 1e9
    #     print(f"GFLOPS: {gflops:.2f}")
    
    
    from ptflops import get_model_complexity_info
    model.eval().to(device)
    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model,
            input_res=(3, 640, 640),
            as_strings=True,
            print_per_layer_stat=True,
            verbose=True
        )
    print(f"FLOPs: {macs}, Params: {params}")

    model.train()
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, scaler)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            
            # save checkpoint 3 epoch frequency
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)

'''



# Mask R-CNN

torchrun --nproc_per_node=2 train.py \
    --dataset coco --data-path /media/data/coco \
    --model maskrcnn_resnet50_fpn \
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 \
    --weights-backbone ResNet50_Weights.IMAGENET1K_V1 \
    --batch-size 2 --lr 0.005 \
    --output-dir ./outputs/maskrcnn-r50/*SemanticNormAttn \
    --print-freq 100 \
    2>&1 | tee ./outputs/maskrcnn-r50/*SemanticNormAttn/train_posEmbed.log

 
# Faster R-CNN

torchrun --nproc_per_node=8 train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1
 

torchrun --nproc_per_node=2 python train.py \
    --dataset coco --data-path /media/data/coco \
    --model fasterrcnn_resnet50_fpn \
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 \
    --weights-backbone ResNet50_Weights.IMAGENET1K_V1 \
    --batch-size 2 --lr 0.005 \
    --output-dir ./outputs/fasterrcnn-r50/Deconv \
    --print-freq 100 \
    2>&1 | tee ./outputs/fasterrcnn-r50/Deconv/flops.log
    
torchrun --nproc_per_node=4 train.py \
    --dataset coco --data-path /home/hslee/FINE-FPN/data/coco \
    --model fasterrcnn_resnet50_fpn \
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 \
    --weights-backbone ResNet50_Weights.IMAGENET1K_V1 \
    --batch-size 2 --lr 0.01 \
    --output-dir ./outputs/fasterrcnn-r50/Deconv \
    --print-freq 100 \
    2>&1 | tee ./outputs/fasterrcnn-r50/Deconv/train.log
    
python train.py \
    --dataset coco --data-path /home/hslee/FINE-FPN/data/coco \
    --model fasterrcnn_resnet50_fpn \
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 \
    --weights-backbone ResNet50_Weights.IMAGENET1K_V1 \
    --batch-size 2 --lr 0.01 \
    --output-dir ./outputs/fasterrcnn-r50/Deconv \
    --print-freq 100 \
    2>&1 | tee ./outputs/fasterrcnn-r50/Deconv/train.log
 
# RetinaNet

torchrun --nproc_per_node=8 train.py\
    --dataset coco --model retinanet_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --lr 0.01 --weights-backbone ResNet50_Weights.IMAGENET1K_V1

torchrun --nproc_per_node=2 train.py \
    --dataset coco --data-path /media/data/coco \
    --model retinanet_resnet50_fpn \
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 \
    --weights-backbone ResNet50_Weights.IMAGENET1K_V1 \
    --batch-size 2 --lr 0.005 \
    --output-dir ./outputs/retinanet-r50/*SemanticNormAttn \
    --print-freq 100 \
    --resume /home/hslee/SONeck/vision/references/detection/outputs/retinanet-r50/*SemanticNormAttn/train_PosEmbed_checkpoint.pth \
    2>&1 | tee -a ./outputs/retinanet-r50/*SemanticNormAttn/train_PosEmbed.log 
 
# FCOS

torchrun --nproc_per_node=8 train.py\
    --dataset coco --model fcos_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3  --lr 0.01 --amp --weights-backbone ResNet50_Weights.IMAGENET1K_V1
    
torchrun --nproc_per_node=2 train.py \
    --dataset coco --data-path /media/data/coco \
    --model fcos_resnet50_fpn \
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 \
    --amp --weights-backbone ResNet50_Weights.IMAGENET1K_V1 \
    --batch-size 2 --lr 0.005 \
    --output-dir ./outputs/fcos-r50/SemanticAlign_TransNormer \
    --print-freq 100 \
    2>&1 | tee ./outputs/fcos-r50/SemanticAlign_TransNormer/train_1e-06_noLN_noPosEmed_l1norm-HW.log
 
torchrun --nproc_per_node=4 train.py \
    --dataset coco --data-path /home/hslee/FINE-FPN/data/coco \
    --model fcos_resnet50_fpn \
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 \
    --amp --weights-backbone ResNet50_Weights.IMAGENET1K_V1 \
    --batch-size 2 --lr 0.01 --world-size 4 \
    --output-dir ./outputs/fcos-r50/SemanticAlign_TransNormer \
    --print-freq 100 \
    2>&1 | tee ./outputs/fcos-r50/SemanticAlign_TransNormer/train_posEmed_norm.log
 
# visualize

## baseline
python train.py \
    --model fasterrcnn_resnet50_fpn \
    --test-only \
    --visualize baseline \
    2>&1 | tee ./test.log
    
## Ours
python train.py \
    --model fasterrcnn_resnet50_fpn \
    --test-only \
    --visualize /home/hslee/Desktop/Embedded_AI/EXP/vision/references/detection/pretrained/fasterrcnn_best_model_23.pth \
    2>&1 | tee ./test.log
    

python train.py \
    --model maskrcnn_resnet50_fpn \
    --test-only \
    --semantic-gap baseline \
    2>&1 | tee ./baseline_sim.log
    
python train.py \
    --model fasterrcnn_resnet50_fpn \
    --test-only \
    --semantic-gap baseline \
    2>&1 | tee ./test.log
 
    --semantic-gap /home/hslee/Desktop/Embedded_AI/EXP/vision/references/detection/pretrained/fasterrcnn_best_model_23.pth \
    

# Keypoint R-CNN
torchrun --nproc_per_node=8 train.py\
    --dataset coco_kp --model keypointrcnn_resnet50_fpn --epochs 46\
    --lr-steps 36 43 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1

torchrun --nproc_per_node=4 train.py \
    --dataset coco_kp --model keypointrcnn_resnet50_fpn --epochs 46 \
    --lr-steps 36 43 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1 \
    --lr 0.01 \
    --weights-backbone ResNet50_Weights.IMAGENET1K_V1

torchrun --nproc_per_node=2 train.py \
    --dataset coco_kp --data-path /media/data/coco --model keypointrcnn_resnet50_fpn --epochs 46 \
    --lr-steps 36 43 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1 \
    --lr 0.005 \
    --print-freq 100 \
    2>&1 | tee ./outputs/keypointrcnn-r50/FINE/train2.log

    

# Eval

python train.py \
    --model fasterrcnn_resnet50_fpn \
    --dataset coco --data-path /media/data/coco \
    --test-only \
    2>&1 | tee ./evals/maskrcnn_resnet50_coco_baseline.log

'''
