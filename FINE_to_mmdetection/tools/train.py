# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    print("Current data_root:", cfg.data_root)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()



'''

# Faster RCNN ours
mkdir -p outputs/faster-rcnn_r50_fine_fpn_v2_1x_coco/
bash ./tools/dist_train.sh \
    configs/faster_rcnn/faster-rcnn_r50_fine_fpn_v2_1x_coco.py \
    2 \
    --cfg-options \
        train_dataloader.dataset.data_root=/media/data/coco/ \
        train_dataloader.dataset.data_prefix.img=images/train2017/ \
        val_dataloader.dataset.data_root=/media/data/coco/ \
        val_dataloader.dataset.data_prefix.img=images/val2017/ \
        test_dataloader.dataset.data_root=/media/data/coco/ \
        test_dataloader.dataset.data_prefix.img=images/val2017/ \
        val_evaluator.ann_file=/media/data/coco/annotations/instances_val2017.json \
        test_evaluator.ann_file=/media/data/coco/annotations/instances_val2017.json \
        optim_wrapper.accumulative_counts=4 \
    --work-dir ./outputs/faster-rcnn_r50_fine_fpn_v2_1x_coco/ \
    2>&1 | tee ./outputs/faster-rcnn_r50_fine_fpn_v2_1x_coco/train_chn-wise.log


python tools/analysis_tools/get_flops.py \
    configs/faster_rcnn/faster-rcnn_r50_fine_fpn_1x_coco.py \
    --shape 640 640 \
    --cfg-options \
        train_dataloader.dataset.data_root=/media/data/coco/ \
        train_dataloader.dataset.data_prefix.img=images/train2017/ \
        val_dataloader.dataset.data_root=/media/data/coco/ \
        val_dataloader.dataset.data_prefix.img=images/val2017/ \
        test_dataloader.dataset.data_root=/media/data/coco/ \
        test_dataloader.dataset.data_prefix.img=images/val2017/ \
        val_evaluator.ann_file=/media/data/coco/annotations/instances_val2017.json \
        test_evaluator.ann_file=/media/data/coco/annotations/instances_val2017.json \
    2>&1 | tee ./outputs/faster-rcnn_r50_fine_fpn_1x_coco/flops.log




## Panoptic Segmentation
bash ./tools/dist_train.sh \
    configs/panoptic_fine_fpn/panoptic-fpn_r50_fine_fpn_1x_coco.py \
    2 \
    --cfg-options \
        optim_wrapper.accumulative_counts=4 \
    --work-dir ./outputs/panoptic-fpn_r50_fine_fpn_1x_coco/ \
    2>&1 | tee ./outputs/panoptic-fpn_r50_fine_fpn_1x_coco/train.log


bash ./tools/dist_train.sh \
    configs/faster_rcnn/faster-rcnn_r50_fine_fpn_v2_1x_coco.py \
    1 \
    --cfg-options \
        train_dataloader.dataset.data_root=/media/data/coco/ \
        train_dataloader.dataset.data_prefix.img=images/train2017/ \
        val_dataloader.dataset.data_root=/media/data/coco/ \
        val_dataloader.dataset.data_prefix.img=images/val2017/ \
        test_dataloader.dataset.data_root=/media/data/coco/ \
        test_dataloader.dataset.data_prefix.img=images/val2017/ \
        val_evaluator.ann_file=/media/data/coco/annotations/instances_val2017.json \
        test_evaluator.ann_file=/media/data/coco/annotations/instances_val2017.json \
        optim_wrapper.accumulative_counts=8 \
    --work-dir ./outputs/faster-rcnn_r50_fine_fpn_v2_1x_coco/ \
    2>&1 | tee ./outputs/faster-rcnn_r50_fine_fpn_v2_1x_coco/train.log

'''