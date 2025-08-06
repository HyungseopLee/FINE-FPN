"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS

import os



def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # priority 0
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)

'''

torchrun --nproc_per_node=4 tools/train.py \
    -c /home/hslee/FINE-FPN/new_rtdetr/configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
    2>&1 | tee ./test.log

export CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 \
    tools/train.py \
    -c /home/hslee/FINE-FPN/new_rtdetr/configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --output-dir=/home/hslee/FINE-FPN/new_rtdetr/output/v1_r50vd_72e_coco_TransNormer \
    --use-amp --seed=0 2>&1 | tee /home/hslee/FINE-FPN/new_rtdetr/output/v1_r50vd_72e_coco_TransNormer/train_posEmbed_bs32.log
    
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 \
    tools/train.py \
    -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
    --output-dir=/home/hslee/SONeck/new_rtdetr/output/v1_r18vd_12e_MGC \
    --use-amp --seed=0 2>&1 | tee ./output/v1_r18vd_12e_MGC/train.log
    
    --use-amp
        
        
    python tools/train.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --output-dir=./output/*v1_r50vd_72e_coco_ours \
    --use-amp --seed=0 2>&1 | tee ./output/*v1_r50vd_72e_coco_ours/train_noPosEmbed_nhead-16_L2H.log
    
    
python tools/train.py \
    -c /home/hslee/Desktop/Embedded_AI/CLASS-FPN/new_rtdetr/configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
    --output-dir=./output/v1_18vd_72e_coco_MGC \
    --use-amp --seed=0 2>&1 | tee ./output/v1_18vd_72e_coco_MGC/train.log

'''