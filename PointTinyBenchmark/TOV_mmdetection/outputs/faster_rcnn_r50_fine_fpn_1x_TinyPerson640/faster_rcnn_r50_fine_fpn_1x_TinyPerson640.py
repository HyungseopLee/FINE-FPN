check = dict(stop_while_nan=True)
checkpoint_config = dict(interval=1)
custom_hooks = [
    dict(type='NumClassCheckHook'),
]
data = dict(
    samples_per_gpu=1,
    test=dict(
        ann_file='/workspace/tiny_set/mini_annotations/tiny_set_test_all.json',
        img_prefix='/workspace/tiny_set/test/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                scale_factor=[
                    1.0,
                ],
                tile_overlap=(
                    100,
                    100,
                ),
                tile_shape=(
                    640,
                    512,
                ),
                transforms=[
                    dict(keep_ratio=True, type='Resize'),
                    dict(type='RandomFlip'),
                    dict(
                        mean=[
                            123.675,
                            116.28,
                            103.53,
                        ],
                        std=[
                            58.395,
                            57.12,
                            57.375,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(keys=[
                        'img',
                    ], type='ImageToTensor'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='CroppedTilesFlipAug'),
        ],
        type='CocoFmtDataset'),
    train=dict(
        ann_file=
        '/workspace/tiny_set/mini_annotations/tiny_set_train_sw640_sh512_all_erase.json',
        img_prefix='/workspace/tiny_set/erase_with_uncertain_dataset/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale_factor=[
                1.0,
            ], type='Resize'),
            dict(flip_ratio=0.5, type='RandomFlip'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(size_divisor=32, type='Pad'),
            dict(type='DefaultFormatBundle'),
            dict(
                keys=[
                    'img',
                    'gt_bboxes',
                    'gt_labels',
                    'gt_bboxes_ignore',
                ],
                type='Collect'),
        ],
        type='CocoFmtDataset'),
    val=dict(
        ann_file='/workspace/tiny_set/mini_annotations/tiny_set_test_all.json',
        img_prefix='/workspace/tiny_set/test/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                scale_factor=[
                    1.0,
                ],
                tile_overlap=(
                    100,
                    100,
                ),
                tile_shape=(
                    640,
                    512,
                ),
                transforms=[
                    dict(keep_ratio=True, type='Resize'),
                    dict(type='RandomFlip'),
                    dict(
                        mean=[
                            123.675,
                            116.28,
                            103.53,
                        ],
                        std=[
                            58.395,
                            57.12,
                            57.375,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(keys=[
                        'img',
                    ], type='ImageToTensor'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='CroppedTilesFlipAug'),
        ],
        type='CocoFmtDataset'),
    workers_per_gpu=1)
data_root = '/workspace/tiny_set/'
dataset_type = 'CocoFmtDataset'
device = 'cuda:0'
dist_params = dict(backend='nccl')
evaluation = dict(
    cocofmt_kwargs=dict(
        cocofmt_param=dict(evaluate_standard='tiny'),
        ignore_uncertain=True,
        iod_th_of_iou_f='lambda iou: iou',
        use_ignore_attr=True,
        use_iod_for_ignore=True),
    interval=3,
    iou_thrs=[
        0.25,
        0.5,
        0.75,
    ],
    metric='bbox',
    proposal_nums=[
        1000,
    ])
gpu_ids = range(0, 1)
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
load_from = None
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
    ], interval=50)
log_level = 'INFO'
lr_config = dict(
    policy='step',
    step=[
        8,
        11,
    ],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FINEFPNV2'),
    pretrained='torchvision://resnet50',
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=1,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                2,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            do_tile_as_aug=False,
            max_per_img=-1,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_num=1000,
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_across_levels=False,
            nms_post=1000,
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_num=1000,
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_across_levels=False,
            nms_post=1000,
            nms_pre=2000)),
    type='FasterRCNN')
optimizer = dict(lr=0.005, momentum=0.9, type='SGD', weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
resume_from = None
runner = dict(max_epochs=12, type='EpochBasedRunner')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        flip=False,
        scale_factor=[
            1.0,
        ],
        tile_overlap=(
            100,
            100,
        ),
        tile_shape=(
            640,
            512,
        ),
        transforms=[
            dict(keep_ratio=True, type='Resize'),
            dict(type='RandomFlip'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(size_divisor=32, type='Pad'),
            dict(keys=[
                'img',
            ], type='ImageToTensor'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='CroppedTilesFlipAug'),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale_factor=[
        1.0,
    ], type='Resize'),
    dict(flip_ratio=0.5, type='RandomFlip'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(size_divisor=32, type='Pad'),
    dict(type='DefaultFormatBundle'),
    dict(
        keys=[
            'img',
            'gt_bboxes',
            'gt_labels',
            'gt_bboxes_ignore',
        ],
        type='Collect'),
]
work_dir = './outputs/faster_rcnn_r50_fine_fpn_1x_TinyPerson640/'
workflow = [
    (
        'train',
        1,
    ),
]
