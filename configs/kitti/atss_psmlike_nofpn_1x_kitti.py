_base_ = [
    '../_base_/datasets/kitti_mono.py',
    '../_base_/default_runtime.py'
]
# ATSS Model
model = dict(
    type='ATSS',
    pretrained='torchvision://resnet34',
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        with_max_pool=False,
        strides=(1, 2, 1, 1),
        dilations=(1, 1, 2, 4),
        deep_stem=False,  # TODO: no pretrained model, cannot modify
        block_with_final_relu=True,
        base_channels=64),
    neck=None,
    bbox_head=dict(
        type='ATSSHead',
        num_classes=3,
        in_channels=512,
        stacked_convs=4,
        feat_channels=512,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[16, 32, 64, 128, 256],
            strides=[4]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
# training and testing settings for ATSS
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=100)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
total_epochs = 24
log_config = dict(interval=10)
# For better, more stable performance initialize from COCO: 39.4AP
# load_from = 'http://download.openmmlab.com/mmdetection/v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth'
