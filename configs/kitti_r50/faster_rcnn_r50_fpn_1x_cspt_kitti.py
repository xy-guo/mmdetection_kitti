# COCO pt
# 98.106	92.488	82.635	79.169 (85.197)	71.46 (79.452)	64.008 (71.857)	78.619 (80.546)	55.188 (58.194)	53.121 (55.938)
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/kitti_mono.py',
    '../_base_/default_runtime.py'
]
model = dict(
    pretrained=None,
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=3,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))
# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[14])
total_epochs = 20
log_config = dict(interval=10)
# For better, more stable performance initialize from COCO
# load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth'
# load_from = './checkpoints/faster_rcnn_r50_fpn_1x_cityscapes_20200227-362cfbbf.pth'
# load_from = None
