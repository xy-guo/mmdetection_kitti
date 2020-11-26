import numpy as np

# model settings
model = dict(
    type='StereoThreeStageDebugDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    stereo_backbone=dict(
        type="PSMNet",
        in_planes=3,  # the in planes of feature extraction backbone
        with_pyramid=False,
        with_raw_features=True,
        with_compressed_features=False,
        out_planes=12) if use_stereo_backbone else None,
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=num_classes,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    bbox3d_roi_extractor=dict(
        type='StereoRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_volume_channels=256 if not use_stereo_backbone else 40,
        out_feat_channels=256 if not use_stereo_backbone else 320,
        featmap_strides=[4, 8, 16, 32] if not use_stereo_backbone else [4],
        volume_construction=None if not use_stereo_backbone else dict(
            type="gwc", num_groups=40)),
    depth_roi_extractor=dict(
        type='DepthRoIExtractor', roi_layer=dict(type='RoIDepth', out_size=7))
    if not do_refine else
    None,  # NOTE: update head depth_upsample_ratio acoordingly
    bbox3d_head=dict(
        type='ConvFCBBox3dHeadVariant1Hourglass',
        depth_upsample_ratio=1,  # NOTE: should be update with depth_roi out_size
        net_feature_extraction="avgpool",
        use_abs_center=False,  # NOTE: abs center may be more stable
        concat_img_feature=True,
        in_feat_channels=256 if not use_stereo_backbone else 320,
        in_volume_channels=256 if not use_stereo_backbone else 40,
        compress_concat_feature_channel=None
        if not use_stereo_backbone else 16,
        hidden_channels=128 if not use_stereo_backbone else 32,
        num_classes=num_classes,
        num_angle_bin=8,
        num_depth=80,
        depth_levels=np.arange(0.8, 64.8, 0.8),
        use_alpha_for_regression=True,
        with_only_valid_abs_depth=True,
        regress_classwise_by_softargmin=False,
        use_hg_3d=True,
        use_hg_1d=True,
        loss_angle_reg=dict(
            type='SoftArgmaxAngle', num_bin=8, loss_weight=1.0))
    if not do_refine else None,
    bbox3d_block_extractor=dict(
        type='StereoImageBlockRoIExtractor',
        roi_layer=dict(type='ImageBlockAlign', out_size=(28, 14,
                                                         28)),  # [l, h, w]
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]) if do_refine else None,
    bbox3d_refine_head=dict(
        num_classes=num_classes,
        # loss_cls=dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.5,  # default 0.25
            loss_weight=1.0),
        type='ConvFCBBox3dAnchorRefineHead') if do_refine else None,
    expand_3droi_as_square=False,
    use_raw_proposals_for_3d=False,
    debug_use_gt_volume=debug_use_gt_volume,
    debug_use_gt_2d_bbox=debug_use_gt_2d_bbox,
    debug_always_train_2d=debug_always_train_2d)
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=0.5),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=5,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=0.5),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=5,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False),
    rcnn3d=dict(
        assigner3d=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=0.5),
        sampler3d=dict(
            type='RandomSampler',
            num=32,
            pos_fraction=0.5,  # TODO: default 0.25
            neg_pos_ub=5,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False),
    refine3d=dict(
        assigner_prerefine=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.1,
            neg_iou_thr=0.0,
            min_pos_iou=0.1,
            ignore_iof_thr=0.5,
            overlap_3d_mode=True),
        sampler_prerefine=dict(
            type='RandomSampler',
            num=32,
            pos_fraction=0.5,
            neg_pos_ub=1,
            add_gt_as_proposals=True),
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.4,
            neg_iou_thr=0.25,
            min_pos_iou=0.25,
            ignore_iof_thr=0.5,
            overlap_3d_mode=True,
            constrain_angle_range=True),
        pos_weight=-1,
        debug=False) if do_refine else None)
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100),
    rcnn3d=dict(
        score_thr=0.1,
        nms=dict(type='nms3d', iou_thr=0.5),
        max_per_img=100,
        only_one_per_proposal=True),
    refine3d=dict(
        score_thr=0.2,
        nms=dict(type='nms3d', iou_thr=0.25),
        max_per_img=100,
        only_one_per_proposal=False) if do_refine else None
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_depth=True),
    dict(
        type='LoadProposals',
        load_2d_proposal=False,
        load_3d_proposal=do_refine),
    dict(
        type='Resize',
        img_scale=(1920, 800),
        keep_ratio=True,
        keep_original_size=keep_original_size),
    dict(type='StereoSwap', swap_ratio=0.5, clip_2d_box=True),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ConvertLidarToDepthMap'),  # place after Normalize
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'K', 'fL', 't2to3', 'velo2cam2', 'velo2cam3', 'left_img',
            'right_img', 'depth_img', 'lidar_points', 'uvd_points',
            'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_bboxes_3d',
            'gt_bboxes_3d_ignore', 'proposals3d'
        ],
        meta_keys=[
            'left_filename', 'right_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'swap', 'img_norm_cfg'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',  # multi-scale and flip
        img_scale=(1920, 800),  # only one scale
        flip=False,  # do not use flipping aug
        transforms=[
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_label=True,
                with_depth=True),
            dict(
                type='LoadProposals',
                load_2d_proposal=False,
                load_3d_proposal=do_refine),
            dict(
                type='Resize',
                keep_ratio=True,
                keep_original_size=keep_original_size),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ConvertLidarToDepthMap'),  # place after Normalize
            dict(type='Pad', size_divisor=32),
            # dict(type='ImageToTensor', keys=['left_img', 'right_img']),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'K', 'fL', 't2to3', 'velo2cam2', 'velo2cam3', 'left_img',
                    'right_img', 'depth_img', 'proposals3d', 'gt_bboxes',
                    'gt_labels'
                ],
                meta_keys=[
                    'left_filename', 'right_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip', 'swap',
                    'img_norm_cfg'
                ]),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'kitti_infos_train.pkl',
        img_prefix=data_root,
        pipeline=train_pipeline,
        used_classes=('Car', ) if num_classes == 2 else None,
        proposal3d_file=
        "proposals/var1hg.with_absdepth.7210f1/train_proposals_3d.pkl"
        if do_refine else None),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'kitti_infos_val.pkl',
        img_prefix=data_root,
        pipeline=test_pipeline,
        filter_empty_gt=False,
        proposal3d_file=
        "proposals/var1hg.with_absdepth.7210f1/val_proposals_3d.pkl"
        if do_refine else None),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'kitti_infos_test.pkl',
        img_prefix=data_root,
        pipeline=test_pipeline,
        filter_empty_gt=False))
evaluation = dict(interval=1, metric=['bbox_2d', 'bbox_3d', 'bev', 'aos'])
# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8 * 6, 11 * 6])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='DebugLoggerHook', log_dir='./work_dir/debug_images', enable=False),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12 * 6
dist_params = dict(backend='nccl')
log_level = 'INFO'
# v2: add swap & flip augmentation
# v3: add ignore_iof_thr threshold, and neg_pos_ub to give better supervisions
if not do_refine:
    work_dir = './work_dirs/3d/kitti/faster_rcnn_r50_fpn_1x/var1hg.with_absdepth'
else:
    work_dir = './work_dirs/3d/kitti/faster_rcnn_r50_fpn_1x/refine.debug'
if debug_use_gt_volume:
    work_dir += '.debug-use-gt-volume'
if debug_use_gt_2d_bbox:
    work_dir += '.debug-use-gt-2d-bbox'
if debug_always_train_2d:
    work_dir += '.debug-always-train-2d'
load_from = './checkpoints/faster_rcnn_r50_fpn_1x_cityscapes_20200227-362cfbbf.pth'
resume_from = None
workflow = [('train', 1)]

# NOTE:
# some other useful tricks:
# 1. multi-scale augmentation: img_scale=[(1280, 800), (2560, 800)]
# 2. no-ignore for van/ped (useful for cyclist? 2AP)
# 3. num of epochs?
