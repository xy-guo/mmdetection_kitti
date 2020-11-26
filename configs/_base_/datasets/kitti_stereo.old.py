dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1920, 800), keep_ratio=True),
    # dict(type='StereoSwap', swap_ratio=0.5),
    # dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(1920, 800), keep_ratio=True),
    dict(type='StereoSwap', swap_ratio=0.5),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='Collect',
         keys=['K', 'fL', 't2to3', 'velo2cam2', 'velo2cam3',
               'left_img', 'right_img', 'gt_bboxes',
               'gt_bboxes_ignore', 'gt_labels', 'gt_bboxes_3d', 'gt_bboxes_3d_ignore'],
         meta_keys=['left_filename', 'right_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'swap', 'img_norm_cfg']
         ),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',  # multi-scale and flip
        img_scale=(1920, 800),  # only one scale
        flip=False,  # do not use flipping aug
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # dict(type='ImageToTensor', keys=['img']),
            # dict(type='Collect', keys=['img']),
            dict(type='Collect',
                 keys=['K', 'fL', 't2to3', 'velo2cam2',
                       'velo2cam3', 'left_img', 'right_img'],
                 meta_keys=['left_filename', 'right_filename', 'ori_shape', 'img_shape',
                            'pad_shape', 'scale_factor', 'flip', 'swap', 'img_norm_cfg']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'kitti_infos_train.pkl',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'kitti_infos_val.pkl',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'kitti_infos_test.pkl',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox_2d')
