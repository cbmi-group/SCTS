dataset_type = 'CocoDataset'
data_root = '/workspace/Swin-Transformer-Object-Detection/livecell/'
img_norm_cfg = dict(
    mean=[128, 128, 128], std=[11.578, 11.578, 11.578], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/LIVECell_proceed/livecell_coco_train.json',
        img_prefix=data_root + 'images/livecell_train_val_images',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/LIVECell_proceed/livecell_coco_val.json',
        img_prefix=data_root + 'images/livecell_train_val_images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/LIVECell_proceed/livecell_coco_test.json',
        img_prefix=data_root + 'images/livecell_test_images',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
