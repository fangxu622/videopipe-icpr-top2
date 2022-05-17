
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        patch_size=(2, 4, 4),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=(8, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True),
    cls_head=dict(
        type='I3DHead',
        in_channels=1024,
        num_classes=17,
        loss_cls=dict(
            type='MultiCBFocalLoss',
            samples_per_cls=[1690, 1502, 1128, 770, 646, 270, 492, 
            521, 374, 243, 340, 223, 164, 100, 98, 228, 133],
            loss_weight=1.0, beta=0.999 ,  gamma=0.5 , 
            pos_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0]),
        spatial_type='avg',
        dropout_ratio=0.5),
    test_cfg=dict(average_clips='prob', max_testing_views=4))
dataset_type = 'QvPipeDataset'
data_root = '/home/lizhen/data/QvPipeData'
data_root_val = '/home/lizhen/data/QvPipeData'
ann_file_train = '/home/lizhen/det-prj/qvpipe_train.json'
ann_file_val = '/home/lizhen/det-prj/qvpipe_val.json'
ann_file_test = '/home/lizhen/det-prj/qvpipe_test.json'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=12, frame_interval=8, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=12,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=12,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=4 ,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    train=dict(
        type='QvPipeDataset',
        ann_file='/home/lizhen/det-prj/qvpipe_train.json',
        data_prefix='/home/lizhen/data/QvPipeData',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=12,
                frame_interval=8,
                num_clips=1),
            dict(type='RawFrameDecode'),
            dict(type='RandomRescale', scale_range=(256, 320)),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    val=dict(
        type='QvPipeDataset',
        ann_file='/home/lizhen/det-prj/qvpipe_val.json',
        data_prefix='/home/lizhen/data/QvPipeData',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=12,
                frame_interval=8,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Flip', flip_ratio=0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    test=dict(
        type='QvPipeDataset',
        ann_file='/home/lizhen/det-prj/qvpipe_test.json',
        data_prefix='/home/lizhen/data/QvPipeData',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=12,
                frame_interval=8,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 224)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Flip', flip_ratio=0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
evaluation = dict(interval=2, metrics=['mAP', ], save_best='mAP')
optimizer = dict(
    type='AdamW',
    lr=0.0004,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1))))
optimizer_config = dict(grad_clip=dict(max_norm=10.0, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5)
total_epochs = 100
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=5,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

work_dir = '/home/lizhen/det-prj/Res/swin_qvpipe_balanced_1'
find_unused_parameters = False
opencv_num_threads = 0

#fp16 = dict(loss_scale=512.0)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/lizhen/det-prj/Res/swin_base_patch244_window877_kinetics600_22k.pth'
resume_from = None #"/home/lizhen/det-prj/Res/swin_qvpipe_balanced_1/best_mAP_epoch_44.pth"
workflow = [('train', 1)]
gpu_ids = range(0, 3)
omnisource = False
module_hooks = []
