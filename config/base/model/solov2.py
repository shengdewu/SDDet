# model settings
num_classes = 1
ignore_index = 255

model = dict(
    name='SingleStageInstanceSegmentor',
    backbone=dict(
        name='RegNet',
        stem_channels=32,
        stages=[
            [[48, [1], 16, 2, 4]],
            [[128, [1], 16, 2, 4], *[[128, [1], 16, 1, 4]] * 2],
            [
                [256, [1], 16, 2, 4],
                [256, [1], 16, 1, 4],
                [256, [1, 2], 16, 1, 4],
                *[[256, [1, 4], 16, 1, 4]] * 4,
                *[[256, [1, 14], 16, 1, 4]] * 6,
                [320, [1, 14], 16, 1, 4],
            ],
        ],
        out_indices=(0, 1, 2)
    ),
    necker=dict(name='FPN',
                in_channels=[48, 128, 320],
                out_channels=256,
                start_level=0,
                num_outs=4,
                ),
    header=dict(
        name='SOLOv2Head',
        num_classes=80,
        in_channels=256,
        mask_feature_head=dict(feat_channels=128,
                               start_level=0,
                               end_level=3,
                               out_channels=256,
                               mask_stride=4),
        feat_channels=256,
        stacked_convs=3,
        strides=(4, 8, 16, 32),
        scale_ranges=((48, 192), (96, 384), (192, 768), (384, 2048)),
        pos_scale=0.2,
        num_grids=(40, 36, 24, 16),
        cls_down_index=0,
        loss_mask=[dict(name='BinaryDiceLoss', param=dict(apply_sigmoid=True, lambda_weight=3.0, reduction='none'))],
        loss_cls=[dict(name='FocalLoss', param=dict(apply_sigmoid=True, gamma=2.0, alpha=0.25, reduction='sum', lambda_weight=1.0))]),
)

trainer = dict(
    name='DetTrainer',
    weights='',
    enable_epoch_method=False,
    model=dict(
        name='DetModel',
        generator=model,
    )
)
