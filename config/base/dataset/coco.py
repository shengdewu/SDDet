data_root = '/mnt/sdb/coco'
num_workers = 1
img_size = 670


t_transformer = [
    dict(
        name='RandomFlip',
        direction=['horizontal', 'vertical'],
        p=0.3,
    ),
    dict(
        name='RandomAffine',
        rotate_degree_range=[degree for degree in range(-90, 90, 10)],
        rotate_range=False,
        border_val=0,
        p=0.1,
    ),
    dict(name='RandomCrop',
         min_crop_ratio=0.05,
         max_crop_ratio=0.25,
         crop_step=0.05,
         p=0.8),
    dict(name='Resize',
         interpolation='INTER_LINEAR',
         target_size=img_size,
         keep_ratio=True,
         is_padding=False),
    dict(
        name='RandomCompress',
        quality_lower=75,
        quality_upper=85,
        quality_step=5,
        p=0.1),
    dict(name='RandomColorJitter',
         brightness_limit=[0.5, 1.2],
         brightness_p=0.6,
         contrast_limit=[0.7, 1.3],
         contrast_p=0.6,
         saturation_limit=[0.7, 1.4],
         saturation_p=0.6,
         hue_limit=0.1,
         hue_p=0.3,
         blur_limit=[3, 5],
         sigma_limit=0,
         blur_p=0.2,
         gamma_limit=[0.3, 2.0],
         gamma_p=0.2,
         clahe_limit=4,
         clahe_p=0.2),
    dict(name='Normalize',
         mean=(123.675, 116.28, 103.53),
         std=(58.395, 57.12, 57.375),
         ),
    dict(name='Pad32')
]

v_transformer = [
    dict(name='Resize',
         interpolation='INTER_LINEAR',
         target_size=img_size,
         keep_ratio=True,
         is_padding=False),
    dict(
        name='RandomCompress',
        quality_lower=75,
        quality_upper=99,
        quality_step=5,
        p=0.5),
    dict(name='Normalize',
         mean=(123.675, 116.28, 103.53),
         std=(58.395, 57.12, 57.375),
         ),
    dict(name='Pad32')
]

dataloader = dict(
    num_workers=num_workers,
    train_data_set=dict(
        name='ConcatDataset',
        ann_info=[(f'{data_root}/train2014', f'{data_root}/annotations/instances_train2014.json')],
        use_cate=None,
        transformer=t_transformer,
    ),
    val_data_set=dict(
        name='ConcatDataset',
        select_nums=800,
        ann_info=[(f'{data_root}/val2014', f'{data_root}/annotations/instances_val2014.json')],
        use_cate=None,
        transformer=v_transformer,
    )
)
