_base_ = [
    './base/dataset/coco.py',
    './base/model/solov2.py',
    './base/schedule/sgd_cosine.py',
]

dataloader = dict(
    num_workers=1,
)

lr = 0.01

max_iter = 220000

steps = [50000, 150000, 200000]

solver = dict(
    train_per_batch=16,
    test_per_batch=2,
    max_iter=max_iter,
    max_keep=20,
    checkpoint_period=10000,
    generator=dict(
        lr_scheduler=dict(
            enabled=True,
            type='LRMultiplierScheduler',
            params=dict(
                lr_scheduler_param=dict(
                    name='WarmupCosineLR',
                    gamma=0.1,
                    steps=steps,
                ),
                warmup_factor=0.01,
                warmup_iter=500,
                max_iter=max_iter,
            )
        )
    )
)

output_dir = '/mnt/sda1/train.output/human.seg/test'
output_log_name = 'humanseg'
