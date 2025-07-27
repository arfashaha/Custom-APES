# dataloaders
train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
    dataset=dict(
        type='CustomSegDataset',
        h5_path='/home/s2737104/MiniMarket_dataset_processing/Final_dataset/hazelnut_cocoa_spread_nutella_350gm_1200_4096_segmentation_40960_40960_10',
        split='train',
        split_ratio=0.8,
        pipeline=[
            dict(type='ToSEGTensor'),
            dict(type='PackSEGInputs')
        ]
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate')
)

val_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
    dataset=dict(
        type='CustomSegDataset',
        h5_path='/home/s2737104/MiniMarket_dataset_processing/Final_dataset/hazelnut_cocoa_spread_nutella_350gm_1200_4096_segmentation_40960_40960_10',
        split='val',
        split_ratio=0.8,
        pipeline=[
            dict(type='ToSEGTensor'),
            dict(type='PackSEGInputs')
        ]
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate')
)

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
    dataset=dict(
        type='CustomSegDataset',
        h5_path='/home/s2737104/APES/data/all_test_scenes.h5',
        split='test',
        split_ratio=1.0,
        pipeline=[
            dict(type='ToSEGTensor'),
            dict(type='PackSEGInputs')
        ]
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate')
)

# test_dataloader = val_dataloader

val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='InstanceMeanIoU', mode='val'),
        dict(type='DiceScoreMetric', mode='val'),
        dict(type='MCCMetric', mode='val'),
        dict(type='PrecisionRecallMetric', mode='val'),
    ]
)

test_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='InstanceMeanIoU', mode='test'),
        dict(type='DiceScoreMetric', mode='test'),
        dict(type='MCCMetric', mode='test'),
        dict(type='PrecisionRecallMetric', mode='test'),
    ]
)

