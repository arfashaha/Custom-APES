_base_ = [
    '../_base_/models/apes_seg_local.py',
    '../_base_/datasets/custom_seg.py', # Custom dataset configuration
    '../_base_/schedules/schedule_50epochs.py',
    '../_base_/default_runtime.py'
]

experiment_name = '{{fileBasenameNoExtension}}'
work_dir = f'./work_dirs/colored_{experiment_name}'

visualizer = dict(vis_backends=[dict(type='ModifiedLocalVisBackend')])

default_hooks = dict(
    checkpoint=dict(save_best=['val_instance_mIoU', 'val_dice_score', 'val_mcc', 'val_precision', 'val_recall'])
)

log_processor = dict(
    custom_cfg=[dict(data_src='loss', log_name='loss', method_name='mean', window_size='epoch')]
)
