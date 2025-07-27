import numpy as np
from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric

@METRICS.register_module()
class DiceScoreMetric(BaseMetric):
    def __init__(self, mode='val'):
        super(DiceScoreMetric, self).__init__()
        self.mode = mode

    def process(self, inputs, data_samples: list[dict]):
        for sample in data_samples:
            pred = sample['pred_seg_label'].cpu().numpy().astype(np.uint8)
            gt = sample['gt_seg_label'].cpu().numpy().astype(np.uint8)

            tp = np.sum((pred == 1) & (gt == 1))
            fp = np.sum((pred == 1) & (gt == 0))
            fn = np.sum((pred == 0) & (gt == 1))

            dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
            self.results.append(dice)

    def compute_metrics(self, results) -> dict:
        if self.mode == 'val':
            return dict(val_dice_score=np.mean(results))
        elif self.mode == 'test':
            return dict(test_dice_score=np.mean(results))
        else:
            raise RuntimeError(f'Invalid mode "{self.mode}". Only supports val and test mode')
