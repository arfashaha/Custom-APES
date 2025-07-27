import numpy as np
from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric

@METRICS.register_module()
class PrecisionRecallMetric(BaseMetric):
    def __init__(self, mode='val'):
        super().__init__()
        self.mode = mode

    def process(self, inputs, data_samples: list[dict]):
        for sample in data_samples:
            pred = sample['pred_seg_label'].cpu().numpy().astype(np.uint8)
            gt = sample['gt_seg_label'].cpu().numpy().astype(np.uint8)
            tp = np.sum((pred == 1) & (gt == 1))
            fp = np.sum((pred == 1) & (gt == 0))
            fn = np.sum((pred == 0) & (gt == 1))

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            self.results.append((precision, recall))

    def compute_metrics(self, results) -> dict:
        precisions, recalls = zip(*results)
        if self.mode == 'val':
            return {
                "val_precision": np.mean(precisions),
                "val_recall": np.mean(recalls)
            }
        elif self.mode == 'test':
            return {
                "test_precision": np.mean(precisions),
                "test_recall": np.mean(recalls)
            }
        else:
            raise RuntimeError(f'Invalid mode "{self.mode}". Only supports val and test mode')
