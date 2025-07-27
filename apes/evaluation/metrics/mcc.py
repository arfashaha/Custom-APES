import numpy as np
from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric

@METRICS.register_module()
class MCCMetric(BaseMetric):
    def __init__(self, mode='val'):
        super().__init__()
        self.mode = mode

    def process(self, inputs, data_samples: list[dict]):
        for sample in data_samples:
            pred = sample['pred_seg_label'].cpu().numpy().astype(np.uint8)
            gt = sample['gt_seg_label'].cpu().numpy().astype(np.uint8)

            tp = np.sum((pred == 1) & (gt == 1))
            tn = np.sum((pred == 0) & (gt == 0))
            fp = np.sum((pred == 1) & (gt == 0))
            fn = np.sum((pred == 0) & (gt == 1))

            numerator = (tp * tn) - (fp * fn)
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-6
            mcc = numerator / denominator
            self.results.append(mcc)

    def compute_metrics(self, results) -> dict:
        if self.mode == 'val':
            return dict(val_mcc=np.mean(results))
        elif self.mode == 'test':
            return dict(test_mcc=np.mean(results))
        else:
            raise RuntimeError(f'Invalid mode "{self.mode}". Only supports val and test mode')
