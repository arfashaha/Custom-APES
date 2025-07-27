import h5py
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.registry import DATASETS


@DATASETS.register_module()
class CustomSegDataset(BaseDataset):

    METAINFO = dict(
        classes=('alien', 'target'),
        mapping={0: (0,), 1: (1,)},
        palette=((0, 0, 255), (255, 0, 0))
    )

    def __init__(self, h5_path, split='train', split_ratio=0.8, pipeline=None, test_mode=False, metainfo=METAINFO):
        self.h5_path = h5_path
        self.split = split
        self.split_ratio = split_ratio
        self.test_mode = test_mode

        # Load the entire HDF5 file
        with h5py.File(h5_path, 'r') as f:
            self.all_points = f['seg_points'][:]  # (N, P, 3)
            self.all_labels = f['seg_labels'][:]  # (N, P, 2) — one-hot
            self.all_colors = f['seg_colors'][:] 

        total = len(self.all_points)
        split_idx = int(total * self.split_ratio)

        if split == 'train' or split == 'test':
            self.points = self.all_points[:split_idx]
            self.labels = self.all_labels[:split_idx]
            self.colors = self.all_colors[:split_idx]   # ← Add this
        else:
            self.points = self.all_points[split_idx:]
            self.labels = self.all_labels[split_idx:]
            self.colors = self.all_colors[split_idx:]   # ← Add this

        super().__init__(metainfo=metainfo, pipeline=pipeline, test_mode=test_mode)

    def load_data_list(self):
        data_list = []
        for i in range(len(self.points)):
            points = self.points[i]           # shape (N, 3)
            colors = self.colors[i]           # shape (N, 3)
            seg_label = self.labels[i]

            if seg_label.ndim == 3:
                seg_label = seg_label.squeeze()

            # Convert from one-hot to index-based label (0 or 1)
            seg_label = np.argmax(seg_label, axis=-1)

            # Concatenate XYZ + RGB → (N, 6)
            pcd_with_color = np.concatenate([points, colors], axis=-1)

            data_info = dict(
                classes=self.metainfo['classes'],
                mapping=self.metainfo['mapping'],
                palette=self.metainfo['palette'],
                pcd=pcd_with_color,                     # ← full (N, 6)
                ori_colors=colors,                      # ← still include raw color (optional)
                cls_label=0,
                seg_label=seg_label,
                pcd_path=f'dummy_pcd_{i}.xyz',
                cls_label_path=f'dummy_cls_{i}.txt',
                seg_label_path=f'dummy_seg_{i}.txt'
            )
            data_list.append(data_info)
        return data_list




@DATASETS.register_module()
class CustomSegTestDataset(BaseDataset):

    METAINFO = dict(
        classes=('alien', 'target'),
        mapping={0: (0,), 1: (1,)},
        palette=((0, 0, 255), (255, 0, 0))
    )

    def __init__(self, h5_path, pipeline=None, metainfo=METAINFO):
        self.h5_path = h5_path

        # Load test HDF5 (no labels)
        with h5py.File(h5_path, 'r') as f:
            self.points = f['seg_points'][:]  # (N, P, 3)
            self.colors = f['seg_colors'][:]  # (N, P, 3)

        super().__init__(metainfo=metainfo, pipeline=pipeline, test_mode=True)

    def load_data_list(self):
        data_list = []
        for i in range(len(self.points)):
            pcd_with_color = np.concatenate([self.points[i], self.colors[i]], axis=-1)  # (N, 6)

            # Dummy segmentation label (all -1)
            seg_label = np.zeros(self.points[i].shape[0], dtype=np.int64)

            data_info = dict(
                classes=self.metainfo['classes'],
                mapping=self.metainfo['mapping'],
                palette=self.metainfo['palette'],
                pcd=pcd_with_color,
                ori_colors=self.colors[i],
                seg_label=seg_label,                # 👈 Add dummy labels
                cls_label=0,
                pcd_path=f'test_pcd_{i}.xyz',
                cls_label_path=f'dummy_cls_{i}.txt',
                seg_label_path=f'dummy_seg_{i}.txt'
            )
            data_list.append(data_info)
        return data_list
