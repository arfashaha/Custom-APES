from mmengine.registry import TRANSFORMS
from .basetransform import BaseTransform
from typing import Dict
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn.functional as F


@TRANSFORMS.register_module()
class ToCLSTensor(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        results['pcd'] = rearrange(torch.tensor(results['pcd']).to(torch.float32), 'N C -> C N')  # PyTorch requires (C, N) format
        results['cls_label'] = torch.tensor(results['cls_label']).to(torch.float32)   # array to tensor
        results['cls_label_onehot'] = F.one_hot(results['cls_label'].long(), 40).to(torch.float32)  # shape == (40,)
        return results


@TRANSFORMS.register_module()
class ToSEGTensor(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        points = results['pcd']
        points = torch.tensor(points, dtype=torch.float32)

        # Split into xyz and color (assumes (N, 6) if color present)
        if points.shape[1] >= 6:
            xyz = points[:, :3]
            colors = points[:, 3:6]
        else:
            xyz = points
            colors = torch.ones_like(xyz) * 127  # fallback gray

        # Convert to (C, N)
        if points.shape[1] >= 6:
            results['pcd'] = rearrange(points[:, :6].clone().detach().to(torch.float32), 'N C -> C N')
        else:
            raise ValueError(f"Expected at least 6 dims (XYZRGB), got {points.shape}")
        results['ori_colors'] = rearrange(colors, 'N C -> C N')

        # Dummy or real class label
        cls_label = results.get('cls_label', 0)
        results['cls_label'] = torch.tensor(cls_label, dtype=torch.float32)
        onehot = F.one_hot(results['cls_label'].long(), num_classes=2).to(torch.float32)
        padded = torch.zeros(16, dtype=torch.float32)
        padded[:2] = onehot
        results['cls_label_onehot'] = padded.unsqueeze(1)

        # Segmentation label
        seg = results.get('seg_label', None)
        if seg is not None:
            if isinstance(seg, np.ndarray):
                seg = torch.tensor(seg, dtype=torch.long)
            elif not torch.is_tensor(seg):
                raise TypeError("seg_label must be a numpy array or tensor")

            if seg.ndim == 2 and seg.shape[1] == 2:
                seg = seg.argmax(dim=1)
            elif seg.ndim != 1:
                raise ValueError(f"Unexpected seg_label shape: {seg.shape}")

            seg[seg < 0] = 1  # Mark unknowns as 'alien'
            results['seg_label'] = seg
            results['seg_label_onehot'] = rearrange(
                F.one_hot(seg, num_classes=2).float(), 'N C -> C N'
            )

        return results





@TRANSFORMS.register_module()
class ShufflePointsOrder(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        idx = np.random.choice(results['pcd'].shape[0], results['pcd'].shape[0], replace=False)
        results['pcd'] = results['pcd'][idx]
        if 'seg_label' in results:
            results['seg_label'] = results['seg_label'][idx]
        return results


@TRANSFORMS.register_module()
class DataAugmentation(BaseTransform):
    def __init__(self, axis='y', angle=15, shift=0.2, min_scale=0.66, max_scale=1.5, sigma=0.01, clip=0.05):
        super().__init__()
        jitter = Jitter(sigma, clip)
        rotation = Rotation(axis, angle)
        translation = Translation(shift)
        anisotropic_scaling = AnisotropicScaling(min_scale, max_scale)
        self.aug_list = [jitter, rotation, translation, anisotropic_scaling]

    def transform(self, results: Dict) -> Dict:
        results = np.random.choice(self.aug_list)(results)
        return results


class Jitter(BaseTransform):
    def __init__(self, sigma=0.01, clip=0.05):
        super().__init__()
        self.sigma = sigma
        self.clip = clip

    def transform(self, results: Dict) -> Dict:
        pcd = results['pcd']
        npts, nfeats = pcd.shape
        jit_pts = np.clip(self.sigma * np.random.randn(npts, nfeats), -self.clip, self.clip)
        jit_pts += pcd
        results['pcd'] = jit_pts
        return results


class Rotation(BaseTransform):
    def __init__(self, axis='y', angle=15):
        super().__init__()
        self.axis = axis
        self.angle = angle

    def transform(self, results: Dict) -> Dict:
        pcd = results['pcd']
        angle = np.random.uniform(-self.angle, self.angle)
        angle = np.pi * angle / 180
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        if self.axis == 'x':
            rotation_matrix = np.array([[1, 0, 0], [0, cos_theta, sin_theta], [0, -sin_theta, cos_theta]])
        elif self.axis == 'y':
            rotation_matrix = np.array([[cos_theta, 0, -sin_theta], [0, 1, 0], [sin_theta, 0, cos_theta]])
        elif self.axis == 'z':
            rotation_matrix = np.array([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]])
        else:
            raise ValueError(f'axis should be one of x, y and z, but got {self.axis}!')
        rotated_pts = pcd @ rotation_matrix
        results['pcd'] = rotated_pts
        return results


class Translation(BaseTransform):
    def __init__(self, shift=0.2):
        super().__init__()
        self.shift = shift

    def transform(self, results: Dict) -> Dict:
        pcd = results['pcd']
        npts = pcd.shape[0]
        x_translation = np.random.uniform(-self.shift, self.shift)
        y_translation = np.random.uniform(-self.shift, self.shift)
        z_translation = np.random.uniform(-self.shift, self.shift)
        x = np.full(npts, x_translation)
        y = np.full(npts, y_translation)
        z = np.full(npts, z_translation)
        translation = np.stack([x, y, z], axis=-1)
        translation_pts = pcd + translation
        results['pcd'] = translation_pts
        return results


class AnisotropicScaling(BaseTransform):
    def __init__(self, min_scale=0.66, max_scale=1.5):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def transform(self, results: Dict) -> Dict:
        pcd = results['pcd']
        x_factor = np.random.uniform(self.min_scale, self.max_scale)
        y_factor = np.random.uniform(self.min_scale, self.max_scale)
        z_factor = np.random.uniform(self.min_scale, self.max_scale)
        scale_matrix = np.array([[x_factor, 0, 0], [0, y_factor, 0], [0, 0, z_factor]])
        scaled_pts = pcd @ scale_matrix
        results['pcd'] = scaled_pts
        return results
