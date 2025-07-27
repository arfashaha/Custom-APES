import torch
import numpy as np
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from einops import repeat, pack, rearrange
from typing import List, Dict
from ...structures.seg_data_sample import SegDataSample
from ...structures.cls_data_sample import ClsDataSample


@HOOKS.register_module()
class CLSVisualizationHook(Hook):
    def after_test_iter(self, runner, batch_idx: int, data_batch: Dict=None, outputs: List[ClsDataSample]=None):
        if runner.world_size > 1:
            ds1_idx = runner.model.module.backbone.ds1.idx.cpu()
            ds2_idx = torch.gather(ds1_idx, dim=1, index=runner.model.module.backbone.ds2.idx.cpu())
        else:
            ds1_idx = runner.model.backbone.ds1.idx.cpu()
            ds2_idx = torch.gather(ds1_idx, dim=1, index=runner.model.backbone.ds2.idx.cpu())
        inputs, data_samples = data_batch['inputs'], data_batch['data_samples']
        inputs = rearrange(inputs, 'B C N -> B N C')
        bg_color = repeat(torch.tensor([192., 192., 192.]), 'C -> B N C', B=inputs.shape[0], N=inputs.shape[1])
        red = torch.tensor([255., 0., 0.])
        ds1_rgb = torch.scatter(bg_color, dim=1, index=repeat(ds1_idx, 'B N -> B N C', C=3), src=repeat(red, 'C -> B N C', B=ds1_idx.shape[0], N=ds1_idx.shape[1]))
        ds2_rgb = torch.scatter(bg_color, dim=1, index=repeat(ds2_idx, 'B N -> B N C', C=3), src=repeat(red, 'C -> B N C', B=ds2_idx.shape[0], N=ds2_idx.shape[1]))
        ds1_xyz_rgb, _ = pack([inputs, ds1_rgb], 'B N *')
        ds2_xyz_rgb, _ = pack([inputs, ds2_rgb], 'B N *')
        for i, (ds1, ds2) in enumerate(zip(ds1_xyz_rgb, ds2_xyz_rgb)):
            runner.visualizer.add_image(f'cls_pcd{i+runner.test_dataloader.batch_size*(batch_idx*runner.world_size+runner.rank)}_ds1', ds1.numpy())
            runner.visualizer.add_image(f'cls_pcd{i+runner.test_dataloader.batch_size*(batch_idx*runner.world_size+runner.rank)}_ds2', ds2.numpy())


@HOOKS.register_module()
class SEGVisualizationHook(Hook):
    def after_test_iter(self, runner, batch_idx: int, data_batch: Dict = None, outputs: List[SegDataSample] = None):
        inputs = rearrange(data_batch['inputs'], 'B C N -> B N C').cpu().numpy()  # (B, N, 3)
        # results['ori_colors'] = torch.tensor(colors).permute(1, 0)  # (3, N)
        if 'ori_colors' in data_batch:
            colors = rearrange(data_batch['ori_colors'], 'B C N -> B N C').cpu().numpy()  # (B, N, 3), optional
        else:
            colors = None

        for i, (xyz, output) in enumerate(zip(inputs, outputs)):
            idx = i + runner.test_dataloader.batch_size * (batch_idx * runner.world_size + runner.rank)

            # Get predicted label map and convert to RGB via palette
            palette = np.array(output.metainfo['palette'])  # e.g., shape (2, 3)
            pred_labels = output.pred_seg_label.long().cpu().numpy()  # shape (N,)
            pred_rgb = palette[pred_labels]  # shape (N, 3)

            # Combine xyz and original color (if available)
            if colors is not None:
                orig_rgb = colors[i]
                pcd_rgb = np.concatenate([xyz, orig_rgb], axis=-1)  # shape (N, 6)
            else:
                pcd_rgb = np.concatenate([xyz, np.ones_like(xyz) * 127], axis=-1)  # fallback to gray

            runner.visualizer.add_image(f'seg_pcd{idx}', pcd=pcd_rgb, pred=pred_rgb)
