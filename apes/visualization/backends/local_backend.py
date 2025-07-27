from mmengine.visualization.vis_backend import LocalVisBackend, force_init_env
from mmengine.registry import VISBACKENDS
import numpy as np
import os
import matplotlib.pyplot as plt


@VISBACKENDS.register_module()
class ModifiedLocalVisBackend(LocalVisBackend):

    """this backend won't save config file and any metric values. it is used for saving images only."""

    def add_config(self, config, **kwargs):
        pass

    def add_scalars(self, scalar_dict, step=0, file_path=None, **kwargs):
        pass

    @force_init_env
    def add_image(self, name, pcd: np.ndarray, pred: np.ndarray = None, **kwargs):
        """
        Save a side-by-side visualization of original point cloud and prediction.

        Args:
            name (str): Image name (without .png extension).
            pcd (np.ndarray): Shape (N, 6) with columns [x, y, z, r, g, b].
            pred (np.ndarray): Shape (N,) or (N, 3). If shape (N,), values are class labels.
        """
        os.makedirs(self._img_save_dir, exist_ok=True)
        saved_path = os.path.join(self._img_save_dir, f'{name}.png')

        fig = plt.figure(figsize=(10, 5))  # wider for side-by-side

        # --- Original PCD on the left ---
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.set_xlim3d(-0.6, 0.6)
        ax1.set_ylim3d(-0.6, 0.6)
        ax1.set_zlim3d(-0.6, 0.6)
        ax1.set_title('Original Point Cloud')

        colors = pcd[:, 3:6]
        if colors.max() > 1.0:
            colors = colors / 255.0

        ax1.scatter(pcd[:, 0], pcd[:, 2], pcd[:, 1], c=colors, s=2)
        ax1.axis('off')


        # --- Prediction PCD on the right ---
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.set_xlim3d(-0.6, 0.6)
        ax2.set_ylim3d(-0.6, 0.6)
        ax2.set_zlim3d(-0.6, 0.6)
        ax2.set_title('Model Prediction')
        if pred is not None:
            if pred.ndim == 1:  # labels
                colors = np.zeros((pred.shape[0], 3))
                colors[pred == 0] = [1, 0, 0]  # Red for class 0
                colors[pred == 1] = [0, 0, 1]  # Blue for class 1
                ax2.scatter(pcd[:, 0], pcd[:, 2], pcd[:, 1], c=colors, s=2)
            elif pred.shape[1] == 3:  # assume RGB colors
                ax2.scatter(pcd[:, 0], pcd[:, 2], pcd[:, 1], c=pred / 255.0, s=2)
        else:
            ax2.text2D(0.5, 0.5, "No prediction", transform=ax2.transAxes, ha='center')

        plt.tight_layout()
        plt.savefig(saved_path, bbox_inches='tight')
        plt.close(fig)
