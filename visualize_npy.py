import numpy as np
import open3d as o3d
import h5py
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# --- EDIT THESE PATHS ---
pred_path = '/home/s2737104/APES/predictions.npy'
hdf5_path = '/home/s2737104/APES/data/all_test_scenes.h5'
gt_label_path = None  # set to GT .npy or None if not available
out_vis_dir = 'prediction_pc_vis'
os.makedirs(out_vis_dir, exist_ok=True)

# --- PARAMETERS ---
color_map = np.array([
    [1, 0, 0],   # Class 0: Red
    [0, 0, 1],   # Class 1: Blue
    # Add more for extra classes
])

# --- LOAD DATA ---
preds = np.load(pred_path)  # (num_scenes, num_points)
print(f"Loaded predictions: {preds.shape}")

with h5py.File(hdf5_path, 'r') as f:
    points_set = f['seg_points'][:]
print(f"Loaded {len(points_set)} scenes from HDF5.")

if gt_label_path is not None:
    gt_labels = np.load(gt_label_path)
    assert gt_labels.shape == preds.shape, "GT and preds shape mismatch!"
else:
    gt_labels = None

# --- STATS ---
print("\n==== Per-scene stats ====")
overall_preds = []
overall_gts = []

for i, scene_pred in enumerate(preds):
    unique, counts = np.unique(scene_pred, return_counts=True)
    print(f"Scene {i+1}:")
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} points")
    overall_preds.append(scene_pred)
    if gt_labels is not None:
        overall_gts.append(gt_labels[i])
        print("  Classification report:")
        print(classification_report(gt_labels[i], scene_pred, zero_division=0))
        print("  Confusion matrix:")
        print(confusion_matrix(gt_labels[i], scene_pred))
print("")

all_preds_flat = np.concatenate(overall_preds)
unique, counts = np.unique(all_preds_flat, return_counts=True)
print("==== Overall predictions distribution ====")
for cls, cnt in zip(unique, counts):
    print(f"  Class {cls}: {cnt} points ({cnt/len(all_preds_flat)*100:.2f}%)")
if gt_labels is not None:
    all_gts_flat = np.concatenate(overall_gts)
    print("\n==== Overall GT distribution ====")
    unique, counts = np.unique(all_gts_flat, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} points ({cnt/len(all_gts_flat)*100:.2f}%)")
    print("\n==== Overall classification report ====")
    print(classification_report(all_gts_flat, all_preds_flat, zero_division=0))
    print("==== Overall confusion matrix ====")
    print(confusion_matrix(all_gts_flat, all_preds_flat))

# --- SAVE PCD, PLY, AND PNG ---
print("\n==== Saving colored prediction PCD/PLY/PNG for each scene ====")
for i, (scene_pred, points) in enumerate(zip(preds, points_set)):
    n_cls = color_map.shape[0]
    safe_pred = np.clip(scene_pred, 0, n_cls-1)
    colors = color_map[safe_pred]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # PCD
    vis_path_pcd = os.path.join(out_vis_dir, f'predicted_scene_{i+1}.pcd')
    o3d.io.write_point_cloud(vis_path_pcd, pcd)
    # PLY
    vis_path_ply = os.path.join(out_vis_dir, f'predicted_scene_{i+1}.ply')
    o3d.io.write_point_cloud(vis_path_ply, pcd)
    # PNG snapshot (scatter, side view)
    fig = plt.figure(figsize=(6,6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c=colors, s=0.1)
    ax.set_title(f'Predictions Scene {i+1}')
    ax.set_axis_off()
    plt.tight_layout()
    vis_path_png = os.path.join(out_vis_dir, f'predicted_scene_{i+1}.png')
    plt.savefig(vis_path_png, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"  Saved: {vis_path_pcd}  {vis_path_ply}  {vis_path_png}")

print("\n==== Analysis Complete ====")
