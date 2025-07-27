import apes  # register APES modules
import argparse
import torch
import h5py
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner
import os
from apes.structures.seg_data_sample import SegDataSample  # Adjust this import as needed!

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='model checkpoint file path')
    parser.add_argument('--hdf5', required=True, help='path to test HDF5 file')
    parser.add_argument('--chunk_size', type=int, default=10240, help='points per chunk')
    parser.add_argument('--device', default='cuda', help='device to use for inference')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes in your model')
    parser.add_argument('--out', default='predictions.npy', help='output npy file to save predictions')
    args = parser.parse_args()
    return args

def load_model(cfg, checkpoint, device='cuda'):
    cfg.load_from = checkpoint
    runner = Runner.from_cfg(cfg)
    model = runner.model
    model.eval()
    model.to(device)
    return model

def chunk_point_cloud(points, colors, chunk_size):
    N = points.shape[0]
    chunks = []
    color_chunks = []
    chunk_valid_lengths = []
    for i in range(0, N, chunk_size):
        pt_chunk = points[i:i+chunk_size]
        col_chunk = colors[i:i+chunk_size]
        valid_len = len(pt_chunk)
        # Pad if needed
        if valid_len < chunk_size:
            pad_pts = np.zeros((chunk_size - valid_len, 3))
            pad_cols = np.zeros((chunk_size - valid_len, 3), dtype=col_chunk.dtype)
            pt_chunk = np.vstack([pt_chunk, pad_pts])
            col_chunk = np.vstack([col_chunk, pad_cols])
        chunks.append(pt_chunk)
        color_chunks.append(col_chunk)
        chunk_valid_lengths.append(valid_len)
    return chunks, color_chunks, chunk_valid_lengths

def make_dummy_data_sample(chunk_size, device='cpu'):
    dummy = SegDataSample()
    dummy.gt_cls_label_onehot = torch.zeros((16, 1), dtype=torch.float32, device=device)
    dummy.gt_seg_label_onehot = torch.zeros((2, chunk_size), dtype=torch.float32, device=device)
    return [dummy]


@torch.no_grad()
def infer_on_hdf5(model, hdf5_path, chunk_size=10240, device='cuda', num_classes=2, out_path='predictions.npy'):
    with h5py.File(hdf5_path, 'r') as f:
        points_set = f['seg_points'][:]  # shape: (num_scenes, N, 3)
        colors_set = f['seg_colors'][:]  # shape: (num_scenes, N, 3)
        # If uint8, normalize for APES model (0–255 to 0–1)
        if colors_set.dtype == np.uint8:
            colors_set = colors_set.astype(np.float32) / 255.

        all_scene_preds = []

        for idx in range(points_set.shape[0]):
            points = points_set[idx]
            colors = colors_set[idx]
            N = points.shape[0]
            print(f"\nScene {idx+1}/{points_set.shape[0]}: {N} points")
            chunks, color_chunks, chunk_valid_lengths = chunk_point_cloud(points, colors, chunk_size)
            scene_preds = []
            for ci, (pts, cols, valid_len) in enumerate(zip(chunks, color_chunks, chunk_valid_lengths)):
                input_arr = np.concatenate([pts, cols], axis=1).T  # (6, chunk_size)
                input_tensor = torch.from_numpy(input_arr).float().unsqueeze(0).to(device)
                dummy_data_samples = make_dummy_data_sample(chunk_size, device=device)
                # APES returns a list of SegDataSample, get .pred_seg_label as per-point prediction
                pred_samples = model(input_tensor, dummy_data_samples, mode='predict')
                pred_label = pred_samples[0].pred_seg_label[:valid_len].cpu().numpy()  # shape: (valid_len,)
                scene_preds.append(pred_label)
                print(f"  Chunk {ci+1}/{len(chunks)} done, kept {valid_len} preds")
            # Reconstruct predictions for all points in the scene
            scene_preds = np.concatenate(scene_preds, axis=0)  # (N,)
            assert scene_preds.shape[0] == N, f"Predictions do not match scene size: {scene_preds.shape[0]} != {N}"
            all_scene_preds.append(scene_preds)
            print(f"  Scene {idx+1} finished, total predictions: {scene_preds.shape}")

        all_scene_preds = np.array(all_scene_preds, dtype=np.int32)  # (num_scenes, N)
        # all_scene_preds = 1 - all_scene_preds   # <-- Flip 0<->1 soalnya kebalik labellingnya
        np.save(out_path, all_scene_preds)
        print(f"\nSaved all predictions to: {out_path}")

    print("All done.")

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = load_model(cfg, args.checkpoint, device=args.device)
    infer_on_hdf5(
        model,
        args.hdf5,
        chunk_size=args.chunk_size,
        device=args.device,
        num_classes=args.num_classes,
        out_path=args.out
    )

if __name__ == '__main__':
    main()
