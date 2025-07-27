import h5py, numpy as np
with h5py.File('/home/s2737104/APES/data/all_test_scenes.h5') as f:
    seg_labels = f['seg_labels'][:]
    gt = seg_labels.argmax(axis=2)  # shape: (num_scenes, num_points)
    for i, g in enumerate(gt):
        print(f"Scene {i+1}:")
        unique, counts = np.unique(g, return_counts=True)
        print(dict(zip(unique, counts)))

# import h5py, numpy as np
# with h5py.File('/home/s2737104/APES/data/all_test_scenes.h5') as f:
#     np.save('gt_labels.npy', f['seg_labels'][:].argmax(axis=2))  # shape: (num_scenes, num_points)

# from sklearn.metrics import classification_report, confusion_matrix

# import numpy as np
# gt = np.load('gt_labels.npy')          # shape (num_scenes, num_points)
# pred = np.load('predictions.npy')      # same shape

# for i in range(gt.shape[0]):
#     print(f"Scene {i+1} report:")
#     print(classification_report(gt[i].flatten(), pred[i].flatten(), target_names=["target", "alien"]))
#     print(confusion_matrix(gt[i].flatten(), pred[i].flatten()))
