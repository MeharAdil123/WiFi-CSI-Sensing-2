import os
import glob
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from src.models import Original_CNN_BiLSTM as UT_HAR_CNN_BiLSTM

CLASS_NAMES = ["Empty", "Sitting", "Standing", "Walking", "Falling"]

def audit():
    print("=" * 60)
    print("1. DATASET AUDIT (Data/MyCSI)")
    print("=" * 60)
    
    X_train = np.load("Data/MyCSI/data/X_train.npy")
    y_train = np.load("Data/MyCSI/label/y_train.npy")
    X_val = np.load("Data/MyCSI/data/X_val.npy")
    y_val = np.load("Data/MyCSI/label/y_val.npy")
    X_test = np.load("Data/MyCSI/data/X_test.npy")
    y_test = np.load("Data/MyCSI/label/y_test.npy")

    print(f"X_train: shape={X_train.shape}, dtype={X_train.dtype}, min={X_train.min():.4f}, max={X_train.max():.4f}")
    print(f"X_val:   shape={X_val.shape}, dtype={X_val.dtype}, min={X_val.min():.4f}, max={X_val.max():.4f}")
    print(f"X_test:  shape={X_test.shape}, dtype={X_test.dtype}, min={X_test.min():.4f}, max={X_test.max():.4f}")
    print(f"y_train: shape={y_train.shape}, unique={np.unique(y_train, return_counts=True)}")
    print(f"y_val:   shape={y_val.shape}, unique={np.unique(y_val, return_counts=True)}")
    print(f"y_test:  shape={y_test.shape}, unique={np.unique(y_test, return_counts=True)}")

    total_windows = len(y_train) + len(y_val) + len(y_test)
    print(f"\nTotal windows: {total_windows} (Train: {len(y_train)} / Val: {len(y_val)} / Test: {len(y_test)})")
    
    # Class distribution summary
    for c_idx, c_name in enumerate(CLASS_NAMES):
        tr_c = np.sum(y_train == c_idx)
        val_c = np.sum(y_val == c_idx)
        te_c = np.sum(y_test == c_idx)
        tot_c = tr_c + val_c + te_c
        pct = (tot_c / total_windows) * 100
        print(f"  Class {c_idx} ({c_name:8s}): Total={tot_c:4d} ({pct:5.1f}%) | Train={tr_c:4d} | Val={val_c:3d} | Test={te_c:3d}")

    print("\n" + "=" * 60)
    print("2. CHECKPOINTS AUDIT & EVALUATION")
    print("=" * 60)

    # Let's test with train-range normalization vs test-own normalization
    tr_min, tr_max = X_train.min(), X_train.max()
    te_min, te_max = X_test.min(), X_test.max()
    print(f"Train min-max range: [{tr_min:.4f}, {tr_max:.4f}]")
    print(f"Test min-max range:  [{te_min:.4f}, {te_max:.4f}]")

    # Normalized tensors
    X_test_train_scaled = (X_test.astype(np.float32) - tr_min) / (tr_max - tr_min)
    X_test_self_scaled = (X_test.astype(np.float32) - te_min) / (te_max - te_min)

    t_train_scaled = torch.from_numpy(X_test_train_scaled.astype(np.float32)).unsqueeze(1)
    t_self_scaled = torch.from_numpy(X_test_self_scaled.astype(np.float32)).unsqueeze(1)

    ckpt_files = sorted(glob.glob("checkpoints/cnn_bilstm_seed_*.pth"))
    for ckpt_p in ckpt_files:
        ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        seed = ckpt.get("seed", "unknown")
        stored_acc = ckpt.get("test_accuracy", None)
        
        model = UT_HAR_CNN_BiLSTM(num_classes=5)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            out_self = model(t_self_scaled).argmax(dim=1).numpy()
            out_tr = model(t_train_scaled).argmax(dim=1).numpy()

        acc_self = (out_self == y_test).mean() * 100
        acc_tr = (out_tr == y_test).mean() * 100
        stored_str = f"{stored_acc*100:.2f}%" if stored_acc is not None else "N/A"

        print(f"\nCheckpoint: {os.path.basename(ckpt_p)} (Seed {seed})")
        print(f"  Stored accuracy in metadata: {stored_str}")
        print(f"  Evaluated with self-scaling:  {acc_self:.2f}% ({(out_self == y_test).sum()}/{len(y_test)})")
        print(f"  Evaluated with train-scaling: {acc_tr:.2f}% ({(out_tr == y_test).sum()}/{len(y_test)})")
        
        if seed in [42, 456]:
            print(f"  Confusion Matrix (self-scaling, Seed {seed}):")
            print(confusion_matrix(y_test, out_self))
            print(f"  Per-class metrics (Seed {seed}):")
            print(classification_report(y_test, out_self, target_names=CLASS_NAMES, digits=4, zero_division=0))

if __name__ == "__main__":
    audit()
