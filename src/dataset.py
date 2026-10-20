"""
Dataset and Data Preprocessing Pipeline for WiFi-CSI Human Activity Recognition.

Supports:
1. Legacy Random Split (for reproduction of published & repository baselines).
2. Leakage-Resistant Recording-Grouped Split (strictly zero window overlap across sets).
3. Robust normalization: train-fitted min-max, z-score standardization, dynamic high-pass baseline removal.
4. Class weighting for extreme class imbalance (Falling).
"""

import os
import glob
import re
import csv
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

CLASS_NAMES = ["Empty", "Sitting", "Standing", "Walking", "Falling"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

WINDOW_SIZE = 250
STRIDE = 50
TARGET_FEATURES = 90


class CSIDataset(Dataset):
    """PyTorch Dataset wrapper for CSI windows."""
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        assert len(X) == len(y), f"Length mismatch: {len(X)} vs {len(y)}"
        self.X = torch.from_numpy(X.astype(np.float32)).unsqueeze(1) # [N, 1, 250, 90]
        self.y = torch.from_numpy(y.astype(np.int64))
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.y[idx]


def compute_class_weights(y_train: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    """
    Computes balanced class weights inversely proportional to class frequencies:
    w_c = N / (C * N_c)
    """
    total = len(y_train)
    weights = []
    for c in range(num_classes):
        count = int(np.sum(y_train == c))
        if count == 0:
            weights.append(1.0)
        else:
            weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# 1. LOAD LEGACY RANDOM SPLIT (Data/MyCSI)
# ============================================================================

def load_legacy_split(
    data_dir: str = "Data/MyCSI",
    norm_method: str = "train_minmax"
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Loads the original 2,241 windows (1568 train, 336 val, 337 test)
    used in the repository checkpoints and presentation slides.
    """
    X_train = np.load(os.path.join(data_dir, "data/X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(data_dir, "label/y_train.npy")).astype(np.int64)
    X_val = np.load(os.path.join(data_dir, "data/X_val.npy")).astype(np.float32)
    y_val = np.load(os.path.join(data_dir, "label/y_val.npy")).astype(np.int64)
    X_test = np.load(os.path.join(data_dir, "data/X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(data_dir, "label/y_test.npy")).astype(np.int64)

    stats = {}
    if norm_method == "train_minmax":
        tr_min = float(X_train.min())
        tr_max = float(X_train.max())
        stats["min"] = tr_min
        stats["max"] = tr_max
        denom = max(tr_max - tr_min, 1e-6)
        X_train = (X_train - tr_min) / denom
        X_val = (X_val - tr_min) / denom
        X_test = (X_test - tr_min) / denom
    elif norm_method == "standardize":
        tr_mean = float(X_train.mean())
        tr_std = float(X_train.std())
        stats["mean"] = tr_mean
        stats["std"] = tr_std
        denom = max(tr_std, 1e-6)
        X_train = (X_train - tr_mean) / denom
        X_val = (X_val - tr_mean) / denom
        X_test = (X_test - tr_mean) / denom
    elif norm_method == "dynamic_demean":
        # Subtract per-window temporal mean to remove static background
        X_train = X_train - X_train.mean(axis=1, keepdims=True)
        X_val = X_val - X_val.mean(axis=1, keepdims=True)
        X_test = X_test - X_test.mean(axis=1, keepdims=True)

    data = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }
    return data, stats


# ============================================================================
# 2. LEAKAGE-RESISTANT RECORDING-GROUPED SPLIT
# ============================================================================

def make_windows_from_recording(frames: np.ndarray, window_size: int = 250, stride: int = 50) -> np.ndarray:
    """Slices consecutive packet frames into fixed (N, 250, 90) windows."""
    if len(frames) < window_size:
        return np.empty((0, window_size, frames.shape[1]), dtype=np.float32)
    starts = range(0, len(frames) - window_size + 1, stride)
    return np.stack([frames[s:s + window_size] for s in starts]).astype(np.float32)


def load_and_group_recordings(
    recordings_dir: str = "processed_recordings",
    extra_recordings_dir: Optional[str] = "processed_new_test_recordings"
) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    """
    Loads pre-extracted windowed .npy arrays from recording directories.
    Returns dictionary mapping class name -> list of (filename, windows_array).
    """
    by_class = {c: [] for c in CLASS_NAMES}
    dirs = [recordings_dir]
    if extra_recordings_dir and os.path.exists(extra_recordings_dir):
        dirs.append(extra_recordings_dir)

    for d in dirs:
        for c in CLASS_NAMES:
            c_dir = os.path.join(d, c)
            if not os.path.exists(c_dir):
                continue
            for f in sorted(glob.glob(os.path.join(c_dir, "*.npy"))):
                arr = np.load(f)
                if len(arr) > 0 and arr.ndim == 3 and arr.shape[1:] == (WINDOW_SIZE, TARGET_FEATURES):
                    by_class[c].append((os.path.basename(f), arr))

    return by_class


def build_grouped_dataset(
    recordings_dir: str = "processed_recordings",
    extra_recordings_dir: Optional[str] = "processed_new_test_recordings",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    norm_method: str = "train_minmax"
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, List[str]]]:
    """
    Creates a strictly LEAKAGE-FREE split at the whole-recording level.
    Guarantees that no recording appears in more than one split.
    """
    by_class = load_and_group_recordings(recordings_dir, extra_recordings_dir)
    rng = np.random.RandomState(seed)

    train_recs, val_recs, test_recs = [], [], []
    split_manifest = {"train": [], "val": [], "test": []}

    for c in CLASS_NAMES:
        recs = list(by_class[c])
        rng.shuffle(recs)
        n_recs = len(recs)
        if n_recs < 3:
            raise ValueError(f"Class {c} has only {n_recs} recordings, need at least 3 for train/val/test split.")

        n_train = max(1, int(round(n_recs * train_ratio)))
        n_val = max(1, int(round(n_recs * val_ratio)))
        # Ensure at least 1 in test
        if n_train + n_val >= n_recs:
            n_train = max(1, n_recs - 2)
            n_val = 1
        n_test = n_recs - n_train - n_val

        tr = recs[:n_train]
        vl = recs[n_train:n_train + n_val]
        te = recs[n_train + n_val:]

        for name, w in tr:
            train_recs.append((c, name, w))
            split_manifest["train"].append(f"{c}/{name}")
        for name, w in vl:
            val_recs.append((c, name, w))
            split_manifest["val"].append(f"{c}/{name}")
        for name, w in te:
            test_recs.append((c, name, w))
            split_manifest["test"].append(f"{c}/{name}")

    # Assemble raw arrays
    X_train_raw = np.concatenate([w for _, _, w in train_recs], axis=0)
    y_train = np.concatenate([np.full(len(w), CLASS_TO_IDX[c], dtype=np.int64) for c, _, w in train_recs])

    X_val_raw = np.concatenate([w for _, _, w in val_recs], axis=0)
    y_val = np.concatenate([np.full(len(w), CLASS_TO_IDX[c], dtype=np.int64) for c, _, w in val_recs])

    X_test_raw = np.concatenate([w for _, _, w in test_recs], axis=0)
    y_test = np.concatenate([np.full(len(w), CLASS_TO_IDX[c], dtype=np.int64) for c, _, w in test_recs])

    # Fit normalization strictly on training recordings
    stats = {}
    if norm_method == "train_minmax":
        tr_min = float(X_train_raw.min())
        tr_max = float(X_train_raw.max())
        stats["min"] = tr_min
        stats["max"] = tr_max
        denom = max(tr_max - tr_min, 1e-6)
        X_train = (X_train_raw - tr_min) / denom
        X_val = (X_val_raw - tr_min) / denom
        X_test = (X_test_raw - tr_min) / denom
    elif norm_method == "standardize":
        tr_mean = float(X_train_raw.mean())
        tr_std = float(X_train_raw.std())
        stats["mean"] = tr_mean
        stats["std"] = tr_std
        denom = max(tr_std, 1e-6)
        X_train = (X_train_raw - tr_mean) / denom
        X_val = (X_val_raw - tr_mean) / denom
        X_test = (X_test_raw - tr_mean) / denom
    elif norm_method == "dynamic_demean":
        X_train = X_train_raw - X_train_raw.mean(axis=1, keepdims=True)
        X_val = X_val_raw - X_val_raw.mean(axis=1, keepdims=True)
        X_test = X_test_raw - X_test_raw.mean(axis=1, keepdims=True)
    else:
        X_train, X_val, X_test = X_train_raw, X_val_raw, X_test_raw

    data = {
        "X_train": X_train.astype(np.float32), "y_train": y_train,
        "X_val": X_val.astype(np.float32), "y_val": y_val,
        "X_test": X_test.astype(np.float32), "y_test": y_test,
    }
    return data, stats, split_manifest


def get_dataloaders(
    data: Dict[str, np.ndarray],
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Wraps numpy arrays into PyTorch DataLoaders."""
    train_ds = CSIDataset(data["X_train"], data["y_train"])
    val_ds = CSIDataset(data["X_val"], data["y_val"])
    test_ds = CSIDataset(data["X_test"], data["y_test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
