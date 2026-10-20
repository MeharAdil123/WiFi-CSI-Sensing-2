"""
Evaluation, Benchmarking and Metric Reporting Module for WiFi-CSI HAR.

Generates:
- Overall Accuracy & Balanced Accuracy
- Macro and Weighted Precision, Recall, F1
- Per-class metrics & support
- Raw and row-normalized confusion matrices
- Single-sample inference latency (ms) and throughput (windows/sec)
"""

import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.models import get_model, count_parameters

CLASS_NAMES = ["Empty", "Sitting", "Standing", "Walking", "Falling"]


def evaluate_model_on_loader(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs inference on test_loader.
    Returns: targets, predictions, probabilities.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_targets.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.append(probs)

    targets = np.array(all_targets, dtype=np.int64)
    predictions = np.array(all_preds, dtype=np.int64)
    probabilities = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, len(CLASS_NAMES)))

    return targets, predictions, probabilities


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES
) -> Dict:
    """Computes full suite of classification and fairness metrics."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    raw_cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_cm = raw_cm.astype(float) / raw_cm.sum(axis=1, keepdims=True)
        norm_cm = np.nan_to_num(norm_cm)

    per_class = {}
    for idx, c in enumerate(class_names):
        c_mask = y_true == idx
        support = int(np.sum(c_mask))
        tp = int(raw_cm[idx, idx])
        fn = int(np.sum(raw_cm[idx, :]) - tp)
        fp = int(np.sum(raw_cm[:, idx]) - tp)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        per_class[c] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "support": support,
            "tp": tp, "fp": fp, "fn": fn
        }

    # Minority safety metric: Falling recall and false negative rate
    falling_idx = class_names.index("Falling") if "Falling" in class_names else 4
    fall_support = per_class["Falling"]["support"]
    fall_recall = per_class["Falling"]["recall"]
    fall_fn = per_class["Falling"]["fn"]
    fall_miss_rate = (fall_fn / fall_support) if fall_support > 0 else 0.0

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "falling_recall": float(fall_recall),
        "falling_miss_rate": float(fall_miss_rate),
        "per_class": per_class,
        "confusion_matrix_raw": raw_cm.tolist(),
        "confusion_matrix_norm": norm_cm.tolist()
    }


def benchmark_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 250, 90),
    runs: int = 100,
    warmup: int = 20,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Measures single-window inference latency and throughput on available hardware."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Measurement
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    throughput = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "mean_latency_ms": mean_lat,
        "std_latency_ms": std_lat,
        "p95_latency_ms": p95_lat,
        "throughput_windows_per_sec": throughput,
        "device": str(device)
    }


def print_evaluation_report(metrics: Dict, title: str = "Test Evaluation Report"):
    """Prints a clean, formatted terminal summary of evaluation results."""
    print("\n" + "=" * 65)
    print(f" {title.upper()} ")
    print("=" * 65)
    print(f"Overall Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']*100:.2f}%")
    print(f"Macro F1-Score:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    print(f"Falling Recall:    {metrics['falling_recall']*100:.2f}% (Miss Rate: {metrics['falling_miss_rate']*100:.2f}%)")
    print("-" * 65)
    print(f"{'Class':<12s} | {'Precision':<10s} | {'Recall':<10s} | {'F1-Score':<10s} | {'Support':<8s}")
    print("-" * 65)
    for c, stats in metrics["per_class"].items():
        print(f"{c:<12s} | {stats['precision']:<10.4f} | {stats['recall']:<10.4f} | {stats['f1']:<10.4f} | {stats['support']:<8d}")
    print("-" * 65)
    print("Confusion Matrix (Rows=True, Cols=Predicted):")
    raw_cm = np.array(metrics["confusion_matrix_raw"])
    print(raw_cm)
    print("=" * 65)
