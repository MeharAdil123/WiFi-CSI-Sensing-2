"""
Standalone Script to Check and Verify SE-TCN Accuracy on Both Splits.

Supports:
1. Track B.1: Legacy Random Split (51 original files)
2. Track B.2: Leakage-Free Grouped Split (76 sessions)

Usage:
    python check_setcn_accuracy.py
    python check_setcn_accuracy.py --split grouped --seed 789
    python check_setcn_accuracy.py --split legacy --seed 2024
"""

import argparse
import os
import torch
import numpy as np

from src.models import get_model
from src.dataset import load_legacy_split, build_grouped_dataset, get_dataloaders
from src.evaluate import evaluate_model_on_loader, compute_metrics, CLASS_NAMES


def evaluate_setcn(split: str = "both", seed: int = 42):
    print("=" * 75)
    print("  SE-TCN ACCURACY VERIFICATION TOOL")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Legacy Split Evaluation
    if split in ["legacy", "both"]:
        ckpt_path = f"checkpoints/experiments/legacy/legacy_SE_TCN_seed{seed}.pth"
        if not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint not found at {ckpt_path}")
        else:
            print(f"\n--- [Track B.1] Legacy Split Evaluation (Seed {seed}) ---")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = get_model("SE_TCN", num_classes=5).to(device)
            model.load_state_dict(ckpt["model_state_dict"])

            data, _ = load_legacy_split(norm_method="train_minmax")
            _, _, test_loader = get_dataloaders(data, batch_size=32)

            y_true, y_pred, _ = evaluate_model_on_loader(model, test_loader, device=device)
            m = compute_metrics(y_true, y_pred)

            print(f"Test Accuracy:          {m['accuracy'] * 100:.2f}%")
            print(f"Balanced Accuracy:      {m['balanced_accuracy'] * 100:.2f}%")
            print(f"Macro-F1 Score:         {m['macro_f1']:.4f}")
            print(f"Fall Recall:            {m['falling_recall'] * 100:.2f}%")
            print("\nPer-Class Breakdown:")
            for cname in CLASS_NAMES:
                rec = m['per_class'][cname]['recall'] * 100
                prec = m['per_class'][cname]['precision'] * 100
                supp = m['per_class'][cname]['support']
                print(f"  {cname:<10}: Recall = {rec:6.2f}%, Precision = {prec:6.2f}% (Support = {supp})")

    # 2. Grouped Split Evaluation
    if split in ["grouped", "both"]:
        ckpt_path = f"checkpoints/experiments/grouped/grouped_SE_TCN_seed{seed}.pth"
        if not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint not found at {ckpt_path}")
        else:
            print(f"\n--- [Track B.2] Leakage-Free Grouped Split Evaluation (Seed {seed}) ---")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = get_model("SE_TCN", num_classes=5).to(device)
            model.load_state_dict(ckpt["model_state_dict"])

            data, _, _ = build_grouped_dataset(seed=seed, norm_method="train_minmax")
            _, _, test_loader = get_dataloaders(data, batch_size=32)

            y_true, y_pred, _ = evaluate_model_on_loader(model, test_loader, device=device)
            m = compute_metrics(y_true, y_pred)

            print(f"Test Accuracy:          {m['accuracy'] * 100:.2f}%")
            print(f"Balanced Accuracy:      {m['balanced_accuracy'] * 100:.2f}%")
            print(f"Macro-F1 Score:         {m['macro_f1']:.4f}")
            print(f"Fall Recall:            {m['falling_recall'] * 100:.2f}%")
            print("\nPer-Class Breakdown:")
            for cname in CLASS_NAMES:
                rec = m['per_class'][cname]['recall'] * 100
                prec = m['per_class'][cname]['precision'] * 100
                supp = m['per_class'][cname]['support']
                print(f"  {cname:<10}: Recall = {rec:6.2f}%, Precision = {prec:6.2f}% (Support = {supp})")
            
            print("\nConfusion Matrix (Rows=True, Cols=Pred):")
            cm = np.array(m['confusion_matrix_raw'])
            print(f"{'':<10} " + " ".join([f"{c[:5]:>7}" for c in CLASS_NAMES]))
            for idx, cname in enumerate(CLASS_NAMES):
                row_str = " ".join([f"{val:7d}" for val in cm[idx]])
                print(f"{cname:<10} {row_str}")


def main():
    parser = argparse.ArgumentParser(description="Check accuracy of SE-TCN model checkpoints.")
    parser.add_argument("--split", type=str, default="both", choices=["legacy", "grouped", "both"],
                        help="Which evaluation split to check ('legacy', 'grouped', or 'both').")
    parser.add_argument("--seed", type=int, default=42, choices=[42, 123, 456, 789, 2024],
                        help="Random seed checkpoint to evaluate (42, 123, 456, 789, or 2024).")
    args = parser.parse_args()

    evaluate_setcn(split=args.split, seed=args.seed)


if __name__ == "__main__":
    main()
