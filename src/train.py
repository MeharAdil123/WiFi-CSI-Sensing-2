"""
Reproducible Training Pipeline for WiFi-CSI Human Activity Recognition.

Features:
- Deterministic seeding across Python, NumPy, and PyTorch.
- Loss options: Standard Cross-Entropy, Class-Weighted Cross-Entropy, Focal Loss.
- Validation early stopping monitoring Macro-F1 (prioritizes minority Falling class).
- Full metadata checkpointing with reproducible configuration and history.
"""

import os
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from src.models import get_model, count_parameters


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss to address extreme class imbalance:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """Runs validation and computes loss, accuracy, and macro-F1."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 5,
    epochs: int = 50,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    loss_type: str = "ce",  # "ce", "weighted_ce", or "focal"
    class_weights: Optional[torch.Tensor] = None,
    patience: int = 15,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints/experiments",
    experiment_name: Optional[str] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict:
    """
    Executes a complete, reproducible training experiment.
    """
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name, num_classes=num_classes).to(device)
    params_count = count_parameters(model)

    # Configure loss function
    if loss_type == "weighted_ce" and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    elif loss_type == "focal":
        weights = class_weights.to(device) if class_weights is not None else None
        criterion = FocalLoss(alpha=weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "lr": []
    }

    best_val_f1 = -1.0
    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    start_time = time.time()

    if verbose:
        print(f"\n--- Training {model_name} (Params: {params_count:,d}, Seed: {seed}, Loss: {loss_type}) ---")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if verbose and (epoch % 5 == 0 or epoch == epochs or stale_epochs == 0):
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% Macro-F1: {val_f1:.4f} | "
                f"{'*Best*' if stale_epochs == 0 else ''}"
            )

        if stale_epochs >= patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
            break

    total_time = time.time() - start_time

    # Save checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    exp_tag = experiment_name or f"{model_name}_seed{seed}_{loss_type}"
    ckpt_path = os.path.join(checkpoint_dir, f"{exp_tag}.pth")

    save_dict = {
        "model_state_dict": best_state,
        "model_name": model_name,
        "num_classes": num_classes,
        "seed": seed,
        "loss_type": loss_type,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_acc": best_val_acc,
        "params_count": params_count,
        "total_time_seconds": total_time,
        "history": history
    }
    torch.save(save_dict, ckpt_path)

    return {
        "model_name": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_acc": best_val_acc,
        "params_count": params_count,
        "training_time": total_time,
        "checkpoint_path": ckpt_path,
        "history": history
    }
