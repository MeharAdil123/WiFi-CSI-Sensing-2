"""
Live Streaming CSI Inference Engine.

Designed for future continuous real-time ESP32-S3 deployment:
- Maintains a rolling FIFO queue of 250 processed CSI packet frames.
- Reuses the EXACT feature extraction pipeline (zero-drop, linear interpolation to 90).
- Normalizes using saved training-set parameters (never test/batch statistics).
- Smooths predictions over consecutive inference steps to prevent flicker.
- Provides immediate alerts for Falling events.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.models import get_model

CLASS_NAMES = ["Empty", "Sitting", "Standing", "Walking", "Falling"]


class LiveCSIInferenceEngine:
    def __init__(
        self,
        checkpoint_path: str,
        window_size: int = 250,
        target_features: int = 90,
        smoothing_window: int = 3,
        fall_threshold: float = 0.50,
        device: Optional[torch.device] = None
    ):
        self.window_size = window_size
        self.target_features = target_features
        self.smoothing_window = smoothing_window
        self.fall_threshold = fall_threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint and metadata
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model_name = checkpoint.get("model_name", "CNN_BiLSTM")
        self.num_classes = checkpoint.get("num_classes", len(CLASS_NAMES))
        self.classes = checkpoint.get("classes", CLASS_NAMES)

        # Normalization parameters from training
        self.train_min = checkpoint.get("train_min", 1.0)
        self.train_max = checkpoint.get("train_max", 25.88)
        self.norm_denom = max(self.train_max - self.train_min, 1e-6)

        # Initialize model
        self.model = get_model(self.model_name, num_classes=self.num_classes).to(self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Packet and prediction buffers
        self.packet_buffer = deque(maxlen=window_size)
        self.history_predictions = deque(maxlen=smoothing_window)
        self.packets_since_last_inference = 0

    def raw_csi_to_features(self, raw_csi: Union[List[int], np.ndarray]) -> Optional[np.ndarray]:
        """
        Processes a single raw 384-element CSI payload:
        1. Pair into in-phase (I) and quadrature (Q).
        2. Magnitude: sqrt(I^2 + Q^2).
        3. Remove zero/null subcarriers.
        4. Linear interpolation to 90 features.
        """
        arr = np.asarray(raw_csi, dtype=np.float32)
        if arr.ndim != 1 or arr.size != 384 or not np.isfinite(arr).all():
            return None

        real = arr[0::2]
        imag = arr[1::2]
        mags = np.sqrt(real**2 + imag**2)
        nonzero = mags[mags > 0]

        if len(nonzero) < 2:
            return None

        old_x = np.linspace(0, 1, len(nonzero))
        new_x = np.linspace(0, 1, self.target_features)
        features = np.interp(new_x, old_x, nonzero).astype(np.float32)
        return features

    def push_packet(self, raw_csi: Union[List[int], np.ndarray]) -> bool:
        """
        Pushes a new raw packet. Returns True if packet was valid.
        """
        feat = self.raw_csi_to_features(raw_csi)
        if feat is not None:
            self.packet_buffer.append(feat)
            self.packets_since_last_inference += 1
            return True
        return False

    def is_ready(self) -> bool:
        """Returns True when buffer has accumulated a full 250-packet window."""
        return len(self.packet_buffer) == self.window_size

    def predict(self) -> Optional[Dict]:
        """
        Runs inference on the current 250-packet rolling window.
        Returns:
            Dict containing predicted class, probabilities, smoothed class, and fall alert.
        """
        if not self.is_ready():
            return None

        # Build window array: [250, 90]
        window = np.array(self.packet_buffer, dtype=np.float32)

        # Apply training-fitted min-max scaling
        norm_window = (window - self.train_min) / self.norm_denom

        # Shape: [1, 1, 250, 90]
        x_tensor = torch.from_numpy(norm_window).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

        pred_class = self.classes[pred_idx]
        self.history_predictions.append(pred_idx)

        # Majority vote smoothing
        counts = np.bincount(list(self.history_predictions), minlength=self.num_classes)
        smoothed_idx = int(np.argmax(counts))
        smoothed_class = self.classes[smoothed_idx]

        # Safety Fall Alert
        falling_idx = self.classes.index("Falling") if "Falling" in self.classes else 4
        fall_prob = float(probs[falling_idx])
        is_fall_alert = bool(fall_prob >= self.fall_threshold)

        self.packets_since_last_inference = 0

        return {
            "prediction": pred_class,
            "prediction_index": pred_idx,
            "confidence": float(probs[pred_idx]),
            "smoothed_prediction": smoothed_class,
            "falling_probability": fall_prob,
            "fall_alert": is_fall_alert,
            "probabilities": {name: float(probs[i]) for i, name in enumerate(self.classes)}
        }
