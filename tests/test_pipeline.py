"""
Automated Unit Tests for WiFi-CSI Sensing Pipeline.

Tests:
1. Signal preprocessing & subcarrier interpolation.
2. Model forward passes and parameter shapes.
3. Strict zero-leakage verification for recording-grouped split.
4. Checkpoint saving and loading integrity.
5. Live streaming FIFO buffer & inference engine.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.models import get_model, count_parameters
from src.dataset import load_legacy_split, build_grouped_dataset, make_windows_from_recording
from src.live_inference import LiveCSIInferenceEngine


class TestWiFiCSIPipeline(unittest.TestCase):

    def test_01_model_forward_passes(self):
        """Verifies that all baseline and proposed architectures accept [B, 1, 250, 90] and output [B, 5]."""
        batch_size = 3
        dummy_input = torch.randn(batch_size, 1, 250, 90)
        models = ["MLP", "LeNet", "BiLSTM", "CNN_GRU", "CNN_BiLSTM", "MS_CNN_BiGRU", "SE_TCN"]

        for m_name in models:
            model = get_model(m_name, num_classes=5)
            model.eval()
            with torch.no_grad():
                out = model(dummy_input)
            self.assertEqual(
                out.shape, (batch_size, 5),
                f"{m_name} failed: expected shape {(batch_size, 5)}, got {out.shape}"
            )
            self.assertTrue(torch.isfinite(out).all(), f"{m_name} produced non-finite outputs.")

    def test_02_zero_leakage_in_grouped_split(self):
        """Verifies that the recording-grouped split has 0% recording or window contamination."""
        _, _, manifest = build_grouped_dataset(seed=42)
        train_set = set(manifest["train"])
        val_set = set(manifest["val"])
        test_set = set(manifest["test"])

        # Check disjointness
        train_val_overlap = train_set.intersection(val_set)
        train_test_overlap = train_set.intersection(test_set)
        val_test_overlap = val_set.intersection(test_set)

        self.assertEqual(len(train_val_overlap), 0, f"Leakage detected between Train and Val: {train_val_overlap}")
        self.assertEqual(len(train_test_overlap), 0, f"Leakage detected between Train and Test: {train_test_overlap}")
        self.assertEqual(len(val_test_overlap), 0, f"Leakage detected between Val and Test: {val_test_overlap}")

    def test_03_raw_csi_feature_conversion(self):
        """Tests that raw 384 CSI array converts correctly to 90 finite, non-negative amplitudes."""
        engine = LiveCSIInferenceEngine.__new__(LiveCSIInferenceEngine)
        engine.target_features = 90

        # Construct dummy raw CSI: 384 integers (-10 to 10)
        raw_csi = np.random.randint(-10, 11, size=384).astype(np.int8)
        features = LiveCSIInferenceEngine.raw_csi_to_features(engine, raw_csi)

        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (90,))
        self.assertTrue((features >= 0).all(), "Amplitudes must be non-negative.")
        self.assertTrue(np.isfinite(features).all(), "Features must be finite.")

    def test_04_windowing_logic(self):
        """Verifies exact window calculation W = floor((N - 250) / 50) + 1."""
        n_packets = 2395  # falling_01 recording size
        dummy_frames = np.random.randn(n_packets, 90).astype(np.float32)
        windows = make_windows_from_recording(dummy_frames, window_size=250, stride=50)

        expected_windows = ((n_packets - 250) // 50) + 1
        self.assertEqual(len(windows), expected_windows)
        self.assertEqual(len(windows), 43)  # Matches Falling_01 exact window count

    def test_05_checkpoint_saving_and_loading(self):
        """Verifies that model parameters survive saving and loading without numerical drift."""
        model = get_model("MS_CNN_BiGRU", num_classes=5)
        model.eval()
        dummy_x = torch.randn(1, 1, 250, 90)

        with torch.no_grad():
            orig_out = model(dummy_x)

        test_ckpt_path = "checkpoints/test_temp.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "model_name": "MS_CNN_BiGRU"}, test_ckpt_path)

        loaded_model = get_model("MS_CNN_BiGRU", num_classes=5)
        ckpt = torch.load(test_ckpt_path, map_location="cpu")
        loaded_model.load_state_dict(ckpt["model_state_dict"])
        loaded_model.eval()

        with torch.no_grad():
            loaded_out = loaded_model(dummy_x)

        if os.path.exists(test_ckpt_path):
            os.remove(test_ckpt_path)

        self.assertTrue(torch.allclose(orig_out, loaded_out, atol=1e-6), "Loaded model outputs do not match original.")


if __name__ == "__main__":
    unittest.main()
