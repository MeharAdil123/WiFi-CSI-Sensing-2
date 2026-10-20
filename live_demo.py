"""
Live Streaming CSI Inference & Activity Monitoring Engine.

Run modes:
1. Playback mode (simulates real-time stream from a recorded CSV file at 100 Hz):
   python live_demo.py --mode playback --file recordings/Falling_01.csv

2. Live hardware mode (connects to physical ESP32-S3 over USB serial):
   python live_demo.py --mode live --port COM8 --baud 921600

Supports any trained checkpoint (CNN_BiLSTM, MS_CNN_BiGRU, SE_TCN).
"""

import os
import sys
import time
import argparse
import csv
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.models import get_model
from src.live_inference import LiveCSIInferenceEngine, CLASS_NAMES


def print_banner():
    print("=" * 75)
    print("  ESP32-S3 WiFi-CSI HUMAN ACTIVITY RECOGNITION & FALL DETECTION ENGINE")
    print("  RNS Institute of Technology - Department of ECE (Academic Year 2026-27)")
    print("=" * 75)


def draw_bar(prob: float, length: int = 20) -> str:
    """Renders an ASCII probability bar."""
    filled = int(round(prob * length))
    return f"[{'#' * filled}{'-' * (length - filled)}] {prob * 100:5.1f}%"


def display_dashboard(pred_dict: dict, packet_count: int, elapsed_time: float):
    """Prints a real-time terminal status dashboard."""
    prediction = pred_dict["prediction"]
    smoothed = pred_dict["smoothed_prediction"]
    confidence = pred_dict["confidence"]
    fall_alert = pred_dict["fall_alert"]
    probs = pred_dict["probabilities"]

    # Clear terminal line or display block
    print("\n" + "-" * 65)
    print(f" Elapsed: {elapsed_time:5.1f}s | Packets Processed: {packet_count:5d} | Sampling Rate: ~100 Hz")
    print("-" * 65)

    for c in CLASS_NAMES:
        p = probs.get(c, 0.0)
        marker = " <-- CURRENT" if c == prediction else ""
        print(f" {c:<10s}: {draw_bar(p, length=22)}{marker}")

    print("-" * 65)
    print(f" Instantaneous Detection: {prediction:<10s} (Confidence: {confidence*100:.1f}%)")
    print(f" Smoothed Prediction:     {smoothed:<10s} (3-Window Majority Vote)")

    if fall_alert:
        print("\n" + "!" * 65)
        print("  EMERGENCY ALERT: FALL DETECTED BEHIND PARTITION WALL!")
        print("  TRIGGERING ELDERCARE DISPATCH NOTIFICATION PROTOCOL")
        print("!" * 65 + "\a")
    print("-" * 65)


def run_playback(csv_file: str, engine: LiveCSIInferenceEngine, stride: int = 50, speed_multiplier: float = 1.0):
    """Simulates real-time packet arrival from recorded CSV."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Recording CSV not found: {csv_file}")

    print(f"\n[STREAM] Streaming playback from: {csv_file}")
    print(f"[STREAM] Model: {engine.model_name} on device: {engine.device}")
    print(f"[STREAM] Window length: {engine.window_size} packets (~2.5s) | Stride: {stride} packets (~0.5s)")
    print("[STREAM] Starting packet ingestion...\n")

    packet_counter = 0
    start_time = time.time()
    packet_period = 0.01 / speed_multiplier  # 10 ms per packet at 100 Hz

    with open(csv_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if first_row is None:
            print("Error: CSV is empty.")
            return

        # Determine CSV format
        if first_row[0].strip() == "CSI_DATA":
            rows = [first_row, *reader]
            csi_extractor = lambda row: ast_csi(row[24]) if len(row) == 25 and row[0].strip() == "CSI_DATA" else None
        elif len(first_row) >= 3 and first_row[0].strip().lower() == "activity":
            rows = list(reader)
            csi_extractor = lambda row: parse_format2_line(row[2]) if len(row) >= 3 else None
        else:
            print("Unrecognized CSV format.")
            return

        for row in rows:
            raw_csi = csi_extractor(row)
            if raw_csi is not None:
                valid = engine.push_packet(raw_csi)
                if valid:
                    packet_counter += 1

                # Every stride packets, run inference
                if engine.is_ready() and (engine.packets_since_last_inference >= stride):
                    result = engine.predict()
                    elapsed = time.time() - start_time
                    display_dashboard(result, packet_counter, elapsed)

                time.sleep(packet_period)

    print(f"\nPlayback complete. Processed {packet_counter} valid CSI packets.")


def ast_csi(raw_str: str):
    import ast
    try:
        return ast.literal_eval(raw_str.strip())
    except:
        return None


def parse_format2_line(line: str):
    import ast
    try:
        fields = next(csv.reader([line]))
        if len(fields) == 25 and fields[0].strip() == "CSI_DATA":
            return ast.literal_eval(fields[24].strip())
    except:
        pass
    return None


def run_live(port: str, baud: int, engine: LiveCSIInferenceEngine, stride: int = 50):
    """Streams live CSI frames from ESP32-S3 over serial COM port."""
    try:
        import serial
    except ImportError:
        print("Error: 'pyserial' package required for live serial collection.")
        print("Install via: pip install pyserial")
        return

    print(f"\n[SERIAL] Connecting to ESP32-S3 on {port} @ {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
    except Exception as exc:
        print(f"Error opening serial port {port}: {exc}")
        return

    print("[SERIAL] Connected. Listening for CSI_DATA broadcast frames...")
    packet_counter = 0
    start_time = time.time()

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if line.startswith("CSI_DATA"):
                raw_csi = ast_csi(line.split(",")[-1])
                if raw_csi is not None:
                    valid = engine.push_packet(raw_csi)
                    if valid:
                        packet_counter += 1

                    if engine.is_ready() and (engine.packets_since_last_inference >= stride):
                        result = engine.predict()
                        elapsed = time.time() - start_time
                        display_dashboard(result, packet_counter, elapsed)
    except KeyboardInterrupt:
        print("\nLive capture stopped by user.")
    finally:
        ser.close()


def main():
    parser = argparse.ArgumentParser(description="WiFi-CSI Live Inference & Activity Recognition")
    parser.add_argument("--mode", choices=["playback", "live"], default="playback", help="Execution mode")
    parser.add_argument("--file", type=str, default="recordings/Falling_01.csv", help="CSV recording path for playback")
    parser.add_argument("--port", type=str, default="COM8", help="Serial COM port for ESP32 receiver")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/experiments/grouped/grouped_SE_TCN_seed42.pth", help="Model checkpoint path")
    parser.add_argument("--speed", type=float, default=2.0, help="Playback speed multiplier")
    args = parser.parse_args()

    print_banner()

    # Fallback checkpoint if default does not exist
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        fallback = "checkpoints/cnn_bilstm_seed_456.pth"
        if os.path.exists(fallback):
            print(f"Note: '{ckpt_path}' not found. Using fallback '{fallback}'.")
            ckpt_path = fallback
        else:
            raise FileNotFoundError(f"No valid checkpoint found at {ckpt_path}")

    engine = LiveCSIInferenceEngine(checkpoint_path=ckpt_path)

    if args.mode == "playback":
        run_playback(args.file, engine, stride=50, speed_multiplier=args.speed)
    elif args.mode == "live":
        run_live(args.port, args.baud, engine, stride=50)


if __name__ == "__main__":
    main()
