# Privacy-Preserving Through-Wall Human Activity Recognition Using WiFi-CSI

**Department of Electronics and Communication Engineering**  
**RNS Institute of Technology, Bengaluru – 560 098**  
**Academic Year:** 2026–2027  

**Project Team (Group 10):**
- Md Mehar Adil D H (USN: `1RN23EC094`)
- Mohith Gowda M (USN: `1RN23EC095`)
- Nihar Gowda (USN: `1RN23EC106`)
- Sujal L G (USN: `1RN23EC169`)

**Project Guide:** Dr. Leena Chandrashekar, Associate Professor, Dept. of ECE  
**Reviewer:** Dr. Srikanth G N, Associate Professor, Dept. of ECE  

---

## 1. Project Overview

This repository contains the complete experimental pipeline, model architectures, data audit, and reproducibility suite for **camera-free, privacy-preserving through-wall human activity recognition (HAR)** using commercial commodity Espressif **ESP32-S3** microcontrollers and Deep Learning.

### The Problem
Traditional indoor monitoring systems rely on optical cameras or wearable sensors:
- **Cameras** invade privacy (especially in bedrooms and bathrooms), require optical line-of-sight, and fail completely behind walls, partitions, smoke, or in darkness.
- **Wearables** require active charging, manual compliance, and are frequently forgotten or discarded by elderly users.

### The Solution: WiFi Channel State Information (CSI)
Wi-Fi signals penetrate drywall, wooden partitions, and standard brick walls. As a human performs an activity behind a wall, their moving body alters the multipath propagation paths. By analyzing fine-grained **Channel State Information (CSI)** extracted from everyday 802.11n Wi-Fi packets, our system recognizes activities and detects dangerous falls **contactlessly, unobtrusively, and with zero visual capture**.

---

## 2. Hardware and Radio Specifications

| Component | Specification | Function / Parameter |
| :--- | :--- | :--- |
| **Microcontrollers** | 2× Espressif ESP32-S3-N16R8 | Dual-core Xtensa LX7 @ 240 MHz, 16 MB Flash, 8 MB PSRAM. |
| **Radio Protocol** | ESP-NOW Broadcast | Connectionless action frames, no access point or router needed. |
| **RF Channel** | Channel 11 (2.462 GHz primary) | 40 MHz channel bandwidth (HT40, 802.11n HT mode). |
| **Packet Sampling Rate** | 100 packets/second (10 ms period) | Nyquist rate up to 50 Hz, capturing human motion dynamics (~16 Hz walking Doppler). |
| **TX Antenna** | 6 dBi Omnidirectional (2.4 GHz, 50 $\Omega$) | Broadly illuminates the monitored zone. |
| **RX Antenna** | 8 dBi ALFA Directional Patch (2.4–2.5 GHz) | Concentrates receiver sensitivity toward the sensing zone, rejecting background multipath. |
| **Raw CSI Payload** | 384 signed 8-bit integers per packet | 192 complex subcarrier entries ($I_k + jQ_k$) spanning L-LTF and HT-LTF training fields. |

---

## 3. Dataset & Data Audit Summary

Our system evaluates five discrete activity classes:
- **0: Empty** (static unoccupied room)
- **1: Sitting**
- **2: Standing**
- **3: Walking**
- **4: Falling** (safety-critical emergency detection)

### Key Audit Findings:
1. **Window Overlap Contamination (Legacy Random Split)**: The original 2,241 windows in `Data/MyCSI/` were sliced with 80% overlap (200 packets shared between adjacent windows) and split randomly at the window level. This allowed sister windows from the same continuous recording to appear in both training and test sets, inflating test accuracy to 96%–98% while masking true generalization.
2. **True Generalization Collapse**: Evaluating the 97.63% checkpoint on brand new unseen recordings (`processed_new_test_recordings/`) caused accuracy to drop to **20.55%** (random chance).
3. **The Leakage-Resistant Recording-Grouped Protocol**: To ensure genuine scientific validity, we quarantined entire CSV recordings into disjoint Train, Validation, and Test subsets before windowing (0% window overlap across sets).
4. **Falling Class Balancing**: Merged 25 new sessions from `live_recordings/` with the original 51 sessions, giving **76 total recording sessions** and increasing Falling representations from 43 to 150 windows.

---

## 4. Model Architectures

We provide full PyTorch implementations of all baselines and two proposed novel architectures:

### 1. Proposed Model 1: Multi-Scale CNN-BiGRU (`MS_CNN_BiGRU`) — *Recommended*
- **Parameters:** **188,673** (41% smaller than original CNN-BiLSTM).
- **Core Innovation:**
  - Parallel multi-scale 1D temporal filter banks ($k=3, 5, 7$) capture abrupt fall spikes (30 ms) and walking step cycles (70 ms) simultaneously.
  - Squeeze-and-Excitation (SE) channel attention adaptively suppresses low-SNR fading notch subcarriers.
  - Bidirectional GRU replaces BiLSTM, eliminating the separate cell state, saving 33% recurrent parameters, and accelerating inference on CPU.

### 2. Proposed Model 2: SE Dilated TCN (`SE_TCN`)
- **Parameters:** **110,165** (65.5% smaller than baseline).
- **Core Innovation:**
  - Fully non-recurrent Temporal Convolutional Network with exponential dilations ($d=1, 2, 4, 8$).
  - Full temporal parallelization on CPU/edge devices with constant inference latency.

### 3. Baselines:
- `MLP` (11.5M params)
- `LeNet` (163k params)
- `BiLSTM` (80k params)
- `CNN_GRU` (153k params)
- `CNN_BiLSTM` (Original Baseline: 319k params)

---

## 5. Comparative Results Summary

### Multi-Seed Performance on Legacy Split vs. Leakage-Resistant Grouped Split:

| Architecture | Parameters | Inference Latency (ms) | Legacy Random Split Acc (Mean ± SD) | Legacy Best Acc | Grouped Split Acc (Mean ± SD) | Grouped Balanced Acc | Grouped Falling Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BiLSTM** | 80,517 | 1.56 ms | 72.11% | 72.11% | 41.93% | 48.67% | 94.74% |
| **MLP** | 11,588,101 | 1.37 ms | 89.91% | 89.91% | 46.33% | 39.67% | 26.32% |
| **CNN_GRU** | 153,669 | 2.76 ms | 97.63% | 97.63% | 45.28% | 35.39% | 0.00% |
| **LeNet (2D CNN)** | 163,813 | 4.00 ms | 96.44% | 96.44% | 52.41% | 42.23% | 0.00% |
| **CNN_BiLSTM (Original)** | 319,429 | 2.39 ms | 94.36 ± 2.16% | 97.03% | 49.39 ± 3.89% | 46.26% | 45.26% |
| **MS_CNN_BiGRU (Proposed)** | **188,673** | 12.76 ms | **97.09 ± 0.95%** | **97.92%** | **50.02 ± 4.63%** | 45.28% | 29.47% |
| **SE_TCN (Proposed Champion)** | **110,165** | **2.75 ms** | **97.15 ± 0.61%** | **97.92%** | **58.57 ± 2.64%** | **55.46%** | **47.37%** |

*Note: Grouped Split results evaluate true generalization on completely unseen recording sessions.*

---

## 6. Project Structure

```
WiFi CSI Agent/
├── config/                     # Configuration definitions
├── docs/                       # Technical & scientific documentation
│   ├── complete_model_explanation.md   # Comprehensive ECE viva explanation & math guide
│   ├── data_audit.md                   # Complete data quality & leakage audit report
│   ├── experiment_protocol.md          # Scientific experimental protocol & standards
│   └── model_architecture.md           # Layer-by-layer specifications & tensor diagrams
├── experiments/
│   └── run_benchmarks.py       # Automated multi-seed benchmark & ablation suite
├── live_recordings/            # 25 CSV recordings collected on Oct 12, 2026
├── processed_new_test_recordings/ # Extracted windows for new recordings
├── processed_recordings/       # Extracted windows for original 51 recordings
├── recordings/                 # 51 original CSV recordings
├── research/
│   └── related_work.md         # Comprehensive literature matrix & survey synthesis
├── results/                    # Machine-readable benchmarks, confusion matrices, CSVs
│   ├── benchmark_results.json  # Full multi-seed metrics
│   ├── literature_comparison.md# Scientific literature comparison
│   └── model_comparison.csv    # Summary comparison table
├── src/                        # Modular production code
│   ├── dataset.py              # Preprocessing, windowing, and zero-leakage splits
│   ├── evaluate.py             # Metrics, confusion matrices, latency benchmark
│   ├── live_inference.py       # Real-time streaming FIFO buffer & fall alert engine
│   ├── models.py               # All baseline and proposed neural architectures
│   └── train.py                # Reproducible training with early stopping & class weights
├── tests/
│   └── test_pipeline.py        # Automated unit test suite
├── audit_existing_pipeline.py  # Repository audit script
├── evaluate_model.py           # Single-checkpoint evaluation script
└── test_recorded_csi.py        # Per-recording validation script
```

---

## 7. Quickstart & Reproduction Commands

### 1. Verify Environment and Run Unit Tests
```bash
python tests/test_pipeline.py
```
*Expected Output: `Ran 5 tests in ~0.9s. OK.`*

### 2. Audit Existing Checkpoints on Legacy Split
```bash
python audit_existing_pipeline.py
```
*Evaluates all 5 saved checkpoints (`cnn_bilstm_seed_*.pth`) and confirms the exact source of reported accuracy numbers.*

### 3. Run Complete Multi-Seed Benchmarks & Ablations
```bash
python experiments/run_benchmarks.py
```
*Executes Track B.1 (Legacy Random Split), Track B.2 (Leakage-Resistant Grouped Split), Loss Ablations, and Normalization Ablations across 5 random seeds, saving full metrics to `results/benchmark_results.json` and `results/model_comparison.csv`.*

### 4. Evaluate Real-Time Streaming Inference Engine
```bash
python -c "from src.live_inference import LiveCSIInferenceEngine; engine = LiveCSIInferenceEngine('checkpoints/cnn_bilstm_seed_456.pth'); print('Inference engine ready. Window length:', engine.window_size)"
```

---

## 8. Primary References

1. **J. Strohmayer and M. Kampel**, "WiFi CSI-based long-range through-wall human activity recognition with the ESP32," *Proc. International Conference on Computer Vision Systems (ICVS)*, Springer LNCS, 2023, pp. 41–50. [DOI: 10.1007/978-3-031-44137-0_4](https://doi.org/10.1007/978-3-031-44137-0_4)
2. **J. Yang et al.**, "SenseFi: A library and benchmark on deep-learning-empowered WiFi human sensing," *Patterns*, vol. 4, no. 3, 2023. [DOI: 10.1016/j.patter.2023.100694](https://doi.org/10.1016/j.patter.2023.100694)
3. **C. Chen et al.**, "CSI-Bench: A Multi-Environment Benchmark for Cross-Domain WiFi Sensing," *IEEE Transactions on Mobile Computing*, 2025.
4. **W. Wang et al.**, "Device-free human activity recognition using commercial WiFi devices (CARM)," *IEEE Journal on Selected Areas in Communications*, vol. 35, no. 5, pp. 1118–1131, 2017.
5. **C. Han et al.**, "WiFall: Device-free fall detection by wireless networks," *Proc. IEEE INFOCOM*, 2014, pp. 271–279.
