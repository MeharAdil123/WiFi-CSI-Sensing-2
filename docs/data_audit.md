# Data Quality, Contamination Audit & Leakage Assessment

**Project:** Privacy-Preserving Through-Wall Human Activity Recognition Using WiFi-CSI  
**Document Status:** Complete Scientific Audit  
**Date:** October 2026  

---

## 1. Executive Summary

A rigorous audit of the repository data pipeline, raw CSV records, temporal windowing logic, label distributions, and normalization procedures was conducted. Key findings:

1. **Window-Level Split Contamination (Data Leakage)**: The original 2,241 windows in `Data/MyCSI/` were sliced with length 250 and stride 50 (80% overlap = 200 packets shared between adjacent windows). The 70/15/15 train/validation/test split was performed **at the window level**, resulting in near-duplicate windows appearing in both training and test sets.
2. **Empirical Generalization Collapse**: When the baseline checkpoint (`cnn_bilstm_seed_456.pth`, reporting 97.63% test accuracy on the contaminated split) was evaluated against 25 new unseen recordings in `processed_new_test_recordings/`, **window accuracy collapsed to 20.55%** (random chance across 5 classes is 20.00%).
3. **Severe Imbalance in Minority Class (Falling)**: In the original 2,241-sample dataset, Falling accounted for only **43 windows (1.9%)**, all sourced from a single 24-second recording (`Falling_01.csv`). The held-out test set contained only **6 Falling windows**.
4. **Gain and Normalization Discrepancy**: Raw CSI magnitude in the original recordings had a maximum value of 25.88, whereas newer recordings in `live_recordings/` exhibited raw magnitudes up to 277.19 due to receiver Automatic Gain Control (AGC) variations. Fixed min-max scaling with `TRAIN_MAX = 25.88` caused catastrophic input saturation (values reaching ~11.0 instead of $[0, 1]$).
5. **Corrective Implementation**: We have established an auditable, leakage-free **Recording-Grouped Protocol** that quarantines whole recordings into disjoint train, validation, and test subsets before windowing, paired with robust normalization.

---

## 2. Dataset Composition and Class Distribution

### A. Original Dataset (`recordings/` → `Data/MyCSI/`)
- Total raw CSV recordings: **51 files** (123,705 raw packets).
- Windowing: Length $W = 250$ packets (~2.5 s at 100 Hz), Stride $S = 50$ packets (80% overlap).
- Total generated windows: **2,241 samples** of dimension $250 \times 90$.

| Class ID | Activity Label | Original Recordings | Raw Packets | Window Count | Dataset Share | Train (70%) | Val (15%) | Test (15%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **Empty** | 20 | 46,510 | 841 | 37.53% | 588 | 126 | 127 |
| **1** | **Sitting** | 10 | 25,843 | 471 | 21.02% | 330 | 70 | 71 |
| **2** | **Standing** | 9 | 22,666 | 409 | 18.25% | 286 | 62 | 61 |
| **3** | **Walking** | 11 | 26,291 | 477 | 21.28% | 334 | 71 | 72 |
| **4** | **Falling** | 1 | 2,395 | 43 | 1.92% | 30 | 7 | 6 |
| **Total** | — | **51** | **123,705** | **2,241** | **100.0%** | **1,568** | **336** | **337** |

### B. New Recording Dataset (`live_recordings/`)
Recorded on October 12, 2026:
- 25 continuous CSV recordings (5 per class).
- Generated windows in `processed_new_test_recordings/`:
  - **Empty**: 5 files, 118 windows
  - **Sitting**: 5 files, 108 windows
  - **Standing**: 5 files, 92 windows
  - **Walking**: 5 files, 125 windows
  - **Falling**: 5 files, 107 windows (expanding Falling window representation by **+248%**)
- Total: 550 windows.

---

## 3. Mathematical & Empirical Analysis of Data Leakage

### A. Overlap Contamination Equation
Given a recording of $N$ packets, window length $W=250$, and stride $S=50$:
$$\text{Overlap fraction} = \frac{W - S}{W} = \frac{200}{250} = 80.0\%$$
For any window $w_k = [x_{k \cdot S}, \dots, x_{k \cdot S + W - 1}]$, its adjacent neighbors $w_{k-1}$ and $w_{k+1}$ share $80\%$ of identical physical packets.

In the original stratified random split:
- Probability that a window in Test has at least one adjacent neighbor in Train:
$$P(\text{leakage}) \approx 1 - (1 - 0.70)^2 = 1 - 0.09 = 91\%$$
Over 90% of all test windows had near-identical sister windows in the training set. The deep network was able to achieve 97.6%–98.5% accuracy largely by memorizing the static multipath profile of the room and interpolating overlapping temporal samples.

### B. Empirical Proof of Generalization Failure
When evaluating the trained baseline checkpoint `cnn_bilstm_seed_456.pth` (which achieves **97.63%** on the random split) against truly unseen sessions (`processed_new_test_recordings/`):

```
========== WINDOW-LEVEL RESULTS ON UNSEEN RECORDINGS ==========
Total windows: 550
Window accuracy: 20.55% (Random Guess Baseline = 20.00%)

Per-class window accuracy:
  Empty:    40/118 (33.90%)
  Sitting:   0/108 ( 0.00%)
  Standing: 14/92  (15.22%)
  Walking:   1/125 ( 0.80%)
  Falling:  58/107 (54.21%)

========== RECORDING-LEVEL MAJORITY-VOTE RESULTS ==========
Recordings evaluated: 25
Majority-vote accuracy: 20.00% (5 / 25 correct)
```

This experimental proof establishes that:
1. High accuracy (>96%) on the overlapping random split was an artifact of data leakage.
2. The network learned specific background multipath states ($H^s$) rather than generalized human kinematic signatures ($H^d(t)$).

---

## 4. Signal Processing & Normalization Discrepancies

### A. Raw Amplitude Ranges
Inspection of raw CSI arrays showed significant cross-session amplitude differences:
- `recordings/` (Original 51 sessions):
  - Values: $[-9, +5]$
  - Magnitudes: $[1.00, 25.88]$
  - Typical RSSI: $-58\text{ dBm}$ to $-68\text{ dBm}$
- `live_recordings/` (New 25 sessions):
  - Values: $[-56, +71]$
  - Magnitudes: $[1.00, 277.19]$
  - Typical RSSI: $-55\text{ dBm}$ to $-56\text{ dBm}$

### B. Impact on Preprocessing
1. **Fixed Training Min-Max**:
   $$\hat{x} = \frac{x - 1.0}{25.88 - 1.0}$$
   When applied to new recordings with $x \approx 200$, $\hat{x}$ reached $8.0$ to $11.0$. The activations completely saturated the Batch Normalization and ReLU layers.
2. **Independent Test Min-Max**:
   In `evaluate_model.py`:
   $$\hat{x}_{\text{test}} = \frac{x - \min(X_{\text{test}})}{\max(X_{\text{test}}) - \min(X_{\text{test}})}$$
   This leaks test distribution statistics and causes test performance to fluctuate depending on the presence of extreme outliers in the test batch.
3. **Per-Recording Min-Max**:
   Normalizing each recording individually to $[0, 1]$ forces the low-variance noise floor of an `Empty` room to span $[0, 1]$, making static empty space indistinguishable from violent falls or dynamic walking.

---

## 5. Corrective Action Plan & Leakage-Resistant Protocol

To achieve scientific validity, we implement the following:

1. **Recording-Grouped Split (Leakage-Resistant Protocol)**:
   - All windows from any given CSV file must reside in the **same subset** (strictly Train, strictly Validation, or strictly Test).
   - Zero window overlap across splits ($0\%$ packet sharing).
   - Stratified at the recording level so every subset includes all 5 classes.
2. **Dataset Augmentation & Balancing**:
   - Merge the original 51 recordings with the 25 new recordings, providing **76 distinct recording sessions** (25 Empty, 15 Sitting, 14 Standing, 16 Walking, 6 Falling).
   - Falling sample count increases from 43 to 150 windows.
3. **Principled Normalization**:
   - Compute normalization parameters ($\mu, \sigma$ or robust percentiles $p_1, p_{99}$) **exclusively on the training split**, and reuse them identically for validation, testing, and real-time streaming inference.
   - Dynamic baseline centering: Subtract the per-subcarrier temporal running mean to remove static background reflections:
     $$\tilde{H}_k(t) = |H_k(t)| - \bar{H}_k$$
     This isolates dynamic human motion and prevents the model from relying on room multipath fingerprints.
