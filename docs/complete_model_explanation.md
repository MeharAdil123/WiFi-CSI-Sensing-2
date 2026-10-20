# Complete Technical Explanation: WiFi-CSI Human Activity Recognition

**Authors:** Md Mehar Adil D H, Mohith Gowda M, Nihar Gowda, Sujal L G  
**Institution:** Department of Electronics and Communication Engineering, RNS Institute of Technology  
**Academic Year:** 2026–2027  
**Project Guide:** Dr. Leena Chandrashekar, Associate Professor  
**Reviewer:** Dr. Srikanth G N, Associate Professor  

---

## Table of Contents
1. [How CSI Measurements Represent the Wireless Channel](#1-how-csi-measurements-represent-the-wireless-channel)
2. [How Raw CSI Packets Become Model Input Tensors](#2-how-raw-csi-packets-become-model-input-tensors)
3. [Physical Justification for Each Preprocessing Step](#3-physical-justification-for-each-preprocessing-step)
4. [Detailed Layer-by-Layer Architecture Walkthrough](#4-detailed-layer-by-layer-architecture-walkthrough)
5. [The Mechanics of Hyperparameters: Kernels, Pooling, Hidden Dimensions, and Bidirectionality](#5-the-mechanics-of-hyperparameters)
6. [Mathematical Mechanics of Logits and Cross-Entropy Loss](#6-mathematical-mechanics-of-logits-and-cross-entropy-loss)
7. [How Adam Updates Model Parameters](#7-how-adam-updates-model-parameters)
8. [Backpropagation, Regularization, and Optimization Safeguards](#8-backpropagation-regularization-and-optimization-safeguards)
9. [Differences Between Training, Validation, and Testing](#9-differences-between-training-validation-and-testing)
10. [How to Read and Defend Every Evaluation Metric and Confusion Matrix](#10-how-to-read-and-defend-every-evaluation-metric-and-confusion-matrix)
11. [What Changed Relative to Each Baseline Architecture](#11-what-changed-relative-to-each-baseline-architecture)
12. [Experimentally Verified Claims vs. Remaining Hypotheses](#12-experimentally-verified-claims-vs-remaining-hypotheses)
13. [Step-by-Step Reproduction Guide](#13-step-by-step-reproduction-guide)
14. [Real-World Limitations, Eldercare Applications, and Future Directions](#14-real-world-limitations-eldercare-applications-and-future-directions)

---

## 1. How CSI Measurements Represent the Wireless Channel

### The Multipath Channel Model
In an indoor environment, radio frequency (RF) waves transmitted at 2.462 GHz (Channel 11) propagate along multiple distinct paths before arriving at the receiver antenna:
1. **Direct Path (Line-of-Sight or through partition):** Passes directly from the transmitter (TX) to the receiver (RX).
2. **Static Reflected Paths:** Bounce off stationary drywall, concrete floors, metallic window frames, and desks.
3. **Dynamic Reflected Paths:** Scatter and bounce off the human body.

The composite wireless channel frequency response on subcarrier $k$ at time $t$ is expressed as the superposition of $L$ multipath components:
$$H_k(t) = \sum_{i=1}^{L} a_i(t) \, e^{-j \left(2\pi f_k \tau_i(t) + \phi_i(t)\right)}$$
where:
- $a_i(t)$ is the attenuation of the $i$-th propagation path.
- $\tau_i(t) = d_i(t) / c$ is the propagation delay along path $i$ of length $d_i(t)$ ($c \approx 3 \times 10^8 \text{ m/s}$).
- $f_k$ is the center frequency of subcarrier $k$.
- $\phi_i(t)$ represents phase shifts introduced by material reflections.

We decompose this channel into a **static channel component** $H_k^s$ (from the room and walls) and a **dynamic human-induced component** $H_k^d(t)$:
$$H_k(t) = H_k^s + H_k^d(t)$$

### Why Even Millimeter Motions Change CSI
At $f_c = 2.462\text{ GHz}$, the RF wavelength is:
$$\lambda = \frac{c}{f_c} = \frac{3 \times 10^8 \text{ m/s}}{2.462 \times 10^9 \text{ Hz}} \approx 0.1218\text{ m} \approx 12.2\text{ cm}$$
A displacement of the human body by just $\Delta d = \lambda / 4 \approx 3\text{ cm}$ changes the round-trip reflected path length by $\Delta d_{\text{path}} \approx 2 \Delta d \approx 6\text{ cm} = \lambda / 2$. This shifts the phase of the reflected signal by:
$$\Delta \theta = 2\pi \frac{\Delta d_{\text{path}}}{\lambda} = \pi \text{ radians} (180^\circ)$$
A $180^\circ$ phase shift flips constructive interference into destructive interference. The measured amplitude $|H_k(t)|$ oscillates between:
$$|H_k^s| - |H_k^d| \quad \Longleftrightarrow \quad |H_k^s| + |H_k^d|$$
This physical principle explains why gross motor actions (walking, falling) produce large cyclic or transient swings across multiple subcarriers, while subtle actions (sitting vs. standing) shift the static shadowing profile of the room.

### Why CSI Outperforms RSSI
- **Received Signal Strength Indicator (RSSI)** is a single scalar representing total received RF power averaged across the entire 40 MHz channel and all paths. Destructive interference on one frequency cancels constructive interference on another, hiding fine movement.
- **CSI (Channel State Information)** reports complex transmission coefficients $(I_k, Q_k)$ individually across up to 128 subcarriers. Because each subcarrier operates at a slightly different frequency $f_k = f_0 + k \Delta f$ ($\Delta f = 312.5\text{ kHz}$), different subcarriers experience frequency-selective fading. This provides a rich 90-channel spatial-frequency fingerprint.

---

## 2. How Raw CSI Packets Become Model Input Tensors

Every ESP-NOW broadcast frame received by the ESP32-S3 station produces a raw CSV record containing 25 fields. The final field is a serialized array of 384 signed 8-bit integers:

```
[I_0, Q_0, I_1, Q_1, I_2, Q_2, ..., I_191, Q_191]
```

### Numerical Pipeline: Step by Step

```mermaid
flowchart LR
    A["Raw Serial Line (384 int8)"] --> B["Pair into 192 (I, Q) Values"]
    B --> C["Magnitude |H_k| = sqrt(I^2 + Q^2)"]
    C --> D["Drop Zero / Null Subcarriers"]
    D --> E["Linear Interpolation -> 90 Features"]
    E --> F["Train-Fitted Min-Max Scaling [0, 1]"]
    F --> G["Temporal Windowing: 250 x 90"]
```

#### Step 1: Complex Number Assembly
For each subcarrier entry $k \in [0, 191]$:
$$H_k = I_k + j Q_k, \qquad \text{where } I_k = \text{CSI}[2k], \quad Q_k = \text{CSI}[2k+1]$$
Example: If raw bytes are $[3, 4]$, then $I = 3$, $Q = 4$.

#### Step 2: Amplitude Extraction
$$|H_k| = \sqrt{I_k^2 + Q_k^2} = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = 5.0$$
*Why not phase?* The receiver oscillator and transmitter oscillator are unsynchronized, introducing random Carrier Frequency Offset (CFO) and Sampling Frequency Offset (SFO):
$$\hat{\phi}_k(t) = \phi_k(t) - 2\pi \frac{k}{N} \delta(t) + \beta(t) + Z_k(t)$$
Without multi-antenna conjugate multiplication or complex linear detrending, raw phase jumps randomly between packets. In contrast, amplitude is invariant to oscillator phase offsets ($|e^{j\beta}| = 1$).

#### Step 3: Null-Subcarrier Elimination
In 802.11n HT40 (40 MHz bandwidth), guard bands at the band edges and the central Direct Current (DC) subcarriers carry no energy. Their reported CSI values are zero ($I=0, Q=0 \implies |H|=0$). We drop all zero values, leaving $M$ active subcarrier amplitudes ($M \approx 108$ to $114$).

#### Step 4: Resampling to Exactly 90 Features
Because small wireless glitches can alter the count of valid subcarriers, we map the remaining $M$ active subcarrier values onto a standardized 90-point grid via piecewise linear interpolation:
$$u_i = \frac{i}{M - 1} \in [0, 1] \quad \xrightarrow{\text{Interpolate}} \quad v_j = \frac{j}{89} \in [0, 1] \quad (j = 0, 1, \dots, 89)$$
$$x'_j = x_i + (x_{i+1} - x_i) \frac{v_j - u_i}{u_{i+1} - u_i}$$
This guarantees every packet vector has dimension exactly $1 \times 90$, matching the public UT-HAR SenseFi benchmark standard.

#### Step 5: Normalization
Using the minimum ($x_{\min}^{\text{train}} = 1.0$) and maximum ($x_{\max}^{\text{train}} = 25.88$) computed **strictly on the training recordings**:
$$\hat{x} = \frac{x - x_{\min}^{\text{train}}}{x_{\max}^{\text{train}} - x_{\min}^{\text{train}}}$$
Values are bounded in $[0, 1]$, preventing gradient explosion in early training layers.

#### Step 6: Temporal Windowing
Consecutive normalized packet frames are grouped into overlapping windows:
- Window length: $W = 250$ packets ($2.5\text{ s}$ @ 100 packets/s).
- Stride: $S = 50$ packets ($0.5\text{ s}$ advance between consecutive samples).
- Sample matrix shape: $250 \times 90$ ($250 \text{ time steps} \times 90 \text{ subcarriers}$).

---

## 3. Physical Justification for Each Preprocessing Step

| Preprocessing Step | Signal Processing Justification | What Happens If Omitted |
| :--- | :--- | :--- |
| **I/Q to Amplitude** | Decouples multipath signal strength from unsynchronized oscillator phase drift ($\beta, \delta$). | Raw phase enters model as uniform random noise, destroying convergence. |
| **Null Subcarrier Dropping** | Edge guard bands and DC carriers contain hardware artifacts and zero power. | Artificial zeros create spurious edges that mislead convolutional filters. |
| **Linear Resampling to 90** | Smoothly preserves frequency-selective channel profile while standardizing input dimensions. | Variable-length tensors crash tensor batching; zero-padding injects artificial discontinuities. |
| **Train-Fitted Normalization** | Ensures all features lie in the optimal numerical range for ReLU and Batch Normalization. | Test-set normalization leaks future data; unnormalized inputs saturate activations. |
| **80% Overlapping Windows** | Prevents activities (such as a 1.5-second fall) from being truncated at window boundaries. | Non-overlapping windows divide falls across boundaries, cutting fall detection rate by ~50%. |

---

## 4. Detailed Layer-by-Layer Architecture Walkthrough

We walk through our proposed **Multi-Scale CNN-BiGRU (`MS_CNN_BiGRU`)** with a concrete batch size $B=16$.

```
Input Window: [16, 1, 250, 90]
  │
  ▼ Squeeze + Permute
Tensor: [16, 90, 250]   <-- 90 CSI subcarriers (channels), 250 packets (time)
  │
  ▼ Subcarrier Squeeze-and-Excitation (SE) Attention
Tensor: [16, 90, 250]   <-- Fading notches attenuated, high-SNR subcarriers amplified
  │
  ▼ Multi-Scale Conv1D Stage 1 (k=3, k=5, k=7 in parallel)
Tensor: [16, 64, 250]   <-- Captures micro-transients and step cycles
  │
  ▼ Multi-Scale Conv1D Stage 2 (k=3, k=5, k=7 in parallel)
Tensor: [16, 96, 250]   <-- Hierarchical spatio-temporal representations
  │
  ▼ MaxPool1D (kernel_size=2, stride=2)
Tensor: [16, 96, 125]   <-- Time resolution halved; receptive field expanded
  │
  ▼ Permute for Recurrent Unit
Tensor: [16, 125, 96]   <-- Sequence length=125, Feature dimension=96
  │
  ▼ Bidirectional GRU (hidden_size=96)
Forward  h_T: [16, 96]  <-- Context summarized from packet 0 -> 125
Backward h_0: [16, 96]  <-- Context summarized from packet 125 -> 0
  │
  ▼ Concatenate Hidden States
Tensor: [16, 192]       <-- Complete bidirectional temporal context
  │
  ▼ Dense Classifier (Linear 192 -> 64 -> 5 with Dropout 0.15)
Output Logits: [16, 5]  <-- Raw scores for [Empty, Sitting, Standing, Walking, Falling]
```

---

## 5. The Mechanics of Hyperparameters

### A. Kernel Size ($k=3, 5, 7$)
A 1D convolution slides a filter of length $k$ across time.
- At 100 packets/second, $k=3$ spans 30 ms (capturing sudden accelerations and high-frequency Doppler).
- $k=5$ spans 50 ms.
- $k=7$ spans 70 ms (capturing limb swings).
Using multi-scale kernels in parallel allows the network to simultaneously detect abrupt fall impacts and rhythmic walking strides without forcing a compromise.

### B. Padding ($p = (k - 1) / 2$)
We use `same` padding ($p=1$ for $k=3$, $p=2$ for $k=5$, $p=3$ for $k=7$).
$$\text{Output Length} = L_{\text{in}} + 2p - k + 1 = 250 + 2(2) - 5 + 1 = 250$$
Padding ensures temporal length is preserved across convolutions, preventing boundary truncation.

### C. Max-Pooling (`kernel_size=2, stride=2`)
Downsamples the temporal sequence from 250 to 125 steps:
1. Retains the strongest activation within each 20 ms window.
2. Introduces temporal shift invariance (an activity starting 10 ms earlier produces the same pooled representation).
3. Halves the sequence length fed to the recurrent unit, cutting recurrent matrix multiplications by **50%**.

### D. Bidirectionality (BiGRU vs. Unidirectional GRU)
- A **unidirectional GRU** reads packets sequentially: $t = 0 \to 125$. The final state $h_{125}$ reflects what happened, but the early packets had no benefit of knowing how the action ended.
- A **bidirectional GRU** runs two independent recurrent units:
  $$\vec{h}_t = \text{GRU}_{\text{fwd}}(x_t, \vec{h}_{t-1}), \qquad \overleftarrow{h}_t = \text{GRU}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1})$$
  In our eldercare scenario, an emergency fall consists of rapid downward motion followed by prolonged stillness on the floor. The backward GRU reading from stillness back to the impact reinforces that the preceding motion was indeed a fall rather than sitting down.
  We concatenate the final forward state $\vec{h}_{125}$ and final backward state $\overleftarrow{h}_1$ into a 192-dimensional vector.

---

## 6. Mathematical Mechanics of Logits and Cross-Entropy Loss

### Logits to Probabilities: The Softmax Function
The final linear layer outputs unconstrained real-valued scores called **logits** $\mathbf{z} = [z_0, z_1, z_2, z_3, z_4]$ for the 5 classes.
The Softmax function converts logits into normalized posterior probabilities:
$$P(Y = c \mid \mathbf{x}) = p_c = \frac{e^{z_c}}{\sum_{j=0}^{4} e^{z_j}}$$
*Example Calculation:*
Suppose for a true "Falling" sample ($y=4$), the model outputs logits:
$$\mathbf{z} = [-1.2, -0.5, -0.8, 1.1, 3.4]$$
Computing exponentials:
$$e^{-1.2} = 0.301, \quad e^{-0.5} = 0.607, \quad e^{-0.8} = 0.449, \quad e^{1.1} = 3.004, \quad e^{3.4} = 29.964$$
$$\sum_{j=0}^{4} e^{z_j} = 0.301 + 0.607 + 0.449 + 3.004 + 29.964 = 34.325$$
Probabilities:
$$p_0 = 0.009, \quad p_1 = 0.018, \quad p_2 = 0.013, \quad p_3 = 0.088, \quad p_4 = \frac{29.964}{34.325} = \mathbf{0.873} \quad (87.3\%)$$

### Cross-Entropy Loss
The Cross-Entropy loss for true class label $y \in \{0, 1, 2, 3, 4\}$ is the negative log-likelihood:
$$\mathcal{L} = -\log(p_y)$$
For $p_4 = 0.873$:
$$\mathcal{L} = -\ln(0.873) = 0.136$$
If the model had mistakenly predicted $p_4 = 0.05$ (missed fall):
$$\mathcal{L} = -\ln(0.05) = 2.996$$
The loss penalizes confident wrong predictions with severe exponential gradient pressure.

### Class-Weighted Cross-Entropy
Because Falling has fewer samples, we apply inverse class weights $w_c$:
$$\mathcal{L}_{\text{weighted}} = -w_y \log(p_y), \qquad w_c = \frac{N_{\text{total}}}{C \cdot N_c}$$
For Falling with weight $w_4 \approx 3.78$, a missed fall produces $3.78\times$ larger gradients, compelling the optimizer to adjust weights to avoid missing future falls.

---

## 7. How Adam Updates Model Parameters

Standard Stochastic Gradient Descent (SGD) applies a uniform learning rate $\alpha$ to all weights:
$$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$$
In a hybrid model containing Conv1D, BatchNorm, and BiGRU layers, gradients for convolutional filters differ in scale by orders of magnitude from recurrent transition matrices. SGD either diverges in the CNN or stalls in the GRU.

**Adam (Adaptive Moment Estimation)** maintains running exponential averages of the gradient ($m_t$, first moment / momentum) and squared gradient ($v_t$, second moment / uncentered variance):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\beta_1 = 0.9)$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\beta_2 = 0.999)$$

### Bias Correction
At $t=1$, because $m_0 = 0$ and $v_0 = 0$, $m_1 = 0.1 g_1$ is biased toward zero. Adam corrects for this initialization bias:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

### The Parameter Update
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
where $\alpha = 5 \times 10^{-4}$ and $\epsilon = 10^{-8}$.
- **Weights with frequent, large gradients** have large $v_t$, reducing their effective step size and preventing divergence.
- **Weights with rare gradients** (such as weights responding to the rare Falling class) have small $v_t$, boosting their effective step size so they learn rapidly.

---

## 8. Backpropagation, Regularization, and Optimization Safeguards

### Backpropagation Through Time (BPTT)
Gradients of the loss $\mathcal{L}$ with respect to recurrent weights $W_h$ flow backward across unrolled time steps:
$$\frac{\partial \mathcal{L}}{\partial W_h} = \sum_{t=1}^{125} \frac{\partial \mathcal{L}}{\partial h_t} \frac{\partial h_t}{\partial W_h}$$
In a plain RNN, repeated multiplication by the weight matrix causes $\prod \frac{\partial h_k}{\partial h_{k-1}}$ to vanish or explode exponentially. GRU's additive linear update gate ($h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$) provides an unobstructed highway for gradient flow across all 125 steps.

### Regularization Suite
1. **Dropout ($p=0.15$):** Randomly zeros 15% of activations during each forward training pass, scaling the remainder by $1 / (1 - 0.15) \approx 1.176$. This forces neurons to learn independent, redundant features rather than co-adapting to spurious noise.
2. **Weight Decay ($\lambda = 10^{-4}$):** Adds an $L_2$ penalty to the objective:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \frac{\lambda}{2} \|\theta\|_2^2$$
   This prevents individual weights from growing excessively large, smoothing the model's decision boundaries.
3. **Early Stopping:** Monitored on validation Macro-F1. When validation Macro-F1 does not achieve a new maximum for 6 consecutive epochs, training terminates and the exact model weights from the best epoch are restored.

---

## 9. Differences Between Training, Validation, and Testing

```
┌──────────────────────────────────────────────────────────────────┐
│                   TOTAL RECORDED DATASET (76 CSVs)               │
└──────────────────────────────────────────────────────────────────┘
         │                           │                      │
         ▼ 70%                       ▼ 15%                  ▼ 15%
┌─────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
│    TRAINING SET     │    │   VALIDATION SET   │    │      TEST SET      │
│ (Learns Parameters) │    │  (Model Selection) │    │  (Final Benchmark) │
│                     │    │                    │    │                    │
│ • Backward pass     │    │ • Forward pass only│    │ • Forward pass only│
│ • Adam weight update│    │ • Tunes epochs, lr │    │ • Evaluated ONCE   │
│ • Dropout active    │    │ • Early stopping   │    │ • Strictly held out│
│ • Computes min/max  │    │ • Reuses train norm│    │ • Reuses train norm│
└─────────────────────┘    └────────────────────┘    └────────────────────┘
```

---

## 10. How to Read and Defend Every Evaluation Metric and Confusion Matrix

### The Four Quadrants for Falling Detection
- **True Positive (TP):** Person fell, model sounded fall alert. (Ideal)
- **False Negative (FN):** Person fell, model predicted Sitting/Walking. (**Catastrophic failure in eldercare**).
- **False Positive (FP):** Person sat down fast, model sounded fall alert. (Nuisance alarm).
- **True Negative (TN):** Person walked, model predicted Walking.

### Mathematical Metric Definitions

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \quad (\text{When model says Fall, how often is it right?})$$
$$\text{Recall (Sensitivity)} = \frac{\text{TP}}{\text{TP} + \text{FN}} \quad (\text{Of all real falls, what fraction did we catch?})$$
$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Reading the Confusion Matrix
In our confusion matrix, **rows represent true classes** and **columns represent predicted classes**:
- Diagonal entries ($C_{ii}$) represent correct classifications.
- Off-diagonal entries ($C_{ij}, i \neq j$) represent errors.
- **Sitting $\leftrightarrow$ Standing Confusion:** In static postures, both people produce zero Doppler frequency ($f_D = 0$). The only difference is the physical height of the body blocking different subcarriers. Because static multipath varies with chair position, Sitting and Standing exhibit minor mutual confusion.

---

## 11. What Changed Relative to Each Baseline Architecture

| Architecture | Structural Differences | Why Our Proposed Model Improves |
| :--- | :--- | :--- |
| **vs. MLP** | MLP treats all 22,500 inputs independently with 11.5M parameters. | Our model preserves temporal order with Conv1D and GRU, using **98.4% fewer parameters** and generalizing far better. |
| **vs. LeNet (2D CNN)** | LeNet applies symmetric 2D square kernels ($5 \times 5$) across time and frequency. | Frequency and time have different physical units. We treat frequency as channels and convolve along time, matching RF Doppler physics. |
| **vs. Pure BiLSTM** | BiLSTM processes 250 raw uncompressed noisy time steps. | Our initial multi-scale CNN filters noise and compresses length from 250 to 125, allowing the recurrent unit to operate on clean features. |
| **vs. Original CNN-BiLSTM** | Baseline uses a single kernel size ($k=5$ in conv1, $k=3$ in conv2) and standard BiLSTM (319k params). | Our model adds **multi-scale receptive fields ($k=3, 5, 7$)**, **subcarrier SE attention**, and a **BiGRU**, cutting parameters by **41%** (188k) while improving minority recall. |

---

## 12. Experimentally Verified Claims vs. Remaining Hypotheses

### What Is Experimentally Verified
1. **The Input Layout Correction:** Permuting tensor from $[B, 250, 90]$ to $[B, 90, 250]$ moved accuracy from 91.10% to 97.63%.
2. **The 98.52% Origin:** Verified that Seed 2024 achieved 98.52% on the legacy split under train-range normalization.
3. **Data Leakage in Random Splits:** Verified that testing the 97.63% checkpoint on new recordings collapsed accuracy to 20.55%, proving that random window splits measure memorization rather than true cross-session generalization.
4. **Parameter Efficiency:** The proposed `MS_CNN_BiGRU` achieves equal or higher accuracy than the 319k baseline with only 188k parameters.

### What Remains a Research Hypothesis (Limitations)
1. **Cross-Room Transfer:** We have not demonstrated that a model trained in Room A will operate in Room B without fine-tuning (domain shift from differing room geometries remains an open research problem).
2. **Clinical Fall Safety:** 150 fall windows from 6 controlled recordings demonstrate engineering feasibility, but do not guarantee clinical reliability across diverse elderly body masses and real-world falls.
3. **Multi-Person Interference:** When two people move simultaneously in the sensing area, their Doppler signatures superimpose non-linearly. Multi-person separation requires multi-antenna angle-of-arrival (AoA) processing.

---

## 13. Step-by-Step Reproduction Guide

To reproduce every result from scratch in the repository:

```bash
# 1. Run unit test suite (verifies dimensions, forward passes, zero-leakage split)
python tests/test_pipeline.py

# 2. Evaluate original saved seed checkpoints on legacy split
python audit_existing_pipeline.py

# 3. Run the comprehensive benchmark suite (legacy split, grouped split, ablations)
python experiments/run_benchmarks.py

# 4. Run real-time streaming inference demonstration
python -c "from src.live_inference import LiveCSIInferenceEngine; engine = LiveCSIInferenceEngine('checkpoints/cnn_bilstm_seed_456.pth'); print('Engine initialized successfully')"
```

---

## 14. Real-World Limitations, Eldercare Applications, and Future Directions

### Eldercare Deployment Architecture
In an assisted living facility:
1. Two ESP32-S3 devices are mounted unobtrusively on opposite walls or across a partition.
2. The transmitter broadcasts periodic 100 Hz ESP-NOW beacons.
3. The receiver streams CSI packets over serial/Wi-Fi to a local low-power host (e.g., Raspberry Pi or mini PC).
4. The local host runs `LiveCSIInferenceEngine`. If a fall is detected ($P(\text{Fall}) > 0.50$) followed by prolonged inactivity, an encrypted MQTT alert is dispatched to caregiver smartphones.
5. **No cameras, no microphones, no wearables, and zero visual intrusion.**

### Key Future Research Directions
1. **Phase Sanitization:** Implementing subcarrier linear unwrapping to incorporate phase differences between packets.
2. **Multi-Receiver Fusion:** Combining CSI streams from two receivers for spatial diversity.
3. **Domain-Adversarial Neural Networks (DANN):** Aligning feature distributions across different rooms to achieve true plug-and-play cross-environment operation.
