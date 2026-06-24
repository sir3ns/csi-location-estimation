# Wi-Fi CSI Room Block Classifier — v4 (CNN-Only)
## Theory & Methodology Documentation

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Why CSI Localization Is Hard](#2-why-csi-localization-is-hard)
3. [Feature Engineering — IQ to Amplitude](#3-feature-engineering--iq-to-amplitude)
4. [Empty-Room Reference Subtraction](#4-empty-room-reference-subtraction)
5. [Normalization — StandardScaler](#5-normalization--standardscaler)
6. [Temporal Sliding Windows](#6-temporal-sliding-windows)
7. [Dataset Splitting — Temporal Per-Class Strategy](#7-dataset-splitting--temporal-per-class-strategy)
8. [Coordinate Labels — Topology-Aware Encoding](#8-coordinate-labels--topology-aware-encoding)
9. [The Core Idea: Window Reshape Strategy](#9-the-core-idea-window-reshape-strategy)
10. [Model Architecture](#10-model-architecture)
    - 10.1 [Stem Convolution](#101-stem-convolution)
    - 10.2 [Residual Blocks](#102-residual-blocks)
    - 10.3 [Squeeze-and-Excitation Channel Attention](#103-squeeze-and-excitation-channel-attention)
    - 10.4 [Global Average Pooling](#104-global-average-pooling)
    - 10.5 [Dual Output Heads](#105-dual-output-heads)
11. [Loss Functions](#11-loss-functions)
12. [Training Strategy](#12-training-strategy)
13. [Data Augmentation](#13-data-augmentation)
14. [Inference Pipeline](#14-inference-pipeline)
15. [v4 vs v3 — Architectural Comparison](#15-v4-vs-v3--architectural-comparison)
16. [Design Decision Summary](#16-design-decision-summary)

---

## 1. Problem Definition

A Wi-Fi receiver continuously collects **Channel State Information (CSI)** packets.
A person standing at different positions in a room causes different multipath
scattering patterns in the Wi-Fi signal, leaving detectable fingerprints in the CSI.
The goal is to classify which of the 9 spatial blocks of a 3×3 grid the person
occupies — or detect that the room is empty — using only the CSI signal.

**10 output classes:**

| Label | Meaning  | Grid Position |
|-------|----------|---------------|
| 0     | Empty    | No person     |
| 1     | Block 1  | Row 0, Col 0  |
| 2     | Block 2  | Row 0, Col 1  |
| 3     | Block 3  | Row 0, Col 2  |
| 4     | Block 4  | Row 1, Col 0  |
| 5     | Block 5  | Row 1, Col 1  |
| 6     | Block 6  | Row 1, Col 2  |
| 7     | Block 7  | Row 2, Col 0  |
| 8     | Block 8  | Row 2, Col 1  |
| 9     | Block 9  | Row 2, Col 2  |

**Grid layout:**

```
┌──────────┬──────────┬──────────┐
│ Block 1  │ Block 2  │ Block 3  │
│ (row 0)  │ (row 0)  │ (row 0)  │
├──────────┼──────────┼──────────┤
│ Block 4  │ Block 5  │ Block 6  │
│ (row 1)  │ (row 1)  │ (row 1)  │
├──────────┼──────────┼──────────┤
│ Block 7  │ Block 8  │ Block 9  │
│ (row 2)  │ (row 2)  │ (row 2)  │
└──────────┴──────────┴──────────┘
```

---

## 2. Why CSI Localization Is Hard

Wi-Fi uses **OFDM (Orthogonal Frequency Division Multiplexing)**, which splits the
signal across many subcarrier frequencies. CSI captures the complex channel gain —
amplitude and phase — at each subcarrier per packet. A person's body perturbs the
multipath propagation differently depending on their position, creating a
location-specific fingerprint.

Four compounding problems make this hard:

**Phase noise.** The carrier frequency offset and hardware clock drift between
transmitter and receiver cause random phase rotations that vary between packets
and sessions. Raw I/Q values are therefore unreproducible even at the same location.

**Static room dominance.** Walls, floor, ceiling, and furniture produce strong,
stable reflections that dwarf the human-induced perturbation. Raw CSI is mostly a
fingerprint of the room geometry, not of the person's position.

**Single-frame noise.** Any one CSI packet may be corrupted by momentary interference,
AGC fluctuation, or environmental change. A single measurement has poor
signal-to-noise ratio for localization.

**Adjacent block similarity.** Blocks that are physically close produce nearly
identical CSI signatures. A classifier that treats the 10 classes as arbitrary
unrelated categories has no guidance on this spatial structure.

Every design decision in v4 targets one or more of these four problems.

---

## 3. Feature Engineering — IQ to Amplitude

### Raw IQ representation

Each CSI frame is stored as 128 complex measurements, interleaved as 256 integers:

```
I1, Q1, I2, Q2, ..., I128, Q128
```

`I` (In-phase) and `Q` (Quadrature) are the real and imaginary components of the
complex channel response at each subcarrier.

### Why IQ is unsuitable as features

Raw I and Q values are corrupted by phase offset — a random rotation of every
complex value caused by hardware timing differences. Two measurements taken at
exactly the same location in the same room may have completely different I and Q
values if the phase offset changed. Training on raw IQ would train the model on
this noise component rather than the location-specific signal.

### Amplitude extraction

The amplitude (magnitude) of each subcarrier is computed:

```
amplitude_k = sqrt(I_k² + Q_k²)
```

Geometrically, this is the length of the complex vector in the IQ plane.
Rotating the vector — what phase noise does — does not change its length.
Amplitude is therefore **phase-invariant** and reflects only the physical
signal attenuation and reinforcement caused by the propagation environment.

This reduces each frame from 256 raw values to **128 amplitude values**,
one per subcarrier.

---

## 4. Empty-Room Reference Subtraction

### The problem: static background dominance

The raw CSI amplitude is dominated by the **static room signature** — the
stable multipath pattern created by walls, furniture, and the direct
line-of-sight path. This component is nearly identical regardless of whether
a person is present. The person's contribution is a small perturbation on
top of a large, unchanging background.

A model given raw amplitudes must learn to ignore this background through
gradient descent alone — spending capacity on suppressing irrelevant variation
instead of learning the localization signal.

### Background subtraction

The mean amplitude vector across all **empty-room (class 0)** samples is
computed per subcarrier:

```
empty_ref[k] = (1 / N_empty) × Σ amplitude[k]     over all class-0 frames
```

This 128-element vector is the best available estimate of the static room
signature. It is subtracted from every frame:

```
X_delta[k] = amplitude[k] - empty_ref[k]
```

`X_delta` now represents only the **human-induced deviation** at each subcarrier.
The static background cancels out algebraically, leaving the localization signal.

### Two-channel input

Rather than discarding the raw amplitude, both signals are preserved as
separate channels per frame:

| Channel | Content | What it provides |
|---------|---------|-----------------|
| 0 | Raw amplitude | Absolute signal level; subcarrier-level context |
| 1 | Delta (amp − reference) | Human-specific perturbation; static background removed |

Each frame becomes a **(128, 2)** tensor — 128 subcarrier positions, 2 channels.

### Why the reference must be saved

The same `empty_ref` vector used to create the training features **must** be
applied at inference. It is saved to `scaler/empty_reference.pkl`. Using a
different reference would shift channel 1 by an arbitrary offset that the
model has never encountered during training, producing wrong predictions.

---

## 5. Normalization — StandardScaler

After computing 2-channel features, different subcarriers still differ in
absolute scale. One subcarrier might operate in the range [5, 30] while
another operates in [200, 600]. Without normalization, gradient updates during
training would be dominated by high-amplitude subcarriers simply because their
numerical values are larger — not because they carry more localization
information.

**StandardScaler** transforms each feature dimension to zero mean and unit
variance:

```
X_scaled[k] = (X[k] - mean[k]) / std[k]
```

`mean[k]` and `std[k]` are computed only from the training set, then applied
to validation and test sets unchanged. This prevents **data leakage** — the
scaler must never see test statistics before evaluation.

After scaling, every subcarrier position competes on equal numerical footing.
The network learns which subcarriers are actually informative through gradient
descent.

The scaler is saved to `scaler/csi_scaler.pkl` for reuse at inference.

---

## 6. Temporal Sliding Windows

### Why single frames are insufficient

A single CSI frame is a noisy measurement. Any individual packet may be
disrupted by momentary multipath interference, AGC fluctuation, or
environmental change unrelated to the person's position. Classifying from a
single frame means the model has no mechanism to distinguish a real location
signature from a transient noise spike.

### Sliding window construction

A window of **30 consecutive frames** is used as one input sample. This
represents a short burst of measurements while the person remains stationary.

A sliding window with **stride 5** is applied over each class block
independently:

```
Window 1:  frames [0  .. 29]
Window 2:  frames [5  .. 34]
Window 3:  frames [10 .. 39]
...
Window W:  frames [start .. start+29]
```

With 690 frames per class:

```
Windows per class = (690 - 30) // 5 + 1 = 133
Total windows     = 133 × 10 classes = 1330
```

### Why windows must be constructed within each class

The dataset is ordered by class (rows 0–689 = class 0, rows 690–1379 = class 1,
and so on). If windowing were applied globally, windows near the class boundary
would contain frames from two different classes — producing a contradictory
training signal. Windows are therefore built **separately for each class**,
ensuring no window ever spans a class boundary.

### How the CNN uses the window

Unlike an LSTM, a CNN does not process frames one at a time. The 30-frame window
is fed to the CNN as a single structured input — all frames are visible
simultaneously. The model learns which patterns across the 30-frame window are
consistent indicators of each location. This achieves noise averaging (a
consistent pattern across 30 noisy frames is a reliable signal) without
requiring sequential processing.

---

## 7. Dataset Splitting — Temporal Per-Class Strategy

### Why random splitting causes leakage

With a window size of 30 and stride of 5, consecutive windows share 25 frames.
If window at position `t` is assigned to train and window at position `t+1`
is assigned to validation, 25 of their 30 frames are identical. The model
effectively sees the validation data during training, producing an optimistic
accuracy estimate that does not reflect real-world performance.

### Temporal split per class

For each class independently, windows are split by their temporal position —
not randomly:

```
Windows 0  .. N×0.70         → training set   (70%)
Windows N×0.70 .. N×0.85     → validation set (15%)
Windows N×0.85 .. N          → test set       (15%)
```

Because windows are ordered by start position, the training set contains only
the earliest windows, validation contains the middle, and test contains the
latest. No window from training overlaps with any window from validation or test.

### Post-split shuffle

After the temporal split, training windows from all 10 classes are **shuffled
together**. This prevents the optimizer from exploiting the class ordering
within batches. Validation and test sets are not shuffled — order has no
effect on evaluation metrics.

---

## 8. Coordinate Labels — Topology-Aware Encoding

### Limitation of pure classification

Standard multi-class classification with 10 arbitrary labels treats all errors
equally. Predicting Block 4 when the true label is Block 5 (adjacent) incurs
the same penalty as predicting Block 9 (far away). The model receives no
signal that adjacent blocks are spatially related and should have similar
feature representations.

### Normalised grid coordinates

Each block is assigned a normalised (row, col) coordinate pair in `[0, 1]²`:

```
Block 1 → (0.0, 0.0)    Block 2 → (0.0, 0.5)    Block 3 → (0.0, 1.0)
Block 4 → (0.5, 0.0)    Block 5 → (0.5, 0.5)    Block 6 → (0.5, 1.0)
Block 7 → (1.0, 0.0)    Block 8 → (1.0, 0.5)    Block 9 → (1.0, 1.0)
```

A second output head predicts these coordinates via regression. The shared
backbone — which feeds both classification and regression heads — must now
learn features that encode spatial proximity. Blocks 4 and 5 will share
gradient signal because they are close in coordinate space.

### Occupancy masking

Coordinate regression is meaningless for the empty-room class (label 0).
Including empty samples in the regression loss would train the model to predict
a position when no person is present, confusing the shared representation.

Each sample carries an **occupancy mask** (1 = person present, 0 = empty).
The regression loss is masked so only occupied samples contribute:

```
L_reg = Σ (MSE × mask) / Σ mask
```

### Loss weighting

```
total_loss = 1.0 × L_classification + 0.3 × L_regression
```

The weight 0.3 keeps coordinate regression as a supporting auxiliary task
without overshadowing the primary classification objective.

---

## 9. The Core Idea: Window Reshape Strategy

This is the defining architectural choice that makes v4 a pure CNN solution.

### The input window

After all preprocessing, each training sample is a window of shape:

```
(30, 128, 2)
  ↑    ↑   ↑
frames  subcarriers  channels
```

This cannot be fed directly to a 1D-CNN as a sequence — the CNN would slide
over the 30-frame axis and treat each frame as a single time step, losing the
frequency-domain structure across the 128 subcarriers.

### Transpose and reshape

The window is reorganised in two steps inside the model:

**Step 1 — Permute axes:**
```
(batch, 30, 128, 2)  →  (batch, 128, 30, 2)
```
The subcarrier axis moves to position 1, becoming the primary spatial axis
the CNN will slide over.

**Step 2 — Merge the last two axes:**
```
(batch, 128, 30, 2)  →  (batch, 128, 60)
  30 frames × 2 channels = 60 features per subcarrier position
```

### What the CNN sees after the reshape

The CNN receives a **(128, 60)** tensor:

- **Length 128** = the subcarrier frequency axis. The CNN slides its kernels
  along this axis, learning how groups of nearby subcarriers co-vary.
- **Depth 60** = 30 frames × 2 channels. At every subcarrier position, the
  model sees all 30 frames simultaneously — the raw amplitude and delta from
  every measurement in the window.

This is equivalent to saying: **"At each frequency bin, what did all 30
measurements look like, and how does that pattern relate to the patterns at
nearby frequency bins?"**

### Why this preserves the benefits of windowing without LSTM

The 30-frame window provides noise averaging — a consistent pattern across
30 noisy frames is a reliable signal. By merging the frame dimension into the
channel depth, all 30 frames are simultaneously available to every
convolutional kernel. The CNN does not need to roll through frames one at a
time (as an LSTM would) because it sees all 60 channels at every position in a
single forward pass.

Nearby subcarriers are correlated due to frequency-selective fading — the 1D
convolution across the 128-subcarrier axis captures these correlations exactly
as intended.

---

## 10. Model Architecture

```
Input: (batch, 30, 128, 2)
    │
    ├── Permute(2,1,3)          → (batch, 128, 30, 2)
    ├── Reshape(128, 60)        → (batch, 128, 60)
    │
    ├── Stem: Conv1D(64, k=7) + BN + ReLU + MaxPool(2)    → (64, 64)
    │
    ├── Stage 1: ResBlock(128, k=5) + SE + MaxPool(2)      → (32, 128)
    │
    ├── Stage 2: ResBlock(256, k=3) + SE + MaxPool(2)      → (16, 256)
    │
    ├── Stage 3: ResBlock(512, k=3) + SE                   → (16, 512)
    │
    ├── GlobalAveragePooling1D                              → (512,)
    │
    ├── Dense(256) → Dropout(0.4) → Dense(128)
    │
    ├─── Classification Head: Dense(10, softmax)
    └─── Regression Head:     Dense(64) → Dense(2, sigmoid)
```

### 10.1 Stem Convolution

```
Conv1D(64 filters, kernel_size=7, padding='same')
BatchNormalization
ReLU
MaxPooling1D(2)
```

The large kernel (7) in the stem captures **broad frequency-domain patterns**
spanning 7 adjacent subcarriers. These are the widest features — long-range
correlations in the subcarrier response that are characteristic of certain
propagation paths. The stem halves the sequence length from 128 to 64 via
MaxPooling, reducing computation for subsequent stages.

**Padding='same'** ensures the output length equals the input length before
pooling — no subcarrier information is discarded at the edges.

### 10.2 Residual Blocks

A residual (skip connection) block applies two convolutions and adds the
block input to the output:

```
output = F(x) + x
```

where `F(x)` is:
```
Conv1D(filters, kernel_size, padding='same')
BatchNormalization
ReLU
Conv1D(filters, kernel_size, padding='same')
BatchNormalization
```

**Why residual connections?**

In a deep network, backpropagation must propagate gradients from the output
all the way to the earliest layers. Each layer multiplication reduces the
gradient magnitude slightly. In a deep plain CNN, by the time the gradient
reaches the first layer it has shrunk to nearly zero — the early layers stop
learning. This is the **vanishing gradient problem**.

The skip connection adds an alternative path for the gradient:

```
∂L/∂x = ∂L/∂output × (∂F/∂x + I)
```

The identity matrix `I` ensures a gradient of at least 1.0 flows through
regardless of what `F(x)` contributes. Early layers always receive a usable
gradient signal.

**1×1 projection on the shortcut**

When the number of filters changes between the input and output of a residual
block (e.g., 64 → 128), the shortcut must be projected to match:

```
shortcut = Conv1D(filters, kernel_size=1)(shortcut)
shortcut = BatchNormalization()(shortcut)
```

A 1×1 convolution changes the channel depth without modifying the spatial
length, making the shapes compatible for the addition.

**Kernel sizes decrease across stages (7 → 5 → 3 → 3):**

- Stage 1 (k=5): medium-scale frequency correlations across 5 adjacent subcarriers
- Stage 2 (k=3): fine-grained local patterns within compressed representations
- Stage 3 (k=3): further refinement at the deepest abstraction level

As the sequence gets shorter and representations become more abstract, smaller
kernels are sufficient to capture meaningful patterns.

**Filter counts double across stages (64 → 128 → 256 → 512):**

Early stages learn simple patterns (a dip at a certain frequency range).
Later stages combine these into complex, location-specific signatures. Each
doubling allows the model to represent an exponentially richer vocabulary
of patterns, compensating for the reduced spatial resolution after pooling.

### 10.3 Squeeze-and-Excitation Channel Attention

After each residual block, an SE block recalibrates the **importance of each
feature channel** (filter output) based on its global content.

**Squeeze — global context per channel:**

```
z_c = GlobalAveragePooling1D of feature_map_c
    = (1/L) × Σ feature_map_c[i]   over all positions i
```

This produces a single scalar per channel that summarises what that filter
responded to across all 128 (then 64, 32, 16) subcarrier positions.

**Excitation — learn per-channel importance weights:**

```
s = sigmoid( W2 × relu( W1 × z ) )
```

Two fully connected layers form a small bottleneck network. The first layer
compresses from `C` channels to `C//8`, forcing the network to find a compact
summary of channel relationships. The second layer expands back to `C` channels
and applies sigmoid to produce weights in `(0, 1)`.

**Scale — apply the learned weights:**

```
output_c = s_c × feature_map_c
```

Channels that respond strongly to localization-relevant patterns get upweighted.
Channels that respond to noise or irrelevant environmental variation get
suppressed. The weights are different for each input — the SE block adapts
to each specific CSI measurement.

In the context of this problem, SE attention learns to identify **which
subcarrier ranges** are most informative for distinguishing the 10 classes,
and amplifies those while dampening others.

### 10.4 Global Average Pooling

After Stage 3, the feature map has shape `(16, 512)` — 16 positions along
the subcarrier axis, each described by a 512-dimensional feature vector.

GlobalAveragePooling1D takes the mean across the 16 positions:

```
output_c = (1/16) × Σ feature_map_c[i]   for i = 0..15
```

This produces a single **(512,)** vector that summarises the entire
frequency-domain response. Compared to Flatten (which would produce a
16×512 = 8192-dimensional vector), GAP:

- Is invariant to exact subcarrier position — the same filter activating at
  position 3 or position 11 contributes equally
- Reduces the number of parameters in the following dense layer dramatically
- Acts as a built-in spatial regularizer, reducing overfitting

### 10.5 Dual Output Heads

Both heads receive the same shared representation from `Dense(256) → Dropout →
Dense(128)`.

**Classification head:**

```
Dense(10, activation='softmax')
```

Softmax normalises the 10 raw scores into probabilities that sum to 1:

```
P(class k) = exp(score_k) / Σ exp(score_j)
```

The predicted class is `argmax(P)`. The softmax is numerically stable and
provides calibrated probability estimates.

**Regression head:**

```
Dense(64, activation='relu')
Dense(2,  activation='sigmoid')
```

Two outputs represent predicted normalised (row, col) coordinates in `[0, 1]²`.
Sigmoid ensures the output is bounded within this range, matching the
coordinate labelling scheme.

---

## 11. Loss Functions

### Classification — Sparse Categorical Cross-Entropy

For a 10-class problem with integer labels:

```
L_cls = -log( P(correct class) )
```

A perfect prediction (probability 1.0 on the correct class) gives loss 0.
A confident wrong prediction (probability near 1.0 on a wrong class) gives a
very large loss. This penalty is asymmetric — being confidently wrong is
heavily penalised, pushing the model to be both accurate and calibrated.

### Regression — Masked Mean Squared Error

```
L_reg = Σ_i [ mask_i × ||coord_pred_i - coord_true_i||² ] / Σ_i mask_i
```

- **MSE** penalises larger coordinate errors quadratically — a prediction 0.4
  units away from the true position incurs 4× the loss of one 0.2 units away.
- **mask_i = 1** for occupied samples, **0** for empty-room samples.
- Dividing by `Σ mask` gives the mean over occupied samples only; empty samples
  are fully excluded from the regression gradient.

### Combined loss

```
L_total = 1.0 × L_cls + 0.3 × L_reg
```

The regression task contributes ~23% of the total gradient signal — enough
to shape the backbone towards spatial awareness without displacing the
primary classification objective.

---

## 12. Training Strategy

### Optimiser — Adam

Adam (Adaptive Moment Estimation) maintains running estimates of the first
and second moments of each parameter's gradient:

```
m_t = β1 × m_{t-1} + (1 - β1) × g_t          (momentum — smoothed gradient)
v_t = β2 × v_{t-1} + (1 - β2) × g_t²         (RMSProp — smoothed gradient²)

θ_t = θ_{t-1} - (lr / sqrt(v_t) + ε) × m_t
```

Parameters with large consistent gradients (fast-learning layers) get small
effective learning rates — they are already moving in a reliable direction.
Parameters with small or noisy gradients get larger effective learning rates —
they need more nudging to make progress. This makes Adam robust across layers
of very different depths and scales.

Initial learning rate: `1e-3`.

### ReduceLROnPlateau

If `val_class_output_accuracy` does not improve for **5 consecutive epochs**,
the learning rate is multiplied by 0.5. Minimum learning rate: `1e-6`.

This implements learning rate annealing: large steps early in training find
a good region of the loss landscape quickly; progressively smaller steps
later allow fine-tuning without overshooting the minimum.

`mode='max'` is set explicitly — Keras cannot automatically infer direction
for the compound metric name `val_class_output_accuracy`.

### EarlyStopping

If `val_class_output_accuracy` does not improve for **15 consecutive epochs**,
training stops and the best weights are restored (`restore_best_weights=True`).

This prevents overfitting — the model memorising the training set at the
expense of generalisation — and avoids wasting compute after the model has
converged.

### ModelCheckpoint

Model weights are saved after every epoch that achieves a new best
`val_class_output_accuracy`. Even if training is interrupted, the best
model is available on disk at `model/best_model.keras`.

### Batch size

Batch size is set to **64** (doubled from v3's 32). The CNN-only model has no
recurrent state to maintain, so the forward and backward passes are much
faster per sample. A larger batch provides more stable gradient estimates per
update and makes better use of vectorised hardware.

---

## 13. Data Augmentation

### Gaussian noise injection

During training, small random Gaussian noise is added to each input window:

```
X_augmented = X + ε,    ε ~ N(0, σ²),    σ = 0.05
```

In standardised units (after scaling), σ = 0.05 means noise with standard
deviation equal to 5% of one unit — a mild perturbation that does not
significantly alter the signal structure.

### Why this works

Real CSI measurements always contain noise that was not present in the training
data. By training on slightly perturbed versions of each sample, the model
learns to classify based on the overall pattern structure rather than memorising
exact numerical values. This improves generalisation to real measurements.

Mathematically, training with Gaussian noise is equivalent to adding an L2
regularisation term to the loss — it penalises models that are sensitive to
small input perturbations, which is exactly what overfitting produces.

### Applied to training only

Augmentation is part of the `tf.data` pipeline and runs only during training.
Validation and test evaluation always use clean, unperturbed data to give an
accurate estimate of real-world performance.

### dtype safety

All arrays are `float32` from the start. `tf.random.normal` defaults to
`float32`. The addition `x + noise` is therefore always between two `float32`
tensors — no type mismatch.

---

## 14. Inference Pipeline

The same preprocessing chain applied during training must be reproduced
exactly at inference time.

**Step 1 — Collect 30 consecutive CSI frames**

```python
# Collect WINDOW_SIZE=30 packets from the receiver
frames_iq = [packet_1, packet_2, ..., packet_30]   # each has 256 IQ values
```

**Step 2 — Extract amplitude per frame**

```python
def iq_to_amplitude(iq_row):
    iq = np.array(iq_row, dtype=np.float32)
    I, Q = iq[0::2], iq[1::2]
    return np.sqrt(I**2 + Q**2)   # (128,)

amplitudes = np.stack([iq_to_amplitude(f) for f in frames_iq])  # (30, 128)
```

**Step 3 — Apply empty-room reference subtraction**

```python
empty_ref = joblib.load("scaler/empty_reference.pkl")  # (128,)
delta     = amplitudes - empty_ref                      # (30, 128)
```

**Step 4 — Stack into 2-channel tensor**

```python
window = np.stack([amplitudes, delta], axis=-1)   # (30, 128, 2)
```

**Step 5 — Scale**

```python
scaler = joblib.load("scaler/csi_scaler.pkl")
window_flat   = window.reshape(30, 256)
window_scaled = scaler.transform(window_flat).reshape(30, 128, 2)
window_scaled = window_scaled.astype(np.float32)
```

**Step 6 — Add batch dimension and predict**

```python
model_input = window_scaled[np.newaxis]               # (1, 30, 128, 2)
cls_probs, coords = model.predict(model_input)

predicted_block = cls_probs[0].argmax()               # int 0–9
predicted_row   = float(coords[0, 0]) * 2             # grid index 0–2
predicted_col   = float(coords[0, 1]) * 2             # grid index 0–2
```

The model internally applies `Permute → Reshape` to convert `(1, 30, 128, 2)`
to `(1, 128, 60)` — this happens automatically inside the Keras model graph
and does not need to be done manually at inference.

---

## 15. v4 vs v3 — Architectural Comparison

| Aspect | v3 (CNN + BiLSTM) | v4 (CNN-only) |
|--------|-------------------|---------------|
| Input to model | (batch, 30, 128, 2) | (batch, 30, 128, 2) |
| Frame processing | TimeDistributed CNN — one frame at a time | Reshape → single CNN pass over all frames |
| Temporal modelling | BiLSTM processes frame sequence | All 30 frames merged as 60 channels |
| How frames interact | Sequentially through LSTM hidden state | Simultaneously at every subcarrier position |
| Training speed | Slow (recurrent operations are sequential) | Fast (fully parallel convolutions) |
| Batch size | 32 | 64 |
| Model depth | Encoder CNN (3 stages) + 2× BiLSTM | 4-stage CNN (stem + 3 residual stages) |
| Widest feature map | 128 filters (encoder) + 256 BiLSTM units | 512 filters (Stage 3) |
| What LSTM provided | Learned temporal dynamics across frames | Replaced by simultaneous multi-frame channels |

**Key conceptual difference:**

v3's BiLSTM asked: *"How does the CSI pattern change from frame 1 to frame 30?"*

v4's reshape asks: *"Given all 30 frames simultaneously, what is the stable
subcarrier pattern that identifies this location?"*

Both are valid approaches to using the 30-frame window. The LSTM is better at
capturing motion or drift within a window; the CNN reshape is better at
identifying consistent multi-frame signatures where the person is stationary.
For a stationary person in a fixed block, the CNN reshape approach is a natural
fit.

---

## 16. Design Decision Summary

| Decision | Alternative | Reason for this choice |
|---|---|---|
| Amplitude features | Raw IQ | Phase-invariant; hardware-agnostic |
| Empty-room subtraction | No subtraction | Removes static room background; isolates human signal |
| 2-channel input (raw + delta) | Delta only | Gives model both absolute level and perturbation simultaneously |
| 30-frame sliding window | Single frame | Noise averaging; consistent patterns stand out across frames |
| Window stride = 5 | Stride = 1 | Reduces redundancy while keeping adequate sample count |
| Window within class only | Global window | Prevents windows straddling two class labels |
| Temporal split per class | Random split | Prevents overlapping windows between train and val/test |
| Reshape (30,128,2)→(128,60) | TimeDistributed + LSTM | All frames visible simultaneously; fully parallel; faster |
| CNN slides over subcarrier axis | CNN slides over frame axis | Subcarrier correlations are the physically meaningful structure |
| Residual blocks | Plain CNN | Prevents vanishing gradients; enables deeper architecture |
| SE channel attention | No attention | Learns which subcarrier ranges matter; suppresses noisy channels |
| 4 stages with doubling filters | Fixed-width CNN | Hierarchical abstraction; compensates reduced spatial resolution |
| Kernel 7→5→3→3 across stages | Uniform kernel | Large early kernels capture broad patterns; small later kernels refine |
| GlobalAveragePooling | Flatten | Position-invariant; fewer parameters; built-in regularizer |
| Coordinate regression (auxiliary) | Classification only | Encodes spatial topology; adjacent blocks share gradient signal |
| Masked regression loss | Unmasked | Empty-room has no valid coordinate; masking prevents confusion |
| Gaussian noise augmentation | No augmentation | Equivalent to L2 regularisation on inputs; improves generalisation |
| Batch size 64 | 32 | CNN is fully parallel; larger batches use hardware more efficiently |
| mode='max' on all callbacks | Default (auto) | Keras cannot infer direction for compound metric names |
