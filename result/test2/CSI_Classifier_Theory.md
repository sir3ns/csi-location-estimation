# Wi-Fi CSI Room Block Classifier — v3
## Theory & Methodology Documentation

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Why CSI, and Why It Is Hard](#2-why-csi-and-why-it-is-hard)
3. [Feature Engineering — From Raw IQ to Amplitude](#3-feature-engineering--from-raw-iq-to-amplitude)
4. [Empty-Room Reference Subtraction](#4-empty-room-reference-subtraction)
5. [Normalization — StandardScaler](#5-normalization--standardscaler)
6. [Temporal Sliding Windows](#6-temporal-sliding-windows)
7. [Dataset Splitting — Temporal Per-Class Strategy](#7-dataset-splitting--temporal-per-class-strategy)
8. [Coordinate Labels — Topology-Aware Encoding](#8-coordinate-labels--topology-aware-encoding)
9. [Model Architecture](#9-model-architecture)
   - 9.1 [Frame Encoder — Residual 1D-CNN with SE Attention](#91-frame-encoder--residual-1d-cnn-with-se-attention)
   - 9.2 [Temporal Encoder — Bidirectional LSTM](#92-temporal-encoder--bidirectional-lstm)
   - 9.3 [Dual Output Heads](#93-dual-output-heads)
10. [Loss Functions](#10-loss-functions)
11. [Training Strategy](#11-training-strategy)
12. [Data Augmentation](#12-data-augmentation)
13. [Inference Pipeline](#13-inference-pipeline)
14. [Design Decision Summary](#14-design-decision-summary)

---

## 1. Problem Definition

A Wi-Fi receiver continuously collects **Channel State Information (CSI)** packets. A person
standing at different positions in a room causes different multipath scattering patterns in
the Wi-Fi signal. The goal is to classify which block of a 3×3 grid the person occupies —
using only the CSI signal, with no cameras or wearables.

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
│ (0,0)    │ (0,1)    │ (0,2)    │
├──────────┼──────────┼──────────┤
│ Block 4  │ Block 5  │ Block 6  │
│ (1,0)    │ (1,1)    │ (1,2)    │
├──────────┼──────────┼──────────┤
│ Block 7  │ Block 8  │ Block 9  │
│ (2,0)    │ (2,1)    │ (2,2)    │
└──────────┴──────────┴──────────┘
```

---

## 2. Why CSI, and Why It Is Hard

### What is CSI?

Wi-Fi uses **OFDM (Orthogonal Frequency Division Multiplexing)**, which splits the signal
across many subcarrier frequencies simultaneously. CSI describes the complex gain —
amplitude and phase — that the channel applies to each subcarrier. It is a vector of
complex numbers, one per subcarrier, captured per packet.

When a person stands in a room, their body absorbs and reflects the Wi-Fi signal
differently depending on position. This changes the multipath propagation pattern and
leaves a detectable fingerprint in the CSI.

### Why is localization hard?

The CSI fingerprint of a location is not stable. It is corrupted by:

- **Phase noise** — carrier frequency offset and hardware differences cause random phase
  rotations that are irreproducible between sessions.
- **Static room signature** — walls, furniture, and the floor dominate the raw CSI.
  A person's body contributes only a small perturbation on top of this.
- **Single-snapshot noise** — one CSI packet is noisy. The signal-to-noise ratio for
  a single frame is poor; any single measurement could be an outlier.
- **Adjacent block similarity** — blocks that are physically close produce very similar
  CSI signatures, making them easy to confuse.

These four problems motivated every design decision in v3.

---

## 3. Feature Engineering — From Raw IQ to Amplitude

### Raw IQ representation

Each CSI sample is stored as 128 complex numbers. Each complex number has a real part
(**I — In-phase**) and an imaginary part (**Q — Quadrature**), giving 256 raw values
per sample in alternating order:

```
I1, Q1, I2, Q2, ..., I128, Q128
```

### Why not use IQ directly?

Raw I and Q values are sensitive to **phase offset**, which varies randomly due to:
- Clock drift between transmitter and receiver
- Minor hardware differences between devices
- Automatic gain control fluctuations

Two measurements at exactly the same location may have completely different I and Q
values if the phase offset changed between collection sessions. Using IQ directly as
features would train the model on noise.

### Amplitude extraction

**Amplitude** (also called magnitude) is computed per subcarrier:

```
amplitude_k = sqrt(I_k² + Q_k²)
```

Amplitude is the **length** of the complex vector in the IQ plane. It is invariant to
phase rotation — rotating the IQ vector does not change its length. Amplitude directly
reflects how much signal power was received at each subcarrier frequency, which is
physically determined by the propagation path and therefore by the room occupancy state.

This converts each sample from 256 raw values → **128 amplitude values**, one per
subcarrier. These are the features the model works with.

---

## 4. Empty-Room Reference Subtraction

### The problem: static room dominance

Raw CSI amplitude is dominated by the **static structure of the room** — the geometry
of walls, reflection from furniture, and the direct line-of-sight path. All of this is
present whether or not a person is in the room. The human-induced perturbation is a
relatively small signal sitting on top of a large, stable background.

A model trained on raw amplitude must learn to ignore this background on its own.
This is difficult and uses up model capacity that should be used for localization.

### The solution: background subtraction

The mean amplitude vector of all empty-room samples is computed:

```
empty_ref[k] = mean over all class-0 samples of amplitude_k
```

This gives a 128-element reference vector that captures the static room signature.
It is subtracted from every sample:

```
X_delta[k] = amplitude[k] - empty_ref[k]
```

`X_delta` now contains only the **human-induced change** at each subcarrier. Static
reflections cancel out. What remains is the localization signal.

### Two-channel input

Both the raw amplitude and the delta are kept as separate channels. The model receives
a **(128, 2)** tensor per frame:

- **Channel 0:** raw amplitude — carries absolute signal level information
- **Channel 1:** delta (amplitude − empty reference) — carries the human-specific
  perturbation

This gives the model complementary views. The raw channel provides context about
absolute subcarrier levels; the delta channel isolates the person's contribution.

### Why the reference must be saved

The same `empty_ref` vector used during training **must** be used at inference time.
It is saved to `scaler/empty_reference.pkl` and loaded before any prediction. Applying
a different reference would shift the delta channel in ways the model has never seen.

---

## 5. Normalization — StandardScaler

### Why normalization is needed

After computing the two-channel features, different subcarriers still vary in absolute
scale. One subcarrier might consistently produce amplitudes in [5, 30] while another
produces [100, 400]. Without normalization, the neural network gradient updates would
be dominated by high-amplitude subcarriers simply because their numerical values are
larger — not because they are more informative for localization.

### StandardScaler

StandardScaler transforms each feature dimension to have zero mean and unit variance:

```
X_scaled[k] = (X[k] - mean[k]) / std[k]
```

Where `mean[k]` and `std[k]` are computed across all training samples for feature
dimension `k`.

After scaling, every subcarrier contributes on equal numerical footing. The network
must learn which subcarriers are actually informative by gradient descent, not by
scale.

### Preventing data leakage

The scaler is fitted **only on the training set**, then applied to validation and test
sets using the same mean/std. This is critical — if the scaler saw validation or test
statistics, it would encode information about unseen data into the preprocessing step,
inflating the apparent performance.

The fitted scaler is saved to `scaler/csi_scaler.pkl` for reuse at inference.

---

## 6. Temporal Sliding Windows

### Why single frames are insufficient

A single CSI packet is noisy. Any one frame might be corrupted by:
- Multipath interference from moving objects other than the person
- Environmental fluctuations (temperature, humidity affecting propagation)
- Receiver noise

Using a single frame per prediction means the model has no way to distinguish a
genuine localization signal from momentary noise.

### The sliding window approach

Instead of classifying a single frame, the model classifies a **window of 30
consecutive frames**. This represents a short burst of CSI measurements taken in
rapid succession while the person is stationary in a block.

A sliding window with step STRIDE=5 is applied over each class block independently:

```
Window 1:  frames [0  .. 29]
Window 2:  frames [5  .. 34]
Window 3:  frames [10 .. 39]
...
```

With 690 frames per class and WINDOW_SIZE=30, STRIDE=5:

```
Number of windows per class = (690 - 30) // 5 + 1 = 133 windows
Total windows across 10 classes = 1330 windows
```

### Why windows must stay within a class

The dataset is ordered by class (all class 0 frames first, then class 1, etc.).
If windowing were applied globally across the entire dataset, windows near the
class boundary would contain frames from two different classes — the model would
receive a contradictory training signal (window partially in Block 1, partially
in Block 2, labelled as one of them). All windowing is therefore performed
**separately per class** so no window ever straddles a boundary.

### What the LSTM gains from this

The 30-frame window is fed into the LSTM as a **sequence of feature vectors**, one
per frame. The LSTM learns the temporal dynamics of the CSI signal at each location:
how the subcarrier amplitudes fluctuate over time while a person stands in a specific
block. Adjacent blocks may look similar in any single frame, but their temporal
evolution patterns are different. The LSTM captures this distinction.

---

## 7. Dataset Splitting — Temporal Per-Class Strategy

### Why random splitting is wrong here

The standard approach of randomly splitting the dataset into train/val/test is
**unsafe for windowed temporal data**. With stride=5 and window=30, consecutive
windows overlap by 25 frames. If window at position t goes to train and window
at position t+1 goes to validation, 25 of the 30 frames are shared between them.
The model would see the validation data during training, producing an optimistic
and misleading accuracy estimate.

### Temporal split per class

For each class independently, the windows are split by time position (not randomly):

```
First 70% of windows  → training set
Next  15% of windows  → validation set
Last  15% of windows  → test set
```

Because windows are ordered by their start position within the class, this guarantees
that no training window overlaps with any validation or test window. The gap between
the last training window and first validation window is at least 1 frame with
STRIDE=5 (and more as the slices are contiguous cuts, not overlapping).

### Post-split shuffle of training set

After the temporal split, the training windows from all 10 classes are **shuffled
together**. This is safe because the split has already been done — shuffling the
training set prevents the model from exploiting the ordering of classes within a
batch (it should not be able to predict "class 3 follows class 2 in the batch").

Validation and test sets are kept in their original order — order does not affect
evaluation metrics.

---

## 8. Coordinate Labels — Topology-Aware Encoding

### The problem with arbitrary class labels

Standard multi-class classification treats the 10 classes as **completely unrelated
categories**. The loss function penalises predicting Block 1 when the true label is
Block 9 exactly the same as predicting Block 2 when the true label is Block 9 —
even though Block 2 and Block 9 are far apart spatially while Block 8 and Block 9
are adjacent.

The network has no information about the spatial relationships between blocks.

### Coordinate regression as a secondary task

Each block is assigned normalised (row, col) coordinates:

```
Block 1 → (0.0, 0.0)    Block 2 → (0.0, 0.5)    Block 3 → (0.0, 1.0)
Block 4 → (0.5, 0.0)    Block 5 → (0.5, 0.5)    Block 6 → (0.5, 1.0)
Block 7 → (1.0, 0.0)    Block 8 → (1.0, 0.5)    Block 9 → (1.0, 1.0)
```

A second output head predicts these (row, col) values via regression. The shared
representation — the backbone layers used by both the classification and regression
heads — must now learn features that are useful for predicting position. This forces
the model to treat Block 4 and Block 5 as related (both in row 1) rather than as
arbitrary different categories.

### Occupancy masking

Coordinate regression is only meaningful when a person is present. The empty-room
class (label 0) has no valid spatial coordinate. Including empty-room samples in the
regression loss would confuse the model (it would try to predict a position when
there is no person).

An **occupancy mask** (1 for occupied, 0 for empty) is computed for each sample.
The regression loss is masked:

```
loss = sum(MSE * mask) / sum(mask)
```

Only occupied samples contribute to the coordinate loss. Empty samples contribute
only to the classification loss.

### Loss weighting

The total loss is a weighted sum:

```
total_loss = 1.0 × classification_loss + 0.3 × regression_loss
```

The weight 0.3 keeps coordinate regression as an **auxiliary task** — it guides the
representation towards spatial awareness without overpowering the primary
classification objective.

---

## 9. Model Architecture

The full architecture processes the (30, 128, 2) input — a sequence of 30 frames,
each with 128 subcarriers and 2 channels — in two stages:

```
Input: (batch, 30, 128, 2)
          │
          ▼
TimeDistributed(CNN Encoder)  ← same CNN applied to each of the 30 frames
          │
   (batch, 30, 128)           ← sequence of per-frame feature vectors
          │
          ▼
Bidirectional LSTM (×2)       ← learns temporal dynamics across 30 frames
          │
   (batch, 256)               ← single vector summarising the whole window
          │
          ▼
Shared Dense Layers           ← joint representation for both heads
     ┌────┴────┐
     ▼         ▼
Classification  Regression
   Head          Head
(10-class      (x,y coords)
 softmax)
```

### 9.1 Frame Encoder — Residual 1D-CNN with SE Attention

The frame encoder takes a single **(128, 2)** amplitude frame and outputs a
**128-dimensional feature vector**. It is a 1D CNN because the 128 subcarrier
amplitudes are an ordered sequence — nearby subcarriers experience correlated
fading, and convolution captures these local frequency-domain patterns.

#### Residual connections

A plain deep CNN suffers from **vanishing gradients** — as the network gets deeper,
gradients shrink exponentially during backpropagation, and early layers stop learning.
Residual (skip) connections bypass this by adding the input of a block directly to its
output:

```
output = F(x) + x
```

Where `F(x)` is the two-conv transformation. The gradient now flows directly through
the addition — even if `F(x)` has small gradients, the gradient through `x` is
unchanged. This allows training deeper networks effectively.

When the number of filters changes between input and output (e.g., 32 → 64 filters),
a 1×1 convolution is applied to the shortcut to project it to the right dimension
before the addition.

#### Squeeze-and-Excitation (SE) channel attention

After each residual block, an SE block recalibrates the importance of each feature
map (channel) globally. It works in three steps:

**Squeeze:** Global average pooling collapses the entire sequence length into a
single value per channel. This gives a global descriptor of what each filter
responded to across all 128 subcarrier positions:

```
z_c = (1/128) × sum over positions of feature_map_c
```

**Excitation:** Two fully connected layers learn a non-linear mapping from these
global descriptors to per-channel weights in (0, 1):

```
s = sigmoid( W2 × relu( W1 × z ) )
```

The bottleneck (filters // 8) in the first layer forces the network to learn a
compact, global interaction between channels rather than independent weights.

**Scale:** Each channel of the feature map is multiplied by its learned weight:

```
output_c = s_c × feature_map_c
```

Channels that are consistently more informative for localization get upweighted;
channels that fire on noise get suppressed. The weights are learned end-to-end
during training.

#### Architecture stages

```
Input: (128, 2)
  │
  ├── Stem: Conv1D(32, kernel=7) → BN → ReLU → MaxPool(2)   → (64, 32)
  │
  ├── Stage 1: ResBlock(64, kernel=5) → SE → MaxPool(2)       → (32, 64)
  │
  ├── Stage 2: ResBlock(128, kernel=3) → SE                   → (32, 128)
  │
  └── GlobalAveragePooling1D                                   → (128,)
```

The kernel sizes decrease (7 → 5 → 3) because:
- The first layer uses a large kernel to capture broad frequency-domain patterns
  spanning many subcarriers.
- Later layers use smaller kernels to refine fine-grained local patterns within
  the compressed representation.

### 9.2 Temporal Encoder — Bidirectional LSTM

After the CNN encoder processes each of the 30 frames independently, the model has
a sequence of 30 feature vectors — one 128-dimensional vector per frame. This
sequence is fed into a **Bidirectional LSTM**.

#### Why LSTM?

LSTM (Long Short-Term Memory) is a recurrent neural network designed to learn
dependencies across time. It maintains a **hidden state** and a **cell state** that
persist across the sequence. Three gates control information flow:

- **Forget gate:** decides what information to discard from the cell state
- **Input gate:** decides what new information to write into the cell state
- **Output gate:** decides what to output from the cell state at this step

This gating mechanism allows the LSTM to learn **which temporal patterns are
meaningful** — for example, a brief spike in a particular subcarrier at frame 5
that is consistent with Block 3's signature — without being confused by irrelevant
short-term fluctuations.

#### Why Bidirectional?

A standard LSTM processes the sequence left-to-right (frame 1 → frame 30). It can
only use past context when processing each frame.

A **Bidirectional LSTM** runs two LSTMs: one forward, one backward. Their outputs
are concatenated at each step. This means when deciding how to weight a feature at
frame 15, the model can use information from both earlier frames (1–14) and later
frames (16–30). For CSI classification this is beneficial because the entire 30-frame
window is available before making a prediction — there is no real-time streaming
constraint.

#### Stacked LSTMs

Two BiLSTM layers are stacked:

```
BiLSTM(128, return_sequences=True)  → (batch, 30, 256)
  ↓ Dropout(0.3)
BiLSTM(64, return_sequences=False)  → (batch, 128)
```

The first layer returns the full sequence so the second layer can see all 30
time steps. The second layer returns only the final state, which summarises the
entire 30-frame window into a single vector.

### 9.3 Dual Output Heads

Both heads receive the same shared representation, produced by two dense layers
after the LSTM:

```
Dense(128) → Dropout(0.4) → Dense(64)
```

**Classification head:**

```
Dense(10, activation='softmax')
```

Softmax converts raw scores to probabilities summing to 1 across all 10 classes.
The predicted class is the one with the highest probability.

**Regression head:**

```
Dense(32, activation='relu') → Dense(2, activation='sigmoid')
```

Sigmoid bounds the output to (0, 1), matching the normalised coordinate range.
The two outputs represent predicted (row, col) position.

---

## 10. Loss Functions

### Classification — Sparse Categorical Cross-Entropy

For a 10-class problem with integer labels, sparse categorical cross-entropy is:

```
L_cls = -log( p_correct_class )
```

Where `p_correct_class` is the softmax probability assigned to the true label.
A perfect prediction (probability 1.0 on the correct class) gives loss 0. A
confident wrong prediction (probability ~1.0 on a wrong class) gives a very
large loss. This pushes the model to be both correct and confident.

### Regression — Masked Mean Squared Error

For the coordinate head:

```
L_reg = sum( MSE(coord_pred, coord_true) × mask ) / sum(mask)
```

Where:
- `MSE` is the mean squared error between predicted and true (row, col)
- `mask` is 1 if the sample is occupied (person present), 0 if empty
- Dividing by `sum(mask)` gives the mean over occupied samples only

Empty-room samples are masked out entirely from the regression loss. Their
predictions are ignored and do not influence the coordinate head weights.

### Combined loss

```
total_loss = 1.0 × L_cls + 0.3 × L_reg
```

The weight 0.3 means the regression task contributes 23% of the total gradient
signal. It is strong enough to shape the representation towards spatial awareness,
but not strong enough to interfere with classification accuracy.

---

## 11. Training Strategy

### Optimiser — Adam

Adam (Adaptive Moment Estimation) maintains per-parameter learning rate estimates
based on first and second moments of the gradients:

```
m_t = β1 × m_{t-1} + (1 - β1) × g_t       (first moment — mean)
v_t = β2 × v_{t-1} + (1 - β2) × g_t²      (second moment — variance)
θ_t = θ_{t-1} - lr × m_t / (sqrt(v_t) + ε)
```

Parameters with consistently large gradients get smaller effective learning rates
(they are already moving fast). Parameters with small or noisy gradients get larger
effective rates (they need more nudging). This makes Adam robust to the scale
differences that exist between shallow and deep layers in the network.

Initial learning rate: `1e-3`.

### ReduceLROnPlateau

If `val_class_output_accuracy` does not improve for 5 consecutive epochs, the
learning rate is halved (factor=0.5). Minimum learning rate is `1e-6`.

This implements a form of **learning rate annealing**: the model starts with large
steps to find a good region of the loss landscape quickly, then takes progressively
smaller steps to fine-tune within that region.

### EarlyStopping

If `val_class_output_accuracy` does not improve for 15 consecutive epochs, training
stops and the weights from the best epoch are restored.

This prevents **overfitting** — the model memorising the training set at the expense
of generalisation. It also saves computation by not continuing after the model has
stopped improving.

### ModelCheckpoint

The model weights are saved to disk after every epoch that achieves a new best
`val_class_output_accuracy`. Even if training crashes or is interrupted, the best
model is preserved on disk.

---

## 12. Data Augmentation

### Gaussian noise injection

During training, small Gaussian noise is added to each input window:

```
X_augmented = X + N(0, σ²)     where σ = 0.05
```

In standardised units (after scaling), σ=0.05 means noise with standard deviation
equal to 5% of one standard deviation of the feature distribution. This is a mild
perturbation.

### Why this helps

Real CSI measurements always contain noise. By training on slightly perturbed
versions of each sample, the model learns to classify based on the overall pattern
rather than memorising exact numerical values. This improves generalisation to
real-world measurements that will never be identical to the training data.

### Applied only to training

The augmentation is part of the `tf.data` pipeline and is applied **only** to the
training dataset, not to validation or test. Evaluation is always performed on
clean (unperturbed) data to get an accurate estimate of real performance.

---

## 13. Inference Pipeline

At inference time, the same preprocessing chain used during training must be applied
exactly. Deviating from any step will produce incorrect feature representations and
wrong predictions.

**Step 1 — Collect a window of 30 frames**

Capture 30 consecutive CSI packets from the receiver.

**Step 2 — Extract amplitude per frame**

For each frame:
```python
amplitude = np.sqrt(I**2 + Q**2)    # shape: (128,)
```

**Step 3 — Apply empty-room reference subtraction**

```python
empty_ref = joblib.load("scaler/empty_reference.pkl")
delta     = amplitude - empty_ref    # shape: (128,)
```

**Step 4 — Stack into 2-channel input**

```python
frame_2ch = np.stack([amplitude, delta], axis=-1)   # shape: (128, 2)
```

Repeat steps 2–4 for all 30 frames to build:
```python
window = np.stack(frames_2ch, axis=0)   # shape: (30, 128, 2)
```

**Step 5 — Scale**

```python
scaler = joblib.load("scaler/csi_scaler.pkl")
window_flat   = window.reshape(30, 256)
window_scaled = scaler.transform(window_flat).reshape(30, 128, 2)
```

**Step 6 — Add batch dimension and predict**

```python
model_input   = window_scaled[np.newaxis]            # shape: (1, 30, 128, 2)
cls_probs, coords = model.predict(model_input)

predicted_block = cls_probs.argmax()                 # integer 0–9
predicted_row   = coords[0, 0] * 2                  # scale back to grid index
predicted_col   = coords[0, 1] * 2
```

---

## 14. Design Decision Summary

| Decision | Alternative considered | Why this choice was made |
|---|---|---|
| Amplitude features | Raw IQ | Phase-invariant; not corrupted by hardware drift |
| Empty-room subtraction | No subtraction | Removes static room background; isolates human signal |
| 2-channel input (raw + delta) | Delta only | Gives model both absolute level and human perturbation |
| Temporal windows of 30 frames | Single frame | Single frames are too noisy; LSTM needs a sequence |
| Temporal split per class | Random split | Prevents overlapping windows leaking between train and val |
| Windowing within class only | Global windowing | Prevents windows spanning two class labels |
| Residual CNN | Plain CNN | Enables deeper network without vanishing gradients |
| SE attention | No attention | Learns which subcarriers matter; suppresses noisy channels |
| Bidirectional LSTM | Unidirectional | Full window is available at inference; both directions help |
| Stacked BiLSTM (2 layers) | Single layer | Hierarchical temporal abstraction |
| Coordinate regression (auxiliary) | Classification only | Encodes spatial topology; adjacent blocks share gradient signal |
| Masked regression loss | Unmasked | Empty-room has no valid coordinate; masking prevents confusion |
| Loss weight 0.3 on regression | Equal weights | Keeps classification as the primary objective |
| Gaussian noise augmentation | No augmentation | Prevents memorisation of exact numerical values |
| StandardScaler after reference subtraction | Before | Scaling should operate on the final 2-channel feature, not raw amplitude |
