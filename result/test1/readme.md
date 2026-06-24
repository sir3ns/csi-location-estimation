# Wi-Fi CSI-Based Room Block Classification

## Overview

This document explains the theory and methodology behind training a deep learning model
that can identify which block or sector of a room a person is in — using only Wi-Fi
Channel State Information (CSI) signals. No cameras, no wearables. Just the Wi-Fi signal
already present in the environment.

---

## 1. What is CSI?

When a Wi-Fi signal travels from a transmitter to a receiver, it bounces off walls,
furniture, and people before arriving at the receiver. This phenomenon is called
**multipath propagation**. The receiver records how the signal was distorted during
this journey — this recording is called **Channel State Information (CSI)**.

CSI is represented as a set of complex numbers, one per subcarrier frequency:

```
CSI value = I + jQ
```

Where:
- `I` is the **in-phase** (real) component
- `Q` is the **quadrature** (imaginary) component
- Each `(I, Q)` pair corresponds to one subcarrier

When a person stands in a room, their body absorbs and reflects Wi-Fi signals in a way
that is unique to their position. Different room blocks produce measurably different CSI
patterns, which is the core idea this system exploits.

---

## 2. Problem Definition

Given a single CSI snapshot from a Wi-Fi receiver, classify it into one of **10 classes**:

| Label | Meaning         |
|-------|-----------------|
| 0     | Empty room      |
| 1     | Block 1         |
| 2     | Block 2         |
| ...   | ...             |
| 9     | Block 9         |

The room is divided into a 3×3 grid of 9 spatial blocks. The model must also correctly
identify when no person is present (class 0), which prevents false detections.

---

## 3. Dataset

| Property            | Value                        |
|---------------------|------------------------------|
| Total samples       | 6900                         |
| Classes             | 10 (1 empty + 9 blocks)      |
| Samples per class   | 690                          |
| Features per sample | 256 raw values (128 I/Q pairs)|
| Class balance       | Perfectly balanced           |

Each row in the CSV file has the format:

```
label, I1, Q1, I2, Q2, ..., I128, Q128
```

The first value is the class label. The remaining 256 values are 128 complex CSI
measurements, stored as alternating real (I) and imaginary (Q) components.

---

## 4. Feature Extraction — From Raw CSI to Amplitude

Raw I and Q values alone are not ideal features because they are sensitive to phase
shifts caused by minor hardware differences and environmental noise. **Amplitude**
is more stable and physically meaningful.

For each subcarrier, amplitude is computed as:

```
amplitude = sqrt(I² + Q²)
```

This converts each sample from 256 raw values into **128 amplitude values**, one per
subcarrier. Amplitude directly reflects how much the Wi-Fi signal was attenuated at
each frequency, which correlates strongly with the physical obstruction caused by a
person's body at a particular location.

---

## 5. Preprocessing

### 5.1 Normalization

Raw amplitudes across different subcarriers can vary significantly in scale. A subcarrier
at one frequency may consistently produce values in the range [5, 30] while another
produces values in [100, 400]. This scale difference can mislead the neural network into
treating high-amplitude subcarriers as more important.

**StandardScaler** is applied to fix this:

```
X_scaled = (X - mean) / std
```

Mean and standard deviation are computed **only from the training set**, then applied
to validation and test sets. This prevents data leakage — the model never sees statistics
from unseen data during training.

> **Important:** The fitted scaler must be saved and reused at inference time.
> Applying a different scale at inference will produce wrong predictions.

### 5.2 Train / Validation / Test Split

The dataset is split using **stratified sampling**, which guarantees that every class
is represented equally in each subset:

| Split      | Proportion | Samples per class | Total  |
|------------|------------|-------------------|--------|
| Train      | 70%        | ~483              | ~4830  |
| Validation | 15%        | ~103              | ~1035  |
| Test       | 15%        | ~103              | ~1035  |

Without stratification, random splitting could leave some classes underrepresented in
validation, leading to unreliable accuracy estimates.

### 5.3 Reshaping for the Model

Keras Conv1D layers expect input of shape `(samples, length, channels)`.
The 128-amplitude vector is reshaped to `(samples, 128, 1)` — treating it as a
1-dimensional signal with a single channel.

---

## 6. Model Architecture — 1D Convolutional Neural Network

A **1D Convolutional Neural Network (1D-CNN)** is used because the 128 amplitude values
form an ordered sequence across subcarrier frequencies. Nearby subcarriers tend to be
correlated (they experience similar fading), and convolution is well-suited to capturing
these local patterns.

### Architecture Summary

```
Input: (batch, 128, 1)
        │
        ▼
┌─────────────────────────────┐
│  Conv1D(32, kernel=7)       │  ← captures broad frequency patterns
│  BatchNorm → ReLU           │
│  MaxPool(2)  → length: 64   │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Conv1D(64, kernel=5)       │  ← captures mid-scale patterns
│  BatchNorm → ReLU           │
│  MaxPool(2)  → length: 32   │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Conv1D(128, kernel=3)      │  ← captures fine-grained patterns
│  BatchNorm → ReLU           │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  GlobalAveragePooling1D     │  ← collapses sequence → (batch, 128)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Dense(64) → ReLU           │
│  Dropout(0.4)               │  ← prevents overfitting
│  Dense(10) → Softmax        │  ← 10-class probability output
└─────────────────────────────┘
```

### Key Design Choices

**Increasing filter counts (32 → 64 → 128):** Early layers learn simple low-level
patterns (a dip at a certain frequency). Later layers combine these into complex
location-specific signatures.

**BatchNormalization:** Normalizes activations within each mini-batch, stabilizing
training and allowing higher learning rates.

**MaxPooling:** Reduces the sequence length by half at each stage, making the model
progressively more spatially invariant and reducing computation.

**GlobalAveragePooling1D:** Instead of flattening (which is sensitive to input length),
this takes the average of each feature map across the entire sequence. It acts as a
built-in regularizer and produces a compact fixed-size representation regardless of
input length.

**Dropout(0.4):** During training, randomly sets 40% of neurons to zero. This forces
the network to learn redundant representations and reduces overfitting.

**Softmax output:** Converts raw scores into probabilities that sum to 1 across all
10 classes. The predicted class is the one with the highest probability.

---

## 7. Training Strategy

### Loss Function

**Sparse Categorical Cross-Entropy** is used because this is a multi-class classification
problem with integer labels (no one-hot encoding needed):

```
Loss = -log(probability assigned to the correct class)
```

A perfect prediction gives loss ≈ 0. A wrong confident prediction gives a very high loss,
pushing the model to correct itself.

### Optimizer

**Adam** with learning rate `1e-3`. Adam adapts the learning rate individually for each
parameter, making it robust and fast to converge for this type of problem.

### Callbacks

Three callbacks monitor and control the training process:

**ReduceLROnPlateau:** If validation accuracy stops improving for 5 consecutive epochs,
the learning rate is halved. This allows the model to make finer adjustments when it
gets close to a good solution.

**EarlyStopping:** If validation accuracy does not improve for 15 consecutive epochs,
training stops and the best weights are restored. This prevents wasting time and
overfitting past the optimal point.

**ModelCheckpoint:** Saves the model weights to disk whenever a new best validation
accuracy is achieved. Even if training crashes, the best model is preserved.

---

## 8. Evaluation

After training, the model is evaluated on the held-out **test set** — data it has
never seen during training or validation.

### Metrics Used

**Overall accuracy:** Percentage of samples classified correctly across all 10 classes.

**Per-class precision, recall, F1-score:**
- *Precision* — of all samples predicted as Block X, how many actually were Block X
- *Recall* — of all actual Block X samples, how many were correctly identified
- *F1-score* — harmonic mean of precision and recall; useful when both matter

**Confusion matrix:** A 10×10 grid where row `i`, column `j` shows how many samples
of true class `i` were predicted as class `j`. Diagonal entries are correct predictions;
off-diagonal entries are errors. This reveals which blocks the model confuses with each
other (typically adjacent blocks with similar signal patterns).

---

## 9. Inference on New Data

To predict the block for a new CSI sample:

1. Parse the raw CSV line and extract the 256 I/Q values
2. Compute amplitude: `sqrt(I² + Q²)` → 128 values
3. Apply the **saved scaler** (same one fitted on training data)
4. Reshape to `(1, 128, 1)`
5. Pass through the **saved model**
6. Take `argmax` of the 10 softmax outputs → predicted class

```python
amplitude = sqrt(I² + Q²)          # step 2
x = scaler.transform(amplitude)     # step 3
x = x.reshape(1, 128, 1)           # step 4
probs = model.predict(x)            # step 5
pred  = probs.argmax()              # step 6
# pred == 0 → Empty room
# pred == 1..9 → Block number
```

---

## 10. Saved Artifacts

After training, two files are saved and must be kept together:

| File                        | Purpose                                      |
|-----------------------------|----------------------------------------------|
| `model/best_model.keras`    | Trained CNN weights (best validation accuracy)|
| `model/final_model.keras`   | Trained CNN weights (end of training)        |
| `scaler/csi_scaler.pkl`     | Fitted StandardScaler (mean and std per feature)|

> The scaler and model are tightly coupled. Always use the scaler that was fitted
> during the same training run as the model. Using a different scaler will produce
> incorrect predictions.

---

## 11. Summary

```
Raw Wi-Fi CSI (I/Q pairs)
        │
        ▼
Amplitude extraction  →  128 features per sample
        │
        ▼
StandardScaler normalization
        │
        ▼
1D-CNN  →  learns frequency-domain spatial signatures
        │
        ▼
Softmax output  →  10-class prediction
                   (0 = empty, 1–9 = room block)
```

The key insight is that a person's body creates a **unique distortion pattern** in the
Wi-Fi signal that depends on their location. By learning these patterns from labeled
examples, the CNN can reliably infer position from a single snapshot of CSI amplitudes —
with no GPS, no camera, and no device worn by the person.
