"""
CSI Room Block Classifier — v3
═══════════════════════════════════════════════════════════════════
Improvements over v2:
  1. dtype fix: all arrays cast to float32  (fixes AddV2 type mismatch)
  2. Temporal sliding windows within each class
       → 30-frame windows, stride 5
       → windows never cross a class boundary (data is ordered 0→9)
       → split is TEMPORAL per class (not random), preventing leakage
  3. Architecture: TimeDistributed CNN → Bi-LSTM
       → CNN extracts per-frame subcarrier features
       → LSTM captures how the CSI pattern evolves across 30 frames
  4. Empty-room reference subtraction (2-channel input)  [from v2]
  5. Residual CNN + SE channel attention                  [from v2]
  6. Multi-task coordinate regression                     [from v2]
  7. Gaussian noise augmentation (dtype-safe)             [fixed]
═══════════════════════════════════════════════════════════════════
"""

import os
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

os.makedirs("model",  exist_ok=True)
os.makedirs("scaler", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 30      # consecutive CSI frames per sample
STRIDE      = 5       # step between windows (within a class)
BATCH_SIZE  = 32      # smaller batch because sequences are larger
NOISE_STD   = 0.05    # augmentation noise (fraction of 1-std unit)
EPOCHS      = 100

# 3×3 grid coordinates (normalised to [0, 1])
BLOCK_COORDS = {
    0: (0.5, 0.5),                                    # empty — masked out
    1: (0.0, 0.0), 2: (0.0, 0.5), 3: (0.0, 1.0),
    4: (0.5, 0.0), 5: (0.5, 0.5), 6: (0.5, 1.0),
    7: (1.0, 0.0), 8: (1.0, 0.5), 9: (1.0, 1.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load raw CSI → amplitude  (float32 from the start)
# ─────────────────────────────────────────────────────────────────────────────
def load_csi_csv(filepath):
    data, labels = [], []
    with open(filepath) as f:
        for line in f:
            vals  = list(map(int, line.strip().split(',')))
            label = vals[0]
            iq    = np.array(vals[1:], dtype=np.float32)   # ← float32
            I, Q  = iq[0::2], iq[1::2]
            data.append(np.sqrt(I**2 + Q**2))
            labels.append(label)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int32)

X, y = load_csi_csv("data.csv")
print(f"Loaded  X:{X.shape}  y:{y.shape}  dtype:{X.dtype}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Empty-room reference subtraction → 2-channel input
# ─────────────────────────────────────────────────────────────────────────────
empty_ref = X[y == 0].mean(axis=0).astype(np.float32)   # shape (128,)
X_delta   = (X - empty_ref).astype(np.float32)
X2        = np.stack([X, X_delta], axis=-1)              # (N, 128, 2)

joblib.dump(empty_ref, "scaler/empty_reference.pkl")
print(f"2-channel features: {X2.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-channel StandardScaler  (fit on ALL data before windowing;
#    we scale the raw frames, then build windows — no leakage because
#    scaler sees amplitude statistics, not labels or window membership)
# ─────────────────────────────────────────────────────────────────────────────
N, L, C  = X2.shape
X_flat   = X2.reshape(N, L * C)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_flat).reshape(N, L, C).astype(np.float32)

joblib.dump(scaler, "scaler/csi_scaler.pkl")
print("Scaler saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build temporal sliding windows WITHIN each class
#
#    Data layout: rows 0–689 = class 0, 690–1379 = class 1, …
#    We slide a window of length WINDOW_SIZE with step STRIDE over each
#    class block independently.  No window ever spans two classes.
#
#    Temporal split per class (to prevent label leakage):
#      first 70 % of windows  → train
#      next  15 %             → val
#      last  15 %             → test
# ─────────────────────────────────────────────────────────────────────────────
def make_windows_for_class(X_cls, label):
    """
    X_cls : (M, 128, 2) — all frames for one class, in collection order
    Returns windows (W, WINDOW_SIZE, 128, 2), labels (W,), coords (W, 2), mask (W,)
    """
    M = len(X_cls)
    wins, lbls, crds, msks = [], [], [], []
    row, col = BLOCK_COORDS[label]
    occ      = 0.0 if label == 0 else 1.0

    for start in range(0, M - WINDOW_SIZE + 1, STRIDE):
        wins.append(X_cls[start : start + WINDOW_SIZE])
        lbls.append(label)
        crds.append([row, col])
        msks.append(occ)

    return (np.array(wins,  dtype=np.float32),
            np.array(lbls,  dtype=np.int32),
            np.array(crds,  dtype=np.float32),
            np.array(msks,  dtype=np.float32))


def temporal_split(wins, lbls, crds, msks, train_frac=0.70, val_frac=0.15):
    n       = len(wins)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    tr  = slice(0,             n_train)
    va  = slice(n_train,       n_train + n_val)
    te  = slice(n_train + n_val, None)
    return (wins[tr], wins[va], wins[te],
            lbls[tr], lbls[va], lbls[te],
            crds[tr], crds[va], crds[te],
            msks[tr], msks[va], msks[te])


split_parts = {'tr': [], 'va': [], 'te': []}

for cls in range(10):
    idx   = np.where(y == cls)[0]
    X_cls = X_scaled[idx]                           # frames for this class

    W, L_, C_, M_ = make_windows_for_class(X_cls, cls)
    parts = temporal_split(W, L_, C_, M_)
    # temporal_split returns 12 values in this order:
    #   index:  0     1     2     3     4     5     6     7     8     9     10    11
    #   value:  Xtr   Xva   Xte   ytr   yva   yte   ctr   cva   cte   mtr   mva  mte
    #
    # For each split we need (X, y, c, m) → offsets 0, 3, 6, 9 within that split column
    for col, split in enumerate(['tr', 'va', 'te']):
        split_parts[split].append((
            parts[col + 0],   # X  (offset 0,1,2)
            parts[col + 3],   # y  (offset 3,4,5)
            parts[col + 6],   # c  (offset 6,7,8)
            parts[col + 9],   # m  (offset 9,10,11)
        ))

def concat_parts(split_key):
    arrs = [np.concatenate([part[i] for part in split_parts[split_key]], axis=0)
            for i in range(4)]
    return arrs  # [X, y, c, m]

X_tr,  y_tr,  c_tr,  m_tr  = concat_parts('tr')
X_val, y_val, c_val, m_val = concat_parts('va')
X_te,  y_te,  c_te,  m_te  = concat_parts('te')

# Shuffle train (temporal order within class kept during split, shuffle after)
rng   = np.random.default_rng(42)
perm  = rng.permutation(len(X_tr))
X_tr, y_tr, c_tr, m_tr = X_tr[perm], y_tr[perm], c_tr[perm], m_tr[perm]

print(f"Windows → Train:{X_tr.shape}  Val:{X_val.shape}  Test:{X_te.shape}")
# Expected shape: (batch, WINDOW_SIZE, 128, 2)

# Pack coords + occupancy mask as (N, 3) for the regression head
def pack_reg(coords, mask):
    return np.concatenate([coords, mask[:, np.newaxis]], axis=-1).astype(np.float32)

reg_tr  = pack_reg(c_tr,  m_tr)
reg_val = pack_reg(c_val, m_val)
reg_te  = pack_reg(c_te,  m_te)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Model: TimeDistributed CNN encoder → Bidirectional LSTM → dual heads
#
#    Each of the WINDOW_SIZE frames is independently encoded by the same
#    small CNN (shared weights via TimeDistributed).  The resulting sequence
#    of feature vectors is fed into a Bi-LSTM that learns the temporal
#    evolution of the CSI pattern across the window.
# ─────────────────────────────────────────────────────────────────────────────

def se_block(x, ratio=8):
    """Squeeze-and-Excitation: re-weights channels by learned importance."""
    filters = x.shape[-1]
    sq = layers.GlobalAveragePooling1D()(x)
    sq = layers.Dense(max(filters // ratio, 1), activation='relu')(sq)
    sq = layers.Dense(filters, activation='sigmoid')(sq)
    sq = layers.Reshape((1, filters))(sq)
    return layers.Multiply()([x, sq])


def frame_encoder(input_shape):
    """
    Small CNN that encodes a single (128, 2) frame → 1D feature vector.
    Wrapped in TimeDistributed so it runs identically on every frame.
    """
    inp = tf.keras.Input(shape=input_shape)

    # Stage 1
    x = layers.Conv1D(32, 7, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)                    # → (64, 32)

    # Residual Stage 2
    shortcut = layers.Conv1D(64, 1, padding='same')(x)
    shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Conv1D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    x = se_block(x)
    x = layers.MaxPooling1D(2)(x)                    # → (32, 64)

    # Residual Stage 3
    shortcut2 = layers.Conv1D(128, 1, padding='same')(x)
    shortcut2 = layers.BatchNormalization()(shortcut2)
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut2])
    x = layers.ReLU()(x)
    x = se_block(x)

    x = layers.GlobalAveragePooling1D()(x)           # → (128,)
    return Model(inp, x, name='frame_encoder')


def build_model(window_size=WINDOW_SIZE, frame_shape=(128, 2), num_classes=10):
    inp = tf.keras.Input(shape=(window_size, *frame_shape), name='seq_input')
    # inp: (batch, T, 128, 2)

    # ── Per-frame CNN (shared weights across all T frames) ────────────────
    encoder  = frame_encoder(frame_shape)
    features = layers.TimeDistributed(encoder, name='td_cnn')(inp)
    # features: (batch, T, 128)

    # ── Temporal modelling: Bidirectional LSTM ────────────────────────────
    x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True), name='bilstm_1')(features)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(
            layers.LSTM(64,  return_sequences=False), name='bilstm_2')(x)
    x = layers.Dropout(0.3)(x)

    # ── Shared dense representation ───────────────────────────────────────
    shared = layers.Dense(128, activation='relu')(x)
    shared = layers.Dropout(0.4)(shared)
    shared = layers.Dense(64, activation='relu')(shared)

    # ── Classification head ───────────────────────────────────────────────
    cls_out = layers.Dense(num_classes, activation='softmax',
                           name='class_output')(shared)

    # ── Coordinate regression head (topology-aware) ───────────────────────
    reg = layers.Dense(32, activation='relu')(shared)
    reg_out = layers.Dense(2, activation='sigmoid',
                           name='coord_output')(reg)

    return Model(inputs=inp, outputs=[cls_out, reg_out])

model = build_model()
model.summary()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Masked MSE loss for the coordinate regression head
# ─────────────────────────────────────────────────────────────────────────────
class MaskedCoordMSE(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        coords_true = y_true[:, :2]
        mask        = y_true[:, 2:3]
        mse         = tf.reduce_mean(tf.square(coords_true - y_pred),
                                     axis=-1, keepdims=True)
        denom       = tf.maximum(tf.reduce_sum(mask), 1.0)
        return tf.reduce_sum(mse * mask) / denom

# ─────────────────────────────────────────────────────────────────────────────
# 7. tf.data pipelines with dtype-safe augmentation
# ─────────────────────────────────────────────────────────────────────────────
def augment(x, cls_lbl, reg_lbl):
    # x is float32; tf.random.normal defaults to float32 → types match
    noise = tf.random.normal(tf.shape(x), stddev=NOISE_STD, dtype=tf.float32)
    return x + noise, cls_lbl, reg_lbl

def repack(x, cls_lbl, reg_lbl):
    return x, {'class_output': cls_lbl, 'coord_output': reg_lbl}

train_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_tr, y_tr, reg_tr))
    .shuffle(len(X_tr), seed=42)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .map(repack,  num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_val, y_val, reg_val))
    .map(repack,  num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Compile and train
# ─────────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss={
        'class_output': 'sparse_categorical_crossentropy',
        'coord_output': MaskedCoordMSE(),
    },
    loss_weights={
        'class_output': 1.0,
        'coord_output': 0.3,
    },
    metrics={'class_output': 'accuracy'}
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_class_output_accuracy', mode='max', factor=0.5,
        patience=5, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_class_output_accuracy', mode='max', patience=15,
        restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        'model/best_model.keras',
        monitor='val_class_output_accuracy', mode='max',
        save_best_only=True, verbose=1),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

model.save("model/final_model.keras")
print("Saved → model/final_model.keras")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Evaluate on held-out test set
# ─────────────────────────────────────────────────────────────────────────────
class_names = ["Empty"] + [f"Block {i}" for i in range(1, 10)]

preds       = model.predict(X_te, batch_size=BATCH_SIZE)
y_pred      = preds[0].argmax(axis=1)
coords_pred = preds[1]

print(f"\nTest accuracy: {(y_pred == y_te).mean():.4f}\n")
print(classification_report(y_te, y_pred, target_names=class_names))

# Coordinate error — occupied samples only
occ_mask = m_te.astype(bool)
if occ_mask.sum() > 0:
    err = np.linalg.norm(coords_pred[occ_mask] - c_te[occ_mask], axis=1)
    print(f"Mean coord error (occupied): {err.mean():.4f} "
          f"(≈ {err.mean() * 3:.3f} block-widths)")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm     = confusion_matrix(y_te, y_pred)
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks(range(10)); ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_yticks(range(10)); ax.set_yticklabels(class_names)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Confusion Matrix — v3 (CNN + BiLSTM)')
thresh = cm.max() / 2
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')
plt.tight_layout()
plt.savefig('confusion_matrix_v3.png', dpi=150)
plt.show()

# ── Training curves ───────────────────────────────────────────────────────────
h = history.history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(h['class_output_accuracy'],     label='train acc')
axes[0].plot(h['val_class_output_accuracy'], label='val acc')
axes[0].set_title('Classification Accuracy'); axes[0].legend()
axes[1].plot(h['class_output_loss'],         label='train cls loss')
axes[1].plot(h['val_class_output_loss'],     label='val cls loss')
axes[1].set_title('Classification Loss'); axes[1].legend()
plt.tight_layout()
plt.savefig('training_curves_v3.png', dpi=150)
plt.show()

print("\nDone. Artefacts:")
print("  model/best_model.keras          ← best val accuracy weights")
print("  model/final_model.keras         ← final epoch weights")
print("  scaler/csi_scaler.pkl           ← StandardScaler (must use at inference)")
print("  scaler/empty_reference.pkl      ← empty-room mean (must use at inference)")
print("  confusion_matrix_v3.png")
print("  training_curves_v3.png")
