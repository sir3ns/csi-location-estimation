"""
CSI Room Block Classifier — v4 (CNN-only, no LSTM)
═══════════════════════════════════════════════════════════════════
Everything from v3 is kept exactly:
  ✓ float32 throughout
  ✓ Empty-room reference subtraction (2-channel input)
  ✓ StandardScaler
  ✓ Temporal sliding windows within each class (window=30, stride=5)
  ✓ Temporal per-class split (no random split / no window leakage)
  ✓ Multi-task coordinate regression with occupancy-masked MSE
  ✓ Gaussian noise augmentation
  ✓ mode='max' on all callbacks

Change vs v3:
  ✗ TimeDistributed CNN + BiLSTM  →  removed

  ✓ Pure CNN on the full window
    The (30, 128, 2) window is reshaped to (128, 60):
      • axis 0 (length 128) = subcarrier frequency axis  ← CNN slides here
      • axis 1 (depth   60) = 30 frames × 2 channels    ← all frames at once
    The CNN therefore sees every frame simultaneously, treating the 30
    measurements as a rich multi-channel view of each subcarrier position.
    Residual blocks + SE attention are then applied on this (128, 60) input.
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
WINDOW_SIZE = 30
STRIDE      = 5
BATCH_SIZE  = 64      # CNN-only is much faster → larger batch is fine
NOISE_STD   = 0.05
EPOCHS      = 120

BLOCK_COORDS = {
    0: (0.5, 0.5),
    1: (0.0, 0.0), 2: (0.0, 0.5), 3: (0.0, 1.0),
    4: (0.5, 0.0), 5: (0.5, 0.5), 6: (0.5, 1.0),
    7: (1.0, 0.0), 8: (1.0, 0.5), 9: (1.0, 1.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load raw CSI → amplitude
# ─────────────────────────────────────────────────────────────────────────────
def load_csi_csv(filepath):
    data, labels = [], []
    with open(filepath) as f:
        for line in f:
            vals  = list(map(int, line.strip().split(',')))
            label = vals[0]
            iq    = np.array(vals[1:], dtype=np.float32)
            I, Q  = iq[0::2], iq[1::2]
            data.append(np.sqrt(I**2 + Q**2))
            labels.append(label)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int32)

X, y = load_csi_csv("data.csv")
print(f"Loaded  X:{X.shape}  y:{y.shape}  dtype:{X.dtype}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Empty-room reference subtraction → 2-channel input
# ─────────────────────────────────────────────────────────────────────────────
empty_ref = X[y == 0].mean(axis=0).astype(np.float32)   # (128,)
X_delta   = (X - empty_ref).astype(np.float32)
X2        = np.stack([X, X_delta], axis=-1)              # (N, 128, 2)

joblib.dump(empty_ref, "scaler/empty_reference.pkl")
print(f"2-channel features: {X2.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. StandardScaler
# ─────────────────────────────────────────────────────────────────────────────
N, L, C  = X2.shape
X_flat   = X2.reshape(N, L * C)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_flat).reshape(N, L, C).astype(np.float32)

joblib.dump(scaler, "scaler/csi_scaler.pkl")
print("Scaler saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Temporal sliding windows within each class
# ─────────────────────────────────────────────────────────────────────────────
def make_windows_for_class(X_cls, label):
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
    tr = slice(0,             n_train)
    va = slice(n_train,       n_train + n_val)
    te = slice(n_train + n_val, None)
    return (wins[tr], wins[va], wins[te],
            lbls[tr], lbls[va], lbls[te],
            crds[tr], crds[va], crds[te],
            msks[tr], msks[va], msks[te])

split_parts = {'tr': [], 'va': [], 'te': []}

for cls in range(10):
    idx   = np.where(y == cls)[0]
    X_cls = X_scaled[idx]
    W, L_, C_, M_ = make_windows_for_class(X_cls, cls)
    parts = temporal_split(W, L_, C_, M_)
    for col_idx, split in enumerate(['tr', 'va', 'te']):
        split_parts[split].append((
            parts[col_idx + 0],   # X
            parts[col_idx + 3],   # y
            parts[col_idx + 6],   # c
            parts[col_idx + 9],   # m
        ))

def concat_parts(split_key):
    return [np.concatenate([part[i] for part in split_parts[split_key]], axis=0)
            for i in range(4)]

X_tr,  y_tr,  c_tr,  m_tr  = concat_parts('tr')
X_val, y_val, c_val, m_val = concat_parts('va')
X_te,  y_te,  c_te,  m_te  = concat_parts('te')

# Shuffle train set
rng  = np.random.default_rng(42)
perm = rng.permutation(len(X_tr))
X_tr, y_tr, c_tr, m_tr = X_tr[perm], y_tr[perm], c_tr[perm], m_tr[perm]

print(f"Windows → Train:{X_tr.shape}  Val:{X_val.shape}  Test:{X_te.shape}")
# Shape is (N, 30, 128, 2) — will be reshaped inside the model

def pack_reg(coords, mask):
    return np.concatenate([coords, mask[:, np.newaxis]], axis=-1).astype(np.float32)

reg_tr  = pack_reg(c_tr,  m_tr)
reg_val = pack_reg(c_val, m_val)
reg_te  = pack_reg(c_te,  m_te)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Model: Pure CNN — no LSTM
#
# Input window shape: (batch, 30, 128, 2)
#
# Reshape strategy:
#   (30, 128, 2)  →  (128, 60)
#    ↑ frames         ↑ subcarriers   ↑ 30 frames × 2 channels = 60 features
#
# The 1D-CNN slides over the 128-subcarrier axis.
# At each subcarrier position it sees 60 values — the amplitude and delta
# readings from all 30 frames simultaneously.
# This is equivalent to treating the 30-frame window as a 60-channel
# multi-measurement snapshot of the frequency response.
#
# Architecture:
#   Stem conv → ResBlock(64) + SE → ResBlock(128) + SE → ResBlock(256) + SE
#   → GlobalAveragePooling → Dense → Classification + Regression heads
# ─────────────────────────────────────────────────────────────────────────────

def se_block(x, ratio=8):
    """Squeeze-and-Excitation channel attention."""
    filters = x.shape[-1]
    sq = layers.GlobalAveragePooling1D()(x)
    sq = layers.Dense(max(filters // ratio, 1), activation='relu')(sq)
    sq = layers.Dense(filters, activation='sigmoid')(sq)
    sq = layers.Reshape((1, filters))(sq)
    return layers.Multiply()([x, sq])


def res_block(x, filters, kernel_size):
    """Residual block with optional 1×1 projection on the shortcut."""
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    return layers.ReLU()(layers.Add()([x, shortcut]))


def build_model(window_size=WINDOW_SIZE, num_classes=10):
    # ── Input & reshape ───────────────────────────────────────────────────────
    # Raw window: (batch, 30, 128, 2)
    inp = tf.keras.Input(shape=(window_size, 128, 2), name='window_input')

    # Permute → (batch, 128, 30, 2), then merge last two axes → (batch, 128, 60)
    # Result: CNN slides over 128 subcarriers; each position has 60 features
    #         (30 frames × 2 channels)
    x = layers.Permute((2, 1, 3))(inp)                    # (batch, 128, 30, 2)
    x = layers.Reshape((128, window_size * 2))(x)         # (batch, 128, 60)

    # ── Stem ──────────────────────────────────────────────────────────────────
    x = layers.Conv1D(64, kernel_size=7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)                         # → (64, 64)

    # ── Stage 1: ResBlock(128, k=5) + SE ──────────────────────────────────────
    x = res_block(x, 128, kernel_size=5)
    x = se_block(x)
    x = layers.MaxPooling1D(2)(x)                         # → (32, 128)

    # ── Stage 2: ResBlock(256, k=3) + SE ──────────────────────────────────────
    x = res_block(x, 256, kernel_size=3)
    x = se_block(x)
    x = layers.MaxPooling1D(2)(x)                         # → (16, 256)

    # ── Stage 3: ResBlock(512, k=3) + SE ──────────────────────────────────────
    x = res_block(x, 512, kernel_size=3)
    x = se_block(x)
    # No more pooling — sequence length is already 16; GAP handles the rest

    # ── Global pooling → shared dense ─────────────────────────────────────────
    x      = layers.GlobalAveragePooling1D()(x)           # → (512,)
    x      = layers.Dense(256, activation='relu')(x)
    x      = layers.Dropout(0.4)(x)
    shared = layers.Dense(128, activation='relu')(x)

    # ── Classification head ───────────────────────────────────────────────────
    cls_out = layers.Dense(num_classes, activation='softmax',
                           name='class_output')(shared)

    # ── Coordinate regression head ────────────────────────────────────────────
    reg     = layers.Dense(64, activation='relu')(shared)
    reg_out = layers.Dense(2, activation='sigmoid',
                           name='coord_output')(reg)

    return Model(inputs=inp, outputs=[cls_out, reg_out])


model = build_model()
model.summary()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Masked MSE loss
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
# 7. tf.data pipelines
# ─────────────────────────────────────────────────────────────────────────────
def augment(x, cls_lbl, reg_lbl):
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
# 9. Evaluate
# ─────────────────────────────────────────────────────────────────────────────
class_names = ["Empty"] + [f"Block {i}" for i in range(1, 10)]

preds       = model.predict(X_te, batch_size=BATCH_SIZE)
y_pred      = preds[0].argmax(axis=1)
coords_pred = preds[1]

print(f"\nTest accuracy: {(y_pred == y_te).mean():.4f}\n")
print(classification_report(y_te, y_pred, target_names=class_names))

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
ax.set_title('Confusion Matrix — v4 (CNN-only)')
thresh = cm.max() / 2
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')
plt.tight_layout()
plt.savefig('confusion_matrix_v4.png', dpi=150)
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
plt.savefig('training_curves_v4.png', dpi=150)
plt.show()

print("\nDone. Artefacts:")
print("  model/best_model.keras")
print("  model/final_model.keras")
print("  scaler/csi_scaler.pkl")
print("  scaler/empty_reference.pkl")
print("  confusion_matrix_v4.png")
print("  training_curves_v4.png")