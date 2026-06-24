import os
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# Create folders if they don't exist
os.makedirs("model", exist_ok=True)
os.makedirs("scaler", exist_ok=True)


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_csi_csv(filepath):
    data, labels = [], []
    with open(filepath) as f:
        for line in f:
            vals = list(map(int, line.strip().split(',')))
            label = vals[0]
            iq = np.array(vals[1:], dtype=float)
            I = iq[0::2]
            Q = iq[1::2]
            amplitude = np.sqrt(I**2 + Q**2)
            data.append(amplitude)
            labels.append(label)
    return np.array(data), np.array(labels)

X, y = load_csi_csv("data.csv")
print(X.shape, y.shape)


# ── 2. Normalize and split ────────────────────────────────────────────────────

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "scaler/csi_scaler.pkl")
print("Scaler saved to scaler/csi_scaler.pkl")

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

X_train = X_train[..., np.newaxis]
X_val   = X_val[..., np.newaxis]
X_test  = X_test[..., np.newaxis]

print(X_train.shape)


# ── 3. Build model ────────────────────────────────────────────────────────────

def build_model(num_classes=10):
    inputs = tf.keras.Input(shape=(128, 1))

    x = layers.Conv1D(32, kernel_size=7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

model = build_model(num_classes=10)
model.summary()


# ── 4. Train ──────────────────────────────────────────────────────────────────

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5,
        patience=5, verbose=1),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15,
        restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        'model/best_model.keras', monitor='val_accuracy',
        save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=64,
    callbacks=callbacks
)

# Save final model too (in case early stopping rolled back weights)
model.save("model/final_model.keras")
print("Model saved to model/final_model.keras")


# ── 5. Evaluate ───────────────────────────────────────────────────────────────

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

y_pred = model.predict(X_test).argmax(axis=1)
class_names = ["Empty"] + [f"Block {i}" for i in range(1, 10)]
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks(range(10), class_names, rotation=45, ha='right')
plt.yticks(range(10), class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                 color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy'); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss'); plt.legend()
plt.tight_layout()
plt.show()