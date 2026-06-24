import numpy as np
import tensorflow as tf
import joblib

# ── Load saved model and scaler ───────────────────────────────────────────────

model  = tf.keras.models.load_model("model/best_model.keras")
scaler = joblib.load("scaler/csi_scaler.pkl")

CLASS_NAMES = ["Empty"] + [f"Block {i}" for i in range(1, 10)]


# ── Helper: parse one CSV line into amplitude ─────────────────────────────────

def parse_line(line: str, has_label: bool = True):
    """
    has_label=True  → line format: label,I1,Q1,I2,Q2,...  (like your training data)
    has_label=False → line format: I1,Q1,I2,Q2,...         (raw, no label)
    """
    vals = list(map(int, line.strip().split(',')))
    iq = np.array(vals[1:] if has_label else vals, dtype=float)
    I, Q = iq[0::2], iq[1::2]
    return np.sqrt(I**2 + Q**2), (vals[0] if has_label else None)


def predict_amplitude(amplitude: np.ndarray):
    x = scaler.transform(amplitude[np.newaxis, :])   # normalize
    x = x[..., np.newaxis]                           # (1, 128, 1)
    probs = model.predict(x, verbose=0)[0]
    pred  = probs.argmax()
    return CLASS_NAMES[pred], probs


# ── Option 1: predict a single raw line ──────────────────────────────────────

def predict_line(line: str, has_label: bool = True):
    amplitude, true_label = parse_line(line, has_label)
    name, probs = predict_amplitude(amplitude)
    print(f"Predicted : {name}  (confidence: {probs.max()*100:.1f}%)")
    if true_label is not None:
        print(f"True label: {CLASS_NAMES[true_label]}")
    return name


# ── Option 2: predict an entire CSV file and show accuracy ───────────────────

def predict_file(filepath: str, has_label: bool = True):
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt

    all_preds, all_true = [], []

    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            amplitude, true_label = parse_line(line, has_label)
            name, probs = predict_amplitude(amplitude)
            all_preds.append(probs.argmax())
            if true_label is not None:
                all_true.append(true_label)

    print(f"\nTotal samples predicted: {len(all_preds)}")

    if all_true:
        correct = sum(p == t for p, t in zip(all_preds, all_true))
        print(f"Accuracy: {correct}/{len(all_true)} = {correct/len(all_true)*100:.2f}%\n")
        print(classification_report(all_true, all_preds, target_names=CLASS_NAMES))

        cm = confusion_matrix(all_true, all_preds)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        plt.xticks(range(10), CLASS_NAMES, rotation=45, ha='right')
        plt.yticks(range(10), CLASS_NAMES)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        for i in range(10):
            for j in range(10):
                plt.text(j, i, cm[i, j], ha='center', va='center',
                         color='white' if cm[i, j] > cm.max()/2 else 'black')
        plt.tight_layout()
        plt.savefig('confusion_matrix_inference.png', dpi=150)
        plt.show()

    return all_preds


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # --- Single line prediction ---
    sample = "3,94,96,5,0,14,-1,14,-2,13,-1,11,-1,9,0,8,-1,7,-3,6,-5,4,-6,3,-8,3,-9,3,-11,2,-11,2,-12,1,-12,0,-12,0,-13,1,-15,2,-16,2,-15,2,-13,2,-11,3,-12,4,-12,4,-12,1,-3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,23,0,24,0,26,1,27,2,26,3,26,5,27,6,28,7,28,6,30,6,30,7,28,9,27,11,28,11,27,9,27,8,26,8,25,8,24,8,24,6,23,6,22,7,21,7,21,5,20,3,20,2,8,0,29,1,27,1,25,0,23,-1,20,-3,17,-4,15,-6,12,-8,10,-10,7,-11,5,-14,4,-16,4,-19,3,-20,2,-21,0,-22,-1,-23,-1,-24,0,-25,1,-26,2,-26,2,-25,2,-24,4,-23,6,-23,6,-22,6,-21,4,-21,1,-4,0,-2,-1,0,-2,0,-1,0,2,-1,6,-2,39,-10,40,-8,44,-3,46,-2,48,0,50,2,51,4,52,6,53,7,55,7,56,7,58,8,57,11,56,13,55,15,54,15,53,15,53,14,52,14,50,13,48,12,47,9,45,8,44,9,43,8,41,6,38,4,37,3"
    predict_line(sample, has_label=True)

    print("\n" + "─"*50 + "\n")

    # --- Full file prediction ---
    predict_file("new_data.csv", has_label=True)
    # set has_label=False if your new file has no label column