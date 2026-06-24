"""
CSI Room Block Classifier — Prediction Script (v3)
═══════════════════════════════════════════════════
Supports three modes:

  1. SINGLE WINDOW  — predict one 30-frame window from a CSV snippet
  2. FULL FILE      — run a sliding window over an entire CSV file and
                      report per-window predictions + overall accuracy
                      if ground-truth labels are present
  3. LIVE BUFFER    — class you can import in your own code to feed
                      frames one at a time and get a prediction whenever
                      the buffer fills

Usage (command line):
  python predict_csi.py --mode single --file my_30_frames.csv
  python predict_csi.py --mode file   --file data.csv
  python predict_csi.py --mode file   --file data.csv --no-labels
"""

import argparse
import sys
import numpy as np
import joblib
import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (must match train_csi_v3.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE  = 30
MODEL_PATH   = "model/best_model.keras"
SCALER_PATH  = "scaler/csi_scaler.pkl"
EMPTY_REF_PATH = "scaler/empty_reference.pkl"

CLASS_NAMES = ["Empty"] + [f"Block {i}" for i in range(1, 10)]

# 3×3 grid visual layout for printing
GRID_LAYOUT = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────

def load_artefacts():
    """Load model, scaler, and empty-room reference once at startup."""
    print(f"Loading model      : {MODEL_PATH}")
    model     = tf.keras.models.load_model(MODEL_PATH)

    print(f"Loading scaler     : {SCALER_PATH}")
    scaler    = joblib.load(SCALER_PATH)

    print(f"Loading empty ref  : {EMPTY_REF_PATH}")
    empty_ref = joblib.load(EMPTY_REF_PATH)

    print("Artefacts loaded.\n")
    return model, scaler, empty_ref


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING  (mirrors train_csi_v3.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def iq_to_amplitude(iq_row: np.ndarray) -> np.ndarray:
    """
    Convert one row of 256 raw IQ values → 128 amplitude values.
    iq_row : 1-D array of length 256, alternating I0,Q0,I1,Q1,...
    """
    iq = iq_row.astype(np.float32)
    I  = iq[0::2]   # indices 0, 2, 4, ...
    Q  = iq[1::2]   # indices 1, 3, 5, ...
    return np.sqrt(I**2 + Q**2)   # shape (128,)


def preprocess_window(frames_amp: np.ndarray,
                      empty_ref:  np.ndarray,
                      scaler) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to one window of amplitude frames.

    frames_amp : (WINDOW_SIZE, 128)  — raw amplitudes, one row per frame
    Returns    : (1, WINDOW_SIZE, 128, 2)  — ready for model.predict()
    """
    # 1. Empty-room reference subtraction → delta channel
    delta = frames_amp - empty_ref                           # (W, 128)

    # 2. Stack into 2-channel tensor
    x2 = np.stack([frames_amp, delta], axis=-1)             # (W, 128, 2)

    # 3. Flatten → scale → reshape  (scaler was fitted on flattened 256-d vectors)
    x_flat   = x2.reshape(WINDOW_SIZE, 256)
    x_scaled = scaler.transform(x_flat).reshape(WINDOW_SIZE, 128, 2)
    x_scaled = x_scaled.astype(np.float32)

    # 4. Add batch dimension
    return x_scaled[np.newaxis]                             # (1, W, 128, 2)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def predict_window(model, x_input: np.ndarray):
    """
    Run model on a preprocessed (1, WINDOW_SIZE, 128, 2) tensor.
    Returns:
      predicted_class  : int
      class_name       : str
      confidence       : float  (0–1)
      probabilities    : np.ndarray shape (10,)
      coord            : np.ndarray shape (2,)  — (row, col) in [0,1]
    """
    cls_probs, coords = model.predict(x_input, verbose=0)
    cls_probs = cls_probs[0]       # shape (10,)
    coord     = coords[0]          # shape (2,)

    predicted_class = int(cls_probs.argmax())
    confidence      = float(cls_probs[predicted_class])

    return predicted_class, CLASS_NAMES[predicted_class], confidence, cls_probs, coord


def print_result(predicted_class, class_name, confidence, probabilities, coord,
                 true_label=None):
    """Pretty-print a single prediction with the room grid."""
    print("─" * 50)

    if true_label is not None:
        correct = "✓" if predicted_class == true_label else "✗"
        print(f"  True label  : {CLASS_NAMES[true_label]} (class {true_label})")
        print(f"  Prediction  : {class_name} (class {predicted_class})  {correct}")
    else:
        print(f"  Prediction  : {class_name} (class {predicted_class})")

    print(f"  Confidence  : {confidence * 100:.1f}%")

    if predicted_class != 0:
        print(f"  Est. coords : row={coord[0]*2:.2f}  col={coord[1]*2:.2f}  "
              f"(grid index, 0–2)")

    # ── Room grid visualisation ──────────────────────────────────────────────
    print()
    print("  Room grid:")
    print("  ┌─────────┬─────────┬─────────┐")
    for r, row in enumerate(GRID_LAYOUT):
        cells = []
        for block in row:
            if block == predicted_class:
                cells.append(f" ▓▓▓B{block}▓▓▓ ")
            else:
                cells.append(f"  Block {block}  ")
        print("  │" + "│".join(cells) + "│")
        if r < 2:
            print("  ├─────────┼─────────┼─────────┤")
    print("  └─────────┴─────────┴─────────┘")

    # ── Per-class probability bar chart ─────────────────────────────────────
    print()
    print("  Class probabilities:")
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
        bar    = "█" * int(prob * 30)
        marker = " ◄" if i == predicted_class else ""
        print(f"    {name:<8} {prob:5.1%}  {bar}{marker}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CSV READING
# ─────────────────────────────────────────────────────────────────────────────

def read_csv(filepath, has_labels=True):
    """
    Read a CSI CSV file.
    If has_labels=True  → first column is the class label, remaining 256 are IQ.
    If has_labels=False → all 256 columns are IQ (no labels).

    Returns:
      amplitudes : (N, 128)  float32
      labels     : (N,)      int32   or  None
    """
    amplitudes, labels = [], []

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            vals = list(map(int, line.split(',')))

            if has_labels:
                if len(vals) != 257:
                    print(f"  Warning: line {line_num} has {len(vals)} values "
                          f"(expected 257). Skipping.")
                    continue
                labels.append(vals[0])
                iq = np.array(vals[1:], dtype=np.float32)
            else:
                if len(vals) != 256:
                    print(f"  Warning: line {line_num} has {len(vals)} values "
                          f"(expected 256). Skipping.")
                    continue
                iq = np.array(vals, dtype=np.float32)

            amplitudes.append(iq_to_amplitude(iq))

    amp_array = np.array(amplitudes, dtype=np.float32)
    lbl_array = np.array(labels, dtype=np.int32) if has_labels else None

    print(f"  Read {len(amp_array)} frames from '{filepath}'")
    return amp_array, lbl_array


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — SINGLE WINDOW
# Predict a single 30-frame window.
# Input CSV must have exactly 30 rows.
# Each row: label(optional), I1, Q1, ..., I128, Q128
# ─────────────────────────────────────────────────────────────────────────────

def mode_single(filepath, model, scaler, empty_ref, has_labels=True):
    print(f"\n── Single-window prediction ──────────────────────────────────")
    amplitudes, labels = read_csv(filepath, has_labels)

    if len(amplitudes) < WINDOW_SIZE:
        print(f"ERROR: file has only {len(amplitudes)} frames; "
              f"need {WINDOW_SIZE}. Exiting.")
        sys.exit(1)

    if len(amplitudes) > WINDOW_SIZE:
        print(f"  Note: file has {len(amplitudes)} frames; "
              f"using first {WINDOW_SIZE}.")

    window = amplitudes[:WINDOW_SIZE]
    x      = preprocess_window(window, empty_ref, scaler)

    pred_class, pred_name, conf, probs, coord = predict_window(model, x)

    true_label = int(labels[0]) if has_labels and labels is not None else None
    print_result(pred_class, pred_name, conf, probs, coord, true_label)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — FULL FILE
# Slide a window over the entire file with STRIDE=1 (every possible position).
# Reports each prediction and final accuracy if labels are present.
# ─────────────────────────────────────────────────────────────────────────────

def mode_file(filepath, model, scaler, empty_ref, has_labels=True,
              stride=1, verbose=True):
    print(f"\n── Full-file prediction ──────────────────────────────────────")
    amplitudes, labels = read_csv(filepath, has_labels)

    n_frames  = len(amplitudes)
    positions = list(range(0, n_frames - WINDOW_SIZE + 1, stride))
    n_windows = len(positions)

    if n_windows == 0:
        print(f"ERROR: file has only {n_frames} frames; "
              f"need at least {WINDOW_SIZE}. Exiting.")
        sys.exit(1)

    print(f"  Frames: {n_frames}  |  Windows: {n_windows}  |  Stride: {stride}\n")

    predictions = []
    true_labels = []
    confidences = []

    for w_idx, start in enumerate(positions):
        window     = amplitudes[start : start + WINDOW_SIZE]
        x          = preprocess_window(window, empty_ref, scaler)
        pred_class, pred_name, conf, probs, coord = predict_window(model, x)

        # True label for this window = label of the middle frame
        mid_frame  = start + WINDOW_SIZE // 2
        true_label = int(labels[mid_frame]) if has_labels else None

        predictions.append(pred_class)
        confidences.append(conf)
        if has_labels:
            true_labels.append(true_label)

        if verbose:
            correct = ""
            if has_labels:
                correct = "✓" if pred_class == true_label else "✗"
            print(f"  Window {w_idx+1:>4}/{n_windows}  "
                  f"frames [{start:>4}–{start+WINDOW_SIZE-1:>4}]  "
                  f"→  {pred_name:<8}  ({conf*100:5.1f}%)  {correct}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("  SUMMARY")
    print("═" * 50)
    print(f"  Total windows  : {n_windows}")
    print(f"  Mean confidence: {np.mean(confidences)*100:.1f}%")

    if has_labels and true_labels:
        predictions  = np.array(predictions)
        true_arr     = np.array(true_labels)
        accuracy     = (predictions == true_arr).mean()
        print(f"  Accuracy       : {accuracy*100:.2f}%\n")

        # Per-class breakdown
        print(f"  {'Class':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
        print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*10}")
        for cls in range(10):
            mask    = true_arr == cls
            if mask.sum() == 0:
                continue
            correct = (predictions[mask] == cls).sum()
            total   = mask.sum()
            acc     = correct / total
            print(f"  {CLASS_NAMES[cls]:<10} {correct:>8} {total:>8} {acc:>10.1%}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 — LIVE BUFFER (importable class)
# Feed one frame at a time. A prediction is returned whenever the buffer
# contains WINDOW_SIZE frames (i.e. every STRIDE frames after the first fill).
#
# Usage:
#   from predict_csi import LivePredictor
#   predictor = LivePredictor()
#   for packet in csi_stream:
#       result = predictor.push(packet)   # None until buffer fills
#       if result:
#           print(result['class_name'], result['confidence'])
# ─────────────────────────────────────────────────────────────────────────────

class LivePredictor:
    """
    Feed raw CSI IQ rows one at a time.
    Returns a prediction dict every STRIDE frames once the buffer is full.
    Returns None otherwise.

    Parameters
    ----------
    stride : int
        How many new frames between predictions (default 5, matches training).
    model_path, scaler_path, empty_ref_path : str
        Paths to saved artefacts.
    """

    def __init__(self,
                 stride         = 5,
                 model_path     = MODEL_PATH,
                 scaler_path    = SCALER_PATH,
                 empty_ref_path = EMPTY_REF_PATH):

        self.stride        = stride
        self.model, self.scaler, self.empty_ref = load_artefacts()
        self._buffer       = []          # list of amplitude arrays (128,)
        self._frames_since = 0           # frames since last prediction

    def push(self, iq_row) -> dict | None:
        """
        Push one raw IQ row (length 256, int or float).
        Returns a prediction dict when ready, otherwise None.

        Dict keys:
          class_id    : int      (0–9)
          class_name  : str
          confidence  : float    (0–1)
          probabilities : np.ndarray (10,)
          coord       : np.ndarray (2,)   — normalised (row, col)
          coord_grid  : np.ndarray (2,)   — grid index (0–2)
        """
        amp = iq_to_amplitude(np.array(iq_row, dtype=np.float32))
        self._buffer.append(amp)

        # Keep buffer size bounded
        if len(self._buffer) > WINDOW_SIZE:
            self._buffer.pop(0)

        self._frames_since += 1

        # Emit a prediction when buffer is full AND stride is satisfied
        if (len(self._buffer) == WINDOW_SIZE and
                self._frames_since >= self.stride):

            self._frames_since = 0
            window  = np.stack(self._buffer, axis=0)   # (30, 128)
            x       = preprocess_window(window, self.empty_ref, self.scaler)
            cls_id, cls_name, conf, probs, coord = predict_window(self.model, x)

            return {
                'class_id'      : cls_id,
                'class_name'    : cls_name,
                'confidence'    : conf,
                'probabilities' : probs,
                'coord'         : coord,
                'coord_grid'    : coord * 2,   # scale to 0–2 grid index
            }

        return None

    def reset(self):
        """Clear the frame buffer (call when moving to a new measurement session)."""
        self._buffer       = []
        self._frames_since = 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CSI Room Block Classifier — Prediction Script (v3)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mode", choices=["single", "file"], default="file",
        help=(
            "single : predict one 30-frame window from a CSV\n"
            "file   : slide over an entire CSV and report all predictions"
        )
    )
    parser.add_argument(
        "--file", required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="CSV has no label column (pure IQ data, 256 values per row)"
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Window stride for 'file' mode (default: 1)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="In 'file' mode, suppress per-window output; show summary only"
    )
    parser.add_argument(
        "--model",     default=MODEL_PATH,     help="Path to .keras model file"
    )
    parser.add_argument(
        "--scaler",    default=SCALER_PATH,    help="Path to scaler .pkl"
    )
    parser.add_argument(
        "--empty-ref", default=EMPTY_REF_PATH, help="Path to empty_reference .pkl"
    )

    args = parser.parse_args()

    # Allow overriding artefact paths from CLI
    global MODEL_PATH, SCALER_PATH, EMPTY_REF_PATH
    MODEL_PATH     = args.model
    SCALER_PATH    = args.scaler
    EMPTY_REF_PATH = args.empty_ref

    model, scaler, empty_ref = load_artefacts()
    has_labels = not args.no_labels

    if args.mode == "single":
        mode_single(args.file, model, scaler, empty_ref, has_labels)

    elif args.mode == "file":
        mode_file(args.file, model, scaler, empty_ref, has_labels,
                  stride=args.stride, verbose=not args.quiet)


if __name__ == "__main__":
    main()
ENDOFFILE