"""
Class Uniqueness Analyzer
=========================
Loads 10 class files (class_0_detrended.txt ... class_9_detrended.txt)
and runs multiple algorithms to check if the 10 classes are unique/separable.

Each algorithm outputs:
  - A 10x10 pairwise score matrix (heatmap)
  - Pairwise scores for all 45 pairs
  - An overall separability verdict
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend — must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations
from scipy.stats import f_oneway, kruskal, entropy
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# ─────────────────────────────────────────────
#  Data Loader
# ─────────────────────────────────────────────

def load_data(folder_path):
    """Load all 10 class files. Returns list of 10 arrays (each 690,)."""
    data = []
    for i in range(10):
        fpath = os.path.join(folder_path, f"class_{i}_detrended.txt")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing: {fpath}")
        arr = np.loadtxt(fpath).flatten()
        if len(arr) != 690:
            print(f"  [WARN] class_{i} has {len(arr)} samples, expected 690")
        data.append(arr)
        print(f"  Loaded class_{i}: {len(arr)} samples | mean={arr.mean():.4f} std={arr.std():.4f}")
    return data


# ─────────────────────────────────────────────
#  Plotting Helper
# ─────────────────────────────────────────────

def plot_matrix(matrix, title, algo_name, cmap, out_dir, higher_is_better=True):
    """Plot a 10x10 pairwise score matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax, label='Score')
    ax.set_title(f"{title}\n({'higher = more unique' if higher_is_better else 'lower = more unique'})",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Class")
    ax.set_ylabel("Class")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([f"C{i}" for i in range(10)])
    ax.set_yticklabels([f"C{i}" for i in range(10)])

    # Annotate cells
    for i in range(10):
        for j in range(10):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    fontsize=7, color='white' if abs(val) > matrix.max() * 0.6 else 'black')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{algo_name}_matrix.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")


def print_verdict(scores_dict, algo_name, higher_is_better=True):
    """Print pairwise scores and overall verdict."""
    print(f"\n{'='*55}")
    print(f"  {algo_name}")
    print(f"{'='*55}")
    scores = list(scores_dict.values())
    mean_s = np.mean(scores)
    std_s  = np.std(scores)
    min_s  = np.min(scores)
    max_s  = np.max(scores)

    print(f"  Pairwise scores (45 pairs):")
    for (i, j), s in scores_dict.items():
        flag = ""
        if higher_is_better and s < mean_s - std_s:
            flag = "  ← LOW (similar)"
        elif not higher_is_better and s > mean_s + std_s:
            flag = "  ← HIGH (similar)"
        print(f"    Class {i} vs Class {j}: {s:.4f}{flag}")

    print(f"\n  Summary:")
    print(f"    Mean  : {mean_s:.4f}")
    print(f"    Std   : {std_s:.4f}")
    print(f"    Min   : {min_s:.4f}")
    print(f"    Max   : {max_s:.4f}")

    # Verdict
    cv = std_s / (abs(mean_s) + 1e-9)
    if cv > 0.3:
        verdict = "MIXED — some class pairs are unique, others are not"
    elif higher_is_better and mean_s > 0.5:
        verdict = "UNIQUE — classes are generally well separated"
    elif not higher_is_better and mean_s < 0.1:
        verdict = "UNIQUE — classes are generally well separated"
    else:
        verdict = "NOT UNIQUE — classes are similar / hard to distinguish"

    print(f"    Verdict: {verdict}")


# ─────────────────────────────────────────────
#  Algorithm 1: ANOVA F-statistic
# ─────────────────────────────────────────────

def algo_anova(data, out_dir):
    """
    One-way ANOVA between each class pair.
    High F → distributions have different means → classes are unique.
    """
    matrix = np.zeros((10, 10))
    scores = {}

    for i, j in combinations(range(10), 2):
        f, p = f_oneway(data[i], data[j])
        matrix[i, j] = f
        matrix[j, i] = f
        scores[(i, j)] = f

    plot_matrix(matrix, "ANOVA F-Statistic (pairwise)", "anova", "YlOrRd", out_dir, higher_is_better=True)
    print_verdict(scores, "ANOVA F-Statistic", higher_is_better=True)
    return scores


# ─────────────────────────────────────────────
#  Algorithm 2: Kruskal-Wallis (non-parametric)
# ─────────────────────────────────────────────

def algo_kruskal(data, out_dir):
    """
    Kruskal-Wallis H test between each class pair.
    Non-parametric version of ANOVA — no normality assumption.
    High H → classes differ significantly.
    """
    matrix = np.zeros((10, 10))
    scores = {}

    for i, j in combinations(range(10), 2):
        h, p = kruskal(data[i], data[j])
        matrix[i, j] = h
        matrix[j, i] = h
        scores[(i, j)] = h

    plot_matrix(matrix, "Kruskal-Wallis H Statistic (pairwise)", "kruskal", "Blues", out_dir, higher_is_better=True)
    print_verdict(scores, "Kruskal-Wallis H", higher_is_better=True)
    return scores


# ─────────────────────────────────────────────
#  Algorithm 3: Jensen-Shannon Divergence
# ─────────────────────────────────────────────

def algo_jsd(data, out_dir, n_bins=50):
    """
    Jensen-Shannon Divergence between class distribution pairs.
    Measures how different the probability distributions are.
    High JSD (max 1.0) → very different distributions → unique classes.
    """
    matrix = np.zeros((10, 10))
    scores = {}

    # Build histogram (PDF) for each class over shared bin range
    all_vals = np.concatenate(data)
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

    pdfs = []
    for arr in data:
        hist, _ = np.histogram(arr, bins=bin_edges, density=True)
        hist = hist + 1e-10  # smooth to avoid log(0)
        hist /= hist.sum()
        pdfs.append(hist)

    for i, j in combinations(range(10), 2):
        jsd = jensenshannon(pdfs[i], pdfs[j])  # 0 to 1
        matrix[i, j] = jsd
        matrix[j, i] = jsd
        scores[(i, j)] = jsd

    plot_matrix(matrix, "Jensen-Shannon Divergence (pairwise)", "jsd", "Greens", out_dir, higher_is_better=True)
    print_verdict(scores, "Jensen-Shannon Divergence", higher_is_better=True)
    return scores


# ─────────────────────────────────────────────
#  Algorithm 4: Random Forest Feature Importance
# ─────────────────────────────────────────────

def algo_random_forest(data, out_dir):
    """
    Train a Random Forest classifier on all 10 classes.
    Pairwise score = cross-val accuracy between each class pair.
    High accuracy → classes are separable → unique.
    """
    matrix = np.zeros((10, 10))
    np.fill_diagonal(matrix, 1.0)
    scores = {}

    for i, j in combinations(range(10), 2):
        X = np.concatenate([data[i], data[j]]).reshape(-1, 1)
        y = np.array([0] * len(data[i]) + [1] * len(data[j]))
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
        acc = cv_scores.mean()
        matrix[i, j] = acc
        matrix[j, i] = acc
        scores[(i, j)] = acc

    plot_matrix(matrix, "Random Forest Pairwise Accuracy", "rf", "Purples", out_dir, higher_is_better=True)
    print_verdict(scores, "Random Forest Accuracy", higher_is_better=True)
    return scores


# ─────────────────────────────────────────────
#  Algorithm 5: Mutual Information
# ─────────────────────────────────────────────

def algo_mutual_info(data, out_dir):
    """
    Mutual Information between class label and signal values.
    Computed per class pair: how much does the signal tell us about which class it is?
    High MI → signal is informative → classes are unique.
    """
    matrix = np.zeros((10, 10))
    scores = {}

    for i, j in combinations(range(10), 2):
        X = np.concatenate([data[i], data[j]]).reshape(-1, 1)
        y = np.array([0] * len(data[i]) + [1] * len(data[j]))
        mi = mutual_info_classif(X, y, random_state=42)[0]
        matrix[i, j] = mi
        matrix[j, i] = mi
        scores[(i, j)] = mi

    plot_matrix(matrix, "Mutual Information (pairwise)", "mi", "Oranges", out_dir, higher_is_better=True)
    print_verdict(scores, "Mutual Information", higher_is_better=True)
    return scores


# ─────────────────────────────────────────────
#  Algorithm 6: LDA Separability Score
# ─────────────────────────────────────────────

def algo_lda(data, out_dir):
    """
    Linear Discriminant Analysis separability between each class pair.
    Score = LDA cross-val accuracy.
    High accuracy → linearly separable → unique classes.
    """
    matrix = np.zeros((10, 10))
    np.fill_diagonal(matrix, 1.0)
    scores = {}

    for i, j in combinations(range(10), 2):
        X = np.concatenate([data[i], data[j]]).reshape(-1, 1)
        y = np.array([0] * len(data[i]) + [1] * len(data[j]))
        lda = LinearDiscriminantAnalysis()
        cv_scores = cross_val_score(lda, X, y, cv=5, scoring='accuracy')
        acc = cv_scores.mean()
        matrix[i, j] = acc
        matrix[j, i] = acc
        scores[(i, j)] = acc

    plot_matrix(matrix, "LDA Pairwise Accuracy", "lda", "RdPu", out_dir, higher_is_better=True)
    print_verdict(scores, "LDA Separability", higher_is_better=True)
    return scores


# ─────────────────────────────────────────────
#  Summary Plot: All algorithms together
# ─────────────────────────────────────────────

def plot_summary(all_scores, out_dir):
    """
    Bar chart: for each of the 45 pairs, show scores from all algorithms (normalized).
    """
    pairs = [(i, j) for i, j in combinations(range(10), 2)]
    pair_labels = [f"C{i}-C{j}" for i, j in pairs]
    algo_names = list(all_scores.keys())
    n_algos = len(algo_names)

    # Normalize each algo's scores to [0, 1]
    norm_scores = {}
    for algo, scores in all_scores.items():
        vals = np.array([scores[(i, j)] for i, j in pairs])
        mn, mx = vals.min(), vals.max()
        norm_scores[algo] = (vals - mn) / (mx - mn + 1e-9)

    fig, ax = plt.subplots(figsize=(20, 6))
    x = np.arange(len(pairs))
    width = 0.8 / n_algos

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    for idx, algo in enumerate(algo_names):
        offset = (idx - n_algos / 2) * width + width / 2
        ax.bar(x + offset, norm_scores[algo], width, label=algo, color=colors[idx % len(colors)], alpha=0.85)

    ax.set_xlabel("Class Pair", fontsize=11)
    ax.set_ylabel("Normalized Score (0=similar, 1=unique)", fontsize=11)
    ax.set_title("All Algorithm Scores per Class Pair (Normalized)", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=90, fontsize=7)
    ax.legend(loc='upper right')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='threshold')
    plt.tight_layout()
    out_path = os.path.join(out_dir, "summary_all_algorithms.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n  Summary plot saved: {out_path}")


def plot_consensus(all_scores, out_dir):
    """
    10x10 consensus heatmap: average normalized score across all algorithms.
    """
    pairs = [(i, j) for i, j in combinations(range(10), 2)]
    matrix = np.zeros((10, 10))

    # Normalize and average
    norm_matrices = []
    for algo, scores in all_scores.items():
        vals = np.array([scores[(i, j)] for i, j in pairs])
        mn, mx = vals.min(), vals.max()
        norm_vals = (vals - mn) / (mx - mn + 1e-9)
        m = np.zeros((10, 10))
        for idx, (i, j) in enumerate(pairs):
            m[i, j] = norm_vals[idx]
            m[j, i] = norm_vals[idx]
        norm_matrices.append(m)

    matrix = np.mean(norm_matrices, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Consensus Score (0=similar, 1=unique)')
    ax.set_title("Consensus Uniqueness Matrix\n(Average across all algorithms)", fontsize=13, fontweight='bold')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([f"C{i}" for i in range(10)])
    ax.set_yticklabels([f"C{i}" for i in range(10)])
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha='center', va='center',
                    fontsize=7.5, color='black')
    plt.tight_layout()
    out_path = os.path.join(out_dir, "consensus_matrix.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Consensus matrix saved: {out_path}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main(i):
    # if len(sys.argv) < 2:
    #     print("Usage: python class_uniqueness.py <folder_path>")
    #     print("Example: python class_uniqueness.py ./data/subcarrier_10")
    #     sys.exit(1)

    folder_path = folder_path = f'./experiment/phase/detrended/sc_{i}'
    out_dir = os.path.join(folder_path, "uniqueness_results")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Class Uniqueness Analyzer")
    print(f"  Folder : {folder_path}")
    print(f"  Output : {out_dir}")
    print(f"{'='*55}\n")

    # Load data
    print("[1/7] Loading data...")
    data = load_data(folder_path)

    all_scores = {}

    print("\n[2/7] Running ANOVA F-statistic...")
    all_scores['ANOVA'] = algo_anova(data, out_dir)

    print("\n[3/7] Running Kruskal-Wallis...")
    all_scores['Kruskal'] = algo_kruskal(data, out_dir)

    print("\n[4/7] Running Jensen-Shannon Divergence...")
    all_scores['JSD'] = algo_jsd(data, out_dir)

    # print("\n[5/7] Running Random Forest accuracy...")
    # all_scores['RandomForest'] = algo_random_forest(data, out_dir)

    print("\n[6/7] Running Mutual Information...")
    all_scores['MutualInfo'] = algo_mutual_info(data, out_dir)

    print("\n[7/7] Running LDA separability...")
    all_scores['LDA'] = algo_lda(data, out_dir)

    # Summary plots
    # print("\n[Summary] Generating summary plots...")
    # plot_summary(all_scores, out_dir)
    # plot_consensus(all_scores, out_dir)

    print(f"\n{'='*55}")
    print(f"  All done. Results saved to: {out_dir}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    for i in range(1, 129):
        main(i)