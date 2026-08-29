"""
Phase 6.5 — Evaluate the Baseline model
=======================================

Runs inference on test.csv once and writes:

    results/baseline_metrics.json          accuracy, macro/weighted P/R/F1,
                                           per-class metrics, confusion matrix
    results/baseline_predictions.csv       per-image true/predicted + confidence
    results/baseline_confusion_matrix.png   6x6 heatmap

Also breaks results down by illumination (Low/Medium/High), because SOP1
and SOP2 ask specifically about performance under varying lighting. An
overall accuracy figure alone cannot answer them.

Undertone is NOT scored. The methodology excludes it from the metrics —
there's no validated undertone ground truth in this dataset, so reporting
an accuracy for it would imply a reference that doesn't exist.

    python src/baseline/evaluate.py
    python src/baseline/evaluate.py --model models/baseline_stage1_best.keras

RUN THIS ONCE, when you have stopped changing the model. Every look at
the test set costs a little of its independence, and the whole point of
the frozen split is that Baseline and HueView face the same unseen data.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

try:
    from data_pipeline import load_split, make_dataset, SCC_CLASSES, NUM_CLASSES
except ImportError:
    from src.baseline.data_pipeline import load_split, make_dataset, SCC_CLASSES, NUM_CLASSES

PROC = Path("data/processed")
RESULTS = Path("results")
MODELS = Path("models")


def confusion_matrix(y_true, y_pred, n=NUM_CLASSES):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_metrics(cm):
    """Precision, recall, F1 and support per class, computed from the matrix."""
    out = {}
    for i, cls in enumerate(SCC_CLASSES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        out[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }
    return out


def aggregate(per_class, cm):
    """Macro (unweighted) and weighted averages."""
    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total else 0.0

    macro = {
        m: round(float(np.mean([per_class[c][m] for c in SCC_CLASSES])), 4)
        for m in ("precision", "recall", "f1")
    }
    weighted = {}
    for m in ("precision", "recall", "f1"):
        weighted[m] = round(float(
            sum(per_class[c][m] * per_class[c]["support"] for c in SCC_CLASSES) / total
        ), 4) if total else 0.0

    return {"accuracy": round(accuracy, 4), "macro": macro, "weighted": weighted}


def plot_confusion(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Row-normalized: with 6:1 class imbalance, raw counts make the majority
    # class dominate visually and hide how the small classes actually did.
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(NUM_CLASSES), SCC_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES), SCC_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Baseline — SCC confusion matrix\n(cell = count, shade = row proportion)")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm[i, j]}\n{100*norm[i, j]:.0f}%",
                    ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")

    fig.colorbar(im, ax=ax, label="proportion of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def illumination_breakdown(test_df, y_true, y_pred):
    """
    Per-illumination accuracy — the evidence SOP1 and SOP2 need.

    Returns None if the manifest has no illumination column.
    """
    manifest = pd.read_csv(PROC / "manifest.csv")
    col = next((c for c in manifest.columns
                if "illum" in c.lower()), None)
    if col is None:
        return None

    lookup = dict(zip(manifest["filename"], manifest[col]))
    illum = test_df["filename"].map(lookup)

    out = {}
    correct = (y_true == y_pred)
    for level in sorted(set(illum.dropna())):
        mask = (illum == level).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        sub_cm = confusion_matrix(y_true[mask], y_pred[mask])
        sub_pc = per_class_metrics(sub_cm)
        out[str(level)] = {
            "n": n,
            "accuracy": round(float(correct[mask].mean()), 4),
            "macro_f1": round(float(np.mean([sub_pc[c]["f1"] for c in SCC_CLASSES])), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="Path to the model. Defaults to models/baseline_effnet.keras")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    model_path = Path(args.model) if args.model else MODELS / "baseline_effnet.keras"
    if not model_path.exists():
        alt = MODELS / "baseline_effnet.h5"
        if alt.exists():
            model_path = alt
        else:
            print(f"!! No model at {model_path}. Run train.py first.")
            return

    RESULTS.mkdir(exist_ok=True)

    print("=" * 68)
    print("PHASE 6.5 — BASELINE EVALUATION")
    print("=" * 68)
    print(f"\nModel: {model_path}")

    model = keras.models.load_model(model_path)
    test_df = load_split("test")
    print(f"Test set: {len(test_df)} images\n")

    test_ds = make_dataset(test_df, batch_size=args.batch_size)
    probs = model.predict(test_ds, verbose=1)

    y_true = test_df["label_index"].to_numpy()
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    cm = confusion_matrix(y_true, y_pred)
    per_class = per_class_metrics(cm)
    agg = aggregate(per_class, cm)

    # ---------------- Report ----------------
    print("\n" + "-" * 68)
    print("OVERALL")
    print("-" * 68)
    print(f"\n  Accuracy           {agg['accuracy']:.4f}")
    print(f"  Macro    P/R/F1    {agg['macro']['precision']:.4f} / "
          f"{agg['macro']['recall']:.4f} / {agg['macro']['f1']:.4f}")
    print(f"  Weighted P/R/F1    {agg['weighted']['precision']:.4f} / "
          f"{agg['weighted']['recall']:.4f} / {agg['weighted']['f1']:.4f}")

    if agg["weighted"]["f1"] - agg["macro"]["f1"] > 0.10:
        print("\n  Weighted F1 exceeds macro F1 by a wide margin, which means the")
        print("  model does noticeably better on the larger classes. Lead with the")
        print("  macro figure — it's the honest summary when classes are imbalanced.")

    print("\n" + "-" * 68)
    print("PER CLASS")
    print("-" * 68)
    print(f"\n  {'class':<8} {'prec':>7} {'recall':>8} {'f1':>7} {'support':>8}")
    for cls in SCC_CLASSES:
        m = per_class[cls]
        print(f"  {cls:<8} {m['precision']:>7.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>7.4f} {m['support']:>8}")

    weak = [c for c in SCC_CLASSES if per_class[c]["recall"] < 0.5
            and per_class[c]["support"] > 0]
    if weak:
        print(f"\n  Recall below 0.50 for: {', '.join(weak)}")
        print("  The model is missing most true examples of these classes.")
        print("  Report this per-class rather than only the overall accuracy.")

    print("\n" + "-" * 68)
    print("CONFUSION MATRIX  (rows = true, columns = predicted)")
    print("-" * 68 + "\n")
    print("           " + "".join(f"{c:>9}" for c in SCC_CLASSES))
    for i, cls in enumerate(SCC_CLASSES):
        print(f"  {cls:<8} " + "".join(f"{cm[i, j]:>9}" for j in range(NUM_CLASSES)))

    # Adjacent-class errors are worth separating out: SCC is an ordinal
    # scale, so confusing SCC-3 with SCC-4 is a different kind of mistake
    # than confusing SCC-1 with SCC-6.
    off = cm.sum() - np.trace(cm)
    adjacent = sum(cm[i, j] for i in range(NUM_CLASSES) for j in range(NUM_CLASSES)
                   if abs(i - j) == 1)
    if off:
        print(f"\n  {adjacent} of {off} errors ({100*adjacent/off:.0f}%) are to an")
        print("  adjacent SCC class. On an ordinal scale those are near-misses,")
        print("  and worth distinguishing from distant confusions in the writeup.")

    illum = illumination_breakdown(test_df, y_true, y_pred)
    if illum:
        print("\n" + "-" * 68)
        print("BY ILLUMINATION  (evidence for SOP1 / SOP2)")
        print("-" * 68)
        print(f"\n  {'level':<12} {'n':>7} {'accuracy':>10} {'macro F1':>10}")
        for level, m in illum.items():
            print(f"  {level:<12} {m['n']:>7} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}")
        accs = [m["accuracy"] for m in illum.values()]
        if accs and max(accs) - min(accs) > 0.05:
            print(f"\n  Spread of {max(accs)-min(accs):.3f} between best and worst")
            print("  lighting condition. This is the number SOP1/SOP2 turn on, and")
            print("  the comparison HueView needs to improve on.")

    # ---------------- Write outputs ----------------
    preds = pd.DataFrame({
        "filename": test_df["filename"],
        "true": [SCC_CLASSES[i] for i in y_true],
        "predicted": [SCC_CLASSES[i] for i in y_pred],
        "correct": y_true == y_pred,
        "confidence": confidence.round(4),
    })
    for i, cls in enumerate(SCC_CLASSES):
        preds[f"p_{cls}"] = probs[:, i].round(4)
    preds.to_csv(RESULTS / "baseline_predictions.csv", index=False)

    metrics = {
        "model": "Baseline (EfficientNetB0 + global RGB)",
        "evaluated": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(model_path),
        "test_images": int(len(test_df)),
        "classes": SCC_CLASSES,
        "overall": agg,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": SCC_CLASSES,
        "by_illumination": illum,
        "adjacent_error_rate": round(float(adjacent / off), 4) if off else None,
        "note": ("Undertone is excluded from these metrics per the methodology — "
                 "the dataset carries no validated undertone ground truth."),
    }
    with open(RESULTS / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        plot_confusion(cm, RESULTS / "baseline_confusion_matrix.png")
        plotted = True
    except ImportError:
        plotted = False

    print("\n" + "=" * 68)
    print("WRITTEN")
    print("=" * 68)
    print(f"\n  {RESULTS / 'baseline_metrics.json'}")
    print(f"  {RESULTS / 'baseline_predictions.csv'}")
    if plotted:
        print(f"  {RESULTS / 'baseline_confusion_matrix.png'}")
    else:
        print("  (matplotlib missing — no confusion matrix image)")

    print("""
  These are the Baseline numbers HueView must beat. Both models must be
  evaluated on this same test set, unchanged, or the comparison in Phase
  12 measures nothing.""")


if __name__ == "__main__":
    main()
