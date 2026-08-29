"""
Phase 6.4 — Train the Baseline model
====================================

Two-stage transfer learning:

    Stage 1  backbone frozen, train the classifier head    (lr 1e-3)
    Stage 2  unfreeze, fine-tune the whole network         (lr 1e-5)

Stage 1 first because a randomly-initialised head produces large
gradients, and letting those flow into pretrained ImageNet weights
destroys them in the first few steps. Train the head until it's sane,
then fine-tune gently.

RUN THIS ON COLAB WITH A GPU. TensorFlow dropped native Windows GPU
support at 2.11, so on Windows this runs on CPU regardless of hardware —
43k images through EfficientNetB0 would take days. See the Colab notes
at the bottom of this file.

Everything is checkpointed. If the session dies mid-run, rerun and it
picks up from the last saved epoch rather than starting over.

    python src/baseline/train.py
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    from data_pipeline import build_datasets, SCC_CLASSES
    from model import build_baseline_model, unfreeze_for_finetuning
except ImportError:
    from src.baseline.data_pipeline import build_datasets, SCC_CLASSES
    from src.baseline.model import build_baseline_model, unfreeze_for_finetuning

MODELS = Path("models")
LOGS = Path("logs")

STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 20
STAGE1_LR = 1e-3
STAGE2_LR = 1e-5
PATIENCE = 5


def callbacks_for(stage: str, monitor: str = "val_loss"):
    """
    Checkpoint + early stopping + LR reduction.

    Monitors val_loss, as the methodology specifies. Loss is the more
    sensitive signal on imbalanced data: accuracy can sit flat for several
    epochs while the model is still improving its confidence, and with
    SCC-5 at ~6x SCC-6 a model can post decent accuracy by leaning on the
    majority class.

    save_best_only means an overfitting tail can't overwrite a good
    checkpoint. restore_best_weights means the model you keep is the best
    one seen, not whatever the last epoch produced.
    """
    MODELS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS / f"baseline_{stage}_best.keras"),
            monitor=monitor, mode="min",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor, mode="min",
            patience=PATIENCE, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=0.5, patience=3, min_lr=1e-7, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(LOGS / f"baseline_{stage}.csv"), append=True),
    ]


def report_environment():
    gpus = tf.config.list_physical_devices("GPU")
    print("=" * 68)
    print("PHASE 6.4 — BASELINE TRAINING")
    print("=" * 68)
    print(f"\nTensorFlow {tf.__version__}")
    if gpus:
        print(f"GPU: {len(gpus)} device(s) — {[g.name for g in gpus]}")
    else:
        print("GPU: none detected — running on CPU.")
        print("""
  On CPU this will take many hours to days for 43k images. If you are on
  Windows, note that TensorFlow >= 2.11 has no native Windows GPU support
  at all, so installing CUDA will not help. Use Colab.""")
    return bool(gpus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--stage1-epochs", type=int, default=STAGE1_EPOCHS)
    ap.add_argument("--stage2-epochs", type=int, default=STAGE2_EPOCHS)
    ap.add_argument("--skip-finetune", action="store_true",
                    help="Stage 1 only — useful for a quick sanity run")
    ap.add_argument("--limit", type=int, default=None,
                    help="Train on N steps per epoch only (smoke test)")
    args = ap.parse_args()

    report_environment()

    train_ds, val_ds, test_ds, class_weights = build_datasets(batch_size=args.batch_size)

    if args.limit:
        train_ds = train_ds.take(args.limit)
        val_ds = val_ds.take(max(1, args.limit // 4))
        print(f"\n  SMOKE TEST: {args.limit} steps/epoch. Results are not meaningful.")

    MODELS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    started = datetime.now()

    # ---------------- Stage 1: frozen backbone ----------------
    print("\n" + "=" * 68)
    print(f"STAGE 1 — frozen backbone, lr={STAGE1_LR}")
    print("=" * 68 + "\n")

    model = build_baseline_model(freeze_backbone=True, learning_rate=STAGE1_LR)
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.stage1_epochs,
        class_weight=class_weights,
        callbacks=callbacks_for("stage1"),
        verbose=1,
    )

    best1_loss = min(h1.history.get("val_loss", [float("inf")]))
    best1 = max(h1.history.get("val_accuracy", [0]))
    print(f"\nStage 1 — best val_loss {best1_loss:.4f}, best val_accuracy {best1:.4f}")

    # ---------------- Stage 2: fine-tune ----------------
    if args.skip_finetune:
        print("\nSkipping fine-tuning (--skip-finetune).")
        h2 = None
    else:
        print("\n" + "=" * 68)
        print(f"STAGE 2 — fine-tuning, lr={STAGE2_LR}")
        print("=" * 68)
        print("\nBatchNorm layers stay frozen — updating their running statistics")
        print("on small batches destabilizes fine-tuning.\n")

        model = unfreeze_for_finetuning(model, learning_rate=STAGE2_LR)
        trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
        print(f"Trainable parameters now: {trainable:,}\n")

        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.stage2_epochs,
            class_weight=class_weights,
            callbacks=callbacks_for("stage2"),
            verbose=1,
        )
        best2_loss = min(h2.history.get("val_loss", [float("inf")]))
        best2 = max(h2.history.get("val_accuracy", [0]))
        print(f"\nStage 2 — best val_loss {best2_loss:.4f}, best val_accuracy {best2:.4f}")
        if best2_loss > best1_loss:
            print("""
  Fine-tuning did NOT improve val_loss over the frozen-backbone result. That happens
  when the learning rate is too high for the pretrained weights or the
  head hadn't converged before unfreezing. The saved stage1 checkpoint is
  still the better model — use it, and say so in the writeup rather than
  reporting the worse number.""")

    # The plan names models/baseline_effnet.h5, so that's the primary
    # artifact. Also saved as .keras — the native Keras 3 format, which
    # round-trips more reliably and is what evaluate.py loads by default.
    final_h5 = MODELS / "baseline_effnet.h5"
    final_keras = MODELS / "baseline_effnet.keras"
    model.save(final_keras)
    try:
        model.save(final_h5)
    except Exception as e:
        print(f"\n  Could not write .h5 ({type(e).__name__}: {e})")
        print("  The .keras file is the one that matters; evaluate.py uses it.")
        final_h5 = None

    elapsed = (datetime.now() - started).total_seconds()
    summary = {
        "finished": datetime.now().isoformat(timespec="seconds"),
        "minutes": round(elapsed / 60, 1),
        "batch_size": args.batch_size,
        "classes": SCC_CLASSES,
        "class_weights": {str(k): round(v, 4) for k, v in class_weights.items()},
        "monitored": "val_loss",
        "stage1_best_val_loss": round(float(best1_loss), 4),
        "stage1_best_val_accuracy": round(float(best1), 4),
        "stage2_best_val_loss": (
            round(float(min(h2.history.get("val_loss", [float("inf")]))), 4) if h2 else None
        ),
        "stage2_best_val_accuracy": (
            round(float(max(h2.history.get("val_accuracy", [0]))), 4) if h2 else None
        ),
        "stage1_epochs_run": len(h1.history.get("loss", [])),
        "stage2_epochs_run": len(h2.history.get("loss", [])) if h2 else 0,
    }
    with open(LOGS / "baseline_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 68)
    print("TRAINING COMPLETE")
    print("=" * 68)
    print(f"\n  Time: {elapsed/60:.1f} min")
    print(f"  Model: {final_keras}" + (f" and {final_h5}" if final_h5 else ""))
    print(f"  Summary: {LOGS / 'baseline_training_summary.json'}")
    print("""
  Do NOT evaluate on the test set yet if you still intend to change
  anything — architecture, hyperparameters, augmentation. Each look at
  test data costs a little of its independence. Tune against val, then
  run the test set once, in Phase 6.5.""")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------
# COLAB
# ---------------------------------------------------------------------
# Runtime > Change runtime type > T4 GPU, then:
#
#     from google.colab import drive
#     drive.mount('/content/drive')
#     %cd /content/drive/MyDrive/HueView-tool
#     !pip install -q tensorflow
#     !python src/baseline/train.py
#
# Sanity-check the wiring on a few steps before committing to a full run:
#
#     !python src/baseline/train.py --limit 5 --stage1-epochs 1 --skip-finetune
#
# resolved_manifest.csv stores paths relative to the repo root, so as long
# as you %cd there first, they resolve unchanged.
#
# Colab disconnects after ~12h (less when idle). Checkpoints land in
# models/ on Drive, so a dropped session loses at most the current epoch.
# ---------------------------------------------------------------------
