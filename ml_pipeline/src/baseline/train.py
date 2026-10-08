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

Every file this script writes carries the "final" tag, so a retrain never
overwrites artifacts from earlier runs:

    models/baseline_stage1_final_best.keras     best stage-1 checkpoint
    models/baseline_stage2_final_best.keras     best stage-2 checkpoint
    models/baseline_effnet_final.keras          the model to evaluate
    models/baseline_effnet_final.h5             same model, legacy format
    logs/baseline_stage1_final.csv              per-epoch metrics
    logs/baseline_stage2_final.csv
    logs/baseline_stage1_final_done.json        stage-complete markers
    logs/baseline_stage2_final_done.json
    logs/baseline_training_summary_final.json
    backups/baseline_stage1_final/              mid-stage resume state
    backups/baseline_stage2_final/              (deleted when a stage ends)

Resuming: if the session dies mid-stage, rerun the same command. The stage
continues from its last completed epoch, and a stage that already finished
is skipped (its best checkpoint is loaded instead). To throw away a
previous final run and start over, pass --fresh.

Smoke-test runs (--limit) are tagged "smoke" instead of "final", so they
can never overwrite, or be resumed into, the real run.

    python src/baseline/train.py
"""

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    from ml_pipeline.src.baseline.data_pipeline import build_datasets, SCC_CLASSES
    from ml_pipeline.src.baseline.model import build_baseline_model, unfreeze_for_finetuning
except ImportError:
    # Run directly as `python src/baseline/train.py`: Python puts this
    # script's own folder on sys.path, so the sibling modules import by name.
    from data_pipeline import build_datasets, SCC_CLASSES
    from model import build_baseline_model, unfreeze_for_finetuning

MODELS = Path("models")
LOGS = Path("logs")
BACKUPS = Path("backups")

TAG = "final"  # switched to "smoke" for --limit runs

STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 20
STAGE1_LR = 1e-3
STAGE2_LR = 1e-5
PATIENCE = 5


# ---------------------------------------------------------------------
# File names — every artifact goes through these, so the tag is
# applied consistently.
# ---------------------------------------------------------------------
def ckpt_path(stage: str) -> Path:
    return MODELS / f"baseline_{stage}_{TAG}_best.keras"


def csv_path(stage: str) -> Path:
    return LOGS / f"baseline_{stage}_{TAG}.csv"


def done_marker(stage: str) -> Path:
    return LOGS / f"baseline_{stage}_{TAG}_done.json"


def backup_dir(stage: str) -> Path:
    return BACKUPS / f"baseline_{stage}_{TAG}"


def final_keras() -> Path:
    return MODELS / f"baseline_effnet_{TAG}.keras"


def final_h5() -> Path:
    return MODELS / f"baseline_effnet_{TAG}.h5"


def summary_path() -> Path:
    return LOGS / f"baseline_training_summary_{TAG}.json"


def wipe_tagged_artifacts():
    """Delete every file from a previous run with the current tag."""
    removed = 0
    for folder in (MODELS, LOGS):
        if folder.exists():
            for p in folder.glob(f"baseline_*_{TAG}*"):
                if p.is_file():
                    p.unlink()
                    removed += 1
    if BACKUPS.exists():
        for p in BACKUPS.glob(f"baseline_*_{TAG}"):
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
    if removed:
        print(f"  Removed {removed} existing '{TAG}' artifact(s).")


# ---------------------------------------------------------------------
# Metrics come from the CSV log rather than the History object, because
# after a resume History only covers the epochs run in this session.
# ---------------------------------------------------------------------
def best_from_csv(stage: str):
    """Best epoch by validation macro F1, with that epoch's val_loss and
    val_accuracy."""
    p = csv_path(stage)
    if not p.exists():
        return None
    with open(p, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("val_f1_score")]
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r["val_f1_score"]))
    return {
        "best_epoch": int(best["epoch"]) + 1,
        "val_f1_score": round(float(best["val_f1_score"]), 4),
        "val_loss": round(float(best["val_loss"]), 4),
        "val_accuracy": round(float(best.get("val_accuracy") or "nan"), 4),
        "epochs_run": len(rows),
    }


def callbacks_for(stage: str, prior_best_f1=None):
    """
    Checkpoint + early stopping + LR reduction + resume.

    Checkpoint and early stopping monitor validation macro F1, as the
    manuscript specifies (Appendix 1: "checkpoints corresponding to the best
    validation Macro F1-Score"), the same rule HueView's train.py uses.
    Macro F1 weights all six SCC classes equally, so with SCC-5 at ~6x
    SCC-6 a model can't look good by leaning on the majority class.
    ReduceLROnPlateau still watches val_loss: it only paces the learning
    rate and doesn't choose the saved model.

    save_best_only means an overfitting tail can't overwrite a good
    checkpoint. On a resume, initial_value_threshold carries the best
    val macro F1 seen before the crash, so the first resumed epoch can't
    overwrite a better checkpoint either.

    The checkpoint file, not EarlyStopping's restore_best_weights, is
    the source of truth: run_stage reloads it after fit(), because
    restore_best_weights only knows about epochs from the current session.
    """
    MODELS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    BACKUPS.mkdir(exist_ok=True)

    return [
        keras.callbacks.BackupAndRestore(backup_dir=str(backup_dir(stage))),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path(stage)),
            monitor="val_f1_score", mode="max",
            save_best_only=True, verbose=1,
            initial_value_threshold=prior_best_f1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_f1_score", mode="max",
            patience=PATIENCE, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=0.5, patience=3, min_lr=1e-7, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(csv_path(stage)), append=True),
    ]


def run_stage(stage, model, train_ds, val_ds, epochs, class_weights):
    resuming = backup_dir(stage).exists()
    prior = best_from_csv(stage) if resuming else None

    if resuming:
        print(f"  Resuming {stage} from {backup_dir(stage)}")
        if prior:
            print(f"  Best so far: val macro F1 {prior['val_f1_score']} (epoch {prior['best_epoch']})\n")
    elif csv_path(stage).exists():
        # Leftover log from an earlier attempt that never got past epoch 1;
        # appending to it would mix runs.
        csv_path(stage).unlink()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks_for(stage, prior["val_f1_score"] if prior else None),
        verbose=1,
    )

    model.load_weights(str(ckpt_path(stage)))
    best = best_from_csv(stage)
    done_marker(stage).write_text(json.dumps(best, indent=2))
    return best


def report_environment():
    gpus = tf.config.list_physical_devices("GPU")
    print("=" * 68)
    print(f"PHASE 6.4 — BASELINE TRAINING  [tag: {TAG}]")
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
    global TAG

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--stage1-epochs", type=int, default=STAGE1_EPOCHS)
    ap.add_argument("--stage2-epochs", type=int, default=STAGE2_EPOCHS)
    ap.add_argument("--skip-finetune", action="store_true",
                    help="Stage 1 only — useful for a quick sanity run")
    ap.add_argument("--limit", type=int, default=None,
                    help="Train on N steps per epoch only (smoke test)")
    ap.add_argument("--fresh", action="store_true",
                    help="Delete existing 'final' artifacts and start over")
    args = ap.parse_args()

    if args.limit:
        TAG = "smoke"

    report_environment()

    if args.fresh or args.limit:  # smoke tests always start clean
        wipe_tagged_artifacts()

    train_ds, val_ds, test_ds, class_weights = build_datasets(batch_size=args.batch_size)

    if args.limit:
        train_ds = train_ds.take(args.limit)
        val_ds = val_ds.take(max(1, args.limit // 4))
        print(f"\n  SMOKE TEST: {args.limit} steps/epoch. Results are not meaningful.")

    MODELS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    BACKUPS.mkdir(exist_ok=True)
    started = datetime.now()

    # ---------------- Stage 1: frozen backbone ----------------
    print("\n" + "=" * 68)
    print(f"STAGE 1 — frozen backbone, lr={STAGE1_LR}")
    print("=" * 68 + "\n")

    model = build_baseline_model(freeze_backbone=True, learning_rate=STAGE1_LR)
    if done_marker("stage1").exists():
        print("  Stage 1 already complete — loading its best checkpoint.")
        model.load_weights(str(ckpt_path("stage1")))
        s1 = best_from_csv("stage1")
    else:
        s1 = run_stage("stage1", model, train_ds, val_ds, args.stage1_epochs, class_weights)

    print(f"\nStage 1 — best val macro F1 {s1['val_f1_score']:.4f} "
          f"(epoch {s1['best_epoch']}), val_accuracy at that epoch {s1['val_accuracy']:.4f}")

    # ---------------- Stage 2: fine-tune ----------------
    s2 = None
    if args.skip_finetune:
        print("\nSkipping fine-tuning (--skip-finetune).")
    else:
        print("\n" + "=" * 68)
        print(f"STAGE 2 — fine-tuning, lr={STAGE2_LR}")
        print("=" * 68)
        print("\nBatchNorm layers stay frozen — updating their running statistics")
        print("on small batches destabilizes fine-tuning.\n")

        model = unfreeze_for_finetuning(model, learning_rate=STAGE2_LR)
        trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
        print(f"Trainable parameters now: {trainable:,}\n")

        if done_marker("stage2").exists():
            print("  Stage 2 already complete — loading its best checkpoint.")
            model.load_weights(str(ckpt_path("stage2")))
            s2 = best_from_csv("stage2")
        else:
            s2 = run_stage("stage2", model, train_ds, val_ds, args.stage2_epochs, class_weights)

        print(f"\nStage 2 — best val macro F1 {s2['val_f1_score']:.4f} "
              f"(epoch {s2['best_epoch']}), val_accuracy at that epoch {s2['val_accuracy']:.4f}")

    # ---------------- Pick the better stage ----------------
    if s2 is not None and s2["val_f1_score"] > s1["val_f1_score"]:
        chosen = "stage2"  # model already holds the stage-2 best weights
    else:
        chosen = "stage1"
        if s2 is not None:
            print("""
  Fine-tuning did NOT improve val macro F1 over the frozen-backbone result. That
  happens when the learning rate is too high for the pretrained weights or
  the head hadn't converged before unfreezing. The stage-1 weights are being
  saved as the final model — say so in the writeup rather than reporting the
  worse number.""")
        model.load_weights(str(ckpt_path("stage1")))

    # The plan names models/baseline_effnet.h5, so an .h5 copy is kept.
    # The .keras file is the native Keras 3 format and round-trips more
    # reliably; point evaluate.py at it.
    out_keras, out_h5 = final_keras(), final_h5()
    model.save(out_keras)
    try:
        model.save(out_h5)
    except Exception as e:
        print(f"\n  Could not write .h5 ({type(e).__name__}: {e})")
        print(f"  {out_keras} is the one that matters.")
        out_h5 = None

    elapsed = (datetime.now() - started).total_seconds()
    summary = {
        "tag": TAG,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "minutes_this_session": round(elapsed / 60, 1),
        "batch_size": args.batch_size,
        "classes": SCC_CLASSES,
        "class_weights": {str(k): round(v, 4) for k, v in class_weights.items()},
        "monitored": "val_f1_score (macro)",
        "stage1": s1,
        "stage2": s2,
        "final_model_from": chosen,
        "final_model": str(out_keras),
    }
    with open(summary_path(), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 68)
    print("TRAINING COMPLETE")
    print("=" * 68)
    print(f"\n  Time this session: {elapsed/60:.1f} min")
    print(f"  Final model (from {chosen}): {out_keras}" + (f" and {out_h5}" if out_h5 else ""))
    print(f"  Summary: {summary_path()}")
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
# Sanity-check the wiring on a few steps before committing to a full run
# (writes "smoke" files only, never "final"):
#
#     !python src/baseline/train.py --limit 5 --stage1-epochs 1 --skip-finetune
#
# If Colab disconnects, run the exact same command again — it resumes.
# To discard a finished final run and retrain from scratch:
#
#     !python src/baseline/train.py --fresh
#
# resolved_manifest.csv stores paths relative to the repo root, so as long
# as you %cd there first, they resolve unchanged.
#
# Colab disconnects after ~12h (less when idle). Checkpoints and resume
# backups land in models/ and backups/ on Drive, so a dropped session
# loses at most the current epoch.
# ---------------------------------------------------------------------