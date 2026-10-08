"""
Appendix 2, 3 and 5 tables that evaluate.py does not write yet.

Reads ONLY the per-image record (plus Phase 9's regional hue angles for
Table 35), so no model is loaded and the numbers always agree with
metrics_by_illumination.csv / metrics_by_region.csv:

    results/classification_results_record.csv   <- input (evaluate.py)
    data/processed/phase9_undertone.csv          <- input, per-region hue angles

    results/appendix/table12-17_confusion_<model>_<bin>.csv   Appendix 2
    results/appendix/table18_perclass_by_illumination.csv     Appendix 2
    results/appendix/table19_metrics_by_illumination.csv      Appendix 2
    results/appendix/table21-26_confusion_hueview_<region>.csv  Appendix 3
    results/appendix/table27_perclass_by_region.csv           Appendix 3
    results/appendix/table28_metrics_by_region.csv            Appendix 3
    results/appendix/table34_hueview_undertone_by_scc.csv     Appendix 5
    results/appendix/table35_hueview_undertone_by_region.csv  Appendix 5
    results/appendix/table36_baseline_undertone_by_scc.csv    Appendix 5

Run from HueView-tool/:

    python -m ml_pipeline.src.hueview.appendix_tables

Per-class Precision / Recall / F1 use zero_division = 0 over all six SCC
labels, exactly evaluate.metric_row, so macro averages equal the SOP1-3
tables. Table 35 reads the per-region hv_<region>_undertone columns that
evaluate.py writes; a record made before those columns existed falls back to
phase9_undertone.csv (with a warning), whose skin masks differ.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from .undertone import classify_undertone

ROOT = pathlib.Path(__file__).resolve().parents[2]
RECORD_CSV = ROOT / "results" / "classification_results_record.csv"
PHASE9_CSV = ROOT / "data" / "processed" / "phase9_undertone.csv"
OUT_DIR = ROOT / "results" / "appendix"

LABELS = list(range(1, 7))
BINS = ["Low", "Medium", "High"]
REGIONS = ["forehead", "left_cheek", "right_cheek", "jawline", "nose_bridge"]
CONFIGS = REGIONS + ["full_face"]
UNDERTONES = ["Warm", "Cool", "Neutral"]
TITLE = {"forehead": "Forehead", "left_cheek": "Left Cheek", "right_cheek": "Right Cheek",
         "jawline": "Jawline", "nose_bridge": "Nose Bridge", "full_face": "Full Face"}


def confusion(y_true, y_pred) -> pd.DataFrame:
    cm = pd.crosstab(pd.Categorical(y_true, LABELS), pd.Categorical(y_pred, LABELS),
                     dropna=False).reindex(index=LABELS, columns=LABELS, fill_value=0)
    cm.index = [f"SCC-{k}" for k in LABELS]
    cm.columns = [f"SCC-{k}" for k in LABELS]
    cm.index.name = "Actual / Predicted"
    cm["Total"] = cm.sum(axis=1)
    return cm


def per_class(y_true, y_pred) -> pd.DataFrame:
    """One-vs-rest TP/TN/FP/FN per SCC (Tables 5-10 reading of the matrix)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rows = []
    for k in LABELS:
        tp = int(((y_true == k) & (y_pred == k)).sum())
        fp = int(((y_true != k) & (y_pred == k)).sum())
        fn = int(((y_true == k) & (y_pred != k)).sum())
        tn = len(y_true) - tp - fp - fn
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"Class": f"SCC-{k}", "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                     "Precision": p, "Recall": r, "F1-Score": f})
    return pd.DataFrame(rows)


def summary(y_true, y_pred) -> dict:
    pc = per_class(y_true, y_pred)
    return {"n": len(y_true),
            "Accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
            "Macro Precision": pc["Precision"].mean(),
            "Macro Recall": pc["Recall"].mean(),
            "Macro F1-Score": pc["F1-Score"].mean()}


def distribution(groups: pd.Series, labels: pd.Series, group_name: str, order) -> pd.DataFrame:
    ct = pd.crosstab(groups, labels).reindex(index=order, columns=UNDERTONES, fill_value=0)
    ct.loc["Total"] = ct.sum()
    out = pd.DataFrame({f"{u} Count": ct[u] for u in UNDERTONES})
    out["Total"] = ct.sum(axis=1)
    for u in UNDERTONES:
        out[f"% {u}"] = (100 * ct[u] / out["Total"].where(out["Total"] > 0)).round(2)
    out.index.name = group_name
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = pd.read_csv(RECORD_CSV)
    y = rec["SCC_ground_truth"]
    models = {"baseline": ("Baseline RGB", "baseline_pred"),
              "hueview": ("HueView", "hv_full_face")}

    # ---- Appendix 2: Tables 12-19 (SOP1, SOP2) ----
    t18, t19, n = [], [], 12
    for key, (name, col) in models.items():
        for b in BINS:
            m = rec["illumination_label"] == b
            confusion(y[m], rec.loc[m, col]).to_csv(
                OUT_DIR / f"table{n}_confusion_{key}_{b.lower()}.csv")
            n += 1
            pc = per_class(y[m], rec.loc[m, col])
            pc.insert(0, "Illumination", b)
            pc.insert(0, "Model", name)
            t18.append(pc)
            t19.append({"Illumination": b, "Model": name, **summary(y[m], rec.loc[m, col])})
    pd.concat(t18).to_csv(OUT_DIR / "table18_perclass_by_illumination.csv", index=False)
    t19 = pd.DataFrame(t19).sort_values(["Illumination", "Model"],
                                        key=lambda s: s.map({b: i for i, b in enumerate(BINS)})
                                        if s.name == "Illumination" else s)
    t19.to_csv(OUT_DIR / "table19_metrics_by_illumination.csv", index=False)

    # ---- Appendix 3: Tables 21-28 (SOP3) ----
    t27, t28 = [], []
    for n, cfg in enumerate(CONFIGS, start=21):
        pred = rec[f"hv_{cfg}"]
        confusion(y, pred).to_csv(OUT_DIR / f"table{n}_confusion_hueview_{cfg}.csv")
        pc = per_class(y, pred)
        pc.insert(0, "Region", TITLE[cfg])
        t27.append(pc)
        t28.append({"Region": TITLE[cfg], **summary(y, pred)})
    pd.concat(t27).to_csv(OUT_DIR / "table27_perclass_by_region.csv", index=False)
    pd.DataFrame(t28).to_csv(OUT_DIR / "table28_metrics_by_region.csv", index=False)

    # ---- Appendix 5: Tables 34-36 (exploratory undertone) ----
    scc = y.map(lambda k: f"SCC-{k}")
    scc_order = [f"SCC-{k}" for k in LABELS]
    distribution(scc, rec["hv_majority_undertone"], "SCC", scc_order).to_csv(
        OUT_DIR / "table34_hueview_undertone_by_scc.csv")
    distribution(scc, rec["baseline_undertone"], "SCC", scc_order).to_csv(
        OUT_DIR / "table36_baseline_undertone_by_scc.csv")

    region_cols = [f"hv_{r}_undertone" for r in REGIONS]
    if set(region_cols) <= set(rec.columns):
        # Same raw CIELAB evaluate.py voted with, so Table 35 agrees with Table 34
        long = rec[["filename"] + region_cols].melt(
            id_vars="filename", var_name="region", value_name="undertone")
        long = long[long["undertone"].isin(UNDERTONES)]
        long["region"] = long["region"].str[3:-len("_undertone")].map(TITLE)
    else:
        print("[appendix] WARNING: the record has no per-region undertone columns; "
              "Table 35 falls back to phase9_undertone.csv, whose skin masks differ "
              "from evaluate.py's, so it will NOT agree with Table 34. Re-run "
              "evaluate.py and this script before reporting Table 35.")
        hue = pd.read_csv(PHASE9_CSV, usecols=["filename"] + [f"{r}_hue_deg" for r in REGIONS])
        hue = hue[hue["filename"].isin(rec["filename"])]
        long = hue.melt(id_vars="filename", var_name="region", value_name="hue").dropna()
        long["region"] = long["region"].str.removesuffix("_hue_deg").map(TITLE)
        long["undertone"] = long["hue"].map(classify_undertone)
    distribution(long["region"], long["undertone"], "Facial Region",
                 [TITLE[r] for r in REGIONS]).to_csv(
        OUT_DIR / "table35_hueview_undertone_by_region.csv")

    print(f"[appendix] {len(rec)} test images -> {OUT_DIR}")
    print(t19.round(4).to_string(index=False))
    print(pd.DataFrame(t28).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
