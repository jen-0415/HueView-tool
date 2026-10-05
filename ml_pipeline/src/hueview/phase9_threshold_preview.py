"""
Phase 9 -- Undertone threshold preview (analysis only, no images read)

Reads the per-region hue angles already saved in phase9_undertone.csv and
re-runs the majority vote under candidate hue thresholds, so the team can
compare them before changing Stage 6 / Label Generation.

Every candidate is DERIVED FROM THE TRAIN SPLIT ONLY and then applied
unchanged to train, val and test -- the threshold is never chosen by looking
at val/test.

Candidates (all keep the researchers' symmetric +/- HALF_BAND Neutral band):

  manuscript   centre = 60 deg (Pantone SkinTone Guide, Sirisayan 2022).
               The current reported rule.

  global       centre = median hue of all measured regions in TRAIN.
               "Warm/Cool relative to this dataset's typical skin hue."
               One number for every face; needs nothing but the hue.

  per_scc      centre = median hue of that SCC's regions in TRAIN.
               "Warm/Cool relative to other faces of the same skin depth"
               (the Pantone SkinTone Guide is itself laid out as lightness
               levels x undertone). Removes the depth confound by design.
               CAVEAT: at inference the SCC is not known -- the centre would
               come from HueView's PREDICTED SCC, so a skin tone error also
               shifts the undertone. State this if you choose it.

Output:
    data/processed/phase9_threshold_preview.csv   -- label % per SCC per rule
    data/processed/phase9_threshold_centres.csv   -- the centres used

Run from the HueView-tool repo root (after run_phase9_batch):
    python -m ml_pipeline.src.hueview.phase9_threshold_preview
    python -m ml_pipeline.src.hueview.phase9_threshold_preview --half-band 7.5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root(start: Path, marker: str = "data") -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / marker).is_dir():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from ml_pipeline.src.hueview.regions import REGION_ORDER  # noqa: E402
from ml_pipeline.src.hueview.undertone import (  # noqa: E402
    REFERENCE_ANGLE_DEG, majority_undertone)

PROCESSED = PROJECT_ROOT / "data" / "processed"
IN_PATH = PROCESSED / "phase9_undertone.csv"
OUT_PATH = PROCESSED / "phase9_threshold_preview.csv"
CENTRES_PATH = PROCESSED / "phase9_threshold_centres.csv"
LABELS = ("Cool", "Neutral", "Warm")
HUE_COLS = [f"{r}_hue_deg" for r in REGION_ORDER]


def signed(h: pd.Series) -> pd.Series:
    """0-360 reported hue -> signed (-180..180), same as undertone.py."""
    return h.where(h <= 180, h - 360)


def vote(df: pd.DataFrame, centres: pd.Series, half_band: float) -> pd.Series:
    """Majority label per image, using a per-row centre."""
    out = []
    for hues, c in zip(df[HUE_COLS].itertuples(index=False), centres):
        regions = {r: h for r, h in zip(REGION_ORDER, hues) if pd.notna(h)}
        out.append(majority_undertone(regions, center=c, half_band=half_band)[0]
                   if regions else None)
    return pd.Series(out, index=df.index)


def pct_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    t = pd.crosstab(df["scc_label"], df[col], normalize="index") * 100
    t.loc["ALL"] = df[col].value_counts(normalize=True) * 100
    return t.reindex(columns=list(LABELS)).fillna(0).round(1)


def main(half_band: float):
    if not IN_PATH.is_file():
        raise SystemExit(f"ERROR: {IN_PATH} not found -- run run_phase9_batch first.")
    df = pd.read_csv(IN_PATH)
    missing = [c for c in ["split", "scc_label"] + HUE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: {IN_PATH.name} has no {missing} -- rerun the new run_phase9_batch.")
    for c in HUE_COLS:
        df[c] = signed(df[c])

    train = df[df["split"] == "train"]
    if train.empty:
        raise SystemExit("ERROR: no train rows (split column empty?).")
    print(f"{IN_PATH.name}: {len(df)} images, {len(train)} in train "
          f"(centres are computed from train only)")
    print(f"Neutral band: +/- {half_band} deg")

    # -------------------------------------------------- train hue distribution
    long = train.melt(id_vars="scc_label", value_vars=HUE_COLS, value_name="hue").dropna()
    q = long.groupby("scc_label")["hue"].quantile([.10, .25, .50, .75, .90]).unstack()
    q.columns = ["p10", "p25", "median", "p75", "p90"]
    q.loc["ALL"] = long["hue"].quantile([.10, .25, .50, .75, .90]).values
    print("\nTRAIN REGION HUE (deg), by SCC")
    print("-" * 60)
    print(q.round(1).to_string())

    # ------------------------------------------------------------ centres
    global_c = round(float(long["hue"].median()), 1)
    scc_c = long.groupby("scc_label")["hue"].median().round(1)
    rules = {
        "manuscript": pd.Series(REFERENCE_ANGLE_DEG, index=df.index),
        "global": pd.Series(global_c, index=df.index),
        "per_scc": df["scc_label"].map(scc_c).fillna(global_c),
    }
    centres = pd.DataFrame(
        [("manuscript", "ALL", REFERENCE_ANGLE_DEG), ("global", "ALL", global_c)]
        + [("per_scc", s, c) for s, c in scc_c.items()],
        columns=["rule", "scc_label", "centre_deg"])
    centres["warm_above"] = centres["centre_deg"] + half_band
    centres["cool_below"] = centres["centre_deg"] - half_band
    centres.to_csv(CENTRES_PATH, index=False)
    print("\nCENTRES (Warm > centre + band, Cool < centre - band)")
    print("-" * 60)
    print(centres.to_string(index=False))

    # ------------------------------------------------------------ compare
    rows = []
    for name, c in rules.items():
        col = f"undertone_{name}"
        df[col] = vote(df, c, half_band)
        for part, sub in [("all", df), ("test", df[df["split"] == "test"])]:
            t = pct_table(sub, col)
            if part == "all":
                print(f"\nMAJORITY UNDERTONE % BY SCC -- {name}  (all splits)")
                print("-" * 60)
                print(t.to_string())
            long_t = t.reset_index().melt(id_vars="scc_label", var_name="undertone",
                                          value_name="percent")
            long_t.insert(0, "split", part)
            long_t.insert(0, "rule", name)
            rows.append(long_t)

    test = df[df["split"] == "test"]
    print("\nTEST SPLIT ONLY -- ALL row (should look like the train-derived result)")
    print("-" * 60)
    print(pd.DataFrame({n: test[f"undertone_{n}"].value_counts(normalize=True)
                        .reindex(list(LABELS)).fillna(0).mul(100).round(1)
                        for n in rules}).to_string())

    pd.concat(rows).to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}\nSaved {CENTRES_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preview undertone hue thresholds")
    ap.add_argument("--half-band", type=float, default=5.0,
                    help="Neutral band half-width in degrees (manuscript: 5)")
    main(ap.parse_args().half_band)
