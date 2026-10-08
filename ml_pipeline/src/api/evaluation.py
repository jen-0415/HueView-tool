"""
GET /api/evaluation -- the evaluation results (baseline and HueView by
lighting, HueView by facial region, significance tests), read from the result
files the evaluation scripts write. Nothing is recomputed here, so the tab shows exactly the numbers that go into Chapter 4:

    results/classification_results_record.csv   evaluate.py       (per image)
    results/appendix/*.csv                      appendix_tables.py (lighting, regions)
    results/sop4/*.csv                          significance.py    (significance tests)

Undertone (Appendix 5) is exploratory and not reported for now, so it is
left out here; appendix_tables.py still writes Tables 34-36.

When a later script has not been re-run after an earlier one, the payload
carries a warning instead of silently showing mixed results.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..inference.paths import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"
RECORD = RESULTS / "classification_results_record.csv"
APPENDIX = RESULTS / "appendix"
SOP4 = RESULTS / "sop4"

BINS = ["Low", "Medium", "High"]
CONFIGS = ["forehead", "left_cheek", "right_cheek", "jawline", "nose_bridge", "full_face"]
REGION_NAME = {"forehead": "Forehead", "left_cheek": "Left Cheek", "right_cheek": "Right Cheek",
               "jawline": "Jawline", "nose_bridge": "Nose Bridge", "full_face": "Full Face"}


def _rows(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    return df.replace({np.nan: None}).to_dict("records")


def _matrix(path: Path) -> dict:
    df = pd.read_csv(path, index_col=0)
    return {"labels": [c for c in df.columns if c != "Total"],
            "rows": df.drop(columns="Total").to_numpy().tolist()}


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


def _stamp(t: float) -> str:
    return datetime.datetime.fromtimestamp(t).isoformat(timespec="seconds")


def build_evaluation() -> dict:
    needed = [RECORD, APPENDIX / "table19_metrics_by_illumination.csv",
              SOP4 / "table33_decisions.csv"]
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing result files: {', '.join(missing)}")

    record_cols = pd.read_csv(RECORD, nrows=0).columns
    n_test = sum(1 for _ in open(RECORD, encoding="utf-8")) - 1
    t_rec = _mtime(RECORD)
    t_app = min(_mtime(p) for p in APPENDIX.glob("*.csv"))
    t_sop4 = min(_mtime(p) for p in SOP4.glob("*.csv"))

    warnings = []
    if "hv_forehead_p1" not in record_cols:
        warnings.append("These results come from the old Full Face (a separately trained model), "
                        "not the manuscript's five-region majority vote. Re-run evaluate.py, "
                        "appendix_tables.py and significance.py.")
    if t_app < t_rec:
        warnings.append("The lighting and region tables are older than the per-image results. "
                        "Re-run appendix_tables.py.")
    if t_sop4 < t_rec:
        warnings.append("The significance tests are older than the per-image results. "
                        "Re-run significance.py.")

    metrics = _rows(APPENDIX / "table19_metrics_by_illumination.csv")
    perclass = _rows(APPENDIX / "table18_perclass_by_illumination.csv")
    by_model = {}
    for key, name, first in (("baseline_lighting", "Baseline RGB", 12),
                             ("hueview_lighting", "HueView", 15)):
        by_model[key] = {
            "metrics": [r for r in metrics if r["Model"] == name],
            "perclass": [r for r in perclass if r["Model"] == name],
            "confusion": {b: _matrix(APPENDIX / f"table{first + i}_confusion_"
                                     f"{key.split('_')[0]}_{b.lower()}.csv")
                          for i, b in enumerate(BINS)},
        }

    regions = {
        "metrics": _rows(APPENDIX / "table28_metrics_by_region.csv"),
        "perclass": _rows(APPENDIX / "table27_perclass_by_region.csv"),
        "confusion": {REGION_NAME[c]: _matrix(APPENDIX / f"table{21 + i}_confusion_hueview_{c}.csv")
                      for i, c in enumerate(CONFIGS)},
    }

    significance = {
        "h01": _rows(SOP4 / "table31a_h01.csv"),
        "h02_omnibus": _rows(SOP4 / "table31b_h02_omnibus.csv"),
        "h02_pairwise": _rows(SOP4 / "table31c_h02_pairwise.csv"),
        "h03": _rows(SOP4 / "table31d_h03.csv"),
        "decisions": _rows(SOP4 / "table33_decisions.csv"),
        "paired_h01": _rows(SOP4 / "table29a_h01_paired_outcomes.csv"),
        "paired_h03": _rows(SOP4 / "table29c_h03_paired_outcomes.csv"),
    }

    return {
        "n_test": n_test,
        "generated": {"record": _stamp(t_rec), "tables": _stamp(t_app), "significance": _stamp(t_sop4)},
        "warnings": warnings,
        **by_model,
        "regions": regions,
        "significance": significance,
    }
