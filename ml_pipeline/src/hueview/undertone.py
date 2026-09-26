"""
Phase 9 -- Undertone Descriptor (Exploratory, no training needed)
Manuscript: Chapter 3, Stage 6 (Undertone Descriptor Computation); Data
Generation, Label Generation; Definition of Terms ("Hue Angle", "Undertone",
"Undertone Chromatic Descriptor"); Appendix 1; Appendix 5.

INPUT -- no new images. Stage 6 "utilizes the CIELAB Feature Vector produced
in Stage 5", and Stage 5 converts "the per-region skin samples produced by the
SSR stage". So the input is Phase 8.2's per-region mean [L*, a*, b*], computed
from the SSR-normalized image (sigma = 30 in the current pipeline; the
manuscript still says 80) over each region's skin mask.

-------------------------------------------------------------------------------
Rules, as stated in the manuscript
-------------------------------------------------------------------------------
1. Hue angle per region: h_ab = atan2(b*, a*), "converted to degrees and
   mapped to the 0-360 range" (Label Generation).
2. Per-region descriptor (60 deg Pantone SkinTone Guide reference, +/-5 deg):
       Warm    : h_ab > 65 deg      ("yellow-dominant base")
       Neutral : 55 <= h_ab <= 65
       Cool    : h_ab < 55 deg      ("red or pink-dominant base")
3. Majority vote across the five regions -> Majority Undertone Label.
4. Tie: "the descriptor closest to the 60 deg reference threshold is
   selected, ensuring deterministic label assignment" (Label Generation).
5. Exploratory: never used as a label, never scored against ground truth.

-------------------------------------------------------------------------------
Implementation decisions the manuscript leaves open (state them in Ch. 3)
-------------------------------------------------------------------------------
a. Tie-break applies AMONG THE TIED DESCRIPTORS. With 2 Warm / 2 Cool /
   1 Neutral, the winner is Warm or Cool -- whichever tied region's hue is
   closest to 60 deg. (Choosing among ALL regions would always pick the
   single Neutral region, since Neutral is by definition within 5 deg of 60,
   so a 1-vote label would beat two 2-vote labels.) If two tied regions are
   exactly equally close, the earlier region in REGION_ORDER wins, so the
   result is always deterministic.
b. Missing regions do not vote. A region with no usable skin pixels (Phase
   7.4/7.5) has no measured colour; the value Phase 8.2 imputes for it in the
   15-value vector is an estimate, not a measurement, so it is excluded here.
   The vote uses the regions that were actually measured.
c. Hues just below 0 deg (pink: a* > 0, b* < 0) are Cool. The manuscript
   reports hue on 0-360, but a pink hue of e.g. 350 deg would read as "> 65,
   Warm" if thresholds were applied to the 0-360 number, contradicting the
   manuscript's own "Cool = red or pink-dominant". So classification uses the
   signed angle from atan2 (-180 to 180); the REPORTED hue stays 0-360.
   For every hue from 0 to 180 deg the two are identical.

Usage (run from ml_pipeline/):
    python -m src.hueview.undertone --smoke
"""
from __future__ import annotations

import argparse
from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np

from .region_selector import ConfigurationOutput
from .regions import REGION_ORDER

WARM = "Warm"
NEUTRAL = "Neutral"
COOL = "Cool"
UNDERTONES = (WARM, NEUTRAL, COOL)

REFERENCE_ANGLE_DEG = 60.0
WARM_THRESHOLD_DEG = 65.0
COOL_THRESHOLD_DEG = 55.0


# ---------------------------------------------------------------------------
# Step 1 -- hue angle
# ---------------------------------------------------------------------------
def signed_hue_deg(a: float, b: float) -> float:
    """atan2(b*, a*) in degrees, -180 to 180 (used for classification)."""
    return float(np.degrees(np.arctan2(b, a)))


def hue_angle_deg(a: float, b: float) -> float:
    """h_ab = atan2(b*, a*), converted to degrees and mapped to 0-360 (reported)."""
    return float(signed_hue_deg(a, b) % 360.0)


# ---------------------------------------------------------------------------
# Step 2 -- per-region descriptor
# ---------------------------------------------------------------------------
def classify_undertone(hue_deg: float) -> str:
    """
    Warm > 65 deg, Cool < 55 deg, Neutral 55-65 deg (inclusive).

    Accepts either the 0-360 reported hue or the signed hue; values above
    180 deg are read as their signed equivalent (e.g. 350 -> -10, pink,
    Cool). See decision (c) in the module docstring.
    """
    if not np.isfinite(hue_deg):
        raise ValueError("Hue angle is NaN/inf -- the region has no measured colour.")
    h = hue_deg - 360.0 if hue_deg > 180.0 else hue_deg
    if h > WARM_THRESHOLD_DEG:
        return WARM
    if h < COOL_THRESHOLD_DEG:
        return COOL
    return NEUTRAL


# ---------------------------------------------------------------------------
# Step 3 -- majority vote with the manuscript's tie-break
# ---------------------------------------------------------------------------
def _distance_to_reference(hue_deg: float) -> float:
    h = hue_deg - 360.0 if hue_deg > 180.0 else hue_deg
    return abs(h - REFERENCE_ANGLE_DEG)


def majority_undertone(per_region_hues: Dict[str, float]) -> Tuple[str, bool]:
    """
    Majority vote over the regions' descriptors -> (label, was_tied).

    Tie: among the regions whose descriptor is one of the TIED labels, the
    region with the hue closest to 60 deg decides; equal distances fall back
    to REGION_ORDER. Only measured regions should be passed in.
    """
    if not per_region_hues:
        raise ValueError("No measured regions to vote with.")
    labels = {r: classify_undertone(h) for r, h in per_region_hues.items()}
    counts = Counter(labels.values())
    top = max(counts.values())
    leaders = {lab for lab, c in counts.items() if c == top}
    if len(leaders) == 1:
        return next(iter(leaders)), False

    order = {r: i for i, r in enumerate(REGION_ORDER)}
    candidates = [r for r in per_region_hues if labels[r] in leaders]
    winner = min(candidates,
                 key=lambda r: (_distance_to_reference(per_region_hues[r]),
                                order.get(r, len(order))))
    return labels[winner], True


# ---------------------------------------------------------------------------
# Full descriptor for one configuration
# ---------------------------------------------------------------------------
def compute_undertone_descriptor(cfg: ConfigurationOutput,
                                 cielab_vector: Optional[np.ndarray]) -> Optional[dict]:
    """
    Undertone descriptor for one ConfigurationOutput, from Phase 8.2's vector.

    single region (3 values) : that region's hue + descriptor
    full_face     (15 values): per-region hues + descriptors of the MEASURED
                               regions, and their Majority Undertone Label

    Returns None when nothing was measured (cielab_vector is None, or every
    component region was imputed). Never scored against ground truth.
    """
    if cielab_vector is None:
        return None
    order = cfg.component_order()
    if cielab_vector.shape[0] != 3 * len(order):
        raise ValueError(
            f"[{cfg.config}] CIELAB vector is {cielab_vector.shape[0]}-d, "
            f"expected {3 * len(order)} for {len(order)} region(s).")

    imputed = set(cfg.imputed_regions)
    regions: Dict[str, dict] = {}
    for name, (L, a, b) in zip(order, cielab_vector.reshape(len(order), 3)):
        if name in imputed or not np.isfinite([a, b]).all():
            continue                                    # decision (b)
        hue = hue_angle_deg(a, b)
        regions[name] = {"L": float(L), "a": float(a), "b": float(b),
                         "hue_deg": hue, "undertone": classify_undertone(hue)}
    if not regions:
        return None

    label, tied = majority_undertone({r: v["hue_deg"] for r, v in regions.items()})
    return {
        "config": cfg.config,
        "regions": regions,
        "majority_undertone": label,
        "was_tied": tied,
        "n_regions_voted": len(regions),
        "excluded_regions": [r for r in order if r not in regions],
    }


# ---------------------------------------------------------------------------
# Step 4 -- store alongside (but separate from) the skin tone predictions
# ---------------------------------------------------------------------------
def undertone_record(image_id: str, descriptor: Optional[dict]) -> dict:
    """
    One flat CSV row per image: the Majority Undertone Label (Appendix 2,
    Table 10 column) plus every region's hue and descriptor (Appendix 5,
    Table 34). Kept in its own file -- never joined into metric computation.
    """
    row = {"image_id": image_id}
    if descriptor is None:
        row.update({"majority_undertone": "", "was_tied": "", "n_regions_voted": 0})
        for r in REGION_ORDER:
            row[f"{r}_hue_deg"], row[f"{r}_undertone"] = "", ""
        return row
    row.update({"majority_undertone": descriptor["majority_undertone"],
                "was_tied": descriptor["was_tied"],
                "n_regions_voted": descriptor["n_regions_voted"]})
    for r in REGION_ORDER:
        v = descriptor["regions"].get(r)
        row[f"{r}_hue_deg"] = round(v["hue_deg"], 3) if v else ""
        row[f"{r}_undertone"] = v["undertone"] if v else ""
    return row


# ---------------------------------------------------------------------------
# Smoke test (no dataset needed)
# ---------------------------------------------------------------------------
def _smoke() -> int:
    from .cielab_features import build_cielab_vector
    from .region_selector import FULL_FACE, IMPUTE_NAN, RegionalConfigurationSelector
    from .regions import STATUS_EMPTY, STATUS_OK, RegionPatch
    from skimage.color import lab2rgb

    passed = failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  [ok]   {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name} {detail}")

    print("\n== step 1: hue angle ==")
    check("a*=b* -> 45 deg", np.isclose(hue_angle_deg(10, 10), 45))
    check("pure +b* (yellow) -> 90 deg", np.isclose(hue_angle_deg(0, 10), 90))
    check("atan2 quadrant: a*<0, b*>0 -> 135 deg", np.isclose(hue_angle_deg(-10, 10), 135))
    check("mapped to 0-360: a*>0, b*<0 -> 350 deg",
          np.isclose(hue_angle_deg(10, -10 * np.tan(np.radians(10))), 350))

    print("\n== step 2: thresholds (Pantone 60 deg +/- 5) ==")
    cases = {70: WARM, 65.01: WARM, 65: NEUTRAL, 60: NEUTRAL, 55: NEUTRAL,
             54.99: COOL, 40: COOL}
    check("Warm > 65, Neutral 55-65 inclusive, Cool < 55",
          all(classify_undertone(h) == lab for h, lab in cases.items()),
          str({h: classify_undertone(h) for h in cases}))
    check("pink hue 350 deg (a*>0, b*<0) -> Cool, not Warm",
          classify_undertone(350.0) == COOL)
    check("blue-green 250 deg -> not Warm", classify_undertone(250.0) != WARM)
    try:
        classify_undertone(float("nan")); check("NaN hue raises (was silently Neutral)", False)
    except ValueError:
        check("NaN hue raises (was silently Neutral)", True)

    print("\n== step 3: majority vote + tie-break ==")
    lab, tied = majority_undertone({"forehead": 70, "left_cheek": 72, "right_cheek": 40,
                                    "nose_bridge": 75, "jawline": 60})
    check("3 Warm / 1 Cool / 1 Neutral -> Warm, no tie", (lab, tied) == (WARM, False))
    lab, tied = majority_undertone({"forehead": 67, "left_cheek": 80, "right_cheek": 52,
                                    "nose_bridge": 30, "jawline": 60})
    check("2W/2C/1N tie -> Warm (67 deg is the tied region closest to 60)",
          (lab, tied) == (WARM, True), str((lab, tied)))
    check("tie never returns the 1-vote Neutral (old behaviour did)", lab != NEUTRAL)
    lab, _ = majority_undertone({"forehead": 80, "left_cheek": 68, "right_cheek": 53,
                                 "nose_bridge": 30, "jawline": 60})
    check("2W/2C/1N tie -> Cool when 53 deg (7 away) beats 68 deg (8 away)", lab == COOL)
    lab, _ = majority_undertone({"forehead": 70, "jawline": 50})
    check("exact distance tie -> earlier region in REGION_ORDER (deterministic)",
          lab == WARM)

    print("\n== with the real 7.5 selector + 8.2 CIELAB ==")
    H = W = 224

    def rgb_for_hue(hue, L=60, C=20):
        lab = np.array([[[L, C * np.cos(np.radians(hue)), C * np.sin(np.radians(hue))]]])
        return tuple(int(round(v)) for v in (lab2rgb(lab)[0, 0] * 255))

    boxes = {"forehead": (10, 50, 50, 170), "left_cheek": (80, 140, 130, 190),
             "right_cheek": (80, 140, 30, 90), "nose_bridge": (80, 140, 95, 125),
             "jawline": (160, 210, 90, 135)}
    hues = {"forehead": 75, "left_cheek": 72, "right_cheek": 45,
            "nose_bridge": 58, "jawline": 80}

    def patch(r, usable=True):
        y0, y1, x0, x1 = boxes[r]
        m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = True
        img = np.zeros((H, W, 3), np.uint8); img[m] = rgb_for_hue(hues[r])
        skin = m if usable else np.zeros_like(m)
        return RegionPatch(r, img, m, skin, STATUS_OK if usable else STATUS_EMPTY,
                           int(m.sum()), int(skin.sum()))

    sel = RegionalConfigurationSelector(skip_unusable=False)
    outs = sel.route_all({r: patch(r) for r in REGION_ORDER}, "img1")
    d = compute_undertone_descriptor(outs[FULL_FACE], build_cielab_vector(outs[FULL_FACE]))
    got = {r: v["undertone"] for r, v in d["regions"].items()}
    want = {r: classify_undertone(h) for r, h in hues.items()}
    check("per-region descriptors from real CIELAB match the intended hues",
          got == want, f"{got} vs {want}")
    check("full face -> Majority Undertone Label Warm (3 of 5)",
          d["majority_undertone"] == WARM and d["n_regions_voted"] == 5)
    single = compute_undertone_descriptor(outs["right_cheek"],
                                          build_cielab_vector(outs["right_cheek"]))
    check("single region -> its own descriptor", single["majority_undertone"] == COOL)

    faces = {r: patch(r) for r in REGION_ORDER}
    faces["forehead"] = patch("forehead", usable=False)
    faces["left_cheek"] = patch("left_cheek", usable=False)
    outs2 = sel.route_all(faces, "img2")
    d2 = compute_undertone_descriptor(outs2[FULL_FACE], build_cielab_vector(outs2[FULL_FACE]))
    check("missing regions do NOT vote (3 measured regions voted)",
          d2["n_regions_voted"] == 3
          and d2["excluded_regions"] == ["forehead", "left_cheek"], str(d2["excluded_regions"]))
    check("  -> 1W/1C/1N three-way tie resolved by closest-to-60 (58 deg, Neutral)",
          d2["majority_undertone"] == NEUTRAL and d2["was_tied"])
    d3 = compute_undertone_descriptor(
        outs2[FULL_FACE], build_cielab_vector(outs2[FULL_FACE], impute_policy=IMPUTE_NAN))
    check("NaN-imputed regions are skipped too", d3["n_regions_voted"] == 3)
    check("nothing measured -> None", compute_undertone_descriptor(outs2[FULL_FACE], None) is None)

    print("\n== step 4: stored separately, one row per image ==")
    row = undertone_record("img2", d2)
    check("row has majority label + per-region hue/descriptor",
          row["majority_undertone"] == NEUTRAL and row["forehead_undertone"] == ""
          and row["jawline_undertone"] == WARM)
    check("row has no skin tone / SCC fields", not any("scc" in k.lower() for k in row))
    check("unmeasured image still gets an (empty) row",
          undertone_record("img3", None)["n_regions_voted"] == 0)

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 9 undertone descriptor")
    ap.add_argument("--smoke", action="store_true", help="run checks without any dataset")
    if ap.parse_args().smoke:
        raise SystemExit(_smoke())
    ap.print_help()
