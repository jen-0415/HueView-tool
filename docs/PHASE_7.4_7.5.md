# Phase 7.4 and 7.5 — Implementation Notes

Files added:

```
src/hueview/regions.py            shared vocabulary (7.3-7.5 + Phase 8)
src/hueview/hsv_skin_filter.py    Phase 7.4
src/hueview/region_selector.py    Phase 7.5
src/hueview/run_region_qa.py      Phase 7 checkpoint driver
src/hueview/__init__.py
configs/hsv_skin_thresholds.json  frozen thresholds (needed again at Phase 14.1)
tests/test_phase7_smoke.py        36 assertions, no dataset required
docs/PHASE_7.4_7.5.md             this file
```

---

## What each stage does

**7.4 — Localized HSV skin filtering.** Takes the five polygon-bounded
patches from 7.3 and removes what is inside the polygon but is not skin:
eyebrows, eyelashes, hair, eyes, lips. Threshold on hue and saturation, leave
Value nearly unconstrained. That last part is the tone-fairness argument — hue
is stable across SCC-1 to SCC-6, brightness is not, so a rule that leans on
Value would systematically strip more from deeper skin tones, in a study whose
whole point is not doing that.

**7.5 — Regional Configuration Selector.** A routing matrix, not a transform.
It holds no parameters and changes no pixels; it decides which of the six
setups gets fed to Phase 8, in what order, and how Full Face is assembled. The
manuscript's phrase is "feeds the isolated regional skin masks one by one",
so it is implemented as a generator.

Despite the build plan's wording, 7.5 has no UI. The user-facing region toggle
is Phase 16.3, and it should import `CONFIGURATIONS` and `REGION_DISPLAY` from
this module rather than hardcoding names.

---

## Contract with 7.1–7.3

7.4 and 7.5 are written against an interface, so they can be reviewed and
tested now and wired up when 7.1–7.3 land. Two functions are expected:

```python
# src/hueview/ssr.py            (Phase 7.2)
def apply_ssr(image_rgb: np.ndarray, sigma: float = 80.0) -> np.ndarray:
    """(224,224,3) uint8 RGB in, (224,224,3) uint8 RGB out, rescaled to 0-255."""

# src/hueview/segmentation.py   (Phase 7.3)
def segment_regions(image_rgb, landmarks=None) -> dict[str, np.ndarray]:
    """{region_name: (224,224) bool}. Keys must be exactly REGION_ORDER.
       Masks must be disjoint — the label-map codec assumes it."""
```

`run_region_qa.py` imports both if they exist and falls back to reference
implementations otherwise, printing which path it took. The fallback masks are
**rectangles** derived from Table 3's coverage percentages. Fine for validating
7.4/7.5; never train on them.

Three things 7.3 must get right, because 7.4 cannot detect any of them:

1. **Disjoint masks.** Table 3's rectangles overlap (the nose bridge band sits
   inside the cheek band). Landmark polygons should not, but assert it.
2. **Left is the subject's left**, which is the right side of the image.
   MediaPipe uses the same convention, so this is automatic — until someone
   adds a `cv2.flip` during augmentation. Augmentation is train-only per Phase
   5, so horizontal flip must either be dropped from the HueView pipeline or
   applied to the masks too, or the per-region table in Appendix 3 becomes
   meaningless.
3. **Masks in the same coordinate frame as the SSR image** (both 224×224,
   both post-crop).

---

## How Phase 8 consumes this

```python
from src.hueview import (
    filter_all_regions, RegionalConfigurationSelector, assert_feature_shapes,
)

patches  = filter_all_regions(ssr_image, geometric_masks)   # 7.4
selector = RegionalConfigurationSelector()                  # 7.5

for cfg in selector.route(patches, image_id=filename):
    cnn_vec = efficientnet_features(cfg.image)              # 8.1 -> 1280-d
    lab_vec = cielab_features(cfg)                          # 8.2 -> 3 or 15
    assert_feature_shapes(cfg, cnn_vec, lab_vec)            # 8.3 checkpoint
    logits  = heads[cfg.config](np.concatenate([cnn_vec, lab_vec]))
```

Two rules for 8.2:

- Average CIELAB over `patch.skin_mask`, never `geometric_mask` and never the
  whole image. `RegionPatch.masked_pixels()` returns the flat `(N, 3)` array
  to convert; use it. Averaging the masked *image* includes the zeroed
  background and drags every L\* toward black — the same failure already
  documented for the Baseline undertone branch.
- Concatenate in `REGION_ORDER`, not `REPORTING_ORDER`. `cfg.component_order()`
  gives the exact sequence.

`cfg.cielab_dim` is 3 for a single region and 15 for `full_face`.

Phase 9's undertone majority vote reads the same per-region CIELAB values, so
it needs no separate routing — iterate `cfg.components` for the `full_face`
output.

---

## Running the checkpoint

```powershell
cd C:\Users\Joanne\HueView-Tool\HueView-tool
.\.venv312\Scripts\Activate.ps1

python -m tests.test_phase7_smoke                    # no dataset needed
python -m src.hueview.run_region_qa --n 12           # contact sheets + stats
python -m src.hueview.run_region_qa --n 40 --compare-hsv-source
```

Outputs: `results/phase7_qa/*.png` (four panels: original | SSR | polygons |
post-HSV skin), `results/phase7_coverage_stats.csv`,
`results/phase7_routing_log.csv`.

The coverage CSV is the evidence for the Sampling Data write-up and for the
likely defense question "how often did your filter fall back?"

---

## Caching recommendation

MediaPipe is ~15–30 ms per image on CPU. Across 29,020 training images times
epochs, recomputing it per batch is hours of waste. SSR is one Gaussian blur
and is cheap.

So: **cache the masks, recompute SSR.** `encode_label_map` packs all five
geometric masks and all five skin masks into one 224×224 uint8 PNG (region id
in bits 0–2, skin flag in bit 3) that compresses to a few KB. Roughly 200 MB
for the whole dataset, versus several GB if you cache SSR images too — which
matters because this has to fit in a Drive upload for Colab.

---

## Three things to raise with your adviser

**1. The six configurations don't match between plan and manuscript.**

The manuscript (Appendix 3, Tables 19–27) evaluates: Forehead, Left Cheek,
Right Cheek, Nose Bridge, Jawline, Full Face (All Five Combined).

The build plan's 7.5 step 2 says: Forehead only, Cheeks combined, Nose Bridge
only, Jawline only, All Regions Combined, Full Face — which invents a "Cheeks
combined" setup, splits "All Regions Combined" from "Full Face", and drops the
individual cheeks.

The code follows the manuscript, because the results tables are already built
to that shape and Phase 12's Cochran's Q runs across those six. Worth a
one-line correction to the plan so the two stop disagreeing.

**2. The HSV thresholds aren't in the manuscript.**

The methodology says HSV filtering happens but gives no numbers, so a panel can
ask where they came from and there is currently no answer in the paper. The
defaults implement Kolkur et al. (2017), *Human Skin Detection Using RGB, HSV
and YCbCr Color Models*: `0° ≤ H ≤ 50°`, `0.23 ≤ S ≤ 0.68`, plus a documented
`V ≥ 0.15` floor that Kolkur's HSV rule does not include (near-black pixels
have numerically unstable hue, so without a floor some pass at random).

Add a threshold table to Stage 3 of Feature Selection and add Kolkur to the
references. `configs/hsv_skin_thresholds.json` is the frozen artifact to quote
from, and Phase 14.1 needs it again at inference time.

**3. HSV-on-SSR looks like a real problem. Test it before training.**

SSR normalizes each of R, G, B against its own blurred surround, which is an
implicit per-channel white balance. It shifts hue and flattens saturation. On
the synthetic test face, a cheek measured:

| | mean hue | mean saturation |
|---|---|---|
| original crop | 44.4° | 0.444 |
| after SSR | 79.2° | 0.167 |

44° is inside the skin band; 79° is not. Saturation fell below the 0.23 floor.
Result: the primary rule retained **0%** on every region and all five fell
through to the geometry fallback. On the original-crop path the same faces
passed cleanly at 40–100% retention.

That test is synthetic — flat colour regions exaggerate what SSR does, and real
photographs will be less extreme. But the direction of the effect is inherent
to SSR, not to the test image, so **run `--compare-hsv-source` on 40 real STW
images before training anything.**

If the SSR path underperforms there too, switch `hsv_source` to `"original"`.
That means the skin/not-skin *decision* is made on the un-normalized crop while
Phase 8 still receives SSR pixels — the output is identical in geometry, only
the pixel selection changes, and it stays consistent with the manuscript's
parallel-streams design (MediaPipe already reads the un-normalized image for
exactly this kind of reason: some decisions are more reliable on natural
colour). Document it in Discussion as an evidenced refinement. Both paths are
implemented; the default is currently the manuscript-literal `"ssr"`.

---

## What is not in this commit

- 7.1 landmarks, 7.2 SSR, 7.3 segmentation (the reference implementations in
  `run_region_qa.py` are QA scaffolding, not deliverables)
- 8.2's actual RGB→XYZ→CIELAB conversion — 7.5 declares the vector width and
  component order, Phase 8 computes the values
- The Phase 9 hue-angle undertone rule
