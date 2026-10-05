# Fixing the Phase 7.3 Region Definitions

The cheek index groups in `segment_regions.py` are wrong. Both cheek masks
currently span the entire face and overlap each other almost completely.
This is a guide to fixing them, verifying the fix, and knowing what has to
be re-run afterwards.

Found during the Phase 7.4 checkpoint, which measures 7.3's masks as a side
effect of filtering them.

Roughly 20 minutes of work plus one 7.3 re-run.

---

## 1. What is wrong

Two naming conventions are in play and they are mirror images of each other.

MediaPipe names landmark groups by **subject anatomy**:
`FACE_LANDMARKS_LEFT_EYE` is the subject's left eye, which appears on the
**right** side of a front-facing photo.

Table 3 of the manuscript names regions by **image position**: "Left Cheek —
left 15–45% width."

So an image-left region must draw its landmarks from MediaPipe's `RIGHT_*`
groups. `derive_regions.py` documents this correctly at the top of the file
and then does the opposite in the cheek block.

### Bug 1 — each cheek is built from opposite sides of the face

```python
left_band          = [i for i in FACE_OVAL if ... and xs[i] < face_center_x]  # image-LEFT
left_eye_lower_pts = [i for i in LEFT_EYE  if ys[i] >= left_eye_mid_y]        # image-RIGHT
left_cheek = left_band + left_nose_edge + left_eye_lower_pts
```

`left_band` selects face-oval points 93, 127, 132, 234 — the image-left edge.
`left_eye_lower_pts` selects 249, 362, 373, 374, 380, 381, 382, 390 — the
image-right eye. Verified against the official groups:

| group | face-oval points | eye points |
|---|---|---|
| `left_cheek` | 93, 127, 132, 234 (image-left) | 249, 362, 373, 374, 380, 381, 382, 390 (`LEFT_EYE`, image-**right**) |
| `right_cheek` | 323, 356, 361, 454 (image-right) | 7, 133, 144, 145, 153, 154, 155, 163 (`RIGHT_EYE`, image-**left**) |

`build_region_mask` takes `cv2.convexHull` of the combined set. A convex hull
is the smallest polygon containing every point, so it necessarily stretches
from one face edge across to the opposite eye and fills in everything
between. Mirrored on the other side, both cheeks end up covering most of the
face — visible as the two overlapping wedges in `figures/region_reference.png`.

### Bug 2 — nose points pulled into the cheeks

```python
left_nose_edge = [i for i in NOSE if ... and xs[i] < face_center_x]
```

Midline nose landmarks sit fractionally left of centre, so they qualify.
`left_cheek` shares 8 indices with `nose_bridge` (2, 5, 6, 19, 45, 94, 195,
197); `right_cheek` shares 2 (4, 275). Shared points guarantee overlapping
hulls even after Bug 1 is fixed.

### Bug 3 — latent cheek/jawline overlap

The cheek band's lower bound is `lips_top_y`; the jawline's upper bound is
`nose_tip_y + 0.12`. These are computed independently and can cross, giving
the two regions an overlapping y-window. It did not fire on the reference
photo but will on some faces.

### Why it matters

- **Disjointness.** `encode_label_map` stores one region id per pixel, so a
  contested pixel is assigned to whichever region comes later in
  `REGION_ORDER` and is lost from the earlier one. Nose bridge is small and
  mostly inside the cheek hulls; it would effectively disappear.
- **Feature independence.** Phase 8.2 builds a 15-element CIELAB vector as
  five separate regional means. Overlapping masks average the same pixels
  into several regions, so features reported as independent measurements are
  partly the same measurement repeated.
- **SOP3.** Appendix 3 reports separate Left Cheek and Right Cheek columns.
  With these masks both cheeks cover nearly the same area, so the two columns
  would come out with similar accuracy — which looks like a finding, reads as
  reasonable, and means nothing. That is worse than a crash.

---

## 2. Replace `derive_regions.py`

Drop in the corrected file. Changes:

- Cheeks pair with the eye on the **same image side**, via explicit
  `IMAGE_LEFT_EYE` / `IMAGE_RIGHT_EYE` aliases so the mirror confusion is
  stated once instead of implied everywhere
- Nose points dropped from the cheeks entirely (Table 3 says "lateral
  cheek"), plus a lateral keep-out margin scaled to the nose's own width
- Cheek band bounded by `min(lips_top_y, jaw_cutoff_y)`, fixing Bug 3
- Per-side x filter applied to the eye points as well, so swapping the groups
  back cannot silently reintroduce Bug 1
- `check_disjoint()` reports shared indices after every run
- Detection code moved under `if __name__ == "__main__":` — as written, the
  file ran MediaPipe at import time, so importing `derive_regions()` would
  trigger a detection on the hardcoded `IMAGE_PATH`

The logic was tested on a constructed landmark layout: cheeks separated
cleanly by side with no shared indices against each other, the nose, or the
jaw. Constructed positions prove the logic, not what real faces will produce
— that is what step 4 is for.

---

## 3. Pick a good reference face

`IMAGE_PATH` at the top of the file. This matters more than it looks.

The index groups are derived **once**, from **one** face, then frozen into
`segment_regions.py` and applied to all 43,000+ images. The derivation uses
geometric rules (y-thresholds relative to eyebrows, nose tip, lips), so a
reference face with an unusual pose produces index groups that are subtly
wrong everywhere else.

Choose one that is frontal, neutral expression, eyes open, no glasses, hair
off the forehead, evenly lit. The existing default is reasonable if it meets
that description.

---

## 4. Run and verify

```powershell
cd C:\Users\Joanne\HueView-Tool\HueView-tool
.\.venv312\Scripts\Activate.ps1
python src\hueview\derive_regions.py
```

**Check the console.** Should end with:

```
Disjointness (index level): OK, no shared landmarks.
```

If it lists clashes instead, stop — do not paste. Send the output on.

**Check `region_reference.png`.** Green (`left_cheek`) on the image-left
cheek only, orange (`right_cheek`) on the image-right cheek only, with a
visible gap between them across the nose. Neither should cross the midline
or touch an eye on the far side.

The other three were already fine and should be unchanged: blue covering the
forehead down to the eyebrows, red a narrow nose midline, magenta the jaw
below the lips.

**Repeat on two or three more faces** with different tones and face shapes.
Edit `IMAGE_PATH`, re-run, look. You are checking that the derivation is
stable, not just that it worked once. Note which face's output you finally
keep — that belongs in Chapter 3.

---

## 5. Paste into `segment_regions.py`

Copy the printed index groups into the `REGIONS` dict.

**Do not hand-edit individual indices.** These are generated output. Nobody,
including you in three months, can verify by reading that landmark 390
belongs in `left_cheek`. Change the derivation and re-run; the same reason
you would fix a formula rather than the cell it produced.

Update the comment above `REGIONS` to say v4 rather than v3.

Optional guard — add directly below the `REGIONS` dict so overlapping groups
can never be pasted in again unnoticed:

```python
from itertools import combinations

_clashes = {
    f"{a} + {b}": sorted(set(REGIONS[a]) & set(REGIONS[b]))
    for a, b in combinations(REGIONS, 2)
    if set(REGIONS[a]) & set(REGIONS[b])
}
if _clashes:
    raise ValueError(
        f"REGIONS index groups overlap, so their convex hulls will too: {_clashes}. "
        "Re-derive with derive_regions.py rather than editing indices by hand."
    )
```

This raises at import, which is intended — 7.3 should not be runnable with
overlapping groups. `run_region_qa.py` catches the import failure and prints
the reason rather than falling back silently.

---

## 6. Pixel-level check on real faces

Index disjointness is necessary but **not sufficient**: two groups with no
shared indices can still produce overlapping convex hulls. This measures the
actual masks.

```powershell
python -m src.hueview.run_region_qa --n 40 --overlap-only
```

Confirm the banner reads `segment_regions + landmarks.npy (7.1/7.3)`. If it
says `REFERENCE fallback`, the import failed and the reason is printed
directly below.

Every pair should read close to 0%. Anything above 2% is flagged.

Then look at the contact sheets, panel 3, across several faces:

```powershell
python -m src.hueview.run_region_qa --n 12
```

This is the real generalisation test — frozen indices rendered on faces other
than the reference.

---

## 7. Re-run 7.3

`data/processed/regions/` was generated from the bad groups and is stale.

```powershell
python src\hueview\segment_regions.py
```

**Before starting this**, check with whoever owns Phase 8 whether it reads
`data/processed/regions/` at all. If Phase 8 instead recomputes masks from
`landmarks.npy` (or uses the cached label maps described in
`docs/PHASE_7.4_7.5.md`), then `regions/` is only a QA artifact and 43,000
files do not need regenerating. That check costs a message and may save the
whole re-run.

`landmarks.npy` is unaffected — 7.1 did not change, so there is no need to
re-run landmark extraction.

---

## 8. Then unblock 7.4

Retention figures, filter-tier counts and the `hsv_source` decision all have
to be measured on corrected masks. The current polygons contain hair, eyes
and nose, which depresses retention for reasons unrelated to the HSV filter —
tuning thresholds against that would produce a filter calibrated on
contamination.

```powershell
python -m src.hueview.run_region_qa --n 40 --compare-hsv-source
```

---

## Checklist

- [ ] Corrected `derive_regions.py` in place
- [ ] `IMAGE_PATH` points at a frontal, neutral, well-lit face
- [ ] Console prints `Disjointness (index level): OK`
- [ ] `region_reference.png` — cheeks on their own sides, gap across the nose
- [ ] Verified on 2–3 additional faces
- [ ] Index groups pasted into `segment_regions.py`, comment updated to v4
- [ ] Optional guard added
- [ ] `--overlap-only` on 40 real faces reads ~0% for every pair
- [ ] Contact sheets panel 3 checked across several faces
- [ ] Asked Phase 8 owner whether `regions/` needs regenerating
- [ ] 7.3 re-run if required
- [ ] Reference face recorded for Chapter 3
