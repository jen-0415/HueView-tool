"""
Phase 7.4 / 7.5 smoke test.

Builds a synthetic 224x224 "face" with known non-skin features (dark
eyebrows, red lips, dark hair) and checks that:

  1. HSV filtering removes the eyebrows and lips but keeps cheek skin
  2. escalation tiers fire in the right order
  3. the selector yields exactly six configurations in fixed order
  4. full_face has cielab_dim 15 and single regions have 3
  5. the label-map codec round-trips
  6. assert_feature_shapes catches a mismatched vector

Run from the repo root:
    python -m tests.test_phase7_smoke
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hueview import (  # noqa: E402
    CIELAB_DIMS, CONFIGURATIONS, FULL_FACE, REGION_ORDER, STATUS_OK,
    FilterConfig, RegionalConfigurationSelector, assert_feature_shapes,
    decode_label_map, encode_label_map, filter_all_regions, skin_mask_hsv,
    summarize,
)
from src.hueview.run_region_qa import reference_region_masks, reference_ssr  # noqa: E402

PASS, FAIL = "  PASS", "  FAIL"
failures = []


def check(name, condition, detail=""):
    print(f"{PASS if condition else FAIL}  {name}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        failures.append(name)


def synthetic_face(size=224):
    """Mid-brown skin oval on a blue-grey background, plus eyebrows and lips."""
    img = np.full((size, size, 3), (70, 80, 110), dtype=np.uint8)   # background
    yy, xx = np.mgrid[0:size, 0:size]
    face = ((xx - size / 2) ** 2 / (size * 0.34) ** 2
            + (yy - size / 2) ** 2 / (size * 0.44) ** 2) <= 1.0
    img[face] = (168, 122, 92)                                       # skin
    img[int(size * .30):int(size * .34), int(size * .22):int(size * .44)] = (35, 25, 20)  # brow L
    img[int(size * .30):int(size * .34), int(size * .56):int(size * .78)] = (35, 25, 20)  # brow R
    img[int(size * .72):int(size * .79), int(size * .38):int(size * .62)] = (172, 40, 55)  # lips
    img[:int(size * .12), :] = (28, 22, 18)                          # hair
    # Mild per-pixel noise so morphology has something realistic to clean.
    rng = np.random.default_rng(0)
    return np.clip(img.astype(np.int16) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)


def main():
    print("=" * 66)
    print("Phase 7.4 / 7.5 smoke test")
    print("=" * 66)

    face = synthetic_face()
    ssr = reference_ssr(face)
    check("SSR output shape/dtype", ssr.shape == face.shape and ssr.dtype == np.uint8)

    masks = reference_region_masks(face.shape[:2])
    check("five region masks produced", set(masks) == set(REGION_ORDER))
    overlap = sum(m.astype(int) for m in masks.values()).max()
    check("region masks are disjoint", overlap <= 1, f"max overlap {overlap}")

    # --- 7.4 on the ORIGINAL image (clean colour, predictable behaviour) ---
    cfg = FilterConfig(hsv_source="original")
    patches = filter_all_regions(ssr, masks, config=cfg, reference_image=face)
    stats = summarize(patches)
    print("\n  region        status              valid/geom      retention")
    for name in REGION_ORDER:
        s = stats[name]
        print(f"  {name:<13} {s['status']:<19} {s['n_valid_px']:>6}/{s['n_geometric_px']:<7} {s['retention']:.1%}")

    check("all five regions returned", set(patches) == set(REGION_ORDER))
    check("cheek skin retained", patches["left_cheek"].retention > 0.5,
          f"{patches['left_cheek'].retention:.1%}")

    # Eyebrows sit in the forehead polygon's lower band; lips in the jawline's.
    brow_band = np.zeros(face.shape[:2], dtype=bool)
    brow_band[int(224 * .30):int(224 * .34), :] = True
    brow_in_geom = int((patches["forehead"].geometric_mask & brow_band).sum())
    brow_in_skin = int((patches["forehead"].skin_mask & brow_band).sum())
    check("eyebrow pixels stripped from forehead",
          brow_in_geom == 0 or brow_in_skin < 0.25 * brow_in_geom,
          f"{brow_in_skin}/{brow_in_geom} kept")

    lip_band = np.zeros(face.shape[:2], dtype=bool)
    lip_band[int(224 * .72):int(224 * .79), int(224 * .38):int(224 * .62)] = True
    lip_in_geom = int((patches["jawline"].geometric_mask & lip_band).sum())
    lip_in_skin = int((patches["jawline"].skin_mask & lip_band).sum())
    check("lip pixels stripped from jawline",
          lip_in_geom == 0 or lip_in_skin < 0.5 * lip_in_geom,
          f"{lip_in_skin}/{lip_in_geom} kept")

    check("skin mask never exceeds polygon",
          all((p.skin_mask & ~p.geometric_mask).sum() == 0 for p in patches.values()))
    check("masked_pixels returns (N,3)",
          patches["left_cheek"].masked_pixels().shape[1] == 3)

    # --- escalation ------------------------------------------------------
    impossible = FilterConfig(hsv_source="original", min_retention=0.999)
    esc = filter_all_regions(ssr, masks, config=impossible, reference_image=face)
    check("impossible retention triggers escalation",
          all(p.status != STATUS_OK for p in esc.values()),
          {p.status for p in esc.values()}.pop() if esc else "")

    empty_masks = {n: np.zeros(face.shape[:2], dtype=bool) for n in REGION_ORDER}
    dead = filter_all_regions(ssr, empty_masks, config=cfg, reference_image=face)
    check("empty polygons return STATUS_EMPTY without raising",
          all(p.status == "empty" for p in dead.values()))

    # --- 7.5 routing -----------------------------------------------------
    selector = RegionalConfigurationSelector()
    outputs = list(selector.route(patches, image_id="synthetic.jpg"))
    check("six configurations emitted", len(outputs) == 6, f"got {len(outputs)}")
    check("configuration order is canonical",
          [o.config for o in outputs] == list(CONFIGURATIONS))

    by_cfg = {o.config: o for o in outputs}
    check("single-region cielab_dim == 3",
          all(by_cfg[r].cielab_dim == 3 for r in REGION_ORDER))
    check("full_face cielab_dim == 15", by_cfg[FULL_FACE].cielab_dim == 15)
    check("CIELAB_DIMS table agrees",
          all(by_cfg[c].cielab_dim == CIELAB_DIMS[c] for c in CONFIGURATIONS))
    check("full_face components in REGION_ORDER",
          by_cfg[FULL_FACE].component_order() == tuple(REGION_ORDER))

    union = int(sum(p.skin_mask.sum() for p in patches.values()))
    check("full_face mask == union of region skin masks",
          by_cfg[FULL_FACE].n_valid_px == union,
          f"{by_cfg[FULL_FACE].n_valid_px} vs {union}")
    check("full_face image is masked, not the raw face",
          (by_cfg[FULL_FACE].image[~by_cfg[FULL_FACE].mask] == 0).all())
    check("no imputed regions on a healthy face",
          by_cfg[FULL_FACE].imputed_regions == [])

    record = selector.routing_record("synthetic.jpg", outputs)
    check("routing record has a column per configuration",
          all(f"{c}_status" in record for c in CONFIGURATIONS))
    check("routing record counts 5 present regions",
          record["full_face_regions_present"] == 5)

    # Degraded face: three regions wiped out.
    partial = dict(patches)
    for name in ("nose_bridge", "jawline", "right_cheek"):
        partial[name] = dead[name]
    sel2 = RegionalConfigurationSelector()
    out2 = {o.config: o for o in sel2.route(partial, image_id="partial.jpg")}
    check("unusable single regions are skipped",
          all(r not in out2 for r in ("nose_bridge", "jawline", "right_cheek")))
    check("full_face survives with the remaining regions", FULL_FACE in out2)
    check("imputed regions recorded",
          set(out2[FULL_FACE].imputed_regions) == {"nose_bridge", "jawline", "right_cheek"})
    check("skip_unusable=False keeps all six",
          len(list(RegionalConfigurationSelector(skip_unusable=False)
                   .route(partial, image_id="p"))) == 6)

    # --- shape contract --------------------------------------------------
    cnn = np.zeros(1280, dtype=np.float32)
    check("assert_feature_shapes accepts 1280+15",
          assert_feature_shapes(by_cfg[FULL_FACE], cnn, np.zeros(15)) == 1295)
    try:
        assert_feature_shapes(by_cfg[FULL_FACE], cnn, np.zeros(3))
        check("assert_feature_shapes rejects wrong CIELAB width", False)
    except ValueError:
        check("assert_feature_shapes rejects wrong CIELAB width", True)

    # --- codec -----------------------------------------------------------
    label = encode_label_map(patches, face.shape[:2])
    restored = decode_label_map(label, ssr)
    check("label map round-trips geometry",
          all(np.array_equal(restored[n].geometric_mask, patches[n].geometric_mask)
              for n in REGION_ORDER))
    check("label map round-trips skin masks",
          all(np.array_equal(restored[n].skin_mask, patches[n].skin_mask)
              for n in REGION_ORDER))
    check("label map fits in uint8", label.dtype == np.uint8 and label.max() <= 13)

    # --- threshold config freeze ----------------------------------------
    out = Path("/tmp/hsv_skin_thresholds.json")
    FilterConfig().to_json(out)
    check("FilterConfig JSON round-trips", FilterConfig.from_json(out) == FilterConfig())

    check("skin_mask_hsv returns a bool mask",
          skin_mask_hsv(face, FilterConfig().primary).dtype == bool)

    print("=" * 66)
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
