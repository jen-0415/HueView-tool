"""
Phase 7.5 — Regional Configuration Selector
===========================================

WHAT THIS STAGE DOES

Phase 7.4 produces five refined regional skin masks. Something has to decide
*which* of them gets fed into the Hybrid Feature Extraction module, in what
order, and how the "Full Face" setup is assembled. That is this module.

From the manuscript (System Architecture, Stage 3):

    "These refined regional outputs are subsequently passed to the Regional
    Configuration Selector. This component acts as an automated routing
    matrix, separating the facial data so that the system can process each
    anatomical region independently. Rather than evaluating the face as a
    single combined entity, the 'Selected Configuration Output' establishes
    an iterative pipeline. This pipeline feeds the isolated regional skin
    masks one by one into the downstream Hybrid Skin Tone Feature Extraction
    & Classification module."

So: a generator, not a transform. It holds no learned parameters, changes no
pixels, and produces no features. Its entire job is *iteration order and
composition*, which is why it is 200 lines of routing rather than an
algorithm.

The word "interface" in the build plan is a little misleading here — 7.5 has
no UI. The user-facing region toggle lives in Phase 16.3, and it reads the
configuration names exported from this module (see `CONFIGURATIONS`) so the
two cannot drift apart.


THE SIX CONFIGURATIONS

Appendix 3 of the manuscript is explicit:

    "The HueView framework is evaluated on six region setups: the five
    individual facial regions extracted by the MediaPipe Face Mesh
    (Forehead, Left Cheek, Right Cheek, Jawline, Nose Bridge) and the Full
    Face setup formed by combining ... all five regions."

    -> forehead, left_cheek, right_cheek, nose_bridge, jawline, full_face

DISCREPANCY TO RAISE WITH YOUR ADVISER: the build plan (Phase 7.5, step 2)
lists the six setups as "Forehead only, Cheeks combined, Nose Bridge only,
Jawline only, All Regions Combined, Full Face" — which has a *Cheeks
combined* setup, treats "All Regions Combined" and "Full Face" as two
different things, and drops the individual left/right cheek setups. That
does not match the manuscript, and it does not match Tables 19-27, which
have separate Left Cheek and Right Cheek columns and a single Full Face
column. The manuscript wins here because the results tables are already
built to its shape. This module implements the manuscript's six. If your
adviser wants the plan's version instead, the change is one line in
CONFIGURATIONS plus a new composite builder — but Appendix 3's tables would
need rewriting.


HOW "FULL FACE" IS BUILT

Full Face is the union of all five regions, not the raw 224x224 image. The
distinction matters: an unmasked full face would reintroduce hair,
background and eyes — the exact contamination Phases 7.3 and 7.4 exist to
remove, and the same contamination already documented as a limitation of
the Baseline undertone branch. "Full Face" here means "all the validated
skin on this face, together."

Concretely, for the full_face configuration:

    CNN branch (8.1):    one composite image = SSR pixels wherever ANY
                         region's skin mask is True, zeros elsewhere.
                         Built by having EACH region copy its own pixels
                         into the composite at its own mask — not by
                         reusing one region's already-masked .image as if
                         it held every region's pixels (it can't: .image
                         is zeroed outside that region's own mask).
    CIELAB branch (8.2): 15 features = the five per-region (L*, a*, b*)
                         triples concatenated in REGION_ORDER. NOT the mean
                         over the union — the manuscript specifies "3 values
                         x 5 regions", and a union mean would collapse the
                         regional information that is the point of the study.

The `cielab_dim` field on every ConfigurationOutput tells Phase 8.3 which
of those two shapes to expect: 3 for a single region, 15 for full_face. The
build plan's Phase 8 checkpoint ("confirm your concatenated feature vector
shapes are correct, CNN output dim + 3 or +15") is checkable directly
against that field.


MISSING REGIONS

If MediaPipe placed a bad mesh or a region came back STATUS_EMPTY, the
full_face CIELAB vector has a hole in it. Silently writing zeros would be
wrong — (0, 0, 0) in CIELAB is pure black, which pulls the classifier toward
SCC-6 and would bias the study in precisely the direction it is trying to
measure fairly. Three policies are offered; the default imputes from the
same face's surviving regions, which is the mildest assumption available
(regions of one face are strongly correlated in colour). Every imputation is
recorded in `imputed_regions` and written to the routing log, so the count
is reportable rather than hidden.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

from .regions import (
    MIN_VALID_PIXELS,
    REGION_DISPLAY,
    REGION_ORDER,
    STATUS_EMPTY,
    STATUS_OK,
    STATUS_RANK,
    RegionPatch,
)

# --------------------------------------------------------------------------
# Configuration vocabulary
# --------------------------------------------------------------------------

FULL_FACE = "full_face"

#: The six evaluated setups, in the order Phases 8-11 iterate them.
#: Appendix 3 Tables 20-25 are one confusion matrix per entry here.
CONFIGURATIONS: tuple = REGION_ORDER + (FULL_FACE,)

#: Which regions each configuration draws on. Single-region setups map to
#: themselves; full_face maps to all five. Phase 8.2 reads this to know how
#: many CIELAB triples to concatenate.
CONFIGURATION_COMPONENTS: Dict[str, tuple] = {
    **{name: (name,) for name in REGION_ORDER},
    FULL_FACE: REGION_ORDER,
}

#: CIELAB feature width per configuration: 3 per component region.
CIELAB_DIMS: Dict[str, int] = {
    name: 3 * len(components) for name, components in CONFIGURATION_COMPONENTS.items()
}

# Policies for a component region that produced no usable skin pixels.
IMPUTE_FACE_MEAN = "face_mean"  # average the same face's surviving regions
IMPUTE_NAN = "nan"              # leave NaN, let Phase 8 decide
IMPUTE_ZEROS = "zeros"          # write zeros (NOT recommended — see module docstring)


# --------------------------------------------------------------------------
# What the selector emits
# --------------------------------------------------------------------------


@dataclass
class ConfigurationOutput:
    """
    One unit of work handed to Phase 8.

    Phase 8.1 feeds `image` to EfficientNetB0. Phase 8.2 computes CIELAB
    means over each entry in `components` and concatenates them in order,
    producing a vector of length `cielab_dim`. Phase 8.3 concatenates the two
    and runs the Dense(6, softmax) head.
    """

    image_id: str
    config: str
    #: (H, W, 3) uint8 RGB — SSR pixels inside the configuration's skin
    #: mask, zeros elsewhere. Always full-frame 224x224.
    image: np.ndarray
    #: (H, W) bool — the union of the component regions' skin masks.
    mask: np.ndarray
    #: Ordered {region_name: RegionPatch}, in REGION_ORDER. Length 1 for a
    #: single-region config, 5 for full_face.
    components: "OrderedDict[str, RegionPatch]"
    #: 3 or 15. Assert against this in the Phase 8 checkpoint.
    cielab_dim: int
    n_valid_px: int
    #: Worst status among the component regions.
    status: str
    #: Component regions with no usable pixels, needing imputation in 8.2.
    imputed_regions: List[str] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return REGION_DISPLAY.get(self.config, self.config)

    @property
    def is_usable(self) -> bool:
        return self.n_valid_px >= MIN_VALID_PIXELS

    def component_order(self) -> tuple:
        """The exact region sequence Phase 8.2 must concatenate in."""
        return tuple(self.components.keys())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ConfigurationOutput({self.image_id}, {self.config}, "
            f"cielab_dim={self.cielab_dim}, valid={self.n_valid_px}, "
            f"status={self.status})"
        )


# --------------------------------------------------------------------------
# Composite construction
# --------------------------------------------------------------------------


def build_full_face_composite(
    patches: Dict[str, RegionPatch],
    image_id: str = "",
) -> ConfigurationOutput:
    """
    Assemble the Full Face (All Five Combined) configuration.

    The composite image is the union of the five *skin* masks — not the
    geometric polygons, and not the unmasked face. Its CIELAB width is 15,
    because Phase 8.2 keeps the five regional triples separate rather than
    averaging them into one.
    """
    usable = [p for p in patches.values() if p.is_usable]
    if not usable:
        # Every region failed. Return an empty composite; the selector's
        # skip_unusable filter will drop it, and the caller logs the image.
        any_patch = next(iter(patches.values()))
        h, w = any_patch.image.shape[:2]
        return ConfigurationOutput(
            image_id=image_id,
            config=FULL_FACE,
            image=np.zeros_like(any_patch.image),
            mask=np.zeros((h, w), dtype=bool),
            components=OrderedDict((n, patches[n]) for n in REGION_ORDER if n in patches),
            cielab_dim=CIELAB_DIMS[FULL_FACE],
            n_valid_px=0,
            status=STATUS_EMPTY,
            imputed_regions=list(REGION_ORDER),
            meta={"reason": "all_regions_empty"},
        )

    # Used only as a shape/dtype template for composite/union_mask below --
    # NOT as the pixel source for the whole union. RegionPatch.image is
    # zeroed everywhere outside that region's own mask, so reading from any
    # single region's .image for pixels outside its own footprint would
    # only ever yield zeros there. Each region writes its own real pixels
    # into `composite` at its own mask inside the loop below instead.
    reference = usable[0].image
    union_mask = np.zeros(reference.shape[:2], dtype=bool)
    composite = np.zeros_like(reference)

    components: "OrderedDict[str, RegionPatch]" = OrderedDict()
    imputed: List[str] = []
    worst = STATUS_OK

    for name in REGION_ORDER:
        patch = patches.get(name)
        if patch is None:
            imputed.append(name)
            continue
        components[name] = patch
        if not patch.is_usable:
            imputed.append(name)
            continue
        union_mask |= patch.skin_mask
        # Each region contributes its OWN pixel values at its OWN mask --
        # patch.image is zeroed everywhere outside that region, so pulling
        # from any single region's .image for the whole union (the
        # previous approach) silently dropped every other region's pixels,
        # making the composite numerically identical to whichever region
        # happened to be usable first in REGION_ORDER.
        composite[patch.skin_mask] = patch.image[patch.skin_mask]
        if STATUS_RANK.get(patch.status, 99) > STATUS_RANK.get(worst, 0):
            worst = patch.status

    return ConfigurationOutput(
        image_id=image_id,
        config=FULL_FACE,
        image=composite,
        mask=union_mask,
        components=components,
        cielab_dim=CIELAB_DIMS[FULL_FACE],
        n_valid_px=int(union_mask.sum()),
        status=worst,
        imputed_regions=imputed,
        meta={"n_component_regions": len(components) - len(imputed)},
    )


def _build_single_region(
    region: str,
    patch: RegionPatch,
    image_id: str = "",
) -> ConfigurationOutput:
    """Wrap one region as a configuration. CIELAB width 3."""
    mask = patch.skin_mask if patch.skin_mask is not None else patch.geometric_mask
    image = np.zeros_like(patch.image)
    image[mask] = patch.image[mask]
    return ConfigurationOutput(
        image_id=image_id,
        config=region,
        image=image,
        mask=mask,
        components=OrderedDict([(region, patch)]),
        cielab_dim=CIELAB_DIMS[region],
        n_valid_px=int(mask.sum()),
        status=patch.status,
        imputed_regions=[] if patch.is_usable else [region],
        meta={"tier": patch.meta.get("tier", "none")},
    )


# --------------------------------------------------------------------------
# The selector
# --------------------------------------------------------------------------


class RegionalConfigurationSelector:
    """
    The automated routing matrix described in Stage 3 of the manuscript.

    Usage — the iterative pipeline the manuscript describes:

        selector = RegionalConfigurationSelector()
        patches = filter_all_regions(ssr_image, geometric_masks)

        for cfg in selector.route(patches, image_id=filename):
            cnn_vec   = efficientnet_features(cfg.image)          # Phase 8.1
            lab_vec   = cielab_features(cfg)                      # Phase 8.2
            assert lab_vec.shape[0] == cfg.cielab_dim             # 8.3 checkpoint
            logits    = models[cfg.config].predict([cnn_vec, lab_vec])

    Args:
        configurations:  which of the six setups to emit, in order. Restrict
                         this for a faster deployment path (Phase 14.7) or
                         to run a single region during debugging.
        skip_unusable:   drop configurations with fewer than `min_valid_px`
                         pixels instead of yielding them. True during
                         training (a 12-pixel patch is noise); consider False
                         at inference so the Phase 16 UI can show "region
                         unavailable" rather than silently omitting a card.
        min_valid_px:    the usability floor.
        impute_policy:   what Phase 8.2 should do about missing regions in
                         the full_face vector. Recorded on the output, not
                         acted on here — the selector routes, it does not
                         compute features.
    """

    def __init__(
        self,
        configurations: Sequence[str] = CONFIGURATIONS,
        skip_unusable: bool = True,
        min_valid_px: int = MIN_VALID_PIXELS,
        impute_policy: str = IMPUTE_FACE_MEAN,
    ):
        unknown = [c for c in configurations if c not in CONFIGURATION_COMPONENTS]
        if unknown:
            raise ValueError(
                f"Unknown configuration(s) {unknown}. Valid: {list(CONFIGURATION_COMPONENTS)}"
            )
        if impute_policy not in (IMPUTE_FACE_MEAN, IMPUTE_NAN, IMPUTE_ZEROS):
            raise ValueError(f"Unknown impute_policy {impute_policy!r}")

        self.configurations = tuple(configurations)
        self.skip_unusable = skip_unusable
        self.min_valid_px = min_valid_px
        self.impute_policy = impute_policy

        # Running tallies for the Phase 7 checkpoint and the Sampling Data
        # write-up. Reset with `reset_stats()` between splits.
        self.stats: Dict[str, int] = {
            "images_routed": 0,
            "configs_emitted": 0,
            "configs_skipped": 0,
        }

    def __len__(self) -> int:
        return len(self.configurations)

    def reset_stats(self) -> None:
        for k in self.stats:
            self.stats[k] = 0

    def route(
        self,
        patches: Dict[str, RegionPatch],
        image_id: str = "",
    ) -> Iterator[ConfigurationOutput]:
        """
        Yield one ConfigurationOutput per configuration, in fixed order.

        This is a generator on purpose. Materializing all six configurations
        for a batch would hold six 224x224x3 arrays per image in memory
        (~900 KB per face), which at batch size 32 is 29 MB of masks alone
        before EfficientNetB0 has allocated anything. Yielding one at a time
        keeps the footprint flat and matches the manuscript's "one by one"
        wording literally.

        Ordering is deterministic across runs. That is not cosmetic: Phase 11
        joins per-region predictions back into Table 19 by position, and
        Phase 12's Cochran's Q test requires each image's six outcomes to
        line up in the same column every time.
        """
        self.stats["images_routed"] += 1

        for config in self.configurations:
            if config == FULL_FACE:
                out = build_full_face_composite(patches, image_id=image_id)
            else:
                patch = patches.get(config)
                if patch is None:
                    self.stats["configs_skipped"] += 1
                    continue
                out = _build_single_region(config, patch, image_id=image_id)

            out.meta["impute_policy"] = self.impute_policy

            if self.skip_unusable and out.n_valid_px < self.min_valid_px:
                self.stats["configs_skipped"] += 1
                continue

            self.stats["configs_emitted"] += 1
            yield out

    def route_all(
        self,
        patches: Dict[str, RegionPatch],
        image_id: str = "",
    ) -> "OrderedDict[str, ConfigurationOutput]":
        """
        Eager version of `route`, keyed by configuration name.

        Convenient at inference time (Phase 14.6), where you want all six
        results in one dict to assemble the response JSON, and the memory
        argument above does not apply because you are handling one image.
        """
        return OrderedDict((c.config, c) for c in self.route(patches, image_id=image_id))

    # ---------------------------------------------------------------- logs

    def routing_record(
        self,
        image_id: str,
        outputs: Sequence[ConfigurationOutput],
        extra: Optional[Dict] = None,
    ) -> Dict:
        """
        Build one flat row describing how a single image was routed.

        Collect these into `results/phase7_routing_log.csv`. It is the
        evidence for two things the panel is likely to ask about: how many
        regions needed the relaxed or fallback filter, and how many images
        contributed fewer than five regions to their Full Face vector.

        It also matters for Phase 12: Cochran's Q compares HueView across six
        related groups and assumes every image has an outcome in each. Images
        that lost a region are the ones to check first if the omnibus test
        behaves strangely.
        """
        by_config = {o.config: o for o in outputs}
        row: Dict = {"image_id": image_id, "n_configs": len(outputs)}

        for config in self.configurations:
            out = by_config.get(config)
            row[f"{config}_status"] = out.status if out else "skipped"
            row[f"{config}_valid_px"] = out.n_valid_px if out else 0

        ff = by_config.get(FULL_FACE)
        row["full_face_regions_present"] = (
            len(REGION_ORDER) - len(ff.imputed_regions) if ff else 0
        )
        row["full_face_imputed"] = "|".join(ff.imputed_regions) if ff else ""

        if extra:
            row.update(extra)
        return row


# --------------------------------------------------------------------------
# Shape contract helper (Phase 8.3 checkpoint)
# --------------------------------------------------------------------------


def assert_feature_shapes(
    output: ConfigurationOutput,
    cnn_vector: np.ndarray,
    cielab_vector: np.ndarray,
    cnn_dim: int = 1280,
) -> int:
    """
    Verify the fused vector before it reaches the Dense layer.

    The build plan makes this an explicit Phase 8 checkpoint. Calling it
    once per configuration during your first training run catches the
    failure mode where a 15-element vector is handed to a head built for 3 —
    which does not crash if the Dense layer was built lazily, it just trains
    on garbage and produces a plausible-looking but meaningless accuracy.

    Returns the expected fused width (cnn_dim + cielab_dim).
    """
    if cnn_vector.shape[-1] != cnn_dim:
        raise ValueError(
            f"[{output.config}] CNN vector is {cnn_vector.shape[-1]}-d, expected {cnn_dim}. "
            "EfficientNetB0 with include_top=False and global average pooling gives 1280."
        )
    if cielab_vector.shape[-1] != output.cielab_dim:
        raise ValueError(
            f"[{output.config}] CIELAB vector is {cielab_vector.shape[-1]}-d, expected "
            f"{output.cielab_dim} (3 per region x {len(output.components)} region(s)). "
            "Check that Phase 8.2 concatenated in REGION_ORDER."
        )
    return cnn_dim + output.cielab_dim
