"""
HueView pipeline package (Phases 7-9).

Phase 7.1  landmarks.py       MediaPipe Face Mesh, 468 points   [to build]
Phase 7.2  ssr.py             Single-Scale Retinex, sigma = 80  [to build]
Phase 7.3  segmentation.py    landmark polygons -> region masks [to build]
Phase 7.4  hsv_skin_filter.py localized HSV skin filtering      [this commit]
Phase 7.5  region_selector.py the routing matrix                [this commit]

`regions.py` holds the vocabulary all five share.
"""

from .regions import (  # noqa: F401
    MIN_VALID_PIXELS,
    REGION_DISPLAY,
    REGION_IDS,
    REGION_ORDER,
    REPORTING_ORDER,
    STATUS_EMPTY,
    STATUS_GEOMETRY_FALLBACK,
    STATUS_OK,
    STATUS_RELAXED,
    RegionPatch,
    decode_label_map,
    encode_label_map,
)
from .hsv_skin_filter import (  # noqa: F401
    DEFAULT_CONFIG,
    PRIMARY_THRESHOLDS,
    RELAXED_THRESHOLDS,
    FilterConfig,
    HSVSkinThresholds,
    filter_all_regions,
    filter_region,
    skin_mask_hsv,
    summarize,
)
from .region_selector import (  # noqa: F401
    CIELAB_DIMS,
    CONFIGURATION_COMPONENTS,
    CONFIGURATIONS,
    FULL_FACE,
    ConfigurationOutput,
    RegionalConfigurationSelector,
    assert_feature_shapes,
    build_full_face_composite,
)

__all__ = [
    "REGION_ORDER", "REPORTING_ORDER", "REGION_DISPLAY", "REGION_IDS",
    "RegionPatch", "encode_label_map", "decode_label_map", "MIN_VALID_PIXELS",
    "STATUS_OK", "STATUS_RELAXED", "STATUS_GEOMETRY_FALLBACK", "STATUS_EMPTY",
    "HSVSkinThresholds", "FilterConfig", "DEFAULT_CONFIG",
    "PRIMARY_THRESHOLDS", "RELAXED_THRESHOLDS",
    "skin_mask_hsv", "filter_region", "filter_all_regions", "summarize",
    "CONFIGURATIONS", "CONFIGURATION_COMPONENTS", "CIELAB_DIMS", "FULL_FACE",
    "ConfigurationOutput", "RegionalConfigurationSelector",
    "build_full_face_composite", "assert_feature_shapes",
]
