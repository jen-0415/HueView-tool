"""
Suffix-aware multi-root image path resolver
===========================================

The dataset was assembled in batches. When a later batch contained a
filename already used by an earlier one, the CSV recorded the collision
with a "(n)" marker while the file itself was written into that batch's
own folder under the ORIGINAL name. So the suffix identifies WHICH FOLDER
the row refers to — it is not a duplicate artifact.

Verified empirically (test_suffix_hypothesis.py, n=250 per group,
matching each row's stored mean_Y against every root):

    no suffix   -> processed        247/250 (99%)
    "(2)"       -> c2_processed     250/250 (100%)

So:

    MST-8/foo.jpg        ->  processed/MST-8/foo.jpg
    MST-8/foo (2).jpg    ->  c2_processed/MST-8/foo.jpg
    MST-2/bar.png        ->  v5_processed/MST-2/bar.bmp   (the v5 batch)

`processed - 7-26` matched suffixed rows at 96%, so it appears to be an
earlier snapshot of the same batch as c2_processed. It's kept only as a
fallback, never as a first choice.

Why the routing matters: the SCC and illumination labels on a suffixed
row were computed from the c2_processed rendering. Stripping the suffix
and serving the `processed` version pairs ~28% of the dataset with labels
derived from a different image. Nothing errors — the numbers just come
out wrong.

Import resolve_image_path from here everywhere so Baseline and HueView
read identical bytes for identical rows.
"""

import re
from pathlib import Path

IMAGES = Path("data/processed/images")

ROOT_PROCESSED = IMAGES / "processed"
ROOT_C2 = IMAGES / "c2_processed"
ROOT_726 = IMAGES / "c1_processed"
ROOT_V5 = IMAGES / "v5_processed"

# Search order per suffix value. First hit wins.
# None = no suffix on the row.
SUFFIX_ROUTING = {
    None: [ROOT_PROCESSED, ROOT_C2, ROOT_726],
    1:    [ROOT_726, ROOT_C2, ROOT_PROCESSED],
    2:    [ROOT_C2, ROOT_726, ROOT_PROCESSED],
    3:    [ROOT_726, ROOT_C2, ROOT_PROCESSED],
}
FALLBACK_ORDER = [ROOT_PROCESSED, ROOT_C2, ROOT_726]

CANDIDATE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

# Matches " (2)" or "(1)" immediately before the extension.
SUFFIX_RE = re.compile(r"\s*\((\d+)\)$")

# Kept for backwards compatibility with earlier scripts.
DEFAULT_ROOTS = [ROOT_PROCESSED, ROOT_C2, ROOT_726]
V5_ROOT = ROOT_V5


def parse_suffix(stem: str):
    """'foo (2)' -> ('foo', 2);  'foo(1)' -> ('foo', 1);  'foo' -> ('foo', None)."""
    m = SUFFIX_RE.search(stem)
    if not m:
        return stem, None
    return stem[:m.start()], int(m.group(1))


def strip_dup_suffix(stem: str) -> str:
    """Backwards-compatible helper: returns the stem without any '(n)'."""
    return parse_suffix(stem)[0]


def resolve_image_path(rel_path, roots=None, v5_root=None):
    """
    Resolve a manifest filename entry to the physical file it refers to.

    Args:
        rel_path: manifest filename value, e.g. "MST-8/foo (2).jpg"
        roots:    optional override of the search order (ignores routing)
        v5_root:  optional override for the v5 batch root

    Returns:
        (Path or None, rule, root_name)

        rule is one of:
          exact          - no suffix, found in its routed root
          suffix_routed  - "(n)" row found in the root that suffix maps to
          v5_bmp         - found in v5_processed as .bmp
          fallback       - found somewhere outside its routed order
          not_found
    """
    v5 = Path(v5_root) if v5_root is not None else ROOT_V5

    rel = Path(str(rel_path).replace("\\", "/"))
    subdir, raw_stem, ext = rel.parent, rel.stem, rel.suffix
    stem, suffix = parse_suffix(raw_stem)

    if roots is not None:
        order = [Path(r) for r in roots]
    else:
        order = SUFFIX_ROUTING.get(suffix, FALLBACK_ORDER)

    # --- Primary: the root this suffix routes to ---
    for i, root in enumerate(order):
        p = root / subdir / f"{stem}{ext}"
        if p.is_file():
            rule = "exact" if (suffix is None and i == 0) else (
                "suffix_routed" if i == 0 else "fallback")
            return p, rule, root.name

    # --- v5 batch: CSV records .png, disk holds .bmp ---
    p = v5 / subdir / f"{stem}.bmp"
    if p.is_file():
        return p, "v5_bmp", v5.name

    # --- Last resort: any known extension, any root ---
    for root in order + [v5]:
        for alt in CANDIDATE_EXTS:
            if alt == ext:
                continue
            p = root / subdir / f"{stem}{alt}"
            if p.is_file():
                return p, "fallback", root.name

    # --- Also try the literal name with the suffix intact ---
    for root in order + [v5]:
        p = root / subdir / f"{raw_stem}{ext}"
        if p.is_file():
            return p, "fallback", root.name

    return None, "not_found", None


def find_all_matches(rel_path, roots=None, v5_root=None):
    """
    Every file across all roots that shares this entry's base name.
    Used by the diagnostic scripts to detect cross-root conflicts.
    """
    v5 = Path(v5_root) if v5_root is not None else ROOT_V5
    order = [Path(r) for r in roots] if roots is not None else FALLBACK_ORDER

    rel = Path(str(rel_path).replace("\\", "/"))
    subdir, raw_stem = rel.parent, rel.stem
    stem, _ = parse_suffix(raw_stem)
    stems = {raw_stem, stem}

    hits = []
    for root in order + [v5]:
        folder = root / subdir
        if not folder.is_dir():
            continue
        for p in folder.iterdir():
            if p.is_file() and p.stem in stems:
                hits.append(p)
    return hits


if __name__ == "__main__":
    # Report how the rare "(1)" rows behave — only 18 exist, too few to
    # infer a routing rule from statistically, so check them individually.
    import pandas as pd
    import cv2

    MANIFEST = Path("data/processed/manifest.csv")
    if not MANIFEST.exists():
        print(f"!! {MANIFEST} not found.")
        raise SystemExit

    m = pd.read_csv(MANIFEST)
    col = "filename" if "filename" in m.columns else m.columns[0]
    y_col = next((c for c in m.columns
                  if c.lower() in ("mean_y", "meany", "y_mean", "luminance")), None)

    m["_suffix"] = m[col].apply(lambda v: parse_suffix(Path(str(v)).stem)[1])
    ones = m[m["_suffix"] == 1]

    print("=" * 70)
    print(f'THE {len(ones)} ROWS WITH A "(1)" SUFFIX')
    print("=" * 70)
    print()

    def mean_y(path):
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].mean())

    for _, row in ones.iterrows():
        entry = row[col]
        p, rule, root = resolve_image_path(entry)
        print(f"  {entry}")
        print(f"      resolved -> {root}  ({rule})")
        if y_col and not pd.isna(row[y_col]) and p:
            stored = float(row[y_col])
            print(f"      stored mean_Y = {stored:.3f}")
            for cand in find_all_matches(entry):
                v = mean_y(cand)
                if v is None:
                    continue
                mark = "  <-- MATCH" if abs(v - stored) <= 0.5 else ""
                try:
                    tag = cand.parts[cand.parts.index("images") + 1]
                except (ValueError, IndexError):
                    tag = str(cand.parent)
                print(f"          {tag:<20} {v:8.3f}{mark}")
        print()

    print("If the MATCH lands consistently on one root, add that mapping to")
    print("SUFFIX_ROUTING above. With only 18 rows, excluding them is also a")
    print("defensible and easily documented choice.")
