"""
Image path resolver
===================

All source images are stored under:

    ml_pipeline/data/processed/images/

Manifest filenames may contain "(n)" suffixes. These suffixes are
meaningful and should not automatically be stripped, because files such
as "foo.jpg" and "foo(1).jpg" may represent different images.

Resolution order:
1. Exact literal filename under the main images root.
2. Filename with the "(n)" suffix removed, if the literal file is absent.
3. Same base filename using another supported extension.
"""

import re
from pathlib import Path


# Main source of original images
IMAGES = Path("ml_pipeline/data/processed/images")

CANDIDATE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

# Matches " (2)" or "(1)" immediately before the extension.
SUFFIX_RE = re.compile(r"\s*\((\d+)\)$")


def parse_suffix(stem: str):
    """
    'foo (2)' -> ('foo', 2)
    'foo(1)'  -> ('foo', 1)
    'foo'     -> ('foo', None)
    """
    m = SUFFIX_RE.search(stem)

    if not m:
        return stem, None

    return stem[:m.start()], int(m.group(1))


def strip_dup_suffix(stem: str) -> str:
    """Backwards-compatible helper: returns the stem without '(n)'."""
    return parse_suffix(stem)[0]


def resolve_image_path(rel_path, roots=None, v5_root=None):
    """
    Resolve a manifest filename to a physical image.

    Mapping rules:
        foo.jpg       -> foo.jpg
        foo(1).jpg   -> foo(1).jpg
        foo (2).jpg  -> foo_1.jpg
        foo (3).jpg  -> foo_2.jpg

    Alternate extensions are also supported.
    """

    if roots is not None:
        search_roots = [Path(r) for r in roots]
    else:
        search_roots = [IMAGES]

    rel = Path(str(rel_path).replace("\\", "/"))
    subdir = rel.parent
    raw_stem = rel.stem
    ext = rel.suffix

    stem, suffix = parse_suffix(raw_stem)

    # ---------------------------------------------------------
    # 1. PRIMARY: exact literal filename
    #
    # Important:
    # foo.jpg and foo(1).jpg can be different images.
    # ---------------------------------------------------------
    for root in search_roots:
        p = root / subdir / f"{raw_stem}{ext}"

        if p.is_file():
            rule = "exact" if suffix is None else "suffix_literal"
            return p, rule, root.name

    # ---------------------------------------------------------
    # 2. FLATTENED BATCH MAPPING
    #
    # In the current flattened dataset:
    #
    #   foo (2).jpg -> foo_1.jpg
    #   foo (3).jpg -> foo_2.jpg
    #
    # This must happen BEFORE suffix stripping because:
    #
    #   foo (2).jpg
    #
    # must NOT resolve to:
    #
    #   foo.jpg
    # ---------------------------------------------------------
    if suffix is not None and suffix >= 2:
        flattened_index = suffix - 1

        for root in search_roots:
            candidate_stems = [
                f"{stem}_{flattened_index}",
            ]

            for candidate_stem in candidate_stems:
                # Same extension first
                p = root / subdir / f"{candidate_stem}{ext}"

                if p.is_file():
                    return p, "flattened_suffix", root.name

                # Then alternate extensions
                for alt in CANDIDATE_EXTS:
                    if alt == ext:
                        continue

                    p = root / subdir / f"{candidate_stem}{alt}"

                    if p.is_file():
                        return p, "flattened_suffix", root.name

    # ---------------------------------------------------------
    # 3. SUFFIX-STRIPPED FALLBACK
    #
    # Only used if the more specific flattened mapping above
    # does not exist.
    # ---------------------------------------------------------
    for root in search_roots:
        p = root / subdir / f"{stem}{ext}"

        if p.is_file():
            return p, "suffix_routed", root.name

    # ---------------------------------------------------------
    # 4. ALTERNATE EXTENSIONS
    # ---------------------------------------------------------
    for root in search_roots:
        for alt in CANDIDATE_EXTS:

            if alt == ext:
                continue

            # Literal filename with alternate extension
            p = root / subdir / f"{raw_stem}{alt}"

            if p.is_file():
                return p, "fallback", root.name

            # Suffix-stripped filename with alternate extension
            p = root / subdir / f"{stem}{alt}"

            if p.is_file():
                return p, "fallback", root.name

    return None, "not_found", None


def find_all_matches(rel_path, roots=None, v5_root=None):
    """
    Find every file under the current image root that matches either:

        - the literal filename stem
        - the suffix-stripped filename stem

    Used for diagnostics.
    """

    if roots is not None:
        search_roots = [Path(r) for r in roots]
    else:
        search_roots = [IMAGES]

    rel = Path(str(rel_path).replace("\\", "/"))

    subdir = rel.parent
    raw_stem = rel.stem
    stem, _ = parse_suffix(raw_stem)

    stems = {raw_stem, stem}

    hits = []

    for root in search_roots:
        folder = root / subdir

        if not folder.is_dir():
            continue

        for p in folder.iterdir():
            if p.is_file() and p.stem in stems:
                hits.append(p)

    return hits


# Backwards compatibility for older scripts that import these names.
DEFAULT_ROOTS = [IMAGES]
ROOT_PROCESSED = IMAGES
ROOT_C2 = IMAGES
ROOT_726 = IMAGES
ROOT_V5 = IMAGES
V5_ROOT = IMAGES


if __name__ == "__main__":
    print("=" * 70)
    print("IMAGE PATH RESOLVER")
    print("=" * 70)
    print(f"Image root: {IMAGES}")
    print(f"Exists: {IMAGES.is_dir()}")