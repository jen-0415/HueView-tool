"""
Build a verified, frozen resolution table
=========================================

Routing rules get 17 of the 18 "(1)" rows right. The exception:

    MST-8/0606_0_0_0_05(1).jpg
        stored = 119.686
        processed        119.686   <- correct
        c2_processed     151.684
        processed - 7-26 151.684   <- where the rule sends it

No rule inferred from 18 examples will be dependable. But the manifest's
stored mean_Y is a per-row fingerprint of the exact image its labels were
computed from, so each row can be VERIFIED instead of guessed.

This runs once and writes resolved_manifest.csv, mapping every manifest
row to the file whose luminance actually matches its stored value. Every
later phase reads that table instead of re-deriving paths, so Baseline
and HueView are guaranteed to see identical bytes and no future script
can silently pick a different folder.

Run from the repo root:  python src/baseline/resolve_manifest.py

Takes a few minutes — it loads images for rows with multiple candidates.
Results are cached per file, and the output is written once.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from ml_pipeline.src.baseline.path_resolver import find_all_matches, resolve_image_path, parse_suffix
except ImportError:
    from ml_pipeline.src.baseline.path_resolver import find_all_matches, resolve_image_path, parse_suffix

PROC = Path("data/processed")
MANIFEST = PROC / "manifest.csv"
OUT = PROC / "resolved_manifest.csv"
TOL = 0.5

_cache = {}


def mean_y(path):
    key = str(path)
    if key in _cache:
        return _cache[key]
    bgr = cv2.imread(key)
    if bgr is None:
        _cache[key] = None
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    v = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].mean())
    _cache[key] = v
    return v


def root_of(path):
    parts = Path(path).parts
    try:
        return parts[parts.index("images") + 1]
    except (ValueError, IndexError):
        return "?"


def main():
    if not MANIFEST.exists():
        print(f"!! {MANIFEST} not found. Run from the repo root.")
        return

    manifest = pd.read_csv(MANIFEST)
    col = "filename" if "filename" in manifest.columns else manifest.columns[0]
    y_col = next((c for c in manifest.columns
                  if c.lower() in ("mean_y", "meany", "y_mean", "luminance")), None)

    if y_col is None:
        print("!! No mean_Y column — cannot verify. Falling back to routing rules only.")
        return

    total = len(manifest)
    print("=" * 70)
    print("BUILDING VERIFIED RESOLUTION TABLE")
    print("=" * 70)
    print(f"\nRows: {total}")
    print("Each row is matched to the file reproducing its stored mean_Y.\n")

    rows = []
    stats = {
        "single_candidate": 0,
        "verified_unique": 0,
        "verified_tie": 0,
        "no_match_used_rule": 0,
        "unresolved": 0,
    }
    problems = []
    start = time.time()

    for i, (_, row) in enumerate(manifest.iterrows(), 1):
        entry = row[col]
        stored = row[y_col]
        candidates = find_all_matches(entry)

        chosen, how = None, None

        if len(candidates) == 1:
            chosen, how = candidates[0], "single_candidate"

        elif len(candidates) > 1 and not pd.isna(stored):
            stored_f = float(stored)
            matching = [c for c in candidates
                        if (v := mean_y(c)) is not None and abs(v - stored_f) <= TOL]

            if len(matching) == 1:
                chosen, how = matching[0], "verified_unique"
            elif len(matching) > 1:
                # Several byte-identical files match; routing order breaks the tie.
                routed, _, _ = resolve_image_path(entry)
                chosen = routed if routed in matching else matching[0]
                how = "verified_tie"
            else:
                # Nothing reproduces the stored value; keep the rule's answer
                # but record it — these need a human decision.
                routed, _, _ = resolve_image_path(entry)
                chosen, how = routed, "no_match_used_rule"
                if len(problems) < 25:
                    vals = {root_of(c): mean_y(c) for c in candidates}
                    problems.append((entry, stored_f, vals))

        elif len(candidates) > 1:
            routed, _, _ = resolve_image_path(entry)
            chosen, how = routed, "no_match_used_rule"

        if chosen is None:
            routed, _, _ = resolve_image_path(entry)
            if routed is not None:
                chosen, how = routed, "single_candidate"
            else:
                stats["unresolved"] += 1
                rows.append({"filename": entry, "resolved_path": None,
                             "root": None, "how": "unresolved"})
                continue

        stats[how] = stats.get(how, 0) + 1
        rows.append({
            "filename": entry,
            "resolved_path": Path(chosen).as_posix(),
            "root": root_of(chosen),
            "how": how,
        })

        if i % 2000 == 0 or i == total:
            el = time.time() - start
            rate = i / el if el else 0
            eta = (total - i) / rate if rate else 0
            sys.stdout.write(f"\r  {i}/{total} rows  ({el:.0f}s elapsed, ~{eta:.0f}s left)")
            sys.stdout.flush()

    print("\n")
    res = pd.DataFrame(rows)

    print("-" * 70)
    print("HOW EACH ROW WAS RESOLVED")
    print("-" * 70)
    labels = {
        "single_candidate": "only one candidate file existed",
        "verified_unique": "one candidate matched stored mean_Y",
        "verified_tie": "several identical candidates matched",
        "no_match_used_rule": "NO candidate matched; used routing rule",
        "unresolved": "no file found at all",
    }
    for k in ["single_candidate", "verified_unique", "verified_tie",
              "no_match_used_rule", "unresolved"]:
        n = stats.get(k, 0)
        if n:
            print(f"  {n:>7}  ({100*n/total:5.1f}%)  {labels[k]}")

    print("\n  Files drawn from each root:")
    for root, n in res["root"].value_counts(dropna=False).items():
        print(f"    {n:>7}  {root}")

    # Duplicates: two rows landing on one file
    ok = res[res["resolved_path"].notna()]
    dup = ok["resolved_path"].value_counts()
    dup = dup[dup > 1]
    print(f"\n  Distinct images: {ok['resolved_path'].nunique()} of {len(ok)} rows")
    if len(dup):
        print(f"  Files used by more than one row: {len(dup)}")
        for p in dup.head(5).index:
            names = ok[ok["resolved_path"] == p]["filename"].tolist()
            print(f"    {Path(p).name}")
            for n in names[:4]:
                print(f"        <- {n}")

    if problems:
        print("\n" + "-" * 70)
        print("ROWS WHERE NO CANDIDATE MATCHED THE STORED mean_Y")
        print("-" * 70)
        print("\n  These fell back to the routing rule and may be wrong:\n")
        for entry, stored, vals in problems[:10]:
            print(f"    {entry}   stored={stored:.3f}")
            for r, v in vals.items():
                print(f"        {r:<20} {v:.3f}" if v is not None else f"        {r:<20} unreadable")
        if len(problems) > 10:
            print(f"    ... and {len(problems)-10} more")

    res.to_csv(OUT, index=False)
    print("\n" + "=" * 70)
    print(f"Wrote {OUT}  ({len(res)} rows)")
    print("=" * 70)
    print("""
This table is now the single source of truth for filename -> file.
Phase 6.1 reads it directly, so the routing rules never run again and
no later script can pick a different folder by accident.

Freeze it alongside train/val/test.csv and commit it — it's part of
what makes the Baseline/HueView comparison reproducible.""")


if __name__ == "__main__":
    main()
