"""
Dataset integrity check (run before Phase 6.1)
==============================================

"Every row resolves" is necessary but not sufficient. This checks the
three things a resolve-rate can't tell you:

  A. COLLISIONS   - do several manifest rows resolve to the SAME file?
                    ~12,268 rows resolve only after stripping a "(n)"
                    suffix. If "foo.jpg" is also its own row, both point
                    at one image.

  B. LEAKAGE      - do any of those collisions cross split boundaries?
                    If "foo.jpg" is in train and "foo (2).jpg" is in test,
                    the model trains on a test image. Phase 5's check
                    compares filename strings and can't see this.

  C. ROOT CONFLICT- does the same relative path exist under more than one
                    root with DIFFERENT content? Those roots are separate
                    preprocessing runs. Search order silently picks one.

Run from the repo root:  python src/baseline/check_dataset.py

Read-only. Nothing is written except a report CSV.
"""

import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    from ml_pipeline.src.baseline.path_resolver import resolve_image_path, find_all_matches
except ImportError:
    from ml_pipeline.src.baseline.path_resolver import resolve_image_path, find_all_matches

PROC = Path("data/processed")
SPLITS = ["train", "val", "test"]
HASH_SAMPLE = 300  # files to hash for the cross-root content check


def file_hash(path, chunk=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_splits():
    frames = {}
    for s in SPLITS:
        p = PROC / f"{s}.csv"
        if not p.exists():
            print(f"!! {p} not found.")
            return None
        df = pd.read_csv(p)
        col = "filename" if "filename" in df.columns else df.columns[0]
        df = df.rename(columns={col: "filename"})
        frames[s] = df
    return frames


def main():
    frames = load_splits()
    if frames is None:
        return

    print("=" * 70)
    print("DATASET INTEGRITY CHECK")
    print("=" * 70)

    total = sum(len(f) for f in frames.values())
    print(f"\nRows: " + ", ".join(f"{s}={len(frames[s])}" for s in SPLITS) + f"  (total {total})")
    print("\nResolving every row across all four roots...")

    records = []
    for split, df in frames.items():
        for _, row in df.iterrows():
            p, rule, root = resolve_image_path(row["filename"])
            records.append({
                "split": split,
                "filename": row["filename"],
                "resolved": p.as_posix() if p else None,
                "rule": rule,
                "root": root,
            })

    res = pd.DataFrame(records)

    print("\n" + "-" * 70)
    print("RESOLUTION")
    print("-" * 70)
    for rule, n in res["rule"].value_counts().items():
        print(f"    {n:>7}  ({100*n/len(res):5.1f}%)  {rule}")
    print("\n  By root:")
    for root, n in res["root"].value_counts(dropna=False).items():
        print(f"    {n:>7}  {root}")

    unresolved = int((res["rule"] == "not_found").sum())
    ok = res[res["resolved"].notna()]

    # ---------------- A. COLLISIONS ----------------
    print("\n" + "=" * 70)
    print("A. DUPLICATE COLLISIONS")
    print("=" * 70)

    counts = ok["resolved"].value_counts()
    colliding = counts[counts > 1]
    redundant = int((colliding - 1).sum())

    print(f"\n    Manifest rows resolved:      {len(ok)}")
    print(f"    Distinct physical images:    {len(counts)}")
    print(f"    Images used by >1 row:       {len(colliding)}")
    print(f"    Redundant rows:              {redundant}")

    if len(colliding):
        pct = 100 * redundant / len(ok)
        print(f"\n    {pct:.1f}% of your rows are pointing at an image another row already uses.")
        print("\n    Examples:")
        for path in colliding.head(3).index:
            rows = ok[ok["resolved"] == path]
            print(f"      {Path(path).name}")
            for _, r in rows.head(4).iterrows():
                print(f"          <- [{r['split']:<5}] {r['filename']}")

    # ---------------- B. LEAKAGE ----------------
    print("\n" + "=" * 70)
    print("B. CROSS-SPLIT LEAKAGE")
    print("=" * 70)

    by_split = {s: set(ok[ok["split"] == s]["resolved"]) for s in SPLITS}
    print()
    for s in SPLITS:
        n_rows = int((ok["split"] == s).sum())
        print(f"    {s:<6} {n_rows:>7} rows -> {len(by_split[s]):>7} distinct images")

    print("\n    Shared images between splits:")
    leaks = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = by_split[a] & by_split[b]
        leaks[(a, b)] = shared
        flag = "   <-- LEAKAGE" if shared else ""
        print(f"      {a:<6} n {b:<6} : {len(shared):>6}{flag}")

    any_leak = any(len(v) for v in leaks.values())
    if any_leak:
        print("\n    Example leaked images:")
        for (a, b), shared in leaks.items():
            for path in list(shared)[:2]:
                rows = ok[ok["resolved"] == path]
                print(f"      {Path(path).name}")
                for _, r in rows.iterrows():
                    print(f"          [{r['split']:<5}] {r['filename']}")
            if shared:
                break

    # ---------------- C. ROOT CONFLICTS ----------------
    print("\n" + "=" * 70)
    print("C. CROSS-ROOT CONTENT CONFLICTS")
    print("=" * 70)
    print(f"\n    Hashing a sample of {HASH_SAMPLE} rows that match under >1 root...")

    sample = ok.sample(n=min(HASH_SAMPLE * 3, len(ok)), random_state=7)
    multi, identical, differing = 0, 0, 0
    examples = []

    for _, r in sample.iterrows():
        matches = find_all_matches(r["filename"])
        if len(matches) < 2:
            continue
        multi += 1
        try:
            hashes = {file_hash(m) for m in matches}
        except OSError:
            continue
        if len(hashes) == 1:
            identical += 1
        else:
            differing += 1
            if len(examples) < 3:
                examples.append((r["filename"], [str(m) for m in matches]))
        if multi >= HASH_SAMPLE:
            break

    print(f"    Rows matching under more than one root: {multi}")
    print(f"      byte-identical copies:  {identical}")
    print(f"      DIFFERENT content:      {differing}")

    if differing:
        print("\n    !! The same filename holds different pixels under different roots.")
        print("       Search order alone decides which one you train on.")
        for fn, paths in examples:
            print(f"\n      {fn}")
            for p in paths:
                print(f"          {p}")

    # ---------------- VERDICT ----------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    problems = []
    if unresolved:
        problems.append(f"{unresolved} rows resolve to nothing")
    if any_leak:
        n = sum(len(v) for v in leaks.values())
        problems.append(f"{n} images appear in more than one split")
    if redundant:
        problems.append(f"{redundant} redundant rows point at already-used images")
    if differing:
        problems.append(f"{differing}/{multi} sampled rows differ in content across roots")

    if not problems:
        print("\n  Clean. Distinct images, no leakage, no root conflicts.")
        print("  Safe to proceed to feature extraction.")
    else:
        print("\n  Issues found:")
        for p in problems:
            print(f"    - {p}")

        if any_leak:
            print("""
  LEAKAGE IS THE BLOCKING ONE.

  The same photo is in training and test under two different names, so
  test accuracy will be inflated for Baseline AND HueView. That doesn't
  just shift one number — it undermines the comparison the study rests on,
  and it's the kind of thing a panelist can find in five minutes.

  Before training:
    1. Decide what the "(n)" rows actually are — stale duplicates, or
       distinct photos that went missing. Check whether any "(n)" file
       still exists anywhere on disk or in a backup.
    2. If stale: drop them, regenerate the Phase 5 splits from the
       de-duplicated manifest, and record the new dataset size.
    3. If distinct-but-missing: those rows can't be used as-is, because
       resolving them to the base file assigns the wrong image to a label.

  Either way the splits get regenerated, not patched — patching breaks
  the stratification proportions.""")
        elif redundant:
            print("""
  No leakage, so this isn't urgent — but duplicated rows within a split
  do weight those images more heavily in training, and they inflate your
  reported dataset size. Worth resolving before you write the methodology
  section, since the honest count is the distinct-image count.""")

    out = PROC / "dataset_check.csv"
    res.to_csv(out, index=False)
    print(f"\n  Per-row detail written to {out}")


if __name__ == "__main__":
    main()
