"""
Does the "(n)" suffix select a ROOT, not a duplicate?
=====================================================

Evidence so far points somewhere unexpected. Three manifest rows whose
stored mean_Y disagreed with `processed`:

    MST-10/0177_0_0_0_02 (2).jpg   stored=99.541   c2=99.541   processed=73.709
    MST-9/0839_6_0_0_10 (2).jpg    stored=151.875  c2=151.875  processed=119.046
    MST-8/1101_0_0_0_07.jpg        stored=76.886   c2=111.812  processed=76.886

Rows WITH "(2)" matched c2_processed. The row WITHOUT matched processed.

If that generalizes, the suffix is a VERSION SELECTOR, not a duplicate
artifact:

    foo.jpg      -> processed/foo.jpg      (version 1)
    foo (2).jpg  -> c2_processed/foo.jpg   (version 2, brightness-adjusted)

That would mean the current resolver is wrong for ~12,268 rows: it strips
the suffix and serves the `processed` version, when the label those rows
carry was computed from the `c2_processed` version. Those rows would be
distinct images, not duplicates, and the dataset is genuinely 43,221.

This script tests the hypothesis by cross-tabulating suffix presence
against which root reproduces the stored mean_Y.

Run from the repo root:  python src/baseline/test_suffix_hypothesis.py

Read-only.
"""

import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from path_resolver import DEFAULT_ROOTS, V5_ROOT, strip_dup_suffix
except ImportError:
    from src.baseline.path_resolver import DEFAULT_ROOTS, V5_ROOT, strip_dup_suffix

PROC = Path("data/processed")
MANIFEST = PROC / "manifest.csv"
SAMPLE_PER_GROUP = 250
TOL = 0.5

SUFFIX_RE = re.compile(r"\s*\((\d+)\)$")

ALL_ROOTS = [Path(r) for r in DEFAULT_ROOTS] + [Path(V5_ROOT)]


def suffix_number(stem: str):
    """Return the n in 'foo (n)', or None."""
    m = SUFFIX_RE.search(stem)
    return int(m.group(1)) if m else None


def mean_y(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].mean())


def versions_for(rel_path):
    """
    All files across roots matching this entry's base name (suffix stripped),
    keyed by root name.
    """
    rel = Path(str(rel_path).replace("\\", "/"))
    subdir, stem, ext = rel.parent, rel.stem, rel.suffix
    base = strip_dup_suffix(stem)

    out = {}
    for root in ALL_ROOTS:
        folder = root / subdir
        if not folder.is_dir():
            continue
        for cand_stem in (stem, base):
            for cand_ext in (ext, ".jpg", ".png", ".bmp", ".jpeg"):
                p = folder / f"{cand_stem}{cand_ext}"
                if p.is_file():
                    out.setdefault(root.name, p)
                    break
            if root.name in out:
                break
    return out


def main():
    if not MANIFEST.exists():
        print(f"!! {MANIFEST} not found.")
        return

    manifest = pd.read_csv(MANIFEST)
    col = "filename" if "filename" in manifest.columns else manifest.columns[0]
    y_col = next((c for c in manifest.columns
                  if c.lower() in ("mean_y", "meany", "y_mean", "luminance")), None)
    if y_col is None:
        print("!! No mean_Y column found; cannot run this test.")
        return

    manifest["_stem"] = manifest[col].apply(lambda v: Path(str(v).replace("\\", "/")).stem)
    manifest["_suffix"] = manifest["_stem"].apply(suffix_number)

    n_suffix = int(manifest["_suffix"].notna().sum())
    print("=" * 72)
    print("IS THE \"(n)\" SUFFIX A VERSION SELECTOR?")
    print("=" * 72)
    print(f"\nManifest rows: {len(manifest)}")
    print(f"  with a (n) suffix: {n_suffix}")
    print(f"  without:           {len(manifest) - n_suffix}")

    suffix_counts = Counter(manifest["_suffix"].dropna().astype(int))
    print(f"\n  Suffix values present: " +
          ", ".join(f"({k})×{v}" for k, v in sorted(suffix_counts.items())))

    with_suf = manifest[manifest["_suffix"].notna()]
    without = manifest[manifest["_suffix"].isna()]

    print(f"\nTesting up to {SAMPLE_PER_GROUP} rows from each group...\n")

    # group -> root -> match count
    table = defaultdict(Counter)
    tested = Counter()
    no_match_examples = defaultdict(list)

    for label, df in [("no suffix", without), ("has (n)", with_suf)]:
        n = 0
        for _, row in df.sample(frac=1.0, random_state=31).iterrows():
            stored = row[y_col]
            if pd.isna(stored):
                continue
            vers = versions_for(row[col])
            if not vers:
                continue

            matched = []
            values = {}
            for rname, p in vers.items():
                v = mean_y(p)
                if v is None:
                    continue
                values[rname] = v
                if abs(v - float(stored)) <= TOL:
                    matched.append(rname)

            if not values:
                continue
            tested[label] += 1
            if matched:
                # If several roots match (identical files), credit all
                for rname in matched:
                    table[label][rname] += 1
                if len(matched) == len(values):
                    table[label]["(all agree)"] += 1
            else:
                table[label]["(none)"] += 1
                if len(no_match_examples[label]) < 3:
                    no_match_examples[label].append(
                        (row[col], float(stored), values))

            n += 1
            if n >= SAMPLE_PER_GROUP:
                break

    print("-" * 72)
    print("WHICH ROOT REPRODUCES THE STORED mean_Y?")
    print("-" * 72)

    roots_seen = sorted({r for c in table.values() for r in c
                         if r not in ("(none)", "(all agree)")})
    header = f"  {'group':<12} {'tested':>7}" + "".join(f" {r[:16]:>17}" for r in roots_seen) + f" {'none':>7}"
    print("\n" + header)
    for label in ("no suffix", "has (n)"):
        if not tested[label]:
            continue
        line = f"  {label:<12} {tested[label]:>7}"
        for r in roots_seen:
            c = table[label][r]
            line += f" {c:>8} ({100*c/tested[label]:>4.0f}%)"
        line += f" {table[label]['(none)']:>7}"
        print(line)

    for label in ("no suffix", "has (n)"):
        if table[label]["(all agree)"]:
            print(f"\n  '{label}': {table[label]['(all agree)']} of {tested[label]} rows had "
                  f"ALL available roots agree (identical files — uninformative).")

    for label, ex in no_match_examples.items():
        if ex:
            print(f"\n  '{label}' rows where NO root matched:")
            for fn, st, vals in ex:
                print(f"    {fn}  stored={st:.3f}")
                for r, v in vals.items():
                    print(f"        {r} = {v:.3f}")

    # ---------------- Verdict ----------------
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    def rate(label, root):
        return table[label][root] / tested[label] if tested[label] else 0.0

    c2_when_suffix = rate("has (n)", "c2_processed")
    proc_when_suffix = rate("has (n)", "processed")
    c2_when_plain = rate("no suffix", "c2_processed")
    proc_when_plain = rate("no suffix", "processed")

    print(f"\n  has (n)   -> c2_processed {100*c2_when_suffix:.0f}% | processed {100*proc_when_suffix:.0f}%")
    print(f"  no suffix -> c2_processed {100*c2_when_plain:.0f}% | processed {100*proc_when_plain:.0f}%")

    hypothesis = (c2_when_suffix > proc_when_suffix + 0.15 and
                  proc_when_plain > c2_when_plain + 0.15)

    if hypothesis:
        print("""
  HYPOTHESIS CONFIRMED — the suffix selects a root.

    foo.jpg      -> processed/foo.jpg
    foo (2).jpg  -> c2_processed/foo.jpg

  Consequences, and they're all good news:

    * The 12,268 "redundant rows" are DISTINCT images, not duplicates.
      Your dataset is genuinely 43,221, and your methodology chapter's
      count stands as written.

    * There is no duplicate-weighting problem in training.

    * BUT the current resolver is wrong for those 12,268 rows. It strips
      the suffix and serves the `processed` version, while the SCC and
      illumination labels on those rows were derived from `c2_processed`.
      Training on it now would pair ~28% of your images with labels
      computed from a different version of that image.

  Fix: rewrite resolve_image_path so a "(n)" suffix routes to the nth
  root rather than being stripped. Say the word and I'll write it.""")
    elif c2_when_suffix > proc_when_suffix:
        print("""
  PARTIAL SUPPORT. Suffixed rows lean toward c2_processed but not
  cleanly enough to encode as a rule. The manifest was probably assembled
  from several runs without a consistent convention.

  Rather than guess, ask whoever built manifest.csv how the "(n)" rows
  were generated. This is a five-minute question for them and hours of
  inference for us — and getting it wrong mislabels 28% of your data.""")
    else:
        print("""
  HYPOTHESIS REJECTED. The suffix does not select a root.

  So the "(n)" rows really are duplicates, and the earlier reading holds:
  ~12,268 redundant rows, 30,953 distinct images. Since check_dataset.py
  already showed no cross-split leakage, this is a reporting and
  training-weight issue rather than a validity threat.""")

    print("\n  Note: rows where all roots agree are uninformative — identical")
    print("  files can't distinguish which one the manifest came from. Only")
    print("  the disagreeing rows carry signal here.")


if __name__ == "__main__":
    main()
