"""
Diagnostic: for each row in landmarks_index.csv, check whether its
corresponding SSR-normalized image can be found UNAMBIGUOUSLY in
images_ssr/, given that images_ssr/ mirrors four overlapping cohort
folders (processed, c1_processed, c2_processed, v5_processed) known to
contain duplicate filenames from repeated processing runs.

Also checks, for ambiguous cases, whether the duplicate copies are
byte-identical (harmless) or actually different files (a real problem --
landmarks from one version applied to a different version's pixels).

This only reads and reports -- doesn't move or change anything.
"""

import hashlib
import pandas as pd
from pathlib import Path

IMAGES_SSR_ROOT = Path("data/processed/images_ssr")
LANDMARKS_INDEX_PATH = Path("data/processed/landmarks_index.csv")

index_df = pd.read_csv(LANDMARKS_INDEX_PATH)
cohorts = sorted(p.name for p in IMAGES_SSR_ROOT.iterdir() if p.is_dir())
print("SSR cohort folders found:", cohorts)
print()


def file_hash(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


unambiguous = 0
ambiguous_identical = 0
ambiguous_different = 0
missing = 0
different_examples = []

for _, row in index_df.iterrows():
    filename = row["filename"]  # e.g. "MST-1/xxx.jpg"
    path_source = row["path_source"]

    matches = [c for c in cohorts if (IMAGES_SSR_ROOT / c / filename).exists()]

    if len(matches) == 0:
        missing += 1
    elif len(matches) == 1:
        unambiguous += 1
    else:
        hashes = {c: file_hash(IMAGES_SSR_ROOT / c / filename) for c in matches}
        if len(set(hashes.values())) == 1:
            ambiguous_identical += 1
        else:
            ambiguous_different += 1
            if len(different_examples) < 10:
                different_examples.append((filename, path_source, matches))

total = len(index_df)
print(f"Total landmark rows checked: {total}")
print(f"Unambiguous (exactly one SSR copy exists): {unambiguous} ({unambiguous/total:.1%})")
print(f"Ambiguous but identical (multiple copies, same bytes -- harmless): {ambiguous_identical} ({ambiguous_identical/total:.1%})")
print(f"Ambiguous AND different (multiple copies, DIFFERENT bytes -- real problem): {ambiguous_different} ({ambiguous_different/total:.1%})")
print(f"Missing (no SSR copy found in any cohort): {missing} ({missing/total:.1%})")

if different_examples:
    print("\nExamples where copies genuinely differ (filename, path_source used for landmarks, cohorts with SSR output):")
    for ex in different_examples:
        print(" ", ex)