# Throwaway diagnostic -- not part of the pipeline, safe to delete after use.
# Run from ml_pipeline/:  python scripts/diagnostics/split_leakage.py

from collections import Counter
from pathlib import Path

import pandas as pd

resolved = pd.read_csv("data/processed/resolved_manifest.csv")

# Group manifest filenames by the physical file they actually resolve to --
# any group with more than one filename is a same-image duplicate.
dup_groups = resolved.groupby("resolved_path")["filename"].apply(list)
dup_groups = dup_groups[dup_groups.apply(len) > 1]
print(f"Duplicate-content groups: {len(dup_groups)}")

split_map = {}
for name, path in [("train", "data/processed/train.csv"),
                    ("val", "data/processed/val.csv"),
                    ("test", "data/processed/test.csv")]:
    for fn in pd.read_csv(path)["filename"]:
        split_map[fn] = name

leaky = []
safe = 0
for resolved_path, filenames in dup_groups.items():
    splits_involved = sorted({split_map[fn] for fn in filenames if fn in split_map})
    if len(splits_involved) > 1:
        leaky.append({
            "resolved_path": resolved_path,
            "filenames": filenames,
            "splits": splits_involved,
        })
    else:
        safe += 1

print(f"Safe (all copies in the same split): {safe}")
print(f"LEAKY (copies span multiple splits): {len(leaky)}")

if leaky:
    pd.DataFrame(leaky).to_csv("data/processed/split_leakage.csv", index=False)
    print(f"\nWrote {len(leaky)} leaky groups to split_leakage.csv")
    print("\nBreakdown by which splits are involved:")
    pair_counts = Counter(tuple(l["splits"]) for l in leaky)
    for pair, count in pair_counts.most_common():
        print(f"  {' <-> '.join(pair)}: {count}")