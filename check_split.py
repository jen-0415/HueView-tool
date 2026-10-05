import pandas as pd
from pathlib import Path

PROC = Path("data/processed")

manifest = pd.read_csv(PROC / "manifest.csv")
label_col = next(c for c in manifest.columns if c.lower() in ("scc_label", "scc", "label"))
labels = manifest[["filename", label_col]].rename(columns={label_col: "scc"})

counts = {}
total = 0
for split in ["train", "val", "test"]:
    df = pd.read_csv(PROC / f"{split}.csv")
    df = df.rename(columns={df.columns[0]: "filename"}) if "filename" not in df.columns else df
    df = df.merge(labels, on="filename", how="left")
    counts[split] = df["scc"].value_counts().sort_index()
    total += len(df)
    print(f"{split:<6} {len(df):>6} rows")

print(f"total  {total:>6}\n")
for split, c in counts.items():
    print(f"{split:<6} {c.sum() / total:.1%}")

print("\nPer class:")
print(pd.DataFrame(counts).fillna(0).astype(int))