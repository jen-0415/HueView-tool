from pathlib import Path

for p in sorted(Path("data").rglob("*.csv")):
    print(p)