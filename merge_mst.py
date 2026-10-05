from pathlib import Path
import shutil

# Correct path from HueView-tool root
base_dir = Path("ml_pipeline/data/processed/images")

# Folders containing the MST folders
source_folders = [
    "c1_processed",
    "c2_processed",
    "processed",
    "v5_processed",
]

# Actual folder names in your dataset
mst_folders = [f"MST-{i}" for i in range(1, 11)]


def move_contents(source, destination):
    destination.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        target = destination / item.name

        # Prevent overwriting duplicate files
        if target.exists():
            counter = 1

            while True:
                if item.is_file():
                    new_target = destination / f"{item.stem}_{counter}{item.suffix}"
                else:
                    new_target = destination / f"{item.name}_{counter}"

                if not new_target.exists():
                    target = new_target
                    break

                counter += 1

        shutil.move(str(item), str(target))


for source_name in source_folders:
    source_dir = base_dir / source_name

    if not source_dir.exists():
        print(f"⚠️ Source folder not found: {source_dir}")
        continue

    for mst in mst_folders:
        source_mst = source_dir / mst
        destination_mst = base_dir / mst

        if not source_mst.exists():
            print(f"⚠️ Missing: {source_mst}")
            continue

        print(f"Moving {source_mst} -> {destination_mst}")

        move_contents(source_mst, destination_mst)

        # Remove empty MST folder
        try:
            source_mst.rmdir()
        except OSError:
            pass

    # Remove source folder if completely empty
    try:
        source_dir.rmdir()
    except OSError:
        pass


print("\n✅ Done! All MST folders have been merged.")