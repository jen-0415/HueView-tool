"""
One-time cleanup: convert the 2,497 leftover .bmp files to .png so they're
included by extract_luminance.py's VALID_EXTENSIONS filter. Lossless
conversion — BMP is uncompressed, so nothing is lost going to PNG.

Originals are left in place (not deleted) so nothing is destructive; the
.bmp files will just keep getting silently skipped by Step 2 going forward,
same as before, now alongside their .png replacements.
"""

import cv2
from pathlib import Path

face_dir = Path("data/processed/faces")  # matches your last successful run

converted = 0
failed = []

for bmp_path in face_dir.rglob("*.bmp"):
    img = cv2.imread(str(bmp_path))
    if img is None:
        failed.append(str(bmp_path))
        continue

    png_path = bmp_path.with_suffix(".png")
    cv2.imwrite(str(png_path), img)
    converted += 1

print(f"Converted {converted} BMP files to PNG")
print(f"Failed to read: {len(failed)}")
if failed:
    print("Examples:", failed[:10])