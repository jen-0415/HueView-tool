"""One-time: download face_landmarker.task for MediaPipe (Phase 7.1 model)."""
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
dest = Path(__file__).resolve().parents[1] / "face_landmarker.task"  # -> ml_pipeline/
dest.parent.mkdir(parents=True, exist_ok=True)
print(f"Downloading to {dest} ...")
urllib.request.urlretrieve(URL, dest)
print(f"Done: {dest.stat().st_size} bytes")