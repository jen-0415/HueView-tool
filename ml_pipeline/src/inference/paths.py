"""
Project-root anchoring, shared by the Phase 14 modules.

Mirrors the find_project_root() pattern src/hueview/ssr_normalization.py
already uses: walk up from this file until a directory containing "data/"
turns up. That makes every path here independent of the current working
directory, so these modules behave the same whether you run them from
ml_pipeline/, from hueview_tool/, or through an IDE's run button.

Without this, Path("configs") silently means a different folder depending
on where the process was started -- which is exactly the class of bug that
cost a whole session earlier in this project.
"""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path, marker: str = "data") -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / marker).is_dir():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

#: Phase 7.1's MediaPipe model bundle. Checked in both the project root and
#: its parent, since it has lived in both places in this repo.
def find_landmarker() -> Path:
    for candidate in (
        PROJECT_ROOT / "face_landmarker.task",
        PROJECT_ROOT.parent / "face_landmarker.task",
    ):
        if candidate.is_file():
            return candidate
    return PROJECT_ROOT / "face_landmarker.task"  # reported as missing downstream
