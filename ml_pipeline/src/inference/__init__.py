"""Phase 14 -- Inference Pipeline.

Single-image path through the real Phase 7-9 modules, for the demo API.
"""

from .pipeline import classify_image, NoFaceDetected
from .models import load_models, models_loaded

__all__ = ["classify_image", "NoFaceDetected", "load_models", "models_loaded"]
