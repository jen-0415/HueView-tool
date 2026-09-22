"""
Phase 14.3 -- preprocess_image()

Mirrors src/preprocessing/mtcnn_filter.py (Phase 2) EXACTLY, so an uploaded
image goes through the same crop the training data did. Four details here
are not interchangeable with the "obvious" alternatives:

  1. facenet_pytorch.MTCNN, not the `mtcnn` pip package. Different API
     (detect() -> boxes as [x1,y1,x2,y2] + probs), different weights,
     different boxes. Same detector config as Phase 2: keep_all=True,
     min_face_size=40, thresholds=[0.6, 0.7, 0.9], post_process=False.

  2. 15% margin padded around the detected box before cropping. Without it
     the crop is tighter than every image the model trained on.

  3. PIL's Image.resize() with its default resample (BICUBIC), not
     cv2.INTER_AREA. Measured mean absolute difference between the two is
     ~5.9 per pixel on a 400->224 downscale -- large enough to matter to a
     colour-based model.

  4. Multiple faces are REJECTED, not disambiguated. Phase 2 dropped those
     images to avoid identity ambiguity, so no multi-face image is in the
     training distribution. See MULTI_FACE_POLICY below if you'd rather the
     demo be more forgiving than the study was.

Also carried over: images under 100px in either dimension are skipped, and
box coordinates are clamped before padding and again after.
"""

from __future__ import annotations

import io
from typing import Dict, Tuple

import numpy as np
from PIL import Image

from .config import MIN_FACE_CONFIDENCE

IMG_SIZE = 224
MIN_INPUT_DIMENSION = 100  # Phase 2 skipped anything smaller
MARGIN = 0.15              # Phase 2's padding around the detected box

#: "reject" matches Phase 2 exactly -- the study never classified a
#: multi-face image, so neither does the demo. Switch to "highest_confidence"
#: only as a deliberate, documented choice (dev plan 17.2 asks you to pick
#: one and write it down); it would mean the demo accepts inputs the study
#: excluded.
MULTI_FACE_POLICY = "reject"


class NoFaceDetected(Exception):
    """No usable face: none found, too many, or below the confidence gate."""


_detector = None


def _get_detector():
    """Built once -- MTCNN has real startup cost, and Phase 15 loads it at
    app startup rather than per request."""
    global _detector
    if _detector is None:
        try:
            from facenet_pytorch import MTCNN
        except ImportError as e:
            raise RuntimeError(
                "facenet_pytorch is not installed. pip install facenet-pytorch\n"
                "(Phase 2 used this, not the `mtcnn` package -- they are different "
                "detectors and produce different boxes.)"
            ) from e

        _detector = MTCNN(
            keep_all=True,        # detect all faces so multi-face can be rejected
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.9],
            post_process=False,   # raw box coordinates
        )
    return _detector


def preprocess_image(image_bytes: bytes) -> Tuple[np.ndarray, Dict]:
    """
    Returns (crop_224_rgb_uint8, detection_info).

    Raises NoFaceDetected -- the API turns that into a 422, not a 500.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError("Could not read the uploaded file as an image.") from e

    if img.width < MIN_INPUT_DIMENSION or img.height < MIN_INPUT_DIMENSION:
        raise NoFaceDetected(
            f"Image is {img.width}x{img.height}. This study skipped anything "
            f"under {MIN_INPUT_DIMENSION}px -- too little detail for skin tone."
        )

    try:
        boxes, probs, landmarks = _get_detector().detect(img, landmarks=True)
    except ValueError as e:
        # facenet_pytorch crashes instead of returning None when the image is
        # too small for min_face_size=40 to produce any scale-pyramid
        # candidates -- observed as "torch.cat(): expected a non-empty list of
        # Tensors" on a 33x33 input. Treat it as "no face" so the API returns
        # a clean 422 rather than a 500 with a stack trace.
        if "non-empty list of Tensors" in str(e):
            raise NoFaceDetected(
                "No face could be detected -- the image or the face in it is too small."
            ) from e
        raise

    if boxes is None or probs is None or len(boxes) == 0:
        raise NoFaceDetected("No face detected in the uploaded image.")

    if len(boxes) > 1 and MULTI_FACE_POLICY == "reject":
        raise NoFaceDetected(
            f"{len(boxes)} faces detected. Please upload a photo with a single "
            f"face -- images with multiple faces were excluded from this study."
        )

    # With "highest_confidence" policy, fall through on the best box instead.
    best = int(np.argmax(probs))
    confidence = float(probs[best])

    if confidence < MIN_FACE_CONFIDENCE:
        raise NoFaceDetected(
            f"A face was found, but confidence {confidence:.2f} is below the "
            f"{MIN_FACE_CONFIDENCE} threshold this study requires."
        )

    # --- Phase 2's crop, step for step -----------------------------------
    x1, y1, x2, y2 = [int(b) for b in boxes[best]]

    # Clamp to image bounds first
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)

    # 15% margin, then clamp again
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = int(MARGIN * w), int(MARGIN * h)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img.width, x2 + pad_x)
    y2 = min(img.height, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        raise NoFaceDetected("The detected face box was empty after clamping.")

    face = img.crop((x1, y1, x2, y2))
    upscaled = bool(face.width < IMG_SIZE or face.height < IMG_SIZE)

    # PIL default resample (BICUBIC) -- matches Phase 2's bare .resize() call
    face = face.resize((IMG_SIZE, IMG_SIZE))
    crop = np.asarray(face, dtype=np.uint8)

    n_landmarks = 0
    if landmarks is not None and len(landmarks) > best:
        n_landmarks = int(len(landmarks[best]))

    detection = {
        "confidence": round(confidence, 4),
        "landmarks_found": n_landmarks,     # MTCNN's 5 keypoints, not MediaPipe's 468
        "bbox": [int(x1), int(y1), int(x2), int(y2)],  # padded box, as Phase 2 logged it
        "upscaled": upscaled,
        "n_faces_found": int(len(boxes)),
    }
    return crop, detection