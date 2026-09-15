import json  
import os 
from PIL import (
    Image,
)  
from facenet_pytorch import MTCNN  
from tqdm import tqdm  

# -------------------------
# CONFIG
# -------------------------
INPUT_DIR = "data/c2_by_class"  # Directory containing original images grouped in class folders
OUTPUT_DIR = "data/c2_processed"  # Target directory to save cropped, processed face images
METADATA_FILE = (
    "data/c2_face_metadata.json"  # Output JSON log file tracking crop results
)

IMG_SIZE = 224  # Standard width and height (224x224 px) for training neural networks
CONF_THRESHOLD = (
    0.90  # Ignore any face detections with less than 90% model confidence
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# MTCNN (DETECTION ONLY)
# -------------------------
# Initialize MTCNN detector settings
mtcnn = MTCNN(
    keep_all=True,  # Detect all faces in an image so we can reject multi-face photos
    min_face_size=40,  # Minimum face size in pixels to consider for detection
    thresholds=[
        0.6,
        0.7,
        0.9,
    ],  # Stage thresholds for the 3 MTCNN neural network steps
    post_process=False,  # Return raw detected bounding box coordinates without normalization
)

print("Loading MTCNN...")

# Dictionary to collect success/failure status and metadata for every image
metadata = {}

# -------------------------
# PROCESS DATASET
# -------------------------
# Loop through each subfolder (class) inside the input directory
for class_folder in sorted(os.listdir(INPUT_DIR)):

    class_path = os.path.join(INPUT_DIR, class_folder)

    # Skip any plain files; only process folders
    if not os.path.isdir(class_path):
        continue

    # Create matching class subfolder in the output directory
    output_class_path = os.path.join(OUTPUT_DIR, class_folder)
    os.makedirs(output_class_path, exist_ok=True)

    images = os.listdir(class_path)

    print(f"\nProcessing {class_folder} ({len(images)} images)")

    success_count = 0

    # Loop through each image file with a visual progress bar
    for img_name in tqdm(images):

        img_path = os.path.join(class_path, img_name)

        try:
            # Load image and convert to 3-channel RGB (removes alpha channels or grayscale issues)
            img = Image.open(img_path).convert("RGB")

            # Skip very small images that lack sufficient detail for accurate skin tone modeling
            if img.width < 100 or img.height < 100:
                continue

            # -------------------------
            # FACE DETECTION
            # -------------------------
            # Run MTCNN to get bounding box coordinates [x1, y1, x2, y2] and confidence scores
            boxes, probs = mtcnn.detect(img)

            # CASE 1: No face detected anywhere in the image
            if boxes is None or probs is None:
                metadata[img_name] = {
                    "class": class_folder,
                    "status": "no_face_detected",
                }
                continue

            # CASE 2: Multiple faces detected (reject image to prevent identity ambiguity)
            if len(boxes) > 1:
                metadata[img_name] = {
                    "class": class_folder,
                    "status": "multiple_faces_detected",
                    "faces_found": int(len(boxes)),
                }
                continue

            # CASE 3: Exactly one face found - check detection confidence
            confidence = probs[0]

            # Reject detections below our 90% confidence threshold
            if confidence < CONF_THRESHOLD:
                metadata[img_name] = {
                    "class": class_folder,
                    "status": "low_confidence",
                    "confidence": float(confidence),
                }
                continue

            box = boxes[0]

            # -------------------------
            # CLEAN CROPPING (NO FILTER LOOK)
            # -------------------------
            # Convert float bounding box coordinates to integers
            x1, y1, x2, y2 = [int(b) for b in box]

            # Clamp bounding box coordinates so they stay within image dimensions
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.width, x2)
            y2 = min(img.height, y2)

            # Calculate box width and height to add a 15% margin around the face
            w, h = x2 - x1, y2 - y1
            pad_x, pad_y = int(0.15 * w), int(0.15 * h)

            # Expand the bounding box by 15% padding while ensuring it stays inside image edges
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(img.width, x2 + pad_x)
            y2 = min(img.height, y2 + pad_y)

            # Crop face area using extended coordinates
            face = img.crop((x1, y1, x2, y2))

            # Resize face crop to standard 224x224 pixels
            face = face.resize((IMG_SIZE, IMG_SIZE))

            # -------------------------
            # SAVE IMAGE
            # -------------------------
            save_path = os.path.join(output_class_path, img_name)
            face.save(save_path)

            # Log successfully processed image details in metadata
            metadata[img_name] = {
                "class": class_folder,
                "confidence": float(confidence),
                "bbox": [x1, y1, x2, y2],
            }

            success_count += 1

        except Exception as e:
            # Catch corrupt images or I/O issues without crashing the whole pipeline
            print(f"Error processing {img_name}: {e}")

    print(f"✔ {class_folder}: {success_count} images saved")

# -------------------------
# SAVE METADATA
# -------------------------
# Export metadata dictionary as a readable JSON file
with open(METADATA_FILE, "w") as f:
    json.dump(metadata, f, indent=4)

print("\nDONE: Clean face dataset created.")