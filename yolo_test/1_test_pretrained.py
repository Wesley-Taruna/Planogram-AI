"""
1_test_pretrained.py
--------------------
STEP 1 — See what YOLOv8 can detect OUT OF THE BOX (no training needed).

This uses the pre-trained YOLOv8 model (trained on COCO dataset — 80 common objects).
It won't know specific FamilyMart products, but it will detect:
  - bottles, cups, bowls → useful for beverages
  - boxes → useful for packaged snacks

Run:
    python yolo_test/1_test_pretrained.py --photo <path_to_shelf_photo>

Output:
    yolo_test/results/pretrained_result.jpg  ← annotated image with boxes
    Console: list of all detected objects + confidence scores
"""

import argparse
import sys
from pathlib import Path

# ── Check imports ──
try:
    from ultralytics import YOLO
    import cv2
except ImportError:
    print("[ERROR] Missing packages. Run: pip install ultralytics opencv-python-headless")
    sys.exit(1)


def run_pretrained_detection(photo_path: str):
    photo = Path(photo_path)
    if not photo.exists():
        print(f"[ERROR] Photo not found: {photo_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  YOLOv8 Pre-trained Detection Test")
    print(f"  Photo: {photo.name}")
    print(f"{'='*60}\n")

    # Load the smallest pre-trained YOLOv8 model (downloads ~6MB on first run)
    print("[yolo] Loading YOLOv8n (nano) pre-trained model...")
    model = YOLO("yolov8n.pt")

    # Run detection
    print("[yolo] Running detection...")
    results = model(str(photo), conf=0.25, iou=0.45)
    result = results[0]

    # Print all detections
    print(f"\n── Detections ({len(result.boxes)} objects found) ──────────────")
    if len(result.boxes) == 0:
        print("  No objects detected above 25% confidence threshold.")
    else:
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            w = x2 - x1
            h = y2 - y1
            detections.append((confidence, class_name, x1, y1, w, h))

        detections.sort(reverse=True)
        for conf, name, x, y, w, h in detections:
            print(f"  {name:<20} conf: {conf:.0%}   bbox: x={x} y={y} w={w} h={h}px")

    # Save annotated image
    output_path = Path("yolo_test/results/pretrained_result.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = result.plot()
    cv2.imwrite(str(output_path), annotated)
    print(f"\n✓ Annotated image saved: {output_path}")

    # Show what this means
    print(f"\n── What this tells us ──────────────────────────────────")
    print("  The pre-trained model detects GENERIC objects (bottle, box, cup).")
    print("  It does NOT know specific products like 'Chitato' or 'Pringles'.")
    print("  → This is why we need CUSTOM TRAINING on FamilyMart SKUs.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", required=True, help="Path to shelf photo (JPG or PNG)")
    args = parser.parse_args()
    run_pretrained_detection(args.photo)
