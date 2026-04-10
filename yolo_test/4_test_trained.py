"""
4_test_trained.py
-----------------
STEP 4 — Test your TRAINED custom model on a shelf photo.

Run:
    python yolo_test/4_test_trained.py --photo <shelf_photo>

    Optional:
    --weights yolo_test/results/train/weights/best.pt  (default)
    --conf 0.4   (confidence threshold, default 0.4)

Output:
    yolo_test/results/trained_result.jpg  ← annotated image
    Console: detected products + positions
"""

import argparse
import sys
import json
from pathlib import Path

try:
    from ultralytics import YOLO
    import cv2
except ImportError:
    print("[ERROR] Run: pip install ultralytics opencv-python-headless")
    sys.exit(1)

DEFAULT_WEIGHTS = "yolo_test/results/train/weights/best.pt"


def run_trained_detection(photo_path: str, weights_path: str, conf_threshold: float):
    photo = Path(photo_path)
    if not photo.exists():
        print(f"[ERROR] Photo not found: {photo_path}")
        sys.exit(1)

    weights = Path(weights_path)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        print("        Train the model first: python yolo_test/3_train.py")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  YOLOv8 Custom Model Detection Test")
    print(f"  Photo  : {photo.name}")
    print(f"  Weights: {weights_path}")
    print(f"  Conf   : {conf_threshold}")
    print(f"{'='*60}\n")

    model = YOLO(str(weights))
    results = model(str(photo), conf=conf_threshold, iou=0.45)
    result = results[0]

    # Get image dimensions for position normalisation
    img = cv2.imread(str(photo))
    img_h, img_w = img.shape[:2]

    print(f"── Detected Products ({len(result.boxes)} found) ──────────────────")

    detections = []
    for box in result.boxes:
        class_id   = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

        # Normalised center position (0.0–1.0)
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h

        # Human-readable position
        h_pos = "left" if cx < 0.33 else ("center" if cx < 0.66 else "right")
        v_pos = "top"  if cy < 0.33 else ("mid"    if cy < 0.66 else "bottom")

        detections.append({
            "product": class_name,
            "confidence": round(confidence, 3),
            "position": f"{v_pos}-{h_pos}",
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "center_norm": {"cx": round(cx, 3), "cy": round(cy, 3)},
        })

    # Sort top-to-bottom, left-to-right
    detections.sort(key=lambda d: (d["center_norm"]["cy"], d["center_norm"]["cx"]))

    for d in detections:
        print(
            f"  {d['product']:<40} "
            f"conf: {d['confidence']:.0%}  "
            f"pos: {d['position']}"
        )

    # Save annotated image
    output_path = Path("yolo_test/results/trained_result.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = result.plot()
    cv2.imwrite(str(output_path), annotated)
    print(f"\n✓ Annotated image: {output_path}")

    # Save detections JSON (this is what the main system will consume later)
    json_path = Path("yolo_test/results/trained_detections.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)
    print(f"✓ Detections JSON: {json_path}")

    # Accuracy hints
    high_conf = [d for d in detections if d["confidence"] >= 0.7]
    low_conf  = [d for d in detections if d["confidence"] < 0.5]

    print(f"\n── Confidence Summary ──────────────────────────────────")
    print(f"  High confidence (≥70%): {len(high_conf)} products  ← reliable")
    print(f"  Low  confidence (<50%): {len(low_conf)} products  ← may need more training data")

    if len(low_conf) > len(high_conf):
        print("\n  [TIP] Many low-confidence detections — consider:")
        print("        1. Adding more labeled images for these products")
        print("        2. Training for more epochs (--epochs 100)")
        print("        3. Using a larger model (--model yolov8m)")

    print(f"{'='*60}\n")
    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo",   required=True,            help="Path to shelf photo")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,  help="Path to trained weights")
    parser.add_argument("--conf",    type=float, default=0.4,  help="Confidence threshold (default 0.4)")
    args = parser.parse_args()

    run_trained_detection(args.photo, args.weights, args.conf)
