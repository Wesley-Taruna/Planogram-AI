"""
2_label_guide.py
----------------
STEP 2 — Understand the training data format + generate a class list
         from the existing planogram JSON files.

This script:
  1. Reads all planogram JSON files and extracts unique product names
  2. Generates a classes.txt file (required for YOLO training)
  3. Prints labeling instructions for use with LabelImg or Roboflow

Run:
    python yolo_test/2_label_guide.py

Output:
    yolo_test/training_data/classes.txt   ← all product classes
    yolo_test/training_data/dataset.yaml  ← YOLO training config
"""

import json
from pathlib import Path


PLANOGRAM_DIR = Path("planograms")
OUTPUT_DIR = Path("yolo_test/training_data")


def extract_classes_from_planograms() -> list:
    """Read all planogram JSONs and collect unique product names."""
    products = set()

    json_files = list(PLANOGRAM_DIR.glob("*.json"))
    if not json_files:
        print("[ERROR] No planogram JSON files found in planograms/")
        print("        Run the PDF extractor first: python pdf_extractor.py planograms/SNACK3C.pdf")
        return []

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sec in data.get("gondola_sections", []):
            for p in sec.get("products", []):
                name = p.get("product_name", "").strip()
                if name:
                    products.add(name)

    return sorted(products)


def write_classes_file(classes: list):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    classes_path = OUTPUT_DIR / "classes.txt"
    with open(classes_path, "w", encoding="utf-8") as f:
        for cls in classes:
            f.write(cls + "\n")
    print(f"✓ classes.txt saved: {classes_path}  ({len(classes)} classes)")
    return classes_path


def write_dataset_yaml(classes: list):
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    lines = [
        "# YOLOv8 Dataset Configuration",
        "# Generated from FamilyMart planogram JSON files",
        "",
        f"path: {OUTPUT_DIR.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(classes)}  # number of classes",
        "",
        "names:",
    ]
    for i, cls in enumerate(classes):
        lines.append(f"  {i}: '{cls}'")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✓ dataset.yaml saved: {yaml_path}")
    return yaml_path


def print_labeling_guide(classes: list):
    print(f"""
{'='*60}
  LABELING GUIDE — How to create training data
{'='*60}

You need labeled shelf photos to train YOLOv8.
For each product on a shelf photo, you draw a bounding box
and assign it a class (product name).

── HOW MANY PHOTOS DO YOU NEED? ────────────────────────────
  Minimum : 50–100 shelf photos (more = better accuracy)
  Per class: at least 20–30 examples of each product
  Recommended: 200–500 photos for production quality

── RECOMMENDED LABELING TOOLS ──────────────────────────────

  OPTION A — Roboflow (easiest, web-based, free tier):
    1. Go to https://roboflow.com
    2. Create a new project → Object Detection
    3. Upload your shelf photos
    4. Draw bounding boxes and assign class names
    5. Export in "YOLOv8" format
    6. Download → put images in training_data/images/train/
                    labels in training_data/labels/train/

  OPTION B — LabelImg (free, offline):
    Install: pip install labelImg
    Run:     labelImg
    Set format to YOLO before labeling
    Output goes to training_data/labels/

── YOLO LABEL FORMAT ────────────────────────────────────────
  One .txt file per image, same filename.
  Each line = one object:
    <class_id> <x_center> <y_center> <width> <height>
  All values normalized 0.0–1.0 relative to image size.

  Example (Pringles can at center-top of image):
    0 0.52 0.15 0.08 0.22

── YOUR CLASSES ({len(classes)} products from planograms) ──
""")
    for i, cls in enumerate(classes[:20]):
        print(f"  {i:>3}: {cls}")
    if len(classes) > 20:
        print(f"  ... and {len(classes) - 20} more (see classes.txt)")

    print(f"""
── FOLDER STRUCTURE EXPECTED ────────────────────────────────
  yolo_test/
  └── training_data/
      ├── images/
      │   ├── train/    ← 80% of your labeled photos
      │   └── val/      ← 20% of your labeled photos
      ├── labels/
      │   ├── train/    ← .txt label files (same names as images)
      │   └── val/      ← .txt label files
      ├── classes.txt   ← already generated ✓
      └── dataset.yaml  ← already generated ✓

{'='*60}
  NEXT STEP: Once you have labeled photos, run:
             python yolo_test/3_train.py
{'='*60}
""")


if __name__ == "__main__":
    print("\n── Extracting product classes from planogram JSONs...")
    classes = extract_classes_from_planograms()

    if not classes:
        exit(1)

    print(f"   Found {len(classes)} unique products across all planograms.\n")
    write_classes_file(classes)
    write_dataset_yaml(classes)
    print_labeling_guide(classes)
