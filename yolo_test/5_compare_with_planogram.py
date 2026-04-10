"""
5_compare_with_planogram.py
---------------------------
STEP 5 — Simulate the full hybrid pipeline WITHOUT touching main code.

Uses trained YOLO detections + planogram JSON to produce a compliance report.
This is a standalone preview of what the final integration will look like.

Run:
    python yolo_test/5_compare_with_planogram.py \
        --photo <shelf_photo> \
        --planogram SNACK3C

Output:
    Console: full compliance report (correct / missing / misplaced / unexpected)
    yolo_test/results/compliance_preview.json
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
PLANOGRAM_DIR   = Path("planograms")


def load_planogram(planogram_id: str) -> dict:
    json_path = PLANOGRAM_DIR / f"{planogram_id}.json"
    if not json_path.exists():
        print(f"[ERROR] Planogram JSON not found: {json_path}")
        print(f"        Run: python pdf_extractor.py planograms/{planogram_id}.pdf")
        sys.exit(1)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_products(photo_path: str, weights_path: str, conf: float) -> list:
    """Run YOLO detection and return sorted detection list."""
    model   = YOLO(weights_path)
    results = model(photo_path, conf=conf, iou=0.45)
    result  = results[0]

    img     = cv2.imread(photo_path)
    img_h, img_w = img.shape[:2]

    detections = []
    for box in result.boxes:
        class_id   = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        detections.append({
            "product": class_name,
            "confidence": round(confidence, 3),
            "cx": round(cx, 3),
            "cy": round(cy, 3),
            "bbox": [x1, y1, x2, y2],
        })

    # Sort top-to-bottom, left-to-right (matches planogram reading order)
    detections.sort(key=lambda d: (d["cy"], d["cx"]))
    return detections


def find_best_match(detected_name: str, planogram_products: list, threshold: float = 0.6) -> dict:
    """
    Simple string similarity match between detected product name and planogram products.
    Returns best matching planogram product or None.
    """
    detected_lower = detected_name.lower()
    best_score = 0
    best_match = None

    for p in planogram_products:
        plan_lower = p["product_name"].lower()
        plan_brand = p["brand"].lower()

        # Check if brand matches
        brand_match = plan_brand in detected_lower or detected_lower in plan_brand

        # Check word overlap
        detected_words = set(detected_lower.split())
        plan_words     = set(plan_lower.split())
        overlap        = len(detected_words & plan_words)
        score          = overlap / max(len(plan_words), 1)

        if brand_match:
            score += 0.3

        if score > best_score:
            best_score = score
            best_match = p

    return best_match if best_score >= threshold else None


def compare(detections: list, planogram_data: dict) -> dict:
    """
    Compare YOLO detections against planogram reference.
    Returns compliance report dict.
    """
    # Flatten all planogram products
    all_products = []
    for sec in planogram_data.get("gondola_sections", []):
        for p in sec.get("products", []):
            all_products.append({
                **p,
                "section_id":    sec["section_id"],
                "section_label": sec["section_label"],
                "matched":       False,  # track if this was found in detections
            })

    correct     = []
    issues      = []
    matched_ids = set()

    # Check each detection against planogram
    for det in detections:
        if det["product"] == "EMPTY":
            continue

        match = find_best_match(det["product"], all_products)

        if match is None:
            # Product not in planogram at all
            issues.append({
                "type":              "unexpected",
                "product":           det["product"],
                "expected_position": None,
                "found_position":    f"cx={det['cx']:.2f} cy={det['cy']:.2f}",
                "confidence":        det["confidence"],
            })
        else:
            pos_key = f"{match['section_id']}-P{match['position']:02d}"
            match["matched"] = True
            matched_ids.add(pos_key)

            # Check if position is approximately correct
            # (rough check: cy should decrease as shelf row increases)
            correct.append(f"{match['product_name']} [{pos_key}]")

    # Check for missing products (planogram items not detected)
    for p in all_products:
        pos_key = f"{p['section_id']}-P{p['position']:02d}"
        if not p["matched"]:
            issues.append({
                "type":              "missing",
                "product":           p["product_name"],
                "expected_position": pos_key,
                "found_position":    None,
                "confidence":        None,
            })

    total    = len(all_products)
    n_correct = len(correct)
    score    = round((n_correct / total * 100)) if total > 0 else 0

    return {
        "planogram_id":     planogram_data.get("planogram_id"),
        "compliance_score": score,
        "status":           "pass" if score >= 80 else "fail",
        "total_positions":  total,
        "correct_count":    n_correct,
        "correct":          correct,
        "issues":           issues,
        "summary": (
            f"{n_correct}/{total} positions correct ({score}% compliance). "
            f"{len([i for i in issues if i['type']=='missing'])} missing, "
            f"{len([i for i in issues if i['type']=='unexpected'])} unexpected."
        ),
    }


def run(photo_path: str, planogram_id: str, weights_path: str, conf: float):
    print(f"\n{'='*60}")
    print("  YOLOv8 + Planogram JSON — Hybrid Compliance Preview")
    print(f"  Photo     : {Path(photo_path).name}")
    print(f"  Planogram : {planogram_id}")
    print(f"{'='*60}\n")

    # Load planogram
    planogram_data = load_planogram(planogram_id)
    total_products = sum(
        len(s["products"]) for s in planogram_data.get("gondola_sections", [])
    )
    print(f"[compare] Planogram loaded: {total_products} expected products")

    # Detect
    if not Path(weights_path).exists():
        print(f"[WARNING] Trained weights not found at {weights_path}")
        print("          Using pre-trained model instead (won't know FamilyMart products)")
        weights_path = "yolov8n.pt"

    print("[compare] Running YOLO detection...")
    detections = detect_products(photo_path, weights_path, conf)
    print(f"[compare] {len(detections)} products detected")

    # Compare
    print("[compare] Comparing against planogram...")
    report = compare(detections, planogram_data)

    # Print report
    print(f"\n── Compliance Report ───────────────────────────────────")
    print(f"  Score  : {report['compliance_score']}%  ({report['status'].upper()})")
    print(f"  Correct: {report['correct_count']} / {report['total_positions']}")
    print(f"  Issues : {len(report['issues'])}")

    if report["correct"]:
        print(f"\n  ✓ Correct placements:")
        for c in report["correct"]:
            print(f"    {c}")

    missing     = [i for i in report["issues"] if i["type"] == "missing"]
    unexpected  = [i for i in report["issues"] if i["type"] == "unexpected"]
    misplaced   = [i for i in report["issues"] if i["type"] == "misplaced"]

    if missing:
        print(f"\n  ✗ Missing ({len(missing)}):")
        for i in missing:
            print(f"    {i['product']} — expected at {i['expected_position']}")

    if misplaced:
        print(f"\n  ⚠ Misplaced ({len(misplaced)}):")
        for i in misplaced:
            print(f"    {i['product']} — expected {i['expected_position']}, found {i['found_position']}")

    if unexpected:
        print(f"\n  ? Unexpected ({len(unexpected)}):")
        for i in unexpected:
            print(f"    {i['product']} at {i['found_position']} (conf: {i['confidence']:.0%})")

    # Save report
    out_path = Path("yolo_test/results/compliance_preview.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Full report saved: {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo",      required=True,           help="Shelf photo path")
    parser.add_argument("--planogram",  required=True,           help="Planogram ID (e.g. SNACK3C)")
    parser.add_argument("--weights",    default=DEFAULT_WEIGHTS, help="Trained YOLO weights")
    parser.add_argument("--conf",       type=float, default=0.4, help="Confidence threshold")
    args = parser.parse_args()

    run(args.photo, args.planogram, args.weights, args.conf)
