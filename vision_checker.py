"""
vision_checker.py
-----------------
Hybrid Gemini + Cloud Vision compliance checker.

Pipeline:
  [AI]     Step 1 — Gemini 2.5 Pro scans the shelf photo BLINDLY.
           No planogram hints. Reports only what it actually sees.
           Returns product names, brands, confidence, and bounding boxes.

  [AI]     Step 2 — For ambiguous/low-confidence detections, Gemini
           uses Google Search to confirm the exact product variant.

  [CV]     Step 2b (optional) — If Cloud Vision Product Search is
           configured, cross-check each detection against the reference
           catalog for extra confidence. This is additive — Gemini does
           the heavy lifting, Cloud Vision just validates.

  [Code]   Step 3 — Pure Python comparison of the verified detection
           list against the planogram JSON.

Output fields:
  found_on_shelf     : in planogram AND detected in photo  ✅
  missing_from_shelf : in planogram but NOT in photo       ❌
  not_in_planogram   : detected in photo but NOT in planogram ⚠️
"""

from google import genai
from google.genai import types
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
from difflib import SequenceMatcher
from typing import List, Optional
import io
import os
import json

from pdf_extractor import extract_planogram

load_dotenv()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class DetectedProduct(BaseModel):
    product_name:    str
    brand:           str
    color_hint:      str
    confidence:      int            # 0-100
    needs_web_check: bool           # True if ambiguous / low confidence
    ambiguity_note:  Optional[str]  # e.g. "unsure if Balado or BBQ flavor"
    bbox:            Optional[List[int]]  # [y_min, x_min, y_max, x_max] 0-1000

class DetectionResult(BaseModel):
    detected: List[DetectedProduct]

class DetectedProductDualRow(BaseModel):
    product_name:    str
    brand:           str
    color_hint:      str
    confidence:      int
    shelf_row:       int            # 1 = top shelf row in photo, 2 = bottom shelf row
    needs_web_check: bool
    ambiguity_note:  Optional[str]
    bbox:            Optional[List[int]]

class DualRowDetectionResult(BaseModel):
    detected: List[DetectedProductDualRow]

class VerifiedProduct(BaseModel):
    original_name:  str   # what Gemini saw
    verified_name:  str   # confirmed name after web search
    verified_brand: str
    confidence:     int
    search_used:    bool
    search_query:   Optional[str]
    reasoning:      str   # why this verification was made

class VerificationResult(BaseModel):
    products: List[VerifiedProduct]


# ── Cloud Vision availability check ──────────────────────────────────────────
# Cloud Vision Product Search is OPTIONAL. If the deps or env vars are missing
# the pipeline still works — Gemini alone handles detection.

_CLOUD_VISION_AVAILABLE = False
try:
    from google.cloud import vision as cloud_vision
    from PIL import Image
    _gcp_project = os.getenv("GCP_PROJECT_ID")
    _gcp_location = os.getenv("GCP_LOCATION", "asia-east1")
    _product_set_id = os.getenv("PRODUCT_SET_ID", "familymart-snacks")
    _product_category = os.getenv("PRODUCT_CATEGORY", "packagedgoods-v1")
    _cv_threshold = float(os.getenv("PRODUCT_MATCH_THRESHOLD", "0.55"))
    if _gcp_project and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        _CLOUD_VISION_AVAILABLE = True
        print("[vision_checker] ✓ Cloud Vision Product Search is ENABLED (will cross-check detections)")
    else:
        print("[vision_checker] · Cloud Vision deps installed but GCP env vars not set — running Gemini-only mode")
except ImportError:
    print("[vision_checker] · google-cloud-vision not installed — running Gemini-only mode")


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_DETECTION = """
You are a shelf scanning system for a retail store in Indonesia.
Your only job: look at the photo and report exactly what products you can see.
Do NOT guess. Do NOT assume. Only report what is clearly visible.
If a label is unreadable, describe the packaging visually instead of guessing a name.
Return JSON only.
"""

SYSTEM_PROMPT_VERIFICATION = """
You are a product verification assistant for an Indonesian retail store.
You have access to Google Search to look up snack product details.
Your job: confirm the exact product name, brand, and variant for ambiguous detections.

When searching:
- Search in Indonesian or English: e.g. "QTELA singkong varian rasa Indonesia"
- Look for packaging images to match color and design
- Focus on confirming the FLAVOR/VARIANT specifically
- Indonesian snack brands to know: Chitato, Qtela, Kusuka, Tao Kae Noi, Potabee,
  Japota, Chiki, Cheetos, Oishi, Piattos, Taro, Happytos, Garuda, Dua Kelinci, etc.
"""

MATCH_THRESHOLD = 0.70


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_bbox(raw) -> Optional[dict]:
    """Convert Gemini's [y_min, x_min, y_max, x_max] 0-1000 scale to {x, y, w, h} normalized 0-1."""
    if not raw or len(raw) < 4:
        return None
    y_min, x_min, y_max, x_max = raw[0], raw[1], raw[2], raw[3]
    x = max(0.0, min(1.0, x_min / 1000.0))
    y = max(0.0, min(1.0, y_min / 1000.0))
    w = max(0.01, min(1.0 - x, (x_max - x_min) / 1000.0))
    h = max(0.01, min(1.0 - y, (y_max - y_min) / 1000.0))
    return {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)}


def _parse_json(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[parser] JSON truncated at char {e.pos}. Attempting partial recovery...")
        try:
            cutoff = raw.rfind("},")
            if cutoff > 0:
                partial = raw[:cutoff + 1] + "]}"
                return json.loads(partial)
        except Exception:
            pass
        print(f"[parser] Could not recover. Raw snippet: {raw[:200]}")
        return {"detected": [], "products": []}


def _similarity(a: str, b: str) -> float:
    a = a.lower().strip()
    b = b.lower().strip()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.85
    a_words = set(a.split())
    b_words = set(b.split())
    common  = a_words & b_words
    if len(common) >= 2:
        word_score = len(common) / max(len(a_words), len(b_words))
        return max(word_score, SequenceMatcher(None, a, b).ratio())
    return SequenceMatcher(None, a, b).ratio()


def _get_planogram_products(planogram_data: dict) -> list:
    seen, products = set(), []
    for sec in planogram_data.get("gondola_sections", []):
        for p in sec.get("products", []):
            key = p["product_name"].lower().strip()
            if key not in seen:
                seen.add(key)
                products.append({
                    "product_name": p["product_name"],
                    "brand":        p["brand"],
                    "color_hint":   p.get("color_hint", ""),
                    "size_hint":    p.get("size_hint", ""),
                    "section_id":   sec["section_id"],
                })
    return products


# ── Step 1 (AI): Blind detection ──────────────────────────────────────────────

def _image_to_json(client, image_part: types.Part) -> list:
    """
    Gemini scans the shelf photo with ZERO knowledge of the planogram.
    Returns (reliable_list, ambiguous_list).
    """
    prompt = """
Scan this retail shelf photo carefully — top to bottom, left to right.
List every product you can clearly see.

Rules:
- Only report products that are CLEARLY VISIBLE in the photo
- Do NOT guess or invent products
- Read the label as printed on the packaging
- product_name    : name as printed on packaging
- brand           : brand name as printed on packaging
- color_hint      : dominant packaging color(s), be specific
                    e.g. "dark green bag with yellow logo"
- confidence      : 100 = clearly readable, 60 = partially visible, 30 = guessing
- needs_web_check : true if you are UNSURE of the exact flavor/variant/size
                    Common cases:
                    · Multiple variants with similar packaging colors
                      (e.g. QTELA has Balado=red, Original=yellow, BBQ=brown)
                    · Label is partially blocked or at an angle
                    · You can read the brand but NOT the variant name
- ambiguity_note  : describe WHY you're unsure, what visual cues you see
                    e.g. "can see QTELA brand clearly, packaging is reddish-orange
                    but cannot confirm if Balado or Spicy variant"
- bbox            : bounding box [y_min, x_min, y_max, x_max] as integers 0-1000
                    where 0=top/left edge and 1000=bottom/right edge of the image
                    Draw the box tight around the visible product label or packaging face
                    Set to null only if the product position is truly unclear
"""

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=["SHELF PHOTO:", image_part, prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_DETECTION,
            response_mime_type="application/json",
            response_schema=DetectionResult,
            temperature=0.0,
        ),
    )

    if response.parsed:
        detected = []
        for p in response.parsed.detected:
            d = p.model_dump()
            d["bbox"] = _normalize_bbox(d.get("bbox"))
            detected.append(d)
    else:
        data     = _parse_json(response.text)
        raw_list = data.get("detected", [])
        detected = []
        for d in raw_list:
            d["bbox"] = _normalize_bbox(d.get("bbox"))
            detected.append(d)

    needs_check = [p for p in detected if p.get("needs_web_check")]
    reliable    = [p for p in detected if not p.get("needs_web_check") and p["confidence"] >= 50]

    print(f"\n[step-1] Gemini blind detection — {len(detected)} total products seen:")
    for p in detected:
        flag  = "✓" if p["confidence"] >= 60 else "~"
        web   = " 🔍[needs web check]" if p.get("needs_web_check") else ""
        note  = f" → {p['ambiguity_note']}" if p.get("ambiguity_note") else ""
        print(f"  {flag} [{p['confidence']}%] {p['product_name']} | {p['brand']} | {p['color_hint']}{web}{note}")

    print(f"\n[step-1] Summary: {len(reliable)} clear | {len(needs_check)} need web verification | "
          f"{len(detected) - len(reliable) - len(needs_check)} dropped (low confidence)")

    return reliable, needs_check


# ── Step 2 helpers ────────────────────────────────────────────────────────────

def _parse_verification_response(raw_text: str, fallback_products: list) -> list:
    """
    Parses the free-form text response from Step 2 (google_search mode).
    Gemini's google_search tool is INCOMPATIBLE with response_mime_type +
    response_schema — we parse the text output ourselves.
    """
    text = raw_text.strip()

    # Remove markdown fences if present
    if "```" in text:
        for part in text.split("```"):
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                text = candidate
                break

    # Find outermost { ... } block
    start = text.find("{")
    if start != -1:
        depth, end, in_str, escape = 0, -1, False, False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end != -1:
            json_str = text[start:end + 1]
            try:
                data = json.loads(json_str)
                products = data.get("products", [])
                if products:
                    print(f"[step-2] Parsed {len(products)} verified product(s) from JSON.")
                    return products
            except json.JSONDecodeError as e:
                print(f"[step-2] JSON parse failed: {e}. Trying partial recovery...")
                cutoff = json_str.rfind("},")
                if cutoff > 0:
                    try:
                        data = json.loads(json_str[:cutoff + 1] + "]}")
                        products = data.get("products", [])
                        if products:
                            print(f"[step-2] Partial recovery: {len(products)} product(s).")
                            return products
                    except Exception:
                        pass

    # Passthrough fallback — keep original names, pipeline never crashes
    print(f"[step-2] ⚠️  JSON extraction failed. Passthrough for {len(fallback_products)} product(s).")
    print(f"[step-2] Raw snippet: {raw_text[:300]}")
    return [
        {
            "original_name":  p["product_name"],
            "verified_name":  p["product_name"],
            "verified_brand": p["brand"],
            "confidence":     p["confidence"],
            "search_used":    False,
            "search_query":   None,
            "reasoning":      "Passthrough — JSON parse failed, using original detection.",
        }
        for p in fallback_products
    ]


# ── Step 2 (AI + Web Search): Verify ambiguous detections ────────────────────

def _verify_with_web(client, ambiguous_products: list) -> list:
    """
    For each product flagged as ambiguous, Gemini uses Google Search
    to look up packaging images and confirm the exact variant.
    """
    if not ambiguous_products:
        print("\n[step-2] No ambiguous products — skipping web verification.")
        return []

    print(f"\n[step-2] Web verification for {len(ambiguous_products)} ambiguous product(s)...")

    products_text = "\n".join([
        f"{i+1}. Brand: {p['brand']} | Seen as: {p['product_name']} | "
        f"Color: {p['color_hint']} | Confidence: {p['confidence']}% | "
        f"Ambiguity: {p.get('ambiguity_note', 'unclear variant')}"
        for i, p in enumerate(ambiguous_products)
    ])

    prompt = f"""
For each product below, search Google to confirm the EXACT product name and variant.

PRODUCTS TO VERIFY:
{products_text}

For each product, you MUST:
1. Search: "[brand name] snack Indonesia varian" OR "[brand] [color] packaging"
   Examples:
   - "QTELA singkong snack Indonesia semua varian"
   - "Kusuka keripik singkong varian rasa kemasan"
   - "Chitato Lite varian rasa packaging warna"
   - "Tao Kae Noi Big Sheet varian Indonesia"

2. Use packaging COLOR as your primary discriminator:
   Common color-variant patterns for Indonesian snacks:
   · QTELA: Balado=red/orange, Original=yellow, BBQ=brown/dark, Barbeque=dark brown
   · Kusuka: Balado=red, BBQ=brown, Original=yellow, Keju=orange, Super Pedas=red+chili
   · Chitato Lite: Nori Seaweed=dark green, Sour Cream=white/blue, Ayam Bawang=yellow
   · Chitato regular: Ayam Bumbu=yellow, Sapi BBQ=red, Original=plain
   · Tao Kae Noi Big Sheet: Classic=green, Spicy=red, Chili Garlic=orange/yellow
   · Potabee: BBQ=dark, Seaweed=green, Salted Egg=yellow/gold, Spicy=red
   · Japota: Original=yellow, Honey Butter=yellow/gold, Seaweed=green
   · Happytos: Merah (red bag)=Corn Chips, Hijau (green bag)=Tortilla
   · Tos Tos: Original=blue/yellow, Nacho Cheese=orange, Korean BBQ=dark red

3. verified_name must be the EXACT product name with variant
   e.g. "Qtela Singkong Balado 100g" not just "Qtela Singkong"

IMPORTANT: You MUST respond with ONLY a valid JSON object.
No markdown, no explanation text, no code fences.
Exact format required:
{{
  "products": [
    {{
      "original_name": "...",
      "verified_name": "...",
      "verified_brand": "...",
      "confidence": 85,
      "search_used": true,
      "search_query": "...",
      "reasoning": "..."
    }}
  ]
}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_VERIFICATION,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.0,
        ),
    )

    raw_text = response.text if response.text else ""
    verified = _parse_verification_response(raw_text, ambiguous_products)

    print(f"\n[step-2] Web verification results:")
    for v in verified:
        search_flag = "🔍 searched" if v.get("search_used") else "💭 inferred"
        changed     = " ← CORRECTED" if v["original_name"].lower() != v["verified_name"].lower() else ""
        print(f"  {search_flag} | {v['original_name']} → {v['verified_name']}{changed}")
        if v.get("search_query"):
            print(f"           Query: \"{v['search_query']}\"")
        print(f"           Reasoning: {v['reasoning'][:100]}...")

    # Convert back to DetectedProduct format — preserve bbox from step 1
    orig_bbox_map = {p["product_name"]: p.get("bbox") for p in ambiguous_products}
    result = []
    for v in verified:
        orig_bbox = orig_bbox_map.get(v.get("original_name", ""))
        result.append({
            "product_name":    v["verified_name"],
            "brand":           v["verified_brand"],
            "color_hint":      "",
            "confidence":      v["confidence"],
            "needs_web_check": False,
            "ambiguity_note":  None,
            "bbox":            orig_bbox,
        })

    return result


# ── Step 2b (optional): Cloud Vision Product Search cross-check ──────────────

def _cloud_vision_crosscheck(detections: list, shelf_photo_bytes: bytes) -> list:
    """
    OPTIONAL enhancement: for each Gemini detection with a bbox, crop the
    region and run Vision Product Search. If Product Search returns a match,
    we boost confidence. If it returns a DIFFERENT name, we flag it.

    This does NOT replace Gemini — it supplements. If Cloud Vision is not
    configured, this function is a no-op and returns detections unchanged.
    """
    if not _CLOUD_VISION_AVAILABLE:
        return detections

    print(f"\n[step-2b] Cloud Vision cross-check for {len(detections)} detection(s)…")

    try:
        cv_client = cloud_vision.ImageAnnotatorClient()
        pil_image = Image.open(io.BytesIO(shelf_photo_bytes))
        W, H = pil_image.size

        product_set_path = (
            f"projects/{_gcp_project}/locations/{_gcp_location}"
            f"/productSets/{_product_set_id}"
        )

        for idx, det in enumerate(detections):
            bbox = det.get("bbox")
            if not bbox:
                continue

            try:
                # Crop the bbox region
                left   = max(0, int(round(bbox["x"] * W)))
                top    = max(0, int(round(bbox["y"] * H)))
                right  = min(W, int(round((bbox["x"] + bbox["w"]) * W)))
                bottom = min(H, int(round((bbox["y"] + bbox["h"]) * H)))
                if right <= left or bottom <= top:
                    continue

                crop = pil_image.crop((left, top, right, bottom))
                if crop.mode != "RGB":
                    crop = crop.convert("RGB")
                buf = io.BytesIO()
                crop.save(buf, format="JPEG", quality=90)
                crop_bytes = buf.getvalue()

                # Run Product Search on the crop
                image = cloud_vision.Image(content=crop_bytes)
                ps_params = cloud_vision.ProductSearchParams(
                    product_set=product_set_path,
                    product_categories=[_product_category],
                )
                ctx = cloud_vision.ImageContext(product_search_params=ps_params)
                response = cv_client.product_search(image=image, image_context=ctx)

                results = response.product_search_results.results
                if results:
                    best = max(results, key=lambda r: r.score)
                    if best.score >= _cv_threshold:
                        cv_name = best.product.display_name or best.product.name.split("/")[-1]
                        sim = _similarity(det["product_name"], cv_name)
                        if sim >= 0.6:
                            # Matches Gemini — boost confidence
                            det["confidence"] = min(100, det["confidence"] + 10)
                            det["cv_confirmed"] = True
                            print(f"  [{idx+1}] ✓ CV confirms: {det['product_name']} (boost +10)")
                        else:
                            # Disagrees — log but keep Gemini's answer (it reads labels)
                            det["cv_suggestion"] = cv_name
                            print(f"  [{idx+1}] ~ CV suggests '{cv_name}' but Gemini says '{det['product_name']}' — keeping Gemini")
                    else:
                        print(f"  [{idx+1}] · CV no confident match for '{det['product_name']}'")
                else:
                    print(f"  [{idx+1}] · CV returned no results for '{det['product_name']}'")

            except Exception as exc:
                print(f"  [{idx+1}] · CV error: {exc}")
                continue

    except Exception as exc:
        print(f"[step-2b] Cloud Vision cross-check failed: {exc} — continuing with Gemini results")

    return detections


# ── Step 3 (Code): Compare detected vs planogram ─────────────────────────────

def _compare(detected: list, planogram_products: list) -> dict:
    """
    Pure Python comparison — no AI, no hallucination.

    Side effect: mutates each dict in `detected` to add:
        - "status"            → "correct" | "unexpected" | "unknown"
        - "matched_planogram" → planogram product name (if matched)
    """
    found_on_shelf     = []
    missing_from_shelf = []
    not_in_planogram   = []
    matched_indices    = set()

    # Every detection starts as "unknown"
    for d in detected:
        d["status"] = "unknown"
        d["matched_planogram"] = None

    print(f"\n[step-3] Matching {len(detected)} detected → {len(planogram_products)} planogram products")
    print(f"         Match threshold: {MATCH_THRESHOLD}\n")

    for plan_p in planogram_products:
        best_score = 0.0
        best_idx   = -1
        best_name  = ""

        for i, det_p in enumerate(detected):
            if i in matched_indices:
                continue
            score = _similarity(plan_p["product_name"], det_p["product_name"])
            pb = plan_p["brand"].lower()
            db = det_p["brand"].lower()
            if pb and db and (pb in db or db in pb):
                score = min(1.0, score + 0.10)
            if score > best_score:
                best_score = score
                best_idx   = i
                best_name  = det_p["product_name"]

        if best_score >= MATCH_THRESHOLD and best_idx != -1:
            found_on_shelf.append(plan_p["product_name"])
            matched_indices.add(best_idx)
            detected[best_idx]["status"] = "correct"
            detected[best_idx]["matched_planogram"] = plan_p["product_name"]
            match_note = f" ← matched '{best_name}'" if best_name.lower() != plan_p["product_name"].lower() else ""
            print(f"  ✅ FOUND   : {plan_p['product_name']}{match_note} (score: {best_score:.2f})")
        else:
            missing_from_shelf.append(plan_p["product_name"])
            print(f"  ❌ MISSING : {plan_p['product_name']}"
                  + (f" (closest: '{best_name}' score: {best_score:.2f})" if best_name else ""))

    for i, det_p in enumerate(detected):
        if i not in matched_indices:
            det_p["status"] = "unexpected"
            not_in_planogram.append(det_p["product_name"])
            print(f"  ⚠️  EXTRA  : {det_p['product_name']} [{det_p.get('brand', '')}] — not in planogram")

    total            = len(planogram_products)
    n_found          = len(found_on_shelf)
    compliance_score = round((n_found / total * 100)) if total > 0 else 0
    status           = "pass" if compliance_score >= 95 else "fail"

    summary_parts = [
        f"{n_found} of {total} planogram products found on shelf ({compliance_score}% compliance)."
    ]
    if missing_from_shelf:
        preview = ", ".join(missing_from_shelf[:3])
        extra   = f" and {len(missing_from_shelf) - 3} more" if len(missing_from_shelf) > 3 else ""
        summary_parts.append(f"Missing: {preview}{extra}.")
    if not_in_planogram:
        summary_parts.append(f"{len(not_in_planogram)} product(s) on shelf not in planogram.")

    return {
        "found_on_shelf":     found_on_shelf,
        "missing_from_shelf": missing_from_shelf,
        "not_in_planogram":   not_in_planogram,
        "compliance_score":   compliance_score,
        "status":             status,
        "summary":            " ".join(summary_parts),
    }


# ── Dual-row image scan (Testing mode) ───────────────────────────────────────

def _image_to_json_dual_row(client, image_part: types.Part) -> tuple:
    """
    Gemini scans a photo that contains TWO shelf rows stacked vertically.
    Returns (reliable_row1, reliable_row2, ambiguous_row1, ambiguous_row2).
    shelf_row=1 → top half of image, shelf_row=2 → bottom half.
    """
    prompt = """
This shelf photo contains exactly TWO horizontal shelf rows stacked on top of each other.
The TOP portion of the image is Shelf Row 1.
The BOTTOM portion of the image is Shelf Row 2.

Scan every product visible in BOTH rows — left to right within each row.

For every product set:
- product_name    : name exactly as printed on the packaging
- brand           : brand name on the packaging
- color_hint      : dominant packaging color(s), e.g. "dark green bag with yellow logo"
- confidence      : 100=clearly readable, 60=partially visible, 30=guessing
- shelf_row       : 1 if the product is in the TOP row, 2 if it is in the BOTTOM row
                    Use the bbox y-coordinates as your guide:
                    if the product center y < 500 it is Row 1, else Row 2
- needs_web_check : true only if you cannot confirm the exact flavor/variant
- ambiguity_note  : why you are unsure (if needs_web_check is true)
- bbox            : [y_min, x_min, y_max, x_max] as integers 0-1000
                    tight around the product label/face, null only if truly unknown

IMPORTANT: every product must be assigned shelf_row 1 or 2. Do not skip this field.
"""

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=["DUAL-ROW SHELF PHOTO:", image_part, prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_DETECTION,
            response_mime_type="application/json",
            response_schema=DualRowDetectionResult,
            temperature=0.0,
        ),
    )

    if response.parsed:
        raw_list = [p.model_dump() for p in response.parsed.detected]
    else:
        data = _parse_json(response.text)
        raw_list = data.get("detected", [])

    detected = []
    for d in raw_list:
        d["bbox"] = _normalize_bbox(d.get("bbox"))
        # Fallback: derive shelf_row from bbox.y if Gemini forgot to set it
        if not d.get("shelf_row") and d.get("bbox"):
            d["shelf_row"] = 1 if d["bbox"]["y"] < 0.5 else 2
        elif not d.get("shelf_row"):
            d["shelf_row"] = 1
        detected.append(d)

    row1 = [d for d in detected if d.get("shelf_row") == 1]
    row2 = [d for d in detected if d.get("shelf_row") == 2]

    print(f"\n[dual-row] Detected: {len(row1)} in Row 1 (top) · {len(row2)} in Row 2 (bottom)")
    for r, label in [(row1, "Row1"), (row2, "Row2")]:
        for p in r:
            flag = "✓" if p["confidence"] >= 60 else "~"
            web  = " 🔍" if p.get("needs_web_check") else ""
            print(f"  {label} {flag} [{p['confidence']}%] {p['product_name']}{web}")

    reliable_r1  = [p for p in row1 if not p.get("needs_web_check") and p["confidence"] >= 50]
    ambiguous_r1 = [p for p in row1 if p.get("needs_web_check")]
    reliable_r2  = [p for p in row2 if not p.get("needs_web_check") and p["confidence"] >= 50]
    ambiguous_r2 = [p for p in row2 if p.get("needs_web_check")]

    return reliable_r1, reliable_r2, ambiguous_r1, ambiguous_r2


def _positional_compare(section_id: str, section_products: list, detected_ordered: list) -> dict:
    """
    Position-by-position comparison of one row's detected products against planogram.
    detected_ordered must already be sorted left-to-right by bbox.x.
    """
    plan_products = sorted(section_products, key=lambda p: p.get("position", 99))
    n_plan = len(plan_products)
    n_det  = len(detected_ordered)
    n_max  = max(n_plan, n_det) if (n_plan or n_det) else 0

    position_results = []
    correct_count = 0

    for i in range(n_max):
        plan_p = plan_products[i] if i < n_plan else None
        det_p  = detected_ordered[i] if i < n_det else None

        sim = None
        if plan_p and det_p:
            sim = _similarity(plan_p["product_name"], det_p["product_name"])
            pb = plan_p.get("brand", "").lower()
            db = det_p.get("brand",  "").lower()
            if pb and db and (pb in db or db in pb):
                sim = min(1.0, sim + 0.10)
            status = "correct" if sim >= MATCH_THRESHOLD else "wrong_product"
            if status == "correct":
                correct_count += 1
        elif plan_p and not det_p:
            status = "missing"
        else:
            status = "unexpected"

        icon = {"correct": "✅", "wrong_product": "⚠️", "missing": "❌", "unexpected": "⚠️"}.get(status, "?")
        print(f"  {icon} [{section_id}] Pos {i+1}: "
              + (plan_p["product_name"][:25] if plan_p else "—")
              + " vs "
              + (det_p["product_name"][:25] if det_p else "—")
              + (f" (sim={sim:.2f})" if sim is not None else ""))

        position_results.append({
            "position":            i + 1,
            "planogram_product":   plan_p["product_name"]       if plan_p else None,
            "planogram_color":     plan_p.get("color_hint", "") if plan_p else None,
            "detected_product":    det_p["product_name"]        if det_p  else None,
            "detected_brand":      det_p.get("brand", "")       if det_p  else None,
            "detected_confidence": det_p.get("confidence", 0)   if det_p  else 0,
            "detected_bbox":       det_p.get("bbox")            if det_p  else None,
            "similarity_score":    round(sim, 2)                if sim is not None else None,
            "status":              status,
        })

    compliance_score = round(correct_count / n_plan * 100) if n_plan > 0 else 0
    return {
        "section_id":       section_id,
        "compliance_score": compliance_score,
        "status":           "pass" if compliance_score >= 95 else "fail",
        "planogram_count":  n_plan,
        "detected_count":   n_det,
        "correct_count":    correct_count,
        "position_results": position_results,
    }


# ── Positional entry point (Testing mode) ────────────────────────────────────

def check_compliance_positional(
    section_id: str,
    section_products: list,
    shelf_photo_bytes: bytes,
    planogram_id: str,
    store_id: str,
    image_media_type: str = "image/jpeg",
) -> dict:
    """Single-row position-aware check (kept for compatibility)."""
    print(f"\n[positional] ── Section {section_id} · {planogram_id} / {store_id} ──")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set.")
    client = genai.Client(api_key=api_key)

    image_part = types.Part.from_bytes(data=shelf_photo_bytes, mime_type=image_media_type)

    clear_detections, ambiguous = _image_to_json(client, image_part)
    verified = _verify_with_web(client, ambiguous)
    all_detections = clear_detections + verified
    all_detections = _cloud_vision_crosscheck(all_detections, shelf_photo_bytes)

    with_bbox    = sorted([d for d in all_detections if d.get("bbox")],
                          key=lambda d: d["bbox"]["x"])
    without_bbox = [d for d in all_detections if not d.get("bbox")]
    detected_ordered = with_bbox + without_bbox

    return _positional_compare(section_id, section_products, detected_ordered)


def check_compliance_positional_dual_row(
    section_id_1: str,
    section_products_1: list,
    section_id_2: str,
    section_products_2: list,
    shelf_photo_bytes: bytes,
    planogram_id: str,
    store_id: str,
    image_media_type: str = "image/jpeg",
) -> dict:
    """
    Position-aware compliance check for a photo that shows TWO shelf rows.

    Pipeline:
      1. Gemini scans the photo identifying products per row (top/bottom)
         and returning bboxes + shelf_row assignment.
      2. Web-verify any ambiguous detections (per row).
      3. Sort each row's detections left-to-right by bbox.x.
      4. Compare each row position-by-position against its planogram section.

    Returns results for both sections under keys 'row1' and 'row2'.
    """
    print(f"\n[dual-row] ── {section_id_1}+{section_id_2} · {planogram_id} / {store_id} ──")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set.")
    client = genai.Client(api_key=api_key)

    image_part = types.Part.from_bytes(data=shelf_photo_bytes, mime_type=image_media_type)

    # Step 1 — dual-row blind scan
    print(f"[dual-row] Step 1 — Scanning both rows…")
    reliable_r1, reliable_r2, ambig_r1, ambig_r2 = _image_to_json_dual_row(client, image_part)

    # Step 2 — web-verify ambiguous products per row
    verified_r1 = _verify_with_web(client, ambig_r1)
    verified_r2 = _verify_with_web(client, ambig_r2)

    all_r1 = reliable_r1 + verified_r1
    all_r2 = reliable_r2 + verified_r2

    # Step 2b — optional Cloud Vision cross-check (runs on all detections)
    all_combined = _cloud_vision_crosscheck(all_r1 + all_r2, shelf_photo_bytes)
    # Re-split after cross-check (shelf_row field is preserved)
    all_r1 = [d for d in all_combined if d.get("shelf_row", 1) == 1]
    all_r2 = [d for d in all_combined if d.get("shelf_row", 1) == 2]

    # Step 3 — sort each row left-to-right by bbox.x
    def _sort_lr(detections):
        with_bbox    = sorted([d for d in detections if d.get("bbox")],
                              key=lambda d: d["bbox"]["x"])
        without_bbox = [d for d in detections if not d.get("bbox")]
        return with_bbox + without_bbox

    ordered_r1 = _sort_lr(all_r1)
    ordered_r2 = _sort_lr(all_r2)

    print(f"\n[dual-row] {section_id_1} (top row) order: "
          + " | ".join(d["product_name"][:18] for d in ordered_r1))
    print(f"[dual-row] {section_id_2} (bottom row) order: "
          + " | ".join(d["product_name"][:18] for d in ordered_r2))

    # Step 4 — position-by-position comparison for each row
    result_r1 = _positional_compare(section_id_1, section_products_1, ordered_r1)
    result_r2 = _positional_compare(section_id_2, section_products_2, ordered_r2)

    total_correct = result_r1["correct_count"] + result_r2["correct_count"]
    total_plan    = result_r1["planogram_count"] + result_r2["planogram_count"]
    overall_score = round(total_correct / total_plan * 100) if total_plan > 0 else 0

    print(f"\n[dual-row] {section_id_1}: {result_r1['correct_count']}/{result_r1['planogram_count']} → {result_r1['compliance_score']}%")
    print(f"[dual-row] {section_id_2}: {result_r2['correct_count']}/{result_r2['planogram_count']} → {result_r2['compliance_score']}%")
    print(f"[dual-row] Combined: {total_correct}/{total_plan} → {overall_score}%")

    return {
        "photo_key":        f"{section_id_1}_{section_id_2}",
        "compliance_score": overall_score,
        "status":           "pass" if overall_score >= 95 else "fail",
        "row1":             result_r1,
        "row2":             result_r2,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def check_compliance(
    pdf_path: str,
    shelf_photo_bytes: bytes,
    planogram_id: str,
    store_id: str,
    image_media_type: str = "image/jpeg",
) -> dict:
    print(f"\n[vision_checker] ── Check: {planogram_id} / {store_id} ──")

    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        client = genai.Client(api_key=api_key)

        # Load planogram reference
        print("[vision_checker] Loading planogram JSON...")
        planogram_data     = extract_planogram(pdf_path)
        planogram_products = _get_planogram_products(planogram_data)
        print(f"[vision_checker] {len(planogram_products)} unique products in planogram.")

        # Prepare image
        image_part = types.Part.from_bytes(
            data=shelf_photo_bytes,
            mime_type=image_media_type,
        )

        # ── Step 1: Blind detection (Gemini) ────────────────────────────────
        print("\n[vision_checker] Step 1 — Blind shelf scan (no hints)...")
        clear_detections, ambiguous = _image_to_json(client, image_part)

        # ── Step 2: Web search verification for ambiguous products ──────────
        print("\n[vision_checker] Step 2 — Web verification for ambiguous products...")
        verified_detections = _verify_with_web(client, ambiguous)

        # Merge clear + verified into one final detection list
        all_detections = clear_detections + verified_detections
        print(f"\n[vision_checker] Final detection list: "
              f"{len(clear_detections)} clear + {len(verified_detections)} verified "
              f"= {len(all_detections)} total")

        # ── Step 2b: Cloud Vision cross-check (optional) ───────────────────
        all_detections = _cloud_vision_crosscheck(all_detections, shelf_photo_bytes)

        # ── Step 3: Python comparison ───────────────────────────────────────
        print("\n[vision_checker] Step 3 — Comparing against planogram...")
        result = _compare(all_detections, planogram_products)

        print(
            f"\n[vision_checker] ── Final Result ──\n"
            f"  found_on_shelf     : {len(result['found_on_shelf'])}\n"
            f"  missing_from_shelf : {len(result['missing_from_shelf'])}\n"
            f"  not_in_planogram   : {len(result['not_in_planogram'])}\n"
            f"  compliance_score   : {result['compliance_score']}%\n"
            f"  status             : {result['status']}"
        )

        # Build position list for frontend canvas overlay
        detected_with_positions = [
            {
                "product_name":      d["product_name"],
                "brand":             d.get("brand", ""),
                "bbox":              d["bbox"],   # {x, y, w, h} normalized 0-1
                "status":            d.get("status", "unknown"),
                "matched_planogram": d.get("matched_planogram"),
            }
            for d in all_detections
            if d.get("bbox")
        ]
        print(f"[vision_checker] {len(detected_with_positions)} of {len(all_detections)} detections have bbox data.")

        return {
            "planogram_id":            planogram_id,
            "store_id":                store_id,
            "status":                  result["status"],
            "compliance_score":        result["compliance_score"],
            "found_on_shelf":          result["found_on_shelf"],
            "missing_from_shelf":      result["missing_from_shelf"],
            "not_in_planogram":        result["not_in_planogram"],
            "detected_with_positions": detected_with_positions,
            "summary":                 result["summary"],
            "timestamp":               datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        print(f"[vision_checker] ERROR: {e}")
        return {
            "planogram_id":            planogram_id,
            "store_id":                store_id,
            "status":                  "error",
            "compliance_score":        0,
            "found_on_shelf":          [],
            "missing_from_shelf":      [],
            "not_in_planogram":        [],
            "detected_with_positions": [],
            "summary":                 f"AI Engine Error: {str(e)}",
            "timestamp":               datetime.utcnow().isoformat() + "Z",
        }
