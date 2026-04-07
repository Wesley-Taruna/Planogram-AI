"""
vision_checker.py
-----------------
3-step AI compliance engine:

  Step A — Section identification:
            Photo → which planogram section is this? (fingerprint matching)

  Step B — Product detection:
            Photo → structured list of every visible product with row + position

  Step C — Compliance comparison:
            Detected products + planogram reference → correct / missing / misplaced / unexpected
"""

from google import genai
from google.genai import types
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional
import os
import json

from models import CheckResult
from pdf_extractor import extract_planogram

load_dotenv()

SYSTEM_PROMPT = """
You are a planogram compliance checker for FamilyMart Indonesia.
You help store supervisors verify that shelf products match the official planogram layout.
Always be thorough, precise, and objective. Never skip products.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_response(text: str) -> dict:
    """Strip markdown fences and parse JSON from a Gemini response."""
    raw = text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def _build_planogram_text(planogram_data: dict, section_id: Optional[str] = None) -> str:
    """
    Render the extracted planogram JSON as a numbered position list.
    If section_id is given, only render that section.
    """
    lines = []
    lines.append(f"Planogram: {planogram_data.get('planogram_id')} — {planogram_data.get('planogram_title')}")
    lines.append(f"Category: {planogram_data.get('category')}")
    lines.append("")

    for sec in planogram_data.get("gondola_sections", []):
        sid = sec.get("section_id")
        if section_id and sid != section_id:
            continue

        lines.append(f"SECTION {sid} — {sec.get('section_label')} [{sec.get('shelf_level')}]")
        lines.append(f"  Description : {sec.get('description')}")
        lines.append(f"  Fingerprint : {sec.get('visual_fingerprint')}")
        lines.append("  Expected products (left → right):")
        for p in sec.get("products", []):
            facing_note = f" (×{p['facings']} facings)" if p.get("facings", 1) > 1 else ""
            lines.append(
                f"    [{sid}-P{p['position']:02d}] {p['product_name']}"
                f" | {p['brand']} | {p['color_hint']}, {p['size_hint']}{facing_note}"
            )
        lines.append("")

    return "\n".join(lines)


def _get_section_products(planogram_data: dict, section_id: Optional[str]) -> list:
    """Return the flat product list for a section (or all sections if None)."""
    products = []
    for sec in planogram_data.get("gondola_sections", []):
        if section_id and sec.get("section_id") != section_id:
            continue
        for p in sec.get("products", []):
            products.append({
                "section_id": sec.get("section_id"),
                "section_label": sec.get("section_label"),
                "position_key": f"{sec.get('section_id')}-P{p['position']:02d}",
                "position": p["position"],
                "product_name": p["product_name"],
                "brand": p["brand"],
                "color_hint": p["color_hint"],
                "size_hint": p["size_hint"],
                "facings": p.get("facings", 1),
            })
    return products


# ── Step A: Section Identification ────────────────────────────────────────────

def _identify_section(client, image_part, planogram_data: dict) -> Optional[str]:
    """
    Identify which planogram section the shelf photo shows by fingerprint matching.
    Returns section_id or None if confidence < 55%.
    """
    section_lines = []
    for sec in planogram_data.get("gondola_sections", []):
        section_lines.append(
            f"  {sec['section_id']}: {sec['section_label']} "
            f"| Fingerprint: {sec['visual_fingerprint']}"
        )

    prompt = f"""
Look at this shelf photo carefully.

These are all planogram sections and their visual fingerprints:
{chr(10).join(section_lines)}

1. Describe the brands and product types you can see in this shelf photo.
2. Match your observation against the section fingerprints above.
3. Pick the single best-matching section_id.
4. Rate your confidence 0–100.

Return ONLY this JSON:
{{
  "visible_summary": "<what brands/products you see>",
  "best_match_section_id": "<section_id>",
  "confidence": <0-100>,
  "reason": "<brief explanation>"
}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["SHELF PHOTO:", image_part, prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    data = _parse_json_response(response.text)
    confidence = data.get("confidence", 0)
    section_id = data.get("best_match_section_id")

    print(f"[step-A] Matched section: {section_id}  confidence: {confidence}%")
    print(f"[step-A] Visible: {data.get('visible_summary')}")
    print(f"[step-A] Reason:  {data.get('reason')}")

    return section_id if confidence >= 55 else None


# ── Step B: Product Detection ─────────────────────────────────────────────────

def _detect_products(client, image_part, planogram_text: str) -> list:
    """
    Scan the shelf photo and return a structured list of every visible product
    with its row and left-to-right position.
    """
    prompt = f"""
You are scanning a store shelf photo to catalog every visible product.

For reference, this shelf section should contain:
{planogram_text}

TASK — Scan the shelf photo completely:
- Divide the shelf into rows from top to bottom (Row 1 = top, Row 2 = next, etc.)
- Within each row scan left to right (Position 1 = leftmost)
- Record EVERY product you can see, even if partially visible
- Use the planogram reference only as a hint for product names you cannot read clearly

Return ONLY this JSON:
{{
  "total_rows_detected": <number>,
  "rows": [
    {{
      "row": <row number>,
      "row_description": "<e.g. top shelf / second shelf>",
      "products": [
        {{
          "position": <left-to-right number starting at 1>,
          "product_name": "<best guess at product name>",
          "brand": "<brand name>",
          "color_hint": "<dominant packaging color>",
          "confidence": <0-100, how sure you are about this product>
        }}
      ]
    }}
  ]
}}

Rules:
- Do NOT skip empty slots — mark them as {{"product_name": "EMPTY", "brand": "", "color_hint": "", "confidence": 100}}
- List every product you can see, do not skip any
- confidence = how certain you are about the product identity (100 = clearly readable label)
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["SHELF PHOTO TO SCAN:", image_part, prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    data = _parse_json_response(response.text)
    rows = data.get("rows", [])
    total = sum(len(r.get("products", [])) for r in rows)
    print(f"[step-B] Detected {total} product slots across {data.get('total_rows_detected', len(rows))} rows.")
    for row in rows:
        print(f"[step-B]   Row {row['row']} ({row.get('row_description', '')}): "
              f"{len(row.get('products', []))} products")

    return rows


# ── Step C: Compliance Comparison ─────────────────────────────────────────────

def _compare(
    client,
    image_part,
    planogram_text: str,
    planogram_products: list,
    detected_rows: list,
) -> dict:
    """
    Compare detected products against the planogram reference.
    Returns the full CheckResult-compatible dict.
    """
    detected_flat = []
    for row in detected_rows:
        for p in row.get("products", []):
            detected_flat.append(
                f"  Row {row['row']} Pos {p['position']}: {p['product_name']} "
                f"[{p['brand']}] — {p['color_hint']} (confidence: {p['confidence']}%)"
            )

    planogram_flat = []
    for p in planogram_products:
        planogram_flat.append(
            f"  [{p['position_key']}] Position {p['position']}: {p['product_name']} "
            f"[{p['brand']}] — {p['color_hint']}, {p['size_hint']}"
        )

    prompt = f"""
You are producing a COMPLETE planogram compliance report.

═══ PLANOGRAM REFERENCE (what SHOULD be on the shelf) ═══
{planogram_text}

Numbered reference positions:
{chr(10).join(planogram_flat)}

═══ DETECTED PRODUCTS IN SHELF PHOTO ═══
{chr(10).join(detected_flat)}

═══ INSTRUCTIONS ═══
Go through EVERY planogram reference position one by one:

1. CORRECT — The expected product is present at the correct position
   → Add the product name to the "correct" list

2. MISSING — The expected product is NOT visible in the shelf photo at all
   → Add to "issues" with type="missing", include expected_position

3. MISPLACED — A product is visible but in the WRONG shelf position
   (e.g. product from position 3 is sitting at position 1)
   → Add to "issues" with type="misplaced", include expected_position and found_position

4. UNEXPECTED — A product is visible in the photo that does NOT appear
   anywhere in the planogram reference
   → Add to "issues" with type="unexpected", include found_position

SCORING:
  compliance_score = round((correct_count / total_planogram_positions) × 100)
  status = "pass" if compliance_score >= 80 else "fail"

Write a concise summary (1-2 sentences) describing the overall shelf condition.

Return ONLY valid JSON matching the schema exactly. Be exhaustive — do not omit any position.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["SHELF PHOTO (for final verification):", image_part, prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=CheckResult,
            temperature=0.0,
        ),
    )

    if response.parsed:
        return response.parsed.model_dump()
    return _parse_json_response(response.text)


# ── Public entry point ─────────────────────────────────────────────────────────

def check_compliance(
    pdf_path: str,
    shelf_photo_bytes: bytes,
    planogram_id: str,
    store_id: str,
    image_media_type: str = "image/jpeg",
) -> dict:
    """
    Full 3-step compliance check.

    Args:
        pdf_path: Path to planogram PDF (JSON sidecar is loaded/created automatically)
        shelf_photo_bytes: Raw bytes of the shelf photo
        planogram_id: e.g. "SNACK3C"
        store_id: e.g. "FM-CIBUBUR-001"
        image_media_type: "image/jpeg" or "image/png"

    Returns:
        dict with planogram_id, store_id, status, compliance_score,
              correct[], issues[], summary, timestamp
    """
    print(f"\n[vision_checker] ── Starting check: {planogram_id} / {store_id} ──")

    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        client = genai.Client(api_key=api_key)

        # ── Load planogram JSON (cached; extracts from PDF if missing) ──────
        print(f"[vision_checker] Loading planogram JSON...")
        planogram_data = extract_planogram(pdf_path)
        total_sections = len(planogram_data.get("gondola_sections", []))
        print(f"[vision_checker] {total_sections} sections, {planogram_data.get('total_products')} products loaded.")

        # ── Prepare image ─────────────────────────────────────────────────────
        image_part = types.Part.from_bytes(data=shelf_photo_bytes, mime_type=image_media_type)

        # ── Step A: Identify which section the photo shows ───────────────────
        print("[vision_checker] Step A — Identifying shelf section...")
        section_id = _identify_section(client, image_part, planogram_data)
        if section_id:
            print(f"[vision_checker] Locked to section {section_id}.")
        else:
            print("[vision_checker] Low confidence — using full planogram as reference.")

        planogram_text = _build_planogram_text(planogram_data, section_id)
        planogram_products = _get_section_products(planogram_data, section_id)
        print(f"[vision_checker] Reference: {len(planogram_products)} positions to check.")

        # ── Step B: Detect all products in the shelf photo ───────────────────
        print("[vision_checker] Step B — Scanning shelf photo for products...")
        detected_rows = _detect_products(client, image_part, planogram_text)

        # ── Step C: Compare and produce full report ───────────────────────────
        print("[vision_checker] Step C — Running compliance comparison...")
        result = _compare(client, image_part, planogram_text, planogram_products, detected_rows)

        result["planogram_id"] = planogram_id
        result["store_id"] = store_id
        result["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Summary log
        n_correct = len(result.get("correct", []))
        n_issues = len(result.get("issues", []))
        print(f"[vision_checker] Done — score: {result.get('compliance_score')}% | "
              f"correct: {n_correct} | issues: {n_issues} | status: {result.get('status')}")

        return result

    except Exception as e:
        print(f"[vision_checker] ERROR: {e}")
        return {
            "planogram_id": planogram_id,
            "store_id": store_id,
            "status": "error",
            "compliance_score": 0,
            "issues": [],
            "correct": [],
            "summary": f"AI Engine Error: {str(e)}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
