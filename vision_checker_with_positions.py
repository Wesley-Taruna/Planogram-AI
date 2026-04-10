"""
vision_checker.py
-----------------
3-step AI compliance engine:

  Step A — Gondola identification:
            Photo → which gondola group is this? (e.g. "1", "2", "3")
            Returns ALL sections belonging to that gondola.

  Step B — Product detection:
            Photo → every visible product, row by row, position by position.

  Step C — Full compliance comparison:
            Each detected row is mapped to its planogram section level.
            Outputs complete correct[] + issues[] (missing/misplaced/unexpected).
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
Always be thorough and precise. Never leave correct[] empty if products are correctly placed.
"""

SHELF_LEVEL_ORDER = ["top", "upper-mid", "mid", "lower-mid", "bottom"]


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


def _get_gondola_groups(planogram_data: dict) -> dict:
    """
    Group sections by gondola prefix.
    e.g. sections 1.1, 1.2, 1.3 → gondola "1"
         sections 2.1, 2.2      → gondola "2"
    Returns: {"1": [sec1, sec2, ...], "2": [...], ...}
    """
    groups = {}
    for sec in planogram_data.get("gondola_sections", []):
        sid = str(sec.get("section_id", ""))
        # prefix is everything before the first "." or the whole string if no "."
        prefix = sid.split(".")[0] if "." in sid else sid
        groups.setdefault(prefix, []).append(sec)

    # Sort each group's sections by shelf_level order
    for prefix in groups:
        groups[prefix].sort(
            key=lambda s: SHELF_LEVEL_ORDER.index(s.get("shelf_level", "mid"))
            if s.get("shelf_level") in SHELF_LEVEL_ORDER else 99
        )
    return groups


def _build_gondola_text(sections: list) -> str:
    """
    Render a list of sections (one gondola) as a numbered reference text.
    Rows are numbered 1..N top-to-bottom matching photo row order.
    """
    lines = []
    for row_idx, sec in enumerate(sections, start=1):
        sid = sec.get("section_id")
        lines.append(
            f"ROW {row_idx} → SECTION {sid}: {sec.get('section_label')} "
            f"[{sec.get('shelf_level')}]"
        )
        lines.append(f"  Description : {sec.get('description')}")
        lines.append(f"  Fingerprint : {sec.get('visual_fingerprint')}")
        lines.append("  Expected products (left → right):")
        for p in sec.get("products", []):
            facing_note = f" (×{p['facings']} facings)" if p.get("facings", 1) > 1 else ""
            lines.append(
                f"    [{sid}-P{p['position']:02d}] "
                f"{p['product_name']} | {p['brand']} | "
                f"{p['color_hint']}, {p['size_hint']}{facing_note}"
            )
        lines.append("")
    return "\n".join(lines)


def _get_all_planogram_products(sections: list) -> list:
    """
    Flat list of all expected products across all sections, each tagged with
    its row_index (1-based, top-to-bottom) for comparison.
    """
    products = []
    for row_idx, sec in enumerate(sections, start=1):
        for p in sec.get("products", []):
            products.append({
                "row_index": row_idx,
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


# ── Step A: Gondola Identification ────────────────────────────────────────────

def _identify_gondola(client, image_part, planogram_data: dict) -> tuple:
    """
    Identify which gondola group the shelf photo shows.
    Returns (gondola_prefix, sections_list) or (None, all_sections) if low confidence.
    """
    gondola_groups = _get_gondola_groups(planogram_data)

    group_lines = []
    for prefix, sections in gondola_groups.items():
        # Summarise fingerprints for all shelf levels in this gondola
        fingerprints = " | ".join(
            f"Row {i+1}: {s.get('visual_fingerprint', '')}"
            for i, s in enumerate(sections)
        )
        group_lines.append(f"  Gondola {prefix}: {fingerprints}")

    prompt = f"""
Look at this shelf photo. It shows a full gondola (multiple shelf rows stacked vertically).

These are the available gondola groups and what each row should look like:
{chr(10).join(group_lines)}

1. Describe ALL the shelf rows you can see from top to bottom.
   For each row, note the dominant brands and packaging colors.
2. Match the overall combination against the gondola fingerprints above.
3. Pick the single best-matching gondola number.
4. Rate your confidence 0–100.

Return ONLY this JSON:
{{
  "rows_observed": [
    {{"row": 1, "description": "<brands/colors you see in top row>"}},
    {{"row": 2, "description": "<brands/colors you see in second row>"}}
  ],
  "best_match_gondola": "<gondola prefix, e.g. 1 or 2 or 3>",
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
    gondola_prefix = str(data.get("best_match_gondola", ""))

    print(f"[step-A] Matched gondola: {gondola_prefix}  confidence: {confidence}%")
    print(f"[step-A] Reason: {data.get('reason')}")
    for row in data.get("rows_observed", []):
        print(f"[step-A]   Row {row.get('row')}: {row.get('description')}")

    if confidence >= 55 and gondola_prefix in gondola_groups:
        return gondola_prefix, gondola_groups[gondola_prefix]

    # Low confidence — use ALL sections
    print("[step-A] Low confidence — using all sections as reference.")
    all_sections = []
    for secs in gondola_groups.values():
        all_sections.extend(secs)
    return None, all_sections


# ── Step B: Product Detection ─────────────────────────────────────────────────

def _detect_products(client, image_part, gondola_text: str, num_expected_rows: int) -> list:
    """
    Scan every row and position in the shelf photo.
    Returns list of row dicts: [{row, row_description, products: [...]}]
    """
    prompt = f"""
You are scanning a store shelf photo to catalog EVERY visible product.

The planogram expects {num_expected_rows} shelf rows. Scan the photo top to bottom.

PLANOGRAM REFERENCE (use only as a naming hint):
{gondola_text}

SCANNING RULES:
- Scan each shelf row from top (Row 1) to bottom (Row {num_expected_rows})
- Within each row scan left to right (Position 1 = leftmost slot)
- Record EVERY product slot you can see — even if partially visible
- For empty slots: use product_name = "EMPTY", brand = "", confidence = 100
- Use the planogram reference to help name products you cannot read clearly
- confidence = how certain you are (100 = clearly readable label, 50 = educated guess)

Return ONLY this JSON:
{{
  "total_rows_detected": <number>,
  "rows": [
    {{
      "row": <row number 1=top>,
      "row_description": "<e.g. top shelf, second shelf from top>",
      "products": [
        {{
          "position": <left-to-right number starting at 1>,
          "product_name": "<product name>",
          "brand": "<brand>",
          "color_hint": "<dominant packaging color>",
          "confidence": <0-100>
        }}
      ]
    }}
  ]
}}
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
    print(f"[step-B] Detected {total} product slots across {len(rows)} rows.")
    for r in rows:
        prods = r.get("products", [])
        names = ", ".join(p.get("product_name", "?") for p in prods)
        print(f"[step-B]   Row {r['row']} ({r.get('row_description', '')}): {names}")

    return rows


# ── Step C: Compliance Comparison ─────────────────────────────────────────────

def _compare(client, image_part, gondola_text: str, planogram_products: list, detected_rows: list) -> dict:
    """
    Full position-by-position comparison.
    Produces complete correct[], issues[] (missing/misplaced/unexpected), score, summary.
    """
    # Format detected products for the prompt
    detected_lines = []
    for row in detected_rows:
        for p in row.get("products", []):
            detected_lines.append(
                f"  Row {row['row']} Pos {p['position']}: "
                f"{p['product_name']} [{p['brand']}] — {p['color_hint']} "
                f"(confidence: {p['confidence']}%)"
            )

    # Format planogram reference
    ref_lines = []
    for p in planogram_products:
        ref_lines.append(
            f"  Row {p['row_index']} Pos {p['position']} [{p['position_key']}]: "
            f"{p['product_name']} | {p['brand']} | {p['color_hint']}"
        )

    total_positions = len(planogram_products)

    prompt = f"""
Produce a COMPLETE planogram compliance report.

═══ PLANOGRAM REFERENCE — what SHOULD be on the shelf ═══
{gondola_text}

All expected positions ({total_positions} total):
{chr(10).join(ref_lines)}

═══ DETECTED PRODUCTS — what IS on the shelf ═══
{chr(10).join(detected_lines)}

═══ YOUR TASK ═══
Go through EVERY planogram position (all {total_positions} of them) one by one:

STEP 1 — CORRECT placements:
  For each planogram position where the correct product IS present at the right row and position:
  → Add the product name string to the "correct" list.
  → Example: "Pringles Original 102g at Row 1 Pos 1 ✓"
  IMPORTANT: Do NOT leave correct[] empty if products match — fill it thoroughly.

STEP 2 — MISSING products:
  For each planogram position where the expected product is NOT visible anywhere in the photo:
  → Add to "issues": type="missing", product=<name>, expected_position=<position_key>

STEP 3 — MISPLACED products:
  For each product that IS visible but in a DIFFERENT row or position than expected:
  → Add to "issues": type="misplaced", product=<name>,
    expected_position=<planogram position_key>, found_position="Row X Pos Y"

STEP 4 — UNEXPECTED products:
  For each detected product that does NOT appear anywhere in the planogram reference:
  → Add to "issues": type="unexpected", product=<name>, found_position="Row X Pos Y"

STEP 5 — SCORE:
  compliance_score = round((len(correct) / {total_positions}) * 100)
  status = "pass" if compliance_score >= 80 else "fail"

STEP 6 — SUMMARY:
  Write 2 sentences: overall shelf condition + most critical issues.

Return ONLY valid JSON matching the schema. Be exhaustive — cover every single position.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["SHELF PHOTO (final verification):", image_part, prompt],
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
    Full 3-step compliance check against the full gondola visible in the photo.

    Returns dict with: planogram_id, store_id, status, compliance_score,
                       correct[], issues[], summary, timestamp
    """
    print(f"\n[vision_checker] ── Starting check: {planogram_id} / {store_id} ──")

    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        client = genai.Client(api_key=api_key)

        # ── Load planogram JSON ──────────────────────────────────────────────
        print("[vision_checker] Loading planogram JSON...")
        planogram_data = extract_planogram(pdf_path)
        total_sections = len(planogram_data.get("gondola_sections", []))
        print(f"[vision_checker] {total_sections} sections loaded.")

        # ── Prepare image ────────────────────────────────────────────────────
        image_part = types.Part.from_bytes(data=shelf_photo_bytes, mime_type=image_media_type)

        # ── Step A: Identify gondola group ───────────────────────────────────
        print("[vision_checker] Step A — Identifying gondola...")
        gondola_prefix, matched_sections = _identify_gondola(client, image_part, planogram_data)
        print(f"[vision_checker] Using {len(matched_sections)} sections as reference.")

        gondola_text = _build_gondola_text(matched_sections)
        planogram_products = _get_all_planogram_products(matched_sections)
        print(f"[vision_checker] {len(planogram_products)} total positions to check.")

        # ── Step B: Detect all products in photo ─────────────────────────────
        print("[vision_checker] Step B — Scanning shelf photo...")
        detected_rows = _detect_products(
            client, image_part, gondola_text, num_expected_rows=len(matched_sections)
        )

        # ── Step C: Full comparison ──────────────────────────────────────────
        print("[vision_checker] Step C — Running compliance comparison...")
        result = _compare(client, image_part, gondola_text, planogram_products, detected_rows)

        result["planogram_id"] = planogram_id
        result["store_id"] = store_id
        result["timestamp"] = datetime.utcnow().isoformat() + "Z"

        n_correct = len(result.get("correct", []))
        n_issues = len(result.get("issues", []))
        print(
            f"[vision_checker] Done — score: {result.get('compliance_score')}% | "
            f"correct: {n_correct} | issues: {n_issues} | status: {result.get('status')}"
        )
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
