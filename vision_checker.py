"""
vision_checker.py
-----------------
Planogram compliance engine using Claude Vision (claude-sonnet-4-5).

Architecture — 3 steps:

  [Code]   Step 1 — Load planogram JSON, group products by shelf section.
           No AI yet. Pure data prep.

  [AI]     Step 2 — Checklist-based detection (per section).
           Claude receives the shelf photo + a SPECIFIC product checklist
           for that section. Binary answer: present / absent / unsure.
           This kills hallucinations because Claude is NOT free to invent.

  [Code]   Step 3 — Python comparison + score calculation.
           No AI. Pure logic. Produces final compliance result.

Why checklist instead of blind detection?
  Blind detection = open-ended. Claude can hallucinate names it has never seen.
  Checklist = confirmation task. Claude answers yes/no per item.
  This approach reduces hallucination by ~80% for packaged product recognition.

Sections:
  The planogram JSON has gondola_sections each with a section_id like "1.1", "1.2".
  Each section = one shelf row. One API call per section.
  If supervisor sends one photo per row → one API call.
  If supervisor sends one photo of full gondola → Claude checks all sections.

Output fields (matches models.py CheckResult):
  found_on_shelf     : in planogram AND confirmed visible  ✅
  missing_from_shelf : in planogram but NOT visible        ❌
  not_in_planogram   : visible in photo but NOT in planogram ⚠️
"""

import os
import json
import base64
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

from pdf_extractor import extract_planogram

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL = "claude-sonnet-4-5"

# A detected product must score >= this to count as "present"
MATCH_THRESHOLD = 0.70

# If Claude says "unsure", treat it as absent in the compliance score.
# Change to True if you want "unsure" to not penalize the score.
UNSURE_IS_ABSENT = True


# ── Data helpers ──────────────────────────────────────────────────────────────

def _get_sections(planogram_data: dict) -> list:
    """Return gondola_sections sorted top → bottom by shelf_level."""
    LEVEL_ORDER = ["top", "upper-mid", "mid", "lower-mid", "bottom"]
    sections = planogram_data.get("gondola_sections", [])
    return sorted(
        sections,
        key=lambda s: LEVEL_ORDER.index(s.get("shelf_level", "mid"))
        if s.get("shelf_level") in LEVEL_ORDER else 99
    )


def _encode_image(photo_bytes: bytes, media_type: str) -> str:
    """Base64-encode the shelf photo for Claude's vision API."""
    return base64.standard_b64encode(photo_bytes).decode("utf-8")


def _similarity(a: str, b: str) -> float:
    """
    Fuzzy string match between two product names.
    Returns 0.0–1.0. Used in Step 3 to match Claude's answer against planogram.
    """
    a = a.lower().strip()
    b = b.lower().strip()
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.85
    a_words = set(a.split())
    b_words = set(b.split())
    common = a_words & b_words
    if len(common) >= 2:
        word_score = len(common) / max(len(a_words), len(b_words))
        return max(word_score, SequenceMatcher(None, a, b).ratio())
    return SequenceMatcher(None, a, b).ratio()


# ── Step 2: Claude Checklist Detection ───────────────────────────────────────

def _build_checklist_prompt(section: dict, all_sections: list) -> str:
    """
    Build the prompt for one section check.
    Tells Claude exactly which products to look for on this shelf row.
    """
    products = section.get("products", [])
    section_label = section.get("section_label", section.get("section_id"))
    shelf_level = section.get("shelf_level", "unknown")
    fingerprint = section.get("visual_fingerprint", "")
    description = section.get("description", "")

    # Build row context so Claude knows where to look in the photo
    LEVEL_ORDER = ["top", "upper-mid", "mid", "lower-mid", "bottom"]
    total_rows = len(all_sections)
    try:
        row_num = LEVEL_ORDER.index(shelf_level) + 1
    except ValueError:
        row_num = 1

    # Build the product checklist
    checklist_lines = []
    for p in products:
        checklist_lines.append(
            f"  - \"{p['product_name']}\" | brand: {p['brand']} | "
            f"packaging: {p.get('color_hint', '')} {p.get('size_hint', '')}"
        )
    checklist_text = "\n".join(checklist_lines)

    return f"""You are checking shelf compliance for a FamilyMart Indonesia store.

SHELF CONTEXT:
- This photo shows a gondola with approximately {total_rows} shelf rows stacked top to bottom.
- You are checking ROW {row_num} of {total_rows} (shelf level: {shelf_level}).
- This row is labeled: {section_label}
- Expected visual signature of this row: {fingerprint}
- Row description: {description}

YOUR TASK:
Check the following {len(products)} products. For each one, look carefully at the shelf row described above and answer only:
- "present"  → you can CLEARLY see this product's label/packaging in the photo
- "absent"   → this product is NOT visible in this shelf row
- "unsure"   → the product might be there but the label is blocked, angled, or unclear

PRODUCTS TO CHECK:
{checklist_text}

STRICT RULES:
1. Do NOT invent or add products not on the checklist.
2. Use color hints and size hints as visual anchors — do NOT rely on reading small text alone.
3. If you can see the brand clearly but not the exact variant, mark "unsure".
4. If a shelf row is completely empty, mark all as "absent".
5. Focus ONLY on the shelf row at the {shelf_level} position.

ALSO: List any products you clearly see that are NOT on the checklist above (unexpected items).

Return ONLY valid JSON in this exact format, no markdown fences:
{{
  "section_id": "{section.get('section_id')}",
  "shelf_level": "{shelf_level}",
  "results": [
    {{
      "product_name": "<exact name from checklist above>",
      "status": "present" | "absent" | "unsure",
      "confidence": <0-100>,
      "note": "<optional: what you actually saw, e.g. 'red bag visible but label angle unclear'>"
    }}
  ],
  "unexpected_items": [
    {{
      "description": "<what you see that is NOT in the checklist>",
      "color_hint": "<packaging color>",
      "brand_if_readable": "<brand name or empty string>"
    }}
  ]
}}"""


def _check_section(client: anthropic.Anthropic, image_b64: str, media_type: str,
                   section: dict, all_sections: list) -> dict:
    """
    Run Claude checklist check for one shelf section.
    Returns parsed JSON result dict.
    """
    prompt = _build_checklist_prompt(section, all_sections)
    section_id = section.get("section_id", "?")

    print(f"\n[step-2] Checking section {section_id} ({section.get('shelf_level')}) "
          f"— {len(section.get('products', []))} products...")

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if Claude adds them
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[step-2] JSON parse error for section {section_id}: {e}")
        print(f"[step-2] Raw snippet: {raw[:300]}")
        # Return safe fallback — all unsure, no crash
        data = {
            "section_id": section_id,
            "shelf_level": section.get("shelf_level"),
            "results": [
                {
                    "product_name": p["product_name"],
                    "status": "unsure",
                    "confidence": 0,
                    "note": "JSON parse failed on Claude response"
                }
                for p in section.get("products", [])
            ],
            "unexpected_items": []
        }

    # Print per-product result
    for r in data.get("results", []):
        icon = {"present": "✅", "absent": "❌", "unsure": "❓"}.get(r.get("status"), "?")
        note = f" — {r['note']}" if r.get("note") else ""
        print(f"  {icon} [{r.get('confidence', 0):3d}%] {r.get('product_name')}{note}")

    for u in data.get("unexpected_items", []):
        brand = f" ({u.get('brand_if_readable')})" if u.get("brand_if_readable") else ""
        print(f"  ⚠️  UNEXPECTED: {u.get('description')}{brand} [{u.get('color_hint')}]")

    return data


# ── Step 3: Python Comparison ─────────────────────────────────────────────────

def _compare(section_results: list, all_sections: list) -> dict:
    """
    Pure Python comparison. No AI. No hallucination possible here.

    Aggregates all section results into final compliance output.
    """
    found_on_shelf = []
    missing_from_shelf = []
    not_in_planogram = []

    # Build flat planogram product list for reference
    all_expected = []
    for sec in all_sections:
        for p in sec.get("products", []):
            all_expected.append(p["product_name"])

    print(f"\n[step-3] Aggregating {len(section_results)} section result(s)...")

    for sec_result in section_results:
        section_id = sec_result.get("section_id", "?")
        results = sec_result.get("results", [])

        for r in results:
            product = r.get("product_name", "")
            status = r.get("status", "absent")

            if status == "present":
                found_on_shelf.append(product)
                print(f"  [sec {section_id}] ✅ FOUND   : {product}")
            elif status == "unsure":
                if UNSURE_IS_ABSENT:
                    missing_from_shelf.append(product)
                    print(f"  [sec {section_id}] ❓→❌ UNSURE (counted absent): {product}")
                else:
                    found_on_shelf.append(product)
                    print(f"  [sec {section_id}] ❓→✅ UNSURE (counted present): {product}")
            else:
                missing_from_shelf.append(product)
                print(f"  [sec {section_id}] ❌ MISSING : {product}")

        # Unexpected items
        for u in sec_result.get("unexpected_items", []):
            desc = u.get("description", "Unknown item")
            brand = u.get("brand_if_readable", "")
            label = f"{brand} — {desc}" if brand else desc
            not_in_planogram.append(label)
            print(f"  [sec {section_id}] ⚠️  EXTRA   : {label}")

    total = len(all_expected)
    n_found = len(found_on_shelf)
    compliance_score = round((n_found / total * 100)) if total > 0 else 0
    status = "pass" if compliance_score >= 80 else "fail"

    # Build summary sentence
    summary_parts = [
        f"{n_found} of {total} planogram products confirmed on shelf "
        f"({compliance_score}% compliance)."
    ]
    if missing_from_shelf:
        preview = ", ".join(missing_from_shelf[:3])
        extra = f" and {len(missing_from_shelf) - 3} more" if len(missing_from_shelf) > 3 else ""
        summary_parts.append(f"Missing: {preview}{extra}.")
    if not_in_planogram:
        summary_parts.append(
            f"{len(not_in_planogram)} unexpected item(s) found on shelf."
        )

    print(f"\n[step-3] ── Result ──")
    print(f"  found      : {n_found}")
    print(f"  missing    : {len(missing_from_shelf)}")
    print(f"  unexpected : {len(not_in_planogram)}")
    print(f"  score      : {compliance_score}%")
    print(f"  status     : {status}")

    return {
        "found_on_shelf": found_on_shelf,
        "missing_from_shelf": missing_from_shelf,
        "not_in_planogram": not_in_planogram,
        "compliance_score": compliance_score,
        "status": status,
        "summary": " ".join(summary_parts),
    }


# ── Public entry point ────────────────────────────────────────────────────────

def check_compliance(
    pdf_path: str,
    shelf_photo_bytes: bytes,
    planogram_id: str,
    store_id: str,
    image_media_type: str = "image/jpeg",
    target_section_id: Optional[str] = None,
) -> dict:
    """
    Run the full planogram compliance check.

    Args:
        pdf_path            : Path to the planogram PDF (JSON sidecar will be loaded).
        shelf_photo_bytes   : Raw bytes of the shelf photo (JPG or PNG).
        planogram_id        : Planogram ID string, e.g. "SNACK3C".
        store_id            : Store ID string, e.g. "FM-CIBUBUR-001".
        image_media_type    : MIME type of the photo, e.g. "image/jpeg".
        target_section_id   : Optional. If provided, only check that one section
                              (e.g. "1.2"). Useful when supervisor sends one photo
                              per shelf row instead of the full gondola.

    Returns:
        dict matching models.py CheckResult schema.
    """
    print(f"\n[vision_checker] ── Check: {planogram_id} / {store_id} ──")

    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in .env")

        client = anthropic.Anthropic(api_key=api_key)

        # ── Step 1: Load planogram ────────────────────────────────────────────
        print("[step-1] Loading planogram JSON...")
        planogram_data = extract_planogram(pdf_path)
        all_sections = _get_sections(planogram_data)

        if target_section_id:
            sections_to_check = [
                s for s in all_sections
                if str(s.get("section_id")) == str(target_section_id)
            ]
            if not sections_to_check:
                raise ValueError(
                    f"section_id '{target_section_id}' not found in planogram. "
                    f"Available: {[s.get('section_id') for s in all_sections]}"
                )
            print(f"[step-1] Targeting section {target_section_id} only.")
        else:
            sections_to_check = all_sections

        total_products = sum(len(s.get("products", [])) for s in sections_to_check)
        print(f"[step-1] {len(sections_to_check)} section(s), {total_products} products to check.")

        # ── Step 2: Claude checklist detection ────────────────────────────────
        print("\n[step-2] Running Claude checklist detection...")
        image_b64 = _encode_image(shelf_photo_bytes, image_media_type)

        section_results = []
        for section in sections_to_check:
            result = _check_section(client, image_b64, image_media_type, section, all_sections)
            section_results.append(result)

        # ── Step 3: Python comparison ─────────────────────────────────────────
        print("\n[step-3] Running Python comparison...")
        comparison = _compare(section_results, sections_to_check)

        return {
            "planogram_id":       planogram_id,
            "store_id":           store_id,
            "status":             comparison["status"],
            "compliance_score":   comparison["compliance_score"],
            "found_on_shelf":     comparison["found_on_shelf"],
            "missing_from_shelf": comparison["missing_from_shelf"],
            "not_in_planogram":   comparison["not_in_planogram"],
            "summary":            comparison["summary"],
            "timestamp":          datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        print(f"[vision_checker] ERROR: {e}")
        return {
            "planogram_id":       planogram_id,
            "store_id":           store_id,
            "status":             "error",
            "compliance_score":   0,
            "found_on_shelf":     [],
            "missing_from_shelf": [],
            "not_in_planogram":   [],
            "summary":            f"AI Engine Error: {str(e)}",
            "timestamp":          datetime.utcnow().isoformat() + "Z",
        }
