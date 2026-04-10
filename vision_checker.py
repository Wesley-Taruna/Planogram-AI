"""
vision_checker.py
-----------------
Three clearly separated responsibilities:

  [AI]     Step 1 — Gemini scans the shelf photo BLINDLY.
           No planogram hints. Reports only what it actually sees.

  [AI]     Step 2 — For ambiguous/low-confidence detections, Gemini
           uses Google Search to confirm the exact product variant
           (flavor, size, color) based on packaging visuals.

  [Code]   Step 3 — Python compares the verified detection list
           against the planogram JSON.

Why web search verification?
  Snack products in Indonesia often share very similar packaging.
  Example: All QTELA variants have similar orange/yellow colors.
  By searching "QTELA snack Indonesia varian" Gemini can distinguish
  between Balado, Original, Barbeque by packaging color + design.

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
import os
import json

from pdf_extractor import extract_planogram


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class DetectedProduct(BaseModel):
    product_name:    str
    brand:           str
    color_hint:      str
    confidence:      int            # 0-100
    needs_web_check: bool           # True if ambiguous / low confidence
    ambiguity_note:  Optional[str]  # e.g. "unsure if Balado or BBQ flavor"

class DetectionResult(BaseModel):
    detected: List[DetectedProduct]

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


load_dotenv()

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
    Now also flags products that need web verification.
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
        detected = [p.model_dump() for p in response.parsed.detected]
    else:
        data     = _parse_json(response.text)
        detected = data.get("detected", [])

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

    Why needed:
      Gemini's google_search tool is INCOMPATIBLE with response_mime_type +
      response_schema. We must parse the text output ourselves.

    Strategy:
      1. Extract JSON block from text (handles markdown fences, extra prose)
      2. Partial recovery if JSON is truncated
      3. Passthrough fallback — never crashes the pipeline
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

    Uses Gemini's built-in Google Search grounding tool.
    Returns a list of verified products with corrected names.
    """
    if not ambiguous_products:
        print("\n[step-2] No ambiguous products — skipping web verification.")
        return []

    print(f"\n[step-2] Web verification for {len(ambiguous_products)} ambiguous product(s)...")

    # Build the verification prompt with all ambiguous products
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

    # ── IMPORTANT: google_search tool is incompatible with response_mime_type
    # ── and response_schema. Use free-form text, then parse manually.
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_VERIFICATION,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.0,
            # NO response_mime_type here — incompatible with tool use
            # NO response_schema here — incompatible with tool use
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

    # Convert back to DetectedProduct format for step 3
    result = []
    for v in verified:
        result.append({
            "product_name": v["verified_name"],
            "brand":        v["verified_brand"],
            "color_hint":   "",
            "confidence":   v["confidence"],
            "needs_web_check": False,
            "ambiguity_note": None,
        })

    return result


# ── Step 3 (Code): Compare detected vs planogram ─────────────────────────────

def _compare(detected: list, planogram_products: list) -> dict:
    """
    Pure Python comparison — no AI, no hallucination.
    Combines clear detections + web-verified detections.
    """
    found_on_shelf     = []
    missing_from_shelf = []
    not_in_planogram   = []
    matched_indices    = set()

    print(f"\n[step-3] Matching {len(detected)} detected → {len(planogram_products)} planogram products")
    print(f"         Match threshold: {MATCH_THRESHOLD}\n")

    for plan_p in planogram_products:
        best_score = 0.0
        best_idx   = -1
        best_name  = ""

        for i, det_p in enumerate(detected):
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
            match_note = f" ← matched '{best_name}'" if best_name.lower() != plan_p["product_name"].lower() else ""
            print(f"  ✅ FOUND   : {plan_p['product_name']}{match_note} (score: {best_score:.2f})")
        else:
            missing_from_shelf.append(plan_p["product_name"])
            print(f"  ❌ MISSING : {plan_p['product_name']}"
                  + (f" (closest: '{best_name}' score: {best_score:.2f})" if best_name else ""))

    for i, det_p in enumerate(detected):
        if i not in matched_indices:
            not_in_planogram.append(det_p["product_name"])
            print(f"  ⚠️  EXTRA  : {det_p['product_name']} [{det_p['brand']}] — not in planogram")

    total            = len(planogram_products)
    n_found          = len(found_on_shelf)
    compliance_score = round((n_found / total * 100)) if total > 0 else 0
    status           = "pass" if compliance_score >= 80 else "fail"

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

        # ── Step 1: Blind detection ──────────────────────────────────────────
        print("\n[vision_checker] Step 1 — Blind shelf scan (no hints)...")
        clear_detections, ambiguous = _image_to_json(client, image_part)

        # ── Step 2: Web search verification for ambiguous products ───────────
        print("\n[vision_checker] Step 2 — Web verification for ambiguous products...")
        verified_detections = _verify_with_web(client, ambiguous)

        # Merge clear + verified into one final detection list
        all_detections = clear_detections + verified_detections
        print(f"\n[vision_checker] Final detection list: "
              f"{len(clear_detections)} clear + {len(verified_detections)} verified "
              f"= {len(all_detections)} total")

        # ── Step 3: Python comparison ────────────────────────────────────────
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

        return {
            "planogram_id":       planogram_id,
            "store_id":           store_id,
            "status":             result["status"],
            "compliance_score":   result["compliance_score"],
            "found_on_shelf":     result["found_on_shelf"],
            "missing_from_shelf": result["missing_from_shelf"],
            "not_in_planogram":   result["not_in_planogram"],
            "summary":            result["summary"],
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