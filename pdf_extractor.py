"""
pdf_extractor.py
----------------
Extracts structured planogram data from a PDF using Gemini.
Saves the result as a JSON sidecar file next to the PDF.

Usage (standalone test):
    python pdf_extractor.py planograms/SNACK3C.pdf

Usage (from code):
    from pdf_extractor import extract_planogram
    data = extract_planogram("planograms/SNACK3C.pdf")
"""

import os
import json
from pathlib import Path
from datetime import datetime

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── Extraction prompt ────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """
You are a planogram data extraction specialist for FamilyMart Indonesia.
Your job is to read a planogram PDF and extract its structure as precise,
machine-readable JSON. Be thorough and accurate — this data will be used
by an AI compliance checker to verify real store shelves.
"""

EXTRACTION_PROMPT = """
Analyze this planogram PDF carefully. Extract ALL product placement data.

Return a JSON object with this exact structure:

{
  "planogram_id": "<the planogram code from the document, e.g. SNACK3C>",
  "planogram_title": "<full title from the document>",
  "category": "<product category, e.g. Snacks, Beverages>",
  "gondola_sections": [
    {
      "section_id": "<letter or number, e.g. A, B, 1, 2>",
      "section_label": "<human-readable label, e.g. 'Top Shelf', 'Shelf 1 - Left'>",
      "shelf_level": "<top / upper-mid / lower-mid / bottom>",
      "description": "<brief description of what products dominate this section>",
      "visual_fingerprint": "<comma-separated list of distinctive visual features: colors, brand logos, packaging shapes that identify this section at a glance>",
      "products": [
        {
          "position": <integer, left-to-right order starting at 1>,
          "product_name": "<full product name>",
          "brand": "<brand name>",
          "variant": "<flavor, size, or variant if specified>",
          "facings": <number of product facings side-by-side, default 1 if not specified>,
          "color_hint": "<dominant packaging color(s), e.g. 'red and white bag', 'blue box'>",
          "size_hint": "<packaging size or shape if visible, e.g. 'tall slim box', 'wide flat bag'>",
          "notes": "<any special instructions or notes from the planogram, empty string if none>"
        }
      ]
    }
  ],
  "total_products": <total number of unique product positions across all sections>,
  "extracted_at": "<ISO timestamp>"
}

Important rules:
- Extract EVERY shelf level as a separate section
- Position numbers must be left-to-right, starting at 1 per shelf level
- If a product spans multiple facings, set facings > 1 but list it as ONE entry
- visual_fingerprint is critical — describe colors and brands visible, this is used to identify which shelf a photo is showing
- If the PDF has multiple pages, read ALL pages
- Do not skip any products
"""


def extract_planogram(pdf_path: str, force: bool = False) -> dict:
    """
    Extract structured planogram data from a PDF.

    Saves a JSON sidecar file at the same path as the PDF (with .json extension).
    Returns the extracted data as a dict.

    Args:
        pdf_path: Path to the planogram PDF file
        force: If True, re-extract even if a cached JSON already exists

    Returns:
        dict with the full planogram structure
    """
    pdf_path = Path(pdf_path)
    json_path = pdf_path.with_suffix(".json")

    # Return cached version if it exists and force=False
    if json_path.exists() and not force:
        print(f"[pdf_extractor] Using cached extraction: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment variables.")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    client = genai.Client(api_key=api_key)
    uploaded_file = None

    try:
        print(f"[pdf_extractor] Uploading {pdf_path.name} to Gemini...")
        uploaded_file = client.files.upload(file=str(pdf_path))

        print("[pdf_extractor] Running extraction with Gemini 2.5 Flash...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Here is the planogram PDF to extract:",
                uploaded_file,
                EXTRACTION_PROMPT,
            ],
            config=types.GenerateContentConfig(
                system_instruction=EXTRACTION_SYSTEM,
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        raw = response.text.strip()

        # Strip markdown code fences if Gemini wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)

        # Inject timestamp if Gemini left it blank
        if not data.get("extracted_at"):
            data["extracted_at"] = datetime.utcnow().isoformat() + "Z"

        # Save to JSON sidecar
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[pdf_extractor] Saved extraction to {json_path}")
        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned invalid JSON: {e}\nRaw response:\n{raw[:500]}")

    finally:
        if uploaded_file:
            print(f"[pdf_extractor] Cleaning up Gemini file {uploaded_file.name}...")
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as err:
                print(f"[pdf_extractor] Warning: could not delete Gemini file: {err}")


# ── Standalone test entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    target_pdf = sys.argv[1] if len(sys.argv) > 1 else "planograms/SNACK3C.pdf"
    force_rerun = "--force" in sys.argv

    print(f"\n{'='*60}")
    print(f"  Planogram PDF Extractor — Test Run")
    print(f"  PDF : {target_pdf}")
    print(f"  Force re-extract: {force_rerun}")
    print(f"{'='*60}\n")

    try:
        result = extract_planogram(target_pdf, force=force_rerun)

        print("\n── Extraction Summary ──────────────────────────────────")
        print(f"  Planogram ID   : {result.get('planogram_id', 'N/A')}")
        print(f"  Title          : {result.get('planogram_title', 'N/A')}")
        print(f"  Category       : {result.get('category', 'N/A')}")
        print(f"  Sections found : {len(result.get('gondola_sections', []))}")
        print(f"  Total products : {result.get('total_products', 'N/A')}")
        print(f"  Extracted at   : {result.get('extracted_at', 'N/A')}")

        print("\n── Section Breakdown ───────────────────────────────────")
        for sec in result.get("gondola_sections", []):
            products = sec.get("products", [])
            print(f"\n  [{sec.get('section_id')}] {sec.get('section_label')} ({sec.get('shelf_level')})")
            print(f"       {sec.get('description')}")
            print(f"       Fingerprint: {sec.get('visual_fingerprint')}")
            print(f"       Products ({len(products)}):")
            for p in products:
                print(f"         {p.get('position'):>2}. {p.get('product_name')} — {p.get('brand')} "
                      f"[{p.get('color_hint')}] x{p.get('facings', 1)} facing(s)")

        json_output = Path(target_pdf).with_suffix(".json")
        print(f"\n✓ Full JSON saved to: {json_output}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
