"""
pdf_extractor.py
----------------
Converts a planogram PDF into a list of page images (base64 encoded).
Each page of the PDF = one shelf/section reference image for Claude Vision.
"""

import fitz  # PyMuPDF
import base64
import os
from pathlib import Path


PLANOGRAM_DIR = Path("planograms")  # folder where you store the PDF files


def get_planogram_path(planogram_id: str) -> Path:
    """
    Given a planogram ID like 'SNACK2C', find the matching PDF file.
    Looks for SNACK2C.pdf inside the /planograms folder.
    """
    pdf_path = PLANOGRAM_DIR / f"{planogram_id}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Planogram PDF not found: {pdf_path}. "
            f"Make sure you put the PDF in the /planograms folder."
        )
    return pdf_path


def extract_pages_as_base64(planogram_id: str, dpi: int = 150) -> list[dict]:
    """
    Opens the planogram PDF and converts each page to a base64 PNG image.

    Returns a list of dicts:
    [
        { "page": 1, "base64": "iVBORw0KGgo..." },
        { "page": 2, "base64": "iVBORw0KGgo..." },
        ...
    ]

    dpi=150 is a good balance between quality and file size.
    Increase to 200 if images look blurry.
    """
    pdf_path = get_planogram_path(planogram_id)
    doc = fitz.open(str(pdf_path))
    pages = []

    zoom = dpi / 72  # PyMuPDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
        img_bytes = pixmap.tobytes("png")
        b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
        pages.append({
            "page": page_num + 1,
            "base64": b64,
            "width": pixmap.width,
            "height": pixmap.height,
        })

    doc.close()
    print(f"[pdf_extractor] Extracted {len(pages)} page(s) from {planogram_id}.pdf")
    return pages


def extract_first_page_as_base64(planogram_id: str, dpi: int = 150) -> str:
    """
    Convenience function — just returns the base64 string of page 1.
    Use this for planograms that are a single shelf layout per file.
    """
    pages = extract_pages_as_base64(planogram_id, dpi)
    if not pages:
        raise ValueError(f"PDF {planogram_id} has no pages.")
    return pages[0]["base64"]


# Quick test — run this file directly to test extraction
if __name__ == "__main__":
    import sys
    planogram_id = sys.argv[1] if len(sys.argv) > 1 else "SNACK2C"

    print(f"Testing extraction for: {planogram_id}")
    try:
        pages = extract_pages_as_base64(planogram_id)
        for p in pages:
            print(f"  Page {p['page']}: {p['width']}x{p['height']}px, "
                  f"base64 length={len(p['base64'])}")
        print("Extraction successful!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
