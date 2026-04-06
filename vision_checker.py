"""
vision_checker.py
-----------------
The core AI engine.
Takes two images:
  1. Reference image extracted from the planogram PDF
  2. Real shelf photo taken by the store supervisor

Sends both to Claude Vision with a structured prompt.
Returns a compliance result as a Python dict (later converted to JSON).
"""

import anthropic
import json
import base64
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()  # automatically picks up ANTHROPIC_API_KEY from .env

SYSTEM_PROMPT = """
You are a planogram compliance checker for FamilyMart Indonesia.

Your job is to compare a real shelf photo against a planogram reference image 
(uploaded by the FamilyMart planogram team) and determine whether the product 
placement is correct.

You must respond ONLY with a valid JSON object. No explanation, no markdown, 
no extra text — just the raw JSON.

The JSON must follow this exact structure:
{
  "status": "pass" or "fail",
  "compliance_score": integer from 0 to 100,
  "issues": [
    {
      "type": "missing" | "misplaced" | "unexpected",
      "product": "product name or description",
      "expected_position": "row X, slot Y or zone description",
      "found_position": "where it actually is (if misplaced)",
      "note": "optional extra detail"
    }
  ],
  "correct": [
    "Product A is correctly placed at row 1 slot 1",
    "Product B is correctly placed at row 2"
  ],
  "summary": "One sentence summary of the compliance check result"
}

Rules:
- "missing": product should be there per planogram but not visible in shelf photo
- "misplaced": product is visible but in wrong position compared to planogram
- "unexpected": product visible in shelf photo that is NOT in the planogram at all
- compliance_score: 100 = perfect match, 0 = completely wrong
- If you cannot clearly read a product label, describe it visually (e.g. "red snack bag, top-right")
- status is "pass" if compliance_score >= 80, otherwise "fail"
"""

USER_PROMPT = """
I need you to check if this store shelf matches the planogram reference.

IMAGE 1 is the PLANOGRAM REFERENCE — this is how the shelf SHOULD look.
IMAGE 2 is the ACTUAL SHELF PHOTO — this is how the shelf looks RIGHT NOW.

Compare them carefully:
- Which products are correctly placed?
- Which products are missing from the shelf?
- Which products are in the wrong position?
- Are there any products on the shelf that should NOT be there?

Return your analysis as JSON following the exact structure I specified.
"""


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode a local image file to base64 string.
    Use this when the shelf photo is uploaded as a file.
    """
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def check_compliance(
    reference_base64: str,
    shelf_photo_base64: str,
    planogram_id: str,
    store_id: str,
    image_media_type: str = "image/jpeg",
) -> dict:
    """
    Main function. Sends both images to Claude Vision and returns
    the compliance result as a Python dict.

    Args:
        reference_base64:   Base64 string of the planogram PDF page image
        shelf_photo_base64: Base64 string of the real shelf photo
        planogram_id:       e.g. "SNACK2C"
        store_id:           e.g. "FM-CIBUBUR-001"
        image_media_type:   "image/jpeg" or "image/png"

    Returns:
        dict with status, compliance_score, issues, correct, summary
    """
    print(f"[vision_checker] Checking {planogram_id} for store {store_id}...")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    # IMAGE 1: Planogram reference (from PDF)
                    {
                        "type": "text",
                        "text": "IMAGE 1 — PLANOGRAM REFERENCE (how the shelf should look):"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",  # PDF pages are always PNG
                            "data": reference_base64,
                        },
                    },
                    # IMAGE 2: Real shelf photo
                    {
                        "type": "text",
                        "text": "IMAGE 2 — ACTUAL SHELF PHOTO (how the shelf looks right now):"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": shelf_photo_base64,
                        },
                    },
                    # The actual instruction
                    {
                        "type": "text",
                        "text": USER_PROMPT
                    },
                ],
            }
        ],
    )

    raw_text = response.content[0].text.strip()
    print(f"[vision_checker] Claude responded ({len(raw_text)} chars)")

    # Parse the JSON response
    try:
        # Strip markdown code fences if Claude adds them despite instructions
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        result = json.loads(raw_text.strip())
    except json.JSONDecodeError as e:
        print(f"[vision_checker] JSON parse error: {e}")
        print(f"[vision_checker] Raw response: {raw_text[:500]}")
        # Return a graceful error result instead of crashing
        result = {
            "status": "error",
            "compliance_score": 0,
            "issues": [],
            "correct": [],
            "summary": f"AI parsing error. Raw response: {raw_text[:200]}",
        }

    # Add metadata
    result["planogram_id"] = planogram_id
    result["store_id"] = store_id
    result["timestamp"] = datetime.utcnow().isoformat()

    return result


# Quick test — run this file directly with two test images
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python vision_checker.py <reference_image> <shelf_photo>")
        print("Example: python vision_checker.py planogram_ref.png shelf_photo.jpg")
        sys.exit(1)

    ref_path = sys.argv[1]
    shelf_path = sys.argv[2]

    ref_b64 = encode_image_to_base64(ref_path)

    # Detect media type from file extension
    ext = shelf_path.split(".")[-1].lower()
    media_type = "image/png" if ext == "png" else "image/jpeg"
    shelf_b64 = encode_image_to_base64(shelf_path)

    result = check_compliance(
        reference_base64=ref_b64,
        shelf_photo_base64=shelf_b64,
        planogram_id="TEST",
        store_id="FM-TEST-001",
        image_media_type=media_type,
    )

    print("\n===== COMPLIANCE RESULT =====")
    print(json.dumps(result, indent=2, ensure_ascii=False))
