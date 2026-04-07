"""
vision_checker.py
-----------------
The core AI engine.
Takes a planogram PDF and a real shelf photo.
Uploads the PDF via Gemini File API, passes image bytes directly,
and enforces a structured JSON dictionary output using the google-genai SDK.
"""

from google import genai
from google.genai import types
from datetime import datetime
from dotenv import load_dotenv
import os
import json

from models import CheckResult

load_dotenv()

SYSTEM_PROMPT = """
You are a planogram compliance checker for FamilyMart Indonesia.

Your job is to compare a real shelf photo against a planogram reference PDF image 
(uploaded by the FamilyMart planogram team) and determine whether the product 
placement is correct.

Rules:
- "missing": product should be there per planogram but not visible in shelf photo
- "misplaced": product is visible but in wrong position compared to planogram
- "unexpected": product visible in shelf photo that is NOT in the planogram at all
- compliance_score: 100 = perfect match, 0 = completely wrong
- If you cannot clearly read a product label, describe it visually (e.g. "red snack bag, top-right")
- status is "pass" if compliance_score >= 80, otherwise "fail"
"""

USER_PROMPT = """
You are a planogram compliance auditor. Your job is to compare the SHELF PHOTO 
against the PLANOGRAM PDF.

STEP 1 — ZONE IDENTIFICATION:
Divide the shelf photo into 3 horizontal zones: LEFT, CENTER, RIGHT.
Identify which planogram zone each corresponds to.

STEP 2 — MATCH CONFIDENCE:
If visual overlap with planogram is below 40%, set overall_confidence = "LOW" 
and return an empty violations list. DO NOT fabricate missing items for 
unrelated shelves.

STEP 3 — VIOLATIONS (only if confidence >= 40%):
Report at most 5 violations. Prioritize: MISPLACED > MISSING > UNEXPECTED.

Return ONLY valid JSON matching the schema. No explanation text.
"""

def check_compliance(
    pdf_path: str,
    shelf_photo_bytes: bytes,
    planogram_id: str,
    store_id: str,
    image_media_type: str = "image/jpeg",
) -> dict:
    """
    Main function. Sends PDF and image to Gemini and returns
    the compliance result as a Python dict.

    Args:
        pdf_path: Absolute or relative path to the planogram PDF file
        shelf_photo_bytes: Raw bytes of the real shelf photo
        planogram_id: e.g. "SNACK2C"
        store_id: e.g. "FM-CIBUBUR-001"
        image_media_type: "image/jpeg" or "image/png"

    Returns:
        dict with status, compliance_score, issues, correct, summary, timestamp
    """
    print(f"[vision_checker] Checking {planogram_id} for store {store_id}...")
    uploaded_file = None

    try:
        # Initialize client here to avoid crashing FastAPI at boot if GEMINI_API_KEY is missing
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")
        client = genai.Client(api_key=api_key)

        # Step 1: Upload the PDF file to Gemini
        print(f"[vision_checker] Uploading PDF {pdf_path} to Gemini...")
        uploaded_file = client.files.upload(file=pdf_path)

        # Step 2: Prepare the image part
        image_part = types.Part.from_bytes(data=shelf_photo_bytes, mime_type=image_media_type)

        # Step 3: Run the structured generation
        print("[vision_checker] Running inference with Gemini 2.5 Flash...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "IMAGE 1 — PLANOGRAM REFERENCE (how the shelf should look):",
                uploaded_file,
                "IMAGE 2 — ACTUAL SHELF PHOTO (how the shelf looks right now):",
                image_part,
                USER_PROMPT
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=CheckResult,
                temperature=0.0,
            )
        )

        print("[vision_checker] Inference complete.")
        
        # result.parsed holds the Pydantic instance if response_schema is set.
        if response.parsed:
            result_dict = response.parsed.model_dump()
        else:
            # Fallback
            result_dict = json.loads(response.text)
            
        result_dict["planogram_id"] = planogram_id
        result_dict["store_id"] = store_id
        result_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        return result_dict

    except Exception as e:
        print(f"[vision_checker] Error during generation: {e}")
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
    finally:
        # Step 4: CRITICAL cleanup
        if uploaded_file:
            print(f"[vision_checker] Deleting temporary Gemini file {uploaded_file.name}...")
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as delete_err:
                print(f"[vision_checker] Failed to delete Gemini file {uploaded_file.name}: {delete_err}")