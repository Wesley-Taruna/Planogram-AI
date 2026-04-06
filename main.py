"""
main.py
-------
FastAPI server — the entry point for the Planogram AI engine.

Endpoints:
  POST /check     — submit a shelf photo + planogram ID, get compliance result
  GET  /health    — simple health check
  GET  /docs      — auto-generated API docs (FastAPI built-in)

Run with:
  uvicorn main:app --reload --port 8000

Then open: http://localhost:8000/docs to test it in the browser.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import base64
import os
from pathlib import Path
from dotenv import load_dotenv

from pdf_extractor import extract_first_page_as_base64
from vision_checker import check_compliance

load_dotenv()

app = FastAPI(
    title="Planogram AI Engine",
    description="AI-powered planogram compliance checker for FamilyMart Indonesia",
    version="1.0.0",
)

# Make sure planograms folder exists
Path("planograms").mkdir(exist_ok=True)


@app.get("/health")
def health_check():
    """Simple ping to confirm the server is running."""
    return {"status": "ok", "service": "Planogram AI Engine v1.0"}


@app.post("/check")
async def check_planogram(
    planogram_id: str = Form(..., description="Planogram file ID, e.g. SNACK2C"),
    store_id: str = Form(..., description="Store ID, e.g. FM-CIBUBUR-001"),
    shelf_photo: UploadFile = File(..., description="Photo of the actual shelf taken by supervisor"),
):
    """
    Main endpoint. Receives:
    - planogram_id: which planogram to check against (must have matching PDF in /planograms folder)
    - store_id: which store this is
    - shelf_photo: image file of the real shelf (JPG or PNG)

    Returns a JSON compliance report.
    """

    # Validate file type
    allowed_types = {"image/jpeg", "image/jpg", "image/png"}
    if shelf_photo.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {shelf_photo.content_type}. Only JPG and PNG allowed."
        )

    # Read the uploaded shelf photo
    photo_bytes = await shelf_photo.read()
    if len(photo_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    shelf_photo_b64 = base64.standard_b64encode(photo_bytes).decode("utf-8")

    # Determine media type
    media_type = "image/png" if shelf_photo.content_type == "image/png" else "image/jpeg"

    # Extract reference image from planogram PDF
    try:
        reference_b64 = extract_first_page_as_base64(planogram_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {str(e)}")

    # Run Claude Vision compliance check
    try:
        result = check_compliance(
            reference_base64=reference_b64,
            shelf_photo_base64=shelf_photo_b64,
            planogram_id=planogram_id,
            store_id=store_id,
            image_media_type=media_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI check error: {str(e)}")

    return JSONResponse(content=result)


@app.get("/planograms")
def list_planograms():
    """
    List all available planogram PDFs in the /planograms folder.
    Useful for FamilyLink to know which planogram IDs are valid.
    """
    planogram_dir = Path("planograms")
    files = [f.stem for f in planogram_dir.glob("*.pdf")]
    return {"available_planograms": sorted(files), "count": len(files)}


@app.post("/upload-planogram")
async def upload_planogram(
    pdf_file: UploadFile = File(..., description="Planogram PDF file from planogram team"),
):
    """
    Upload a new planogram PDF to the engine.
    The filename becomes the planogram_id (e.g. SNACK2C.pdf → planogram_id: SNACK2C).
    """
    if not pdf_file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    save_path = Path("planograms") / pdf_file.filename
    content = await pdf_file.read()

    with open(save_path, "wb") as f:
        f.write(content)

    planogram_id = pdf_file.filename.replace(".pdf", "")
    return {
        "message": "Planogram uploaded successfully.",
        "planogram_id": planogram_id,
        "filename": pdf_file.filename,
        "size_bytes": len(content),
    }
