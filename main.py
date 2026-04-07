"""
main.py
-------
FastAPI server — the entry point for the Planogram AI engine.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import httpx
from pathlib import Path
from dotenv import load_dotenv

from vision_checker import check_compliance

load_dotenv()

app = FastAPI(
    title="Planogram AI Engine",
    description="AI-powered planogram compliance checker for FamilyMart Indonesia",
    version="1.0.0",
)

# Make sure planograms folder exists
Path("planograms").mkdir(exist_ok=True)


async def send_to_supabase(result: dict):
    """Background task to save the check result to Supabase."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase_table = os.getenv("SUPABASE_TABLE", "compliance_logs")
    
    if not supabase_url or not supabase_key:
        print("[main] Warning: SUPABASE_URL or SUPABASE_KEY not found. Skipping DB logging.")
        return

    # Trim trailing slash if present
    base_url = supabase_url.rstrip('/')
    endpoint = f"{base_url}/rest/v1/{supabase_table}"
    
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(endpoint, json=result, headers=headers)
            res.raise_for_status()
            print(f"[main] Successfully logged result to Supabase ({supabase_table}) for store {result.get('store_id')}")
    except Exception as e:
        print(f"[main] Failed to log result to Supabase: {e}")


@app.get("/health")
def health_check():
    """Simple ping to confirm the server is running."""
    return {"status": "ok", "service": "Planogram AI Engine v1.0"}


@app.post("/check")
async def check_planogram(
    background_tasks: BackgroundTasks,
    planogram_id: str = Form(..., description="Planogram file ID, e.g. SNACK2C"),
    store_id: str = Form(..., description="Store ID, e.g. FM-CIBUBUR-001"),
    shelf_photo: UploadFile = File(..., description="Photo of the actual shelf taken by supervisor"),
):
    """
    Main endpoint. Receives:
    - planogram_id: which planogram to check against
    - store_id: which store this is
    - shelf_photo: image file of the real shelf (JPG or PNG)
    """
    # Validate file type
    allowed_types = {"image/jpeg", "image/jpg", "image/png"}
    if shelf_photo.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {shelf_photo.content_type}. Only JPG and PNG allowed."
        )

    # Read the uploaded shelf photo bytes
    photo_bytes = await shelf_photo.read()
    if len(photo_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Locate the PDF file and pass its path
    pdf_path = Path("planograms") / f"{planogram_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"Planogram PDF {planogram_id}.pdf not found.")

    # Run Gemini Vision compliance check
    try:
        result = check_compliance(
            pdf_path=str(pdf_path),
            shelf_photo_bytes=photo_bytes,
            planogram_id=planogram_id,
            store_id=store_id,
            image_media_type=shelf_photo.content_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI check error: {str(e)}")

    # Schedule background task to upload result to DB only if it's not an internal handled error
    if result.get("status") != "error":
        background_tasks.add_task(send_to_supabase, result)
    else:
        print(f"[main] Compliance check returned error status, skipping DB upload: {result.get('summary')}")

    return JSONResponse(content=result)


@app.get("/planograms")
def list_planograms():
    """List all available planogram PDFs in the /planograms folder."""
    planogram_dir = Path("planograms")
    files = [f.stem for f in planogram_dir.glob("*.pdf")]
    return {"available_planograms": sorted(files), "count": len(files)}


@app.post("/upload-planogram")
async def upload_planogram(
    pdf_file: UploadFile = File(..., description="Planogram PDF file from planogram team"),
):
    """Upload a new planogram PDF to the engine."""
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