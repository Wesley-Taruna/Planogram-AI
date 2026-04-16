"""
main.py
-------
FastAPI server — the entry point for the Planogram AI engine.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import httpx
from pathlib import Path
from dotenv import load_dotenv

from vision_checker import check_compliance, check_compliance_positional, check_compliance_positional_dual_row
from pdf_extractor import extract_planogram

# ── In-memory cache for planogram JSON ─────────────────────────────────────
# Avoids re-reading the .json file from disk on every /info request.
# Key: planogram_id (str), Value: dict (full planogram JSON)
_planogram_cache: dict = {}

load_dotenv()

app = FastAPI(
    title="Planogram AI Engine",
    description="AI-powered planogram compliance checker for FamilyMart Indonesia",
    version="1.0.0",
)

# CORS is now largely redundant since FastAPI serves the frontend from the same
# origin, but we leave it open for dev-time tooling (e.g. hitting /check from a
# standalone HTML page during local testing).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Make sure planograms folder exists
Path("planograms").mkdir(exist_ok=True)

# ── Static frontend (monorepo) ────────────────────────────────────────────────
# The HTML dashboard + its assets live in ./static/. We mount them at /static
# and serve index.html at the root route so the whole app runs as ONE server.
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_frontend():
    """Serve the FamilyMart Planogram dashboard at the root URL."""
    return FileResponse(str(STATIC_DIR / "index.html"))


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


@app.post("/check-full-gondola")
async def check_full_gondola(
    background_tasks: BackgroundTasks,
    planogram_id: str = Form(..., description="Planogram ID, e.g. SNACK5C"),
    store_id: str = Form(..., description="Store ID, e.g. FM-CIBUBUR-001"),
    section_keys: list[str] = Form(..., description="Ordered list of section keys, e.g. ['1.1_1.2', '4.1_4.2']"),
    section_photos: list[UploadFile] = File(..., description="Photos matching each section_key (same order)"),
):
    """
    Full gondola compliance check — supports any subset of columns checked simultaneously.
    Photos are processed in PARALLEL so checking shelf 1 + shelf 4 takes the same time
    as checking just one shelf (~60s instead of sequential ~2min).
    """
    import json as _json
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime

    if len(section_keys) != len(section_photos):
        raise HTTPException(
            status_code=400,
            detail=f"Mismatch: {len(section_keys)} keys but {len(section_photos)} photos."
        )

    allowed_types = {"image/jpeg", "image/jpg", "image/png"}
    for photo in section_photos:
        if photo.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {photo.content_type}. Only JPG and PNG allowed."
            )

    pdf_path = Path("planograms") / f"{planogram_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"Planogram PDF {planogram_id}.pdf not found.")

    # Load planogram JSON — extract on demand if missing
    json_path = Path("planograms") / f"{planogram_id}.json"
    if not json_path.exists():
        try:
            extract_planogram(str(pdf_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not extract planogram: {e}")

    with open(json_path, "r", encoding="utf-8") as f:
        planogram_data = _json.load(f)

    sections_by_id = {
        sec["section_id"]: sec
        for sec in planogram_data.get("gondola_sections", [])
    }

    # ── Read all photo bytes concurrently (fast I/O) ──────────────────────────
    photo_bytes_list = await asyncio.gather(*[photo.read() for photo in section_photos])
    photo_types = [photo.content_type for photo in section_photos]

    # ── Build per-section task payloads ───────────────────────────────────────
    tasks = []
    for key, photo_bytes, content_type in zip(section_keys, photo_bytes_list, photo_types):
        if not photo_bytes:
            continue
        sec_ids = key.split("_")
        # Column number = first digit of first section id (e.g. "4" from "4.1")
        column = sec_ids[0].split(".")[0] if sec_ids else "?"
        section_products = []
        for sid in sec_ids:
            sec_data = sections_by_id.get(sid, {})
            section_products.extend(
                p["product_name"] for p in sec_data.get("products", [])
            )
        tasks.append({
            "key": key,
            "sec_ids": sec_ids,
            "column": column,
            "photo_bytes": photo_bytes,
            "content_type": content_type,
            "section_products": section_products,
        })

    if not tasks:
        raise HTTPException(status_code=400, detail="No valid photos received.")

    print(f"[check-full-gondola] Processing {len(tasks)} sections in PARALLEL — "
          f"columns: {sorted(set(t['column'] for t in tasks))}")

    # ── Run all Gemini calls in parallel via thread pool ──────────────────────
    # check_compliance is synchronous/blocking, so we use run_in_executor to
    # avoid blocking the async event loop and allow true parallelism.
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=len(tasks))

    def run_check(task: dict) -> dict:
        """Synchronous worker — called in thread pool."""
        key = task["key"]
        expected = task["section_products"]
        try:
            result = check_compliance(
                pdf_path=str(pdf_path),
                shelf_photo_bytes=task["photo_bytes"],
                planogram_id=planogram_id,
                store_id=store_id,
                image_media_type=task["content_type"],
            )
            found      = result.get("found_on_shelf", [])
            missing    = result.get("missing_from_shelf", [])
            unexpected = result.get("not_in_planogram", [])
            positions  = result.get("detected_with_positions", [])
            score      = result.get("compliance_score", 0)
            print(f"[check-full-gondola] ✓ Section {key} done — score: {score}%")
            return {
                "section_key":             key,
                "section_ids":             task["sec_ids"],
                "column":                  task["column"],
                "compliance_score":        score,
                "expected_products":       expected,
                "found_on_shelf":          found,
                "missing_from_shelf":      missing,
                "not_in_planogram":        unexpected,
                "detected_with_positions": positions,
                "found_count":             len(found),
                "missing_count":           len(missing),
                "unexpected_count":        len(unexpected),
                "status":                  result.get("status", "error"),
                "section_products_count":  len(expected),
            }
        except Exception as e:
            print(f"[check-full-gondola] ✗ Section {key} error: {e}")
            return {
                "section_key":             key,
                "section_ids":             task["sec_ids"],
                "column":                  task["column"],
                "compliance_score":        0,
                "expected_products":       expected,
                "found_on_shelf":          [],
                "missing_from_shelf":      expected,
                "not_in_planogram":        [],
                "detected_with_positions": [],
                "found_count":             0,
                "missing_count":           len(expected),
                "unexpected_count":        0,
                "status":                  "error",
                "error":                   str(e),
                "section_products_count":  len(expected),
            }

    # Fire all tasks at once — they all run simultaneously
    section_results = list(await asyncio.gather(
        *[loop.run_in_executor(executor, run_check, task) for task in tasks]
    ))
    executor.shutdown(wait=False)

    # ── Aggregate results ─────────────────────────────────────────────────────
    all_found      = [p for r in section_results for p in r["found_on_shelf"]]
    all_missing    = [p for r in section_results for p in r["missing_from_shelf"]]
    all_unexpected = [p for r in section_results for p in r["not_in_planogram"]]
    total_planogram_products = sum(t["section_products_count"] for t in tasks)

    overall_score = (
        round(len(all_found) / total_planogram_products * 100)
        if total_planogram_products > 0 else 0
    )
    overall_status = "pass" if overall_score >= 80 else "fail"

    # ── Group section results by column for frontend display ──────────────────
    columns_checked = sorted(set(r["column"] for r in section_results))
    column_summaries = {}
    for col in columns_checked:
        col_secs = [r for r in section_results if r["column"] == col]
        col_found = sum(r["found_count"] for r in col_secs)
        col_total = sum(r["section_products_count"] for r in col_secs)
        col_score = round(col_found / col_total * 100) if col_total > 0 else 0
        column_summaries[f"column_{col}"] = {
            "column": col,
            "compliance_score": col_score,
            "sections": len(col_secs),
            "found": col_found,
            "missing": sum(r["missing_count"] for r in col_secs),
            "unexpected": sum(r["unexpected_count"] for r in col_secs),
            "status": "pass" if col_score >= 80 else "fail",
        }

    aggregate_result = {
        "planogram_id":           planogram_id,
        "store_id":               store_id,
        "status":                 overall_status,
        "compliance_score":       overall_score,
        "columns_checked":        columns_checked,
        "column_summaries":       column_summaries,
        "found_on_shelf":         all_found,
        "missing_from_shelf":     all_missing,
        "not_in_planogram":       all_unexpected,
        "sections_checked":       len(section_results),
        "section_results":        section_results,
        "total_planogram_products": total_planogram_products,
        "summary": (
            f"Checked columns {', '.join(columns_checked)} — {overall_score}% compliance "
            f"across {len(section_results)} sections. "
            f"{len(all_found)} found, {len(all_missing)} missing, {len(all_unexpected)} unexpected."
        ),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if overall_status != "error":
        background_tasks.add_task(send_to_supabase, aggregate_result)

    return JSONResponse(content=aggregate_result)


@app.post("/check-positions")
async def check_positions(
    planogram_id: str = Form(..., description="Planogram ID, e.g. SNACK5C"),
    store_id: str = Form(..., description="Store ID"),
    section_pair_keys: list[str] = Form(..., description="Pairs like ['1.1_1.2','1.3_1.4','1.5_1.6']"),
    section_photos: list[UploadFile] = File(..., description="One photo per pair (same order)"),
):
    """
    Dual-row position-aware compliance check (Testing page).

    Each photo covers TWO shelf rows:
      photo 1 → rows 1.1 + 1.2
      photo 2 → rows 1.3 + 1.4
      photo 3 → rows 1.5 + 1.6

    Pipeline per photo:
      1. Gemini detects products and assigns each to top row (1) or bottom row (2)
      2. Sort each row left-to-right by bbox.x → image JSON with positions
      3. Compare position-by-position against the planogram JSON
    """
    import json as _json

    if len(section_pair_keys) != len(section_photos):
        raise HTTPException(
            status_code=400,
            detail=f"Mismatch: {len(section_pair_keys)} keys but {len(section_photos)} photos."
        )

    allowed_types = {"image/jpeg", "image/jpg", "image/png"}
    for photo in section_photos:
        if photo.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {photo.content_type}."
            )

    json_path = Path("planograms") / f"{planogram_id}.json"
    pdf_path  = Path("planograms") / f"{planogram_id}.pdf"

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"{planogram_id}.pdf not found.")

    if not json_path.exists():
        try:
            extract_planogram(str(pdf_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not extract planogram: {e}")

    with open(json_path, "r", encoding="utf-8") as f:
        planogram_data = _json.load(f)

    sections_by_id = {
        sec["section_id"]: sec
        for sec in planogram_data.get("gondola_sections", [])
    }

    photo_results  = []
    overall_correct     = 0
    overall_plan_count  = 0

    for pair_key, photo in zip(section_pair_keys, section_photos):
        # Parse "1.1_1.2" → ("1.1", "1.2")
        parts = pair_key.split("_")
        if len(parts) != 2:
            photo_results.append({
                "photo_key": pair_key,
                "status": "error",
                "error": f"Expected pair key like '1.1_1.2', got '{pair_key}'.",
            })
            continue

        sid1, sid2 = parts[0], parts[1]
        sec1 = sections_by_id.get(sid1)
        sec2 = sections_by_id.get(sid2)

        missing_secs = [s for s, d in [(sid1, sec1), (sid2, sec2)] if not d]
        if missing_secs:
            photo_results.append({
                "photo_key": pair_key,
                "status": "error",
                "error": f"Section(s) not found in {planogram_id}: {missing_secs}",
            })
            continue

        photo_bytes = await photo.read()
        if len(photo_bytes) == 0:
            continue

        try:
            result = check_compliance_positional_dual_row(
                section_id_1=sid1,
                section_products_1=sec1.get("products", []),
                section_id_2=sid2,
                section_products_2=sec2.get("products", []),
                shelf_photo_bytes=photo_bytes,
                planogram_id=planogram_id,
                store_id=store_id,
                image_media_type=photo.content_type,
            )
            photo_results.append(result)
            overall_correct    += result["row1"]["correct_count"] + result["row2"]["correct_count"]
            overall_plan_count += result["row1"]["planogram_count"] + result["row2"]["planogram_count"]

        except Exception as e:
            print(f"[check-positions] Error for pair {pair_key}: {e}")
            photo_results.append({
                "photo_key": pair_key,
                "status": "error",
                "compliance_score": 0,
                "error": str(e),
            })

    overall_score = round(overall_correct / overall_plan_count * 100) if overall_plan_count > 0 else 0

    return JSONResponse(content={
        "planogram_id":     planogram_id,
        "store_id":         store_id,
        "compliance_score": overall_score,
        "status":           "pass" if overall_score >= 95 else "fail",
        "photos":           photo_results,
        "timestamp":        __import__("datetime").datetime.utcnow().isoformat() + "Z",
    })


@app.get("/planograms/{planogram_id}/pdf")
def serve_planogram_pdf(planogram_id: str):
    """Serve the raw PDF file so the frontend iframe can display it."""
    pdf_path = Path("planograms") / f"{planogram_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"{planogram_id}.pdf not found.")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"{planogram_id}.pdf",
        headers={"Content-Disposition": "inline"},
    )


@app.get("/planograms/{planogram_id}/info")
def get_planogram_info(planogram_id: str):
    """
    Return the extracted JSON structure for a planogram.
    Used by the frontend to build its PLANS row/slot grid for any
    dynamically discovered planogram (not just hardcoded ones).

    Uses an in-memory cache (_planogram_cache) so the .json file is only
    read from disk once per server lifetime — every subsequent request for
    the same planogram_id is served instantly from RAM.
    """
    # 1. Fast path: already in memory
    if planogram_id in _planogram_cache:
        print(f"[info] Cache hit for {planogram_id}")
        return _planogram_cache[planogram_id]

    json_path = Path("planograms") / f"{planogram_id}.json"
    pdf_path  = Path("planograms") / f"{planogram_id}.pdf"

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"{planogram_id}.pdf not found.")

    # 2. Read cached JSON from disk (fast — no Gemini call)
    if json_path.exists():
        print(f"[info] Loading {planogram_id}.json from disk into memory cache")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _planogram_cache[planogram_id] = data
        return data

    # 3. JSON doesn't exist yet — trigger Gemini extraction (slow, one-time)
    print(f"[info] No JSON sidecar found — running Gemini extraction for {planogram_id}")
    try:
        data = extract_planogram(str(pdf_path))
        _planogram_cache[planogram_id] = data   # cache the freshly extracted data
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.get("/planograms")
def list_planograms():
    """List all available planogram PDFs in the /planograms folder."""
    planogram_dir = Path("planograms")
    files = [f.stem for f in planogram_dir.glob("*.pdf")]
    return {"available_planograms": sorted(files), "count": len(files)}


async def send_planogram_to_supabase(planogram_id: str, planogram_data: dict, size_bytes: int):
    """
    Background task — store uploaded planogram metadata in Supabase.
    Table: planogram_library  (create with supabase_schema.sql if needed)
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        print("[main] Supabase not configured — skipping planogram metadata upload.")
        return

    base_url = supabase_url.rstrip("/")
    endpoint = f"{base_url}/rest/v1/planogram_library"
    headers  = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",   # upsert on conflict
    }
    payload = {
        "planogram_id":    planogram_id,
        "planogram_title": planogram_data.get("planogram_title", planogram_id),
        "category":        planogram_data.get("category", ""),
        "total_products":  planogram_data.get("total_products", 0),
        "sections_count":  len(planogram_data.get("gondola_sections", [])),
        "extracted_at":    planogram_data.get("extracted_at", ""),
        "size_bytes":      size_bytes,
    }
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(endpoint, json=payload, headers=headers)
            res.raise_for_status()
            print(f"[main] Planogram metadata saved to Supabase: {planogram_id}")
    except Exception as e:
        print(f"[main] Failed to save planogram metadata to Supabase: {e}")


@app.post("/upload-planogram")
async def upload_planogram(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(..., description="Planogram PDF file from planogram team"),
):
    """
    Upload a new planogram PDF.
    1. Saves to /planograms/ folder.
    2. Extracts structured JSON via Gemini (cached as .json sidecar).
    3. Returns full gondola_sections so the frontend can build its slot grid.
    4. Logs planogram metadata to Supabase in background.
    """
    if not pdf_file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    save_path = Path("planograms") / pdf_file.filename
    content   = await pdf_file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    with open(save_path, "wb") as f:
        f.write(content)

    planogram_id = pdf_file.filename.replace(".pdf", "")
    print(f"[main] PDF saved: {save_path} ({len(content)} bytes)")

    # Extract structured JSON (force=True so we always re-extract on fresh upload)
    try:
        planogram_data = extract_planogram(str(save_path), force=True)
        sections       = len(planogram_data.get("gondola_sections", []))
        total_products = planogram_data.get("total_products", 0)
        extraction_status = "ok"
        print(f"[main] Extraction OK — {sections} sections, {total_products} products")
    except Exception as e:
        print(f"[main] Extraction failed for {planogram_id}: {e}")
        planogram_data    = {"gondola_sections": []}
        sections          = 0
        total_products    = 0
        extraction_status = f"failed: {str(e)}"

    # Always update the in-memory cache after upload so /info is instant immediately
    if extraction_status == "ok":
        _planogram_cache[planogram_id] = planogram_data
        print(f"[main] Updated in-memory cache for {planogram_id}")

    # Save planogram metadata to Supabase in background
    if extraction_status == "ok":
        background_tasks.add_task(
            send_planogram_to_supabase, planogram_id, planogram_data, len(content)
        )

    # Return full gondola_sections so the frontend can build the PLANS row grid
    return {
        "message":        "Planogram uploaded and extracted successfully.",
        "planogram_id":   planogram_id,
        "filename":       pdf_file.filename,
        "size_bytes":     len(content),
        "gondola_sections": planogram_data.get("gondola_sections", []),
        "extraction": {
            "status":          extraction_status,
            "sections_found":  sections,
            "total_products":  total_products,
        },
    }