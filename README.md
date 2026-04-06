# Planogram AI Engine
AI-powered planogram compliance checker for FamilyMart Indonesia.  
Built with Python, FastAPI, and Claude Vision API.

---

## What this does
Staff takes a photo of a shelf → AI compares it against the planogram PDF reference → Returns what's correct, missing, or misplaced.

---

## Setup (do this once)

### 1. Clone / open this folder in VS Code

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your API key
Copy `.env.example` to `.env` and fill in your Anthropic API key:
```bash
cp .env.example .env
```
Then edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxx
```

### 5. Add planogram PDFs
Put the planogram PDF files inside the `/planograms` folder.  
Example: `planograms/SNACK2C.pdf`, `planograms/BISCUIT.pdf`

---

## Running the server

```bash
uvicorn main:app --reload --port 8000
```

Open your browser at: **http://localhost:8000/docs**  
This gives you an interactive UI to test all endpoints.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check server is running |
| GET | `/planograms` | List all available planogram IDs |
| POST | `/upload-planogram` | Upload a new planogram PDF |
| POST | `/check` | Submit shelf photo → get compliance result |

---

## Testing the check endpoint

Using the `/docs` UI:
1. Open http://localhost:8000/docs
2. Click `POST /check` → Try it out
3. Fill in `planogram_id` (e.g. `SNACK2C`) and `store_id` (e.g. `FM-CIBUBUR-001`)
4. Upload a shelf photo JPG
5. Click Execute

Using curl:
```bash
curl -X POST http://localhost:8000/check \
  -F "planogram_id=SNACK2C" \
  -F "store_id=FM-CIBUBUR-001" \
  -F "shelf_photo=@/path/to/shelf_photo.jpg"
```

---

## Example response
```json
{
  "planogram_id": "SNACK2C",
  "store_id": "FM-CIBUBUR-001",
  "status": "fail",
  "compliance_score": 72,
  "issues": [
    {
      "type": "missing",
      "product": "Chitato 68g",
      "expected_position": "row 2, slot 3",
      "found_position": null,
      "note": null
    },
    {
      "type": "misplaced",
      "product": "Piattos Cheese",
      "expected_position": "row 1, slot 2",
      "found_position": "row 3, slot 1",
      "note": null
    }
  ],
  "correct": [
    "Lays Original 68g correctly placed at row 1 slot 1",
    "Doritos Nacho correctly placed at row 2 slot 1"
  ],
  "summary": "Shelf is 72% compliant. 2 issues found: 1 missing product, 1 misplaced product.",
  "timestamp": "2026-04-02T08:30:00"
}
```

---

## Project structure

```
planogram-ai/
├── main.py              ← FastAPI server (start here)
├── pdf_extractor.py     ← Converts planogram PDF → image
├── vision_checker.py    ← Claude Vision comparison logic
├── models.py            ← Data shapes
├── requirements.txt     ← Python dependencies
├── .env.example         ← Copy to .env and add API key
├── planograms/          ← Put planogram PDFs here
└── test_images/         ← Put test shelf photos here
```

---

## Testing individual modules

Test PDF extraction only:
```bash
python pdf_extractor.py SNACK2C
```

Test vision checker only (needs 2 image files):
```bash
python vision_checker.py test_images/reference.png test_images/shelf.jpg
```
