from pydantic import BaseModel
from typing import List, Optional


class CheckRequest(BaseModel):
    planogram_id: str       # e.g. "SNACK2C"
    store_id: str           # e.g. "FM-CIBUBUR-001"


class PlacementIssue(BaseModel):
    type: str               # "missing" | "misplaced" | "unexpected"
    product: str
    expected_position: Optional[str] = None
    found_position: Optional[str] = None
    note: Optional[str] = None


class CheckResult(BaseModel):
    planogram_id: str
    store_id: str
    status: str             # "pass" | "fail" | "error"
    compliance_score: int   # 0-100
    issues: List[PlacementIssue]
    correct: List[str]
    summary: str
    timestamp: str
