from pydantic import BaseModel
from typing import List


class CheckRequest(BaseModel):
    planogram_id: str       # e.g. "SNACK3C"
    store_id: str           # e.g. "FM-CIBUBUR-001"


class CheckResult(BaseModel):
    """
    Compliance check result.

    found_on_shelf      : products that ARE in the planogram AND visible in the shelf photo
    missing_from_shelf  : products that ARE in the planogram but NOT visible in the shelf photo
    not_in_planogram    : products visible in the shelf photo that are NOT in the planogram
    """
    planogram_id:       str
    store_id:           str
    status:             str     # "pass" | "fail" | "error"
    compliance_score:   int     # 0-100

    found_on_shelf:     List[str]   # ✅ in planogram + visible in photo
    missing_from_shelf: List[str]   # ❌ in planogram + NOT visible in photo
    not_in_planogram:   List[str]   # ⚠️  visible in photo + NOT in planogram

    summary:    str
    timestamp:  str
