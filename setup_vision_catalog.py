"""
setup_vision_catalog.py
-----------------------
One-off utility that bootstraps a Google Cloud Vision Product Search catalog
for the FamilyMart Planogram AI Engine.

WHAT IT DOES
============
1. Reads a local folder of reference product images (clear, front-facing
   photos of snacks — one product per image, filename = SKU / product name).
2. Uploads each image to a Google Cloud Storage bucket.
3. For each image, creates (or re-uses):
     - a Product in Vision Product Search
     - a ReferenceImage attached to that Product pointing at the GCS URI
     - membership of that Product in a ProductSet (e.g. "familymart-snacks")

After you run this once, vision_checker.py can issue product_search() calls
against the ProductSet and identify snacks on real shelf photos.

REQUIRED ENVIRONMENT VARIABLES
==============================
  GOOGLE_APPLICATION_CREDENTIALS   Absolute path to a GCP service-account JSON
                                   key with these roles:
                                     - Cloud Vision AI Service Agent
                                     - Storage Object Admin (on the bucket)

  GCP_PROJECT_ID                   Your Google Cloud project ID
                                   (e.g. "familymart-planogram-prod")

  GCP_LOCATION                     Vision Product Search region.
                                   Valid: us-west1, us-east1, europe-west1,
                                          asia-east1. Pick the one nearest
                                          your FamilyMart stores — for
                                          Indonesia, "asia-east1" is closest.

  GCS_BUCKET                       Name of an existing Cloud Storage bucket
                                   for the reference images
                                   (e.g. "familymart-product-refs").

  PRODUCT_SET_ID                   The ProductSet to create / append to.
                                   Default: "familymart-snacks"

  PRODUCT_CATEGORY                 Vision product category. Must be one of:
                                     packagedgoods-v1  ← recommended for snacks
                                     apparel-v2, toys-v2, homegoods-v2, general-v1
                                   Default: "packagedgoods-v1"

  REFERENCE_IMAGES_DIR             Local folder containing reference images.
                                   Filenames become product display names
                                   and product IDs.
                                   Default: "./reference_images"

USAGE
=====
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
  export GCP_PROJECT_ID=familymart-planogram-prod
  export GCP_LOCATION=asia-east1
  export GCS_BUCKET=familymart-product-refs
  python setup_vision_catalog.py

  # or override a single knob:
  PRODUCT_SET_ID=familymart-beverages python setup_vision_catalog.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud import storage, vision

load_dotenv()

# ── Config (env vars) ─────────────────────────────────────────────────────────
PROJECT_ID        = os.getenv("GCP_PROJECT_ID")
LOCATION          = os.getenv("GCP_LOCATION", "asia-east1")
GCS_BUCKET        = os.getenv("GCS_BUCKET")
PRODUCT_SET_ID    = os.getenv("PRODUCT_SET_ID", "familymart-snacks")
PRODUCT_CATEGORY  = os.getenv("PRODUCT_CATEGORY", "packagedgoods-v1")
REF_IMAGES_DIR    = Path(os.getenv("REFERENCE_IMAGES_DIR", "./reference_images"))

VALID_IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".webp"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_env() -> None:
    """Fail fast with a clear message if the user forgot something."""
    missing = []
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        missing.append("GOOGLE_APPLICATION_CREDENTIALS")
    if not PROJECT_ID:
        missing.append("GCP_PROJECT_ID")
    if not GCS_BUCKET:
        missing.append("GCS_BUCKET")
    if missing:
        print("[setup] ERROR: missing required env vars: " + ", ".join(missing))
        print("[setup] See the docstring at the top of this file for details.")
        sys.exit(1)

    if not REF_IMAGES_DIR.is_dir():
        print(f"[setup] ERROR: reference images folder not found: {REF_IMAGES_DIR.resolve()}")
        print("[setup] Create it and drop clear, front-facing product photos inside.")
        sys.exit(1)


def _sanitize_product_id(stem: str) -> str:
    """Vision product IDs must match ^[a-zA-Z0-9_-]+$ (<=128 chars)."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", stem).strip("-").lower()
    return cleaned[:120] or "unnamed-product"


def _iter_reference_images(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            yield p


# ── GCS upload ────────────────────────────────────────────────────────────────

def upload_to_gcs(local_path: Path, bucket_name: str, blob_name: str) -> str:
    """Upload a local file to GCS and return its gs:// URI."""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    print(f"[gcs] Uploading {local_path.name} → gs://{bucket_name}/{blob_name}")
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{blob_name}"


# ── Vision Product Search setup ───────────────────────────────────────────────

def ensure_product_set(client: vision.ProductSearchClient, location_path: str) -> str:
    """Create the ProductSet if it does not exist yet. Returns its full path."""
    product_set_path = client.product_set_path(
        project=PROJECT_ID, location=LOCATION, product_set=PRODUCT_SET_ID
    )
    try:
        existing = client.get_product_set(name=product_set_path)
        print(f"[vision] ProductSet already exists: {existing.name}")
        return existing.name
    except NotFound:
        pass

    print(f"[vision] Creating ProductSet '{PRODUCT_SET_ID}' in {LOCATION}...")
    product_set = vision.ProductSet(display_name=f"FamilyMart {PRODUCT_SET_ID}")
    created = client.create_product_set(
        parent=location_path,
        product_set=product_set,
        product_set_id=PRODUCT_SET_ID,
    )
    print(f"[vision] Created ProductSet: {created.name}")
    return created.name


def ensure_product(
    client: vision.ProductSearchClient,
    location_path: str,
    product_id: str,
    display_name: str,
) -> str:
    """Create a Product if missing; return its full resource name."""
    product_path = client.product_path(
        project=PROJECT_ID, location=LOCATION, product=product_id
    )
    try:
        existing = client.get_product(name=product_path)
        print(f"[vision]   ↳ Product exists: {existing.display_name}")
        return existing.name
    except NotFound:
        pass

    product = vision.Product(
        display_name=display_name,
        product_category=PRODUCT_CATEGORY,
    )
    created = client.create_product(
        parent=location_path,
        product=product,
        product_id=product_id,
    )
    print(f"[vision]   ↳ Created Product: {created.display_name}")
    return created.name


def add_reference_image(
    client: vision.ProductSearchClient,
    product_name: str,
    gcs_uri: str,
    reference_image_id: str,
) -> None:
    """Attach a reference image (stored in GCS) to an existing Product."""
    reference_image = vision.ReferenceImage(uri=gcs_uri)
    try:
        client.create_reference_image(
            parent=product_name,
            reference_image=reference_image,
            reference_image_id=reference_image_id,
        )
        print(f"[vision]   ↳ Added ReferenceImage '{reference_image_id}'")
    except AlreadyExists:
        print(f"[vision]   ↳ ReferenceImage '{reference_image_id}' already attached")


def add_product_to_set(
    client: vision.ProductSearchClient,
    product_set_name: str,
    product_name: str,
) -> None:
    """Idempotently make sure the Product is a member of the ProductSet."""
    try:
        client.add_product_to_product_set(
            name=product_set_name,
            product=product_name,
        )
        print("[vision]   ↳ Linked Product → ProductSet")
    except AlreadyExists:
        # add_product_to_product_set is idempotent in practice but older
        # client libs can raise — safe to swallow.
        print("[vision]   ↳ Product already in ProductSet")


# ── Main flow ────────────────────────────────────────────────────────────────

def main() -> None:
    _require_env()

    print("=" * 64)
    print("  FamilyMart Vision Product Search — Catalog Setup")
    print("=" * 64)
    print(f"  Project     : {PROJECT_ID}")
    print(f"  Location    : {LOCATION}")
    print(f"  Bucket      : gs://{GCS_BUCKET}")
    print(f"  ProductSet  : {PRODUCT_SET_ID}")
    print(f"  Category    : {PRODUCT_CATEGORY}")
    print(f"  Images dir  : {REF_IMAGES_DIR.resolve()}")
    print("=" * 64)

    images = list(_iter_reference_images(REF_IMAGES_DIR))
    if not images:
        print(f"[setup] No reference images found in {REF_IMAGES_DIR}. Nothing to do.")
        return
    print(f"[setup] Found {len(images)} reference image(s).\n")

    vision_client = vision.ProductSearchClient()
    location_path = f"projects/{PROJECT_ID}/locations/{LOCATION}"

    product_set_name = ensure_product_set(vision_client, location_path)

    for idx, image_path in enumerate(images, start=1):
        stem = image_path.stem
        product_id = _sanitize_product_id(stem)
        display_name = stem  # keep the human-readable filename as display name
        blob_name = f"reference_images/{image_path.name}"

        print(f"\n[{idx}/{len(images)}] {display_name}")

        try:
            gcs_uri = upload_to_gcs(image_path, GCS_BUCKET, blob_name)

            product_name = ensure_product(
                vision_client, location_path, product_id, display_name
            )

            # Use a stable reference-image ID so re-runs are idempotent.
            ref_image_id = _sanitize_product_id(f"{stem}-{image_path.stem}-ref")[:120]
            add_reference_image(vision_client, product_name, gcs_uri, ref_image_id)

            add_product_to_set(vision_client, product_set_name, product_name)

        except Exception as exc:  # noqa: BLE001 — log and continue to the next image
            print(f"[setup] ERROR on {image_path.name}: {exc}")
            continue

    print("\n" + "=" * 64)
    print(f"[setup] Done. ProductSet ready: {product_set_name}")
    print("[setup] You can now run the compliance checker against this catalog.")
    print("=" * 64)


if __name__ == "__main__":
    main()
