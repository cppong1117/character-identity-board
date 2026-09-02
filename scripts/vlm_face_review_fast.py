#!/usr/bin/env python3
"""
Vision LLM Face Verification — Fast version with image resize.
Resizes face crops to 128x128 before sending to Vision LLM for speed.
"""
import sys
import os
import json
import base64
import io
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# ─── Config ──────────────────────────────────────────────
CIB_API = "http://127.0.0.1:8322"
PROJECT_ID = 15
VISION_API_BASE = os.environ.get("VISION_API_BASE", "http://127.0.0.1:8011/v1")
VISION_API_KEY = os.environ.get("VISION_API_KEY", "local")
VISION_MODEL = os.environ.get("VISION_MODEL", "mage-vl")
BATCH_SIZE = 50
MAX_WORKERS = 10
RESIZE_TO = 128  # Resize images for faster processing

FACE_CHECK_PROMPT = """Is there a clearly visible human face in this image? 
Answer ONLY JSON: {"is_face": true/false, "reason": "brief"}
- true: clear face (even partial/blurry/angle)
- false: body/hands/objects/background/nothing/animal/unclear"""


def load_image_resized(path: str, size: int = 128) -> str:
    """Load image, resize, and return base64."""
    img = Image.open(path)
    img = img.resize((size, size), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=70)
    return base64.b64encode(buffer.getvalue()).decode()


def check_face(image_path: str) -> dict:
    """Send resized image to Vision LLM."""
    try:
        b64 = load_image_resized(image_path, RESIZE_TO)
        resp = requests.post(
            f"{VISION_API_BASE}/chat/completions",
            json={
                "model": VISION_MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": FACE_CHECK_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}],
                "max_tokens": 100,
                "temperature": 0.0
            },
            headers={"Authorization": f"Bearer {VISION_API_KEY}", "Content-Type": "application/json"},
            timeout=15
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        result = json.loads(content)
        return {"is_face": result.get("is_face", True), "reason": result.get("reason", "")}
    except Exception as e:
        return {"is_face": True, "reason": f"Error: {e}"}


def get_review_items(project_id: int) -> list:
    """Get all pending review items from CIB API."""
    resp = requests.get(f"{CIB_API}/projects/{project_id}/review-queue")
    resp.raise_for_status()
    return resp.json()


def resolve_crop_path(path_str: str) -> str:
    """Resolve crop path to absolute filesystem path."""
    if not path_str:
        return None
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str
    base = Path(os.path.expanduser("~/character-identity-board-data"))
    if "character-identity-board-data/" in path_str:
        rel = path_str.split("character-identity-board-data/", 1)[1]
        full = base / rel
        if full.exists():
            return str(full)
    return None


def exclude_tracklet(tracklet_id: int, reason: str):
    """Mark tracklet as excluded via CIB API."""
    resp = requests.patch(
        f"{CIB_API}/tracklets/{tracklet_id}/assignment",
        json={"review_status": "confirmed", "note": f"VLM excluded: {reason}"},
        headers={"Content-Type": "application/json"}
    )
    resp.raise_for_status()


def main():
    print(f"=== Vision LLM Face Verification (Fast) ===")
    print(f"Vision API: {VISION_API_BASE}")
    print(f"Model: {VISION_MODEL}")
    print(f"Resize: {RESIZE_TO}x{RESIZE_TO}")
    print()

    # Get review items
    print("Fetching review queue...")
    items = get_review_items(PROJECT_ID)
    print(f"Found {len(items)} items")

    # Resolve paths
    valid_items = []
    for item in items:
        if not item.get("face_crop_path"):
            continue
        crop_path = resolve_crop_path(item["face_crop_path"])
        if crop_path and os.path.exists(crop_path):
            item["_crop_path"] = crop_path
            valid_items.append(item)
    print(f"Valid crops: {len(valid_items)}")

    if not valid_items:
        print("No valid crops found.")
        return

    # Process
    print(f"\nProcessing with {MAX_WORKERS} workers, batch size {BATCH_SIZE}...")
    start_time = time.time()
    results = {"face": 0, "not_face": 0}
    excluded = []

    for i in range(0, len(valid_items), BATCH_SIZE):
        batch = valid_items[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(valid_items) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"Batch {batch_num}/{total_batches} ({len(batch)})... ", end="", flush=True)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(check_face, item["_crop_path"]): item for item in batch}
            for future in as_completed(futures):
                item = futures[future]
                result = future.result()
                if result["is_face"]:
                    results["face"] += 1
                else:
                    results["not_face"] += 1
                    excluded.append({
                        "tracklet_id": item["tracklet_id"],
                        "shot_number": item["shot_number"],
                        "reason": result["reason"]
                    })

        batch_faces = results["face"]  # running total
        print(f"running: {results['face']} faces, {results['not_face']} not-faces")
        time.sleep(0.2)

    elapsed = time.time() - start_time
    print(f"\n=== Results ===")
    print(f"Faces: {results['face']}")
    print(f"Not faces: {results['not_face']}")
    print(f"Time: {elapsed:.1f}s ({(results['face'] + results['not_face']) / elapsed:.1f} items/s)")

    # Auto-exclude
    if excluded:
        print(f"\nExcluding {len(excluded)} non-faces...")
        ok = 0
        for ex in excluded:
            try:
                exclude_tracklet(ex["tracklet_id"], ex["reason"])
                ok += 1
            except Exception as e:
                print(f"  Failed tracklet {ex['tracklet_id']}: {e}")
        print(f"✓ Excluded {ok}/{len(excluded)}")

    # Save report
    report_path = os.path.expanduser("~/character-identity-board/reports/vlm_face_review.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "project_id": PROJECT_ID,
            "total_items": len(items),
            "valid_crops": len(valid_items),
            "faces": results["face"],
            "not_faces": results["not_face"],
            "excluded_count": len(excluded),
            "excluded_samples": excluded[:20],
            "elapsed_seconds": elapsed,
            "model": VISION_MODEL,
            "resize": RESIZE_TO
        }, f, indent=2)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
