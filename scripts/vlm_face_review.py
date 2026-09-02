#!/usr/bin/env python3
"""
Vision LLM Face Verification — Auto-exclude non-faces from Review Queue.

Uses a Vision LLM (via OpenAI-compatible API) to examine face crops
and determine if they actually contain a human face.

Batch processes all review queue items, marks non-faces for exclusion.
"""
import sys
import os
import json
import base64
import sqlite3
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Config ──────────────────────────────────────────────
CIB_API = "http://127.0.0.1:8322"
CIB_DB = os.path.expanduser("~/character-identity-board-data/cib.sqlite3")
PROJECT_ID = 15  # Main project

# Vision LLM config — use any OpenAI-compatible endpoint
# Options:
#   - LiteLLM: http://localhost:4000/v1
#   - Ollama: http://localhost:11434/v1
#   - OpenAI: https://api.openai.com/v1
VISION_API_BASE = os.environ.get("VISION_API_BASE", "http://localhost:11434/v1")
VISION_API_KEY = os.environ.get("VISION_API_KEY", "ollama")
VISION_MODEL = os.environ.get("VISION_MODEL", "llava")  # or minicpm-v, qwen2.5-vl, etc.

BATCH_SIZE = 20  # Concurrent requests
MAX_WORKERS = 5  # Parallel threads

# ─── Prompt ──────────────────────────────────────────────
FACE_CHECK_PROMPT = """You are a face detection verifier. Look at this image crop and determine if it contains a clearly visible HUMAN FACE.

Answer ONLY with a JSON object:
{"is_face": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}

Rules:
- is_face=true: The image contains a clearly visible human face (even if partial, blurry, or at an angle)
- is_face=false: The image does NOT contain a human face — it could be: body/hands/objects/background/texture/nothing/animal/etc.
- Be strict: if you're not sure it's a face, say false
- Focus on whether you can see eyes/nose/mouth/face shape

Respond ONLY with the JSON object, nothing else."""


def load_image_as_base64(path: str) -> str:
    """Load image file and return base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def check_face_with_vlm(image_path: str) -> dict:
    """Send image to Vision LLM and get face classification."""
    try:
        b64 = load_image_as_base64(image_path)
        
        headers = {
            "Authorization": f"Bearer {VISION_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": FACE_CHECK_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }
            ],
            "max_tokens": 200,
            "temperature": 0.0
        }
        
        resp = requests.post(
            f"{VISION_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        
        content = resp.json()["choices"][0]["message"]["content"]
        
        # Parse JSON from response (handle markdown code blocks)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        result = json.loads(content)
        return {
            "is_face": result.get("is_face", True),
            "confidence": result.get("confidence", 0.5),
            "reason": result.get("reason", "")
        }
    except Exception as e:
        return {"is_face": True, "confidence": 0.5, "reason": f"Error: {e}"}


def get_review_items(project_id: int) -> list:
    """Get all pending review items from CIB API."""
    resp = requests.get(f"{CIB_API}/projects/{project_id}/review-queue")
    resp.raise_for_status()
    return resp.json()


def exclude_tracklet(tracklet_id: int, reason: str):
    """Mark a tracklet as excluded via CIB API."""
    resp = requests.patch(
        f"{CIB_API}/tracklets/{tracklet_id}/assignment",
        json={"review_status": "confirmed", "note": f"VLM excluded: {reason}"},
        headers={"Content-Type": "application/json"}
    )
    resp.raise_for_status()


def resolve_crop_path(path_str: str) -> str:
    """Resolve the crop path to absolute filesystem path."""
    if not path_str:
        return None
    # Path format: .../character-identity-board-data/projects/...
    # We need to find the actual file
    base = Path(os.path.expanduser("~/character-identity-board-data"))
    
    # Try absolute first
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str
    
    # Try relative to data dir
    # Remove leading project path prefix
    rel = path_str
    if "character-identity-board-data/" in rel:
        rel = rel.split("character-identity-board-data/", 1)[1]
    
    full = base / rel
    if full.exists():
        return str(full)
    
    # Try searching in projects dir
    projects_dir = base / "projects"
    if projects_dir.exists():
        # Just use filename match
        filename = os.path.basename(path_str)
        for f in projects_dir.rglob(filename):
            return str(f)
    
    return None


def main():
    print(f"=== Vision LLM Face Verification ===")
    print(f"API: {VISION_API_BASE}")
    print(f"Model: {VISION_MODEL}")
    print(f"Project: {PROJECT_ID}")
    print()
    
    # Test connection
    print("Testing Vision LLM connection...")
    try:
        resp = requests.get(f"{VISION_API_BASE}/models", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        available = [m["id"] for m in models]
        print(f"Available models: {available[:5]}")
        if VISION_MODEL not in available:
            print(f"⚠ Warning: {VISION_MODEL} not found in available models!")
    except Exception as e:
        print(f"⚠ Could not list models: {e}")
    
    # Get review items
    print("\nFetching review queue...")
    items = get_review_items(PROJECT_ID)
    print(f"Found {len(items)} review items")
    
    # Resolve crop paths
    print("\nResolving crop paths...")
    valid_items = []
    for item in items:
        if not item.get("face_crop_path"):
            continue
        crop_path = resolve_crop_path(item["face_crop_path"])
        if crop_path and os.path.exists(crop_path):
            item["_crop_path"] = crop_path
            valid_items.append(item)
    
    print(f"Found {len(valid_items)} items with valid crop paths")
    
    if not valid_items:
        print("No valid crops found. Check paths.")
        return
    
    # Process in batches
    print(f"\nProcessing {len(valid_items)} images with Vision LLM...")
    print(f"Batch size: {BATCH_SIZE}, Workers: {MAX_WORKERS}")
    print()
    
    results = {"face": 0, "not_face": 0, "error": 0}
    excluded = []
    
    start_time = time.time()
    
    for i in range(0, len(valid_items), BATCH_SIZE):
        batch = valid_items[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(valid_items) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"Batch {batch_num}/{total_batches} ({len(batch)} items)...", end=" ", flush=True)
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for item in batch:
                future = executor.submit(check_face_with_vlm, item["_crop_path"])
                futures[future] = item
            
            batch_results = []
            for future in as_completed(futures):
                item = futures[future]
                result = future.result()
                batch_results.append((item, result))
                
                if result["is_face"]:
                    results["face"] += 1
                else:
                    results["not_face"] += 1
                    excluded.append({
                        "tracklet_id": item["tracklet_id"],
                        "shot_number": item["shot_number"],
                        "reason": result["reason"],
                        "confidence": result["confidence"]
                    })
        
        # Report batch results
        batch_faces = sum(1 for _, r in batch_results if r["is_face"])
        batch_not_faces = len(batch_results) - batch_faces
        print(f"faces={batch_faces}, not_faces={batch_not_faces}")
        
        # Rate limiting
        time.sleep(0.5)
    
    elapsed = time.time() - start_time
    
    print(f"\n=== Results ===")
    print(f"Total processed: {results['face'] + results['not_face']}")
    print(f"Faces: {results['face']}")
    print(f"Not faces: {results['not_face']}")
    print(f"Time: {elapsed:.1f}s ({(results['face'] + results['not_face']) / elapsed:.1f} items/s)")
    
    # Ask to confirm exclusion
    if excluded:
        print(f"\n--- Items to exclude ({len(excluded)}) ---")
        for ex in excluded[:20]:
            print(f"  Shot #{ex['shot_number']} tracklet={ex['tracklet_id']} conf={ex['confidence']:.2f} reason={ex['reason'][:50]}")
        if len(excluded) > 20:
            print(f"  ... and {len(excluded) - 20} more")
        
        print(f"\nExcluding {len(excluded)} non-face items...")
        excluded_count = 0
        for ex in excluded:
            try:
                exclude_tracklet(ex["tracklet_id"], ex["reason"])
                excluded_count += 1
            except Exception as e:
                print(f"  Failed to exclude tracklet {ex['tracklet_id']}: {e}")
        
        print(f"✓ Excluded {excluded_count}/{len(excluded)} non-face items")
    else:
        print("\nNo non-faces detected!")
    
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
            "excluded": excluded,
            "elapsed_seconds": elapsed,
            "vision_model": VISION_MODEL,
            "vision_api": VISION_API_BASE
        }, f, indent=2)
    
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
