#!/usr/bin/env python3
"""
Fast non-face detection: file size heuristic + Vision LLM for borderline cases.
"""
import os, json, base64, io, time, requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "http://127.0.0.1:8322"
VISION = "http://127.0.0.1:8011/v1/chat/completions"
MODEL = "mage-vl"
DATA = os.path.expanduser("~/character-identity-board-data")
PROJECT_ID = 15

PROMPT = """This is a face crop from a video. Examine carefully.

A REAL HUMAN FACE must show at least 2 of: eyes, nose, mouth, eyebrows, chin, cheekbones, forehead.

If the image shows: blurry mess, dark blob, clothing, hair only, background, texture, object, body part only (hand/ear/neck), or anything NOT a recognizable face → is_face=false.

Answer ONLY JSON: {"is_face": true/false, "reason": "brief"}"""

# Thresholds
SMALL_FILE = 4000    # < 4KB = definitely not a face
BORDERLINE_FILE = 12000  # 4-12KB = check with Vision LLM


def resolve(p):
    if not p: return None
    if "character-identity-board-data/" in p:
        full = os.path.join(DATA, p.split("character-identity-board-data/", 1)[1])
        if os.path.exists(full): return full
    return None


def check_vlm(path):
    """Check with Vision LLM."""
    try:
        img = Image.open(path).resize((128, 128), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode()
        resp = requests.post(VISION, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}],
            "max_tokens": 100, "temperature": 0.0
        }, headers={"Authorization": "Bearer local", "Content-Type": "application/json"}, timeout=45)
        resp.raise_for_status()
        c = resp.json()["choices"][0]["message"]["content"].strip()
        if c.startswith("```"):
            c = c.split("```")[1]
            if c.startswith("json"): c = c[4:]
            c = c.strip()
        r = json.loads(c)
        return {"is_face": r.get("is_face", True), "reason": r.get("reason", "")}
    except Exception as e:
        return {"is_face": None, "reason": f"Error: {e}"}


def main():
    print("=== Fast Non-Face Detection ===")
    
    items = requests.get(f"{API}/projects/{PROJECT_ID}/review-queue").json()
    print(f"Review queue: {len(items)} items")

    # Classify by file size
    definite_nonface = []  # < 4KB
    borderline = []        # 4-12KB
    likely_face = []       # > 12KB

    for it in items:
        cp = resolve(it.get("face_crop_path", ""))
        if not cp: continue
        size = os.path.getsize(cp)
        it["_path"] = cp
        it["_size"] = size
        
        if size < SMALL_FILE:
            definite_nonface.append(it)
        elif size < BORDERLINE_FILE:
            borderline.append(it)
        else:
            likely_face.append(it)

    print(f"Definite non-face (< {SMALL_FILE}B): {len(definite_nonface)}")
    print(f"Borderline ({SMALL_FILE}-{BORDERLINE_FILE}B): {len(borderline)}")
    print(f"Likely face (> {BORDERLINE_FILE}B): {len(likely_face)}")

    # Auto-exclude definite non-faces
    print(f"\n--- Excluding {len(definite_nonface)} definite non-faces ---")
    excluded_count = 0
    for it in definite_nonface:
        try:
            requests.patch(f"{API}/tracklets/{it['tracklet_id']}/assignment",
                json={"review_status": "confirmed", "note": f"Auto-excluded: file too small ({it['_size']}B)"},
                headers={"Content-Type": "application/json"}, timeout=10)
            excluded_count += 1
        except: pass
    print(f"✓ Excluded {excluded_count}/{len(definite_nonface)}")

    # Check borderline with Vision LLM
    if borderline:
        print(f"\n--- Checking {len(borderline)} borderline items with Vision LLM ---")
        vlm_nonface = 0
        vlm_errors = 0
        
        for i, it in enumerate(borderline):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(borderline)} (non-face: {vlm_nonface}, errors: {vlm_errors})")
            
            result = check_vlm(it["_path"])
            if result["is_face"] is False:
                vlm_nonface += 1
                try:
                    requests.patch(f"{API}/tracklets/{it['tracklet_id']}/assignment",
                        json={"review_status": "confirmed", "note": f"VLM: {result['reason']}"},
                        headers={"Content-Type": "application/json"}, timeout=10)
                except: pass
            elif result["is_face"] is None:
                vlm_errors += 1
            
            time.sleep(0.5)  # Rate limiting
        
        print(f"  Done: non-face={vlm_nonface}, errors={vlm_errors}")

    total_excluded = excluded_count + (len(borderline) - vlm_errors if 'vlm_nonface' in dir() else 0)
    print(f"\n=== SUMMARY ===")
    print(f"Total review items: {len(items)}")
    print(f"Definite non-faces (auto-excluded): {excluded_count}")
    print(f"Borderline checked: {len(borderline)}")
    print(f"Remaining for human review: {len(likely_face)}")
    
    # Save report
    report = {
        "total": len(items),
        "definite_nonface": len(definite_nonface),
        "definite_excluded": excluded_count,
        "borderline": len(borderline),
        "likely_face": len(likely_face)
    }
    rpt_path = os.path.expanduser("~/character-identity-board/reports/nonface_detection.json")
    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {rpt_path}")


if __name__ == "__main__":
    main()
