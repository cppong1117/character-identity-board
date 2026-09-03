#!/usr/bin/env python3
"""
VLM Face Review — Strict prompt version.
Correctly identifies non-faces from the review queue.
"""
import os, json, base64, io, time, requests, sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

CIB_API = "http://127.0.0.1:8322"
PROJECT_ID = 15
VISION_API = "http://127.0.0.1:8011/v1"
MODEL = "mage-vl"
RESIZE = 128
BATCH_SIZE = 50
WORKERS = 10

STRICT_PROMPT = """This is a face crop from a video. Examine carefully.

A REAL HUMAN FACE must show at least 2 of: eyes, nose, mouth, eyebrows, chin, cheekbones, forehead.

If the image shows: blurry mess, dark blob, clothing, hair only, background, texture, object, body part only (hand/ear/neck), or anything NOT a recognizable face → is_face=false.

Answer ONLY: {"is_face": true/false, "confidence": 0.0-1.0, "reason": "brief"}"""


def load_resized(path):
    img = Image.open(path).resize((RESIZE, RESIZE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def check_face(path):
    try:
        b64 = load_resized(path)
        resp = requests.post(VISION_API, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": STRICT_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}],
            "max_tokens": 150, "temperature": 0.0
        }, headers={"Authorization": "Bearer local", "Content-Type": "application/json"}, timeout=20)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"): content = content[4:]
            content = content.strip()
        r = json.loads(content)
        return {"is_face": r.get("is_face", True), "confidence": r.get("confidence", 0), "reason": r.get("reason", "")}
    except Exception as e:
        return {"is_face": True, "confidence": 0, "reason": f"Error: {e}"}


def resolve_path(p):
    if not p: return None
    if os.path.isabs(p) and os.path.exists(p): return p
    base = Path(os.path.expanduser("~/character-identity-board-data"))
    if "character-identity-board-data/" in p:
        full = base / p.split("character-identity-board-data/", 1)[1]
        if full.exists(): return str(full)
    return None


def main():
    print(f"=== VLM Face Review (Strict) ===")
    print(f"Model: {MODEL} | Workers: {WORKERS} | Batch: {BATCH_SIZE}")

    items = requests.get(f"{CIB_API}/projects/{PROJECT_ID}/review-queue").json()
    print(f"Review queue: {len(items)} items")

    valid = []
    for item in items:
        cp = resolve_path(item.get("face_crop_path", ""))
        if cp:
            item["_crop"] = cp
            valid.append(item)
    print(f"Valid crops: {len(valid)}")

    start = time.time()
    faces, not_faces = 0, 0
    excluded = []

    for i in range(0, len(valid), BATCH_SIZE):
        batch = valid[i:i+BATCH_SIZE]
        bn = i//BATCH_SIZE + 1
        tb = (len(valid)+BATCH_SIZE-1)//BATCH_SIZE
        print(f"Batch {bn}/{tb}... ", end="", flush=True)

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(check_face, it["_crop"]): it for it in batch}
            for f in as_completed(futs):
                it = futs[f]
                r = f.result()
                if r["is_face"]:
                    faces += 1
                else:
                    not_faces += 1
                    excluded.append({"tracklet_id": it["tracklet_id"], "shot": it["shot_number"],
                                     "reason": r["reason"], "conf": r["confidence"]})
        print(f"faces={faces} not_faces={not_faces}")
        time.sleep(0.2)

    elapsed = time.time() - start
    print(f"\n=== DONE ===")
    print(f"Faces: {faces}")
    print(f"Not faces: {not_faces}")
    print(f"Time: {elapsed:.0f}s ({(faces+not_faces)/elapsed:.1f}/s)")

    # Exclude non-faces
    if excluded:
        print(f"\nExcluding {len(excluded)} non-faces...")
        ok = 0
        for ex in excluded:
            try:
                requests.patch(f"{CIB_API}/tracklets/{ex['tracklet_id']}/assignment",
                    json={"review_status": "confirmed", "note": f"VLM: {ex['reason']}"},
                    headers={"Content-Type": "application/json"})
                ok += 1
            except: pass
        print(f"✓ Excluded {ok}/{len(excluded)}")

    # Save report
    rpt = os.path.expanduser("~/character-identity-board/reports/vlm_face_review.json")
    os.makedirs(os.path.dirname(rpt), exist_ok=True)
    with open(rpt, "w") as f:
        json.dump({"faces": faces, "not_faces": not_faces, "excluded": excluded, "elapsed": elapsed}, f, indent=2)
    print(f"Report: {rpt}")


if __name__ == "__main__":
    main()
