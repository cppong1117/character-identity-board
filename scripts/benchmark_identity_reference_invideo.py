"""Reference-mode with in-video reference crops (the intended usage).

User supplies a representative frame per character (as they would in the UI).
Here we take the highest-quality observation crop of shot1 (person A) and
shot2 (person B) as the two references, then match all tracklets against them.
This is the realistic Reference Mode path (not a synthetic upscale).
"""
from __future__ import annotations

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "backend"))

import json
import cv2
import numpy as np
from backend.app.database import SessionLocal
from backend.app import models as M
from backend.app.face_engine import FaceEngine
from backend.app.clustering import match_to_references, _cosine_sim

ROOT = Path.home() / "character-identity-board-data"
OUT = ROOT / "benchmarks" / "V0.1" / "results"
GT = {"1": "A", "2": "B", "3": "A", "4": "B"}


def obs_embedding_from_crop(face_crop_path):
    """Re-embed a stored face crop to get its SFace vector (standalone check)."""
    fe = FaceEngine()
    img = cv2.imread(str(face_crop_path))
    if img is None:
        return None
    faces = fe.detect_faces(img)
    if not faces:
        return None
    best = max(faces, key=lambda f: f["score"])
    return fe.get_embedding(img, best["bbox"])


VIDEO_ID = 12  # completed dual-closeup benchmark


def main():
    vid_id = int(sys.argv[1]) if len(sys.argv) > 1 else VIDEO_ID
    db = SessionLocal()
    # Reference = best-face observation crop for shot 1 (person A) and shot 2 (person B)
    refs = {}
    for shot_num, char in [(1, "A"), (2, "B")]:
        sh = db.query(M.Shot).filter(M.Shot.video_id == vid_id, M.Shot.shot_number == shot_num).first()
        if sh is None:
            raise SystemExit(f"shot {shot_num} not found for video {vid_id}")
        t = db.query(M.Tracklet).filter(M.Tracklet.shot_id == sh.id).first()
        if t is None:
            raise SystemExit(f"no tracklet for shot {shot_num}")
        obs = db.query(M.FaceObservation).filter(M.FaceObservation.tracklet_id == t.id).all()
        best = max(obs, key=lambda o: o.quality_score)
        emb = obs_embedding_from_crop(best.face_crop_path)
        refs[char] = emb
        print(f"reference {char}: crop={best.face_crop_path} quality={best.quality_score:.3f} embedded={emb is not None}")

    # Match each shot's tracklet vs references
    fe = FaceEngine()
    matches = {}
    for shot_num in ["1", "2", "3", "4"]:
        sh = db.query(M.Shot).filter(M.Shot.video_id == vid_id, M.Shot.shot_number == int(shot_num)).first()
        if sh is None:
            matches[shot_num] = {"pred": "NO_SHOT", "sim": 0}
            continue
        t = db.query(M.Tracklet).filter(M.Tracklet.shot_id == sh.id).first()
        obs = db.query(M.FaceObservation).filter(M.FaceObservation.tracklet_id == t.id).all() if t else []
        embs = [np.frombuffer(o.embedding, dtype=np.float32).reshape(-1) for o in obs if o.embedding is not None]
        track = np.mean(np.stack(embs), axis=0) if embs else None
        if track is None:
            matches[shot_num] = {"pred": "UNKNOWN", "sim": 0}
            continue
        best_c, best_s = match_to_references(
            track, [{"character_id": k, "embedding": v} for k, v in refs.items()], threshold=0.85)
        matches[shot_num] = {"pred": best_c or "UNKNOWN", "sim": round(best_s, 3)}
        print(f"shot {shot_num} GT={GT[shot_num]}  pred={best_c or 'UNKNOWN'}  best_sim={best_s:.3f}")

    correct = sum(1 for sn, m in matches.items() if m["pred"] == GT[sn])
    unknown = sum(1 for m in matches.values() if m["pred"] == "UNKNOWN")
    # purity among non-unknown
    non_unknown = {sn: m for sn, m in matches.items() if m["pred"] != "UNKNOWN"}
    purity = sum(1 for sn, m in non_unknown.items() if m["pred"] == GT[sn]) / len(non_unknown) if non_unknown else 0.0
    result = {
        "fixture": "testA_dual_closeup_dialogue.mp4 (in-video reference mode)",
        "mode": "reference_mode_in_video_refs",
        "reference_sources": {"A": "shot1 best crop", "B": "shot2 best crop"},
        "matches": matches,
        "reference_mode_accuracy": round(correct / len(matches), 4),
        "reference_mode_purity_nonunknown": round(purity, 4),
        "unknown_count": unknown,
        "unknown_rate": round(unknown / len(matches), 4),
        "note": "Uses actual in-video best-face crops as references (realistic UI usage).",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "identity_dual_closeup_reference_invideo.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    db.close()


if __name__ == "__main__":
    main()
