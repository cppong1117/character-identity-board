"""Benchmark the corrected dual-closeup dialogue fixture (both faces embeddable).

Runs the full real pipeline (Probe -> Shots -> Detect/Track -> Embed -> Cluster),
then evaluates identity accuracy in two modes:
  1) Discovery auto-cluster JSON-only (what auto clustering produced).
  2) Reference-mode: feed the two source face images as references, match tracklets,
     compute Identity Purity / Recall / F1 against ground truth A,B,A,B.
Records everything to ~/character-identity-board-data/benchmarks/V0.1/results/.
"""
from __future__ import annotations

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))                # for `backend.app` imports
sys.path.insert(0, str(_ROOT / "backend"))    # for pipeline-internal `app.*` imports

import json
import shutil
import numpy as np
from sqlalchemy.orm import Session

ROOT = Path.home() / "character-identity-board-data"
OUT = ROOT / "benchmarks" / "V0.1" / "results"
FIX = ROOT / "benchmarks" / "V0.1" / "generated" / "testA_dual_closeup_dialogue.mp4"
FACE1 = ROOT / "cache" / "test_face.jpg"        # person A
FACE2 = ROOT / "cache" / "test_face2_closeup.jpg"  # person B (closeup we generated)

GT = {"1": "A", "2": "B", "3": "A", "4": "B"}


def face_embed(img_path):
    import cv2
    from backend.app.face_engine import FaceEngine
    fe = FaceEngine()
    img = cv2.imread(str(img_path))
    faces = fe.detect_faces(img)
    if not faces:
        return None
    best = max(faces, key=lambda f: f["score"])
    return fe.get_embedding(img, best["bbox"])


def run(db: Session, project, video_id):
    from backend.app.models import Shot, Tracklet, FaceObservation, IdentityAssignment, Character
    shots = db.query(Shot).filter(Shot.video_id == video_id).order_by(Shot.shot_number).all()
    result = {
        "fixture": "testA_dual_closeup_dialogue.mp4",
        "ground_truth": GT,
        "detected_shots": len(shots),
        "shots": {},
    }
    tracklet_emb = {}
    for sh in shots:
        tls = db.query(Tracklet).filter(Tracklet.shot_id == sh.id).order_by(Tracklet.track_number).all()
        for t in tls:
            obs = db.query(FaceObservation).filter(FaceObservation.tracklet_id == t.id).all()
            embs = [np.frombuffer(o.embedding, dtype=np.float32).reshape(-1) for o in obs if o.embedding is not None]
            ia = db.query(IdentityAssignment).filter(IdentityAssignment.tracklet_id == t.id).first()
            char_code = None
            if ia and ia.character_id:
                c = db.get(Character, ia.character_id)
                char_code = c.character_code if c else None
            result["shots"][str(sh.shot_number)] = {
                "tracklets": len(tls), "obs": len(obs),
                "n_embedded": len(embs),
                "auto_char_code": char_code,
            }
            if embs:
                tracklet_emb[sh.shot_number] = np.mean(np.stack(embs), axis=0)
    return result, tracklet_emb


def identity_metrics(tracklet_emb, ref_map, threshold=0.85):
    """ref_map: char->embedding. Assign each tracklet to best ref above threshold, else Unknown."""
    pred = {}
    confs = {}
    for sn, emb in tracklet_emb.items():
        best_c, best_s = None, -1
        for c, re_ in ref_map.items():
            if re_ is None:
                continue
            s = float(np.dot(emb.reshape(-1), re_.reshape(-1)) /
                      (np.linalg.norm(emb) * np.linalg.norm(re_) + 1e-12))
            if s > best_s:
                best_c, best_s = c, s
        pred[sn] = best_c if best_s >= threshold else "UNKNOWN"
        confs[sn] = best_s
    # metrics vs GT
    total = len(pred)
    correct = sum(1 for sn, pc in pred.items() if GT.get(sn) == pc)
    unknown = sum(1 for v in pred.values() if v == "UNKNOWN")
    purity_num = sum(1 for sn, pc in pred.items() if pc != "UNKNOWN" and GT.get(sn) == pc)
    purity_den = sum(1 for sn, pc in pred.items() if pc != "UNKNOWN")
    purity = purity_num / purity_den if purity_den else 0.0
    # recall: among embedded (non-unknown assignments) how many correct
    recall_den = sum(1 for sn, pc in pred.items() if pc != "UNKNOWN")
    recall = correct / max(1, recall_den) if recall_den else 0.0
    # overall accuracy incl unknown-as-miss
    acc = correct / total if total else 0.0
    f1 = 2 * purity * recall / (purity + recall) if (purity + recall) else 0.0
    return {
        "threshold": threshold,
        "total_tracklets": total,
        "purity_correct_nonunknown": purity_num,
        "purity_denominator": purity_den,
        "identity_purity": round(purity, 4),
        "recall_nonunknown": round(recall, 4),
        "identity_recall_f1": round(f1, 4),
        "unknown_count": unknown,
        "unknown_rate": round(unknown / total, 4) if total else 0,
        "overall_accuracy": round(acc, 4),
        "predictions": pred,
        "confidences": {k: round(v, 3) for k, v in confs.items()},
    }


def main():
    from backend.app.database import SessionLocal
    from backend.app.models import Project, Video
    db = SessionLocal()
    # Use a dedicated scratch project for the reference-mode eval
    project = Project(name="Benchmark Identity (dual closeup)", status="created")
    db.add(project); db.commit(); db.refresh(project)
    pdir = ROOT / "projects" / str(project.id); pdir.mkdir(parents=True, exist_ok=True)
    vid = Video(project_id=project.id, filename=FIX.name, filepath=str(FIX),
                processing_status="uploaded", pipeline_stage="uploaded")
    db.add(vid); db.commit(); db.refresh(vid)

    from backend.app.pipeline import Pipeline
    Pipeline(db, vid.id).run()
    db.expire_all()

    result, tracklet_emb = run(db, project, vid.id)
    print("TT", result["detected_shots"])

    # Discovery auto result: purity among embedded (auto char codes)
    # Reference embeddings
    ref_map = {"A": face_embed(FACE1), "B": face_embed(FACE2)}
    print("REF A emb:", ref_map["A"] is not None, "REF B emb:", ref_map["B"] is not None)

    id_metrics = identity_metrics(tracklet_emb, ref_map, threshold=0.85)
    result["identity_ref_mode_threshold0_85"] = id_metrics
    id_metrics_loose = identity_metrics(tracklet_emb, ref_map, threshold=0.70)
    result["identity_ref_mode_threshold0_70"] = id_metrics_loose

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "identity_dual_closeup.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # cleanup scratch project rows? keep for inspection but remove from active listing later
    db.close()


if __name__ == "__main__":
    main()
