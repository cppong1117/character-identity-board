"""Run Reference Mode with persisted reference images on a real BBC clip."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path[:0] = [str(ROOT), str(BACKEND)]

from backend.app.config import settings
from backend.app.database import SessionLocal
from backend.app.face_engine import get_face_engine
from backend.app.models import Character, FaceObservation, IdentityAssignment, Tracklet, Shot, Video
from backend.app.pipeline import _char_reference_embeddings
from backend.app.reference_embeddings import embed_reference_image
from backend.app.clustering import evaluate_reference_match


def crop_path_for(db, tracklet_id):
    row = db.query(FaceObservation).filter(FaceObservation.tracklet_id == tracklet_id).order_by(FaceObservation.quality_score.desc()).first()
    return row.face_crop_path if row else None


def main(video_id: int = 14):
    db = SessionLocal()
    video = db.get(Video, video_id)
    if not video:
        raise SystemExit(f"video {video_id} not found")
    shots = db.query(Shot).filter(Shot.video_id == video_id).order_by(Shot.shot_number).all()
    tracks = db.query(Tracklet).filter(Tracklet.shot_id.in_([s.id for s in shots])).order_by(Tracklet.id).all()
    # The first and third visible speaking tracklets were manually verified as A/B exemplars.
    if len(tracks) < 4:
        raise SystemExit("expected four tracklets")
    char_a = db.query(Character).filter_by(project_id=video.project_id, character_code="BBC_A_IMG").first()
    char_b = db.query(Character).filter_by(project_id=video.project_id, character_code="BBC_B_IMG").first()
    if not char_a:
        char_a = Character(project_id=video.project_id, display_name="BBC_A_IMG", character_code="BBC_A_IMG", status="manual", created_by="benchmark")
        db.add(char_a)
    if not char_b:
        char_b = Character(project_id=video.project_id, display_name="BBC_B_IMG", character_code="BBC_B_IMG", status="manual", created_by="benchmark")
        db.add(char_b)
    db.commit()
    ref_specs = [(char_a, tracks[0]), (char_b, tracks[2])]
    for char, track in ref_specs:
        path = crop_path_for(db, track.id)
        char.reference_image = path
        char.reference_pack = [path]
    db.commit()
    refs = {char.id: _char_reference_embeddings(db, char) for char, _ in ref_specs}
    negatives = {
        cid: [vector for other_id, values in refs.items() if other_id != cid for vector in values]
        for cid in refs
    }
    rows = []
    engine = get_face_engine()
    for track in tracks:
        target_path = crop_path_for(db, track.id)
        target = embed_reference_image(target_path, engine) if target_path else None
        if not target or target.embedding is None:
            continue
        emb = target.embedding
        result = evaluate_reference_match(emb, refs, negatives=negatives, threshold=0.80, margin=0.10)
        shot = db.get(Shot, track.shot_id)
        rows.append({"tracklet_id": track.id, "shot": shot.shot_number if shot else None,
                     "target_reference_image": target_path,
                     "target_embedding_dim": target.embedding_dim,
                     "match": result.__dict__})
    out = ROOT / "evidence" / "V0.1" / "real_bbc_image_reference_report.json"
    out.write_text(json.dumps({"status": "PARTIAL", "video_id": video_id, "reference_source": "persisted face crop images re-embedded with YuNet+SFace", "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "rows": len(rows), "reference_dims": {str(k): [int(x.size) for x in v] for k, v in refs.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
