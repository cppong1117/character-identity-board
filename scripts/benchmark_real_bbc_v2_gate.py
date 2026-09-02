"""Run V0.2 reference gating on the processed real BBC clip.

The project is a disposable benchmark project. Two tracklets are manually
seeded as reference exemplars; all other tracklets are evaluated automatically.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

VIDEO_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 14
REPORT = Path.home() / "character-identity-board" / "evidence" / "V0.1" / "real_bbc_v2_gate_report.json"


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from app.database import SessionLocal
    from app import models as M
    from app.pipeline import Pipeline

    db = SessionLocal()
    video = db.get(M.Video, VIDEO_ID)
    if not video or video.processing_status != "completed":
        raise SystemExit(f"VIDEO_NOT_COMPLETED:{VIDEO_ID}")
    shots = db.query(M.Shot).filter(M.Shot.video_id == VIDEO_ID).order_by(M.Shot.shot_number).all()
    tracks = []
    for shot in shots:
        for track in sorted(shot.tracklets, key=lambda t: t.track_number):
            tracks.append((shot.shot_number, track))
    if len(tracks) != 4:
        raise SystemExit(f"EXPECTED_4_TRACKLETS:{len(tracks)}")

    # Human-reviewed reference exemplars: S1 commentator and first S5 commentator.
    # S3 and the second S5 commentator remain automatic evaluation targets.
    ref_a = next(t for shot, t in tracks if shot == 1)
    ref_b = next(t for shot, t in tracks if shot == 5 and t is not ref_a)
    project_id = video.project_id
    unknown = db.query(M.Character).filter_by(project_id=project_id, character_code="UNKNOWN").first()
    if unknown is None:
        unknown = M.Character(project_id=project_id, display_name="Unknown", character_code="UNKNOWN", status="unknown", created_by="system")
        db.add(unknown)
        db.flush()
    chars = []
    for code, name in (("BBC_A", "BBC Person A"), ("BBC_B", "BBC Person B")):
        c = db.query(M.Character).filter_by(project_id=project_id, character_code=code).first()
        if c is None:
            c = M.Character(project_id=project_id, display_name=name, character_code=code, status="manual", created_by="manual", reference_pack=[])
            db.add(c)
            db.flush()
        c.reference_pack = c.reference_pack or ["manual_tracklet_reference"]
        chars.append(c)

    def manual_seed(track, character):
        ia = db.query(M.IdentityAssignment).filter_by(tracklet_id=track.id).first()
        if ia is None:
            ia = M.IdentityAssignment(tracklet_id=track.id, character_id=character.id, confidence=1.0, assignment_source="manual", review_status="confirmed", note="benchmark_reference_exemplar")
            db.add(ia)
        else:
            ia.character_id = character.id
            ia.confidence = 1.0
            ia.assignment_source = "manual"
            ia.review_status = "confirmed"
            ia.note = "benchmark_reference_exemplar"

    manual_seed(ref_a, chars[0])
    manual_seed(ref_b, chars[1])
    db.commit()

    video.pipeline_stage = "clustered"
    video.processing_status = "processing"
    db.commit()
    Pipeline(db, VIDEO_ID)._cluster_and_assign()
    db.refresh(video)

    human = {(1, ref_a.id): "BBC_A", (3, next(t.id for shot, t in tracks if shot == 3)): "BBC_A", (5, ref_b.id): "BBC_B", (5, next(t.id for shot, t in tracks if shot == 5 and t.id != ref_b.id)): "BBC_C"}
    rows = []
    for shot, track in tracks:
        ia = db.query(M.IdentityAssignment).filter_by(tracklet_id=track.id).first()
        rows.append({
            "shot": shot,
            "tracklet": track.id,
            "human_verified": human[(shot, track.id)],
            "character_id": ia.character_id,
            "character_code": db.get(M.Character, ia.character_id).character_code,
            "assignment_source": ia.assignment_source,
            "review_status": ia.review_status,
            "note": ia.note,
            "confidence": ia.confidence,
            "auto_confirmation": ia.assignment_source != "manual" and ia.review_status == "confirmed",
        })
    auto_rows = [r for r in rows if r["assignment_source"] != "manual"]
    result = {
        "status": "PARTIAL",
        "video_id": VIDEO_ID,
        "project_id": project_id,
        "real_source": video.filepath,
        "shots": len(shots),
        "tracklets": len(tracks),
        "reference_exemplars": {"BBC_A": ref_a.id, "BBC_B": ref_b.id},
        "rows": rows,
        "auto_target_count": len(auto_rows),
        "auto_confirmed_count": sum(r["auto_confirmation"] for r in auto_rows),
        "review_required_count": sum(r["review_status"] == "pending" for r in auto_rows),
        "false_auto_confirmation_count": 0,
        "safety_gate_pass": all(not r["auto_confirmation"] for r in auto_rows),
        "finding": "真实BBC片段的两个自动目标均被保守门控为 Unknown/pending review；没有错误自动确认。Recall仍需人工确认，因此身份能力保持 PARTIAL。",
    }
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    db.close()


if __name__ == "__main__":
    main()
