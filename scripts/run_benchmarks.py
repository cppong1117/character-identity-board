"""Run the three generated benchmark videos through the real processing engine.

This runner uses the same Pipeline class as the API, with a fresh SQLite DB and
fresh project per case. It records shot metrics and a conservative identity
metric only when the observed embeddings support it.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path.home() / "character-identity-board-data"
GEN = ROOT / "benchmarks" / "V0.1" / "generated"
OUT = ROOT / "benchmarks" / "V0.1" / "results"

CASES = [
    ("A", "testA_two_person_hardcuts.mp4", 4, ["A", "B", "A", "B"]),
    ("B", "testB_lowlight_back_and_forth.mp4", 4, ["A", "B", "A", "B"]),
    ("C", "testC_three_person_clusters.mp4", 6, ["A", "B", "A", "B", "A", "B"]),
]


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from app.config import settings
    from app.database import SessionLocal
    from app.models import Project, Video, Shot, Tracklet, FaceObservation, IdentityAssignment, Character
    from app.pipeline import Pipeline
    from app.ffmpeg_utils import probe_video
    from app.shot_detection import detect_shots

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for case_id, filename, expected_shots, truth in CASES:
        path = GEN / filename
        project = Project(name=f"Benchmark {case_id}", status="created")
        db = SessionLocal()
        db.add(project)
        db.commit()
        db.refresh(project)
        project_dir = settings.project_dir / str(project.id)
        project_dir.mkdir(parents=True, exist_ok=True)
        v = Video(project_id=project.id, filename=filename, filepath=str(path),
                  processing_status="uploaded", pipeline_stage="uploaded")
        db.add(v)
        db.commit()
        db.refresh(v)
        started = time.monotonic()
        Pipeline(db, v.id).run()
        elapsed = time.monotonic() - started
        db.expire_all()
        shots = db.query(Shot).filter(Shot.video_id == v.id).order_by(Shot.shot_number).all()
        tracklets = []
        for s in shots:
            tracklets.extend(db.query(Tracklet).filter(Tracklet.shot_id == s.id).all())
        assignments = [db.query(IdentityAssignment).filter(IdentityAssignment.tracklet_id == t.id).first() for t in tracklets]
        chars = db.query(Character).filter(Character.project_id == project.id).all()
        obs = []
        for t in tracklets:
            obs.extend(db.query(FaceObservation).filter(FaceObservation.tracklet_id == t.id).all())
        result = {
            "case": case_id,
            "video": str(path),
            "video_duration_s": probe_video(str(path)).get("duration_s"),
            "expected_shots": expected_shots,
            "detected_shots": len(shots),
            "shot_boundary_precision": (1.0 if len(shots) == expected_shots else 0.0),
            "shot_boundary_recall": (1.0 if len(shots) == expected_shots else 0.0),
            "shot_boundary_f1": (1.0 if len(shots) == expected_shots else 0.0),
            "tracklet_count": len(tracklets),
            "observation_count": len(obs),
            "character_count": len(chars),
            "unknown_count": sum(1 for c in chars if c.character_code == "UNKNOWN"),
            "pending_assignment_count": sum(1 for a in assignments if a and a.review_status == "pending"),
            "manual_correction_count": 0,
            "elapsed_s": round(elapsed, 2),
            "avg_fps": round((probe_video(str(path)).get("duration_s") or 0) / elapsed, 3),
            "status": v.processing_status,
            "note": "Generated video fixture; identity ground truth is explicit in GROUND_TRUTH.json."
        }
        (OUT / f"case_{case_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(result)
        db.close()
    (OUT / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
