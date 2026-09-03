"""Export endpoints: JSON/CSV/Contact Sheet/HTML + recluster."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Project, Video, IdentityAssignment, Tracklet, FaceObservation, Shot
from ..schemas import ExportRequest
from .. import reporting
from .. import corrections

log = logging.getLogger("cib.api.exports")
router = APIRouter(tags=["exports"])


@router.post("/projects/{project_id}/export")
def export_project(project_id: int, body: ExportRequest = None,
                   db: Session = Depends(get_db)):
    p = db.get(Project, project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    formats = body.formats if body else ["json", "csv", "contact_sheet", "html"]
    manifest = reporting.export_project(db, project_id, formats)
    # relativize to /media for UI display
    media_manifest = {
        "exports_dir": manifest["exports_dir"],
        "files": manifest["files"],
    }
    return media_manifest


@router.post("/projects/{project_id}/recluster")
def recluster(project_id: int, db: Session = Depends(get_db)):
    p = db.get(Project, project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    result = corrections.recluster_project(db, project_id)
    return {"result": "ok", **result}


# ---- Review Queue ----
@router.get("/projects/{project_id}/review-queue")
def review_queue(project_id: int, status: str = "pending", db: Session = Depends(get_db)):
    """Central queue of low-confidence / unknown / ambiguous samples.
    
    Query param: status=pending (default) | confirmed | all
    """
    vids = db.query(Video).filter(Video.project_id == project_id).all()
    vid_ids = [v.id for v in vids]
    items = []
    if not vid_ids:
        return []
    shot_ids = [s.id for s in db.query(Shot).filter(Shot.video_id.in_(vid_ids)).all()]
    if not shot_ids:
        return []
    trs = db.query(Tracklet).filter(Tracklet.shot_id.in_(shot_ids)).all()
    for t in trs:
        ia = db.query(IdentityAssignment).filter(
            IdentityAssignment.tracklet_id == t.id).first()
        if not ia:
            continue
        # Filter by status (default: pending only)
        if status != "all" and ia.review_status and ia.review_status != status:
            continue
        s = db.get(Shot, t.shot_id)
        o = (db.get(FaceObservation, t.best_face_observation_id)
             if t.best_face_observation_id else None)
        # reasons
        reasons = []
        if ia.review_status == "pending":
            reasons.append("pending_review")
        if o and o.quality_score < 0.45:
            reasons.append("low_quality")
        if o and o.occlusion_score > 0.7:
            reasons.append("possible_occlusion")
        if ia.confidence and ia.confidence < 0.6:
            reasons.append("low_confidence")
        from ..models import Character
        char = db.get(Character, ia.character_id) if ia.character_id else None
        if char and char.character_code == "UNKNOWN":
            reasons.append("unknown")
        gate_reason = None
        if ia.note and ia.note.startswith("reference_gate:"):
            gate_reason = ia.note.split(":", 1)[1]
            reasons.append(f"gate:{gate_reason}")
        if gate_reason:
            reasons = [f"gate:{gate_reason}"] + [r for r in reasons if not r.startswith("gate:")]
        items.append({
            "tracklet_id": t.id, "shot_number": s.shot_number if s else None,
            "timecode_start": s.timecode_start if s else None,
            "character_id": ia.character_id,
            "character_name": char.display_name if char else None,
            "character_code": char.character_code if char else None,
            "identity_confidence": ia.confidence,
            "review_status": ia.review_status,
            "assignment_source": ia.assignment_source,
            "gate_reason": gate_reason,
            "face_crop_path": o.face_crop_path if o else None,
            "quality_score": o.quality_score if o else None,
            "blur_score": o.blur_score if o else None,
            "occlusion_score": o.occlusion_score if o else None,
            "reasons": reasons,
        })
    return items
