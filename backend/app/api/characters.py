"""Character endpoints: list / create / patch / merge / split / mark unknown /
reference / history."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import (
    Character, IdentityAssignment, Tracklet, Shot, FaceObservation,
    ReviewAction, Project,
)
from ..schemas import (
    CharacterCreate, CharacterPatch, CharacterOut,
    CharacterMergeRequest, CharacterSplitRequest, ReviewActionOut,
)
from .. import corrections

log = logging.getLogger("cib.api.characters")
router = APIRouter(tags=["characters"])


def _char_out(db: Session, c: Character) -> CharacterOut:
    # counts
    tr_count = (db.query(IdentityAssignment)
                .filter(IdentityAssignment.character_id == c.id).count())
    # shots: distinct shot ids via tracklets of assigned tracklets
    trs = (db.query(Tracklet)
           .join(IdentityAssignment, IdentityAssignment.tracklet_id == Tracklet.id)
           .filter(IdentityAssignment.character_id == c.id).all())
    shot_ids = {t.shot_id for t in trs}
    pending = (db.query(IdentityAssignment)
               .filter(IdentityAssignment.character_id == c.id,
                       IdentityAssignment.review_status == "pending").count())
    confs = [ia.confidence for ia in (db.query(IdentityAssignment)
             .filter(IdentityAssignment.character_id == c.id).all()) if ia.confidence]
    avg = sum(confs) / len(confs) if confs else 0.0
    return CharacterOut(
        id=c.id, project_id=c.project_id, display_name=c.display_name,
        character_code=c.character_code, reference_image=c.reference_image,
        status=c.status, created_by=c.created_by,
        tracklet_count=tr_count, shot_count=len(shot_ids),
        avg_confidence=round(avg, 4), pending_review=pending,
    )


@router.get("/projects/{project_id}/characters", response_model=list[CharacterOut])
def list_characters(project_id: int, db: Session = Depends(get_db)):
    chars = db.query(Character).filter(Character.project_id == project_id) \
        .order_by(Character.id).all()
    return [_char_out(db, c) for c in chars]


@router.post("/characters", response_model=CharacterOut, status_code=201)
def create_character(body: CharacterCreate, db: Session = Depends(get_db)):
    p = db.get(Project, body.project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    code = body.character_code or f"MAN{_next_code(db, body.project_id)}"
    c = Character(
        project_id=body.project_id, display_name=body.display_name,
        character_code=code, reference_image=body.reference_image,
        status="manual", created_by="manual")
    db.add(c)
    db.commit()
    db.refresh(c)
    return _char_out(db, c)


def _next_code(db: Session, project_id: int) -> int:
    existing = db.query(Character.character_code).filter(
        Character.character_code.like("MAN%")).all()
    maxn = 0
    for (code,) in existing:
        try:
            maxn = max(maxn, int(code[3:]))
        except ValueError:
            pass
    return maxn + 1


@router.patch("/characters/{char_id}", response_model=CharacterOut)
def patch_character(char_id: int, body: CharacterPatch, db: Session = Depends(get_db)):
    c = db.get(Character, char_id)
    if not c:
        raise HTTPException(404, "Character not found")
    if body.display_name is not None:
        corrections.rename_character(db, char_id, body.display_name)
    if body.reference_image is not None:
        corrections.set_reference_image(db, char_id, body.reference_image)
    if body.status is not None:
        c.status = body.status
        db.commit()
    db.refresh(c)
    return _char_out(db, c)


@router.post("/characters/merge", response_model=CharacterOut)
def merge(body: CharacterMergeRequest, db: Session = Depends(get_db)):
    tgt = corrections.merge_characters(
        db, body.source_character_id, body.target_character_id)
    return _char_out(db, tgt)


@router.post("/characters/{char_id}/split", response_model=dict)
def split(body: CharacterSplitRequest, char_id: int, db: Session = Depends(get_db)):
    orig, new = corrections.split_character(
        db, char_id, body.tracklet_ids, new_name=body.new_character_name)
    return {"original_character_id": orig.id,
            "new_character_id": new.id,
            "moved_tracklets": len(body.tracklet_ids)}


@router.post("/characters/{char_id}/mark-unknown", response_model=CharacterOut)
def mark_unknown(char_id: int, db: Session = Depends(get_db)):
    c = corrections.mark_unknown(db, char_id)
    return _char_out(db, c)


@router.delete("/characters/{char_id}", status_code=204)
def delete_char(char_id: int, db: Session = Depends(get_db)):
    corrections.delete_character(db, char_id)
    return None


@router.get("/characters/{char_id}/observations")
def char_observations(char_id: int, limit: int = 48, offset: int = 0,
                      include_excluded: bool = False, db: Session = Depends(get_db)):
    """Face observations for a character, ordered by shot.

    Returns observation id so UI can exclude/select individual faces.
    Default: skip excluded faces. Use limit/offset for paging.
    """
    c = db.get(Character, char_id)
    if not c:
        raise HTTPException(404, "Character not found")
    q = (db.query(IdentityAssignment, Tracklet, Shot, FaceObservation)
         .join(Tracklet, IdentityAssignment.tracklet_id == Tracklet.id)
         .join(Shot, Tracklet.shot_id == Shot.id)
         .join(FaceObservation, Tracklet.best_face_observation_id == FaceObservation.id)
         .filter(IdentityAssignment.character_id == char_id))
    if not include_excluded:
        q = q.filter((FaceObservation.excluded.is_(False)) | (FaceObservation.excluded.is_(None)))
    assoc = q.order_by(Shot.shot_number).offset(max(0, offset)).limit(max(1, min(limit, 500))).all()
    rows = []
    for ia, t, s, o in assoc:
        rows.append({
            "id": o.id,
            "shot_number": s.shot_number,
            "tracklet_id": t.id,
            "timecode_start": s.timecode_start,
            "timecode_end": s.timecode_end,
            "face_crop_path": o.face_crop_path,
            "portrait_crop_path": o.portrait_crop_path,
            "body_crop_path": o.body_crop_path,
            "face_bbox": o.face_bbox,
            "quality_score": o.quality_score,
            "blur_score": o.blur_score,
            "identity_confidence": ia.confidence,
            "review_status": ia.review_status,
            "excluded": bool(o.excluded),
            "exclude_reason": o.exclude_reason,
        })
    return rows


@router.get("/projects/{project_id}/review-actions", response_model=list[ReviewActionOut])
def review_actions(project_id: int, db: Session = Depends(get_db)):
    acts = db.query(ReviewAction).filter(ReviewAction.project_id == project_id) \
        .order_by(ReviewAction.id.desc()).all()
    return [ReviewActionOut(
        id=a.id, project_id=a.project_id, action_type=a.action_type,
        source_character_id=a.source_character_id,
        target_character_id=a.target_character_id, tracklet_id=a.tracklet_id,
        before_state=a.before_state, after_state=a.after_state,
        created_at=a.created_at) for a in acts]
