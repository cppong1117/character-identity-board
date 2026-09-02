"""Deep-dive: what does the pipeline actually store and how does clustering assign?"""
import sys, numpy as np
sys.path.insert(0, "/home/ponky_re6000/character-identity-board/backend")
from app.database import SessionLocal
from app.models import Tracklet, FaceObservation, IdentityAssignment, Shot, Character

db = SessionLocal()
# Look at ALL projects/videos currently in DB
from app.models import Project, Video
for p in db.query(Project).all():
    print(f"\n=== Project {p.id} {p.name} ===")
    for v in db.query(Video).filter(Video.project_id==p.id).all():
        print(f"  Video {v.id} {v.filename} stage={v.pipeline_stage} status={v.processing_status}")
        shots = db.query(Shot).filter(Shot.video_id==v.id).order_by(Shot.shot_number).all()
        print(f"    shots: {len(shots)}")
        # list all characters & assignments
        chars = db.query(Character).filter(Character.project_id==p.id).all()
        print(f"    characters: {[(c.id,c.display_name,c.character_code,c.status) for c in chars]}")
        for s in shots:
            trs = db.query(Tracklet).filter(Tracklet.shot_id==s.id).order_by(Tracklet.track_number).all()
            for t in trs:
                best = db.get(FaceObservation, t.best_face_observation_id)
                ia = db.query(IdentityAssignment).filter(IdentityAssignment.tracklet_id==t.id).first()
                n_obs = db.query(FaceObservation).filter(FaceObservation.tracklet_id==t.id).count()
                emb_info = "no-emb"
                if best and best.embedding:
                    e = np.frombuffer(best.embedding, dtype=np.float32)
                    emb_info = f"emb[:4]={[round(float(x),3) for x in e[:4]]} norm={float(np.linalg.norm(e)):.3f}"
                q = f"q={best.quality_score:.2f}" if best else "q=none"
                assign = f"char={ia.character_id} conf={ia.confidence if ia else '-'} src={ia.assignment_source if ia else '-'} st={ia.review_status if ia else '-'}" if ia else "no-assign"
                print(f"      shot{s.shot_number} tracklet#{t.track_number} (id {t.id}) n_obs={n_obs} {q} {emb_info} | {assign}")
