import sys, cv2, numpy as np
sys.path.insert(0, "/home/ponky_re6000/character-identity-board/backend")
from app.database import SessionLocal
from app.models import Tracklet, FaceObservation
from app.face_engine import get_face_engine

db = SessionLocal()
fe = get_face_engine()
for tid in [1,2,3,4]:
    t = db.get(Tracklet, tid)
    best = db.get(FaceObservation, t.best_face_observation_id)
    print(f"--- tracklet {tid} (shot {t.shot_id}) best_obs={best.id if best else None} q={best.quality_score if best else 0:.3f} ---")
    if best and best.face_crop_path:
        crop = cv2.imread(best.face_crop_path)
        print(f"   crop path: {best.face_crop_path}")
        print(f"   crop size: {crop.shape if crop is not None else 'None'}")
        if crop is not None:
            faces = fe.detect_faces(crop)
            print(f"   faces in crop: {len(faces)}")
            if faces:
                e = np.asarray(fe.get_embedding(crop, faces[0]['bbox'])).reshape(-1)
                print(f"   fresh embed[:6]: {[round(x,3) for x in e[:6]]}")
