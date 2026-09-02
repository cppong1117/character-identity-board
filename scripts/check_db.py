import sys
sys.path.insert(0, "/home/ponky_re6000/character-identity-board/backend")
import numpy as np
from app.database import SessionLocal
from app.models import Tracklet, FaceObservation

db = SessionLocal()
trs = db.query(Tracklet).filter(Tracklet.shot_id.in_([1,2,3,4])).all()
got = {}
for t in trs:
    obs = db.query(FaceObservation).filter(FaceObservation.tracklet_id==t.id).all()
    # use best observation embedding (the one stored & used)
    best = db.get(FaceObservation, t.best_face_observation_id)
    e = None
    if best and best.embedding:
        e = np.frombuffer(best.embedding, dtype=np.float32)
        got[t.id] = (t.shot_id, e)

ids = list(got.keys())
truth = {1:(1,"lena"),2:(2,"messi"),3:(3,"lena"),4:(4,"messi")}
def csim(a,b):
    a=a.reshape(-1); b=b.reshape(-1)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
print("DB best-observation embeddings, cross-tracklet cosine:")
for i in range(len(ids)):
    for j in range(i+1,len(ids)):
        a=got[ids[i]][1]; b=got[ids[j]][1]
        c=csim(a,b)
        tag="SAME" if truth[ids[i]][1]==truth[ids[j]][1] else "DIFF"
        print(f"  t{ids[i]}[shot{truth[ids[i]][0]},{truth[ids[i]][1]}] vs t{ids[j]}[shot{truth[ids[j]][0]},{truth[ids[j]][1]}]: {c:.4f} [{tag}]")
