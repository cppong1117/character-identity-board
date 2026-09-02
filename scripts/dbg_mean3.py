import sys
sys.path.insert(0, "/home/ponky_re6000/character-identity-board/backend")
import numpy as np
from app.database import SessionLocal
from app.models import Tracklet, FaceObservation

db = SessionLocal()
trs = db.query(Tracklet).filter(Tracklet.shot_id.in_([1,2,3,4])).all()
reps = {}
for t in trs:
    obs = db.query(FaceObservation).filter(FaceObservation.tracklet_id == t.id).all()
    vals = [(o.quality_score, np.frombuffer(o.embedding, dtype=np.float32)) for o in obs if o.embedding]
    vals_sorted = sorted(vals, key=lambda x: -x[0])[:3]
    reps[t.id] = np.mean(np.stack([e for _, e in vals_sorted]), axis=0)

print("Robust (mean-of-top3) representative, cross-shot cosine:")
ids = list(reps.keys())
shot_of = {1:1, 2:2, 3:3, 4:4}  # tracklet->shot (from earlier: t1=s1,t2=s2,t3=s3,t4=s4)
truth = {1:"lena",2:"messi",3:"lena",4:"messi"}
for i in range(len(ids)):
    for j in range(i+1, len(ids)):
        a,b = reps[ids[i]], reps[ids[j]]
        c = float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)))
        tag = "SAME" if truth[ids[i]]==truth[ids[j]] else "DIFF"
        print(f"  t{ids[i]}({truth[ids[i]]}) vs t{ids[j]}({truth[ids[j]]}): {c:.4f} [{tag}]")
