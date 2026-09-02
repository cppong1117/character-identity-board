import sys
sys.path.insert(0, "/home/ponky_re6000/character-identity-board/backend")
from app.clustering import cluster_hdbscan
import numpy as np
from app.database import SessionLocal
from app.models import Tracklet, FaceObservation, IdentityAssignment

# Load the 4 tracklet best embeddings from DB
db = SessionLocal()
trs = db.query(Tracklet).filter(Tracklet.shot_id.in_([1,2,3,4])).all()
for t in trs:
    best = db.get(FaceObservation, t.best_face_observation_id)
    if best and best.embedding:
        e = np.frombuffer(best.embedding, dtype=np.float32).reshape(-1)
        print(f"tracklet {t.id} (shot {t.shot_id}): norm={np.linalg.norm(e):.3f}")
print("---")
ems = []
for t in trs:
    best = db.get(FaceObservation, t.best_face_observation_id)
    if best and best.embedding:
        ems.append(np.frombuffer(best.embedding, dtype=np.float32).reshape(-1))
print("n embeddings:", len(ems))
labels = cluster_hdbscan(ems, min_cluster_size=2)
print("HDBSCAN labels:", labels)
# pairwise cosine
for i in range(len(ems)):
    for j in range(i+1, len(ems)):
        c = float(ems[i] @ ems[j] / (np.linalg.norm(ems[i])*np.linalg.norm(ems[j])))
        print(f"  e{i}-e{j}: {c:.4f}")
