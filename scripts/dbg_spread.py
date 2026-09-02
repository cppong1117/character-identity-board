import sys
sys.path.insert(0, "/home/ponky_re6000/character-identity-board/backend")
import numpy as np
from app.database import SessionLocal
from app.models import Tracklet, FaceObservation

db = SessionLocal()
# For each tracklet, examine observation embedding spread + quality
for tid in [1, 2, 3, 4]:
    obs = db.query(FaceObservation).filter(FaceObservation.tracklet_id == tid).all()
    embs = []
    for o in obs:
        if o.embedding:
            embs.append((o.frame_number, o.quality_score, np.frombuffer(o.embedding, dtype=np.float32).reshape(-1)))
    if not embs:
        continue
    # sort by quality desc, take top 5, compute pairwise sim & mean
    embs_sorted = sorted(embs, key=lambda x: -x[1])
    top = [e for _, _, e in embs_sorted[:8]]
    print(f"--- tracklet {tid}: {len(embs)} obs ---")
    # pairwise sim among top-quality
    for i in range(min(4, len(top))):
        for j in range(i+1, min(4, len(top))):
            c = float(top[i] @ top[j] / (np.linalg.norm(top[i])*np.linalg.norm(top[j])))
            print(f"  top{i}-top{j}: {c:.3f}")
    # centroid of top-3
    mean_top = np.mean(np.stack([top[0], top[1%len(top)], top[2%len(top)]]), axis=0)
    print("  mean_top norm:", round(float(np.linalg.norm(mean_top)),3))
