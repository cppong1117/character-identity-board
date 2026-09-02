import sys, cv2, numpy as np
sys.path.insert(0, "/home/ponky_re6000/character-identity-board/backend")
from app.face_engine import get_face_engine
fe = get_face_engine()
# find face sizes in each test video's shots, and test embedding consistency for clean framing
base = "/home/ponky_re6000/character-identity-board-data/benchmarks/V0.1/generated/"
import subprocess, os
for vid in ["testA_two_person_hardcuts.mp4"]:
    cap = cv2.VideoCapture(base+vid)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in [40, 60, 190, 210, 340, 360, 490, 510]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok: continue
        fs = fe.detect_faces(fr)
        desc = [(int(f['bbox'][2]), int(f['bbox'][3]), round(f['score'],2)) for f in fs]
        emb_repr = ""
        if fs:
            e = fe.get_embedding(fr, fs[0]['bbox'])
            if e is not None:
                emb_repr = f"emb[:3]={[round(float(x),3) for x in np.asarray(e).reshape(-1)[:3]]}"
        print(f"frame {idx}: faces={desc} {emb_repr}")
    cap.release()
