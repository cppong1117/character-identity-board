import sys, cv2, numpy as np
sys.path.insert(0, "/home/ponky_re6000/character-identity-board/backend")
from pathlib import Path
from app.face_engine import get_face_engine, FaceEngine
from app.processors import ShotProcessor
from app.shot_detection import detect_shots, shots_to_frames
from app.ffmpeg_utils import probe_video

vp = "/home/ponky_re6000/character-identity-board-data/cache/synth2/testA_conversation.mp4"
meta = probe_video(vp)
shots = detect_shots(vp)
fr = shots_to_frames(shots, meta["fps"])
print("shots:", len(shots))
fe = get_face_engine()
d = Path("/tmp/cib_checkA")
import shutil
if d.exists(): shutil.rmtree(d)
sp = ShotProcessor(fe, d)
truth = {0:"lena",1:"messi",2:"lena",3:"messi"}
reps = {}
for i in range(len(shots)):
    res = sp.process(vp, shots[i], i, fr[i], fps=meta["fps"])
    for tr in res["tracklets"]:
        obs = tr["observations"]
        vals = [(o["quality_score"], np.asarray(o["embedding"]).reshape(-1)) for o in obs if o.get("embedding") is not None]
        if vals:
            vs = sorted(vals, key=lambda x:-x[0])[:3]
            reps[i] = {"emb": np.mean(np.stack([e for _,e in vs]),axis=0), "n": len(vals)}
print("n tracklets:", len(reps))
for i in range(len(shots)):
    for j in range(i+1, len(shots)):
        a,b = reps[i]["emb"], reps[j]["emb"]
        c = float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)))
        tag = "SAME" if truth[i]==truth[j] else "DIFF"
        print(f"  shot{i}({truth[i]}) vs shot{j}({truth[j]}): {c:.4f} [{tag}]")
