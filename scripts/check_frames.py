import sys, cv2, numpy as np
sys.path.insert(0, "/home/ponky_re6000/character-identity-board/backend")
from pathlib import Path
from app.face_engine import get_face_engine
import shutil

vp = "/home/ponky_re6000/character-identity-board-data/cache/synth2/testA_conversation.mp4"
fe = get_face_engine()
cap = cv2.VideoCapture(vp)
print("cap opened:", cap.isOpened(), "frames:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Read frames at midpoints of each shot
for si,(f0,f1) in enumerate([(0,149),(150,299),(300,449),(450,599)]):
    idx = (f0+f1)//2
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    faces = fe.detect_faces(frame)
    tgt = Path(f"/tmp/cib_frames/shot{si}.jpg")
    tgt.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tgt), frame)
    print(f"shot{si} frame{idx}: faces={len(faces)}", [ [int(v) for v in f['bbox']] for f in faces])
    if faces:
        # embed
        e = fe.get_embedding(frame, faces[0]["bbox"])
        print(f"   emb[:6]", [round(x,3) for x in e.reshape(-1)[:6]])
cap.release()
