import sys, cv2, numpy as np
sys.path.insert(0, "/home/ponky_re6000/character-identity-board/backend")
from app.face_engine import get_face_engine

fe = get_face_engine()
face2 = "/home/ponky_re6000/character-identity-board-data/cache/test_face2.jpg"
face1 = "/home/ponky_re6000/character-identity-board-data/cache/test_face.jpg"

def rv(e):
    e = np.asarray(e).reshape(-1)
    return e / (np.linalg.norm(e)+1e-12)

for name, path in [("lena", face1), ("messi", face2)]:
    img = cv2.imread(path)
    fb = fe.detect_faces(img)[0]["bbox"]
    e_full = rv(fe.get_embedding(img, fb))
    print(f"--- {name}: face {int(fb[2])}x{int(fb[3])} ---")
    for sz in [160, 120, 90, 64, 48, 40]:
        scale = sz / max(fb[2], fb[3])
        small = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
        fb_s = [fb[0]*scale, fb[1]*scale, fb[2]*scale, fb[3]*scale]
        try:
            e = rv(fe.get_embedding(small, fb_s))
            c = float(e_full @ e)
            print(f"  face~{sz}px sim-to-full: {c:.4f}")
        except Exception as ex:
            print(f"  face~{sz}px: FAIL {str(ex)[:40]}")

img1=cv2.imread(face1); img2=cv2.imread(face2)
a=rv(fe.get_embedding(img1, fe.detect_faces(img1)[0]["bbox"]))
b=rv(fe.get_embedding(img2, fe.detect_faces(img2)[0]["bbox"]))
print("lena vs messi full-res:", round(float(a@b),4))
