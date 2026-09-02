import sys, cv2, numpy as np
sys.path.insert(0, "/home/ponky_re6000/character-identity-board/backend")
from app.face_engine import FaceEngine
fe = FaceEngine(); fe._init()

def norm(img, bbox):
    b = np.asarray(bbox, np.float32).reshape(-1)
    ft = np.asarray(fe.sface.feature(fe.sface.alignCrop(img, b))).reshape(-1).astype(np.float32)
    n = np.linalg.norm(ft)+1e-12
    return ft/n

def face(img):
    h,w=img.shape[:2]; fe.yunet.setInputSize((w,h)); ok,f=fe.yunet.detect(img)
    f = sorted(f, key=lambda r:-r[-1])[0] if (ok and f is not None and len(f)) else None
    return f[:4] if f is not None else None

A = cv2.imread("/home/ponky_re6000/character-identity-board-data/cache/test_face.jpg")
B = cv2.imread("/home/ponky_re6000/character-identity-board-data/cache/test_face2.jpg")
fa, fb = face(A), face(B)
print("faceA", [int(v) for v in fa], "area", int(fa[2]*fa[3]))
print("faceB", [int(v) for v in fb], "area", int(fb[2]*fb[3]))

eA = norm(A, fa); eB0 = norm(B, fb); eB1 = norm(B, fb)
print("A vs B:", round(float(eA@eB0),4))
print("B vs B (repeat):", round(float(eB0@eB1),4))  # determinism
# consistency when the SAME person's image is scaled to realistic sizes
for s in [0.6, 1.0, 1.8, 2.5]:
    for name, im, ff in [("A",A,fa),("B",B,fb)]:
        scaled = cv2.resize(im, (int(im.shape[1]*s), int(im.shape[0]*s)))
        fb_s = [v*s for v in ff]
        es = norm(scaled, fb_s)
        cA = round(float(es@eA),3); cB = round(float(es@eB0),3)
        print(f"  {name}@x{s}: cosA={cA} cosB={cB}")
