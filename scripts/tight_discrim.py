import sys, cv2, numpy as np
sys.path.insert(0, "/home/ponky_re6000/character-identity-board/backend")
from app.face_engine import FaceEngine
fe = FaceEngine(); fe._init()

def norm_emb(img, bbox):
    b = np.asarray(bbox, dtype=np.float32).reshape(-1)
    al = fe.sface.alignCrop(img, b)
    ft = np.asarray(fe.sface.feature(al)).reshape(-1).astype(np.float32)
    n = np.linalg.norm(ft)+1e-12
    return ft/n

def top_face(img):
    h,w = img.shape[:2]; fe.yunet.setInputSize((w,h))
    ok,f = fe.yunet.detect(img)
    if not ok or f is None or len(f)==0: return None
    f = sorted(f, key=lambda r: -r[-1])[0]
    return f[:4]

f1 = cv2.imread("/home/ponky_re6000/character-identity-board-data/cache/test_face.jpg")
f2 = cv2.imread("/home/ponky_re6000/character-identity-board-data/cache/test_face2.jpg")

lf = top_face(f1); mf = top_face(f2)
print("lena bbox", [int(v) for v in lf], "messi bbox", [int(v) for v in mf])

# full-res embeddings
e_l_full = norm_emb(f1, lf)
e_m_full = norm_emb(f2, mf)
print("lena full [:4]", [round(x,3) for x in e_l_full[:4]])
print("messi full [:4]", [round(x,3) for x in e_m_full[:4]])
print("lena-vs-messi FULL cosine:", round(float(e_l_full@e_m_full),4))

# at 50px likeness (downscale whole frame so face ~50px)
for target in [100, 60, 50]:
    for name, im, fb in [("lena",f1,lf), ("messi",f2,mf)]:
        scale = target / max(fb[2], fb[3])
        small = cv2.resize(im, (int(im.shape[1]*scale), int(im.shape[0]*scale)))
        fb_s = [fb[0]*scale, fb[1]*scale, fb[2]*scale, fb[3]*scale]
        e = norm_emb(small, fb_s)
        c_full = round(float(e@e_l_full),3)
        c_messi = round(float(e@e_m_full),3)
        print(f"  {name}~{target}px: cos-lena={c_full} cos-messi={c_messi}")
