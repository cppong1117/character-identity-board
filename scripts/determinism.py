import sys, cv2, numpy as np
sys.path.insert(0, "/home/ponky_re6000/character-identity-board/backend")
from app.face_engine import FaceEngine
fe = FaceEngine(); fe._init()
img = cv2.imread("/home/ponky_re6000/character-identity-board-data/cache/test_face.jpg")

# Determinism check: exact same input to SFace twice.
crop = img[100:400, 100:400]
al = fe.sface.alignCrop(img, np.asarray([150.0,120.0,240.0,300.0], np.float32))
f1 = np.asarray(fe.sface.feature(al)).reshape(-1)
f2 = np.asarray(fe.sface.feature(al)).reshape(-1)
f3 = np.asarray(fe.sface.feature(al)).reshape(-1)
print("same-input twice equal:", np.array_equal(f1,f2), np.array_equal(f2,f3))
print("f1[:6]", [round(x,3) for x in f1[:6]])
print("f2[:6]", [round(x,3) for x in f2[:6]])

# Does the SAME aligned crop re-feature deterministically?
e1 = np.asarray(fe.sface.feature(fe.sface.alignCrop(img, np.asarray([150.,120.,240.,300.],np.float32)))).reshape(-1)
e2 = np.asarray(fe.sface.feature(fe.sface.alignCrop(img, np.asarray([150.,120.,240.,300.],np.float32)))).reshape(-1)
print("re-align same-bbox equal:", np.array_equal(e1,e2), "cos:", round(float(e1@e2/(np.linalg.norm(e1)*np.linalg.norm(e2))),4))

# Try a tiny perturbation of bbox -> does embedding jump wildly?
for dx in [0, 1, 3, 5, 10]:
    b = np.asarray([150.+dx,120.,240.,300.], np.float32)
    e = np.asarray(fe.sface.feature(fe.sface.alignCrop(img,b))).reshape(-1)
    c = float(e@e1/(np.linalg.norm(e)*np.linalg.norm(e1)))
    print(f"  bbox+{dx}: cos-vs-base={round(c,4)}")
