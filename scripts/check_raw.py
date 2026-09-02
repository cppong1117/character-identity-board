import sys, cv2, numpy as np
sys.path.insert(0, "/home/ponky_re6000/character-identity-board/backend")
from app.face_engine import FaceEngine
import app.face_engine as FE

fe = FaceEngine()
fe._init()
sface, yunet = fe.sface, fe.yunet
face1 = "/home/ponky_re6000/character-identity-board-data/cache/test_face.jpg"
face2 = "/home/ponky_re6000/character-identity-board-data/cache/test_face2.jpg"

for name, path in [("lena", face1), ("messi", face2)]:
    img = cv2.imread(path)
    h,w = img.shape[:2]
    yunet.setInputSize((w,h))
    ok, faces = yunet.detect(img)
    fb = faces[0][:4]
    aligned = sface.alignCrop(img, fb.astype(np.float32))
    feat = sface.feature(aligned)
    print(f"--- {name}: feature shape={feat.shape} dtype={feat.dtype}")
    print("   feat[:8]:", np.asarray(feat).reshape(-1)[:8])
    print("   norm:", float(np.linalg.norm(np.asarray(feat).reshape(-1))))

print("\n=== direct bytes comparison ===")
a = fe.get_embedding(cv2.imread(face1), fe.detect_faces(cv2.imread(face1))[0]['bbox'])
b = fe.get_embedding(cv2.imread(face2), fe.detect_faces(cv2.imread(face2))[0]['bbox'])
print("a (lena) shape", np.asarray(a).shape, "b (messi) shape", np.asarray(b).shape)
print("a[:6]", np.asarray(a).reshape(-1)[:6])
print("b[:6]", np.asarray(b).reshape(-1)[:6])
