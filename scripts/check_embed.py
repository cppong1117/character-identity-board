import sys, cv2, numpy as np
sys.path.insert(0, "/home/ponky_re6000/character-identity-board/backend")
from app.face_engine import FaceEngine

fe = FaceEngine()
fe._init()
img = cv2.imread("/home/ponky_re6000/character-identity-board-data/cache/test_face.jpg")  # lena
h, w = img.shape[:2]
fe.yunet.setInputSize((w, h))
ok, faces = fe.yunet.detect(img)
fb = faces[0][:4]
print("detected face bbox:", fb, fb.dtype, fb.shape)

# path A: get_embedding
eA = fe.get_embedding(img, list(fb))
print("get_embedding output shape:", np.asarray(eA).shape)
print("get_embedding[:6]:", [round(x,3) for x in np.asarray(eA).reshape(-1)[:6]])

# path B: manual (check_raw style)
aligned = fe.sface.alignCrop(img, fb.astype(np.float32))
feat = fe.sface.feature(aligned)
print("manual feature shape:", feat.shape)
print("manual feature[:6]:", [round(x,3) for x in np.asarray(feat).reshape(-1)[:6]])

# path C: manual but via list
fb_list = np.asarray(list(fb), dtype=np.float32)
aligned2 = fe.sface.alignCrop(img, fb_list)
feat2 = fe.sface.feature(aligned2)
print("manual(list) feature[:6]:", [round(x,3) for x in np.asarray(feat2).reshape(-1)[:6]])

# Is alignCrop producing different crops?
print("alignCrop shapes:", np.asarray(aligned).shape)
