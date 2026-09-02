import sys, cv2, numpy as np
sys.path.insert(0, "/home/ponky_re6000/character-identity-board/backend")
from app.face_engine import FaceEngine
fe = FaceEngine(); fe._init()
img = cv2.imread("/home/ponky_re6000/character-identity-board-data/cache/test_face.jpg")
h, w = img.shape[:2]
fe.yunet.setInputSize((w, h))
ok, faces = fe.yunet.detect(img)
yu_row = faces[0][:4]                    # raw YuNet row (float32, 4,)
det_dict = fe.detect_faces(img)[0]       # via detect_faces
py_list = det_dict["bbox"]               # <-- list of python floats

print("yu_row:", yu_row, yu_row.dtype, yu_row.shape)
print("py_list:", py_list, type(py_list))

def emb(bbox_arr):
    al = fe.sface.alignCrop(img, bbox_arr)
    ft = fe.sface.feature(al)
    return np.asarray(ft).reshape(-1)[:6]

e_yu     = emb(yu_row)                                    # manual path (worked)
e_yu_f32 = emb(np.asarray(yu_row, dtype=np.float32))       # same as manual
e_list   = emb(py_list)                                    # get_embedding-style (list)
e_listf  = emb(np.asarray(py_list, dtype=np.float32))      # get_embedding style via np

print("from yu_row  :", [round(x,3) for x in e_yu])
print("from yu_f32  :", [round(x,3) for x in e_yu_f32])
print("from list    :", [round(x,3) for x in e_list])
print("from listf32 :", [round(x,3) for x in e_listf])
print()
print("alignCrop sizes:", np.asarray(fe.sface.alignCrop(img, yu_row)).shape, "/", np.asarray(fe.sface.alignCrop(img, py_list)).shape)
