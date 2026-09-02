import sys, cv2
sys.path.insert(0, '/home/ponky_re6000/character-identity-board/backend')
from app.face_engine import get_face_engine
fe = get_face_engine()
for s in 'ABCD':
    im = cv2.imread(f'/tmp/f{s}.jpg')
    fs = fe.detect_faces(im)
    print(f'shot{s}: faces=', [(int(f['bbox'][2]), int(f['bbox'][3])) for f in fs], 'imgsz', im.shape[:2] if im is not None else None)
