#!/usr/bin/env bash
# Character Identity Board V0.1 - backend dependency install (vim: set by us)
# Stack: FastAPI + SQLAlchemy + PySceneDetect + OpenCV(YuNet/SFace) + scikit-learn(HDBSCAN) + torch(CPU/CUDA)
set -e
cd ~/character-identity-board
. .venv/bin/activate
export PATH="/home/ponky_re6000/.local/bin:$PATH"
export HF_HOME=~/character-identity-board-data/cache/hf

echo "=== [1/5] web/api/orm ==="
pip install --quiet \
    fastapi "uvicorn[standard]" pydantic pydantic-settings \
    sqlalchemy alembic python-multipart websockets "httpx" \
    python-jose passlib

echo "=== [2/5] num/vision ==="
pip install --quiet numpy opencv-python-headless pillow \
    scikit-learn scikit-image hdbscan tqdm pymupdf

echo "=== [3/5] PySceneDetect ==="
pip install --quiet scenedetect[opencv]

echo "=== [4/5] torch (linux x86_64 default wheel includes CUDA runtime) ==="
pip install --quiet torch torchvision || pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "=== [5/5] pytest + test util ==="
pip install --quiet pytest pytest-asyncio aiofiles

echo "=== DONE ==="
. .venv/bin/activate
python - <<'PY'
import numpy, cv2, scenedetect, sklearn
print("numpy", numpy.__version__)
print("cv2", cv2.__version__, "| detectors:", hasattr(cv2, "FaceDetectorYN"))
print("scenedetect", scenedetect.__version__)
print("sklearn", sklearn.__version__)
try:
    import torch
    print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(), "cuda_ver", torch.version.cuda)
except Exception as e:
    print("torch optional:", e)
PY
