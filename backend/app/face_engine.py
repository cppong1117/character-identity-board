"""Face detection (YuNet) and face embedding (SFace) via OpenCV DNN.

Licensing: YuNet MIT, SFace Apache-2.0, both from opencv_zoo (Apache-2.0 repo).
Deliberately avoids InsightFace (repo + weights have no clear commercial license),
satisfying the MISSION License Gate cleanly.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import settings


class FaceEngine:
    """Wraps YuNet + SFace OpenCV DNN models with lazy init + CUDA-CPU fallback."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir or settings.model_dir)
        self.yunet = None
        self.sface = None
        self.det_dims = (320, 320)     # detection input size (w, h)
        self._backend = None

    def _init(self):
        if self.yunet is not None:
            return
        yunet_path = self.model_dir / "face_detection_yunet.onnx"
        sface_path = self.model_dir / "face_recognition_sface.onnx"
        if not yunet_path.exists():
            raise FileNotFoundError(
                f"YuNet model missing at {yunet_path}. Download from opencv_zoo.")
        if not sface_path.exists():
            raise FileNotFoundError(f"SFace model missing at {sface_path}.")

        # Prefer CUDA backend if available, else CPU.
        # NOTE: OpenCV 5 new graph engine warns CUDA backend not yet supported and
        # silently falls back to CPU; retain CUDA attempt for future engine versions.
        backends = []
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                backends.append((cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA))
        except Exception:
            pass
        backends.append((cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU))
        used = None
        for b, t in backends:
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    str(yunet_path), "", self.det_dims,
                    score_threshold=0.5, nms_threshold=0.3, top_k=5000,
                    backend_id=b, target_id=t)
                self.sface = cv2.FaceRecognizerSF.create(
                    str(sface_path), "", backend_id=b, target_id=t)
                used = (b, t)
                # quick self-check: verify it can load an image
                break
            except Exception:
                self.yunet = None
                self.sface = None
                continue
        if self.yunet is None:
            raise RuntimeError("Could not init any CV backend for face engine.")
        self._backend = used

    @property
    def backend_label(self) -> str:
        return self._backend

    def detect_faces(self, img_bgr: np.ndarray) -> list[dict]:
        """Return list of face dicts: {bbox:[x,y,w,h], landmarks, score}."""
        self._init()
        h, w = img_bgr.shape[:2]
        self.yunet.setInputSize((w, h))
        ok, faces = self.yunet.detect(img_bgr)
        out = []
        if ok and faces is not None and len(faces) > 0:
            for f in faces:
                x, y, ww, hh = f[:4]
                score = float(f[-1])
                lm = f[4:14].reshape(5, 2)
                out.append({
                    "bbox": [float(x), float(y), float(ww), float(hh)],
                    "landmarks": lm.tolist(),
                    "score": score,
                })
        return out

    @staticmethod
    def crop(img_bgr: np.ndarray, bbox, margin: float = 0.0,
             clamp_to_frame: bool = True) -> np.ndarray:
        x, y, w, h = [float(v) for v in bbox]
        if margin > 0:
            mx = w * margin; my = h * margin
            x -= mx; y -= my; w += 2 * mx; h += 2 * my
        H, W = img_bgr.shape[:2]
        x0 = max(0, int(x)); y0 = max(0, int(y))
        x1 = min(W, int(x + w)); y1 = min(H, int(y + h))
        if x1 <= x0 or y1 <= y0:
            return img_bgr.copy()
        return img_bgr[y0:y1, x0:x1].copy()

    def get_embedding(self, img_bgr: np.ndarray, detection: dict) -> Optional[np.ndarray]:
        """Return 128-d L2-normalized SFace embedding.

        ``detection`` must contain: bbox:[x,y,w,h], landmarks:5x2 list, score:float.
        This ensures SFace ``alignCrop`` receives proper 5-point landmarks for
        face alignment instead of a raw bbox (which produced black images).
        """
        self._init()
        try:
            bbox = detection['bbox']
            lm = detection['landmarks']  # list of 5 [x,y] points
            score = detection['score']
            face_arr = np.asarray(
                bbox + [coord for pt in lm for coord in pt] + [score],
                dtype=np.float32,
            )
            aligned = self.sface.alignCrop(img_bgr, face_arr)
            feat = self.sface.feature(aligned)
            feat = np.asarray(feat).reshape(-1).astype(np.float32)
            n = np.linalg.norm(feat) + 1e-12
            return (feat / n).reshape(1, -1) if n else feat.reshape(1, -1)
        except Exception:
            return None

    def get_embedding_from_bbox(self, img_bgr: np.ndarray, face_bbox) -> Optional[np.ndarray]:
        """Detect face at bbox and embed with proper landmark alignment."""
        self._init()
        detections = self.detect_faces(img_bgr)
        best, best_iou = None, 0.0
        for det in detections:
            db = det['bbox']
            ix = max(0, min(face_bbox[0]+face_bbox[2], db[0]+db[2]) - max(face_bbox[0], db[0]))
            iy = max(0, min(face_bbox[1]+face_bbox[3], db[1]+db[3]) - max(face_bbox[1], db[1]))
            inter = ix * iy
            union = face_bbox[2]*face_bbox[3] + db[2]*db[3] - inter
            iou = inter / (union + 1e-6)
            if iou > best_iou:
                best_iou = iou
                best = det
        if best and best_iou > 0.3:
            return self.get_embedding(img_bgr, best)
        return None

    @staticmethod
    def similarity(e1: np.ndarray, e2: np.ndarray) -> float:
        """Cosine similarity from two L2-normalized embeddings (shape (1,128) or (128,))."""
        a = e1.reshape(-1)
        b = e2.reshape(-1)
        ca = float(np.dot(a, b))
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        return ca / (na * nb + 1e-12)

    def best_observation(self, embeddings: list) -> Optional[np.ndarray]:
        """Return the embedding that is most representative (median of pairwise
        similarities) from a set of embeddings for one tracklet."""
        if not embeddings:
            return None
        if len(embeddings) == 1:
            return np.asarray(embeddings[0]).reshape(1, -1)
        em = np.stack([np.asarray(e).reshape(-1) for e in embeddings])
        sims = em @ em.T          # normalized => cosine sim matrix
        # exclude self-diagonal
        np.fill_diagonal(sims, -1)
        avg = sims.mean(axis=1)
        return em[int(np.argmax(avg))].reshape(1, -1).copy()

    @staticmethod
    def aggregate_embeddings(embeddings: list, top_k: int = 5) -> Optional[np.ndarray]:
        """Return a normalized centroid over a bounded set of embeddings.

        Multiple frames are more stable than a single reference crop when a
        speaker is talking or turns slightly between frames.
        """
        if not embeddings:
            return None
        em = np.stack([np.asarray(e, dtype=np.float32).reshape(-1)
                       for e in embeddings[:max(1, top_k)]])
        cen = np.mean(em, axis=0)
        norm = np.linalg.norm(cen) + 1e-12
        return (cen / norm).reshape(1, -1)


_engine: Optional[FaceEngine] = None


def get_face_engine() -> FaceEngine:
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine
