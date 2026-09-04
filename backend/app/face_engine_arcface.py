"""Face detection (YuNet) + ArcFace recognition with 5-point alignment.

Pipeline (V0.3):
  YuNet bbox + 5 landmarks
    -> similarity transform to canonical 112x112
    -> ArcFace (buffalo_l / w600k_r50.onnx)
    -> 512-d L2-normalized embedding

embedding_version:
  arcface_v1          = legacy unaligned bbox-crop resize (deprecated for identity)
  arcface_v2_aligned  = 5-point similarity aligned (this module default)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .config import settings

# InsightFace / ArcFace canonical 5-point template for 112x112 input
# order: left_eye, right_eye, nose, left_mouth, right_mouth
ARC_FACE_5PTS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

EMBEDDING_VERSION_V1 = "arcface_v1"
EMBEDDING_VERSION_V2 = "arcface_v2_aligned"


class FaceEngineArcFace:
    """YuNet detection + 5-point aligned ArcFace recognition."""

    def __init__(self, model_dir: Optional[Path] = None, arcface_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir or settings.model_dir)
        self.arcface_dir = Path(arcface_dir or Path.home() / ".insightface/models/buffalo_l")
        self.yunet = None
        self.arcface_session = None
        self.det_dims = (320, 320)
        self._backend = None
        self.embedding_version = EMBEDDING_VERSION_V2
        self.canonical_size = 112

    def _init_yunet(self):
        if self.yunet is not None:
            return

        yunet_path = self.model_dir / "face_detection_yunet.onnx"
        if not yunet_path.exists():
            raise FileNotFoundError(f"YuNet model missing at {yunet_path}")

        backends = []
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                backends.append((cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA))
        except Exception:
            pass
        backends.append((cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU))

        for b, t in backends:
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    str(yunet_path),
                    "",
                    self.det_dims,
                    score_threshold=0.5,
                    nms_threshold=0.3,
                    top_k=5000,
                    backend_id=b,
                    target_id=t,
                )
                self._backend = (b, t)
                break
            except Exception:
                self.yunet = None
                continue

        if self.yunet is None:
            raise RuntimeError("Could not init YuNet face detector")

    def _init_arcface(self):
        if self.arcface_session is not None:
            return

        import onnxruntime as ort

        model_path = self.arcface_dir / "w600k_r50.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"ArcFace model missing at {model_path}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        self.arcface_session = ort.InferenceSession(
            str(model_path),
            providers=[p for p in providers if p in available],
        )
        print(f"  ArcFace loaded: {model_path} providers={self.arcface_session.get_providers()}", flush=True)

    def detect_faces(self, img_bgr: np.ndarray) -> list[dict]:
        """Return list of face dicts: {bbox:[x,y,w,h], landmarks[[x,y]*5], score}."""
        self._init_yunet()
        h, w = img_bgr.shape[:2]
        self.yunet.setInputSize((w, h))
        ok, faces = self.yunet.detect(img_bgr)
        out = []
        if ok and faces is not None and len(faces) > 0:
            for f in faces:
                x, y, ww, hh = f[:4]
                score = float(f[-1])
                lm = f[4:14].reshape(5, 2).astype(np.float32)
                out.append(
                    {
                        "bbox": [float(x), float(y), float(ww), float(hh)],
                        "landmarks": lm.tolist(),
                        "score": score,
                    }
                )
        return out

    @staticmethod
    def crop(img_bgr: np.ndarray, bbox, margin: float = 0.0, clamp_to_frame: bool = True) -> np.ndarray:
        x, y, w, h = [float(v) for v in bbox]
        if margin > 0:
            mx = w * margin
            my = h * margin
            x -= mx
            y -= my
            w += 2 * mx
            h += 2 * my
        H, W = img_bgr.shape[:2]
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(W, int(x + w))
        y1 = min(H, int(y + h))
        if x1 <= x0 or y1 <= y0:
            return img_bgr.copy()
        return img_bgr[y0:y1, x0:x1].copy()

    @staticmethod
    def landmarks_array(detection: dict) -> Optional[np.ndarray]:
        lm = detection.get("landmarks")
        if lm is None:
            return None
        arr = np.asarray(lm, dtype=np.float32).reshape(-1, 2)
        if arr.shape != (5, 2):
            return None
        if not np.isfinite(arr).all():
            return None
        return arr

    @staticmethod
    def validate_landmark_geometry(
        landmarks: np.ndarray,
        bbox: Optional[list] = None,
        img_shape: Optional[tuple] = None,
    ) -> dict[str, Any]:
        """Basic geometry checks (Phase 9 precursor). Does not change thresholds."""
        lm = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
        le, re, nose, lmth, rmth = lm
        reasons: list[str] = []

        # eye order: left eye should be left of right eye in image coords
        if le[0] >= re[0]:
            reasons.append("eye_order")

        eye_dist = float(np.linalg.norm(re - le))
        if eye_dist < 1e-3:
            reasons.append("eye_dist_zero")

        # nose roughly between eyes horizontally
        mid_x = 0.5 * (le[0] + re[0])
        if eye_dist > 1e-3 and abs(nose[0] - mid_x) > eye_dist * 0.85:
            reasons.append("nose_not_between_eyes")

        # nose below eyes
        eye_y = 0.5 * (le[1] + re[1])
        if nose[1] < eye_y - 2:
            reasons.append("nose_above_eyes")

        # mouth below nose
        mouth_y = 0.5 * (lmth[1] + rmth[1])
        if mouth_y < nose[1] - 2:
            reasons.append("mouth_above_nose")

        # mouth corners order
        if lmth[0] >= rmth[0]:
            reasons.append("mouth_order")

        if bbox is not None and len(bbox) >= 4:
            x, y, w, h = [float(v) for v in bbox[:4]]
            # allow small slack outside box
            slack = 0.15 * max(w, h)
            for i, (px, py) in enumerate(lm):
                if px < x - slack or py < y - slack or px > x + w + slack or py > y + h + slack:
                    reasons.append(f"landmark_{i}_outside_bbox")
                    break
            if w > 1 and h > 1:
                ratio = h / w
                if ratio < 0.6 or ratio > 2.2:
                    reasons.append("face_ratio")

        if img_shape is not None:
            H, W = img_shape[:2]
            for i, (px, py) in enumerate(lm):
                if px < -5 or py < -5 or px > W + 5 or py > H + 5:
                    reasons.append(f"landmark_{i}_outside_image")
                    break

        return {"ok": len(reasons) == 0, "reasons": reasons}

    def align_face(
        self,
        img_bgr: np.ndarray,
        landmarks: np.ndarray,
        image_size: int = 112,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Similarity-transform face to ArcFace canonical 112x112.

        Returns (aligned_bgr, M 2x3).
        """
        src = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
        dst = ARC_FACE_5PTS_112
        if image_size != 112:
            scale = float(image_size) / 112.0
            dst = dst * scale
        M = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
        if M is None:
            # fallback: cv2.estimateAffinePartial2D with default
            M = cv2.estimateAffinePartial2D(src, dst)[0]
        if M is None:
            # last resort: center crop path caller should handle
            raise RuntimeError("similarity alignment failed")
        aligned = cv2.warpAffine(
            img_bgr,
            M,
            (image_size, image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned, M

    def _preprocess_aligned(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Aligned 112x112 BGR -> model input NCHW float32."""
        if aligned_bgr.shape[0] != 112 or aligned_bgr.shape[1] != 112:
            aligned_bgr = cv2.resize(aligned_bgr, (112, 112))
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) - 127.5) / 127.5
        chw = normalized.transpose(2, 0, 1)
        return np.expand_dims(chw, axis=0)

    def _preprocess_face_unaligned(self, img_bgr: np.ndarray) -> np.ndarray:
        """Legacy v1 path: plain resize (for A/B only)."""
        resized = cv2.resize(img_bgr, (112, 112))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) - 127.5) / 127.5
        chw = normalized.transpose(2, 0, 1)
        return np.expand_dims(chw, axis=0)

    def _run_arcface(self, preprocessed: np.ndarray) -> np.ndarray:
        self._init_arcface()
        input_name = self.arcface_session.get_inputs()[0].name
        outputs = self.arcface_session.run(None, {input_name: preprocessed})
        embedding = outputs[0].flatten().astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        return embedding.reshape(1, -1)

    def prepare_face(
        self,
        img_bgr: np.ndarray,
        detection: dict,
        require_geometry_ok: bool = False,
    ) -> dict[str, Any]:
        """Build raw_crop + aligned_crop + landmarks metadata (no embed yet)."""
        bbox = detection.get("bbox")
        if bbox is None:
            raise ValueError("detection missing bbox")
        raw_crop = self.crop(img_bgr, bbox, margin=0.1)
        lm = self.landmarks_array(detection)
        geo = None
        aligned = None
        M = None
        align_ok = False
        align_error = None
        if lm is not None:
            geo = self.validate_landmark_geometry(lm, bbox=bbox, img_shape=img_bgr.shape)
            if require_geometry_ok and not geo["ok"]:
                return {
                    "raw_crop": raw_crop,
                    "aligned_crop": None,
                    "landmarks": lm.tolist(),
                    "geometry": geo,
                    "align_ok": False,
                    "align_error": "geometry_invalid",
                    "M": None,
                }
            try:
                aligned, M = self.align_face(img_bgr, lm, image_size=self.canonical_size)
                align_ok = True
            except Exception as e:
                align_error = str(e)
                align_ok = False
        else:
            align_error = "missing_landmarks"

        return {
            "raw_crop": raw_crop,
            "aligned_crop": aligned,
            "landmarks": None if lm is None else lm.tolist(),
            "geometry": geo,
            "align_ok": align_ok,
            "align_error": align_error,
            "M": None if M is None else M.tolist(),
            "bbox": bbox,
            "score": detection.get("score"),
        }

    def get_embedding(
        self,
        img_bgr: np.ndarray,
        detection: dict,
        aligned: bool = True,
    ) -> Optional[np.ndarray]:
        """Return 512-d L2-normalized embedding.

        Default aligned=True -> arcface_v2_aligned.
        aligned=False keeps legacy v1 path for A/B tests only.
        """
        try:
            if aligned:
                prep = self.prepare_face(img_bgr, detection)
                if prep["align_ok"] and prep["aligned_crop"] is not None:
                    tensor = self._preprocess_aligned(prep["aligned_crop"])
                    return self._run_arcface(tensor)
                # fallback: legacy crop resize if landmarks fail (still better than crash)
                if prep["raw_crop"] is None or prep["raw_crop"].size == 0:
                    return None
                tensor = self._preprocess_face_unaligned(prep["raw_crop"])
                return self._run_arcface(tensor)
            # explicit v1
            bbox = detection["bbox"]
            face_crop = self.crop(img_bgr, bbox, margin=0.1)
            if face_crop.size == 0:
                return None
            tensor = self._preprocess_face_unaligned(face_crop)
            return self._run_arcface(tensor)
        except Exception:
            return None

    def get_embedding_debug(
        self,
        img_bgr: np.ndarray,
        detection: dict,
    ) -> dict[str, Any]:
        """Embedding + raw_crop + aligned_crop + landmarks for debug dumps."""
        prep = self.prepare_face(img_bgr, detection)
        emb_v2 = None
        emb_v1 = None
        if prep["align_ok"] and prep["aligned_crop"] is not None:
            emb_v2 = self._run_arcface(self._preprocess_aligned(prep["aligned_crop"]))
        if prep["raw_crop"] is not None and prep["raw_crop"].size > 0:
            emb_v1 = self._run_arcface(self._preprocess_face_unaligned(prep["raw_crop"]))
        return {
            **prep,
            "embedding_v2": emb_v2,
            "embedding_v1": emb_v1,
            "embedding_version": EMBEDDING_VERSION_V2 if emb_v2 is not None else EMBEDDING_VERSION_V1,
        }

    @staticmethod
    def similarity(e1: np.ndarray, e2: np.ndarray) -> float:
        a = e1.reshape(-1)
        b = e2.reshape(-1)
        return float(np.dot(a, b))

    def best_observation(self, embeddings: list) -> Optional[np.ndarray]:
        if not embeddings:
            return None
        if len(embeddings) == 1:
            return np.asarray(embeddings[0]).reshape(1, -1)
        em = np.stack([np.asarray(e).reshape(-1) for e in embeddings])
        sims = em @ em.T
        np.fill_diagonal(sims, -1)
        avg = sims.mean(axis=1)
        return em[int(np.argmax(avg))].reshape(1, -1).copy()

    @staticmethod
    def aggregate_embeddings(embeddings: list, top_k: int = 5) -> Optional[np.ndarray]:
        if not embeddings:
            return None
        em = np.stack(
            [np.asarray(e, dtype=np.float32).reshape(-1) for e in embeddings[: max(1, top_k)]]
        )
        cen = np.mean(em, axis=0)
        norm = np.linalg.norm(cen) + 1e-12
        return (cen / norm).reshape(1, -1)


_engine: Optional[FaceEngineArcFace] = None


def get_face_engine_arcface() -> FaceEngineArcFace:
    global _engine
    if _engine is None:
        _engine = FaceEngineArcFace()
    return _engine
