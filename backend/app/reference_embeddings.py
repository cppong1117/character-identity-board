"""Reference-image embedding for Reference Mode."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ReferenceEmbedding:
    def __init__(self, status: str, embedding: np.ndarray | None = None,
                 embedding_dim: int | None = None, reason: str | None = None):
        self.status = status
        self.embedding = embedding
        self.embedding_dim = embedding_dim
        self.reason = reason


def embed_reference_image(path: str | Path, face_engine: Any) -> ReferenceEmbedding:
    """Detect the strongest face and encode it with the existing SFace engine."""
    # Reference files are already face crops; detect once for landmarks, then
    # reuse the crop bbox/landmarks for SFace alignment without a second crop.
    image = cv2.imread(str(path))
    if image is None:
        return ReferenceEmbedding("error", reason="image_unreadable")
    faces = face_engine.detect_faces(image)
    if not faces:
        return ReferenceEmbedding("no_face", reason="no_face_detected")
    face = max(faces, key=lambda item: item.get("score", 0.0))
    bbox = face.get("bbox")
    if not bbox:
        return ReferenceEmbedding("no_face", reason="missing_bbox")
    # Pass the full detection dict to get_embedding for proper alignment.
    embedding = face_engine.get_embedding(image, face)
    if embedding is None:
        return ReferenceEmbedding("no_embedding", reason="engine_no_embedding")
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    return ReferenceEmbedding("embedded", vector, int(vector.size))


def build_reference_sets(reference_packs: dict[Any, list[str]]) -> dict[Any, dict[str, list[str]]]:
    """Partition persisted image paths into per-character positives/negatives."""
    all_paths = [path for paths in reference_packs.values() for path in paths]
    return {
        character_id: {
            "positive_paths": list(paths),
            "negative_paths": [path for path in all_paths if path not in paths],
        }
        for character_id, paths in reference_packs.items()
    }


def embed_reference_packs(reference_packs: dict[Any, list[str]], face_engine: Any) -> dict[Any, dict[str, Any]]:
    """Embed every persisted reference path and retain failures for review."""
    result = {}
    for character_id, paths in reference_packs.items():
        embeddings = []
        failures = []
        for path in paths:
            item = embed_reference_image(path, face_engine)
            if item.embedding is not None:
                embeddings.append(item.embedding)
            else:
                failures.append({"path": path, "status": item.status, "reason": item.reason})
        result[character_id] = {"embeddings": embeddings, "failures": failures}
    return result
