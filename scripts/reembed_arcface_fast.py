#!/usr/bin/env python3
"""
Fast ArcFace re-embedding using only the recognition model.
Skips detection/alignment since face crops are already extracted.
"""
import sys
import os
import sqlite3
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

DB_PATH = str(Path.home() / "character-identity-board-data/cib.sqlite3")
MODEL_DIR = str(Path.home() / ".insightface/models/buffalo_l")
BATCH_SIZE = 200


def load_recognition_model():
    """Load ONLY the recognition model (skip detection)."""
    print("Loading ArcFace recognition model...", flush=True)
    
    import onnxruntime as ort
    
    model_path = os.path.join(MODEL_DIR, "w600k_r50.onnx")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    input_meta = session.get_inputs()[0]
    print(f"  Input shape: {input_meta.shape}", flush=True)
    print(f"  Model loaded!", flush=True)
    
    return session


def preprocess_face(img):
    """Preprocess face crop for ArcFace recognition (112x112, BGR->RGB, normalize)."""
    # Resize to 112x112
    resized = cv2.resize(img, (112, 112))
    
    # BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize: (pixel - 127.5) / 127.5
    normalized = (rgb.astype(np.float32) - 127.5) / 127.5
    
    # HWC to CHW
    chw = normalized.transpose(2, 0, 1)
    
    # Add batch dimension
    batch = np.expand_dims(chw, axis=0)
    
    return batch


def get_embedding(session, img):
    """Get embedding from face crop."""
    preprocessed = preprocess_face(img)
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: preprocessed})
    
    embedding = outputs[0].flatten()
    
    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding.astype(np.float32)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    
    # Load model
    session = load_recognition_model()
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Count total
    cur.execute("SELECT COUNT(*) FROM face_observations WHERE face_crop_path IS NOT NULL")
    total = cur.fetchone()[0]
    print(f"Total observations to re-embed: {total}", flush=True)
    
    # Get all observation IDs and paths
    cur.execute("SELECT id, face_crop_path FROM face_observations WHERE face_crop_path IS NOT NULL ORDER BY id")
    rows = cur.fetchall()
    
    # Process in batches
    success = 0
    failed = 0
    batch_updates = []
    start_time = datetime.now()
    
    for i, (obs_id, face_crop_path) in enumerate(rows):
        try:
            if not Path(face_crop_path).exists():
                failed += 1
                continue
            
            img = cv2.imread(face_crop_path)
            if img is None:
                failed += 1
                continue
            
            embedding = get_embedding(session, img)
            batch_updates.append((embedding.tobytes(), 512, obs_id))
            success += 1
            
        except Exception as e:
            failed += 1
        
        # Commit batch
        if len(batch_updates) >= BATCH_SIZE:
            cur.executemany(
                "UPDATE face_observations SET embedding = ?, embedding_dim = ? WHERE id = ?",
                batch_updates
            )
            conn.commit()
            batch_updates = []
            
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (success + failed) / elapsed if elapsed > 0 else 0
            eta = (total - success - failed) / rate if rate > 0 else 0
            print(f"[{success + failed}/{total}] success={success} failed={failed} rate={rate:.1f}/s ETA={eta:.0f}s", flush=True)
    
    # Final batch
    if batch_updates:
        cur.executemany(
            "UPDATE face_observations SET embedding = ?, embedding_dim = ? WHERE id = ?",
            batch_updates
        )
        conn.commit()
    
    conn.close()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n=== RE-EMBEDDING COMPLETE ===", flush=True)
    print(f"Total: {success + failed}", flush=True)
    print(f"Success: {success}", flush=True)
    print(f"Failed: {failed}", flush=True)
    print(f"Time: {elapsed:.0f}s", flush=True)


if __name__ == '__main__':
    main()
