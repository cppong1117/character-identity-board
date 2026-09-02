#!/usr/bin/env python3
"""Batch re-embed all face observations with correct landmark alignment."""
import sqlite3, cv2, numpy as np, sys, os, time
from collections import defaultdict

sys.path.insert(0, os.path.expanduser('~/character-identity-board/backend'))
from app.face_engine import get_face_engine

DB_PATH = os.path.expanduser('~/character-identity-board-data/cib.sqlite3')
VID = '/home/ponky_re6000/character-identity-board-data/projects/15/videos/c9bd3ce7_mirrored20260828 happy.mp4'

def main():
    engine = get_face_engine()
    engine._init()
    print(f"Face engine initialized: backend={engine._backend}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Get all observations grouped by frame
    cur.execute('''SELECT fo.id, fo.frame_number, fo.face_bbox
        FROM face_observations fo
        JOIN tracklets t ON fo.tracklet_id = t.id
        JOIN shots s ON t.shot_id = s.id
        WHERE s.video_id = 15 AND fo.excluded = 0 AND fo.face_bbox IS NOT NULL
        ORDER BY fo.frame_number''')
    rows = cur.fetchall()
    print(f"Total observations to re-embed: {len(rows)}")

    # Group by frame
    frame_obs = defaultdict(list)
    for obs_id, frame_num, bbox_json in rows:
        bbox = eval(bbox_json) if isinstance(bbox_json, str) else bbox_json
        frame_obs[frame_num].append((obs_id, bbox))

    print(f"Unique frames: {len(frame_obs)}")

    cap = cv2.VideoCapture(VID)
    updated = 0
    skipped = 0
    failed = 0
    batch = []
    t0 = time.time()

    for fi, (frame_num, obs_list) in enumerate(sorted(frame_obs.items())):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            skipped += len(obs_list)
            continue

        detections = engine.detect_faces(frame)

        for obs_id, bbox in obs_list:
            # Find matching detection by IoU
            best_det = None
            best_iou = 0
            for det in detections:
                det_bbox = det['bbox']
                ix = max(0, min(bbox[0]+bbox[2], det_bbox[0]+det_bbox[2]) - max(bbox[0], det_bbox[0]))
                iy = max(0, min(bbox[1]+bbox[3], det_bbox[1]+det_bbox[3]) - max(bbox[1], det_bbox[1]))
                inter = ix * iy
                union = bbox[2]*bbox[3] + det_bbox[2]*det_bbox[3] - inter
                iou = inter / (union + 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            if best_det and best_iou > 0.3:
                emb = engine.get_embedding(frame, best_det)
                if emb is not None:
                    batch.append((emb.astype(np.float32).tobytes(), 128, obs_id))
                    updated += 1
                else:
                    failed += 1
            else:
                skipped += 1

        if len(batch) >= 500:
            cur.executemany('UPDATE face_observations SET embedding=?, embedding_dim=? WHERE id=?', batch)
            conn.commit()
            batch = []

        if (fi + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            eta = (len(frame_obs) - fi - 1) / rate if rate > 0 else 0
            print(f"  Frame {fi+1}/{len(frame_obs)}: updated={updated} skipped={skipped} failed={failed} "
                  f"({rate:.1f} frames/s, ETA {eta/60:.1f}min)")

    # Final commit
    if batch:
        cur.executemany('UPDATE face_observations SET embedding=?, embedding_dim=? WHERE id=?', batch)
        conn.commit()

    cap.release()
    conn.close()

    elapsed = time.time() - t0
    print(f"\n=== RE-EMBED COMPLETE ({elapsed/60:.1f} min) ===")
    print(f"Updated: {updated}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")

if __name__ == '__main__':
    main()
