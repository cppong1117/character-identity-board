#!/usr/bin/env python3
"""PHASE 4+5+6+7: Quality Gate V2 — dual-pool face quality scoring."""
import sqlite3, numpy as np, sys, os, json, time
from collections import defaultdict

DB_PATH = os.path.expanduser('~/character-identity-board-data/cib.sqlite3')
REPORT_DIR = os.path.expanduser('~/character-identity-board/reports/v02_accuracy_recovery')

def compute_quality_score(row):
    """Compute quality score from observation row."""
    (obs_id, frame_num, bbox_json, quality_s, blur_s, occ_s, yaw, pitch, roll, bbox2) = row
    
    bbox = eval(bbox_json) if isinstance(bbox_json, str) and bbox_json else [0,0,0,0]
    face_area = (bbox[2] if len(bbox)>2 else 0) * (bbox[3] if len(bbox)>3 else 0)
    
    quality_s = quality_s or 0
    blur_s = blur_s or 0
    occ_s = occ_s or 0
    yaw = abs(yaw or 0)
    pitch = abs(pitch or 0)
    
    # Sub-scores
    det_s = min(1.0, max(0, (quality_s - 0.3) / 0.6))  # quality 0.3→0, 0.9→1
    blur_sub = min(1.0, blur_s / 200.0)  # HIGHER = SHARPER (Laplacian variance)
    occ_sub = max(0, 1.0 - occ_s)
    yaw_s = max(0, 1.0 - yaw / 60.0)
    pitch_s = max(0, 1.0 - pitch / 45.0)
    pose_s = (yaw_s + pitch_s) / 2
    size_s = min(1.0, face_area / 10000)
    
    # Weighted quality
    quality = (
        0.25 * det_s +
        0.15 * blur_sub +
        0.10 * occ_sub +
        0.20 * pose_s +
        0.15 * size_s +
        0.15 * 0.5  # landmark default
    )
    
    # Identity evidence gate
    reasons = []
    if quality_s < 0.70:
        reasons.append("LOW_DETECTOR_CONFIDENCE")
    if blur_s < 30.0:
        reasons.append("BLUR_TOO_HIGH")
    if occ_s > 0.3:
        reasons.append("OCCLUDED")
    if yaw > 30:
        reasons.append("POSE_TOO_EXTREME")
    if face_area < 500:
        reasons.append("FACE_TOO_SMALL")
    
    return (
        round(quality, 4),
        0 if reasons else 1,
        "|".join(reasons) if reasons else None,
    )

def main():
    print("=== PHASE 4+5+6+7: Quality Gate V2 ===")
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM face_observations WHERE excluded=0 AND embedding IS NOT NULL")
    total_emb = cur.fetchone()[0]
    print(f"Total observations: {total_emb}")
    
    # Add columns
    for col, typ in [
        ("quality_score_v2", "REAL"),
        ("identity_evidence_allowed", "INTEGER DEFAULT 0"),
        ("rejection_reason", "TEXT"),
    ]:
        try:
            cur.execute(f"ALTER TABLE face_observations ADD COLUMN {col} {typ}")
            print(f"Added {col}")
        except:
            pass
    conn.commit()
    
    # Process
    cur.execute('''SELECT id, frame_number, face_bbox, quality_score, blur_score,
                   occlusion_score, yaw, pitch, roll, face_bbox
                   FROM face_observations WHERE excluded = 0 AND embedding IS NOT NULL''')
    rows = cur.fetchall()
    print(f"Processing {len(rows)} observations...")
    
    stats = {'total': 0, 'ok': 0, 'rejected': 0, 'reasons': defaultdict(int)}
    batch = []
    t0 = time.time()
    
    for row in rows:
        q, allowed, reasons = compute_quality_score(row)
        batch.append((q, allowed, reasons, row[0]))
        stats['total'] += 1
        if allowed:
            stats['ok'] += 1
        else:
            stats['rejected'] += 1
            for r in (reasons or '').split('|'):
                if r: stats['reasons'][r] += 1
        
        if len(batch) >= 1000:
            cur.executemany('UPDATE face_observations SET quality_score_v2=?, identity_evidence_allowed=?, rejection_reason=? WHERE id=?', batch)
            conn.commit()
            batch = []
    
    if batch:
        cur.executemany('UPDATE face_observations SET quality_score_v2=?, identity_evidence_allowed=?, rejection_reason=? WHERE id=?', batch)
        conn.commit()
    
    conn.close()
    elapsed = time.time() - t0
    
    print(f"\n=== QUALITY GATE V2 COMPLETE ({elapsed:.1f}s) ===")
    print(f"Total: {stats['total']}")
    print(f"Identity OK: {stats['ok']} ({stats['ok']/max(1,stats['total'])*100:.1f}%)")
    print(f"Rejected: {stats['rejected']} ({stats['rejected']/max(1,stats['total'])*100:.1f}%)")
    print(f"\nRejection reasons:")
    for reason, count in sorted(stats['reasons'].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, 'quality_gate_v2.json'), 'w') as f:
        json.dump({'total': stats['total'], 'ok': stats['ok'], 'rejected': stats['rejected'],
                    'reasons': dict(stats['reasons'])}, f, indent=2)
    print(f"Report saved")

if __name__ == '__main__':
    main()
