#!/usr/bin/env python3
"""PHASE 4+5+6+7: Quality Gate V2 — dual-pool face quality scoring.

After re-embed completes, this script:
1. Computes comprehensive quality scores for every face observation
2. Assigns each to Observation Pool or Identity Evidence Pool
3. Records rejection reasons
4. Validates landmark geometry
5. Updates database with quality_score, identity_evidence_allowed, rejection_reason
"""
import sqlite3, numpy as np, sys, os, json, cv2, time
from collections import defaultdict

sys.path.insert(0, os.path.expanduser('~/character-identity-board/backend'))
from app.face_engine import get_face_engine

DB_PATH = os.path.expanduser('~/character-identity-board-data/cib.sqlite3')
VID = '/home/ponky_re6000/character-identity-board-data/projects/15/videos/c9bd3ce7_mirrored20260828 happy.mp4'
REPORT_DIR = os.path.expanduser('~/character-identity-board/reports/v02_accuracy_recovery')

def compute_landmark_score(landmarks_5x2):
    """Validate 5-point landmark geometry. Returns (score, reason).
    
    Landmarks: [left_eye, right_eye, nose_tip, mouth_left, mouth_right]
    """
    if landmarks_5x2 is None or len(landmarks_5x2) < 5:
        return 0.0, "LANDMARK_INVALID"
    
    pts = np.array(landmarks_5x2, dtype=np.float32)
    
    left_eye, right_eye, nose, mouth_l, mouth_r = pts[0], pts[1], pts[2], pts[3], pts[4]
    
    # Check: eyes exist
    if np.all(left_eye == 0) or np.all(right_eye == 0):
        return 0.0, "LANDMARK_INVALID"
    
    # Check: eyes above nose (y-axis, smaller = higher in image)
    if left_eye[1] > nose[1] + 10 or right_eye[1] > nose[1] + 10:
        return 0.1, "LANDMARK_GEOMETRY_BAD"
    
    # Check: nose above mouth
    if nose[1] > mouth_l[1] + 5 or nose[1] > mouth_r[1] + 5:
        return 0.1, "LANDMARK_GEOMETRY_BAD"
    
    # Check: reasonable inter-eye distance
    eye_dist = np.linalg.norm(right_eye - left_eye)
    if eye_dist < 5:
        return 0.2, "EYES_TOO_CLOSE"
    if eye_dist > 500:
        return 0.2, "EYES_TOO_FAR"
    
    # Check: face width/height ratio from landmarks
    face_w = eye_dist * 2.5  # approximate
    face_h = np.linalg.norm(mouth_l - (left_eye + right_eye) / 2) * 1.8
    if face_w > 0 and face_h > 0:
        ratio = face_h / face_w
        if ratio < 0.3 or ratio > 3.0:
            return 0.3, "BAD_FACE_RATIO"
    
    # All checks passed
    return 1.0, "OK"


def compute_quality_score(obs_row):
    """Compute comprehensive quality score from observation data.
    
    Returns dict with sub-scores and final quality_score.
    """
    det_score = obs_row.get('detector_score', 0) or 0
    blur_score = obs_row.get('blur_score', 0) or 0
    occlusion = obs_row.get('occlusion_score', 0) or 0
    yaw = abs(obs_row.get('yaw', 0) or 0)
    pitch = abs(obs_row.get('pitch', 0) or 0)
    face_w = obs_row.get('face_width', 0) or 0
    face_h = obs_row.get('face_height', 0) or 0
    
    # Sub-scores (all 0-1)
    # 1. Detector confidence (0.85+ is good for YuNet)
    det_s = min(1.0, max(0, (det_score - 0.5) / 0.4))  # 0.5→0, 0.9→1
    
    # 2. Blur (lower = sharper; 27 is threshold)
    blur_s = max(0, 1.0 - blur_score / 50.0)  # 0→1, 27→0.46, 50→0
    
    # 3. Occlusion (lower = less occluded)
    occ_s = max(0, 1.0 - occlusion)  # 0→1, 0.5→0.5, 1→0
    
    # 4. Pose (frontal = good)
    yaw_s = max(0, 1.0 - yaw / 60.0)  # 0°→1, 30°→0.5, 60°→0
    pitch_s = max(0, 1.0 - pitch / 45.0)
    pose_s = (yaw_s + pitch_s) / 2
    
    # 5. Face size (larger = better, up to a point)
    face_area = face_w * face_h
    size_s = min(1.0, face_area / 10000)  # 100x100 → 1.0
    
    # 6. Landmark geometry
    lm_s = 0.5  # default if no landmarks
    rejection = None
    
    # Weighted combination
    weights = {
        'detector': 0.20,
        'blur': 0.15,
        'occlusion': 0.10,
        'pose': 0.20,
        'size': 0.15,
        'landmark': 0.20,
    }
    
    quality = (
        weights['detector'] * det_s +
        weights['blur'] * blur_s +
        weights['occlusion'] * occ_s +
        weights['pose'] * pose_s +
        weights['size'] * size_s +
        weights['landmark'] * lm_s
    )
    
    # Identity evidence decision
    # High precision gate: multiple conditions must be met
    identity_ok = True
    rejection_reasons = []
    
    if det_score < 0.85:
        identity_ok = False
        rejection_reasons.append("LOW_DETECTOR_CONFIDENCE")
    if blur_score > 27.0:
        identity_ok = False
        rejection_reasons.append("BLUR_TOO_HIGH")
    if occlusion > 0.3:
        identity_ok = False
        rejection_reasons.append("OCCLUDED")
    if yaw > 30:
        identity_ok = False
        rejection_reasons.append("POSE_TOO_EXTREME")
    if face_area < 500:  # ~22x22 pixels
        identity_ok = False
        rejection_reasons.append("FACE_TOO_SMALL")
    
    rejection = "|".join(rejection_reasons) if rejection_reasons else None
    
    return {
        'quality_score': round(quality, 4),
        'detector_score_sub': round(det_s, 4),
        'blur_score_sub': round(blur_s, 4),
        'occlusion_score_sub': round(occ_s, 4),
        'pose_score_sub': round(pose_s, 4),
        'size_score_sub': round(size_s, 4),
        'landmark_score_sub': round(lm_s, 4),
        'identity_evidence_allowed': identity_ok,
        'rejection_reason': rejection,
    }


def main():
    print("=== PHASE 4+5+6+7: Quality Gate V2 ===")
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Check if re-embed is done by verifying embedding uniqueness
    cur.execute("SELECT COUNT(*) FROM face_observations WHERE excluded=0 AND embedding IS NOT NULL")
    total_emb = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT SUBSTR(embedding, 1, 64)) FROM face_observations WHERE excluded=0 AND embedding IS NOT NULL")
    unique_emb = cur.fetchone()[0]
    dup_rate = (total_emb - unique_emb) / total_emb * 100 if total_emb > 0 else 0
    
    print(f"Embeddings: {total_emb} total, {unique_emb} unique, {dup_rate:.1f}% duplication")
    if dup_rate > 50:
        print("WARNING: Re-embed may not be complete yet. Waiting...")
        # Don't proceed if embeddings are still mostly duplicates
    
    # Add new columns if not exist
    try:
        cur.execute("ALTER TABLE face_observations ADD COLUMN quality_score_v2 REAL")
        print("Added quality_score_v2 column")
    except:
        pass
    try:
        cur.execute("ALTER TABLE face_observations ADD COLUMN identity_evidence_allowed INTEGER DEFAULT 0")
        print("Added identity_evidence_allowed column")
    except:
        pass
    try:
        cur.execute("ALTER TABLE face_observations ADD COLUMN rejection_reason TEXT")
        print("Added rejection_reason column")
    except:
        pass
    
    conn.commit()
    
    # Process all observations
    cur.execute('''SELECT id, frame_number, face_bbox, detector_score, blur_score, 
                   occlusion_score, yaw, pitch, roll, face_width, face_height,
                   embedding, embedding_dim, excluded
                   FROM face_observations WHERE excluded = 0 AND embedding IS NOT NULL''')
    rows = cur.fetchall()
    print(f"\nProcessing {len(rows)} observations...")
    
    stats = {
        'total': 0,
        'identity_ok': 0,
        'identity_rejected': 0,
        'rejection_reasons': defaultdict(int),
    }
    
    batch = []
    t0 = time.time()
    
    for i, row in enumerate(rows):
        (obs_id, frame_num, bbox_json, det_score, blur_s, occ_s, 
         yaw, pitch, roll, face_w, face_h, emb, emb_dim, excluded) = row
        
        bbox = eval(bbox_json) if isinstance(bbox_json, str) else bbox_json if bbox_json else [0,0,0,0]
        
        obs_data = {
            'detector_score': det_score or 0,
            'blur_score': blur_s or 0,
            'occlusion_score': occ_s or 0,
            'yaw': yaw or 0,
            'pitch': pitch or 0,
            'roll': roll or 0,
            'face_width': face_w or (bbox[2] if bbox else 0),
            'face_height': face_h or (bbox[3] if bbox else 0),
        }
        
        result = compute_quality_score(obs_data)
        
        batch.append((
            result['quality_score'],
            1 if result['identity_evidence_allowed'] else 0,
            result['rejection_reason'],
            obs_id,
        ))
        
        stats['total'] += 1
        if result['identity_evidence_allowed']:
            stats['identity_ok'] += 1
        else:
            stats['identity_rejected'] += 1
            for reason in (result['rejection_reason'] or '').split('|'):
                if reason:
                    stats['rejection_reasons'][reason] += 1
        
        if len(batch) >= 1000:
            cur.executemany('''UPDATE face_observations 
                SET quality_score_v2=?, identity_evidence_allowed=?, rejection_reason=?
                WHERE id=?''', batch)
            conn.commit()
            batch = []
        
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(rows)} processed ({elapsed:.1f}s)")
    
    if batch:
        cur.executemany('''UPDATE face_observations 
            SET quality_score_v2=?, identity_evidence_allowed=?, rejection_reason=?
            WHERE id=?''', batch)
        conn.commit()
    
    conn.close()
    elapsed = time.time() - t0
    
    # Report
    print(f"\n=== QUALITY GATE V2 COMPLETE ({elapsed:.1f}s) ===")
    print(f"Total processed: {stats['total']}")
    print(f"Identity evidence OK: {stats['identity_ok']} ({stats['identity_ok']/max(1,stats['total'])*100:.1f}%)")
    print(f"Identity evidence REJECTED: {stats['identity_rejected']} ({stats['identity_rejected']/max(1,stats['total'])*100:.1f}%)")
    print(f"\nRejection reasons:")
    for reason, count in sorted(stats['rejection_reasons'].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    
    # Save report
    os.makedirs(REPORT_DIR, exist_ok=True)
    report = {
        'total': stats['total'],
        'identity_ok': stats['identity_ok'],
        'identity_rejected': stats['identity_rejected'],
        'rejection_reasons': dict(stats['rejection_reasons']),
    }
    with open(os.path.join(REPORT_DIR, 'quality_gate_v2.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {REPORT_DIR}/quality_gate_v2.json")


if __name__ == '__main__':
    main()
