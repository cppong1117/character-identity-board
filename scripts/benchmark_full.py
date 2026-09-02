#!/usr/bin/env python3
"""PHASE 11-14: Threshold Calibration + Three-Way Decision + Full Benchmark.

After re-embed + quality gate + clustering, this script:
1. Calibrates SFace threshold on actual data
2. Measures best/second-best margin
3. Computes cross-shot similarity for same-person pairs
4. Reports Precision/Recall/FAR/FRR at multiple thresholds
5. Produces final accuracy report
"""
import sqlite3, numpy as np, sys, os, json, time, cv2
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = os.path.expanduser('~/character-identity-board-data/cib.sqlite3')
REPORT_DIR = os.path.expanduser('~/character-identity-board/reports/v02_accuracy_recovery')
THRESHOLDS = np.arange(0.30, 0.62, 0.02)


def load_tracklet_representatives(conn, project_id=15):
    """Load tracklet prototype embeddings for cross-tracklet comparison."""
    cur = conn.cursor()
    
    # Get tracklets with their character assignments (via identity_assignments)
    cur.execute('''SELECT t.id, ia.character_id, t.shot_id, c.display_name, c.character_code
        FROM tracklets t
        LEFT JOIN identity_assignments ia ON ia.tracklet_id = t.id
        LEFT JOIN characters c ON ia.character_id = c.id
        WHERE t.shot_id IN (SELECT id FROM shots WHERE video_id = ?)''', (project_id,))
    tracklet_info = {row[0]: {'character_id': row[1], 'shot_id': row[2], 'name': row[3], 'code': row[4]}
                     for row in cur.fetchall()}
    
    # Get best embedding per tracklet (quality-weighted)
    cur.execute('''SELECT fo.tracklet_id, fo.embedding, fo.quality_score_v2
        FROM face_observations fo
        JOIN tracklets t ON fo.tracklet_id = t.id
        JOIN shots s ON t.shot_id = s.id
        WHERE s.video_id = ? AND fo.excluded = 0 AND fo.embedding IS NOT NULL
        AND fo.identity_evidence_allowed = 1
        ORDER BY fo.tracklet_id, COALESCE(fo.quality_score_v2, 0) DESC''', (project_id,))
    
    tracklet_embs = defaultdict(list)
    for tid, emb_bytes, qs in cur.fetchall():
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        tracklet_embs[tid].append((emb, qs or 0))
    
    # Build prototypes
    prototypes = {}
    for tid, embs_list in tracklet_embs.items():
        if tid not in tracklet_info:
            continue
        # Top-5 quality-weighted centroid
        top = sorted(embs_list, key=lambda x: -x[1])[:5]
        embs = np.array([e[0] for e in top])
        weights = np.array([e[1] for e in top])
        weights = weights / (weights.sum() + 1e-12)
        proto = np.average(embs, axis=0, weights=weights)
        n = np.linalg.norm(proto) + 1e-12
        prototypes[tid] = {
            'embedding': proto / n,
            'info': tracklet_info[tid],
            'n_faces': len(embs_list),
        }
    
    return prototypes


def build_same_person_pairs(prototypes):
    """Find pairs of tracklets that belong to the same character."""
    # Group by character
    char_tracklets = defaultdict(list)
    for tid, data in prototypes.items():
        char_id = data['info']['character_id']
        if char_id is not None:
            char_tracklets[char_id].append(tid)
    
    pairs = []
    for char_id, tids in char_tracklets.items():
        if len(tids) < 2:
            continue
        for i in range(len(tids)):
            for j in range(i+1, len(tids)):
                pairs.append(('same', tids[i], tids[j], char_id))
    
    return pairs


def build_different_person_pairs(prototypes, max_pairs=500):
    """Find pairs of tracklets from different characters."""
    char_tracklets = defaultdict(list)
    for tid, data in prototypes.items():
        char_id = data['info']['character_id']
        if char_id is not None:
            char_tracklets[char_id].append(tid)
    
    char_ids = sorted(char_tracklets.keys())
    pairs = []
    for ci in range(len(char_ids)):
        for cj in range(ci+1, len(char_ids)):
            for ti in char_tracklets[char_ids[ci]][:3]:
                for tj in char_tracklets[char_ids[cj]][:3]:
                    pairs.append(('different', ti, tj, (char_ids[ci], char_ids[cj])))
                    if len(pairs) >= max_pairs:
                        return pairs
    
    return pairs


def build_cross_shot_pairs(prototypes):
    """Same person, different shot — the critical test."""
    char_tracklets = defaultdict(list)
    for tid, data in prototypes.items():
        char_id = data['info']['character_id']
        shot_id = data['info']['shot_id']
        if char_id is not None:
            char_tracklets[char_id].append((tid, shot_id))
    
    pairs = []
    for char_id, tracklets in char_tracklets.items():
        shots_seen = defaultdict(list)
        for tid, shot_id in tracklets:
            shots_seen[shot_id].append(tid)
        
        shot_ids = sorted(shots_seen.keys())
        for i in range(len(shot_ids)):
            for j in range(i+1, len(shot_ids)):
                for ti in shots_seen[shot_ids[i]][:2]:
                    for tj in shots_seen[shot_ids[j]][:2]:
                        pairs.append(('cross_shot', ti, tj, char_id))
    
    return pairs


def evaluate_thresholds(prototypes, same_pairs, diff_pairs, cross_pairs):
    """Evaluate metrics at multiple cosine thresholds."""
    results = []
    
    all_embeddings = {tid: data['embedding'] for tid, data in prototypes.items()}
    
    for threshold in THRESHOLDS:
        # Same-person matching
        tp = 0  # same person, above threshold (correct match)
        fn = 0  # same person, below threshold (missed)
        same_sims = []
        
        for _, ti, tj, _ in same_pairs:
            sim = float(np.dot(all_embeddings[ti], all_embeddings[tj]))
            same_sims.append(sim)
            if sim >= threshold:
                tp += 1
            else:
                fn += 1
        
        # Different-person matching
        fp = 0  # different person, above threshold (false merge)
        tn = 0  # different person, below threshold (correct reject)
        diff_sims = []
        
        for _, ti, tj, _ in diff_pairs:
            sim = float(np.dot(all_embeddings[ti], all_embeddings[tj]))
            diff_sims.append(sim)
            if sim >= threshold:
                fp += 1
            else:
                tn += 1
        
        # Cross-shot same-person
        cross_tp = 0
        cross_fn = 0
        cross_sims = []
        
        for _, ti, tj, _ in cross_pairs:
            sim = float(np.dot(all_embeddings[ti], all_embeddings[tj]))
            cross_sims.append(sim)
            if sim >= threshold:
                cross_tp += 1
            else:
                cross_fn += 1
        
        total_same = tp + fn
        total_diff = fp + tn
        total_cross = cross_tp + cross_fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        cross_recall = cross_tp / (cross_tp + cross_fn) if (cross_tp + cross_fn) > 0 else 0
        
        results.append({
            'threshold': round(float(threshold), 3),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'far': round(far, 4),
            'frr': round(frr, 4),
            'cross_shot_recall': round(cross_recall, 4),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'same_sim_mean': round(float(np.mean(same_sims)), 4) if same_sims else 0,
            'same_sim_median': round(float(np.median(same_sims)), 4) if same_sims else 0,
            'diff_sim_mean': round(float(np.mean(diff_sims)), 4) if diff_sims else 0,
            'diff_sim_median': round(float(np.median(diff_sims)), 4) if diff_sims else 0,
            'cross_sim_mean': round(float(np.mean(cross_sims)), 4) if cross_sims else 0,
            'cross_sim_median': round(float(np.median(cross_sims)), 4) if cross_sims else 0,
        })
    
    return results


def compute_margin_analysis(prototypes):
    """Compute best/second-best margin for each tracklet vs all others."""
    tids = list(prototypes.keys())
    embs = np.array([prototypes[tid]['embedding'] for tid in tids])
    char_ids = [prototypes[tid]['info']['character_id'] for tid in tids]
    
    sim_matrix = cosine_similarity(embs)
    
    margins = []
    for i in range(len(tids)):
        # Get similarities to all other tracklets
        sims = sim_matrix[i]
        # Get the same-character similarity (if any)
        same_char_sims = [sims[j] for j in range(len(tids)) if j != i and char_ids[j] == char_ids[i] and char_ids[i] is not None]
        # Get the best different-character similarity
        diff_char_sims = [sims[j] for j in range(len(tids)) if j != i and char_ids[j] != char_ids[i]]
        
        best_same = max(same_char_sims) if same_char_sims else 0
        best_diff = max(diff_char_sims) if diff_char_sims else 0
        margin = best_same - best_diff
        
        margins.append({
            'tracklet_id': tids[i],
            'character_id': char_ids[i],
            'best_same_sim': round(best_same, 4),
            'best_diff_sim': round(best_diff, 4),
            'margin': round(margin, 4),
        })
    
    return margins


def main():
    print("=== PHASE 11-14: Threshold Calibration + Full Benchmark ===")
    
    conn = sqlite3.connect(DB_PATH)
    t0 = time.time()
    
    # Load prototypes
    prototypes = load_tracklet_representatives(conn)
    print(f"Loaded {len(prototypes)} tracklet prototypes")
    
    # Build pairs
    same_pairs = build_same_person_pairs(prototypes)
    diff_pairs = build_different_person_pairs(prototypes)
    cross_pairs = build_cross_shot_pairs(prototypes)
    
    print(f"Same-person pairs: {len(same_pairs)}")
    print(f"Different-person pairs: {len(diff_pairs)}")
    print(f"Cross-shot pairs: {len(cross_pairs)}")
    
    # Evaluate thresholds
    threshold_results = evaluate_thresholds(prototypes, same_pairs, diff_pairs, cross_pairs)
    
    # Margin analysis
    margins = compute_margin_analysis(prototypes)
    
    conn.close()
    elapsed = time.time() - t0
    
    # Report
    print(f"\n=== BENCHMARK COMPLETE ({elapsed:.1f}s) ===")
    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'FAR':>10} {'FRR':>10} {'CrossRecall':>12}")
    print("-" * 72)
    for r in threshold_results:
        print(f"{r['threshold']:>10.3f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['far']:>10.4f} {r['frr']:>10.4f} {r['cross_shot_recall']:>12.4f}")
    
    print(f"\nScore distributions:")
    if threshold_results:
        r = threshold_results[len(threshold_results)//2]  # middle threshold
        print(f"  Same-person: mean={r['same_sim_mean']}, median={r['same_sim_median']}")
        print(f"  Different-person: mean={r['diff_sim_mean']}, median={r['diff_sim_median']}")
        print(f"  Cross-shot: mean={r['cross_sim_mean']}, median={r['cross_sim_median']}")
    
    # Find best threshold (precision >= 0.99 with highest recall)
    best = None
    for r in threshold_results:
        if r['precision'] >= 0.99:
            if best is None or r['recall'] > best['recall']:
                best = r
    
    if best:
        print(f"\nBest threshold for 99% precision: {best['threshold']}")
        print(f"  Recall: {best['recall']:.4f}")
        print(f"  Cross-shot recall: {best['cross_shot_recall']:.4f}")
    else:
        print("\nNo threshold achieves 99% precision")
        # Find best compromise
        best = max(threshold_results, key=lambda r: r['precision'] * 0.7 + r['recall'] * 0.3)
        print(f"Best compromise: threshold={best['threshold']}, precision={best['precision']:.4f}, recall={best['recall']:.4f}")
    
    # Save
    os.makedirs(REPORT_DIR, exist_ok=True)
    report = {
        'threshold_results': threshold_results,
        'margin_analysis': {
            'n_tracklets': len(margins),
            'mean_margin': round(float(np.mean([m['margin'] for m in margins])), 4),
            'median_margin': round(float(np.median([m['margin'] for m in margins])), 4),
        },
        'same_person_pairs': len(same_pairs),
        'different_person_pairs': len(diff_pairs),
        'cross_shot_pairs': len(cross_pairs),
        'best_threshold_99prec': best['threshold'] if best else None,
    }
    with open(os.path.join(REPORT_DIR, 'benchmark_full.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to {REPORT_DIR}/benchmark_full.json")


if __name__ == '__main__':
    main()
