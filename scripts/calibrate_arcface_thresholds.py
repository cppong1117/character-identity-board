#!/usr/bin/env python3
"""
Threshold calibration for ArcFace R100 embeddings.
Finds optimal thresholds for identity/matching decisions.
"""
import sqlite3
import numpy as np
from pathlib import Path
from collections import defaultdict

DB_PATH = str(Path.home() / "character-identity-board-data/cib.sqlite3")
OUTPUT_DIR = str(Path.home() / "character-identity-board/reports/v02_accuracy_recovery")


def load_arcface_embeddings():
    """Load ArcFace embeddings (512-dim) from database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT fo.id, fo.embedding, fo.quality_score_v2,
               ia.character_id, c.display_name
        FROM face_observations fo
        JOIN tracklets t ON fo.tracklet_id = t.id
        JOIN identity_assignments ia ON t.id = ia.tracklet_id
        JOIN characters c ON ia.character_id = c.id
        WHERE fo.embedding IS NOT NULL
        AND fo.embedding_dim = 512
        AND c.display_name != 'Unknown'
        AND c.status IN ('manual', 'automatic')
    """)
    
    rows = cur.fetchall()
    conn.close()
    
    embeddings = []
    for row in rows:
        emb = np.frombuffer(row[1], dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-6)
        embeddings.append({
            'id': row[0],
            'embedding': emb,
            'quality_score': row[2],
            'char_id': row[3],
            'char_name': row[4]
        })
    
    return embeddings


def generate_pairs(embeddings):
    """Generate same-person and different-person pairs."""
    by_char = defaultdict(list)
    for e in embeddings:
        by_char[e['char_id']].append(e)
    
    same_pairs = []
    diff_pairs = []
    
    # Same-person pairs
    for char_id, char_embs in by_char.items():
        if len(char_embs) < 2:
            continue
        for i in range(len(char_embs)):
            for j in range(i+1, min(i+30, len(char_embs))):
                same_pairs.append((char_embs[i], char_embs[j]))
    
    # Different-person pairs
    char_ids = [c for c, embs in by_char.items() if len(embs) >= 2]
    for i in range(len(char_ids)):
        for j in range(i+1, min(i+5, len(char_ids))):
            embs_a = by_char[char_ids[i]][:10]
            embs_b = by_char[char_ids[j]][:10]
            for a in embs_a:
                for b in embs_b:
                    diff_pairs.append((a, b))
    
    return same_pairs, diff_pairs


def find_optimal_thresholds(same_scores, diff_scores):
    """Find optimal thresholds for different operating points."""
    same_arr = np.array(same_scores)
    diff_arr = np.array(diff_scores)
    
    results = []
    
    for threshold in np.arange(0.10, 0.90, 0.01):
        tp = np.sum(same_arr >= threshold)
        fn = np.sum(same_arr < threshold)
        fp = np.sum(diff_arr >= threshold)
        tn = np.sum(diff_arr < threshold)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        
        # False Accept Rate (FAR) = FP / (FP + TN)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        # False Reject Rate (FRR) = FN / (FN + TP)
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'far': far,
            'frr': frr
        })
    
    return results


def main():
    print("Loading ArcFace embeddings...", flush=True)
    embeddings = load_arcface_embeddings()
    print(f"Loaded {len(embeddings)} embeddings from {len(set(e['char_id'] for e in embeddings))} characters", flush=True)
    
    # Group by character
    by_char = defaultdict(list)
    for e in embeddings:
        by_char[e['char_id']].append(e)
    
    print("\nCharacters:", flush=True)
    for char_id, char_embs in sorted(by_char.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        name = char_embs[0]['char_name']
        print(f"  {name}: {len(char_embs)} faces", flush=True)
    
    print("\nGenerating pairs...", flush=True)
    same_pairs, diff_pairs = generate_pairs(embeddings)
    print(f"Same-person pairs: {len(same_pairs)}", flush=True)
    print(f"Different-person pairs: {len(diff_pairs)}", flush=True)
    
    # Compute similarity scores
    print("\nComputing similarities...", flush=True)
    same_scores = [float(np.dot(a['embedding'], b['embedding'])) for a, b in same_pairs]
    diff_scores = [float(np.dot(a['embedding'], b['embedding'])) for a, b in diff_pairs]
    
    # Basic stats
    same_arr = np.array(same_scores)
    diff_arr = np.array(diff_scores)
    print(f"\nSame-person: mean={np.mean(same_arr):.4f}, median={np.median(same_arr):.4f}", flush=True)
    print(f"Different-person: mean={np.mean(diff_arr):.4f}, median={np.median(diff_arr):.4f}", flush=True)
    print(f"Separation: {np.mean(same_arr) - np.mean(diff_arr):.4f}", flush=True)
    
    # Find optimal thresholds
    results = find_optimal_thresholds(same_scores, diff_scores)
    
    # Find best F1
    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"\n=== OPTIMAL THRESHOLDS ===", flush=True)
    print(f"Best F1: {best_f1['f1']:.4f} @ threshold={best_f1['threshold']:.2f}", flush=True)
    print(f"  Precision: {best_f1['precision']:.4f}", flush=True)
    print(f"  Recall: {best_f1['recall']:.4f}", flush=True)
    print(f"  FAR: {best_f1['far']:.6f}", flush=True)
    print(f"  FRR: {best_f1['frr']:.6f}", flush=True)
    
    # Find threshold for 95% recall
    for r in results:
        if r['recall'] >= 0.95:
            print(f"\n95% Recall: threshold={r['threshold']:.2f}, precision={r['precision']:.4f}, FAR={r['far']:.6f}", flush=True)
            break
    
    # Find threshold for 99% recall
    for r in results:
        if r['recall'] >= 0.99:
            print(f"99% Recall: threshold={r['threshold']:.2f}, precision={r['precision']:.4f}, FAR={r['far']:.6f}", flush=True)
            break
    
    # Save results
    import json
    output = {
        'same_person_stats': {
            'mean': float(np.mean(same_arr)),
            'median': float(np.median(same_arr)),
            'std': float(np.std(same_arr))
        },
        'different_person_stats': {
            'mean': float(np.mean(diff_arr)),
            'median': float(np.median(diff_arr)),
            'std': float(np.std(diff_arr))
        },
        'separation': float(np.mean(same_arr) - np.mean(diff_arr)),
        'optimal_thresholds': {
            'best_f1': best_f1,
        },
        'threshold_sweep': results
    }
    
    with open(f"{OUTPUT_DIR}/arcface_threshold_calibration.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/arcface_threshold_calibration.json", flush=True)


if __name__ == '__main__':
    main()
