#!/usr/bin/env python3
"""
A/B Test: SFace vs ArcFace R100 (buffalo_l)
Using existing embeddings and face crops from database
"""
import sqlite3
import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

DB_PATH = str(Path.home() / "character-identity-board-data/cib.sqlite3")
OUTPUT_DIR = str(Path.home() / "character-identity-board/reports/v02_accuracy_recovery")


def load_embeddings_with_crops():
    """Load embeddings and face crop paths from database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT fo.id, fo.face_crop_path, fo.embedding, fo.quality_score_v2,
               ia.character_id, c.display_name
        FROM face_observations fo
        JOIN tracklets t ON fo.tracklet_id = t.id
        JOIN identity_assignments ia ON t.id = ia.tracklet_id
        JOIN characters c ON ia.character_id = c.id
        WHERE fo.embedding IS NOT NULL
        AND fo.face_crop_path IS NOT NULL
        AND c.display_name != 'Unknown'
        AND c.status IN ('manual', 'automatic')
    """)
    
    rows = cur.fetchall()
    conn.close()
    
    embeddings = []
    for row in rows:
        emb = np.frombuffer(row[2], dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-6)
        embeddings.append({
            'id': row[0],
            'face_crop_path': row[1],
            'embedding_sface': emb,
            'quality_score': row[3],
            'char_id': row[4],
            'char_name': row[5]
        })
    
    return embeddings


def generate_pairs(embeddings, max_same_per_char=50, max_diff_pairs=500):
    """Generate same-person and different-person pairs."""
    by_char = defaultdict(list)
    for e in embeddings:
        by_char[e['char_id']].append(e)
    
    same_pairs = []
    diff_pairs = []
    
    # Same-person pairs (subsample)
    for char_id, char_embs in by_char.items():
        if len(char_embs) < 2:
            continue
        indices = np.random.choice(len(char_embs), 
                                   min(max_same_per_char, len(char_embs)*(len(char_embs)-1)//2),
                                   replace=False)
        count = 0
        for i in range(len(char_embs)):
            for j in range(i+1, len(char_embs)):
                if count >= max_same_per_char:
                    break
                same_pairs.append((char_embs[i], char_embs[j]))
                count += 1
    
    # Different-person pairs (subsample)
    char_ids = [c for c, embs in by_char.items() if len(embs) >= 2]
    count = 0
    for i in range(len(char_ids)):
        for j in range(i+1, len(char_ids)):
            if count >= max_diff_pairs:
                break
            embs_a = by_char[char_ids[i]][:10]
            embs_b = by_char[char_ids[j]][:10]
            for a in embs_a:
                for b in embs_b:
                    if count >= max_diff_pairs:
                        break
                    diff_pairs.append((a, b))
                    count += 1
    
    return same_pairs, diff_pairs


def cosine(a, b):
    return float(np.dot(a, b))


def compute_metrics(same_scores, diff_scores):
    """Compute classification metrics."""
    same_arr = np.array(same_scores)
    diff_arr = np.array(diff_scores)
    
    best_f1 = 0
    best_threshold = 0
    best_precision = 0
    best_recall = 0
    
    for threshold in np.arange(0.10, 0.80, 0.01):
        tp = np.sum(same_arr >= threshold)
        fn = np.sum(same_arr < threshold)
        fp = np.sum(diff_arr >= threshold)
        tn = np.sum(diff_arr < threshold)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    same_mean = float(np.mean(same_arr))
    same_median = float(np.median(same_arr))
    diff_mean = float(np.mean(diff_arr))
    diff_median = float(np.median(diff_arr))
    separation = same_mean - diff_mean
    
    return {
        'same_mean': same_mean,
        'same_median': same_median,
        'diff_mean': diff_mean,
        'diff_median': diff_median,
        'separation': separation,
        'best_threshold': float(best_threshold),
        'best_f1': float(best_f1),
        'best_precision': float(best_precision),
        'best_recall': float(best_recall),
        'same_count': len(same_arr),
        'diff_count': len(diff_arr)
    }


def main():
    np.random.seed(42)
    
    print("Loading embeddings...")
    embeddings = load_embeddings_with_crops()
    print(f"Loaded {len(embeddings)} embeddings from {len(set(e['char_id'] for e in embeddings))} characters")
    
    # Group by character
    by_char = defaultdict(list)
    for e in embeddings:
        by_char[e['char_id']].append(e)
    
    print("\nCharacters:")
    for char_id, char_embs in sorted(by_char.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        name = char_embs[0]['char_name']
        print(f"  {name}: {len(char_embs)} faces")
    
    print("\nGenerating pairs...")
    same_pairs, diff_pairs = generate_pairs(embeddings)
    print(f"Same-person pairs: {len(same_pairs)}")
    print(f"Different-person pairs: {len(diff_pairs)}")
    
    # SFace: use existing embeddings directly
    print("\n=== Testing SFace (existing embeddings) ===")
    same_scores_sface = [cosine(a['embedding_sface'], b['embedding_sface']) for a, b in same_pairs]
    diff_scores_sface = [cosine(a['embedding_sface'], b['embedding_sface']) for a, b in diff_pairs]
    metrics_sface = compute_metrics(same_scores_sface, diff_scores_sface)
    print(f"Same-person mean: {metrics_sface['same_mean']:.4f}")
    print(f"Different-person mean: {metrics_sface['diff_mean']:.4f}")
    print(f"Separation: {metrics_sface['separation']:.4f}")
    print(f"Best F1: {metrics_sface['best_f1']:.4f} @ threshold={metrics_sface['best_threshold']:.2f}")
    print(f"Best Precision: {metrics_sface['best_precision']:.4f}, Recall: {metrics_sface['best_recall']:.4f}")
    
    # ArcFace: use InsightFace for fresh embeddings
    print("\n=== Testing ArcFace R100 (buffalo_l) ===")
    try:
        from insightface.app import FaceAnalysis
        
        print("Loading ArcFace model...")
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1)
        print("Model loaded!")
        
        same_scores_arcface = []
        processed = 0
        for a, b in same_pairs:
            if processed % 50 == 0:
                print(f"  Processing same pairs: {processed}/{len(same_pairs)}")
            processed += 1
            
            if not Path(a['face_crop_path']).exists() or not Path(b['face_crop_path']).exists():
                continue
            
            img_a = cv2.imread(a['face_crop_path'])
            img_b = cv2.imread(b['face_crop_path'])
            if img_a is None or img_b is None:
                continue
            
            faces_a = app.get(img_a)
            faces_b = app.get(img_b)
            
            if not faces_a or not faces_b:
                continue
            
            emb_a = faces_a[0].normed_embedding
            emb_b = faces_b[0].normed_embedding
            score = float(np.dot(emb_a, emb_b))
            same_scores_arcface.append(score)
        
        diff_scores_arcface = []
        processed = 0
        for a, b in diff_pairs:
            if processed % 50 == 0:
                print(f"  Processing diff pairs: {processed}/{len(diff_pairs)}")
            processed += 1
            
            if not Path(a['face_crop_path']).exists() or not Path(b['face_crop_path']).exists():
                continue
            
            img_a = cv2.imread(a['face_crop_path'])
            img_b = cv2.imread(b['face_crop_path'])
            if img_a is None or img_b is None:
                continue
            
            faces_a = app.get(img_a)
            faces_b = app.get(img_b)
            
            if not faces_a or not faces_b:
                continue
            
            emb_a = faces_a[0].normed_embedding
            emb_b = faces_b[0].normed_embedding
            score = float(np.dot(emb_a, emb_b))
            diff_scores_arcface.append(score)
        
        metrics_arcface = compute_metrics(same_scores_arcface, diff_scores_arcface)
        print(f"\nSame-person mean: {metrics_arcface['same_mean']:.4f}")
        print(f"Different-person mean: {metrics_arcface['diff_mean']:.4f}")
        print(f"Separation: {metrics_arcface['separation']:.4f}")
        print(f"Best F1: {metrics_arcface['best_f1']:.4f} @ threshold={metrics_arcface['best_threshold']:.2f}")
        print(f"Best Precision: {metrics_arcface['best_precision']:.4f}, Recall: {metrics_arcface['best_recall']:.4f}")
        
    except Exception as e:
        print(f"ArcFace test failed: {e}")
        import traceback
        traceback.print_exc()
        metrics_arcface = None
    
    # Save results
    results = {
        'sface': metrics_sface,
        'arcface': metrics_arcface
    }
    
    with open(f"{OUTPUT_DIR}/sface_vs_arcface_ab_test.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/sface_vs_arcface_ab_test.json")
    
    # Comparison
    if metrics_arcface:
        print("\n" + "="*60)
        print("COMPARISON: SFace vs ArcFace R100")
        print("="*60)
        print(f"{'Metric':<25} {'SFace':>12} {'ArcFace':>12} {'Improvement':>12}")
        print("-"*60)
        print(f"{'Same-person mean':<25} {metrics_sface['same_mean']:>12.4f} {metrics_arcface['same_mean']:>12.4f} {metrics_arcface['same_mean']/metrics_sface['same_mean']:>11.2f}x")
        print(f"{'Diff-person mean':<25} {metrics_sface['diff_mean']:>12.4f} {metrics_arcface['diff_mean']:>12.4f} {metrics_arcface['diff_mean']/metrics_sface['diff_mean']:>11.2f}x")
        print(f"{'Separation':<25} {metrics_sface['separation']:>12.4f} {metrics_arcface['separation']:>12.4f} {metrics_arcface['separation']/metrics_sface['separation']:>11.2f}x")
        print(f"{'Best F1':<25} {metrics_sface['best_f1']:>12.4f} {metrics_arcface['best_f1']:>12.4f} {metrics_arcface['best_f1']/metrics_sface['best_f1']:>11.2f}x")
        print(f"{'Best Precision':<25} {metrics_sface['best_precision']:>12.4f} {metrics_arcface['best_precision']:>12.4f} {metrics_arcface['best_precision']/metrics_sface['best_precision']:>11.2f}x" if metrics_sface['best_precision'] > 0 else f"{'Best Precision':<25} {metrics_sface['best_precision']:>12.4f} {metrics_arcface['best_precision']:>12.4f} {'N/A':>12}")


if __name__ == '__main__':
    main()
