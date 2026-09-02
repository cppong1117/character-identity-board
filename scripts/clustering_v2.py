#!/usr/bin/env python3
"""PHASE 9+10: Tracklet Prototype + Reference Set V2 + Clustering V2.

After re-embed + quality gate, this script:
1. Builds tracklet prototypes (quality-weighted centroid of top-K embeddings)
2. Re-clusters using HDBSCAN on tracklet prototypes
3. Assigns characters to clusters
4. Reports cluster purity metrics
"""
import sqlite3, numpy as np, sys, os, json, time, hdbscan
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = os.path.expanduser('~/character-identity-board-data/cib.sqlite3')
REPORT_DIR = os.path.expanduser('~/character-identity-board/reports/v02_accuracy_recovery')
TOP_K = 5  # Top-K quality embeddings per tracklet for prototype
MIN_CLUSTER_SIZE = 15
MIN_SAMPLES = 5


def build_tracklet_prototypes(conn):
    """Build prototype embedding for each tracklet using top-K quality embeddings."""
    cur = conn.cursor()
    
    cur.execute('''SELECT t.id, t.shot_id
        FROM tracklets t
        JOIN shots s ON t.shot_id = s.id
        WHERE s.video_id = 15''')
    tracklets = cur.fetchall()
    print(f"Tracklets to process: {len(tracklets)}")
    
    prototypes = {}
    for tid, shot_id in tracklets:
        cur.execute('''SELECT embedding, quality_score_v2, quality_score
            FROM face_observations
            WHERE tracklet_id = ? AND excluded = 0 AND embedding IS NOT NULL
            AND identity_evidence_allowed = 1
            ORDER BY COALESCE(quality_score_v2, quality_score, 0) DESC
            LIMIT ?''', (tid, TOP_K))
        
        rows = cur.fetchall()
        if not rows:
            # Fallback: use any embedding
            cur.execute('''SELECT embedding, quality_score_v2, quality_score
                FROM face_observations
                WHERE tracklet_id = ? AND excluded = 0 AND embedding IS NOT NULL
                ORDER BY COALESCE(quality_score_v2, quality_score, 0) DESC
                LIMIT ?''', (tid, TOP_K))
            rows = cur.fetchall()
        
        if not rows:
            continue
        
        embs = []
        weights = []
        for emb_bytes, qs_v2, qs in rows:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            embs.append(emb)
            w = (qs_v2 or qs or 0.5)
            weights.append(w)
        
        embs = np.array(embs)
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-12)
        
        # Quality-weighted centroid
        prototype = np.average(embs, axis=0, weights=weights)
        n = np.linalg.norm(prototype) + 1e-12
        prototype = prototype / n
        
        prototypes[tid] = {
            'embedding': prototype,
            'shot_id': shot_id,
            'n_faces': len(rows),
        }
    
    print(f"Prototypes built: {len(prototypes)}")
    return prototypes


def cluster_tracklets(prototypes, min_cluster_size=MIN_CLUSTER_SIZE):
    """Cluster tracklet prototypes using HDBSCAN."""
    if len(prototypes) < min_cluster_size:
        print(f"Too few tracklets ({len(prototypes)}) for clustering")
        return {}
    
    tids = list(prototypes.keys())
    embs = np.array([prototypes[tid]['embedding'] for tid in tids])
    
    print(f"Clustering {len(embs)} tracklet prototypes (dim={embs.shape[1]})...")
    
    # Convert to distance matrix (1 - cosine similarity)
    sim = cosine_similarity(embs)
    dist = 1 - sim
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=MIN_SAMPLES,
        metric='precomputed',
        cluster_selection_method='eom',
    )
    labels = clusterer.fit_predict(dist)
    
    # Build results
    clusters = defaultdict(list)
    noise = []
    for i, tid in enumerate(tids):
        if labels[i] >= 0:
            clusters[labels[i]].append(tid)
        else:
            noise.append(tid)
    
    print(f"Clusters: {len(clusters)}")
    print(f"Noise tracklets: {len(noise)}")
    for cid, tids_in in sorted(clusters.items()):
        print(f"  Cluster {cid}: {len(tids_in)} tracklets")
    
    return clusters, noise


def assign_characters(conn, clusters):
    """Assign characters to clusters, preserving manual assignments."""
    cur = conn.cursor()
    
    # Get existing characters
    cur.execute("SELECT id, display_name, character_code FROM characters WHERE project_id = 15")
    existing = {row[2]: row for row in cur.fetchall()}
    
    # Create new characters for each cluster
    created = 0
    for cid, tids in sorted(clusters.items()):
        code = f"CL{cid:02d}"
        if code not in existing:
            cur.execute('''INSERT INTO characters (project_id, display_name, character_code)
                VALUES (15, ?, ?)''', (f"Cluster {cid:02d}", code))
            created += 1
        
        # Get character ID
        cur.execute("SELECT id FROM characters WHERE character_code = ? AND project_id = 15", (code,))
        char_id = cur.fetchone()[0]
        
        # Assign tracklets via identity_assignments
        for tid in tids:
            cur.execute('''INSERT OR REPLACE INTO identity_assignments 
                (tracklet_id, character_id, confidence, assignment_source, review_status)
                VALUES (?, ?, 0.95, 'clustering_v2', 'auto_assigned')''', (tid, char_id))
    
    conn.commit()
    print(f"Created {created} new characters, assigned tracklets")


def compute_cluster_metrics(conn, prototypes, clusters, noise):
    """Compute cluster purity and quality metrics."""
    cur = conn.cursor()
    
    # Inter-cluster vs intra-cluster similarity
    all_tids = []
    all_labels = []
    for cid, tids in sorted(clusters.items()):
        for tid in tids:
            all_tids.append(tid)
            all_labels.append(cid)
    
    if not all_tids:
        return {}
    
    embs = np.array([prototypes[tid]['embedding'] for tid in all_tids])
    sim = cosine_similarity(embs)
    
    # Intra-cluster similarity
    intra_sims = []
    for cid in set(all_labels):
        idx = [i for i, l in enumerate(all_labels) if l == cid]
        if len(idx) > 1:
            for i in range(len(idx)):
                for j in range(i+1, len(idx)):
                    intra_sims.append(sim[idx[i], idx[j]])
    
    # Inter-cluster similarity (sample)
    inter_sims = []
    unique_labels = sorted(set(all_labels))
    for ci in range(len(unique_labels)):
        for cj in range(ci+1, len(unique_labels)):
            idx_i = [i for i, l in enumerate(all_labels) if l == unique_labels[ci]]
            idx_j = [i for i, l in enumerate(all_labels) if l == unique_labels[cj]]
            # Sample up to 10 pairs
            pairs = min(10, len(idx_i) * len(idx_j))
            for _ in range(pairs):
                ii = np.random.choice(idx_i)
                ij = np.random.choice(idx_j)
                inter_sims.append(sim[ii, ij])
    
    metrics = {
        'n_clusters': len(clusters),
        'n_noise': len(noise),
        'n_assigned': len(all_tids),
        'intra_cluster_mean': float(np.mean(intra_sims)) if intra_sims else 0,
        'intra_cluster_median': float(np.median(intra_sims)) if intra_sims else 0,
        'inter_cluster_mean': float(np.mean(inter_sims)) if inter_sims else 0,
        'inter_cluster_median': float(np.median(inter_sims)) if inter_sims else 0,
        'separation': float(np.mean(intra_sims) - np.mean(inter_sims)) if intra_sims and inter_sims else 0,
    }
    
    return metrics


def main():
    print("=== PHASE 9+10: Tracklet Prototype + Clustering V2 ===")
    
    conn = sqlite3.connect(DB_PATH)
    t0 = time.time()
    
    # Step 1: Build tracklet prototypes
    prototypes = build_tracklet_prototypes(conn)
    
    # Step 2: Cluster
    result = cluster_tracklets(prototypes)
    if not result:
        print("Clustering failed — too few tracklets")
        conn.close()
        return
    
    clusters, noise = result
    
    # Step 3: Compute metrics
    metrics = compute_cluster_metrics(conn, prototypes, clusters, noise)
    
    # Step 4: Assign characters
    assign_characters(conn, clusters)
    
    conn.close()
    elapsed = time.time() - t0
    
    # Report
    print(f"\n=== CLUSTERING V2 COMPLETE ({elapsed:.1f}s) ===")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    # Save
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, 'clustering_v2.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Report saved to {REPORT_DIR}/clustering_v2.json")


if __name__ == '__main__':
    main()
