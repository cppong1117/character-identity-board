#!/usr/bin/env python3
"""
Model A/B Test using existing embeddings from database.

This uses the properly aligned embeddings already computed by the production
pipeline, avoiding the landmark issue.

Usage:
    python model_ab_test_existing.py --project 15
"""

import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def load_embeddings_by_character(
    db_path: str,
    project_id: int,
    min_quality: float = 0.6,
    max_per_char: int = 100,
) -> Dict[int, dict]:
    """
    Load embeddings grouped by character.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get characters
    cur.execute("""
        SELECT c.id, c.display_name
        FROM characters c
        WHERE c.project_id = ?
    """, (project_id,))
    characters = cur.fetchall()
    
    char_data = {}
    for char_id, char_name in characters:
        # Get tracklets assigned to this character
        cur.execute("""
            SELECT tracklet_id FROM identity_assignments
            WHERE character_id = ?
        """, (char_id,))
        tracklet_ids = [row[0] for row in cur.fetchall()]
        
        if not tracklet_ids:
            continue
        
        # Get embeddings
        placeholders = ",".join("?" * len(tracklet_ids))
        cur.execute(f"""
            SELECT embedding, embedding_dim, quality_score_v2, tracklet_id
            FROM face_observations
            WHERE tracklet_id IN ({placeholders})
              AND embedding IS NOT NULL
              AND quality_score_v2 >= ?
              AND excluded = 0
            ORDER BY quality_score_v2 DESC
            LIMIT ?
        """, tracklet_ids + [min_quality, max_per_char])
        
        embeddings = []
        seen_tracklets = set()
        for row in cur.fetchall():
            emb_bytes, emb_dim, quality, tracklet_id = row
            
            # Only take one per tracklet for diversity
            if tracklet_id in seen_tracklets:
                continue
            seen_tracklets.add(tracklet_id)
            
            # Convert bytes to numpy array
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            if emb_dim and len(emb) == emb_dim:
                embeddings.append({
                    "embedding": emb,
                    "quality": quality or 0,
                    "tracklet_id": tracklet_id,
                })
        
        if embeddings:
            char_data[char_id] = {
                "name": char_name,
                "embeddings": embeddings,
            }
    
    conn.close()
    return char_data


def generate_pairs(
    char_data: Dict[int, dict],
    same_person_pairs: int = 300,
    diff_person_pairs: int = 200,
) -> Tuple[List[dict], List[dict]]:
    """
    Generate same-person and different-person pairs from existing embeddings.
    """
    same_pairs = []
    diff_pairs = []
    
    char_ids = list(char_data.keys())
    
    # Same-person pairs (across tracklets)
    for char_id in char_ids:
        embs = char_data[char_id]["embeddings"]
        if len(embs) < 2:
            continue
        
        tracklet_groups = {}
        for e in embs:
            tid = e["tracklet_id"]
            if tid not in tracklet_groups:
                tracklet_groups[tid] = e
        
        tracklet_list = list(tracklet_groups.values())
        for i in range(len(tracklet_list)):
            for j in range(i + 1, min(len(tracklet_list), i + 20)):
                if len(same_pairs) >= same_person_pairs:
                    break
                same_pairs.append({
                    "emb1": tracklet_list[i]["embedding"],
                    "emb2": tracklet_list[j]["embedding"],
                    "same_person": True,
                    "character_name": char_data[char_id]["name"],
                })
            if len(same_pairs) >= same_person_pairs:
                break
    
    # Different-person pairs
    count = 0
    attempts = 0
    while count < diff_person_pairs and attempts < diff_person_pairs * 10:
        attempts += 1
        if len(char_ids) < 2:
            break
        
        i, j = np.random.choice(len(char_ids), 2, replace=False)
        embs_a = char_data[char_ids[i]]["embeddings"]
        embs_b = char_data[char_ids[j]]["embeddings"]
        
        if not embs_a or not embs_b:
            continue
        
        e1 = embs_a[np.random.randint(len(embs_a))]
        e2 = embs_b[np.random.randint(len(embs_b))]
        
        diff_pairs.append({
            "emb1": e1["embedding"],
            "emb2": e2["embedding"],
            "same_person": False,
        })
        count += 1
    
    return same_pairs, diff_pairs


def compute_metrics(
    same_scores: List[float],
    diff_scores: List[float],
) -> dict:
    """
    Compute precision, recall, FAR, FRR at various thresholds.
    """
    thresholds = np.arange(0.10, 0.95, 0.02)
    metrics = {"thresholds": []}
    
    for threshold in thresholds:
        tp = sum(1 for s in same_scores if s >= threshold)
        fn = sum(1 for s in same_scores if s < threshold)
        fp = sum(1 for s in diff_scores if s >= threshold)
        tn = sum(1 for s in diff_scores if s < threshold)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics["thresholds"].append({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "far": float(far),
            "frr": float(frr),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        })
    
    # Distribution stats
    metrics["same_person"] = {
        "mean": float(np.mean(same_scores)) if same_scores else 0,
        "median": float(np.median(same_scores)) if same_scores else 0,
        "std": float(np.std(same_scores)) if same_scores else 0,
        "min": float(np.min(same_scores)) if same_scores else 0,
        "max": float(np.max(same_scores)) if same_scores else 0,
    }
    metrics["different_person"] = {
        "mean": float(np.mean(diff_scores)) if diff_scores else 0,
        "median": float(np.median(diff_scores)) if diff_scores else 0,
        "std": float(np.std(diff_scores)) if diff_scores else 0,
        "min": float(np.min(diff_scores)) if diff_scores else 0,
        "max": float(np.max(diff_scores)) if diff_scores else 0,
    }
    
    # Separation
    separation = metrics["same_person"]["mean"] - metrics["different_person"]["mean"]
    metrics["separation"] = float(separation)
    
    # Best precision at >=99% recall
    best_prec_99 = 0
    for t in metrics["thresholds"]:
        if t["recall"] >= 0.99 and t["precision"] > best_prec_99:
            best_prec_99 = t["precision"]
    metrics["best_precision_at_99_recall"] = float(best_prec_99)
    
    # Best precision at >=95% recall
    best_prec_95 = 0
    for t in metrics["thresholds"]:
        if t["recall"] >= 0.95 and t["precision"] > best_prec_95:
            best_prec_95 = t["precision"]
    metrics["best_precision_at_95_recall"] = float(best_prec_95)
    
    return metrics


def generate_report(metrics: dict, output_path: str):
    """Generate markdown report."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# SFace Existing Embeddings Benchmark\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## Score Distribution\n\n")
        f.write("| Metric | Same Person | Different Person |\n")
        f.write("|--------|-------------|------------------|\n")
        sp = metrics["same_person"]
        dp = metrics["different_person"]
        f.write(f"| Mean | {sp['mean']:.4f} | {dp['mean']:.4f} |\n")
        f.write(f"| Median | {sp['median']:.4f} | {dp['median']:.4f} |\n")
        f.write(f"| Std | {sp['std']:.4f} | {dp['std']:.4f} |\n")
        f.write(f"| Min | {sp['min']:.4f} | {dp['min']:.4f} |\n")
        f.write(f"| Max | {sp['max']:.4f} | {dp['max']:.4f} |\n")
        
        f.write(f"\n**Separation**: {metrics['separation']:.4f}\n")
        f.write(f"**Best Precision @ 99% Recall**: {metrics['best_precision_at_99_recall']:.4f}\n")
        f.write(f"**Best Precision @ 95% Recall**: {metrics['best_precision_at_95_recall']:.4f}\n")
        
        f.write("\n## Threshold Performance\n\n")
        f.write("| Threshold | Precision | Recall | FAR | FRR | TP | FN | FP | TN |\n")
        f.write("|-----------|-----------|--------|-----|-----|----|----|----|----|\n")
        for t in metrics["thresholds"]:
            f.write(
                f"| {t['threshold']:.2f} | {t['precision']:.4f} | {t['recall']:.4f} | "
                f"{t['far']:.4f} | {t['frr']:.4f} | {t['tp']} | {t['fn']} | {t['fp']} | {t['tn']} |\n"
            )
        
        f.write("\n## Key Findings\n\n")
        if metrics["separation"] < 0.1:
            f.write("- ❌ **Separation too low** (< 0.1): SFace cannot reliably distinguish same vs different person across shots\n")
        if metrics["best_precision_at_99_recall"] < 0.99:
            f.write(f"- ❌ **Precision@99%Recall = {metrics['best_precision_at_99_recall']:.2%}** < 99% target\n")
        if sp["mean"] < 0.3:
            f.write(f"- ❌ **Same-person mean = {sp['mean']:.4f}** is very low\n")
        
        f.write("\n## Conclusion\n\n")
        if metrics["separation"] >= 0.3 and metrics["best_precision_at_99_recall"] >= 0.99:
            f.write("**SFACE IS VIABLE** for production use.\n")
        elif metrics["separation"] >= 0.15:
            f.write("**SFACE MARGINAL** - may work with careful threshold tuning but review rate will be high.\n")
        else:
            f.write("**SFACE INSUFFICIENT** - Model replacement recommended.\n")


def main():
    db_path = str(Path.home() / "character-identity-board-data/cib.sqlite3")
    output_dir = "reports/v02_accuracy_recovery"
    project_id = 15
    
    print(f"Loading embeddings from project {project_id}...")
    char_data = load_embeddings_by_character(db_path, project_id)
    
    total_embs = sum(len(d["embeddings"]) for d in char_data.values())
    print(f"Loaded {total_embs} embeddings from {len(char_data)} characters")
    
    for cid, d in char_data.items():
        print(f"  {d['name']}: {len(d['embeddings'])} embeddings")
    
    print("\nGenerating pairs...")
    same_pairs, diff_pairs = generate_pairs(char_data)
    print(f"Same-person pairs: {len(same_pairs)}")
    print(f"Different-person pairs: {len(diff_pairs)}")
    
    print("\nComputing similarities...")
    same_scores = [float(np.dot(p["emb1"], p["emb2"])) for p in same_pairs]
    diff_scores = [float(np.dot(p["emb1"], p["emb2"])) for p in diff_pairs]
    
    print("\nComputing metrics...")
    metrics = compute_metrics(same_scores, diff_scores)
    
    print(f"\n=== RESULTS ===")
    print(f"Same-person mean: {metrics['same_person']['mean']:.4f}")
    print(f"Different-person mean: {metrics['different_person']['mean']:.4f}")
    print(f"Separation: {metrics['separation']:.4f}")
    print(f"Best Precision @ 99% Recall: {metrics['best_precision_at_99_recall']:.4f}")
    print(f"Best Precision @ 95% Recall: {metrics['best_precision_at_95_recall']:.4f}")
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "sface_existing_embeddings_report.md")
    generate_report(metrics, report_path)
    print(f"\nReport saved to: {report_path}")
    
    # Save raw metrics
    metrics_path = os.path.join(output_dir, "sface_existing_embeddings_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
