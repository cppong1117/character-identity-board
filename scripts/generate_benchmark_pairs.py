#!/usr/bin/env python3
"""
Generate benchmark pairs from CIB database for Model A/B Test.

Creates pairs of face images with labels:
- same_person: True/False
- image1, image2: file paths
- detection1, detection2: detection dicts

Usage:
    python generate_benchmark_pairs.py --project 15 --output benchmark_pairs.json
"""

import argparse
import json
import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


def get_character_observations(
    db_path: str,
    project_id: int,
    min_quality: float = 0.6,
) -> Dict[int, List[dict]]:
    """
    Get face observations grouped by character.
    
    Returns:
        Dict mapping character_id to list of observations
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get characters in project
    cur.execute("""
        SELECT c.id, c.display_name
        FROM characters c
        WHERE c.project_id = ?
    """, (project_id,))
    characters = cur.fetchall()
    
    char_observations = {}
    for char_id, char_name in characters:
        # Get tracklets assigned to this character
        cur.execute("""
            SELECT tracklet_id
            FROM identity_assignments
            WHERE character_id = ?
        """, (char_id,))
        tracklet_ids = [row[0] for row in cur.fetchall()]
        
        if not tracklet_ids:
            continue
        
        # Get observations for these tracklets
        placeholders = ",".join("?" * len(tracklet_ids))
        cur.execute(f"""
            SELECT id, face_crop_path, face_bbox, quality_score, tracklet_id
            FROM face_observations
            WHERE tracklet_id IN ({placeholders})
              AND embedding IS NOT NULL
              AND (quality_score_v2 >= ? OR quality_score >= ?)
              AND excluded = 0
            ORDER BY quality_score_v2 DESC, quality_score DESC
        """, tracklet_ids + [min_quality, min_quality])
        
        observations = []
        for row in cur.fetchall():
            obs_id, crop_path, bbox_json, quality, tracklet_id = row
            if crop_path and os.path.exists(crop_path):
                observations.append({
                    "id": obs_id,
                    "crop_path": crop_path,
                    "bbox": json.loads(bbox_json) if bbox_json else None,
                    "quality": quality or 0,
                    "tracklet_id": tracklet_id,
                })
        
        if observations:
            char_observations[char_id] = {
                "name": char_name,
                "observations": observations,
            }
    
    conn.close()
    return char_observations


def generate_same_person_pairs(
    char_observations: Dict[int, dict],
    max_pairs_per_char: int = 50,
) -> List[dict]:
    """
    Generate same-person pairs (different shots/tracks).
    """
    pairs = []
    
    for char_id, char_data in char_observations.items():
        observations = char_data["observations"]
        if len(observations) < 2:
            continue
        
        # Group by tracklet
        tracklet_groups = {}
        for obs in observations:
            tid = obs["tracklet_id"]
            if tid not in tracklet_groups:
                tracklet_groups[tid] = []
            tracklet_groups[tid].append(obs)
        
        # Need at least 2 tracklets for cross-shot pairs
        tracklet_ids = list(tracklet_groups.keys())
        if len(tracklet_ids) < 2:
            continue
        
        # Sample pairs across tracklets
        count = 0
        for i in range(len(tracklet_ids)):
            for j in range(i + 1, len(tracklet_ids)):
                if count >= max_pairs_per_char:
                    break
                
                tracklet_a = tracklet_groups[tracklet_ids[i]]
                tracklet_b = tracklet_groups[tracklet_ids[j]]
                
                # Pick best quality from each
                obs_a = tracklet_a[0]
                obs_b = tracklet_b[0]
                
                pairs.append({
                    "pair_id": f"same_{char_id}_{count}",
                    "image1": obs_a["crop_path"],
                    "image2": obs_b["crop_path"],
                    "detection1": {"bbox": obs_a["bbox"], "landmarks": None, "score": obs_a["quality"]},
                    "detection2": {"bbox": obs_b["bbox"], "landmarks": None, "score": obs_b["quality"]},
                    "same_person": True,
                    "character_id": char_id,
                    "character_name": char_data["name"],
                })
                count += 1
    
    return pairs


def generate_different_person_pairs(
    char_observations: Dict[int, dict],
    max_pairs: int = 200,
) -> List[dict]:
    """
    Generate different-person pairs.
    """
    pairs = []
    char_ids = list(char_observations.keys())
    
    if len(char_ids) < 2:
        return pairs
    
    count = 0
    attempts = 0
    max_attempts = max_pairs * 10
    
    while count < max_pairs and attempts < max_attempts:
        attempts += 1
        
        # Random pair of different characters
        i, j = random.sample(range(len(char_ids)), 2)
        char_a = char_observations[char_ids[i]]
        char_b = char_observations[char_ids[j]]
        
        if not char_a["observations"] or not char_b["observations"]:
            continue
        
        obs_a = random.choice(char_a["observations"])
        obs_b = random.choice(char_b["observations"])
        
        pairs.append({
            "pair_id": f"diff_{count}",
            "image1": obs_a["crop_path"],
            "image2": obs_b["crop_path"],
            "detection1": {"bbox": obs_a["bbox"], "landmarks": None, "score": obs_a["quality"]},
            "detection2": {"bbox": obs_b["bbox"], "landmarks": None, "score": obs_b["quality"]},
            "same_person": False,
            "character_id_a": char_ids[i],
            "character_id_b": char_ids[j],
        })
        count += 1
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark pairs")
    parser.add_argument("--project", type=int, required=True, help="Project ID")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--min-quality", type=float, default=0.6, help="Min quality score")
    parser.add_argument("--max-pairs-per-char", type=int, default=50)
    parser.add_argument("--max-diff-pairs", type=int, default=200)
    args = parser.parse_args()
    
    db_path = str(Path.home() / "character-identity-board-data/cib.sqlite3")
    
    print(f"Loading character observations from project {args.project}...")
    char_obs = get_character_observations(db_path, args.project, args.min_quality)
    print(f"Found {len(char_obs)} characters with observations")
    
    for char_id, data in char_obs.items():
        print(f"  Character {data['name']}: {len(data['observations'])} observations")
    
    print("\nGenerating same-person pairs...")
    same_pairs = generate_same_person_pairs(char_obs, args.max_pairs_per_char)
    print(f"Generated {len(same_pairs)} same-person pairs")
    
    print("\nGenerating different-person pairs...")
    diff_pairs = generate_different_person_pairs(char_obs, args.max_diff_pairs)
    print(f"Generated {len(diff_pairs)} different-person pairs")
    
    # Combine and shuffle
    all_pairs = same_pairs + diff_pairs
    random.shuffle(all_pairs)
    
    benchmark_data = {
        "project_id": args.project,
        "min_quality": args.min_quality,
        "total_pairs": len(all_pairs),
        "same_person_count": len(same_pairs),
        "different_person_count": len(diff_pairs),
        "characters": {str(k): v["name"] for k, v in char_obs.items()},
        "pairs": all_pairs,
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to: {args.output}")
    print(f"Total pairs: {len(all_pairs)}")


if __name__ == "__main__":
    main()
