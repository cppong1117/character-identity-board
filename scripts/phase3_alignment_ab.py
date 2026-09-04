#!/usr/bin/env python3
"""Phase 3: Alignment A/B — v1 unaligned vs v2 5-point aligned ArcFace.

Precision-first. No threshold changes. No DB writes.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.face_engine_arcface import FaceEngineArcFace  # noqa: E402

DATA = Path.home() / "character-identity-board-data"
DB = DATA / "cib.sqlite3"
OUT = ROOT / "reports/v03_recovery"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)


def parse_bbox(b):
    if b is None:
        return None
    if isinstance(b, (bytes, bytearray)):
        try:
            return json.loads(b.decode())
        except Exception:
            return None
    if isinstance(b, str):
        try:
            return json.loads(b)
        except Exception:
            return None
    return b


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union


def stats(xs):
    a = np.asarray(xs, dtype=np.float64)
    if len(a) == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "p10": None, "p90": None}
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


def far_at(diff, thr):
    if not diff:
        return None
    return float(np.mean(np.asarray(diff) >= thr))


def frr_at(same, thr):
    if not same:
        return None
    return float(np.mean(np.asarray(same) < thr))


def main():
    import sqlite3

    t0 = time.time()
    engine = FaceEngineArcFace()
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    shots = conn.execute(
        """
        SELECT s.id as shot_id, s.shot_number, COUNT(fo.id) n
        FROM shots s
        JOIN videos v ON s.video_id=v.id
        JOIN tracklets t ON t.shot_id=s.id
        JOIN face_observations fo ON fo.tracklet_id=t.id
        WHERE v.project_id=15 AND IFNULL(fo.excluded,0)=0
        GROUP BY s.id HAVING n >= 5
        ORDER BY n DESC LIMIT 80
        """
    ).fetchall()
    top = list(shots[:20])
    rest = list(shots[20:])
    random.shuffle(rest)
    picked = top + rest[:10]
    shot_ids = [r["shot_id"] for r in picked]
    print("shots", len(picked), [r["shot_number"] for r in picked], flush=True)

    # Prefer named-character observations first for pair stats, then fill
    rows = conn.execute(
        f"""
        SELECT fo.id, fo.face_bbox, fo.quality_score, fo.original_frame_ref,
               fo.tracklet_id, s.shot_number, s.representative_frame, s.id as shot_id,
               ia.character_id, ch.display_name, ch.character_code
        FROM face_observations fo
        JOIN tracklets t ON fo.tracklet_id=t.id
        JOIN shots s ON t.shot_id=s.id
        JOIN videos v ON s.video_id=v.id
        JOIN identity_assignments ia ON ia.tracklet_id=t.id
        JOIN characters ch ON ch.id=ia.character_id
        WHERE v.project_id=15 AND IFNULL(fo.excluded,0)=0
          AND s.id IN ({",".join("?" * len(shot_ids))})
          AND ch.display_name IN ('SMY','ZY','DOCTOR','lw','mbq')
          AND fo.quality_score >= 0.65
        ORDER BY fo.quality_score DESC
        LIMIT 4000
        """,
        shot_ids,
    ).fetchall()
    print("named candidates", len(rows), flush=True)

    frame_cache: dict[str, np.ndarray] = {}

    def load_frame(path: str | None):
        if not path or not os.path.exists(path):
            return None
        if path in frame_cache:
            return frame_cache[path]
        img = cv2.imread(path)
        if img is not None and len(frame_cache) < 120:
            frame_cache[path] = img
        return img

    def shot_frame(shot_number: int):
        d = DATA / f"projects/15/analysis/video_15/shot_{int(shot_number):03d}/frames"
        if not d.exists():
            return None
        files = sorted(d.glob("*.jpg"))
        if not files:
            return None
        return str(files[len(files) // 2])

    records = []
    skipped = 0
    budget = 2500  # enough for pairs; faster than 5000 on CPU ArcFace
    det_cache: dict[str, list] = {}

    for r in rows:
        if len(records) >= budget:
            break
        bbox = parse_bbox(r["face_bbox"])
        if not bbox or len(bbox) < 4:
            skipped += 1
            continue
        fp = r["original_frame_ref"] or r["representative_frame"] or shot_frame(r["shot_number"])
        img = load_frame(fp)
        if img is None:
            skipped += 1
            continue

        if fp not in det_cache:
            det_cache[fp] = engine.detect_faces(img)
        dets = det_cache[fp]
        best, bi = None, -1.0
        for d in dets:
            i = iou(bbox, d["bbox"])
            if i > bi:
                bi, best = i, d
        if best is None or bi < 0.2:
            skipped += 1
            continue

        emb_v1 = engine.get_embedding(img, best, aligned=False)
        emb_v2 = engine.get_embedding(img, best, aligned=True)
        if emb_v1 is None or emb_v2 is None:
            skipped += 1
            continue

        records.append(
            {
                "obs_id": r["id"],
                "shot": r["shot_number"],
                "char": r["display_name"],
                "emb_v1": emb_v1.reshape(-1).astype(np.float32),
                "emb_v2": emb_v2.reshape(-1).astype(np.float32),
                "iou": float(bi),
            }
        )
        if len(records) % 100 == 0:
            print(
                f"embedded {len(records)}/{budget} skipped={skipped} "
                f"frames={len(frame_cache)} {time.time() - t0:.1f}s",
                flush=True,
            )

    print(f"records={len(records)} skipped={skipped} time={time.time() - t0:.1f}s", flush=True)

    by_char = defaultdict(list)
    for rec in records:
        by_char[rec["char"]].append(rec)
    print("per_char", {k: len(v) for k, v in by_char.items()}, flush=True)

    same_v1, same_v2 = [], []
    for ch, items in by_char.items():
        if len(items) < 2:
            continue
        for i, a in enumerate(items):
            for b in items[i + 1 : i + 10]:
                same_v1.append(float(np.dot(a["emb_v1"], b["emb_v1"])))
                same_v2.append(float(np.dot(a["emb_v2"], b["emb_v2"])))
                if len(same_v1) >= 3000:
                    break
            if len(same_v1) >= 3000:
                break

    diff_v1, diff_v2 = [], []
    chars = list(by_char.keys())
    for _ in range(9000):
        if len(diff_v1) >= 3000 or len(chars) < 2:
            break
        c1, c2 = random.sample(chars, 2)
        a = random.choice(by_char[c1])
        b = random.choice(by_char[c2])
        diff_v1.append(float(np.dot(a["emb_v1"], b["emb_v1"])))
        diff_v2.append(float(np.dot(a["emb_v2"], b["emb_v2"])))

    s1, s2 = stats(same_v1), stats(same_v2)
    d1, d2 = stats(diff_v1), stats(diff_v2)
    sep1 = (s1["mean"] - d1["mean"]) if s1["mean"] is not None and d1["mean"] is not None else None
    sep2 = (s2["mean"] - d2["mean"]) if s2["mean"] is not None and d2["mean"] is not None else None

    far = {
        "v1@0.40": far_at(diff_v1, 0.40),
        "v2@0.40": far_at(diff_v2, 0.40),
        "v1@0.50": far_at(diff_v1, 0.50),
        "v2@0.50": far_at(diff_v2, 0.50),
    }
    frr = {
        "v1@0.40": frr_at(same_v1, 0.40),
        "v2@0.40": frr_at(same_v2, 0.40),
        "v1@0.50": frr_at(same_v1, 0.50),
        "v2@0.50": frr_at(same_v2, 0.50),
    }

    pass_sep = sep2 is not None and sep1 is not None and sep2 > sep1
    pass_far = (
        far["v2@0.40"] is not None
        and far["v1@0.40"] is not None
        and far["v2@0.40"] <= far["v1@0.40"] + 0.01
    )
    pass_same = s2["mean"] is not None and s1["mean"] is not None and (s2["mean"] >= s1["mean"] - 0.01)
    overall = bool(pass_sep and pass_far and pass_same)

    report = {
        "phase": 3,
        "n_records": len(records),
        "n_shots": len(picked),
        "shot_numbers": [r["shot_number"] for r in picked],
        "per_character_counts": {k: len(v) for k, v in by_char.items()},
        "time_sec": time.time() - t0,
        "same_person": {"v1": s1, "v2": s2},
        "different_person": {"v1": d1, "v2": d2},
        "separation_mean": {"v1": sep1, "v2": sep2, "delta": None if sep1 is None else sep2 - sep1},
        "thresholds_frozen": {"identity": 0.40, "merge": 0.50, "unknown": 0.30},
        "false_merge_rate_FAR": far,
        "false_reject_rate_FRR": frr,
        "pass": {
            "separation_improved": pass_sep,
            "false_merge_not_worse": pass_far,
            "same_person_mean_ok": pass_same,
            "overall_PASS": overall,
        },
    }
    (OUT / "alignment_ab_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Phase 3 — Alignment A/B Test",
        "",
        f"- Records: **{len(records)}** on **{len(picked)}** shots (named chars, q≥0.65)",
        f"- per_char: `{report['per_character_counts']}`",
        f"- time: {report['time_sec']:.1f}s",
        "- thresholds frozen: 0.40 / 0.50 / 0.30",
        "",
        "## Same-person",
        "",
        "| ver | n | mean | median | std | p10 | p90 |",
        "|-----|--:|-----:|-------:|----:|----:|----:|",
        f"| v1 | {s1['n']} | {s1['mean']:.4f} | {s1['median']:.4f} | {s1['std']:.4f} | {s1['p10']:.4f} | {s1['p90']:.4f} |",
        f"| v2 | {s2['n']} | {s2['mean']:.4f} | {s2['median']:.4f} | {s2['std']:.4f} | {s2['p10']:.4f} | {s2['p90']:.4f} |",
        "",
        "## Different-person",
        "",
        "| ver | n | mean | median | std | p10 | p90 |",
        "|-----|--:|-----:|-------:|----:|----:|----:|",
        f"| v1 | {d1['n']} | {d1['mean']:.4f} | {d1['median']:.4f} | {d1['std']:.4f} | {d1['p10']:.4f} | {d1['p90']:.4f} |",
        f"| v2 | {d2['n']} | {d2['mean']:.4f} | {d2['median']:.4f} | {d2['std']:.4f} | {d2['p10']:.4f} | {d2['p90']:.4f} |",
        "",
        f"## Separation: v1={sep1:.4f}  v2={sep2:.4f}  delta={sep2 - sep1:.4f}",
        "",
        "## FAR/FRR",
        "",
        f"| FAR@0.40 | {far['v1@0.40']:.4f} | {far['v2@0.40']:.4f} |",
        f"| FAR@0.50 | {far['v1@0.50']:.4f} | {far['v2@0.50']:.4f} |",
        f"| FRR@0.40 | {frr['v1@0.40']:.4f} | {frr['v2@0.40']:.4f} |",
        f"| FRR@0.50 | {frr['v1@0.50']:.4f} | {frr['v2@0.50']:.4f} |",
        "",
        "## PASS",
        "",
        f"- separation_improved: **{pass_sep}**",
        f"- false_merge_not_worse: **{pass_far}**",
        f"- same_person_mean_ok: **{pass_same}**",
        f"- **OVERALL: {'PASS' if overall else 'FAIL'}**",
        "",
        "Next: Phase 4 full re-embed only if PASS." if overall else "Next: fix alignment/A-B before Phase 4.",
        "",
    ]
    # fix table headers for FAR
    md[md.index("## FAR/FRR") + 2 : md.index("## FAR/FRR") + 2] = [
        "| metric | v1 | v2 |",
        "|--------|---:|---:|",
    ]
    (OUT / "alignment_ab_test.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report["pass"], indent=2), flush=True)
    print("SEP", sep1, sep2, "FAR", far, flush=True)
    print("WROTE", OUT / "alignment_ab_test.md", flush=True)
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
