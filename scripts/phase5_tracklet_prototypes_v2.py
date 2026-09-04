#!/usr/bin/env python3
"""Phase 5: Tracklet prototypes from embedding_v2 + contamination audit.

RULES:
- Does NOT lower thresholds
- Does NOT overwrite manual/confirmed assignments
- Does NOT recluster / reassign yet (Phase later)

Outputs:
- reports/v03_recovery/tracklet_prototypes_v2.npz  (tid -> 512d)
- reports/v03_recovery/tracklet_prototypes_v2.json  (metadata)
- reports/v03_recovery/phase5_prototype_audit.md
- SQLite table tracklet_prototypes_v2 (created if missing)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "character-identity-board-data"
DEFAULT_DB = str(DATA / "cib.sqlite3")
OUT = ROOT / "reports/v03_recovery"
OUT.mkdir(parents=True, exist_ok=True)

TOP_K = 5
# Within-tracklet pairwise cosine below this → candidate MIXED_TRACKLET
MIXED_MIN_PAIR = 0.35
# Fraction of low-sim pairs to flag mixed
MIXED_FRAC = 0.30


def unpack(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.asarray(struct.unpack(f"<{n}f", blob), dtype=np.float32)


def pack(e: np.ndarray) -> bytes:
    v = np.asarray(e, dtype=np.float32).reshape(-1)
    return struct.pack(f"<{len(v)}f", *v.tolist())


def l2(e: np.ndarray) -> np.ndarray:
    e = np.asarray(e, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(e)) + 1e-12
    return e / n


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tracklet_prototypes_v2 (
            tracklet_id INTEGER PRIMARY KEY,
            project_id INTEGER,
            shot_id INTEGER,
            n_faces INTEGER,
            n_used INTEGER,
            quality_mean REAL,
            purity_min_pair REAL,
            purity_mean_pair REAL,
            mixed_flag INTEGER DEFAULT 0,
            embedding BLOB NOT NULL,
            embedding_dim INTEGER DEFAULT 512,
            source_version TEXT,
            updated_at TEXT
        )
        """
    )
    conn.commit()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--project-id", type=int, default=15)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args()

    t0 = time.time()
    conn = sqlite3.connect(args.db, timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_table(conn)

    # film tracklets
    tracklets = conn.execute(
        """
        SELECT t.id, t.shot_id
        FROM tracklets t
        JOIN shots s ON t.shot_id = s.id
        JOIN videos v ON s.video_id = v.id
        WHERE v.project_id = ?
        ORDER BY t.id
        """,
        (args.project_id,),
    ).fetchall()
    print(f"tracklets={len(tracklets)} project={args.project_id}", flush=True)

    # current assignment map
    assigns = {
        r[0]: (r[1], r[2], r[3], r[4])
        for r in conn.execute(
            """
            SELECT ia.tracklet_id, ia.character_id, ch.display_name, ia.assignment_source, ia.review_status
            FROM identity_assignments ia
            JOIN characters ch ON ch.id = ia.character_id
            """
        )
    }

    protos: dict[int, dict] = {}
    meta_rows = []
    skipped_no_emb = 0
    mixed = 0
    pure = 0

    for i, (tid, shot_id) in enumerate(tracklets, 1):
        rows = conn.execute(
            """
            SELECT id, embedding_v2, embedding_version,
                   COALESCE(quality_score_v2, quality_score, 0.5),
                   IFNULL(identity_evidence_allowed, 0)
            FROM face_observations
            WHERE tracklet_id = ? AND IFNULL(excluded,0)=0 AND embedding_v2 IS NOT NULL
            ORDER BY
              CASE WHEN embedding_version = 'arcface_v2_aligned' THEN 0 ELSE 1 END,
              CASE WHEN IFNULL(identity_evidence_allowed,0)=1 THEN 0 ELSE 1 END,
              COALESCE(quality_score_v2, quality_score, 0) DESC
            LIMIT ?
            """,
            (tid, max(args.top_k * 3, 15)),  # fetch more for purity; use top-k for proto
        ).fetchall()
        if not rows:
            skipped_no_emb += 1
            continue

        # prefer aligned + evidence for prototype pool
        pool = [r for r in rows if r[2] == "arcface_v2_aligned"] or list(rows)
        pool = pool[: args.top_k]
        embs = []
        weights = []
        for _oid, blob, ver, qs, _ev in pool:
            e = l2(unpack(blob))
            embs.append(e)
            weights.append(float(qs) if qs and qs > 0 else 0.5)
        embs_a = np.stack(embs, axis=0)
        w = np.asarray(weights, dtype=np.float64)
        w = w / (w.sum() + 1e-12)
        proto = l2(np.average(embs_a, axis=0, weights=w))

        # purity on up to 15 faces (all fetched)
        all_e = np.stack([l2(unpack(r[1])) for r in rows], axis=0)
        if len(all_e) >= 2:
            sim = all_e @ all_e.T
            iu = np.triu_indices(len(all_e), k=1)
            pairs = sim[iu]
            pmin = float(pairs.min())
            pmean = float(pairs.mean())
            low_frac = float((pairs < MIXED_MIN_PAIR).mean())
            is_mixed = 1 if (low_frac >= MIXED_FRAC and pmin < MIXED_MIN_PAIR) else 0
        else:
            pmin = pmean = 1.0
            low_frac = 0.0
            is_mixed = 0
        if is_mixed:
            mixed += 1
        else:
            pure += 1

        ch = assigns.get(tid)
        rec = {
            "tracklet_id": tid,
            "shot_id": shot_id,
            "n_faces_scored": len(rows),
            "n_used": len(pool),
            "quality_mean": float(np.mean(weights)),
            "purity_min_pair": pmin,
            "purity_mean_pair": pmean,
            "mixed_flag": is_mixed,
            "character_id": ch[0] if ch else None,
            "character_name": ch[1] if ch else None,
            "assignment_source": ch[2] if ch else None,
            "review_status": ch[3] if ch else None,
            "manual_locked": bool(
                ch
                and (
                    ch[2] == "manual"
                    or ch[3] == "confirmed"
                    or (ch[1] and str(ch[1]).lower() != "unknown" and ch[3] == "confirmed")
                )
            ),
        }
        protos[tid] = {"embedding": proto, **rec}
        meta_rows.append(rec)

        conn.execute(
            """
            INSERT INTO tracklet_prototypes_v2(
              tracklet_id, project_id, shot_id, n_faces, n_used, quality_mean,
              purity_min_pair, purity_mean_pair, mixed_flag, embedding, embedding_dim,
              source_version, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(tracklet_id) DO UPDATE SET
              project_id=excluded.project_id,
              shot_id=excluded.shot_id,
              n_faces=excluded.n_faces,
              n_used=excluded.n_used,
              quality_mean=excluded.quality_mean,
              purity_min_pair=excluded.purity_min_pair,
              purity_mean_pair=excluded.purity_mean_pair,
              mixed_flag=excluded.mixed_flag,
              embedding=excluded.embedding,
              embedding_dim=excluded.embedding_dim,
              source_version=excluded.source_version,
              updated_at=excluded.updated_at
            """,
            (
                tid,
                args.project_id,
                shot_id,
                len(rows),
                len(pool),
                rec["quality_mean"],
                pmin,
                pmean,
                is_mixed,
                pack(proto),
                512,
                "arcface_v2_aligned",
                datetime.now().astimezone().isoformat(timespec="seconds"),
            ),
        )
        if i % 200 == 0:
            conn.commit()
            print(f"  {i}/{len(tracklets)} protos={len(protos)} mixed={mixed}", flush=True)

    conn.commit()

    # character centroids from locked/named tracklets (not Unknown)
    char_embs: dict[int, list] = defaultdict(list)
    char_names: dict[int, str] = {}
    for tid, p in protos.items():
        cid = p.get("character_id")
        name = p.get("character_name") or ""
        if not cid or name.lower() == "unknown":
            continue
        # only use non-mixed for reference-ish centroid
        if p.get("mixed_flag"):
            continue
        char_embs[cid].append(p["embedding"])
        char_names[cid] = name
    char_cent = {cid: l2(np.mean(np.stack(vs), axis=0)) for cid, vs in char_embs.items() if vs}

    # Unknown recovery potential: prototype sim to best named centroid
    unknown_stats = {"n": 0, "ge_0.50": 0, "ge_0.40": 0, "ge_0.35": 0, "lt_0.30": 0}
    unknown_hits = []
    for tid, p in protos.items():
        name = (p.get("character_name") or "").lower()
        if name != "unknown":
            continue
        if p.get("mixed_flag"):
            continue
        unknown_stats["n"] += 1
        if not char_cent:
            continue
        sims = {cid: float(p["embedding"] @ c) for cid, c in char_cent.items()}
        best_cid = max(sims.keys(), key=lambda k: sims[k])
        best = sims[best_cid]
        # margin vs second
        ordered = sorted(sims.values(), reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        if best >= 0.50:
            unknown_stats["ge_0.50"] += 1
        if best >= 0.40:
            unknown_stats["ge_0.40"] += 1
        if best >= 0.35:
            unknown_stats["ge_0.35"] += 1
        if best < 0.30:
            unknown_stats["lt_0.30"] += 1
        if best >= 0.40:
            unknown_hits.append(
                {
                    "tracklet_id": tid,
                    "best_character_id": best_cid,
                    "best_name": char_names.get(best_cid),
                    "sim": round(best, 4),
                    "margin": round(margin, 4),
                    "manual_locked": p.get("manual_locked"),
                }
            )

    unknown_hits.sort(key=lambda x: -x["sim"])

    # save npz
    if protos:
        tids = sorted(protos.keys())
        mat = np.stack([protos[t]["embedding"] for t in tids], axis=0)
        np.savez_compressed(OUT / "tracklet_prototypes_v2.npz", tracklet_ids=np.array(tids), embeddings=mat)

    summary = {
        "project_id": args.project_id,
        "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - t0, 2),
        "tracklets_total": len(tracklets),
        "prototypes_built": len(protos),
        "skipped_no_emb": skipped_no_emb,
        "mixed_flagged": mixed,
        "pure": pure,
        "top_k": args.top_k,
        "mixed_min_pair": MIXED_MIN_PAIR,
        "mixed_frac": MIXED_FRAC,
        "named_character_centroids": {str(k): char_names[k] for k in char_cent},
        "unknown_nonmixed": unknown_stats,
        "unknown_recoverable_at_0.40_count": unknown_stats["ge_0.40"],
        "note": "NO reassignment performed. Thresholds unchanged. Manual locks respected.",
    }
    (OUT / "tracklet_prototypes_v2_meta.json").write_text(
        json.dumps({"summary": summary, "unknown_top_hits": unknown_hits[:50], "mixed_examples": [r for r in meta_rows if r["mixed_flag"]][:30]}, indent=2),
        encoding="utf-8",
    )

    # markdown report
    md = f"""# Phase 5 — Tracklet Prototypes V2 + Contamination Audit

- Finished: `{summary['finished']}`
- Project: film (`{args.project_id}`)
- Source: `embedding_v2` (prefer `arcface_v2_aligned`)
- Top-K quality-weighted centroid: **{args.top_k}**
- **No reassignment / no recluster / thresholds frozen**

## Counts

| metric | value |
|--------|------:|
| tracklets total | {len(tracklets)} |
| prototypes built | {len(protos)} |
| skipped (no v2 emb) | {skipped_no_emb} |
| pure | {pure} |
| **mixed_flag** | **{mixed}** ({100*mixed/max(len(protos),1):.1f}%) |

Mixed rule: ≥{int(MIXED_FRAC*100)}% of within-tracklet pairs have cosine < {MIXED_MIN_PAIR}.

## Unknown recovery potential (read-only)

Among non-mixed Unknown tracklets (n={unknown_stats['n']}):

| best sim to named centroid | count |
|----------------------------|------:|
| ≥ 0.50 | {unknown_stats['ge_0.50']} |
| ≥ 0.40 (identity thr) | {unknown_stats['ge_0.40']} |
| ≥ 0.35 | {unknown_stats['ge_0.35']} |
| < 0.30 | {unknown_stats['lt_0.30']} |

These are **candidates** for later calibrated matching — **not auto-applied**.

Named centroids used: {', '.join(f'{n}(id={i})' for i,n in char_names.items() if i in char_cent) or 'none'}

### Top recoverable Unknown (preview)

| tracklet | best char | sim | margin |
|---------:|-----------|----:|-------:|
"""
    for h in unknown_hits[:20]:
        md += f"| {h['tracklet_id']} | {h['best_name']} | {h['sim']:.3f} | {h['margin']:.3f} |\n"
    md += f"""

## Artifacts

- `tracklet_prototypes_v2` SQLite table
- `reports/v03_recovery/tracklet_prototypes_v2.npz`
- `reports/v03_recovery/tracklet_prototypes_v2_meta.json`
- this report

## Next (Phase 6+)

1. Quality evidence gate on faces feeding prototypes
2. Reference bank from locked high-purity tracklets
3. Margin-based matching + calibration
4. Safe reassignment (skip manual_locked)
"""
    (OUT / "phase5_prototype_audit.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("WROTE", OUT / "phase5_prototype_audit.md", flush=True)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
