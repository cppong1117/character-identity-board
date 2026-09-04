#!/usr/bin/env python3
"""Phase 4: Full ArcFace re-embed V2 (5-point aligned) for CIB.

Practical path for film project:
  Most observations only have face_crop_path (no original_frame_ref / sparse frames).
  Re-detect landmarks ON THE FACE CROP with YuNet, then similarity-align to 112,
  then ArcFace -> embedding_v2.

Keeps:
  embedding / embedding_v1 = arcface_v1 (legacy)
Writes:
  embedding_v2, embedding_version, landmarks_json

No threshold changes. No reassignment. Resumable.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.face_engine_arcface import (  # noqa: E402
    EMBEDDING_VERSION_V1,
    EMBEDDING_VERSION_V2,
    FaceEngineArcFace,
)

DATA = Path.home() / "character-identity-board-data"
DEFAULT_DB = DATA / "cib.sqlite3"
OUT = ROOT / "reports/v03_recovery"
OUT.mkdir(parents=True, exist_ok=True)


def ensure_schema(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(face_observations)")}
    for col, typ in [
        ("embedding_v1", "BLOB"),
        ("embedding_v2", "BLOB"),
        ("embedding_version", "TEXT"),
        ("landmarks_json", "TEXT"),
    ]:
        if col not in cols:
            conn.execute(f"ALTER TABLE face_observations ADD COLUMN {col} {typ}")
    conn.commit()


def pack_emb(e: np.ndarray) -> bytes:
    v = np.asarray(e, dtype=np.float32).reshape(-1)
    return struct.pack(f"<{len(v)}f", *v.tolist())


def embed_from_crop(engine: FaceEngineArcFace, crop_bgr: np.ndarray) -> tuple[np.ndarray | None, str, str | None]:
    """Return (emb, version_tag, landmarks_json)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None, "fail_empty", None

    h, w = crop_bgr.shape[:2]
    dets = engine.detect_faces(crop_bgr)
    if dets:
        # pick highest score / largest
        det = max(dets, key=lambda d: (d.get("score") or 0) * (d["bbox"][2] * d["bbox"][3]))
        dbg = engine.get_embedding_debug(crop_bgr, det)
        if dbg.get("embedding_v2") is not None and dbg.get("align_ok"):
            return (
                dbg["embedding_v2"],
                EMBEDDING_VERSION_V2,
                json.dumps(dbg.get("landmarks")),
            )
        if dbg.get("embedding_v1") is not None:
            return dbg["embedding_v1"], f"{EMBEDDING_VERSION_V2}_fallback_unaligned", json.dumps(dbg.get("landmarks"))

    # No det on crop: treat whole crop as face, unaligned resize
    det = {"bbox": [0.0, 0.0, float(w), float(h)], "landmarks": None, "score": 0.0}
    emb = engine.get_embedding(crop_bgr, det, aligned=False)
    if emb is None:
        return None, "fail_embed", None
    return emb, f"{EMBEDDING_VERSION_V2}_fallback_unaligned", None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--project-id", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-excluded", action="store_true")
    ap.add_argument("--batch-commit", type=int, default=100)
    ap.add_argument("--only-missing", action="store_true", default=True)
    args = ap.parse_args()

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)

    n_copy = conn.execute(
        """
        UPDATE face_observations
        SET embedding_v1 = embedding,
            embedding_version = COALESCE(embedding_version, ?)
        WHERE embedding IS NOT NULL AND embedding_v1 IS NULL
        """,
        (EMBEDDING_VERSION_V1,),
    ).rowcount
    conn.commit()
    print(f"copied embedding->embedding_v1 rows={n_copy}", flush=True)

    where = ["fo.face_crop_path IS NOT NULL"]
    params: list = []
    if not args.include_excluded:
        where.append("IFNULL(fo.excluded,0)=0")
    where.append("(fo.embedding_v2 IS NULL)")
    where.append("v.project_id=?")
    params.append(args.project_id)

    sql = f"""
    SELECT fo.id, fo.face_crop_path
    FROM face_observations fo
    JOIN tracklets t ON fo.tracklet_id=t.id
    JOIN shots s ON t.shot_id=s.id
    JOIN videos v ON s.video_id=v.id
    WHERE {' AND '.join(where)}
    ORDER BY fo.id
    """
    if args.limit > 0:
        sql += f" LIMIT {int(args.limit)}"

    rows = conn.execute(sql, params).fetchall()
    total = len(rows)
    print(f"to_process={total} project={args.project_id}", flush=True)

    engine = FaceEngineArcFace()
    ok = fail = aligned = fallback = 0
    t0 = time.time()
    progress_path = OUT / "reembed_v2_progress.json"

    for i, r in enumerate(rows, 1):
        path = r["face_crop_path"]
        emb = None
        ver = "fail"
        lm = None
        if path and os.path.exists(path):
            img = cv2.imread(path)
            emb, ver, lm = embed_from_crop(engine, img)
        if emb is None:
            fail += 1
        else:
            conn.execute(
                """
                UPDATE face_observations
                SET embedding_v2=?, embedding_version=?, landmarks_json=COALESCE(?, landmarks_json)
                WHERE id=?
                """,
                (pack_emb(emb), ver, lm, r["id"]),
            )
            ok += 1
            if ver == EMBEDDING_VERSION_V2:
                aligned += 1
            else:
                fallback += 1

        if i % args.batch_commit == 0 or i == total:
            conn.commit()
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            eta = (total - i) / max(rate, 1e-6)
            prog = {
                "done": i,
                "total": total,
                "ok": ok,
                "fail": fail,
                "aligned": aligned,
                "fallback": fallback,
                "rate_per_s": rate,
                "eta_min": eta / 60,
                "elapsed_sec": elapsed,
            }
            progress_path.write_text(json.dumps(prog, indent=2), encoding="utf-8")
            print(
                f"[{i}/{total}] ok={ok} align={aligned} fb={fallback} fail={fail} "
                f"{rate:.2f}/s eta={eta/60:.1f}m",
                flush=True,
            )

    summary = {
        "project_id": args.project_id,
        "ok": ok,
        "fail": fail,
        "aligned": aligned,
        "fallback": fallback,
        "total": total,
        "elapsed_sec": time.time() - t0,
        "primary_embedding_column": "still_v1",
        "v2_column": "embedding_v2",
        "note": "No cutover of identity matcher yet; Phase 5+ will consume embedding_v2",
    }
    (OUT / "reembed_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
