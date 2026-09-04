#!/usr/bin/env python3
"""Phase 4 parallel re-embed V2 — multi-process CPU workers, resumable."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "character-identity-board-data"
DEFAULT_DB = str(DATA / "cib.sqlite3")
OUT = ROOT / "reports/v03_recovery"
OUT.mkdir(parents=True, exist_ok=True)

V1 = "arcface_v1"
V2 = "arcface_v2_aligned"


def ensure_schema(db: str) -> None:
    conn = sqlite3.connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(face_observations)")}
    for col, typ in [
        ("embedding_v1", "BLOB"),
        ("embedding_v2", "BLOB"),
        ("embedding_version", "TEXT"),
        ("landmarks_json", "TEXT"),
    ]:
        if col not in cols:
            conn.execute(f"ALTER TABLE face_observations ADD COLUMN {col} {typ}")
    n = conn.execute(
        """
        UPDATE face_observations
        SET embedding_v1 = embedding,
            embedding_version = COALESCE(embedding_version, ?)
        WHERE embedding IS NOT NULL AND embedding_v1 IS NULL
        """,
        (V1,),
    ).rowcount
    conn.commit()
    conn.close()
    print(f"schema ok; copied v1 rows={n}", flush=True)


def pack_emb(e: np.ndarray) -> bytes:
    v = np.asarray(e, dtype=np.float32).reshape(-1)
    return struct.pack(f"<{len(v)}f", *v.tolist())


def worker_chunk(args: tuple) -> dict:
    db, ids = args
    sys.path.insert(0, str(ROOT))
    from backend.app.face_engine_arcface import FaceEngineArcFace

    engine = FaceEngineArcFace()
    conn = sqlite3.connect(db, timeout=120)
    conn.row_factory = sqlite3.Row
    ok = fail = aligned = fallback = 0
    batch = []

    def flush():
        nonlocal batch
        if not batch:
            return
        conn.executemany(
            """
            UPDATE face_observations
            SET embedding_v2=?, embedding_version=?, landmarks_json=COALESCE(?, landmarks_json)
            WHERE id=?
            """,
            batch,
        )
        conn.commit()
        batch = []

    for oid in ids:
        row = conn.execute(
            "SELECT id, face_crop_path FROM face_observations WHERE id=?", (oid,)
        ).fetchone()
        if row is None:
            fail += 1
            continue
        path = row["face_crop_path"]
        emb = None
        ver = "fail"
        lm = None
        if path and os.path.exists(path):
            img = cv2.imread(path)
            if img is not None and img.size > 0:
                h, w = img.shape[:2]
                dets = engine.detect_faces(img)
                if dets:
                    det = max(
                        dets,
                        key=lambda d: (d.get("score") or 0) * (d["bbox"][2] * d["bbox"][3] + 1),
                    )
                    dbg = engine.get_embedding_debug(img, det)
                    if dbg.get("embedding_v2") is not None and dbg.get("align_ok"):
                        emb = dbg["embedding_v2"]
                        ver = V2
                        lm = json.dumps(dbg.get("landmarks"))
                    elif dbg.get("embedding_v1") is not None:
                        emb = dbg["embedding_v1"]
                        ver = f"{V2}_fallback_unaligned"
                        lm = json.dumps(dbg.get("landmarks"))
                if emb is None:
                    det = {"bbox": [0.0, 0.0, float(w), float(h)], "landmarks": None, "score": 0.0}
                    emb = engine.get_embedding(img, det, aligned=False)
                    ver = f"{V2}_fallback_unaligned"
        if emb is None:
            fail += 1
        else:
            batch.append((pack_emb(emb), ver, lm, oid))
            ok += 1
            if ver == V2:
                aligned += 1
            else:
                fallback += 1
            if len(batch) >= 40:
                flush()
    flush()
    conn.close()
    return {"ok": ok, "fail": fail, "aligned": aligned, "fallback": fallback, "n": len(ids)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--project-id", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ensure_schema(args.db)
    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        """
        SELECT fo.id FROM face_observations fo
        JOIN tracklets t ON fo.tracklet_id=t.id
        JOIN shots s ON t.shot_id=s.id
        JOIN videos v ON s.video_id=v.id
        WHERE v.project_id=? AND IFNULL(fo.excluded,0)=0
          AND fo.face_crop_path IS NOT NULL AND fo.embedding_v2 IS NULL
        ORDER BY fo.id
        """,
        (args.project_id,),
    ).fetchall()
    ids = [r[0] for r in rows]
    if args.limit > 0:
        ids = ids[: args.limit]
    conn.close()
    total = len(ids)
    print(f"to_process={total} workers={args.workers} chunk={args.chunk_size}", flush=True)
    if total == 0:
        print("nothing to do", flush=True)
        return 0

    chunks = [ids[i : i + args.chunk_size] for i in range(0, total, args.chunk_size)]
    t0 = time.time()
    done = ok = fail = aligned = fallback = 0
    progress_path = OUT / "reembed_v2_progress.json"

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker_chunk, (args.db, ch)) for ch in chunks]
        for fut in as_completed(futs):
            r = fut.result()
            done += r["n"]
            ok += r["ok"]
            fail += r["fail"]
            aligned += r["aligned"]
            fallback += r["fallback"]
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta = (total - done) / max(rate, 1e-6)
            prog = {
                "done": done,
                "total": total,
                "ok": ok,
                "fail": fail,
                "aligned": aligned,
                "fallback": fallback,
                "rate_per_s": rate,
                "eta_min": eta / 60,
                "elapsed_sec": elapsed,
                "workers": args.workers,
            }
            progress_path.write_text(json.dumps(prog, indent=2), encoding="utf-8")
            print(
                f"[{done}/{total}] ok={ok} align={aligned} fb={fallback} fail={fail} "
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
        "workers": args.workers,
    }
    (OUT / "reembed_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
