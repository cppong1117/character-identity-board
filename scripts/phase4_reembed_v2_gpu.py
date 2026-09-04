#!/usr/bin/env python3
"""Phase 4 single-process GPU re-embed (ORT CUDA).

Multiprocess workers lose CUDA device visibility on this WSL host.
Single process + A6000 ≈ 100–200 faces/s after warm-up.
Resumable: skips rows that already have embedding_v2.
"""
from __future__ import annotations

import argparse
import json
import os
import site
import sqlite3
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "character-identity-board-data"
DEFAULT_DB = str(DATA / "cib.sqlite3")
OUT = ROOT / "reports/v03_recovery"
OUT.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT / "reembed_v2_progress.json"

V1 = "arcface_v1"
V2 = "arcface_v2_aligned"


def prep_cuda() -> None:
    libs = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        base = Path(sp) / "nvidia"
        if base.is_dir():
            for d in base.glob("*/lib"):
                libs.append(str(d))
    if Path("/usr/lib/wsl/lib").is_dir():
        libs.append("/usr/lib/wsl/lib")
    if libs:
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(libs + ([cur] if cur else []))
    # Touch torch CUDA before ORT (stabilizes device on WSL)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda not available")
    _ = torch.zeros(1, device="cuda")
    print(f"torch_cuda={torch.cuda.get_device_name(0)}", flush=True)


def ensure_schema(db: str) -> None:
    conn = sqlite3.connect(db, timeout=120)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(face_observations)")}
    for col, typ in [
        ("embedding_v1", "BLOB"),
        ("embedding_v2", "BLOB"),
        ("embedding_version", "TEXT"),
        ("landmarks_json", "TEXT"),
    ]:
        if col not in cols:
            conn.execute(f"ALTER TABLE face_observations ADD COLUMN {col} {typ}")
    conn.execute("PRAGMA journal_mode=WAL")
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


def pack(e: np.ndarray) -> bytes:
    v = np.asarray(e, dtype=np.float32).reshape(-1)
    return struct.pack(f"<{len(v)}f", *v.tolist())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--project-id", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-commit", type=int, default=200)
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    prep_cuda()
    ensure_schema(args.db)

    from backend.app.face_engine_arcface import FaceEngineArcFace

    engine = FaceEngineArcFace()
    engine._init_arcface()
    prov = engine.arcface_session.get_providers()
    print(f"providers={prov}", flush=True)
    if "CUDAExecutionProvider" not in prov:
        print("ERROR: not on GPU — abort (use CPU shards runner instead)", flush=True)
        return 2

    conn = sqlite3.connect(args.db, timeout=600)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    rows = conn.execute(
        """
        SELECT fo.id, fo.face_crop_path
        FROM face_observations fo
        JOIN tracklets t ON fo.tracklet_id=t.id
        JOIN shots s ON t.shot_id=s.id
        JOIN videos v ON s.video_id=v.id
        WHERE v.project_id=? AND IFNULL(fo.excluded,0)=0
          AND fo.face_crop_path IS NOT NULL AND fo.embedding_v2 IS NULL
        ORDER BY fo.id
        """,
        (args.project_id,),
    ).fetchall()
    if args.limit > 0:
        rows = rows[: args.limit]
    total = len(rows)
    print(f"to_process={total} batch_commit={args.batch_commit}", flush=True)
    if total == 0:
        print("nothing to embed", flush=True)
        conn.close()
        return 0

    t0 = time.time()
    ok = fail = aligned = fallback = 0
    batch: list[tuple] = []

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        conn.executemany(
            """
            UPDATE face_observations
            SET embedding_v2=?, embedding_version=?,
                landmarks_json=COALESCE(?, landmarks_json)
            WHERE id=? AND embedding_v2 IS NULL
            """,
            batch,
        )
        conn.commit()
        batch = []

    def write_progress(done: int) -> None:
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-6)
        eta = (total - done) / max(rate, 1e-6)
        prog = {
            "phase": "embed_gpu_single",
            "gpu": True,
            "providers": prov,
            "done": done,
            "total": total,
            "ok": ok,
            "fail": fail,
            "aligned": aligned,
            "fallback": fallback,
            "rate_per_s": rate,
            "eta_min": eta / 60,
            "elapsed_sec": elapsed,
            "workers": 1,
            "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        PROGRESS.write_text(json.dumps(prog, indent=2), encoding="utf-8")

    for i, (oid, path) in enumerate(rows, 1):
        emb = None
        ver = "fail"
        lm = None
        try:
            if path and os.path.exists(path):
                img = cv2.imread(path)
                if img is not None and img.size > 0:
                    h, w = img.shape[:2]
                    dets = engine.detect_faces(img)
                    if dets:
                        det = max(
                            dets,
                            key=lambda d: (d.get("score") or 0)
                            * (d["bbox"][2] * d["bbox"][3] + 1),
                        )
                        dbg = engine.get_embedding_debug(img, det)
                        if dbg.get("embedding_v2") is not None and dbg.get("align_ok"):
                            emb = dbg["embedding_v2"]
                            ver = V2
                            lm = dbg.get("landmarks")
                        elif dbg.get("embedding_v1") is not None:
                            emb = dbg["embedding_v1"]
                            ver = f"{V2}_fallback_unaligned"
                            lm = dbg.get("landmarks")
                    if emb is None:
                        det = {
                            "bbox": [0.0, 0.0, float(w), float(h)],
                            "landmarks": None,
                            "score": 0.0,
                        }
                        emb = engine.get_embedding(img, det, aligned=False)
                        ver = f"{V2}_fallback_unaligned"
        except Exception as e:
            emb = None
            ver = f"fail:{type(e).__name__}"

        if emb is None:
            fail += 1
        else:
            ok += 1
            if ver == V2:
                aligned += 1
            else:
                fallback += 1
            lm_json = json.dumps(lm) if lm is not None else None
            batch.append((pack(emb), ver, lm_json, oid))
            if len(batch) >= args.batch_commit:
                flush()

        if i % args.progress_every == 0 or i == total:
            flush()
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            eta = (total - i) / max(rate, 1e-6)
            write_progress(i)
            print(
                f"[{i}/{total}] ok={ok} align={aligned} fb={fallback} fail={fail} "
                f"{rate:.1f}/s eta={eta/60:.1f}m",
                flush=True,
            )

    flush()
    write_progress(total)
    remain = conn.execute(
        """
        SELECT COUNT(*) FROM face_observations fo
        JOIN tracklets t ON fo.tracklet_id=t.id
        JOIN shots s ON t.shot_id=s.id
        JOIN videos v ON s.video_id=v.id
        WHERE v.project_id=? AND IFNULL(fo.excluded,0)=0 AND fo.embedding_v2 IS NULL
        """,
        (args.project_id,),
    ).fetchone()[0]
    v2 = conn.execute(
        "SELECT COUNT(*) FROM face_observations WHERE embedding_v2 IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    summary = {
        "project_id": args.project_id,
        "v2_total": v2,
        "remain_film_active": remain,
        "ok": ok,
        "fail": fail,
        "aligned": aligned,
        "fallback": fallback,
        "elapsed_sec": time.time() - t0,
        "rate_per_s": total / max(time.time() - t0, 1e-6),
        "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (OUT / "reembed_v2_gpu_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
