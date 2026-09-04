#!/usr/bin/env python3
"""Phase 4 parallel re-embed V2 — workers write shard files; single DB writer.

Avoids multi-process SQLite write contention.
Resumable: skips rows that already have embedding_v2.
"""
from __future__ import annotations

import argparse
import base64
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
SHARD_DIR = OUT / "reembed_v2_shards"
OUT.mkdir(parents=True, exist_ok=True)
SHARD_DIR.mkdir(parents=True, exist_ok=True)

V1 = "arcface_v1"
V2 = "arcface_v2_aligned"


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
    conn.execute("PRAGMA synchronous=NORMAL")
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


def pack_b64(e: np.ndarray) -> str:
    v = np.asarray(e, dtype=np.float32).reshape(-1)
    raw = struct.pack(f"<{len(v)}f", *v.tolist())
    return base64.b64encode(raw).decode("ascii")


def b64_to_blob(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def _prep_cuda_libs() -> None:
    """Put pip nvidia + WSL libcuda on LD_LIBRARY_PATH for ORT CUDA EP."""
    try:
        import site
        from pathlib import Path as _P

        libs = []
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            base = _P(sp) / "nvidia"
            if base.is_dir():
                for d in base.glob("*/lib"):
                    libs.append(str(d))
        if _P("/usr/lib/wsl/lib").is_dir():
            libs.append("/usr/lib/wsl/lib")
        if libs:
            cur = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = ":".join(libs + ([cur] if cur else []))
    except Exception:
        pass


def worker_shard(payload: tuple) -> dict:
    """Process a list of (id, path) and write one jsonl shard. No DB writes."""
    shard_id, items, use_gpu = payload if len(payload) == 3 else (*payload, False)

    sys.path.insert(0, str(ROOT))
    if use_gpu:
        _prep_cuda_libs()
        os.environ.pop("CIB_FORCE_CPU", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        # CPU-only worker path
        os.environ["CIB_FORCE_CPU"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from backend.app.face_engine_arcface import FaceEngineArcFace

    engine = FaceEngineArcFace()
    engine._init_arcface()
    prov = engine.arcface_session.get_providers() if engine.arcface_session else []
    print(f"worker shard={shard_id} gpu={use_gpu} providers={prov}", flush=True)

    shard_path = SHARD_DIR / f"shard_{shard_id:05d}.jsonl"
    ok = fail = aligned = fallback = 0
    t0 = time.time()
    with shard_path.open("w", encoding="utf-8") as f:
        for oid, path in items:
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
                rec = {"id": oid, "ok": False, "ver": ver}
            else:
                ok += 1
                if ver == V2:
                    aligned += 1
                else:
                    fallback += 1
                rec = {
                    "id": oid,
                    "ok": True,
                    "ver": ver,
                    "emb_b64": pack_b64(emb),
                    "lm": lm,
                }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "shard_id": shard_id,
        "path": str(shard_path),
        "ok": ok,
        "fail": fail,
        "aligned": aligned,
        "fallback": fallback,
        "n": len(items),
        "sec": time.time() - t0,
    }


def apply_shards(db: str, shard_paths: list[str]) -> dict:
    conn = sqlite3.connect(db, timeout=600)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    applied = fail = 0
    batch = []

    def flush():
        nonlocal batch, applied
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
        applied += len(batch)
        batch = []

    for sp in shard_paths:
        with open(sp, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if not rec.get("ok"):
                    fail += 1
                    continue
                blob = b64_to_blob(rec["emb_b64"])
                lm = json.dumps(rec["lm"]) if rec.get("lm") is not None else None
                batch.append((blob, rec["ver"], lm, rec["id"]))
                if len(batch) >= 500:
                    flush()
        flush()
    conn.close()
    return {"applied": applied, "fail_rows": fail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--project-id", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--apply-only", action="store_true", help="Only apply existing shards")
    ap.add_argument(
        "--gpu",
        action="store_true",
        help="Use ORT CUDA EP in workers (prefer 1-2 workers; needs LD_LIBRARY_PATH nvidia libs)",
    )
    args = ap.parse_args()

    ensure_schema(args.db)
    if args.gpu:
        _prep_cuda_libs()
        # GPU: fewer workers avoid VRAM thrash; default down if user left 8
        if args.workers > 2 and "--workers" not in sys.argv:
            args.workers = 2

    if not args.apply_only:
        conn = sqlite3.connect(args.db, timeout=120)
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
        items = [(r[0], r[1]) for r in rows]
        conn.close()
        if args.limit > 0:
            items = items[: args.limit]
        total = len(items)
        print(
            f"to_process={total} workers={args.workers} chunk={args.chunk_size} gpu={args.gpu}",
            flush=True,
        )
        if total == 0:
            print("nothing to embed", flush=True)
        else:
            # fresh shard dir for this run (old shards already applied; keep backups)
            run_tag = time.strftime("%Y%m%d_%H%M%S")
            # keep existing shards; use new sequential ids starting after max
            existing = list(SHARD_DIR.glob("shard_*.jsonl"))
            start_id = 0
            if existing:
                try:
                    start_id = max(int(p.stem.split("_")[1]) for p in existing) + 1
                except Exception:
                    start_id = len(existing)
            chunks = [
                items[i : i + args.chunk_size] for i in range(0, total, args.chunk_size)
            ]
            t0 = time.time()
            done = ok = fail = aligned = fallback = 0
            shard_paths: list[str] = []
            progress_path = OUT / "reembed_v2_progress.json"
            progress_path.write_text(
                json.dumps(
                    {
                        "phase": "starting",
                        "gpu": args.gpu,
                        "total": total,
                        "workers": args.workers,
                        "run_tag": run_tag,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {
                    ex.submit(worker_shard, (start_id + i, ch, bool(args.gpu))): i
                    for i, ch in enumerate(chunks)
                }
                for fut in as_completed(futs):
                    r = fut.result()
                    shard_paths.append(r["path"])
                    done += r["n"]
                    ok += r["ok"]
                    fail += r["fail"]
                    aligned += r["aligned"]
                    fallback += r["fallback"]
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-6)
                    eta = (total - done) / max(rate, 1e-6)
                    prog = {
                        "phase": "embed_shards",
                        "gpu": bool(args.gpu),
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
                        "last_shard": r["path"],
                    }
                    progress_path.write_text(json.dumps(prog, indent=2), encoding="utf-8")
                    print(
                        f"[embed {done}/{total}] ok={ok} align={aligned} fb={fallback} "
                        f"fail={fail} {rate:.2f}/s eta={eta/60:.1f}m shard={r['shard_id']}",
                        flush=True,
                    )

            print("applying shards to DB...", flush=True)
            app = apply_shards(args.db, sorted(shard_paths))
            print("apply", app, flush=True)
    else:
        shard_paths = sorted(str(p) for p in SHARD_DIR.glob("shard_*.jsonl"))
        print(f"apply-only shards={len(shard_paths)}", flush=True)
        app = apply_shards(args.db, shard_paths)
        print("apply", app, flush=True)

    # final counts
    conn = sqlite3.connect(args.db)
    v2 = conn.execute(
        "SELECT COUNT(*) FROM face_observations WHERE embedding_v2 IS NOT NULL"
    ).fetchone()[0]
    aligned_n = conn.execute(
        "SELECT COUNT(*) FROM face_observations WHERE embedding_version=?", (V2,)
    ).fetchone()[0]
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
    conn.close()
    summary = {
        "project_id": args.project_id,
        "v2_total": v2,
        "aligned_total": aligned_n,
        "remain_film_active": remain,
    }
    (OUT / "reembed_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
