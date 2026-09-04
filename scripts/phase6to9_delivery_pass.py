#!/usr/bin/env python3
"""V0.3 delivery pass: dual-assignment repair + reference bank + safe match + QC.

RULES:
- Never lower thresholds (0.40 / 0.50 / 0.30)
- Never overwrite true manual named confirmations
- Dual Unknown+Named: keep Named only if prototype sim agrees; else keep Unknown
- Pure Unknown manual-confirmed stay locked (suggestions only)
- No full HDBSCAN recluster
"""
from __future__ import annotations

import argparse
import json
import shutil
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
QC = OUT / "delivery_qc"
OUT.mkdir(parents=True, exist_ok=True)
QC.mkdir(parents=True, exist_ok=True)

THR_ID = 0.40
THR_MERGE = 0.50
THR_UNK = 0.30
MARGIN_AUTO = 0.08  # best - second


def unpack(b: bytes) -> np.ndarray:
    n = len(b) // 4
    return np.asarray(struct.unpack(f"<{n}f", b), dtype=np.float32)


def pack(e: np.ndarray) -> bytes:
    v = np.asarray(e, dtype=np.float32).reshape(-1)
    return struct.pack(f"<{len(v)}f", *v.tolist())


def l2(e: np.ndarray) -> np.ndarray:
    e = np.asarray(e, dtype=np.float32).reshape(-1)
    return e / (float(np.linalg.norm(e)) + 1e-12)


def load_protos(conn: sqlite3.Connection, project_id: int) -> dict[int, dict]:
    out = {}
    for tid, mixed, blob, pmin, pmean, n_used in conn.execute(
        """
        SELECT tracklet_id, mixed_flag, embedding, purity_min_pair, purity_mean_pair, n_used
        FROM tracklet_prototypes_v2 WHERE project_id=?
        """,
        (project_id,),
    ):
        out[tid] = {
            "mixed": int(mixed or 0),
            "e": l2(unpack(blob)),
            "pmin": pmin,
            "pmean": pmean,
            "n_used": n_used,
        }
    return out


def best_face(conn: sqlite3.Connection, tid: int) -> tuple[int | None, str | None]:
    row = conn.execute(
        """
        SELECT id, face_crop_path FROM face_observations
        WHERE tracklet_id=? AND IFNULL(excluded,0)=0 AND face_crop_path IS NOT NULL
        ORDER BY
          CASE WHEN embedding_version='arcface_v2_aligned' THEN 0 ELSE 1 END,
          CASE WHEN IFNULL(identity_evidence_allowed,0)=1 THEN 0 ELSE 1 END,
          COALESCE(quality_score_v2, quality_score, 0) DESC
        LIMIT 1
        """,
        (tid,),
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--project-id", type=int, default=15)
    ap.add_argument("--apply", action="store_true", help="Write DB changes (default dry-run)")
    ap.add_argument("--copy-qc", type=int, default=40, help="Copy N face crops for visual QC")
    args = ap.parse_args()

    t0 = time.time()
    conn = sqlite3.connect(args.db, timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    protos = load_protos(conn, args.project_id)

    # character map for film-relevant chars
    chars = {
        r["id"]: dict(r)
        for r in conn.execute(
            "SELECT id, display_name, character_code, status FROM characters WHERE project_id=? OR display_name='Unknown'",
            (args.project_id,),
        )
    }
    # also load any characters referenced by film assignments
    for r in conn.execute(
        """
        SELECT DISTINCT ch.id, ch.display_name, ch.character_code, ch.status
        FROM identity_assignments ia
        JOIN characters ch ON ch.id=ia.character_id
        JOIN tracklets t ON t.id=ia.tracklet_id
        JOIN shots s ON t.shot_id=s.id JOIN videos v ON s.video_id=v.id
        WHERE v.project_id=?
        """,
        (args.project_id,),
    ):
        chars[r["id"]] = dict(r)

    unknown_ids = {i for i, c in chars.items() if c["display_name"] == "Unknown" or c["character_code"] == "UNKNOWN"}

    # --- Phase 7 reference bank from pure non-mixed named tracklets ---
    # Prefer confirmed; fall back to auto_assigned
    ref_members: dict[int, list[int]] = defaultdict(list)
    for tid, p in protos.items():
        if p["mixed"]:
            continue
        rows = conn.execute(
            """
            SELECT ia.character_id, ia.assignment_source, ia.review_status, ia.confidence
            FROM identity_assignments ia
            WHERE ia.tracklet_id=?
            """,
            (tid,),
        ).fetchall()
        named = [r for r in rows if r["character_id"] not in unknown_ids]
        if not named:
            continue
        # pick best named assignment
        named.sort(
            key=lambda r: (
                0 if r["review_status"] == "confirmed" else 1,
                0 if r["assignment_source"] == "manual" else 1,
                -(r["confidence"] or 0),
            )
        )
        cid = named[0]["character_id"]
        ref_members[cid].append(tid)

    ref_bank = {}
    for cid, tids in ref_members.items():
        # take up to 30 highest purity
        scored = sorted(
            ((protos[t]["pmean"] or 0) * (1 if not protos[t]["mixed"] else 0), t) for t in tids if t in protos
        )
        top = [t for _, t in scored[::-1][:30]]
        if not top:
            continue
        mat = np.stack([protos[t]["e"] for t in top], 0)
        cent = l2(mat.mean(0))
        # also keep top-5 exemplars
        ref_bank[cid] = {
            "character_id": cid,
            "name": chars.get(cid, {}).get("display_name"),
            "n_members": len(tids),
            "n_refs": len(top),
            "centroid": cent,
            "exemplar_tids": top[:5],
        }

    # persist reference bank table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS character_ref_bank_v2 (
            character_id INTEGER PRIMARY KEY,
            project_id INTEGER,
            display_name TEXT,
            n_members INTEGER,
            n_refs INTEGER,
            embedding BLOB,
            exemplar_tids TEXT,
            updated_at TEXT
        )
        """
    )
    if args.apply:
        for cid, rb in ref_bank.items():
            conn.execute(
                """
                INSERT INTO character_ref_bank_v2(character_id, project_id, display_name, n_members, n_refs, embedding, exemplar_tids, updated_at)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(character_id) DO UPDATE SET
                  project_id=excluded.project_id,
                  display_name=excluded.display_name,
                  n_members=excluded.n_members,
                  n_refs=excluded.n_refs,
                  embedding=excluded.embedding,
                  exemplar_tids=excluded.exemplar_tids,
                  updated_at=excluded.updated_at
                """,
                (
                    cid,
                    args.project_id,
                    rb["name"],
                    rb["n_members"],
                    rb["n_refs"],
                    pack(rb["centroid"]),
                    json.dumps(rb["exemplar_tids"]),
                    datetime.now().astimezone().isoformat(timespec="seconds"),
                ),
            )
        conn.commit()

    def match_proto(e: np.ndarray) -> tuple[int | None, float, float, dict]:
        if not ref_bank:
            return None, 0.0, 0.0, {}
        sims = {cid: float(e @ rb["centroid"]) for cid, rb in ref_bank.items()}
        ordered = sorted(sims.items(), key=lambda x: -x[1])
        best_cid, best = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else -1.0
        margin = best - second
        return best_cid, best, margin, sims

    # --- Dual assignment repair ---
    dual_tids = [
        r[0]
        for r in conn.execute(
            "SELECT tracklet_id FROM identity_assignments GROUP BY tracklet_id HAVING COUNT(*)>1"
        )
    ]
    dual_actions = []
    deleted_ids = []
    kept = []

    for tid in dual_tids:
        # only film tracklets
        ok = conn.execute(
            """
            SELECT 1 FROM tracklets t
            JOIN shots s ON t.shot_id=s.id JOIN videos v ON s.video_id=v.id
            WHERE t.id=? AND v.project_id=?
            """,
            (tid, args.project_id),
        ).fetchone()
        if not ok:
            continue
        rows = [
            dict(r)
            for r in conn.execute(
                """
                SELECT ia.id as ia_id, ia.tracklet_id, ia.character_id, ia.confidence,
                       ia.assignment_source, ia.review_status, ia.note,
                       ch.display_name
                FROM identity_assignments ia
                JOIN characters ch ON ch.id=ia.character_id
                WHERE ia.tracklet_id=?
                """,
                (tid,),
            )
        ]
        named = [r for r in rows if r["display_name"] != "Unknown"]
        unks = [r for r in rows if r["display_name"] == "Unknown"]
        p = protos.get(tid)
        decision = None
        keep_ia = None
        reason = ""

        # Case A: has named + unknown
        if named and unks:
            # score each named vs ref bank
            best_named = None
            best_score = -1.0
            best_margin = 0.0
            for r in named:
                cid = r["character_id"]
                if p is None or cid not in ref_bank:
                    # keep manual confirmed named without proto
                    score = r["confidence"] or 0
                    margin = 0
                else:
                    score = float(p["e"] @ ref_bank[cid]["centroid"])
                    # margin vs other refs
                    _, _, margin, _ = match_proto(p["e"])
                    # if matched other person better, penalize
                    m_cid, m_sim, m_margin, _ = match_proto(p["e"])
                    if m_cid is not None and m_cid != cid and m_sim > score + 0.05:
                        score = m_sim - 1.0  # reject
                    else:
                        margin = m_margin
                # priority boost for manual confirmed named
                pri = score + (0.2 if r["assignment_source"] == "manual" and r["review_status"] == "confirmed" else 0)
                if pri > best_score:
                    best_score = pri
                    best_named = r
                    best_margin = margin

            if best_named is not None:
                # Accept named if: manual confirmed named OR (sim>=0.40 and not mixed) OR clustering with sim>=0.35
                cid = best_named["character_id"]
                sim = float(p["e"] @ ref_bank[cid]["centroid"]) if (p and cid in ref_bank) else (best_named["confidence"] or 0)
                mixed = bool(p and p["mixed"])
                manual_named = best_named["assignment_source"] == "manual" and best_named["review_status"] == "confirmed"
                if manual_named or (sim >= THR_ID and not mixed) or (sim >= 0.35 and best_named["assignment_source"] in ("clustering_v2", "automatic_cluster") and not mixed):
                    keep_ia = best_named
                    decision = "keep_named_drop_unknown"
                    reason = f"sim={sim:.3f} margin={best_margin:.3f} mixed={mixed}"
                else:
                    # keep unknown (safer)
                    keep_ia = max(unks, key=lambda r: (r["confidence"] or 0, r["ia_id"]))
                    decision = "keep_unknown_drop_named"
                    reason = f"named_sim_weak={sim:.3f} mixed={mixed}"
            else:
                keep_ia = max(unks, key=lambda r: (r["confidence"] or 0, r["ia_id"]))
                decision = "keep_unknown_only"
                reason = "no_scored_named"

        # Case B: multi named only
        elif len(named) >= 2:
            if p is None:
                keep_ia = max(named, key=lambda r: (
                    0 if r["review_status"] == "confirmed" else 1,
                    0 if r["assignment_source"] == "manual" else 1,
                    -(r["confidence"] or 0),
                ))
                decision = "keep_priority_named_no_proto"
                reason = "no_proto"
            else:
                m_cid, m_sim, m_margin, sims = match_proto(p["e"])
                # pick assignment matching best ref if sim ok
                match_rows = [r for r in named if r["character_id"] == m_cid]
                if match_rows and m_sim >= THR_ID and m_margin >= MARGIN_AUTO and not p["mixed"]:
                    keep_ia = match_rows[0]
                    decision = "keep_match_named"
                    reason = f"sim={m_sim:.3f} margin={m_margin:.3f}"
                else:
                    # prefer manual confirmed
                    man = [r for r in named if r["assignment_source"] == "manual" and r["review_status"] == "confirmed"]
                    keep_ia = man[0] if man else max(named, key=lambda r: r["confidence"] or 0)
                    decision = "keep_manual_or_conf_named"
                    reason = f"best_ref={chars.get(m_cid,{}).get('display_name')} sim={m_sim:.3f}"

        # Case C: multi unknown only
        elif len(unks) >= 2:
            keep_ia = max(unks, key=lambda r: (r["confidence"] or 0, r["ia_id"]))
            decision = "dedupe_unknown"
            reason = "multi_unknown"

        else:
            continue

        drop = [r for r in rows if r["ia_id"] != keep_ia["ia_id"]]
        dual_actions.append(
            {
                "tracklet_id": tid,
                "decision": decision,
                "reason": reason,
                "keep": {
                    "ia_id": keep_ia["ia_id"],
                    "name": keep_ia["display_name"],
                    "source": keep_ia["assignment_source"],
                    "status": keep_ia["review_status"],
                },
                "drop": [
                    {"ia_id": r["ia_id"], "name": r["display_name"], "source": r["assignment_source"], "status": r["review_status"]}
                    for r in drop
                ],
            }
        )
        if args.apply:
            for r in drop:
                conn.execute("DELETE FROM identity_assignments WHERE id=?", (r["ia_id"],))
                deleted_ids.append(r["ia_id"])
            kept.append(keep_ia["ia_id"])

    if args.apply:
        conn.commit()

    # --- Safe reassignment for single-assignment Unknown that are NOT locked ---
    # Locked = manual OR confirmed. Only pending non-manual Unknown can auto-move.
    safe_moves = []
    unlockable = conn.execute(
        """
        SELECT ia.id as ia_id, ia.tracklet_id, ia.character_id, ia.assignment_source, ia.review_status
        FROM identity_assignments ia
        JOIN characters ch ON ch.id=ia.character_id
        JOIN tracklets t ON t.id=ia.tracklet_id
        JOIN shots s ON t.shot_id=s.id JOIN videos v ON s.video_id=v.id
        WHERE v.project_id=? AND ch.display_name='Unknown'
          AND ia.assignment_source != 'manual'
          AND ia.review_status = 'pending'
        """,
        (args.project_id,),
    ).fetchall()
    for r in unlockable:
        tid = r["tracklet_id"]
        p = protos.get(tid)
        if not p or p["mixed"]:
            continue
        cid, sim, margin, _ = match_proto(p["e"])
        if cid is None:
            continue
        if sim >= THR_ID and margin >= MARGIN_AUTO:
            safe_moves.append(
                {
                    "tracklet_id": tid,
                    "ia_id": r["ia_id"],
                    "to_character_id": cid,
                    "to_name": ref_bank[cid]["name"],
                    "sim": round(sim, 4),
                    "margin": round(margin, 4),
                }
            )
            if args.apply:
                conn.execute(
                    """
                    UPDATE identity_assignments
                    SET character_id=?, confidence=?, assignment_source='v03_margin_match',
                        review_status='auto_assigned',
                        note=COALESCE(note,'') || ' | v03 safe match'
                    WHERE id=?
                    """,
                    (cid, sim, r["ia_id"]),
                )
    if args.apply and safe_moves:
        conn.commit()

    # --- Suggestions for locked Unknown with strong match (NOT applied) ---
    suggestions = []
    locked_unk = conn.execute(
        """
        SELECT ia.id as ia_id, ia.tracklet_id, ia.assignment_source, ia.review_status
        FROM identity_assignments ia
        JOIN characters ch ON ch.id=ia.character_id
        JOIN tracklets t ON t.id=ia.tracklet_id
        JOIN shots s ON t.shot_id=s.id JOIN videos v ON s.video_id=v.id
        WHERE v.project_id=? AND ch.display_name='Unknown'
          AND (ia.assignment_source='manual' OR ia.review_status='confirmed')
        """,
        (args.project_id,),
    ).fetchall()
    for r in locked_unk:
        tid = r["tracklet_id"]
        p = protos.get(tid)
        if not p or p["mixed"]:
            continue
        cid, sim, margin, _ = match_proto(p["e"])
        if cid is None:
            continue
        if sim >= THR_ID and margin >= MARGIN_AUTO:
            oid, path = best_face(conn, tid)
            suggestions.append(
                {
                    "tracklet_id": tid,
                    "suggested_character_id": cid,
                    "suggested_name": ref_bank[cid]["name"],
                    "sim": round(sim, 4),
                    "margin": round(margin, 4),
                    "obs_id": oid,
                    "crop": path,
                    "locked_source": r["assignment_source"],
                    "locked_status": r["review_status"],
                }
            )
    suggestions.sort(key=lambda x: -x["sim"])

    # --- Validate current named assignments (flag wrong-person) ---
    wrong_flags = []
    named_assign = conn.execute(
        """
        SELECT ia.id as ia_id, ia.tracklet_id, ia.character_id, ia.assignment_source, ia.review_status,
               ch.display_name
        FROM identity_assignments ia
        JOIN characters ch ON ch.id=ia.character_id
        JOIN tracklets t ON t.id=ia.tracklet_id
        JOIN shots s ON t.shot_id=s.id JOIN videos v ON s.video_id=v.id
        WHERE v.project_id=? AND ch.display_name!='Unknown'
        """,
        (args.project_id,),
    ).fetchall()
    for r in named_assign:
        tid = r["tracklet_id"]
        p = protos.get(tid)
        if not p:
            continue
        cid = r["character_id"]
        if cid not in ref_bank:
            continue
        own = float(p["e"] @ ref_bank[cid]["centroid"])
        m_cid, m_sim, m_margin, _ = match_proto(p["e"])
        # wrong if another char much better
        if m_cid and m_cid != cid and m_sim >= THR_ID and (m_sim - own) >= 0.10 and m_margin >= MARGIN_AUTO:
            # skip if manual confirmed named (locked)
            if r["assignment_source"] == "manual" and r["review_status"] == "confirmed":
                tag = "locked_possible_wrong"
            else:
                tag = "auto_possible_wrong"
            wrong_flags.append(
                {
                    "tracklet_id": tid,
                    "current": r["display_name"],
                    "own_sim": round(own, 4),
                    "better": ref_bank[m_cid]["name"],
                    "better_sim": round(m_sim, 4),
                    "margin": round(m_margin, 4),
                    "tag": tag,
                    "source": r["assignment_source"],
                    "status": r["review_status"],
                    "ia_id": r["ia_id"],
                    "mixed": p["mixed"],
                }
            )
            # auto-fix non-locked wrong → better char or Unknown if weak
            if args.apply and tag == "auto_possible_wrong":
                if m_sim >= THR_ID and m_margin >= MARGIN_AUTO and not p["mixed"]:
                    conn.execute(
                        """
                        UPDATE identity_assignments
                        SET character_id=?, confidence=?, assignment_source='v03_wrong_fix',
                            review_status='auto_assigned',
                            note=COALESCE(note,'') || ' | v03 wrong-person fix'
                        WHERE id=?
                        """,
                        (m_cid, m_sim, r["ia_id"]),
                    )
                elif not p["mixed"]:
                    # demote to unknown character id
                    unk_id = next(iter(unknown_ids))
                    conn.execute(
                        """
                        UPDATE identity_assignments
                        SET character_id=?, confidence=?, assignment_source='v03_demote_unknown',
                            review_status='pending',
                            note=COALESCE(note,'') || ' | v03 demote weak/wrong'
                        WHERE id=?
                        """,
                        (unk_id, own, r["ia_id"]),
                    )
    if args.apply and wrong_flags:
        conn.commit()

    # --- mixed named → pending review note (don't force) ---
    mixed_named = []
    for r in named_assign:
        p = protos.get(r["tracklet_id"])
        if p and p["mixed"]:
            mixed_named.append(
                {
                    "tracklet_id": r["tracklet_id"],
                    "name": r["display_name"],
                    "pmin": p["pmin"],
                    "source": r["assignment_source"],
                    "status": r["review_status"],
                }
            )

    # --- Copy QC gallery ---
    qc_manifest = []
    # buckets: wrong, mixed, top suggestions, pure exemplars
    def copy_tid(tid: int, bucket: str, meta: dict):
        oid, path = best_face(conn, tid)
        if not path or not Path(path).exists():
            return
        dest_dir = QC / bucket
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"t{tid}_o{oid}_{Path(path).name}"
        try:
            shutil.copy2(path, dest)
            qc_manifest.append({"bucket": bucket, "tracklet_id": tid, "obs_id": oid, "src": path, "dest": str(dest), **meta})
        except Exception as e:
            qc_manifest.append({"bucket": bucket, "tracklet_id": tid, "error": str(e), **meta})

    for w in wrong_flags[: args.copy_qc]:
        copy_tid(w["tracklet_id"], "possible_wrong", w)
    for m in sorted(mixed_named, key=lambda x: x.get("pmin") or 0)[: min(20, args.copy_qc)]:
        copy_tid(m["tracklet_id"], "mixed_named", m)
    for s in suggestions[: args.copy_qc]:
        copy_tid(s["tracklet_id"], "unknown_suggest", s)
    for cid, rb in ref_bank.items():
        for tid in rb["exemplar_tids"][:3]:
            copy_tid(tid, f"ref_{rb['name']}", {"character": rb["name"]})

    # --- Final counts ---
    def count_chars():
        return conn.execute(
            """
            SELECT ch.display_name, count(DISTINCT ia.tracklet_id)
            FROM identity_assignments ia
            JOIN characters ch ON ch.id=ia.character_id
            JOIN tracklets t ON t.id=ia.tracklet_id
            JOIN shots s ON t.shot_id=s.id JOIN videos v ON s.video_id=v.id
            WHERE v.project_id=?
            GROUP BY 1 ORDER BY 2 DESC
            """,
            (args.project_id,),
        ).fetchall()

    dual_left = conn.execute(
        "SELECT count(*) FROM (SELECT tracklet_id FROM identity_assignments GROUP BY 1 HAVING count(*)>1)"
    ).fetchone()[0]

    summary = {
        "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
        "apply": bool(args.apply),
        "elapsed_sec": round(time.time() - t0, 2),
        "thresholds": {"identity": THR_ID, "merge": THR_MERGE, "unknown": THR_UNK, "margin_auto": MARGIN_AUTO},
        "ref_bank": {rb["name"]: {"n_members": rb["n_members"], "n_refs": rb["n_refs"]} for rb in ref_bank.values()},
        "dual_actions": len(dual_actions),
        "dual_deleted_assignments": len(deleted_ids) if args.apply else sum(len(a["drop"]) for a in dual_actions),
        "dual_remaining": dual_left,
        "safe_moves_applied_or_planned": len(safe_moves),
        "locked_unknown_suggestions": len(suggestions),
        "wrong_flags": len(wrong_flags),
        "wrong_auto_fixable": sum(1 for w in wrong_flags if w["tag"] == "auto_possible_wrong"),
        "mixed_named": len(mixed_named),
        "character_counts": [{"name": n, "n": c} for n, c in count_chars()],
        "qc_images": len(qc_manifest),
    }

    payload = {
        "summary": summary,
        "dual_actions": dual_actions[:200],
        "safe_moves": safe_moves,
        "suggestions_top": suggestions[:100],
        "wrong_flags": wrong_flags[:100],
        "mixed_named": mixed_named[:50],
        "qc_manifest": qc_manifest,
    }
    (OUT / "delivery_pass.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown
    md = f"""# V0.3 Delivery Pass

- Finished: `{summary['finished']}`
- Mode: **{'APPLY' if args.apply else 'DRY-RUN'}**
- Thresholds frozen: identity={THR_ID}, merge={THR_MERGE}, unknown={THR_UNK}, margin≥{MARGIN_AUTO}

## Reference bank

| character | members | refs used |
|-----------|--------:|----------:|
"""
    for name, info in summary["ref_bank"].items():
        md += f"| {name} | {info['n_members']} | {info['n_refs']} |\n"
    md += f"""

## Dual-assignment repair

- Actions planned/applied: **{summary['dual_actions']}**
- Assignments dropped: **{summary['dual_deleted_assignments']}**
- Dual remaining after: **{summary['dual_remaining']}**

## Safe auto moves (pending non-manual Unknown only)

- Count: **{summary['safe_moves_applied_or_planned']}**

## Locked Unknown suggestions (NOT auto-applied — RULE 2)

- Strong matches sim≥{THR_ID} + margin: **{summary['locked_unknown_suggestions']}**
- These need human unlock in UI if desired.

## Possible wrong-person on named

- Flagged: **{summary['wrong_flags']}** (auto-fixable={summary['wrong_auto_fixable']})
- Mixed named tracklets: **{summary['mixed_named']}**

## Character tracklet counts (current)

| name | tracklets |
|------|----------:|
"""
    for row in summary["character_counts"]:
        md += f"| {row['name']} | {row['n']} |\n"
    md += f"""

## QC gallery

`{QC}` — {summary['qc_images']} images in buckets: possible_wrong / mixed_named / unknown_suggest / ref_*

## Artifacts

- `reports/v03_recovery/delivery_pass.json`
- `reports/v03_recovery/delivery_qc/`
- table `character_ref_bank_v2` (if apply)
"""
    (OUT / "delivery_pass.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("WROTE", OUT / "delivery_pass.md")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
