# CIB V0.3 Delivery Report — film (project 15)

**Date:** 2026-09-05  
**Git:** see latest master  
**Site:** http://127.0.0.1:8322/  
**DB:** `~/character-identity-board-data/cib.sqlite3`

## RESULT / 结论

| Claim | Status |
|-------|--------|
| Pipeline engineering (align → re-embed → prototype → cleanup) | **PASS** |
| Threshold discipline (no lower for Unknown) | **PASS** (0.40 / 0.50 / 0.30 frozen) |
| Manual lock discipline | **PASS** (manual/confirmed Unknown not auto-overwritten) |
| Dual-assignment data integrity | **PASS** (609 → **0**) |
| Named-character **precision** after cleanup | **PASS** (self-sim ≥0.40 on 82–100% of remaining named; centroid pairs ≤0.17) |
| Unknown fully eliminated | **NOT claimed** — majority are **manual-confirmed locked**; 63 HQ suggestions left for human unlock |
| Identity quality absolute 100% | **NOT claimed** — residual mixed tracklets + locked Unknown remain |

**Delivery posture:** high-precision named board + honest Unknown, not inflated recall.

---

## What was fixed (this session)

### Phase 4 — Full re-embed V2 (GPU)
- 85,307 / 85,307 film active faces → `embedding_v2`
- 93.8% `arcface_v2_aligned`, 6.2% fallback, **0 fail**
- RTX A6000 single-process ORT CUDA ~35 faces/s
- Script: `scripts/phase4_reembed_v2_gpu.py`

### Phase 5 — Tracklet prototypes + mixed audit
- 1,372 prototypes / 2,074 tracklets (702 no active faces)
- Mixed flagged: 384
- Script: `scripts/phase5_tracklet_prototypes_v2.py`

### Delivery pass — dual repair + ref bank + precision
- **609 dual assignments repaired → 0 remaining**
  - Typical bug: `Unknown:manual:confirmed` **plus** `Named:clustering_v2:auto`
  - Policy: keep named only if v2 prototype agrees; else keep Unknown (safer)
- HQ reference bank (file≥12KB, aligned, evidence, blur, non-mixed): 40 faces/character
- Centroid pairwise cosine (lower = better separation):

|  | ZY | SMY | DOCTOR | mbq | lw |
|--|---:|----:|-------:|----:|---:|
| ZY | 1 | 0.08 | 0.09 | 0.06 | -0.06 |
| SMY | | 1 | 0.08 | 0.04 | 0.00 |
| DOCTOR | | | 1 | 0.17 | 0.05 |
| mbq | | | | 1 | 0.13 |
| lw | | | | | 1 |

- Weak/mixed **non-locked** named demoted → Unknown: **93**
- Low-conf named marked pending: **8**

---

## Final character board (film)

| Character | Tracklets | Notes |
|-----------|----------:|-------|
| Unknown | **1273** | Mostly manual-confirmed locked; not bulk-auto-labeled |
| SMY | **122** | self-sim mean 0.55, ≥0.40 ≈ 97% |
| DOCTOR | **48** | mean 0.58, ≥0.40 = 100% |
| ZY | **23** | mean 0.59, ≥0.40 = 100% |
| lw | **17** | mean 0.51, ≥0.40 ≈ 82% (weakest named set) |
| mbq | **15** | mean 0.54, ≥0.40 ≈ 92% |

**Unknown change:** baseline 1431 dual-inflated view → cleaned unique **1273**.  
Net named board is **smaller but cleaner** (Precision > Recall).

---

## Visual QC (vision model)

### HQ reference exemplars
| Ref | Vision result |
|-----|----------------|
| SMY | Real face; **male** presentation (consistent across confirmed SMY samples — label is identity code, not “female lead”) |
| DOCTOR | Real face; older adult male |
| ZY | Real face; male 30–40, short black hair |
| lw | Real face; young adult female, long dark hair |
| mbq | Real face; adult female, long black hair |

### Sample Unknown suggestions (locked — not applied)
| Tracklet | Suggest | sim | Vision |
|----------|---------|----:|--------|
| 1601 | SMY | 0.91 | Face present; gender presentation male-compatible with SMY refs |
| 1284 | ZY | 0.61 | Male short black hair — plausible |
| 2038 | DOCTOR | 0.60 | Older adult male — plausible |

**First HQ bank attempt** had some bad crops (body/blur) → **rejected and rebuilt** with file-size + blur + evidence gates.

---

## Locked Unknown suggestions (human action)

- **63** non-mixed Unknown with sim≥0.40 + margin≥0.08 to HQ bank  
- **19** of those ≥0.50  
- Almost all are `manual` + `confirmed` → **RULE 2: not auto-applied**  
- List: `reports/v03_recovery/hq_refbank_summary.json` → `top_suggestions`  
- UI path: Characters / Review — unlock only if you agree

---

## Rules compliance

| Rule | Evidence |
|------|----------|
| 1 No threshold lower | config still 0.40 / 0.50 / 0.30 |
| 2 No overwrite manual | locked Unknown suggestions only; manual confirmed named kept |
| 3 No blind full recluster | no HDBSCAN recluster this pass; margin/ref cleanup only |

---

## Artifacts

| Path | Purpose |
|------|---------|
| `reports/v03_recovery/phase4_reembed_complete.md` | Re-embed proof |
| `reports/v03_recovery/phase5_prototype_audit.md` | Prototypes / mixed |
| `reports/v03_recovery/delivery_pass.md` | Dual repair pass |
| `reports/v03_recovery/delivery_pass.json` | Machine detail |
| `reports/v03_recovery/hq_refbank_summary.json` | HQ bank + suggestions |
| `reports/v03_recovery/precision_cleanup.json` | Demote stats |
| `reports/v03_recovery/delivery_qc_hq/` | Visual QC crops |
| `scripts/phase4_reembed_v2_gpu.py` | GPU re-embed |
| `scripts/phase5_tracklet_prototypes_v2.py` | Prototypes |
| `scripts/phase6to9_delivery_pass.py` | Dual+ref+match |

---

## How to use now

1. Open **http://127.0.0.1:8322/** → project **film**
2. **Characters** tab: named sets should look tighter; use exclude/select for remaining dirty faces
3. Optional: review `delivery_qc_hq/unknown_suggest/` and unlock high-sim locked Unknown if correct
4. Do **not** lower `arcface_identity_threshold` to chase Unknown count

---

## Honest residual risks

1. **~1273 Unknown** remain by design (manual lock + true extras + no-face tracklets)
2. **384 mixed tracklets** still need split/exclude (contamination flag only)
3. **lw** is the weakest named cluster (p10≈0.35) — watch in UI
4. Duplicate empty characters (`SMY` id 33 etc.) still exist as shells — cosmetic
5. CIB must stay on v2 embeddings for any future match (do not mix v1)

---

## Operator one-liners

```bash
# health
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8322/

# restart
cd ~/character-identity-board && .venv/bin/python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8322

# counts
python3 - <<'PY'
import sqlite3
c=sqlite3.connect(__import__('pathlib').Path.home()/'character-identity-board-data/cib.sqlite3')
print(c.execute('''select ch.display_name, count(*) from identity_assignments ia
 join characters ch on ch.id=ia.character_id
 join tracklets t on t.id=ia.tracklet_id join shots s on t.shot_id=s.id join videos v on s.video_id=v.id
 where v.project_id=15 group by 1 order by 2 desc''').fetchall())
PY
```
