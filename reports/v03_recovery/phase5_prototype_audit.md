# Phase 5 — Tracklet Prototypes V2 + Contamination Audit

- Finished: `2026-09-05T00:34:27+08:00`
- Project: film (`15`)
- Source: `embedding_v2` (prefer `arcface_v2_aligned`)
- Top-K quality-weighted centroid: **5**
- **No reassignment / no recluster / thresholds frozen**

## Counts

| metric | value |
|--------|------:|
| tracklets total | 2074 |
| prototypes built | 1372 |
| skipped (no v2 emb) | 702 |
| pure | 988 |
| **mixed_flag** | **384** (28.0%) |

Mixed rule: ≥30% of within-tracklet pairs have cosine < 0.35.

## Unknown recovery potential (read-only)

Among non-mixed Unknown tracklets (n=647):

| best sim to named centroid | count |
|----------------------------|------:|
| ≥ 0.50 | 47 |
| ≥ 0.40 (identity thr) | 110 |
| ≥ 0.35 | 232 |
| < 0.30 | 282 |

These are **candidates** for later calibrated matching — **not auto-applied**.

Named centroids used: SMY(id=27), ZY(id=25), lw(id=36), mbq(id=31), DOCTOR(id=30)

### Top recoverable Unknown (preview)

| tracklet | best char | sim | margin |
|---------:|-----------|----:|-------:|
| 1284 | ZY | 0.761 | 0.435 |
| 1978 | DOCTOR | 0.697 | 0.469 |
| 1988 | ZY | 0.686 | 0.440 |
| 1184 | SMY | 0.684 | 0.498 |
| 1200 | ZY | 0.677 | 0.397 |
| 2039 | DOCTOR | 0.673 | 0.441 |
| 901 | DOCTOR | 0.669 | 0.388 |
| 1294 | SMY | 0.665 | 0.387 |
| 900 | DOCTOR | 0.652 | 0.344 |
| 1679 | DOCTOR | 0.636 | 0.371 |
| 1731 | DOCTOR | 0.635 | 0.461 |
| 947 | SMY | 0.622 | 0.398 |
| 1734 | DOCTOR | 0.619 | 0.408 |
| 1315 | SMY | 0.618 | 0.362 |
| 1314 | SMY | 0.607 | 0.373 |
| 1735 | DOCTOR | 0.600 | 0.394 |
| 1159 | SMY | 0.600 | 0.337 |
| 1183 | SMY | 0.600 | 0.440 |
| 755 | SMY | 0.594 | 0.305 |
| 1304 | SMY | 0.592 | 0.326 |


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
