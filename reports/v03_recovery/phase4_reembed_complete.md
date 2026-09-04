# Phase 4 — Full Re-embed V2 COMPLETE

- Finished: `2026-09-05T00:31:54+08:00`
- Runner: `scripts/phase4_reembed_v2_gpu.py` (single-process ORT CUDA on RTX A6000)
- Project: film (`project_id=15`)

## Results (DB verified)

| metric | value |
|--------|------:|
| film active faces | **85,307** |
| with `embedding_v2` | **85,307** (100%) |
| remain without v2 | **0** |
| `arcface_v2_aligned` | **80,020** (93.8%) |
| fallback unaligned | **5,287** (6.2%) |
| fail | **0** |
| rate | ~34.5 faces/s |
| elapsed (GPU pass) | ~2153 s (~36 min) |

## Schema

- `embedding` / `embedding_v1` = legacy v1 kept
- `embedding_v2` = new aligned (or fallback)
- `embedding_version` = `arcface_v2_aligned` | `arcface_v2_aligned_fallback_unaligned`
- `landmarks_json` populated when available

## Thresholds (FROZEN)

- identity **0.40** / merge **0.50** / unknown **0.30**
- **No reassignment** in this phase

## Artifacts

- `reports/v03_recovery/reembed_v2_gpu_summary.json`
- `reports/v03_recovery/reembed_v2_run.log`
- `scripts/phase4_reembed_v2_gpu.py`
