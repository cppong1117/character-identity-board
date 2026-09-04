# CIB V0.3 Recovery — Live Status

Updated: 2026-09-04 (session)

## Completed

| Phase | Result |
|-------|--------|
| 0 Baseline freeze | ✅ `baseline.md` + snapshots |
| 1 Unknown audit (n=300) | ✅ recognition-fail 43% / alignment 26% / true-unknown 22% |
| 2 5-point alignment | ✅ `face_engine_arcface.py` + 500 debug faces |
| 3 Alignment A/B | ✅ **PASS** |

### Phase 3 headline numbers (2500 named faces, 30 shots)

| metric | v1 unaligned | v2 aligned |
|--------|-------------:|-----------:|
| same-person mean | 0.633 | **0.671** |
| same-person std | 0.415 | **0.367** (more stable) |
| diff-person mean | 0.182 | **0.152** |
| separation | 0.451 | **0.519** (+0.068) |
| FAR @0.40 | 6.1% | **3.1%** |
| FAR @0.50 | 2.4% | **1.4%** |

Thresholds **unchanged** (0.40 / 0.50 / 0.30).

## In progress

### Phase 4 — film full re-embed V2

- Script: `scripts/phase4_reembed_v2_parallel.py` (8 CPU workers)
- Target: project_id=15 active faces (~85k)
- Writes: `embedding_v2`, `embedding_version`, `landmarks_json`
- Keeps: `embedding` / `embedding_v1` = arcface_v1
- Path: re-detect landmarks **on face crop** (no original_frame_ref in DB)
- ORT CUDA unavailable in this WSL context (`cudaError 100`); Torch sees A6000 but ArcFace ONNX is CPU
- Progress: query `SELECT COUNT(*) FROM face_observations WHERE embedding_v2 IS NOT NULL`
- Log: `reports/v03_recovery/reembed_v2_run.log`
- ETA: order of **several hours** on CPU (monitor rate)

## Not started (blocked until Phase 4 finishes)

5 Tracklet prototype → 6 contamination → 7 quality split → 8 size gates → 9 FP geometry → 10 reference bank → 11 matching margin → 12 recalibration → 13 reassignment → 14 UI

## Rules still locked

- No threshold lowering to shrink Unknown
- No overwrite of manual assignments
- No full recluster yet


## Phase 4 runner switch (2026-09-04)

Old multi-writer SQLite parallel stalled (~2%).
New: `scripts/phase4_reembed_v2_shards.py` — workers write jsonl shards, single DB apply.
Smoke 200: 198/200 aligned, 0 fail, DB apply OK.
