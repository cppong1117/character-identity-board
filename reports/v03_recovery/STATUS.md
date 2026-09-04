# CIB V0.3 STATUS — 2026-09-04 23:55:58 CST

## Phase 4 — GPU re-embed (ACTIVE)

- Runner: `scripts/phase4_reembed_v2_gpu.py` (single-process ORT CUDA)
- Device: NVIDIA RTX A6000 via WSL
- Providers: CUDAExecutionProvider + CPUExecutionProvider
- Smoke: 200 faces @ ~25 faces/s, fail=0
- Already in DB embedding_v2: ~10,933
- Remain film active: ~74,374
- ETA: ~50 min → **2026-09-05 00:45 CST**
- Thresholds FROZEN: identity 0.40 / merge 0.50 / unknown 0.30

## Why not multi-process GPU
ProcessPool workers hit `cudaError 100: no CUDA-capable device` on this host.
Main-process + prep_cuda(LD_LIBRARY_PATH + torch touch) works.

## Phase 3 PASS (locked)
separation 0.451→0.519, FAR@0.40 6.1%→3.1%
