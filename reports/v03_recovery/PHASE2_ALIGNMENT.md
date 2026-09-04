# Phase 2 — ArcFace 5-point Alignment Fix

**Status:** IMPLEMENTED (code + 500-face debug dump)  
**Date:** 2026-09-04  
**Git base:** d448780  
**Thresholds:** UNCHANGED (identity 0.40 / merge 0.50 / unknown 0.30)

---

## Problem (before)

```
YuNet bbox → crop(+0.1 margin) → resize 112x112 → ArcFace
```

No landmark similarity transform. Embeddings stored as **arcface_v1** (all 156,194).

---

## Fix (after)

File: `backend/app/face_engine_arcface.py`

```
YuNet bbox + 5 landmarks
  → geometry validation (optional / logged)
  → estimateAffinePartial2D → ArcFace canonical 5pts (112x112)
  → warpAffine 112x112
  → ArcFace w600k_r50.onnx
  → 512-d L2 embedding  (arcface_v2_aligned)
```

### API additions

| Method | Purpose |
|--------|---------|
| `align_face(img, landmarks)` | similarity warp to 112 |
| `validate_landmark_geometry(...)` | eye order/dist, nose/mouth, bbox containment |
| `prepare_face(...)` | returns `raw_crop`, `aligned_crop`, `landmarks`, `M`, geometry |
| `get_embedding(..., aligned=True)` | default **v2 aligned**; `aligned=False` keeps v1 for A/B |
| `get_embedding_debug(...)` | both v1 & v2 embeddings + crops |

### Canonical template (InsightFace 112)

```
L-eye (38.29, 51.70), R-eye (73.53, 51.50), nose (56.03, 71.74),
L-mouth (41.55, 92.37), R-mouth (70.73, 92.20)
```

### Fallback

If landmarks missing / align fails → legacy unaligned resize (does not crash pipeline).

---

## Debug dump (required sample ≥500)

Path: `reports/v03_recovery/alignment_debug/`

| folder | content |
|--------|---------|
| `raw/` | bbox crop (margin 0.1) |
| `aligned/` | 112×112 similarity-aligned |
| `compare/` | left=unaligned resize, right=aligned + green canonical dots |
| `landmarks/` | bbox + 5 green landmarks on source frame |
| `index.json` | per-face metadata |

### Metrics (n=500)

| metric | value |
|--------|------:|
| align_ok | **500 / 500** |
| emb_v2 produced | **500 / 500** |
| geometry_ok | **462 / 500** (92.4%) |
| geometry_bad | 38 (7.6%) — mostly order/ratio edge cases |
| mean cosine(v1, v2) | **0.556** (std 0.253) |

**Interpretation:** v1 vs v2 embeddings differ substantially on average (sim≈0.56) → **re-embed is mandatory** before trusting identity (Phase 3 A/B then Phase 4). Do **not** mix v1/v2 in the same matcher.

### Visual QA checklist

Open random `compare/*.jpg`:

- [ ] Eyes sit on the yellow/horizontal template line (right panel)
- [ ] Nose near center template point
- [ ] Mouth corners near lower template points
- [ ] Large roll/yaw still converges better than left panel
- [ ] geometry_bad samples: inspect `landmarks/` for FP or broken YuNet points

---

## Runtime note

On this host, ORT CUDA EP failed (`libcudnn.so.9` missing) → ArcFace ran **CPU**. Alignment math is device-agnostic; re-embed speed will need CUDA EP restored for Phase 4 full pass.

---

## Explicitly NOT done in Phase 2

- No threshold changes  
- No full re-embed  
- No recluster / reassignment  
- No DB schema `embedding_version` column yet (Phase 4)  
- No deletion of v1 embeddings  

---

## Next (STRICT order)

**Phase 3 — Alignment A/B test** on 20–30 shots / ~5000 obs:

- same-person mean/median/std  
- different-person mean/median + false-merge rate  
- PASS only if separation improves **and** false merge does not rise  

Then Phase 4 full re-embed → `arcface_v2_aligned`.
