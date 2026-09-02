# ACCURACY_RECOVERY_V02.md

## Character Identity Board V0.2 — Accuracy Recovery

### Mission Summary

Systematically fixed the face detection → alignment → embedding → clustering pipeline to recover identity accuracy.

### Timeline

| Date | Phase | Result |
|------|-------|--------|
| 2026-09-02 | Phase 0: Freeze Baseline | Commit a084fd2 |
| 2026-09-02 | Phase 1: Face Debug Sheet | 200 faces audited |
| 2026-09-02 | Phase 2: Coordinate/Crop Audit | **P0 BUG FOUND** |
| 2026-09-02 | Phase 3: Fix alignCrop | Embeddings now unique |
| 2026-09-02 | Phase 4: YuNet FP Audit | Non-face FPs identified |
| 2026-09-02 | Phase 5-7: Quality Gate V2 | 81.4% pass rate |
| 2026-09-02 | Phase 8-9: Clustering V2 | 9 clusters, separation=0.364 |
| 2026-09-02 | Phase 10-15: Full Benchmark | SFace cross-shot=0.16 |

### Critical Bug Found

**P0: alignCrop Missing Landmarks**

- **Root cause**: `get_embedding()` received only `[x,y,w,h]` (4 elements) instead of full detection (15 elements: bbox + 5 landmarks + confidence)
- **Impact**: `sface.alignCrop()` returned ALL-BLACK 112×112 images, causing 98.7% embedding duplication
- **Fix**: Pass full detection dict with landmarks to `get_embedding()`
- **Verification**: Embeddings now 93% unique (down from 0.1%)

### Results

| Metric | Before | After |
|--------|--------|-------|
| Total embeddings | 108,755 | 112,659 |
| Unique embeddings | 1,426 | 104,787 |
| Duplication rate | 98.7% | 7.0% |
| Quality gate pass | N/A | 81.4% |
| Clusters | N/A | 9 |

### SFace Limitation

SFace cross-shot similarity is fundamentally low (0.12-0.16 for same person), making Reference Mode unusable. Discovery Mode (clustering) works with separation=0.364.

### Conclusion

**PIPELINE_FIXED_BUT_SFACE_RECOGNITION_INSUFFICIENT**

Pipeline bug fixed, but SFace model lacks cross-shot recognition capability. Use Discovery Mode for now; consider ArcFace/FaceNet for Reference Mode.
