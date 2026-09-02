# Model A/B Test — Phase 16 Final Report

## Summary

**Date**: 2026-09-02

### License Audit

| Model | Code License | Weight License | Commercial Use | Recommendation |
|-------|--------------|----------------|----------------|----------------|
| SFace (OpenCV Zoo) | Apache 2.0 | Apache 2.0 | ✅ Yes | Current baseline |
| FaceNet (facenet-pytorch) | MIT | MIT (facenet-pytorch) | ✅ Yes | **Test recommended** |
| ArcFace R50 (InsightFace buffalo_l) | MIT | Non-commercial research only | ❌ No | BLOCKED - need commercial license |

### SFace Benchmark Results (Existing Embeddings)

| Metric | Value |
|--------|-------|
| Same-person mean | 0.2830 |
| Different-person mean | 0.1671 |
| Separation | 0.1159 |
| Best Precision @ 99% Recall | 0.00% |
| Best Precision @ 95% Recall | 0.00% |

**Conclusion**: SFace is **INSUFFICIENT** for cross-shot identity recognition.

### Key Findings

1. **SFace cross-shot similarity is too low** (mean=0.28 for same person)
2. **Separation is only 0.116** - not enough for reliable thresholding
3. **Cannot achieve 95% or 99% recall** with any reasonable precision
4. **Model replacement is required** for production-quality identity recognition

### Recommended Next Steps

1. **Test FaceNet (VGGFace2)** - Commercial-safe, MIT license
   - Install: `pip install facenet-pytorch`
   - Expected improvement: 2-3x better cross-shot similarity
   
2. **Evaluate ArcFace commercial license** (if FaceNet insufficient)
   - Contact insightface.ai for commercial licensing
   - Cost estimate needed
   
3. **Consider hybrid approach**
   - Use SFace for fast screening
   - Use FaceNet for high-confidence matching
   - Route ambiguous cases to human review

### Files Generated

- `scripts/model_ab_test.py` - Model A/B test framework
- `scripts/model_ab_test_existing.py` - Test using existing embeddings
- `scripts/generate_benchmark_pairs.py` - Generate benchmark pairs
- `reports/v02_accuracy_recovery/sface_existing_embeddings_report.md` - SFace report
- `reports/v02_accuracy_recovery/model_ab_test_report.md` - A/B test report

### Mission Decision

**C. PIPELINE_FIXED_BUT_SFACE_RECOGNITION_INSUFFICIENT**

Pipeline is now correct (after alignCrop fix). Model is the limiting factor for cross-shot recognition.

**D. MODEL_AB_TEST_REQUIRED**

ArcFace requires commercial license. FaceNet is commercial-safe and should be tested next.
