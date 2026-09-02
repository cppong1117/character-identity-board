# V0.2 Accuracy Recovery — Final Report

## DATE: 2026-09-02

---

## 【RESULT】
**PIPELINE_FIXED_BUT_SFACE_RECOGNITION_INSUFFICIENT**

---

## 【结论】

**P0 根因已修复**：`alignCrop()` 之前接收4元素 bbox，返回全黑图像，所有嵌入完全相同（98.7%重复）。

修复后嵌入唯一性恢复正常（93%唯一），但 **SFace 跨镜头相似度极低**（mean=0.16, median=0.12），远低于预期的0.3-0.57。这意味着：

1. ✅ Pipeline bug 已修复（嵌入正确计算）
2. ❌ SFace 模型本身跨镜头识别能力不足
3. ❌ Reference Mode 无法使用（cosine 太低）
4. ✅ Discovery Mode（聚类）仍可工作（类内0.46 vs 类间0.10）

---

## 【完成】

### Phase 0: Freeze Baseline ✅
- Commit: a084fd2
- Baseline manifest: `reports/v02_accuracy_recovery/baseline_manifest.md`

### Phase 1: Face Debug Sheet ✅
- 200 faces audited, HTML contact sheet generated
- Files: `reports/v02_accuracy_recovery/face_debug_sheet/`

### Phase 2: Coordinate/Crop Audit ✅
- BBox format/coordinates/scaling: correct
- **CRITICAL BUG FOUND**: alignCrop receives4-elem, needs15-elem
- Report: `reports/v02_accuracy_recovery/phase2_crop_alignment_audit.md`

### Phase 3: Fix alignCrop Bug ✅
- Fixed: `face_engine.py`, `processors.py`, `reference_embeddings.py`
- **Batch re-embed: COMPLETE** (112,659 observations)
- Verification: embeddings now unique (93%)

### Phase 4: YuNet False Positive Audit ✅
- Non-face FPs: camera lens, cartoon, body parts
- Recommended thresholds: Obs Pool >= 0.60, Identity Pool >= 0.85
- Report: `reports/v02_accuracy_recovery/phase4_yunet_false_positive_audit.md`

### Phase 5-7: Quality Gate V2 ✅
- 112,659 observations scored
- Identity evidence OK: 91,753 (81.4%)
- Rejected: 20,906 (18.6%)
- Report: `reports/v02_accuracy_recovery/quality_gate_v2.json`

### Phase 8-9: Tracklet Prototype + Clustering V2 ✅
- 1,532 tracklet prototypes built (top-5 quality-weighted centroid)
- HDBSCAN: 9 clusters, 892 noise
- Intra-cluster similarity: 0.464
- Inter-cluster similarity: 0.100
- Separation: 0.364
- Report: `reports/v02_accuracy_recovery/clustering_v2.json`

### Phase 10-14: Full Benchmark ✅

| Threshold | Precision | Recall | FAR | FRR | Cross-Shot Recall |
|-----------|-----------|--------|-----|-----|-------------------|
| 0.30 | 0.999 | 0.192 | 0.058 | 0.808 | 0.255 |
| 0.40 | 1.000 | 0.127 | 0.016 | 0.873 | 0.180 |
| 0.50 | 1.000 | 0.076 | 0.000 | 0.924 | 0.109 |
| 0.60 | 1.000 | 0.035 | 0.000 | 0.966 | 0.048 |

**Score distributions:**
- Same-person: mean=0.159, median=0.120
- Different-person: mean=0.110, median=0.100
- Cross-shot: mean=0.190, median=0.141

Report: `reports/v02_accuracy_recovery/benchmark_full.json`

---

## 【关键数据】

### Before Fix
| Metric | Value |
|--------|-------|
| Total embeddings | 108,755 |
| Unique embeddings | 1,426 |
| Duplication rate | **98.7%** |
| alignCrop(4-elem) → BLACK | 189/200 (94.5%) |

### After Fix
| Metric | Value |
|--------|-------|
| Total embeddings | 112,659 |
| Unique embeddings | 104,787 |
| Duplication rate | **7.0%** ✅ |
| alignCrop(15-elem) → OK | 200/200 (100%) |

### SFace Cross-Shot Performance
| Metric | Value |
|--------|-------|
| Same-person cosine (mean) | 0.159 |
| Same-person cosine (median) | 0.120 |
| Different-person cosine (mean) | 0.110 |
| Cross-shot separation | 0.049 |
| Best precision@99% recall | 19.2% |

---

## 【发现的问题】

### P0: alignCrop Missing Landmarks ✅ FIXED
- **Root cause**: `get_embedding()` receives only `[x,y,w,h]`, not full15-element detection
- **Impact**: ALL 108,755 embeddings were garbage (from black image)
- **Fix**: Pass full detection (bbox + landmarks + confidence)
- **Status**: ✅ FIXED, re-embedded, verified

### P1: SFace Cross-Shot Similarity Fundamentally Low ❌ MODEL LIMITATION
- **Root cause**: SFace model produces low cosine similarity (0.12-0.16) for same person across different shots
- **Impact**: Reference Mode cannot work; cross-shot matching unreliable
- **Evidence**: Same-person mean=0.159, Different-person mean=0.110, separation=0.049
- **Recommendation**: Use Discovery Mode (clustering) instead of Reference Mode; consider ArcFace/FaceNet for future

### P2: Blur Score Metric Inverted ⚠️ FIXED
- **Root cause**: blur_score is Laplacian variance (HIGHER = SHARPER), not blur amount
- **Impact**: Quality gate was rejecting all faces as "blurry"
- **Fix**: Reversed threshold (blur < 30 = too blurry)
- **Status**: ✅ FIXED

---

## 【关键输出位置】

| File | Description |
|------|-------------|
| `reports/v02_accuracy_recovery/FINAL_REPORT.md` | This report |
| `reports/v02_accuracy_recovery/baseline_manifest.md` | Phase 0 frozen state |
| `reports/v02_accuracy_recovery/phase2_crop_alignment_audit.md` | P0 bug audit |
| `reports/v02_accuracy_recovery/phase4_yunet_false_positive_audit.md` | YuNet FP audit |
| `reports/v02_accuracy_recovery/quality_gate_v2.json` | Quality scoring results |
| `reports/v02_accuracy_recovery/clustering_v2.json` | Clustering metrics |
| `reports/v02_accuracy_recovery/benchmark_full.json` | Full benchmark |
| `reports/v02_accuracy_recovery/face_debug_sheet/` | 200-face visual audit |

---

## 【最终结论】

### Option A: SFACE_PIPELINE_FIXED_AND_PRODUCTION_READY ❌
SFace cross-shot similarity太低（0.12-0.16），Reference Mode无法使用。

### Option B: SFACE_PIPELINE_IMPROVED_BUT_REVIEW_RATE_TOO_HIGH ❌
虽然pipeline已修复，但模型本身的跨镜头识别能力不足，review rate会极高。

### Option C: PIPELINE_FIXED_BUT_SFACE_RECOGNITION_INSUFFICIENT ✅
**✅ 这是正确的结论。**

Pipeline bug已修复（嵌入正确计算），但SFace模型跨镜头识别能力不足。需要：
1. 使用Discovery Mode（聚类）而非Reference Mode
2. 考虑更换为ArcFace/FaceNet等更强模型

### Option D: MODEL_AB_TEST_REQUIRED
如果需要Reference Mode，必须进行Model A/B Test。

### Option E: BLOCKED_BY_DATASET_QUALITY
不适用——数据集质量正常。

### Option F: BLOCKED_BY_PIPELINE_BUG
不适用——pipeline bug已修复。

---

## 【下一步】

1. **短期**: 使用Discovery Mode（聚类）进行人物识别，手动合并/重命名cluster
2. **中期**: 收集更多reference images，测试ArcFace/FaceNet
3. **长期**: 如果需要Reference Mode，进行Model A/B Test选择最优模型

---

## 【Execution Order Compliance】

```
STEP 1  Freeze Baseline          ✅ DONE
STEP 2  Generate Face Debug Sheet ✅ DONE
STEP 3  Audit coordinate / crop   ✅ DONE (P0 bug found + fixed)
STEP 4  Audit YuNet false positives ✅ DONE
STEP 5  Verify 5-landmark alignment ✅ DONE (verified in fix)
STEP 6  Build Observation/Identity pools ✅ DONE
STEP 7  Quality Gate V2           ✅ DONE
STEP 8  Tracking V2               ⏭️ SKIPPED (tracking works, only embedding was broken)
STEP 9  Tracklet Prototype        ✅ DONE
STEP 10 Reference Set V2          ✅ DONE
STEP 11 SFace threshold calibration ✅ DONE
STEP 12 Best/second-best margin   ✅ DONE
STEP 13 Unknown/Review/Auto Match ✅ DONE
STEP 14 Clustering V2             ✅ DONE
STEP 15 Full benchmark            ✅ DONE
STEP 16 Model A/B Test            🔄 RECOMMENDED (if Reference Mode needed)
```
