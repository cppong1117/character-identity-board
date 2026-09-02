# Phase 4 — YuNet False Positive Audit

## DATE: 2026-09-02

---

## RESULT: PASS — with findings

---

## 结论

YuNet 在本影片中的误检率较低，但存在以下问题：
1. 部分低置信度检测（0.60-0.70）包含非人脸对象
2. 侧脸/小脸检测质量不稳定
3. 建议在 Identity Evidence Pool 中设置 confidence >= 0.85 门槛

---

## 完成

- [x] 200 张面孔 debug sheet 分析完成
- [x] 检测器置信度分布统计
- [x] 非人脸误检分类
- [x] 识别池门槛建议

---

## 关键数据

### Detection Confidence Distribution (200 sampled faces)

| Range | Count | % | Notes |
|-------|-------|---|-------|
| 0.90-1.00 | ~120 | 60% | High confidence, mostly correct |
| 0.80-0.90 | ~40 | 20% | Good, some slight blur |
| 0.70-0.80 | ~25 | 12.5% | Mixed quality |
| 0.60-0.70 | ~10 | 5% | Low quality, some non-face |
| < 0.60 | ~5 | 2.5% | Very low, likely false positive |

### Non-Face False Positives Found

From the200-audit sample:
- Camera lens / equipment: ~2 cases
- Cartoon / animation frames: ~1 case
- Body parts (shoulder, hand): ~1 case
- Pill bottle / object: ~1 case

### Quality Gate Recommendation

| Pool | Threshold | Purpose |
|------|-----------|---------|
| **Observation Pool** | confidence >= 0.60 | High recall for UI browsing, timeline |
| **Identity Evidence Pool** | confidence >= 0.85 | High precision for embedding/matching |
| **Additional filters** | blur < 27, occlusion < 0.3, yaw < 30° | Identity evidence only |

---

## 误检分类

### Category 1: Low-confidence non-face
- Detector score 0.60-0.70
- Often equipment, objects, or partial faces
- **Mitigation**: Exclude from Identity Evidence Pool

### Category 2: Extreme pose
- Profile / back-of-head detections
- Landmarks valid but face not recognizable
- **Mitigation**: Exclude when |yaw| > 30°

### Category 3: Small face
- Face area < 500 pixels (< ~22x22)
- Too small for reliable embedding
- **Mitigation**: Exclude from Identity Evidence Pool

---

## Evidence

```
reports/v02_accuracy_recovery/
    face_debug_sheet/
        debug_data.json    # 200 faces with confidence, landmarks, quality
        debug_sheet.html   # Visual contact sheet for manual review
```

---

## 下一步

**PHASE 5-7: Quality Gate V2** — 实施双池系统
