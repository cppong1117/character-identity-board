# BENCHMARK_METHOD_V02.md

## Character Identity Board V0.2 — Benchmark Method

### Benchmark Overview

This document describes the benchmark methodology used to evaluate face identity accuracy after the V0.2 pipeline fixes.

### Dataset

- **Source**: Full-length movie (2h45m, 1080p, 24fps)
- **Project ID**: 15
- **Total shots**: 1,078
- **Total tracklets**: 2,074
- **Total embeddings**: 112,659
- **Unique embeddings**: 104,787 (93% unique)

### Benchmark Categories

Following Mission §24, we report metrics for:

1. **Front Face**: |yaw| < 15°, quality > 0.85
2. **Profile**: 15° < |yaw| < 45°
3. **Blur**: blur_score < 50
4. **Small Face**: face_area < 1000 px²
5. **Occlusion**: occlusion_score > 0.3
6. **Low Light**: exposure < 0.3
7. **Backlight**: exposure > 0.7
8. **Fast Motion**: tracking confidence < 0.7
9. **Same Shot**: tracklets from same shot
10. **Cross Shot**: tracklets from different shots, same character
11. **Unknown Person**: no reference available
12. **Similar-looking Different People**: different characters with similar appearance
13. **Multi-person Crossing**: multiple faces in same frame

### Evaluation Protocol

#### 1. Self-Match Sanity Check

- Compare embedding with itself
- Expected: cosine = 1.0
- Purpose: Verify pipeline correctness

#### 2. Same-Shot Matching

- Compare tracklets from same shot, same character
- Expected: high cosine (> 0.5)
- Purpose: Verify intra-shot consistency

#### 3. Cross-Shot Matching

- Compare tracklets from different shots, same character
- Expected: moderate cosine (0.12-0.16 for SFace)
- Purpose: Verify cross-shot invariance

#### 4. Different-Person Matching

- Compare tracklets from different characters
- Expected: low cosine (< 0.12)
- Purpose: Verify inter-person discrimination

### Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Precision | TP / (TP + FP) | >= 0.99 |
| Recall | TP / (TP + FN) | maximize |
| FAR | FP / (FP + TN) | minimize |
| FRR | FN / (FN + TP) | minimize |
| Cross-Shot Recall | cross-shot TP / cross-shot total | maximize |
| Separation | intra-cluster mean - inter-cluster mean | maximize |

### Threshold Sweep

Tested cosine thresholds: 0.30, 0.32, 0.34, ..., 0.60

For each threshold:
- Compute TP, TN, FP, FN
- Calculate Precision, Recall, FAR, FRR
- Find best threshold for 99% Precision

### Results Summary

| Threshold | Precision | Recall | FAR | FRR | Cross-Shot Recall |
|-----------|-----------|--------|-----|-----|-------------------|
| 0.30 | 0.999 | 0.192 | 0.058 | 0.808 | 0.255 |
| 0.40 | 1.000 | 0.127 | 0.016 | 0.873 | 0.180 |
| 0.50 | 1.000 | 0.076 | 0.000 | 0.924 | 0.109 |
| 0.60 | 1.000 | 0.035 | 0.000 | 0.966 | 0.048 |

### Score Distributions

| Category | Mean | Median |
|----------|------|--------|
| Same-person | 0.159 | 0.120 |
| Different-person | 0.110 | 0.100 |
| Cross-shot | 0.190 | 0.141 |

### Cluster Metrics

| Metric | Value |
|--------|-------|
| Number of clusters | 9 |
| Noise tracklets | 892 |
| Intra-cluster similarity | 0.464 |
| Inter-cluster similarity | 0.100 |
| Separation | 0.364 |

### Limitations

1. **SFace Cross-Shot**: Cosine similarity 0.12-0.16 is too low for Reference Mode
2. **No Ground Truth**: Manual labeling required for full accuracy assessment
3. **Single Movie**: Results may vary across different content types

### Recommendations

1. Use Discovery Mode (clustering) for identity assignment
2. Manual review for cluster merge/split decisions
3. Consider ArcFace/FaceNet for Reference Mode if needed
