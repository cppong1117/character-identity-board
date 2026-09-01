# V0.1 Benchmark 报告（BENCHMARK_REPORT）

日期：2026-08-17
结论：**Shot 边界达到 SUCCESS（F1=1.0×3）；人物身份聚类 PARTIAL（合成小脸素材限制）**

素材为本地生成的三个真实 H.264 视频（非静态图）。Ground Truth 显式记录于
`~/character-identity-board-data/benchmarks/V0.1/GROUND_TRUTH.json`。

## 汇总（真实运行结果）

| 指标 | Test A 双人 | Test B 复杂 | Test C 多人 | V0.1 目标 |
|------|-------------|-------------|-------------|-----------|
| 时长 | 24s | 24s | 36s | — |
| 预期镜头 | 4 | 4 | 6 | — |
| 检测镜头 | 4 | 4 | 6 | — |
| Shot Precision | 1.0 | 1.0 | 1.0 | ≥0.90 |
| Shot Recall | 1.0 | 1.0 | 1.0 | ≥0.90 |
| **Shot F1** | **1.0** | **1.0** | **1.0** | **≥0.90 ✅** |
| Tracklets | 4 | 2 | 5 | — |
| Observations | 558 | 300 | 708 | — |
| 识别人物数 | 1 (Unknown) | 1 (Unknown) | 1 (Unknown) | A/B/C |
| 待审 tracklet | 2 | 2 | 3 | — |
| 处理耗时 | 11.6s | 9.9s | 15.2s | — |
| 平均速度 | 2.07 fps | 2.44 fps | 2.36 fps | — |

## Detail 分项

### Shot Detection（SUCCESS）
- ContentDetector 对所有三个视频正确切分，无漏检、无误检（F1=1.0）
- 每个 shot 都有代表帧 + 时间码

### Identity Clustering（PARTIAL）
- 全部 tracklet 在本 fixture 中被归到 Unknown，未分出 A/B/C
- **确诊原因**（非引擎缺陷）：生成的测试人物脸太小（messi 面积仅 ~1229px²，~30-65px），
  低于 `min_face_size_px=90` 嵌入门槛被跳过；而合成缩放让 SFace 对同一人跨尺度自一致
  仅 0.13-0.28，无法稳定判同。
- 这属于“低质量观测进入 Unknown”的正确保守行为（Mission §10 明确要求），
  但也意味着在这套合成素材上身份 accuracy 未达 0.90 → **如实 PARTIAL**。
- **人工确认后 Identity Accuracy 可到 100%**（人工重指派已验证 tracklet→Character 精确归属）

### Track 质量（无碎片化）
- 每个单人镜头稳定产出恰好 1 个 tracklet（无逐帧重复、无跨镜沿用 ID）

## 说明
- 不使用 Mock；所有数字来自真实数据库行数与真实处理计时。
- 若要达到身份 accuracy gate，需以真实/更高质量（≥100px 清晰正脸）人物素材重跑；
  已提供 `scripts/run_benchmarks.py` 可复跑。
