# 追踪方案决策（TRACKING_DECISION）

## 结论
采用**自定义 IoU + Hungarian 匹配**追踪器（`backend/app/tracking.py`），不引入
ByteTrack / BoT-SORT / DeepSORT / OC-SORT。

## 评估摘要

| 候选 | License | 引入依赖 | 确定性 | 镜头重置 | 结论 |
|------|---------|----------|--------|----------|------|
| ByteTrack | MIT | motpy/cython 等 | 中 | 需自己实现 | 评估后未采用 |
| BoT-SORT | MIT | ReID model/权重未知 | 低 | — | 未采用 |
| DeepSORT | GPL variant 风险 | 重 | 低 | — | 未采用 |
| OC-SORT | MIT | 重 | 中 | — | 未采用 |
| **自研 IoU+Hungarian** | 本项目 | 仅 numpy/scipy | 高 | 原生 | ✅ 采用 |

## 理由
1. **确定性 & 无额外依赖**：避免引入带未知权重/许可证的 ReID 模型，规避 License Gate。
2. **Shot 全局重置**：每进入新 Shot，tracker 状态清空，杜绝跨镜头沿用 Track ID。
3. **V0.1 人物级追踪足够**：目标是镜头内的 Tracklet（不要求跨镜追踪，聚类处理由 embedding 完成）。
4. 交叉身份：以 IoU 代价矩阵 + Hungarian 全局最优匹配，降低同帧两人交叉换号概率；帧级漏检用短暂海龟保留。

## 实现
- 人脸检测框（YuNet）作为前景
- 逐帧 Hungarian 匹配到既有 track；IoU < gate 视为新 track
- 漏检延迟存活；Shot 结束 flush 所有 tracklet

## 验证
- `tests/` 中 pipeline integration 确认每 shot 生成恰一个 tracklet（单人画面）
- 已避免：同一人物逐帧重复记录（以 tracklet 聚合）、镜头切换沿用 ID（重置）
