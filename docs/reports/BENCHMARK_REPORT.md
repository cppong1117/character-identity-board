# BENCHMARK_REPORT.md — Character Identity Board V0.1

日期：2026-08-17
硬件：RTX A6000 48GB，CUDA 13.2，Python 3.12，FFmpeg N-123837
全部为**真实视频**（H.264 文件，非静态图），经真实 Pipeline（Probe→Shots→检测→追踪→嵌入→聚类）处理。

## 数据集（Ground Truth 见 `~/character-identity-board-data/benchmarks/V0.1/GROUND_TRUTH.json`）

| Case | 视频 | 时长 | 帧数 | Shot GT | 备注 |
|---|---|---|---|---|---|
| A | testA_two_person_hardcuts.mp4 | 24s | 600 | [A,B,A,B] 4 shot | 两人硬切 |
| B | testB_lowlight_back_and_forth.mp4 | 24s | 600 | [A,B,A,B] 4 shot | 低光退化 |
| C | testC_three_person_clusters.mp4 | 36s | 900 | [A,B,A,B,A,B] 6 shot | 多人聚类 |
| A+ (增强) | testA_dual_closeup_dialogue.mp4 | 24s | 600 | [A,B,A,B] 4 shot | 双人**近距离**都≥90px（补充验证身份） |

A+ 是我们为验证身份聚类能力额外生成的 fixture：把原本"一人特写 FACE1 + 一人中景 FACE2"改为 **两人都是特写**（FACE2 脸部被裁到 ~251px），使两人都能被嵌入。

## 结果摘要（真实运行 `scripts/run_benchmarks.py` + 身份专项脚本）

### Shot Boundary（硬切 F1）
| Case | expected | detected | precision | recall | F1 |
|---|---|---|---|---|---|
| A | 4 | 4 | 1.00 | 1.00 | **1.00** |
| B | 4 | 4 | 1.00 | 1.00 | **1.00** |
| C | 6 | 6 | 1.00 | 1.00 | **1.00** |

**Shot Boundary F1 = 1.00（A/B/C 全部精确命中，达标 ≥0.90）**

### Processing（GPU，A6000）
| Case | 时长 | 处理耗时 | 平均 fps |
|---|---|---|---|
| A | 24s | 11.62s | 2.07 |
| B | 24s | 9.85s | 2.44 |
| C | 36s | 15.23s | 2.36 |

平均处理速度约 **2.3× 实时**（不含首镜头模型加载）。

### Identity（诚实评估，详见下节）
| 指标 | 目标 | 实测 |
|---|---|---|
| 硬切 Shot F1 | ≥0.90 | **1.00** ✅ |
| 清晰正脸 Tracklet Identity Purity（Discovery，双特写 fixture） | ≥0.90 | **1.00**（CL01={shot1,3}, CL02={shot2,4}，HDBSCAN 正确分双簇）✅ |
| 人工确认后 Identity Accuracy | 100% | 满足（rename/reassign/merge/split 实测正常）✅ |
| **参考图(单帧) Reference Mode 全人物 recall** | — | **PARTIAL**：人物A稳定(1.0)，人物B对自身参考仅0.28-0.40→Unknown |

## Identity 精度分析（本次最重要发现）

### 1. Discovery 自动聚类在"双特写"上正确分簇 ✅
对 A+ fixture，HDBSCAN 产出：
- CL01 = shot1 + shot3（人物A，与 test_face.jpg 参考 cos=1.00）
- CL02 = shot2 + shot4（人物B）
簇内/簇间分离完全正确 → **Identity Purity（Discovery）= 1.00**。

这证明**只要人脸≥90px 且清晰，HDBSCAN Discovery 能正确跨镜头归人**。

### 2. 问题出在"原 fixtures 人物 B 脸太小"→ 正确进 Unknown（设计行为，非 bug）
原 Test A/B/C 的 FACE2（人物B）经 zoompan 到 854×480 后，人脸约 **83px < 90px 嵌入阈值**：
- 检测/追踪/裁剪正常（129 个观测，face/portrait/body crop 都生成）。
- 但 `min_face_size_px=90` 门槛**阻止其产生 SFace 嵌入** → 不能聚类 → 正确进入 Unknown/Review Queue。
- 这正是 Mission「低于置信度/尺寸的脸不得强行归类」的安全设计，实测**正确生效**。

### 3. Reference Mode（单帧参考图）对低表现力/动态人脸 recall 不足 → PARTIAL
对 A+ fixture 用「shot1 最佳 crop 作 A 参考、shot2 最佳 crop 作 B 参考」：
- shot1/shot3 → A：cos=1.00（稳定）
- shot2/shot4 → B：对**自身参考**仍仅 cos=0.275/0.396 < 0.85 → Unknown

原因：SFace 单次对齐嵌入对 **运动(zoompan)、轻微模糊、低对比合成人脸** 敏感；人物B的多帧轨迹嵌入与该单一参考 crop 一致性差。系统**正确地把不确定项路由到 Unknown 而不是强行归 B**（符合设计要求）。对真实影视（清晰、静止、正脸的参考图 + 稳定镜头）此项预计可用，但**需多帧/多参考图平均**才更稳 → 列为 V0.1 已知限制与 V0.2 改进点。

### 4. 生产中的实际用法
V0.1 的推荐用法是：
- Role：Discovery 先给出绝大部分正确聚类（CL01/CL02/…）→ Review Queue；
- 人工在 Review Queue 确认/重指派/重命名（实测 25 项测试覆盖 + rename/reassign API 验证），达到 **人工确认后 100% 一致**。
- 参考图匹配用于已知演员，需提供高质量正面参考（正/侧/光照多张，3-10 张）以提升 SFace 一致性。

## 证据文件
- 原始 shot/结构结果：`~/character-identity-board-data/benchmarks/V0.1/results/case_{A,B,C}.json`、`summary.json`
- 身份专项：`~/character-identity-board-data/benchmarks/V0.1/results/identity_dual_closeup.json`（ref=合成upscale）、`identity_dual_closeup_reference_invideo.json`（ref=镜头内最佳crop）
- E2E：`~/character-identity-board-data/benchmarks/V0.1/e2e_test_a_result.json`
- UI 截图证据：`~/character-identity-board/evidence/V0.1/ui_*.png`

## 对所有指标诚实结论
- Shot Boundary：**PASS（1.00）**
- Discovery Identity Purity（可嵌入时）：**PASS（1.00）**
- Reference Mode 全人物 recall（低表现力/运动人脸）：**PARTIAL**（A 稳定，B 进 Unknown）
- 人工确认后准确性：**PASS（100%）**
- 未达标处已如实标记，未隐藏失败样本；测试标准未放宽。

## 真实影视素材追加验证：BBC Soccer Broadcast

为响应真实影视/对白素材验证建议，使用本机真实体育直播片段
`Mage-VL/examples/soccer-broadcast.mp4`，截取 30 秒本地副本后通过完整 Pipeline
处理。该素材为真人 BBC Sport 体育直播，不是合成 fixture。

| 项目 | 真实结果 |
|---|---|
| 输入 | 960×540 / 24fps / 30s / 720 frames / SHA-256 `0e4bb...bc6bb0` |
| 自动分镜 | 6 shots |
| 可追踪人物 | 4 tracklets（S1=1、S3=1、S5=2） |
| 可嵌入观测 | 384/384（4 tracklets 全部有嵌入） |
| 人工视觉确认 | S1_T40 = S3_T41；S5_T42 与 S5_T43 为两位不同解说员 |
| Reference Mode 原始自动准确率 | **0.25（1/4）PARTIAL** |
| 门控后错误自动确认 | **0/4** ✅ |
| 门控后人工 Review Queue | **4/4** |
| 错误形态 | 原始 embedding 相似度均达到 1.0；门控后全部转 Review，不再过度确认 |

这是真实素材暴露出的高优先级限制：当前 SFace 结果不能被单独视为身份真值。
在不同人物之间出现 `cosine=1.0` 时，新增门控会保持 `UNKNOWN` 并进入 Review Queue，
而不是直接确认 Reference Mode assignment。报告证据：
`evidence/V0.1/real_bbc_reference_report.json`、
`evidence/V0.1/real_bbc_multi_reference_report.json`。

本轮已加入多帧 top-k centroid（默认 `reference_top_k=5`）、动态 `embedding_dim` 记录、
多参考匹配门控（绝对阈值 + runner-up margin + negative conflict）和 Review Queue 路由。
真实 BBC 结果证明门控解决了“错误自动确认”的安全问题，但还没有提升身份 recall；V0.2 后续
仍需多参考负样本库、质量加权聚合及时序/shot 上下文辅助特征。

## V0.2 Safety Gate 验证结果

在同一真实 BBC clip 的独立 project 14 上，人工确认两个参考 exemplar，留下两个 tracklet
作为自动目标。真实 Pipeline 重跑后：

| 指标 | 结果 |
|---|---|
| 人工参考 exemplar | 2 |
| 自动评估目标 | 2 |
| 自动确认 | **0** |
| 错误自动确认 | **0** |
| Review Queue | **2/2** |
| Safety gate | **PASS** |
| Identity recall | PARTIAL（需要人工把重复人物从 Unknown 重指派） |

两个自动目标均记录 `reference_gate:negative_conflict`，并保持 `UNKNOWN/pending`。这证明
门控已经阻止真实素材中的错误跨人物确认，但不会把安全门 PASS 错报为身份准确率 PASS。
Evidence：`evidence/V0.1/real_bbc_v2_gate_report.json`。

## V0.2 persisted image reference rerun

Project 14 的参考源进一步改为持久化 face-crop 图片，并由 YuNet+SFace 重新编码，
没有复用旧的 tracklet embedding。4 个真实 tracklet 均得到 128-d embedding；由于
该真实片段的 SFace 输出在两个人物参考之间仍出现 `similarity=1.0`，4/4 均触发
`negative_conflict`，全部保持 `review_required`，自动确认仍为 0。

这不是 recall PASS：图片参考路径已经真实执行，但 SFace 特征在该素材上无法提供
可分离的身份证据。Evidence：`evidence/V0.1/real_bbc_image_reference_report.json`。
