# CHANGELOG.md — Character Identity Board V0.1

## [0.1.0] — 2026-08-17
首个可运行 V0.1 交付。

### 真实素材追加验证
- 使用真实 BBC Sport soccer broadcast 30s clip 完成完整 Pipeline：6 shots、4 tracklets、384/384 embeddings。
- 人工视觉确认 S1/S3 为同一人物、S5 为两位不同人物；Reference Mode 自动准确率 1/4，确认跨人物过度归类风险，最终状态改为 `PARTIAL`。
- 加入多帧 top-k embedding aggregation（默认 5）、动态 `embedding_dim`、可配置 `face_size_gate`；27 项测试 PASS。

### 新增
- 端到端系统：上传视频 → 自动分镜 → 人物检测/追踪 → 人脸嵌入 → 聚类为 A/B/C/Unknown → UI 呈现 → 人工纠错 → 导出。
- 后端 FastAPI + SQLAlchemy + Pydantic（后端 `backend/app/`，~3,500 行）。
- 数据模型按 Mission §4 完整建模（Project/Video/Shot/Tracklet/FaceObservation/Character/IdentityAssignment/ReviewAction），SQLite(WAL)→PostgreSQL 就绪。
- 分镜：PySceneDetect ContentDetector/AdaptiveDetector，可调 threshold/min_len；支持手动增删改/合并/拆分镜头。
- 人脸：YuNet 检测 + SFace 嵌入（License 合规，规避 InsightFace），CUDA→CPU 回退。
- 追踪：镜头内 IoU tracker，Shot 切换重置。
- 聚类：HDBSCAN Discovery + 余弦 Reference 匹配，低置信/Unknown 进 Review。
- 人工纠错：rename/merge/split/reassign/exclude/mark_unknown/set_reference/confirm/recluster（全部记 ReviewAction，为 Undo/Redo 预留，保留 manual 结果）。
- 导出：JSON / CSV / Contact Sheet / HTML 报告。
- 前端：纯静态中文 SPA（零构建），Projects/Upload/People/ShotBoard/Continuity/ReviewQueue 六视图。
- 自动化测试：25 项 PASS（api/database/corrections/pipeline/quality）。
- Benchmark：Test A/B/C 三组真实视频 + GROUND_TRUTH + run_benchmarks 评估（Shot F1=1.0）。
- 证据：Playwright 真实渲染 UI 截图（`evidence/V0.1/`）。
- Docker Compose + Dockerfile 部署；docs（AUDIT/ARCH/DATA_MODEL/API/TEST_PLAN/LICENSE_NOTES）。

### 修复（测试驱动，来自手工验证）
- merge_characters / delete_character 在 ReviewAction 存在 FK 引用时的删除被阻塞 → 重指向并置空引用，保留历史。
- 拆分/合并代码唯一性（character_code unique per project）。
- `clustering._cosine_sim` 仅扁平化了一个向量：当参考嵌入带 batch 维 `(1,N)`（`FaceEngine.get_embedding` 常见返回）时，`match_to_references` 触发 `ValueError: shapes (N,) and (1,N) not aligned`，导致 Reference Mode 直接崩溃。修复：对两个输入都 `.reshape(-1)`（与 `FaceEngine.similarity` 语义一致）。<br/>*复现：用户上传参考图 → reference_match；修复后比对可在该路径进行。*

### 已知限制（诚实记录）
- 自动身份聚类对"合成静态人像 fixture"未完整分离 A/B：FACE2 人脸约 83px < 90px 嵌入阈值 → 正确进入 Unknown（非聚类逻辑错误）。
- SFace 对同人不同姿势（正面 vs 侧脸）可能给出较低相似度（实测 ~0.30），对异人相似姿势可能偏高（0.86）→ 自动聚类对强姿势变化敏感；需人工确认或参考图（Reference Mode）。
- 存储的 `embedding_dim` 元数据写死 128，但 SFace 实际 512-d（不影响运行时，仅元数据字段不准确）。
- 前端为纯静态零构建实现，未使用 Next.js（满足"可运行、隔离、可恢复"，但非 Mission 建议的 React/Tailwind 栈；如需可平滑升级）。

## [0.0.1] — 2026-08-14
- 初始骨架与数据模型草稿。
- 环境审计与 License Gate 决策（YuNet/SFace 采用，InsightFace 规避）。
