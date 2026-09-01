# ARCHITECTURE.md — Character Identity Board V0.1

## 系统定位

用户上传一部影片 → 系统自动识别 Shot → 提取镜头内人物 → 建立人物 Tracklet → 生成人脸/头肩/全身截图 → 参考人物匹配或自动聚类为 人物A/B/C/Unknown → 通过 UI 按镜头顺序排列整场戏人物画面 → 支持人工纠错（改名/合并/拆分/重指派/Unknown/参考图）→ 导出连续性与报告。

V0.1 **不做**自动换脸/重绘/面部修复，但数据模型已为 V0.2 预留接口。

## 模块化架构（后端）

```
frontend/static/  (SPA: index.html + app.js + style.css, 零构建)
       │  同源 HTTP (FastAPI)
backend/app/
    main.py       应用入口，静态托管 frontend + /media
    config.py     Settings（CIB_ env 前缀），数据/模型/日志目录，GPU 探测
    database.py   SQLAlchemy engine/session，SQLite(WAL)→PostgreSQL 就绪
    models.py     ORM（Project/Video/Shot/Tracklet/FaceObservation/
                  Character/IdentityAssignment/ReviewAction…）
    schemas.py    Pydantic 请求/响应
    pipeline.py   视频处理编排（checkpoint/resume 九阶段）
    shot_detection.py   PySceneDetect Content/Adaptive 检测 + 抽帧策略
    processors.py 每 Shot 解码/检测/追踪/crop→tracklet
    face_engine.py YuNet(SFace) 人脸检测/嵌入（CUDA→CPU 回退）
    tracking.py   镜头内追踪器（IoU + 丢失容忍，切换时重置）
    quality.py    人脸质量评分（清晰度/尺寸/遮挡/亮度/置信）
    clustering.py 分布式 HDBSCAN + 参考匹配 + merge 建议
    corrections.py 全部人工纠错服务（rename/merge/split/reassign/…）
    reporting.py  导出（JSON/CSV/Contact Sheet/HTML）
    api/          路由（projects/videos/shots/characters/tracklets/exports/review）
```

### 分层原则
- **ORM 层**：仅数据访问；V0.1 SQLite（WAL），通过 `database_url` 可切 PostgreSQL（psycopg）。
- **服务层**（pipeline/corrections/reporting）：业务规则；不与 HTTP 框架耦合，便于测试。
- **API 层**：FastAPI 路由，薄壳，参数校验 + 状态码。
- **前端**：纯静态 SPA，无 Node 构建链，直接调 API。

## Video Processing Pipeline（九阶段 + 可恢复）

```
uploaded → probed → shots_detected → frames_extracted
        → tracklets_created → faces_embedded → clustered
        → thumbnails_generated → completed
```
- 每个阶段在 `Video.pipeline_stage` 持久化；中断后重跑从该阶段恢复（Mission §15/16）。
- 失败写入 `error_stage` / `error_message`。
- 不会一次把整条影片载入 RAM/VRAM：按 Shot 顺序流式解码，中间结果(crops/frames/overlays)落盘。

## Shot Detection（Mission §6）
- 默认 PySceneDetect `ContentDetector`(threshold=27)，可选 `AdaptiveDetector`。
- 参数：threshold / min_scene_len_frames / fade handling。
- UI/API 支持手动新增/删除/修改/合并/拆分 Shot 边界，全部记录 ReviewAction。

## 镜头内追踪（Mission §8）
- `ShotTracker`：逐帧人脸检测 → 以 IoU 关联建立 Tracklet；丢失容忍 `max_track_frames_lost`。
- **Shot 切换后 Tracker 重置**，避免跨镜头错误沿用 Track ID。
- 每个 Tracklet 选最优观测（质量最高）作为代表；背脸/遮挡/低质量帧不污染聚类。

## 身份识别（Mission §9/10）
- **Reference Mode**：用户上传人物 A/B/C 参考图（3-10 张/人）→ 按 SFace 余弦相似度匹配；低于 `identity_threshold`(0.85) 必须进 Unknown。
- **Discovery Mode**：无参考时用 **HDBSCAN** 自动聚类（无需预设人物数 K，Mission 明确禁用 K-means 作为唯一方案）；噪声→Unknown；用户可将 Cluster 重命名/合并/拆分。
- 质量门槛：人脸 < `min_face_size_px`(90) 或模糊/遮挡/出画均**不产生嵌入**，观测进入 Unknown/Review Queue，不强塞给某个人物。

## 人工纠错（Mission §12）
- rename / merge（FK 安全，来源并入目标后删源，ReviewAction 引用置空以保历史）/ split（指定 tracklet 移入新 character）/ reassign 单个 tracklet / exclude 单条 observation / mark_unknown / set_reference / confirm / recluster（**保留人工确认结果**）。
- 人工确认优先级 > 自动聚类。

## 导出（Mission §13）
- JSON（完整全量）/ CSV（shot/character/观察路径/置信度/review）/ Contact Sheet（每人一图按 Shot 排序）/ HTML（可离线打开的整场戏报告）。

## 数据流（上传→完成）

```
上传视频 → ffprobe → Shot 检测 → 抽帧(前/中/后端采样)
→ 逐帧人脸检测+追踪 → 建立 Tracklet → 生成 face/portrait/body crops
→ 选最佳观测 → 生成 SFace 嵌入 → 参考匹配或 HDBSCAN 聚类
→ 生成 Character A/B/C/Unknown + IdentityAssignment → 落库
→ UI 显示 → 人工确认 → 导出
```

## V0.2 预留（Mission §20）
- `Character.reference_image` / reference-pack（正/左/右脸参考）。
- `Shot.continuity_score` / `Shot.restored`（一致性评分 / 修复状态）。
- `FaceObservation.yaw/pitch/roll`、`excluded`。
- `ReviewAction` 为 Undo/Redo 预留（action_type + before/after 快照）。
- 存储层经 `project_dir` 抽象，未来可切 NAS/MinIO/S3。
- 可为 Face Restore / Face Swap / ComfyUI workflow 调用预留字符/JSON 字段结构。
