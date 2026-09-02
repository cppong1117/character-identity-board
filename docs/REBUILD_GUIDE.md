# Character Identity Board — 完整重建指南

## 📋 项目概述

**目标：** 从视频中自动检测、跟踪并识别不同人物，建立人物身份数据库。

**技术栈：**
- 后端：FastAPI + SQLAlchemy + SQLite
- 前端：原生 HTML/CSS/JS
- AI 模型：YuNet (检测) + SFace (特征)
- 聚类：HDBSCAN

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (HTML/JS)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Dashboard │  │ Upload   │  │ Review   │  │ Export   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (8322)                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    API Layer                          │  │
│  │  /projects  /characters  /tracklets  /shots  /export │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  Core Services                        │  │
│  │  Pipeline │ FaceEngine │ Clustering │ Quality        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer (SQLite)                       │
│  Projects │ Videos │ Shots │ Tracklets │ Observations       │
│  Characters │ Assignments │ References                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 文件结构

```
character-identity-board/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── config.py            # Settings (Pydantic)
│   │   ├── database.py          # SQLAlchemy engine
│   │   ├── models.py            # ORM models
│   │   ├── schemas.py           # Pydantic schemas
│   │   ├── face_engine.py       # YuNet + SFace wrapper
│   │   ├── pipeline.py          # Video processing pipeline
│   │   ├── processors.py        # Frame/face processing
│   │   ├── clustering.py        # HDBSCAN clustering
│   │   ├── tracking.py          # IoU tracking
│   │   ├── quality.py           # Face quality scoring
│   │   ├── reference_embeddings.py  # Reference face handling
│   │   ├── corrections.py       # Merge/split/reassign
│   │   ├── reporting.py         # Export/report generation
│   │   └── api/                 # API endpoints
│   │       ├── projects.py
│   │       ├── characters.py
│   │       ├── tracklets.py
│   │       ├── shots.py
│   │       ├── review.py
│   │       ├── face_review.py
│   │       └── exports.py
│   ├── requirements.txt
│   └── tests/
├── frontend/
│   └── static/
│       ├── index.html           # Main dashboard
│       ├── style.css            # Styles
│       ├── app.js               # Frontend logic
│       ├── upload.html          # Reference photo upload
│       ├── label.html           # Seed labeling
│       └── clusters.html        # Cluster review
├── docs/
│   ├── API.md
│   ├── ARCHITECTURE.md
│   ├── DATA_MODEL.md
│   └── SYSTEM_CONFIG.md
└── scripts/                     # Utility scripts
```

---

## 🧠 核心模块详解

### 1. FaceEngine (`face_engine.py`)

**职责：** 人脸检测 + 特征提取

```python
class FaceEngine:
    def detect_faces(img_bgr) -> list[dict]:
        """返回 [{bbox, landmarks, score}, ...]"""
        
    def get_embedding(img_bgr, face_bbox) -> Optional[np.ndarray]:
        """返回 128-d L2-normalized 向量"""
        
    def similarity(e1, e2) -> float:
        """余弦相似度"""
```

**模型：**
- YuNet: `face_detection_yunet.onnx` (MIT)
- SFace: `face_recognition_sface.onnx` (Apache-2.0)

**关键参数：**
- 检测阈值: 0.5
- NMS 阈值: 0.3
- 输入尺寸: 320x320

---

### 2. Pipeline (`pipeline.py`)

**职责：** 视频处理流水线

**阶段：**
```
uploaded → probed → shots_detected → frames_extracted → 
tracklets_created → faces_embedded → clustered → completed
```

**每个阶段：**
1. 检测镜头边界 (PySceneDetect)
2. 提取关键帧 (ffmpeg)
3. 创建 Tracklet (IoU 跟踪)
4. 检测人脸 (YuNet)
5. 提取特征 (SFace)
6. 聚类/匹配 (HDBSCAN/Reference)

---

### 3. Clustering (`clustering.py`)

**职责：** 身份聚类

**两种模式：**

#### Discovery Mode (自动聚类)
```python
def cluster_hdbscan(embeddings, min_cluster_size=15):
    """HDBSCAN 聚类"""
```

#### Reference Mode (参考匹配)
```python
def match_to_references(tracklet_embs, reference_vectors, threshold=0.85):
    """与参考人脸匹配"""
```

---

### 4. Quality (`quality.py`)

**职责：** 人脸质量评估

```python
def score_face(face_crop, landmarks) -> dict:
    """返回 {quality, blur, occlusion}"""
```

**过滤条件：**
- `quality > 0.6`
- `blur > 27` (Laplacian variance)
- `occlusion < 0.3`

---

## 📊 数据模型

### 核心表

```sql
-- 项目
CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT DEFAULT 'created'
);

-- 视频
CREATE TABLE videos (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL,
    processing_status TEXT DEFAULT 'uploaded',
    pipeline_stage TEXT DEFAULT 'uploaded'
);

-- 镜头
CREATE TABLE shots (
    id INTEGER PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id),
    shot_number INTEGER NOT NULL,
    start_frame INTEGER,
    end_frame INTEGER,
    start_time_ms INTEGER,
    end_time_ms INTEGER
);

-- Tracklet (一个人在一个镜头中的连续出现)
CREATE TABLE tracklets (
    id INTEGER PRIMARY KEY,
    shot_id INTEGER REFERENCES shots(id),
    track_number INTEGER NOT NULL,
    start_frame INTEGER,
    end_frame INTEGER,
    UNIQUE(shot_id, track_number)
);

-- 人脸观测
CREATE TABLE face_observations (
    id INTEGER PRIMARY KEY,
    tracklet_id INTEGER REFERENCES tracklets(id),
    frame_number INTEGER,
    face_crop_path TEXT,
    face_bbox JSON,  -- [x, y, w, h]
    quality_score FLOAT,
    blur_score FLOAT,
    embedding BLOB,  -- 128-d float32
    excluded BOOLEAN DEFAULT FALSE
);

-- 人物
CREATE TABLE characters (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    display_name TEXT NOT NULL,
    character_code TEXT NOT NULL,
    reference_image TEXT,
    reference_pack JSON,  -- list of paths
    UNIQUE(project_id, character_code)
);

-- 身份分配
CREATE TABLE identity_assignments (
    id INTEGER PRIMARY KEY,
    tracklet_id INTEGER REFERENCES tracklets(id),
    character_id INTEGER REFERENCES characters(id),
    confidence FLOAT,
    assignment_source TEXT,  -- automatic_cluster | reference_match | manual
    review_status TEXT DEFAULT 'pending'  -- pending | confirmed | rejected
);
```

---

## 🔧 配置参数

```python
# config.py
class Settings:
    # 网络
    api_host: str = "127.0.0.1"
    api_port: int = 8322
    
    # 存储
    data_dir: Path = "~/character-identity-board-data"
    
    # 数据库
    db_path: Path = data_dir / "cib.sqlite3"
    
    # 镜头检测
    shot_detector: str = "ContentDetector"
    shot_threshold: float = 27.0
    shot_min_len_frames: int = 12
    
    # 跟踪
    track_iou_threshold: float = 0.30
    max_track_frames_lost: int = 12
    
    # 聚类
    hdbscan_min_cluster_size: int = 15
    identity_threshold: float = 0.85
    merge_threshold: float = 0.88
    
    # 质量
    min_face_size_px: int = 90
    blur_threshold: float = 28.0
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd character-identity-board/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 下载模型

```bash
mkdir -p ~/character-identity-board-data/cache/models
cd ~/character-identity-board-data/cache/models

# YuNet
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  -O face_detection_yunet.onnx

# SFace
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx \
  -O face_recognition_sface.onnx
```

### 3. 启动服务

```bash
cd character-identity-board/backend
uvicorn app.main:app --host 127.0.0.1 --port 8322
```

### 4. 访问界面

- Dashboard: http://127.0.0.1:8322
- Upload: http://127.0.0.1:8322/static/upload.html
- API Docs: http://127.0.0.1:8322/docs

---

## 📡 API 端点

### Projects
- `GET /projects` - 列出项目
- `POST /projects` - 创建项目
- `GET /projects/{id}` - 获取项目详情

### Videos
- `POST /projects/{id}/videos` - 上传视频
- `GET /projects/{id}/videos` - 列出视频

### Characters
- `GET /projects/{id}/characters` - 列出人物
- `POST /characters` - 创建人物
- `PATCH /characters/{id}` - 更新人物
- `DELETE /characters/{id}` - 删除人物
- `POST /characters/{id}/upload-reference` - 上传参考照片

### Tracklets
- `GET /projects/{id}/review-queue` - 获取待审核队列
- `PATCH /tracklets/{id}/assignment` - 更新分配

### Export
- `GET /projects/{id}/export` - 导出数据

---

## 🎯 工作流程

### 流程 1：自动处理

```
1. 上传视频
2. 系统自动：
   - 检测镜头边界
   - 提取关键帧
   - 创建 Tracklet
   - 检测人脸
   - 提取特征
   - 聚类
3. 查看结果
```

### 流程 2：参考匹配 (推荐)

```
1. 上传真实人物照片 (1-3张/人)
2. 系统提取特征
3. 与电影中所有人脸比较
4. 相似度 > 0.85 → 自动匹配
5. 查看并确认结果
```

---

## ⚠️ 已知问题与优化方向

### 问题

1. **SFace 跨镜头相似度低** (0.3-0.57)
   - 同一个人在不同镜头得到不同嵌入
   - 导致自动聚类不可靠

2. **误检**
   - YuNet 会检测非人脸物体
   - 需要后处理过滤

3. **性能**
   - CPU 处理慢
   - 1000+ 镜头需要数小时

### 优化方向

| 方向 | 说明 | 优先级 |
|------|------|--------|
| **换 ArcFace** | 更强的特征模型 | ⚠️ 授权问题 |
| **GPU 加速** | CUDA 加速检测+特征 | 高 |
| **姿态过滤** | 只保留正面人脸 | 中 |
| **批量处理** | 多线程/GPU 并行 | 中 |
| **增量处理** | 支持断点续传 | 低 |

---

## 📝 开发笔记

### SFace 的问题

SFace 是 OpenCV 提供的轻量级人脸特征模型，优点是：
- 完全免费 (Apache-2.0)
- 跨平台 (CPU/GPU)
- 轻量级 (< 10MB)

但缺点是：
- 跨镜头相似度低 (同一个人 0.3-0.57)
- 不如同类商业模型 (ArcFace > 0.9)

### 解决方案

由于 SFace 跨镜头不可靠，我们采用 **参考模式**：
1. 用户提供真实人物照片
2. 用这些照片作为锚点
3. 与电影人脸比较
4. 人工确认匹配结果

---

*Generated: 2026-09-01*
