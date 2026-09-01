# Character Identity Board — 系统配置总览

## 🏗️ 基础架构

| 项目 | 值 |
|------|-----|
| **项目路径** | `~/character-identity-board` |
| **数据路径** | `~/character-identity-board-data` |
| **数据库** | SQLite (`cib.sqlite3`) |
| **后端** | FastAPI (Python 3.12) |
| **前端** | 原生 HTML/CSS/JS (无框架) |
| **API 地址** | `http://127.0.0.1:8322` |
| **设备** | CPU (`use_gpu=True` 但实际跑 CPU) |

---

## 🧠 AI 模型

| 功能 | 模型 | 来源 | License | 说明 |
|------|------|------|---------|------|
| **人脸检测** | YuNet | OpenCV Zoo | MIT | 检测人脸 + 5点关键点 (眼/鼻/嘴角) |
| **人脸特征** | SFace | OpenCV Zoo | Apache-2.0 | 128维 embedding, L2 归一化 |
| **聚类** | HDBSCAN | scikit-learn | BSD | `min_cluster_size=15` |
| **镜头检测** | ContentDetector | PySceneDetect | — | `threshold=27.0`, `min_len=12帧` |

---

## ⚙️ 关键参数

```python
# 身份匹配
identity_threshold = 0.85    # 同一个人的最低余弦相似度
merge_threshold    = 0.88    # 合并 Cluster 的阈值
unknown_threshold  = 0.80    # 标记为 Unknown 的阈值
reference_margin   = 0.10    # 参考模式的 margin
reference_top_k    = 5       # 参考模式取前 K 个嵌入

# 质量过滤
blur_threshold     = 28.0    # 模糊分数阈值 (Laplacian variance)
face_size_gate     = either  # 最小 90px, 最大 600px

# 跟踪
track_iou_threshold   = 0.3
max_track_frames_lost = 12

# 聚类
hdbscan_min_cluster_size = 15
```

---

## 📊 当前数据

| 指标 | 数值 |
|------|------|
| **视频镜头数** | 1,078 |
| **Tracklet 数** | 2,074 |
| **人脸检测数** | 有效 114,207 / 排除 41,987 |
| **人物 Character** | 15 |
| **有参考照片** | 14 个 (SMY, 院长, LW 等) |

---

## ⚠️ 已知问题

1. **SFace 跨镜头相似度低** — 同一个人在不同镜头的余弦相似度只有 0.3-0.57（理想应 >0.85），导致自动聚类不可靠
2. **HDBSCAN 聚类失效** — 由于 SFace 嵌入不稳定，聚类结果不对应真实人物
3. **误检** — YuNet 会检测非人脸物体（药瓶、镜头、卡通）

## 💡 推荐优化方向

| 方向 | 说明 | 难度 |
|------|------|------|
| **换 ArcFace/InsightFace** | 更强的特征模型，跨镜头相似度更高 | ⚠️ 授权问题 |
| **GPU 加速** | YuNet + SFace 支持 CUDA，速度提升 10x | 低 |
| **参考模式工作流** | 用户上传真实照片 → 系统匹配（当前方案） | ✅ 已实现 |
| **姿态过滤** | 用 YuNet 关键点估算 yaw/pitch，只保留正面 | 中 |
| **多人模型** | 用 YOLOv8-Face 替代 YuNet，检测更准 | 中 |

---

## 🔄 工作流程

```
1. 上传真实人物正面照片（1-3张/人）
   ↓
2. 系统提取 SFace 嵌入（128维）
   ↓
3. 与电影中所有人脸嵌入做余弦相似度比较
   ↓
4. 相似度 > 0.85 → 匹配成功
   相似度 0.50-0.85 → 需要人工确认
   相似度 < 0.50 → 不匹配
```

---

*Generated: 2026-09-01*
