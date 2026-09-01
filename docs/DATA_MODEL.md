# DATA_MODEL.md — Character Identity Board V0.1

ORM: SQLAlchemy 2.x（`backend/app/models.py`）。V0.1 默认 SQLite（WAL + foreign_keys + busy_timeout），结构设计 PostgreSQL 就绪：不使用 SQLite 特有 idiom，通过 `database_url` 环境变量切换。

## 实体关系（Mission §4 全部覆盖）

```
Project 1──* Video
              │ 1──* Shot
              │              │ 1──* Tracklet 1──* FaceObservation
Project 1──* Character
Tracklet ── IdentityAssignment ──> Character
Project 1──* (ReviewAction)          (历史日志，可 Undo/Redo)
```

## 表设计

### Project
| 字段 | 类型 | 说明 |
|---|---|---|
| id | Integer PK | |
| name | String(200) | 项目名 |
| status | String(50) | created / processing / completed / failed |
| settings | JSON | 项目级设置 |
| created_at / updated_at | DateTime(UTC) | |

### Video
| 字段 | 类型 | 说明 |
|---|---|---|
| id | PK | |
| project_id | FK→Project | |
| filename / filepath | String | 原始文件 + 落盘路径 |
| duration_s / fps / width / height / codec / frame_count | | ffprobe 元数据 |
| processing_status | String(50) | uploaded/processing/completed/failed |
| pipeline_stage | String(50) | checkpoint/resume 阶段 |
| error_stage / error_message | | 失败定位 |

### Shot
| 字段 | 类型 | 说明 |
|---|---|---|
| id | PK | |
| video_id | FK→Video | |
| shot_number | Int | 镜头序号 |
| start_frame / end_frame | Int | 起止帧 |
| start_time_ms / end_time_ms / duration_ms | Int | 时间码 |
| representative_frame | String | 主代表帧路径 |
| detection_method | String(30) | ContentDetector / AdaptiveDetector / manual |
| confidence | Float | |
| **continuity_score** | Float null | **V0.2 预留**：人物一致性评分 |
| **restored** | Bool | **V0.2 预留**：是否已修复 |

### Tracklet（人物在单镜头内的连续出现）
| 字段 | 类型 | 说明 |
|---|---|---|
| id | PK | |
| shot_id | FK→Shot | |
| track_number | Int | 镜头内人物序号 |
| start_frame / end_frame | Int | 轨迹起止 |
| person_bbox | JSON | [x,y,w,h] 原帧坐标 |
| best_face_observation_id | FK→FaceObservation | 最佳脸观测 |
| best_body_observation_id | FK→FaceObservation | 最佳身体观测 |

### FaceObservation
| 字段 | 类型 | 说明 |
|---|---|---|
| id | PK | |
| tracklet_id | FK→Tracklet | |
| frame_number / timestamp_ms | | 帧序号 + 时间戳 |
| face_crop_path / portrait_crop_path / body_crop_path | String | 脸/头肩/全身截图（XML §7 要求全保留） |
| face_bbox | JSON | |
| quality_score / blur_score / occlusion_score | Float | 质量/模糊/遮挡 |
| **yaw / pitch / roll** | Float null | **V0.2 预留**：姿态 |
| embedding | LargeBinary | SFace 512-d float32 原始字节 |
| embedding_dim | Int | 存储维度元数据 |
| **excluded / exclude_reason** | Bool/String | 人工排除 |

### Character
| 字段 | 类型 | 说明 |
|---|---|---|
| id | PK | |
| project_id | FK→Project | |
| display_name | String | 例如 "Character A" / "Unknown" |
| character_code | String | A/B/C, UNKNOWN, CLxx, MANxx |
| reference_image | String null | 代表图 |
| status | String | unknown / manual |
| created_by | String | system / manual |

### IdentityAssignment
| 字段 | 类型 | 说明 |
|---|---|---|
| id | PK | |
| tracklet_id | FK→Tracklet (unique) | 每 Tracklet 一条 |
| character_id | FK→Character | 归属人物 |
| confidence | Float | |
| assignment_source | String | reference_match / automatic_cluster / manual |
| review_status | String | pending / confirmed |
| note | String null | |

### ReviewAction（全部人工修改历史，为 Undo/Redo 预留）
| 字段 | 类型 | 说明 |
|---|---|---|
| id | PK | |
| action_type | String | rename_character/merge_characters/split_character/reassign_tracklet/confirm_tracklet/exclude_observation/manual_shot_edit/set_reference/set_unknown/delete_character/recluster |
| source_character_id / target_character_id | FK→Character null | 注意：merge/delete 时对已删 character 的引用会置 null，历史保留但不会构成 FK 阻塞 |
| tracklet_id | FK→Tracklet null | |
| before_state / after_state | JSON | 前后快照，Undo/Redo 依据 |
| created_at | DateTime | |

## 关键约束
- `Character.character_code` 每项目唯一（测试覆盖）。
- `Tracklet.track_number` 每 Shot 唯一（测试覆盖）。
- `IdentityAssignment.tracklet_id` 唯一（一个人物轨迹一归属）。
- 内置 Unknown 字符不可删（测试覆盖）。
- Merge/Split 不破坏 Shot/Tracklet 原始数据，仅重指向 IdentityAssignment（Mission §12）。

## PostgreSQL 切换说明
```bash
export CIB_DATABASE_URL="postgresql+psycopg://user:pass@localhost/cib"
```
ORM 全部使用跨方言类型（Integer/String/Float/DateTime/JSON/LargeBinary），无需改代码。需安装 `psycopg[binary]`。
