# API.md — Character Identity Board V0.1

FastAPI 自动生成 OpenAPI：`http://127.0.0.1:8322/openapi.json`，Swagger UI 默认在 `/docs`。

## 概览

- Base：`http://127.0.0.1:8322`
- 健康检查：`GET /health` → `{"status":"ok","version":"0.1.0"}`
- 前端：`GET /`（SPA），静态资源 `/static/*`，媒体（crops/thumbnail/video）`/media/*`
- 全部请求/响应为 JSON，文件上传用 `multipart/form-data`

## 端点清单（对应 Mission §14 需求）

### 项目
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | /projects | 创建项目 `{name, settings?}` |
| GET | /projects | 列出全部项目（含 video_count/character_count/shot_count） |
| GET | /projects/{project_id} | 项目详情 |
| DELETE | /projects/{project_id} | 删除项目（级联） |

### 视频
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | /projects/{project_id}/videos | 上传视频（multipart `file`）→ 落盘 + 返回 Video |
| POST | /projects/{project_id}/videos/process | 启动后台 Pipeline（九阶段，可恢复） |
| GET | /projects/{project_id}/videos | 列出该项目的视频 |
| GET | /projects/{project_id}/videos/{video_id}/status | 处理状态 + pipeline_stage + error |

### 镜头
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | /projects/{project_id}/shots | 列出视频的全部 Shot |
| PATCH | /shots/{shot_id} | 手动修改起止帧/时间码 |
| POST | /shots/{shot_id}/split | 在 split_frame 拆分镜头 |
| POST | /shots/merge | 合并多个镜头 |
| GET | /shots/{shot_id}/tracklets | 镜头内 Tracklet |

### 人物
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | /projects/{project_id}/characters | 列出人物（含 tracklet_count/shot_count/avg_confidence/pending_review） |
| POST | /characters | 手动新增人物 |
| PATCH | /characters/{char_id} | 重命名 / 改代码 |
| POST | /characters/{char_id}/split | 指定 tracklet_ids 拆出新人物 |
| POST | /characters/merge | 合并两人物（`{source_character_id, target_character_id}`） |
| POST | /characters/{char_id}/mark-unknown | 该人物全部 tracklet 标记 Unknown |
| POST | /characters/{char_id}/reference | 上传参考图（multipart `file`） |
| DELETE | /characters/{char_id}/reference/{index} | 删除某张参考图 |
| GET | /characters/{char_id}/reference-pack | 获取参考图集合 |
| GET | /characters/{char_id}/observations | 该人物的全部观测 |

### Tracklet / Observation 纠错
| 方法 | 路径 | 说明 |
|---|---|---|
| PATCH | /tracklets/{tracklet_id}/assignment | 重指派到指定 character（`{character_id, note?}`） |
| DELETE | /tracklets/{tracklet_id} / GET | 查看/删除 |
| PATCH | /observations/{obs_id}/exclude | 排除/恢复单条观测 `{exclude: bool, reason?}` |

### 复核与导出
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | /projects/{project_id}/review-queue | 低置信/Unknown/边界样本队列（含 reasons） |
| GET | /projects/{project_id}/review-actions | 全部人工修改历史 |
| POST | /projects/{project_id}/recluster | 重聚类（保留 manual 结果） |
| POST | /projects/{project_id}/export | 生成 JSON+CSV+Contact Sheet+HTML，返回文件清单 |

## 请求/响应示例

### 创建项目
```bash
curl -X POST http://127.0.0.1:8322/projects \
  -H 'Content-Type: application/json' \
  -d '{"name":"My Project"}'
# => {"id":11,"name":"My Project","status":"created",...}
```

### 上传视频
```bash
curl -X POST http://127.0.0.1:8322/projects/11/videos \
  -F "file=@/path/to/clip.mp4"
```

### 启动处理
```bash
curl -X POST http://127.0.0.1:8322/projects/11/videos/process
# => {"video_id":..., "status":"processing","pipeline_stage":"uploaded",
#     "message":"Pipeline started in background thread","log_path":".../backend.log"}
```

### 查询状态
```bash
curl -s http://127.0.0.1:8322/projects/11/videos/{vid}/status
```

### 重命名人物 / 查看修改历史
```bash
curl -X PATCH http://127.0.0.1:8322/characters/{id} -H 'Content-Type: application/json' -d '{"display_name":"Character A"}'
curl -s http://127.0.0.1:8322/projects/11/review-actions
```

## 状态码约定
- 200 / 201：成功
- 204：无内容（删除类）
- 400：参数/校验错误
- 404：资源不存在
- 409：冲突（如 merge-self、删除 Unknown 被拒）
- 500：服务端异常（写入 `Video.error_stage/error_message`）

## 错误处理
- Pydantic 自动校验 body/query（400 带 detail）。
- Pipeline 失败在数据库记录具体失败阶段与信息，API 返回 `failed` 状态 + `error_stage/error_message`。
- 所有修改类操作写入 `ReviewAction` 历史（为 Undo/Redo 预留）。
