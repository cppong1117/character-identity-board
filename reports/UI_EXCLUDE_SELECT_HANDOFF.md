# CIB 废除 / 选择 UI 交接文档（给修 UI 的人）

**Date:** 2026-09-03  
**Project:** Character Identity Board (`film`, project_id=15)  
**Live URL:** http://localhost:8322  
**Repo:** https://github.com/cppong1117/character-identity-board  
**Data DB:** `/home/ponky_re6000/character-identity-board-data/cib.sqlite3`  
**Frontend:** `frontend/static/index.html` + `frontend/static/app.js`  
**Backend:** FastAPI uvicorn on port **8322**

---

## 1. 用户真实痛点（模拟真人操作后的结论）

用户要做的事很简单：

1. 打开某个角色（例如 SMY / ZY / DOCTOR）
2. 看到该角色下面挂着的脸缩略图
3. **挑出“不是这个人 / 不是人脸 / crop 错位”的图**
4. **批量废除**这些图
5. 不要一点就整页刷新、不要难点、不要点了没反应

当前 UI **很难操作**，不是用户不会用，是实现有多个硬 bug。

---

## 2. 当前真人操作路径（实际能点到什么）

### A. 进入项目
1. 打开 http://localhost:8322
2. Dashboard 点项目卡片 `film`（project 15）
3. 进入 detail view（`#view-detail`）
4. Tabs：
   - Shots
   - **Characters** ← 用户主要在这里想废除错脸
   - **Review Queue** ← 待审队列

### B. Characters 页（废除/选择）
每个角色卡显示：
- 角色名输入框
- tracklet/shot/confidence 统计
- 最多 **6 张** face crop 缩略图
- 按钮：
  - `☐ 选择`
  - `✗ 排除选中`
  - `✓ 重命名`
  - `✗ 删除`（删整个 character，不是删脸）

**期望交互：**
- hover 缩略图 → 右上角红 ✕
- 点 ✕ / 点图 → 排除该脸
- 点 `☐ 选择` → 多选
- 点 `✗ 排除选中` → 批量排除

**实际结果：基本不可用（见第 4 节 bug）**

### C. Review Queue 页
每条 pending item：
- checkbox
- face crop
- shot / character / confidence / quality
- 下拉选人物
- `✓ 确认` / `Unknown` / `✗ Not a face`
- 顶部：
  - 全选
  - `✗ Not a face (批量)`
  - Confirm high-conf
  - Exclude low-conf

Review Queue 的批量 not-face 相对 Characters 页更可用，但仍不是“按角色整理错脸”的主路径。

---

## 3. 当前数据现状（project 15）

### Characters
| id | name | code | tracklets | shots | pending | avg conf |
|----|------|------|-----------|-------|---------|----------|
| 20 | Unknown | UNKNOWN | 1645 | 702 | 1 | 0.91 |
| 25 | ZY | CL06 | 55 | 42 | 4 | 0.94 |
| 27 | SMY | CL08 | 206 | 176 | 13 | 0.94 |
| 30 | DOCTOR | CL04 | 87 | 72 | 2 | 0.94 |
| 31 | mbq | CL01 | 36 | 35 | 0 | 0.95 |
| 36 | lw | CL00 | 78 | 76 | 0 | 0.95 |
| 32 | Test Actor | TEST | 0 | 0 | 0 | 0 |
| 33 | SMY | SMY | 0 | 0 | 0 | 0 |
| 34 | 院长 | 院长 | 0 | 0 | 0 | 0 |
| 35 | LW | LW | 0 | 0 | 0 | 0 |

注意：有空壳角色 / 重名（SMY 有两个、lw/LW 重复）。

### Review Queue (pending only)
- **20 pending**
- 分布：SMY 13 / ZY 4 / DOCTOR 2 / Unknown 1

### Face observations 全局
- Total: **156,194**
- Active (not excluded): ~**89,376**
- Excluded: ~**66,818**
- Embedding: **全部 ArcFace 512-dim**
- ArcFace 已启用：`use_arcface=True`
- Thresholds: identity=0.40, merge=0.50, unknown=0.30

### 用户点名的问题 shot
- Shot **#732, #777, #778, #780**
- 现象：crop 不在正确脸部位置 / 场景可能无人脸 / 视觉上“不是人样”
- 这些不一定都在 pending queue 里（很多已 auto_assigned 到角色卡里）

---

## 4. 关键 Bug（为什么难操作）

### BUG-1 【阻断】Characters 页排除 API 缺 observation id
**API：** `GET /characters/{id}/observations`

当前返回字段：
```json
{
  "shot_number": 28,
  "tracklet_id": 56,
  "timecode_start": "...",
  "timecode_end": "...",
  "face_crop_path": "...",
  "portrait_crop_path": "...",
  "body_crop_path": "...",
  "quality_score": 0.71,
  "blur_score": 41.0,
  "identity_confidence": 0.95,
  "review_status": "auto_assigned"
}
```

**缺少：**
- `id`（observation id）
- `excluded`
- `exclude_reason`
- `face_bbox`
- 原图/shot 代表帧路径（方便对照 crop 是否错位）

但前端却写：
```js
data-obs="${o.id}"
onclick="excludeObservation(${o.id}, this)"
```
所以实际发出：
```
PATCH /observations/undefined/exclude
```
或 `NaN` → **排除失败**。

**修复：** `backend/app/api/characters.py` `char_observations()` 必须返回：
```python
"id": o.id,
"excluded": o.excluded,
"exclude_reason": o.exclude_reason,
"face_bbox": o.face_bbox,
"original_frame_ref": getattr(o, "original_frame_ref", None),
```
并 filter `excluded != True`。

---

### BUG-2 【阻断】批量选择缺 `data-char`
前端：
```js
<img ... class="char-face-img" data-obs="${o.id}">
```
但：
```js
batchExcludeFaces(charId) {
  document.querySelectorAll(`.char-face-img[data-char="${charId}"].selected`)
}
toggleFaceSelect(charId) {
  document.querySelectorAll(`.char-face-img[data-char="${charId}"]`)
}
```

**没有 `data-char=...`** → 选择器永远空 → toast “请先选择要排除的脸”。

**修复：** img 加 `data-char="${c.id}" data-tracklet="${o.tracklet_id}"`。

---

### BUG-3 【体验】点图 = 直接排除，不是选择
当前：
- 单击缩略图 = 立即 exclude
- `☐ 选择` 只是“全选/取消全选”该角色当前 6 张，不是进入多选模式
- 没有单张 toggle select
- 没有 shift 多选 / 框选

用户期望更像：
1. 点选（高亮）
2. 再点“排除选中”
或：
- 明确的 checkbox per face

---

### BUG-4 【体验】每个角色只显示 6 张脸
```js
api(`/characters/${c.id}/observations?limit=6`)
faces = obs.filter(...).slice(0, 6)
```
但 backend **忽略 limit**，返回该角色全部 observations（SMY=206），前端再 slice 6。

结果：
- 用户只能看到 6 张
- 绝大多数错脸根本看不到
- 无法在角色内做完整清理

**修复建议：**
- 角色详情页 / 抽屉：分页或虚拟滚动展示全部 faces
- 支持 filter：pending / low quality / low conf / not reviewed
- 支持按 shot 排序、按 similarity-to-centroid 排序（最不像该角色的排前面）

---

### BUG-5 【语义混乱】exclude observation ≠ 从角色移除 tracklet
`PATCH /observations/{id}/exclude` 只把 `face_observations.excluded=true`。

它 **不会**：
- 自动改 `identity_assignments`
- 自动把 tracklet 标成 unknown / not-a-face
- 保证角色卡立刻不再统计该 tracklet（取决于下游 query 是否 filter excluded）

用户口头“废除”通常想要三者之一：
1. **Not a face**：这张/这个 tracklet 不是人脸 → 排除 observation + 确认 review 为 excluded/not-face
2. **Wrong person**：是人脸但不是这个角色 → reassign 到别的角色 / Unknown
3. **Bad crop**：检测框错位 → 排除 observation，最好能触发重检

当前 UI 把 1/2/3 混成一个红叉。

---

### BUG-6 【JS 结构】hover CSS 注入写在 DOMContentLoaded 外
`frontend/static/app.js` 第 17-24 行 style 注入在 listener 回调外，依赖执行时 `document.head` 已存在（通常 OK），但代码结构乱，像半成品 patch。

---

### BUG-7 【Review Queue 过滤】
`GET /projects/{id}/review-queue?status=pending` 已支持，默认 pending。  
但 Characters 页 pending badge 与可操作 faces 没有打通：  
有 pending 的角色，不一定能在 6 张预览里看到 pending 对应的那张脸。

---

## 5. 相关 API 清单

### 读
- `GET /projects`
- `GET /projects/{id}`
- `GET /projects/{id}/characters`
- `GET /characters/{id}/observations`  ← **缺 id，要修**
- `GET /projects/{id}/review-queue?status=pending|confirmed|all`
- `GET /tracklets/{id}`  （含 observations 列表，有 obs id / excluded）
- `GET /projects/{id}/shots`

### 写（废除/选择相关）
- `PATCH /observations/{obs_id}/exclude`
  ```json
  {"excluded": true, "reason": "not a face / wrong crop / ..."}
  ```
- `PATCH /tracklets/{tracklet_id}/assignment`
  ```json
  {"character_id": 27, "review_status": "confirmed"}
  ```
  或 unknown：
  ```json
  {"character_id": 0, "note": "Marked unknown by user"}
  ```
  或 not-a-face（当前用法）：
  ```json
  {"review_status": "confirmed", "note": "Excluded: not a real face"}
  ```

### 注意
- Review Queue 操作对象是 **tracklet assignment**
- Characters 缩略图排除对象是 **observation**
- 两套模型不一致，UI 必须让用户看懂

---

## 6. 建议的正确产品交互（给 UI 重做）

### 角色工作台（优先）
打开角色 SMY 后进入 **Face Grid**：

```
[Filter: All | Suspicious | Pending | Low QS | Low Conf]
[Sort: Least similar first | Shot order | Quality]
[Select mode]

[□ face] [□ face] [□ face] ...  (大图 96~128px，不要 56px)
每张显示：shot# · qs · conf · reason badge

批量动作：
- 废除 Not a face
- 移到 Unknown
- 重指定到角色...
- 取消选择
```

### 单张点开
左侧 crop，右侧 shot 代表帧 + bbox 叠加（确认是否 crop 错位）。

### 快捷键
- `X` not a face
- `U` unknown
- `1..9` assign character
- `J/K` next/prev
- `Space` select

### 不要
- 不要每点一次整页 reload
- 不要只显示 6 张
- 不要 hover-only 的 18px 红点当主操作
- 不要单击直接删除（太容易误触）

---

## 7. 后端最小修复清单（先让现有按钮能用）

1. **`char_observations` 返回 `id` 等字段**，支持 `limit/offset`，默认过滤 excluded
2. **前端 img 加 `data-char` / `data-tracklet` / `data-obs`**
3. **单击 = select toggle；按钮 = exclude selected**
4. **排除成功后不要 `loadCharacters()` 全量刷新**（或刷新时保持 scroll/selection）
5. **排除 observation 后可选同步 tracklet**：
   - 若 tracklet 的 best face 被 exclude，自动挑下一条 best，或把 assignment 打 pending/unknown
6. **角色 faces 支持 “least similar first”**（用 ArcFace embedding 对角色 centroid 的 cosine distance 排序）——这对“找出混进该角色的错脸”最关键

---

## 8. 视觉模型（Mage-VL）现状

- Service: `http://127.0.0.1:8011/v1` model `mage-vl`
- 已用于：
  - pending queue 60 条审查：39 not-face / 21 face
  - 启发式 + VLM：排除大量 small/borderline crops
- **限制：**
  - 高并发会卡死（10 workers 不可用；3 也不稳；1 worker 太慢）
  - 不该替代 UI；应作为后台 “suspicious queue” 生成器
- 建议：
  - 离线任务产出 `suspicious_faces` 表
  - UI 只消费结果：按角色展示 “AI 认为不像该人 / 不是脸” 的列表供一键确认

---

## 9. 关键代码位置

| 区域 | 路径 |
|------|------|
| Dashboard JS | `frontend/static/app.js` |
| HTML shell | `frontend/static/index.html` |
| Char observations API | `backend/app/api/characters.py` L128-155 |
| Exclude API | `backend/app/api/tracklets.py` L64-69 |
| Exclude logic | `backend/app/corrections.py` L247-261 |
| Review queue API | `backend/app/api/exports.py` |
| Assignment patch | `backend/app/api/tracklets.py` L42-61 |
| ArcFace config | `backend/app/config.py` (`use_arcface=True`) |

---

## 10. 复现步骤（给 QA）

1. 打开 http://localhost:8322
2. 进入 project `film`
3. Characters tab → 打开 `SMY`
4. 观察：只有 6 张小图
5. 点 `☐ 选择` → 无高亮（因为缺 data-char）
6. 点 `✗ 排除选中` → toast 请先选择
7. hover 点 ✕ → network 里 `PATCH /observations/undefined/exclude` 或失败
8. Review Queue tab → pending 约 20 条，可勾选批量 Not a face（这条路径相对可用）

---

## 11. 成功标准（修好的定义）

- [ ] 角色内能浏览该角色 **全部** faces（分页/虚拟列表）
- [ ] 能多选，批量 not-face / unknown / reassign
- [ ] 单次操作不整页重载，选中状态不丢
- [ ] 排除请求带正确 observation id，DB `excluded=1`
- [ ] 错位 crop 能对照 shot 原帧 + bbox
- [ ] 支持 “最不像该角色” 排序，方便清脏数据
- [ ] Review Queue 与 Character Face Grid 状态一致

---

## 12. 一句话总结

**不是用户不会选，是 Characters 页的选择/废除链路从头断了：API 不返回 observation id，前端选择器也缺 data-char，再叠加只显示 6 张和单击即删，导致“按人物清理错脸”几乎不可用。Review Queue 批量 not-face 只覆盖 pending，不覆盖已经挂到角色里的大量错脸。**
