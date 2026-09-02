# Phase 2 — Coordinate / Crop / Alignment Audit

## DATE: 2026-09-02

---

## RESULT: FAIL — P0 ALIGNMENT BUG

---

## 结论

**`get_embedding()` 收到的 bbox 只有 4 个元素 (x, y, w, h)，但 SFace `alignCrop()` 需要 15 个元素 (bbox + 5 landmarks + confidence)。**

缺少 landmarks 时，`alignCrop()` 返回全黑的 112×112 图像，导致所有 embedding 完全相同。

---

## 完成

- [x] Phase 0: Baseline frozen (commit a084fd2)
- [x] Phase 1: Face debug sheet generated (200 faces, 165 shots)
- [x] Phase 2: Coordinate/crop/alignment pipeline audited
- [x] P0 root cause identified and confirmed

---

## 关键数据

### Baseline
| Metric | Value |
|--------|-------|
| Total embeddings | 108,755 |
| Unique embeddings | 1,426 |
| Duplication rate | 98.7% |

### Face Debug Sheet (200 audited faces)
| Metric | Value |
|--------|-------|
| alignCrop(4-elem) → BLACK | 189/200 (94.5%) |
| alignCrop(15-elem) → OK | 200/200 (100%) |
| Landmarks re-detected | 200/200 (100%) |
| Stored vs correct cosine | mean=0.0103 |

### Delta (Stored vs Correct)
| Metric | Stored (broken) | Correct (with landmarks) |
|--------|----------------|--------------------------|
| Alignment | ALL-BLACK 112×112 | Proper face alignment |
| Embedding | Same vector for all faces | Unique per face |
| Cosine similarity (stored vs correct) | 0.0103 (random) | — |

---

## 根因分析

### Bug Location
**`backend/app/processors.py` line 134:**
```python
emb = self.face_engine.get_embedding(frame, fb)
```
`fb` is `fc["bbox"]` — only `[x, y, w, h]` (4 elements).

### What SFace `alignCrop()` expects
**`backend/app/face_engine.py` line 91:**
```python
aligned = self.sface.alignCrop(img_bgr, np.asarray(face_bbox, dtype=np.float32))
```
`alignCrop()` requires a 15-element array:
- [0:4] bbox (x, y, w, h)
- [4:6] left eye (x, y)
- [6:8] right eye (x, y)
- [8:10] nose tip (x, y)
- [10:12] left mouth corner (x, y)
- [12:14] right mouth corner (x, y)
- [14] confidence score

### What happens with 4 elements
When `alignCrop()` receives only 4 elements:
1. Landmarks are undefined (garbage/zeros)
2. Alignment fails silently
3. Returns a 112×112 ALL-BLACK image (mean pixel = 0.00)
4. SFace extracts embedding from black image → same vector every time

### Impact
- **ALL identity matching is meaningless** — every face is "identical"
- Clustering produces random groupings based on detection metadata, not face similarity
- Reference Mode always fails — stored embeddings don't represent actual faces
- The system has NEVER produced correct face embeddings

---

## BBox Format Audit

| Check | Result |
|-------|--------|
| BBox format | x, y, w, h ✅ |
| BBox coordinates | In original frame space ✅ |
| BBox boundary clamp | Handled in `FaceEngine.crop()` ✅ |
| BBox rescaling | Detection at original resolution (setInputSize) ✅ |

**BBox format is correct.** The bug is NOT in coordinate scaling or format — it's in what gets passed to `alignCrop()`.

---

## Alignment Audit

| Check | Result |
|-------|--------|
| `alignCrop()` usage | Present in `get_embedding()` ✅ |
| Landmarks passed to `alignCrop()` | ❌ **NOT PASSED** |
| `detect_faces()` returns landmarks | ✅ Returns 5 landmarks per face |
| Landmarks forwarded to embedding | ❌ **DROPPED in processors.py** |

**Landmarks exist in the pipeline but are discarded before reaching `alignCrop()`.**

---

## Evidence Files

```
reports/v02_accuracy_recovery/
    baseline_manifest.md              # Phase 0 frozen state
    face_debug_sheet/
        debug_data.json               # 200 faces with full audit data
        debug_sheet.html              # Visual contact sheet
        frame_0000.jpg ... 0199.jpg   # Frame overlays with bbox+landmarks
        raw_0000.jpg ... 0199.jpg     # Raw face crops
        aligned_0000.jpg ... 0199.jpg # Correctly aligned crops (15-elem)
```

---

## 发现的问题

### P0: alignCrop Missing Landmarks
- **Location**: `processors.py:134` → `face_engine.py:91`
- **Root cause**: `get_embedding()` receives only `[x,y,w,h]`, not full15-element detection
- **Impact**: ALL 108,755 embeddings are garbage (from black image)
- **Fix**: Pass full detection (bbox + landmarks + confidence) to `get_embedding()`
- **Severity**: BLOCKS ALL downstream identity work

### P1: No additional P0 found
- BBox coordinates: correct
- BBox format: correct (x,y,w,h)
- BBox scaling: correct (original frame coords)
- Boundary clamping: correct

---

## 下一步

**PHASE 3: Fix alignCrop bug**

修复 `get_embedding()` 接口，使其接收完整的15元素检测结果：

```python
# face_engine.py
def get_embedding(self, img_bgr: np.ndarray, detection: dict) -> Optional[np.ndarray]:
    """detection = {bbox, landmarks, score}"""
    bbox = detection['bbox']
    landmarks = detection['landmarks']  # 5 points
    score = detection['score']
    face_arr = np.asarray(bbox + [l for pt in landmarks for l in pt] + [score], dtype=np.float32)
    aligned = self.sface.alignCrop(img_bgr, face_arr)
    ...
```

修复后需要：
1. 重新处理所有视频（pipeline 从 faces_embedded 阶段重跑）
2. 重新聚类
3. 重新 benchmark
4. 验证 embedding 唯一性和跨镜头相似度
