# CIB V0.3 Baseline Freeze

- Frozen at: `2026-09-04T13:53:29.491758+00:00`
- Git HEAD: `d4487806dc36d546ef642b96d5a27b72becb7e5f` (`d448780)`
- Config SHA256: `9176d37c06367d38fe0274776d3a1dbac7215f946bb52103c22c41a605b225d2`
- face_engine_arcface SHA256: `9fa7f078ca35c222bb7a85af0c8b4ea8a2d71242cc05036572e6237d43f89836`
- DB: `/home/ponky_re6000/character-identity-board-data/cib.sqlite3` (652.2 MB)
- Snapshots: `reports/v03_recovery/baseline_snapshots/`

## RULE lock

- **No threshold lowering** for Unknown reduction
- **No full recluster** yet
- Thresholds frozen at ArcFace 0.40 / 0.50 / 0.30
- Manual confirmed/excluded/assignments treated as locked going forward

## Global observations

| metric | value |
|--------|------:|
| total | 156194 |
| active (excluded=0) | 89272 |
| excluded | 66922 |
| embedding_dim | [(512, 156194)] |
| identity_evidence_allowed | [(0, 64441), (1, 91753)] |

### Exclude reasons (top 20)

- `blur_below_threshold`: 15840
- `quality_below_0.6`: 12505
- `very_blurry`: 8847
- `false_positive_CL13`: 2823
- `false_positive_CL02`: 1067
- `false_positive_CL04`: 687
- `Auto: size=3330B area=3976px`: 250
- `VLM: the image is blurry and does not show any recognizable facial features`: 201
- `VLM: The image is a blurry mess and does not show any recognizable facial features.`: 151
- `non_frontal_yaw_gt_30`: 120
- `batch excluded by user`: 103
- `VLM: the image is a blurry mess and does not show any recognizable facial features`: 98
- `VLM: the image is a blurry mess and does not show any recognizable features of a face`: 92
- `low_quality_non_face`: 72
- `high_occlusion`: 26
- `Auto: size=2998B area=10323px`: 24
- `Auto: size=2996B area=10323px`: 24
- `Borderline: size=4959B area=10032px qs=0.7012272620371263`: 23
- `Borderline: size=4959B area=10032px qs=0.6930196831609858`: 20
- `Borderline: size=4959B area=10032px qs=0.6906579239156808`: 20

## Project film (id=15)

- Video: `c9bd3ce7_mirrored20260828 happy.mp4`
- 1920x1080 @ 24.0 fps, frames=157363, duration_s=6556.791678
- pipeline_stage=completed processing_status=completed
- shots=1078 tracklets=2074
- observations film: total=149928 active=85307 excluded=64621
- **Unknown tracklets: 1431**

### Characters

| id | name | code | status | tracklets | pending | auto | confirmed | avg_conf |
|---:|------|------|--------|----------:|--------:|-----:|----------:|---------:|
| 20 | Unknown | UNKNOWN | unknown | 1431 | 1 | 0 | 1644 | 0.907 |
| 27 | SMY | CL08 | manual | 205 | 13 | 170 | 23 | 0.943 |
| 30 | DOCTOR | CL04 | manual | 86 | 2 | 71 | 14 | 0.942 |
| 36 | lw | CL00 | manual | 78 | 0 | 78 | 0 | 0.950 |
| 25 | ZY | CL06 | manual | 55 | 4 | 35 | 16 | 0.935 |
| 31 | mbq | CL01 | manual | 36 | 0 | 33 | 3 | 0.953 |
| 32 | Test Actor | TEST | manual | 0 | 0 | 0 | 0 | 0.000 |
| 33 | SMY | SMY | manual | 0 | 0 | 0 | 0 | 0.000 |
| 34 | 院长 | 院长 | manual | 0 | 0 | 0 | 0 | 0.000 |
| 35 | LW | LW | manual | 0 | 0 | 0 | 0 | 0.000 |

### Assignment source × status

- `manual` / `confirmed`: 1629
- `clustering_v2` / `auto_assigned`: 387
- `automatic_cluster` / `confirmed`: 70
- `automatic_cluster` / `pending`: 20
- `clustering_v2` / `confirmed`: 1

### Embedding note

All 156,194 embeddings are **arcface_v1 unaligned** (bbox crop → resize 112, no 5-point similarity transform).
These will be superseded by `arcface_v2_aligned` after Phase 2–4; v1 retained.

### Artifacts

- `baseline_snapshots/config.py.d448780`
- `baseline_snapshots/face_engine_arcface.py.d448780`
- `baseline_snapshots/baseline_meta.json`
- `baseline_snapshots/film_assignments.json`
