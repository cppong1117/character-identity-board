# FACE_PIPELINE_V02.md

## Character Identity Board — Face Pipeline V0.2

### Pipeline Architecture

```
Video Input
    ↓
Shot Detection (PySceneDetect)
    ↓
Frame Extraction
    ↓
Face Detection (YuNet)
    ↓
Landmark Extraction (5-point)
    ↓
Face Alignment (SFace.alignCrop with 15-elem detection)
    ↓
Quality Scoring (V2: detector + blur + occlusion + pose + size + landmark)
    ↓
Dual-Pool Assignment
    ├── Observation Pool (quality >= 0.60): UI browsing, timeline
    └── Identity Evidence Pool (quality >= 0.85): embedding, matching, clustering
    ↓
Embedding Extraction (SFace, 128-d)
    ↓
Tracking (IoU-based)
    ↓
Tracklet Prototype (top-5 quality-weighted centroid)
    ↓
Clustering (HDBSCAN on prototypes)
    ↓
Character Assignment
    ↓
Review Queue (human verification)
    ↓
Identity Database
```

### Key Components

#### 1. Face Detection (YuNet)

- **Model**: YuNet (ONNX, MIT license)
- **Input**: 320×320 resized frame
- **Output**: bbox + 5 landmarks + confidence
- **Threshold**: 0.60 (Observation Pool), 0.85 (Identity Evidence Pool)

#### 2. Face Alignment (SFace.alignCrop)

- **Input**: Full15-element detection (bbox + landmarks + confidence)
- **Output**: Aligned 112×112 face crop
- **Critical**: Must receive landmarks, NOT just bbox

#### 3. Quality Scoring V2

| Component | Weight | Range |
|-----------|--------|-------|
| Detector confidence | 0.25 | 0-1 |
| Blur (Laplacian variance) | 0.15 | 0-1 (higher = sharper) |
| Occlusion | 0.10 | 0-1 (lower = less occluded) |
| Pose (yaw/pitch) | 0.20 | 0-1 (lower = more frontal) |
| Face size | 0.15 | 0-1 (larger = better) |
| Landmark geometry | 0.15 | 0-1 |

**Rejection reasons**:
- LOW_DETECTOR_CONFIDENCE (quality < 0.70)
- BLUR_TOO_HIGH (blur < 30)
- OCCLUDED (occlusion > 0.3)
- POSE_TOO_EXTREME (|yaw| > 30°)
- FACE_TOO_SMALL (area < 500 px²)

#### 4. Embedding Extraction (SFace)

- **Model**: SFace (ONNX, Apache-2.0 license)
- **Output**: 128-d L2-normalized vector
- **Cross-shot similarity**: 0.12-0.16 (fundamentally low)

#### 5. Tracklet Prototype

- **Method**: Top-5 quality-weighted centroid
- **Selection**: Highest quality_score_v2 embeddings
- **Aggregation**: Weighted average, re-normalized

#### 6. Clustering (HDBSCAN)

- **Algorithm**: HDBSCAN with precomputed distance matrix
- **Distance**: 1 - cosine_similarity
- **Parameters**: min_cluster_size=15, min_samples=5
- **Metric**: Precomputed distance matrix

### Database Schema Changes (V0.2)

Added columns to `face_observations`:
- `quality_score_v2` (REAL): Comprehensive quality score
- `identity_evidence_allowed` (INTEGER): 0/1 flag
- `rejection_reason` (TEXT): Pipe-separated rejection reasons

### Configuration

| Parameter | Value | Location |
|-----------|-------|----------|
| YuNet confidence | 0.60 | `config.py` |
| Identity threshold | 0.85 | `config.py` |
| HDBSCAN min_cluster_size | 15 | `config.py` |
| HDBSCAN min_samples | 5 | `clustering_v2.py` |
| Quality gate threshold | 0.85 | `quality_gate_v2.py` |
| Blur threshold | 30 | `quality_gate_v2.py` |
| Occlusion threshold | 0.3 | `quality_gate_v2.py` |
| Pose threshold | 30° | `quality_gate_v2.py` |
