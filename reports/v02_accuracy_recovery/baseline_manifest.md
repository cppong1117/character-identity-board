# BASELINE MANIFEST — V0.2 Accuracy Recovery

## Frozen State (2026-09-02)

### Git
- Commit: a084fd2
- Branch: master

### Runtime
- Python: 3.12.3
- OpenCV: 5.0.0 (DNN backend)
- NumPy: 2.5.2
- FastAPI: 0.141.1
- SQLAlchemy: 2.0.52
- HDBSCAN: installed
- scikit-learn: 1.9.0
- PySceneDetect: 0.7.1

### Models
- YuNet: face_detection_yunet.onnx (from opencv_zoo, MIT)
- SFace: face_recognition_sface.onnx (from opencv_zoo, Apache-2.0)
- Detection input: 320x320
- Embedding dim: 128 (float32, L2-normalized)

### Config Thresholds
| Parameter | Value |
|-----------|-------|
| identity_threshold | 0.85 |
| merge_threshold | 0.88 |
| unknown_threshold | 0.80 |
| track_iou_threshold | 0.30 |
| max_track_frames_lost | 12 |
| hdbscan_min_cluster_size | 15 |
| min_face_size_px | 90 |
| max_face_size_px | 600 |
| face_size_gate | either |
| blur_threshold | 28.0 |
| occlusion_penalty | 0.30 |
| reference_top_k | 5 |
| reference_margin | 0.10 |
| shot_threshold | 27.0 |
| shot_min_len_frames | 12 |
| yunet_score_threshold | 0.5 |
| yunet_nms_threshold | 0.3 |

### Database Stats (Project 15)
| Metric | Value |
|--------|-------|
| Shots | 1,078 |
| Tracklets | 2,074 |
| Total observations | 156,194 |
| Valid observations | 114,207 |
| Excluded observations | 41,987 |
| Unique embeddings | 1,426 (!) |
| Total embeddings | 108,755 |
| Embedding duplication rate | 98.7% |
| Characters | 15 |
| Identity assignments | pending |

### 🔴 P0 BUG: alignCrop receives 4-element bbox (x,y,w,h) instead of 15-element detection (bbox + 5 landmarks + confidence)
- Result: alignCrop returns ALL-BLACK 112x112 image
- Consequence: ALL embeddings are identical (from black image)
- This is the ROOT CAUSE of clustering failure
