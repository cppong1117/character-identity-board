"""Per-shot person/face processing: detection, tracking, crop generation.

Produces for each shot:
  - tracklets (person runs)
  - per-tracklet best face/portrait/body crops
  - per-tracklet best embedding
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import settings
from .face_engine import FaceEngine
from .quality import compute_quality
from .tracking import ShotTracker

log = logging.getLogger(__name__)

# default body/person detect: reuse face bbox expanded to head-shoulders + body
# (V0.1 uses face-first person association; no separate person model needed.)


def _expand_bbox(bbox, frame_w, frame_h, scale=2.6):
    """Expand a face bbox to a head-shoulders / upper-body box."""
    x, y, w, h = [float(v) for v in bbox]
    cx, cy = x + w / 2, y + h / 2
    nw, nh = w * scale, h * scale
    x0 = max(0, cx - nw / 2)
    y0 = max(0, cy - nh * 0.45)   # bias upward to include top of head
    x0 = min(x0, frame_w - 1)
    y0 = min(y0, frame_h - 1)
    x1 = min(frame_w, x0 + nw)
    y1 = min(frame_h, y0 + nh)
    return [x0, y0, x1 - x0, y1 - y0]


class ShotProcessor:
    def __init__(self, face_engine: FaceEngine, work_dir: Path,
                 device=None):
        self.face_engine = face_engine
        self.work_dir = work_dir
        self.device = device

    def process(self, video_path: str, shot: dict, shot_idx: int,
                frame_indices: list[int], fps: float | None = None,
                draw_overlays: bool = True) -> dict:
        """Process one shot by decoding its representative sampled frames.

        Returns dict with:
          representative_frame: path
          tracks: list of track dicts: {track_number, start_frame, end_frame,
                 bbox, observations: [...]}
        Each observation has frame_number, bbox, crops paths, quality, embedding.
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise RuntimeError(f"OpenCV failed to open video {video_path}")

        shot_dir = self.work_dir / f"shot_{shot_idx:03d}"
        shot_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = shot_dir / "frames"
        crops_dir = shot_dir / "crops"
        overlay_dir = shot_dir / "overlays"
        frames_dir.mkdir(exist_ok=True)
        crops_dir.mkdir(exist_ok=True)
        overlay_dir.mkdir(exist_ok=True)

        tracker = ShotTracker(
            iou_threshold=settings.track_iou_threshold,
            max_lost=settings.max_track_frames_lost,
            face_engine=self.face_engine,
        )
        f0, f1 = shot["start_frame"], shot["end_frame"]

        # We'll decode frames sequentially but only process selected indices;
        # if too sparse we decode contiguous for tracking stability.
        to_process = set(frame_indices)

        # Decode contiguous frames from f0..f1 for robust tracking
        # (limit to a budget: process every Nth frame if shot is huge)
        step = 1
        if (f1 - f0 + 1) > 600:
            step = 2
        process_seq = [i for i in range(f0, f1 + 1, step)]

        # ensure we hit the required sample frames
        extra = sorted(to_process - set(process_seq))
        process_frames = sorted(set(process_seq) | set(extra))

        video.set(cv2.CAP_PROP_POS_FRAMES, f0)
        tracks = {}
        frame_idx = f0 - 1
        last_saved_overlay = -1
        track_num_counter = 0

        # capture repr frame (middle sample)
        repr_target = frame_indices[len(frame_indices) // 2] if frame_indices else f0
        repr_path = None
        obs_store = {}   # (track_key) -> list of obs dicts

        processed = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx > f1:
                break
            if frame_idx < process_frames[0]:
                continue
            if frame_idx not in set(process_frames):
                # still advance; skip heavy work
                continue

            H, W = frame.shape[:2]
            # 1) face detection
            faces = self.face_engine.detect_faces(frame)
            dets = []
            for fc in faces:
                fb = fc["bbox"]
                # Track faces down to ~50px but only embed reliably-sized ones.
                if fb[2] < 50 or fb[3] < 50:
                    continue
                if settings.face_size_gate == "both":
                    embed_for = (fb[2] >= settings.min_face_size_px) and (fb[3] >= settings.min_face_size_px)
                else:
                    embed_for = (fb[2] >= settings.min_face_size_px) or (fb[3] >= settings.min_face_size_px)
                # A representative embedding requires a genuine face crop.
                emb = self.face_engine.get_embedding(frame, fc) if embed_for else None
                dets.append({
                    "bbox_abs": _expand_bbox(fb, W, H, scale=1.35),
                    "face_bbox_abs": fb,
                    "face_score": fc["score"],
                    "face_embedding": emb,
                })

            active = tracker.update(dets, frame_idx)

            # persist observations per track
            for d in dets:
                fb = d["face_bbox_abs"]
                # match to track that owns this face bbox
                owner = self._find_track_for_face(tracker.tracks, fb)
                if owner is None:
                    continue
                key = owner.track_id
                ts = int(frame_idx / (fps or 25.0) * 1000)
                obs = self._make_observation(
                    frame, fb, d, frame_idx, ts, W, H,
                    fps=fps, crops_dir=crops_dir,
                )
                if obs is not None:
                    obs_store.setdefault(key, []).append(obs)
                    if frame_idx == repr_target:
                        repr_path = self._save_frame(frame, frames_dir, repr_target)

            # draw overlay for a few frames (proof)
            if draw_overlays and frame_idx % 10 == 0:
                ov = frame.copy()
                for t in active:
                    if t.best_face_bbox:
                        x, y, w, h = [int(v) for v in t.best_face_bbox]
                        cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 200, 0), 2)
                    pb = [int(v) for v in t.bbox]
                    cv2.rectangle(ov, (pb[0], pb[1]), (pb[0] + pb[2], pb[1] + pb[3]), (255, 0, 0), 2)
                    cv2.putText(ov, f"T{t.track_id} f{frame_idx}",
                                (pb[0], max(10, pb[1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                ovp = overlay_dir / f"ov_{frame_idx:06d}.jpg"
                cv2.imwrite(str(ovp), ov, [cv2.IMWRITE_JPEG_QUALITY, 85])

            processed += 1

        video.release()

        # Build track summaries with best crops
        track_out = []
        for t in tracker.tracks:
            obs = obs_store.get(t.track_id, [])
            if not obs:
                continue
            # pick best observation by quality
            best = max(obs, key=lambda o: o["quality_score"])
            track_out.append({
                "track_number": t.track_id,
                "start_frame": t.start_frame,
                "end_frame": t.last_frame,
                "person_bbox": t.bbox,
                "best_face_bbox": t.best_face_bbox,
                "best_face_score": t.best_face_score,
                "best_observation": best,
                "observations": obs,
            })

        if repr_path is None and frames_dir.exists():
            # ensure repr exists
            mid = frame_indices[len(frame_indices) // 2] if frame_indices else f0
            rp = frames_dir / f"frame_{mid:06d}.jpg"
            if rp.exists():
                repr_path = str(rp)

        return {
            "representative_frame": repr_path or str(shot_dir),
            "tracklet_count": len(track_out),
            "tracklets": track_out,
        }

    def _find_track_for_face(self, tracks, face_bbox):
        best_t, best_iou = None, 0.0
        for t in tracks:
            pb = t.bbox
            iou = self._face_in_box(face_bbox, pb)
            if iou > best_iou:
                best_iou = iou
                best_t = t
        return best_t if best_iou > 0.05 else None

    @staticmethod
    def _face_in_box(face, box):
        fx, fy, fw, fh = face
        bx, by, bw, bh = box
        ix = max(0, min(fx + fw, bx + bw) - max(fx, bx))
        iy = max(0, min(fy + fh, by + bh) - max(fy, by))
        inter = ix * iy
        if inter <= 0:
            return 0.0
        area = fw * fh
        return inter / area if area > 0 else 0.0

    def _make_observation(self, frame, face_bbox, det, frame_idx, ts, W, H,
                          fps, crops_dir):
        try:
            fw = crops_dir / f"face_{frame_idx:06d}.jpg"
            pw = crops_dir / f"portrait_{frame_idx:06d}.jpg"
            bw = crops_dir / f"body_{frame_idx:06d}.jpg"
            # face crop (with margin)
            face_c = FaceEngine.crop(frame, face_bbox, margin=0.20)
            # portrait = expanded by 2.4x
            port_bb = _expand_bbox(face_bbox, W, H, scale=2.4)
            port_c = FaceEngine.crop(frame, port_bb)
            # body = expand by 5x
            body_bb = _expand_bbox(face_bbox, W, H, scale=5.0)
            body_c = FaceEngine.crop(frame, body_bb)
            cv2.imwrite(str(fw), face_c)
            cv2.imwrite(str(pw), port_c)
            cv2.imwrite(str(bw), body_c)

            q = compute_quality(frame, face_bbox,
                                detect_confidence=det["face_score"],
                                frame_w=W, frame_h=H)
            emb = det.get("face_embedding")
            return {
                "frame_number": frame_idx,
                "timestamp_ms": ts,
                "face_crop_path": str(fw),
                "portrait_crop_path": str(pw),
                "body_crop_path": str(bw),
                "face_bbox": list(face_bbox),
                "quality_score": q["quality_score"],
                "blur_score": q["blur_score"],
                "occlusion_score": q["occlusion_score"],
                "embedding": emb,
            }
        except Exception as e:
            log.debug("obs fail f%d: %s", frame_idx, e)
            return None

    def _save_frame(self, frame, dirp, idx):
        p = dirp / f"frame_{idx:06d}.jpg"
        if not p.exists():
            cv2.imwrite(str(p), frame)
        return str(p)
