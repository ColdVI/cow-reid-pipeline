from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .config import resolve_model_path
from .tracking import MotionDetector
from .utils import (
    bbox_iou,
    bool_mask,
    crop_box,
    decode_jpeg,
    encode_jpeg,
    expand_box,
    make_contact_sheet,
    save_json,
    sharpness_score,
    torso_box,
)

# Bump whenever the crop/masking logic in this module changes (the animal-
# silhouette masking in _quality_score/_process_video, torso/full-box geometry,
# etc.) so embeddings built from old crops never silently compare as
# current-version against embeddings built after a crop/mask change.
EXTRACTION_PIPELINE_VERSION = "extract_v1_masked_torso"

# Below this min-pairwise color-histogram similarity between a tracklet's own
# candidate frames, something looks visually inconsistent within one
# continuous track -- most often a tracker id-switch silently merging two
# different animals, occasionally an extreme viewpoint change. This only
# raises a quality_flags entry for human review; per the spec, appearance
# inconsistency is a suspicious-track marker, never grounds by itself for an
# automatic identity/validity decision.
APPEARANCE_CONSISTENCY_FLAG_THRESHOLD = 0.65


@dataclass
class Candidate:
    score: float
    frame_idx: int
    timestamp_s: float
    confidence: float
    overlap: float
    bbox: list[float]
    torso_jpeg: bytes
    embedding_jpeg: bytes
    full_jpeg: bytes


@dataclass
class TrackState:
    records: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)


def _resolve_device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _reset_ultralytics_tracker(model: Any) -> None:
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", []) or []:
        reset = getattr(tracker, "reset", None)
        if callable(reset):
            reset()


def _detections_yolo(model: Any, frame: np.ndarray, config: dict[str, Any], device: str):
    result = model.track(
        frame,
        persist=True,
        classes=[int(config.get("cow_class", 19))],
        conf=float(config.get("confidence", 0.16)),
        iou=float(config.get("nms_iou", 0.10)),
        imgsz=int(config.get("imgsz", 768)),
        tracker="bytetrack.yaml",
        verbose=False,
        device=device,
    )[0]
    if result.boxes.id is None:
        return [], [], [], []
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float).tolist()
    ids = result.boxes.id.detach().cpu().numpy().astype(int).tolist()
    confidence = result.boxes.conf.detach().cpu().numpy().astype(float).tolist()
    polygons: list[np.ndarray | None] = [None] * len(boxes)
    if result.masks is not None and len(result.masks.xy) == len(boxes):
        polygons = [np.asarray(polygon, dtype=np.float32) if len(polygon) >= 3 else None for polygon in result.masks.xy]
    return boxes, ids, confidence, polygons


def _quality_score(
    frame: np.ndarray,
    box: list[float],
    confidence: float,
    overlap: float,
    gate: tuple[float, float],
    orientation: str,
    mask_polygon: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    full = crop_box(frame, expand_box(box, width, height))
    torso_bounds = torso_box(box, width, height, orientation)
    torso = crop_box(frame, torso_bounds)
    embedding_crop = torso.copy()
    if mask_polygon is not None and embedding_crop.size:
        tx1, ty1, tx2, ty2 = torso_bounds
        local_polygon = np.rint(mask_polygon - np.asarray([tx1, ty1], dtype=np.float32)).astype(np.int32)
        matte = np.zeros((ty2 - ty1, tx2 - tx1), dtype=np.uint8)
        cv2.fillPoly(matte, [local_polygon], 255)
        matte = cv2.morphologyEx(
            matte,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        coverage = float(np.mean(matte > 0))
        if coverage >= 0.18:
            masked = np.full_like(embedding_crop, 127)
            masked[matte > 0] = embedding_crop[matte > 0]
            embedding_crop = masked
    cx = (x1 + x2) * 0.5 / width
    gate_mid = (gate[0] + gate[1]) * 0.5
    gate_radius = max(0.05, (gate[1] - gate[0]) * 0.5)
    center_score = float(np.clip(1.0 - abs(cx - gate_mid) / gate_radius, 0.0, 1.0))
    area_score = float(np.clip((bw * bh) / (0.12 * width * height), 0.0, 1.0))
    sharp_score = sharpness_score(torso)
    edge_margin = min(x1 / width, (width - x2) / width, y1 / height, (height - y2) / height)
    edge_score = float(np.clip(edge_margin / 0.03, 0.0, 1.0))
    score = (
        0.30 * float(np.clip(confidence, 0.0, 1.0))
        + 0.22 * center_score
        + 0.18 * area_score
        + 0.17 * sharp_score
        + 0.13 * edge_score
        - 0.35 * float(np.clip(overlap, 0.0, 1.0))
    )
    return float(np.clip(score, 0.0, 1.0)), torso, embedding_crop, full


def _classify_direction(net_dx: float, min_span: float, orientation: str) -> tuple[bool, bool, bool]:
    """Returns (direction_ok, wrong_direction, uncertain_direction).

    A track's net displacement can miss the margin for two very different
    reasons: it clearly moved the wrong way (safe to hard reject), or it
    barely moved net either way -- e.g. paused/jostled near the gate --
    which is not evidence of the wrong direction, just inconclusive.
    Callers should route the latter to human review rather than silently
    discarding it.
    """
    margin = min_span * 0.35
    if orientation == "left_to_right":
        direction_ok = net_dx >= margin
        wrong_direction = net_dx <= -margin
    else:
        direction_ok = net_dx <= -margin
        wrong_direction = net_dx >= margin
    uncertain_direction = not direction_ok and not wrong_direction
    return direction_ok, wrong_direction, uncertain_direction


def _intra_track_appearance_consistency(images: list[np.ndarray]) -> float | None:
    if len(images) < 2:
        return None
    features = []
    for image in images:
        small = cv2.resize(image, (48, 48)).astype(np.float32) / 255.0
        channels = cv2.split(small) if small.ndim == 3 else [small]
        histogram = np.concatenate(
            [np.histogram(channel, bins=12, range=(0.0, 1.0))[0].astype(np.float32) for channel in channels]
        )
        norm = float(np.linalg.norm(histogram))
        features.append(histogram / norm if norm > 0 else histogram)
    stacked = np.stack(features)
    similarity = stacked @ stacked.T
    np.fill_diagonal(similarity, 1.0)
    return float(similarity.min())


def _add_candidate(state: TrackState, candidate: Candidate, limit: int) -> None:
    state.candidates.append(candidate)
    state.candidates.sort(key=lambda item: item.score, reverse=True)
    if len(state.candidates) > limit:
        state.candidates.pop()


def _process_video(
    row: pd.Series,
    output_dir: Path,
    config: dict[str, Any],
    model: Any | None,
    device: str,
    max_seconds: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = Path(str(row["path"]))
    video_id = str(row["video_id"])
    session_id = str(row.get("session_id") or video_id)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    sample_fps = float(config.get("sample_fps", 5.0))
    sample_step = max(1, int(round(fps / sample_fps)))
    roi_norm = list(map(float, config.get("roi", [0.0, 0.22, 1.0, 0.80])))
    rx1, ry1, rx2, ry2 = (
        int(roi_norm[0] * width),
        int(roi_norm[1] * height),
        int(roi_norm[2] * width),
        int(roi_norm[3] * height),
    )
    roi_width, roi_height = max(1, rx2 - rx1), max(1, ry2 - ry1)
    gate = tuple(map(float, config.get("gate", [0.18, 0.82])))
    backend = str(config.get("backend", "yolo"))
    orientation = str(config.get("orientation", "left_to_right"))
    motion = MotionDetector(roi_width, roi_height, config.get("motion", {})) if backend == "motion" else None
    if model is not None:
        _reset_ultralytics_tracker(model)

    tracks: dict[int, TrackState] = {}
    frame_idx = 0
    sampled_frames = 0
    max_frame = int(max_seconds * fps) if max_seconds is not None else None
    while True:
        ok, frame = cap.read()
        if not ok or (max_frame is not None and frame_idx >= max_frame):
            break
        if frame_idx % sample_step:
            frame_idx += 1
            continue
        sampled_frames += 1
        roi_frame = frame[ry1:ry2, rx1:rx2]
        if backend == "yolo":
            boxes, ids, confidences, polygons = _detections_yolo(model, roi_frame, config, device)
        else:
            boxes, ids, confidences = motion.detect_and_track(roi_frame)
            polygons = [None] * len(boxes)
        absolute_boxes: list[list[float]] = []
        for x1, y1, x2, y2 in boxes:
            absolute_boxes.append([x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1])
        absolute_polygons = [
            polygon + np.asarray([rx1, ry1], dtype=np.float32) if polygon is not None else None
            for polygon in polygons
        ]

        for index, (track_id, box, confidence, polygon) in enumerate(
            zip(ids, absolute_boxes, confidences, absolute_polygons)
        ):
            x1, y1, x2, y2 = box
            area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, width * height)
            if area_ratio < float(config.get("min_box_area", 0.010)):
                continue
            other_overlap = max(
                [bbox_iou(box, other) for j, other in enumerate(absolute_boxes) if j != index] or [0.0]
            )
            state = tracks.setdefault(int(track_id), TrackState())
            cx = (x1 + x2) * 0.5 / width
            timestamp_s = frame_idx / fps
            score, torso, embedding_crop, full = _quality_score(
                frame,
                box,
                float(confidence),
                other_overlap,
                gate,
                orientation,
                polygon,
            )
            state.records.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_s": timestamp_s,
                    "bbox": [float(v) for v in box],
                    "confidence": float(confidence),
                    "quality": score,
                    "center_x": cx,
                    "overlap": other_overlap,
                }
            )
            if gate[0] <= cx <= gate[1] and torso.size and full.size:
                candidate = Candidate(
                    score=score,
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    confidence=float(confidence),
                    overlap=float(other_overlap),
                    bbox=[float(v) for v in box],
                    torso_jpeg=encode_jpeg(torso, int(config.get("jpeg_quality", 92))),
                    embedding_jpeg=encode_jpeg(embedding_crop, int(config.get("jpeg_quality", 92))),
                    full_jpeg=encode_jpeg(full, int(config.get("jpeg_quality", 92))),
                )
                _add_candidate(state, candidate, int(config.get("best_frames", 12)))
        frame_idx += 1
    cap.release()

    tracklet_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    min_samples = int(config.get("min_track_samples", 8))
    min_span = float(config.get("min_track_span", 0.16))
    min_best = int(config.get("min_best_frames", 3))
    keep_rejected = bool(config.get("keep_rejected", False))
    for track_id, state in sorted(tracks.items()):
        records = sorted(state.records, key=lambda item: item["frame_idx"])
        if not records:
            continue
        centers = np.asarray([record["center_x"] for record in records], dtype=np.float32)
        span = float(centers.max() - centers.min())
        net_dx = float(centers[-1] - centers[0])
        direction_ok, wrong_direction, uncertain_direction = _classify_direction(net_dx, min_span, orientation)
        valid = (
            len(records) >= min_samples
            and span >= min_span
            and direction_ok
            and len(state.candidates) >= min_best
        )
        rejection_reasons: list[str] = []
        if len(records) < min_samples:
            rejection_reasons.append("too_short")
        if span < min_span:
            rejection_reasons.append("small_motion_span")
        if wrong_direction:
            rejection_reasons.append("wrong_direction")
        elif uncertain_direction:
            rejection_reasons.append("uncertain_direction")
        if len(state.candidates) < min_best:
            rejection_reasons.append("not_enough_clean_frames")

        tracklet_id = f"{video_id}_t{track_id:04d}"
        track_dir = output_dir / "tracklets" / video_id / tracklet_id
        contact_path = ""
        metadata_path = ""
        appearance_consistency: float | None = None
        quality_flags: list[str] = []
        if valid or keep_rejected or (uncertain_direction and state.candidates):
            torso_dir = track_dir / "torso"
            embedding_dir = track_dir / "masked_torso"
            full_dir = track_dir / "full"
            torso_dir.mkdir(parents=True, exist_ok=True)
            embedding_dir.mkdir(parents=True, exist_ok=True)
            full_dir.mkdir(parents=True, exist_ok=True)
            contact_images: list[np.ndarray] = []
            for rank, candidate in enumerate(state.candidates, start=1):
                filename = f"rank_{rank:02d}_f{candidate.frame_idx:07d}_q{candidate.score:.3f}.jpg"
                torso_path = torso_dir / filename
                embedding_path = embedding_dir / filename
                full_path = full_dir / filename
                torso_path.write_bytes(candidate.torso_jpeg)
                embedding_path.write_bytes(candidate.embedding_jpeg)
                full_path.write_bytes(candidate.full_jpeg)
                contact_images.append(decode_jpeg(candidate.torso_jpeg))
                frame_rows.append(
                    {
                        "tracklet_id": tracklet_id,
                        "video_id": video_id,
                        "session_id": session_id,
                        "rank": rank,
                        "frame_idx": candidate.frame_idx,
                        "timestamp_s": candidate.timestamp_s,
                        "quality": candidate.score,
                        "confidence": candidate.confidence,
                        "overlap": candidate.overlap,
                        "bbox": ",".join(f"{value:.2f}" for value in candidate.bbox),
                        "torso_path": str(torso_path.resolve()),
                        "embedding_path": str(embedding_path.resolve()),
                        "full_path": str(full_path.resolve()),
                    }
                )
            appearance_consistency = _intra_track_appearance_consistency(contact_images)
            if appearance_consistency is not None and appearance_consistency < APPEARANCE_CONSISTENCY_FLAG_THRESHOLD:
                quality_flags.append("large_appearance_change")
            contact = make_contact_sheet(contact_images, columns=4)
            contact_file = track_dir / "contact.jpg"
            cv2.imwrite(str(contact_file), contact)
            contact_path = str(contact_file.resolve())
            metadata_file = track_dir / "metadata.json"
            save_json(
                metadata_file,
                {
                    "tracklet_id": tracklet_id,
                    "video_id": video_id,
                    "session_id": session_id,
                    "source_video": str(path.resolve()),
                    "valid": valid,
                    "rejection_reasons": rejection_reasons,
                    "start_s": records[0]["timestamp_s"],
                    "end_s": records[-1]["timestamp_s"],
                    "fps": fps,
                    "sample_fps": sample_fps,
                    "records": records,
                },
            )
            metadata_path = str(metadata_file.resolve())

        tracklet_rows.append(
            {
                "tracklet_id": tracklet_id,
                "video_id": video_id,
                "session_id": session_id,
                "session_type": row.get("session_type", "unknown"),
                "source_video": str(path.resolve()),
                "source_filename": path.name,
                "valid": valid,
                "rejection_reasons": ";".join(rejection_reasons),
                "n_observations": len(records),
                "n_best_frames": len(state.candidates),
                "start_s": records[0]["timestamp_s"],
                "end_s": records[-1]["timestamp_s"],
                "duration_s": records[-1]["timestamp_s"] - records[0]["timestamp_s"],
                "x_start": float(centers[0]),
                "x_end": float(centers[-1]),
                "x_span": span,
                "net_dx": net_dx,
                "mean_confidence": float(np.mean([record["confidence"] for record in records])),
                "mean_quality": float(np.mean([candidate.score for candidate in state.candidates]))
                if state.candidates
                else 0.0,
                "contact_path": contact_path,
                "metadata_path": metadata_path,
                "crop_version": EXTRACTION_PIPELINE_VERSION,
                "appearance_consistency": appearance_consistency,
                "quality_flags": ";".join(quality_flags),
            }
        )
    save_json(
        output_dir / "video_summaries" / f"{video_id}.json",
        {
            "video_id": video_id,
            "source": str(path.resolve()),
            "sampled_frames": sampled_frames,
            "raw_tracks": len(tracks),
            "valid_tracks": sum(bool(row["valid"]) for row in tracklet_rows),
        },
    )
    return tracklet_rows, frame_rows


def extract_tracklets(
    manifest: str | Path | pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
    max_videos: int | None = None,
    max_seconds: float | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_frame = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
    if "enabled" in manifest_frame:
        manifest_frame = manifest_frame.loc[bool_mask(manifest_frame["enabled"], default=True)]
    if "is_duplicate" in manifest_frame:
        manifest_frame = manifest_frame.loc[~bool_mask(manifest_frame["is_duplicate"], default=False)]
    if max_videos is not None:
        manifest_frame = manifest_frame.head(max_videos)
    output_dir = Path(output_dir).resolve()
    tracklets_csv = output_dir / "tracklets.csv"
    if tracklets_csv.exists() and not overwrite:
        raise FileExistsError(f"Run already exists: {tracklets_csv}. Choose a new output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Persist the exact manifest rows this run was built from, so downstream
    # steps (training's overlap/date-aware split in particular) have a
    # reliable, in-run record of overlap groups and recording dates without
    # having to guess which manifest file produced this run.
    manifest_frame.to_csv(output_dir / "video_manifest.csv", index=False)
    extract_cfg = config["extract"]
    backend = str(extract_cfg.get("backend", "yolo"))
    model = None
    device = _resolve_device(str(extract_cfg.get("device", "auto")))
    if backend == "yolo":
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is required for the yolo backend. Install the deep extra or use backend=motion.") from exc
        model_path = resolve_model_path(str(extract_cfg.get("model", "yolo11n-seg.pt")), config)
        model = YOLO(model_path)

    all_tracklets: list[dict[str, Any]] = []
    all_frames: list[dict[str, Any]] = []
    for _, row in manifest_frame.iterrows():
        tracklet_rows, frame_rows = _process_video(
            row, output_dir, extract_cfg, model=model, device=device, max_seconds=max_seconds
        )
        all_tracklets.extend(tracklet_rows)
        all_frames.extend(frame_rows)
    tracklet_frame = pd.DataFrame(all_tracklets)
    frame_frame = pd.DataFrame(all_frames)
    tracklet_frame.to_csv(tracklets_csv, index=False)
    frame_frame.to_csv(output_dir / "frames.csv", index=False)
    save_json(
        output_dir / "extract_summary.json",
        {
            "videos": int(len(manifest_frame)),
            "tracklets_total": int(len(tracklet_frame)),
            "tracklets_valid": int(tracklet_frame["valid"].sum()) if not tracklet_frame.empty else 0,
            "selected_frames": int(len(frame_frame)),
            "backend": backend,
            "device": device,
        },
    )
    return tracklet_frame, frame_frame
