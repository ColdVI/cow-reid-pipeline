from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .media import evidence_window


STATUS_COLORS = {
    "HUMAN_CONFIRMED": (55, 190, 70),
    "PREDICTED": (20, 205, 235),
    "AMBIGUOUS": (0, 135, 255),
    "UNKNOWN": (150, 150, 150),
    "TRACKLET_ERROR": (35, 35, 220),
    "COLLECTING": (220, 120, 35),
}

POSE_EDGES = (
    ("nose", "poll"), ("poll", "withers"),
    ("withers", "thoracic_spine"), ("thoracic_spine", "dorsal_apex"),
    ("dorsal_apex", "lumbar_spine"), ("lumbar_spine", "sacrum"),
    ("withers", "near_fore_carpus"), ("near_fore_carpus", "near_fore_hoof"),
    ("withers", "far_fore_carpus"), ("far_fore_carpus", "far_fore_hoof"),
    ("sacrum", "near_hind_hock"), ("near_hind_hock", "near_hind_hoof"),
    ("sacrum", "far_hind_hock"), ("far_hind_hock", "far_hind_hoof"),
)


def _interpolate_record(records: list[dict[str, Any]], timestamp_s: float) -> dict[str, Any] | None:
    if not records:
        return None
    ordered = sorted(records, key=lambda row: float(row["timestamp_s"]))
    if timestamp_s <= float(ordered[0]["timestamp_s"]):
        return ordered[0]
    if timestamp_s >= float(ordered[-1]["timestamp_s"]):
        return ordered[-1]
    for left, right in zip(ordered, ordered[1:]):
        a, b = float(left["timestamp_s"]), float(right["timestamp_s"])
        if a <= timestamp_s <= b:
            alpha = (timestamp_s - a) / max(1e-9, b - a)
            box = (1 - alpha) * np.asarray(left["bbox"], float) + alpha * np.asarray(right["bbox"], float)
            return {"bbox": box.tolist(), "confidence": (1 - alpha) * float(left.get("confidence", 0)) + alpha * float(right.get("confidence", 0))}
    return None


def annotation_sidecar(track: pd.Series | dict[str, Any], cow_id: str, status: str, similarity: float | None = None, candidates: tuple[str, str] | None = None) -> dict[str, Any]:
    row = dict(track)
    metadata_path = Path(str(row.get("metadata_path", "")))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {"records": []}
    label = cow_id or "UNKNOWN"
    if status == "PREDICTED" and similarity is not None:
        label = f"{label} · PREDICTED · similarity={similarity:.3f}"
    elif status == "AMBIGUOUS" and candidates:
        label = f"AMBIGUOUS · {candidates[0]} / {candidates[1]}"
    elif status == "UNKNOWN":
        label = "UNKNOWN · insufficient evidence"
    else:
        label = f"{label} · {status}"
    return {
        "tracklet_id": str(row.get("tracklet_id", "")), "video_id": str(row.get("video_id", "")),
        "start_s": float(row.get("start_s", metadata.get("start_s", 0))), "end_s": float(row.get("end_s", metadata.get("end_s", 0))),
        "label": label, "status": status, "color_bgr": STATUS_COLORS.get(status, STATUS_COLORS["UNKNOWN"]),
        "records": metadata.get("records", []),
    }


def _load_pose_frames(path: Path, min_visibility: float = 0.25) -> tuple[dict[int, dict[str, tuple[int, int]]], str]:
    if not path.is_file():
        return {}, ""
    frame = pd.read_csv(path)
    required = {"frame_idx", "keypoint_name", "x_px", "y_px", "visibility"}
    if frame.empty or not required.issubset(frame.columns):
        return {}, ""
    frame = frame.loc[pd.to_numeric(frame["visibility"], errors="coerce").ge(min_visibility)].copy()
    points: dict[int, dict[str, tuple[int, int]]] = {}
    for frame_idx, group in frame.groupby("frame_idx"):
        points[int(frame_idx)] = {
            str(row["keypoint_name"]): (int(round(float(row["x_px"]))), int(round(float(row["y_px"]))))
            for _, row in group.iterrows()
        }
    versions = frame["model_version"].dropna().astype(str).unique().tolist() if "model_version" in frame else []
    token = f"{path.stat().st_size}:{path.stat().st_mtime_ns}:{','.join(versions)}"
    return points, token


def render_debug_evidence(
    track: pd.Series | dict[str, Any], output_dir: str | Path, cow_id: str = "", status: str = "UNKNOWN",
    similarity: float | None = None, padding_s: float = 2.0,
) -> Path:
    row = dict(track)
    source = Path(str(row.get("source_video", "")))
    if not source.is_file():
        raise FileNotFoundError(f"Source video not found: {source}")
    annotation = annotation_sidecar(row, cow_id, status, similarity)
    target_dir = Path(output_dir); target_dir.mkdir(parents=True, exist_ok=True)
    pose_path = target_dir.parent / "pose" / annotation["tracklet_id"] / "pose_lameness" / "keypoints.csv"
    pose_frames, pose_token = _load_pose_frames(pose_path)
    cache_key = hashlib.sha1(json.dumps({"id": annotation["tracklet_id"], "label": annotation["label"], "start": annotation["start_s"], "end": annotation["end_s"], "pose": pose_token}, sort_keys=True).encode()).hexdigest()[:12]
    output = target_dir / f"{annotation['tracklet_id']}_{cache_key}.mp4"
    sidecar = output.with_suffix(".json")
    if output.exists() and sidecar.exists():
        return output

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    clip_start, clip_end = evidence_window(annotation["start_s"], annotation["end_s"], duration, padding_s)
    cap.set(cv2.CAP_PROP_POS_MSEC, clip_start * 1000)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release(); raise RuntimeError("Could not create debug MP4")
    trajectory: list[tuple[int, int]] = []
    frame_index = int(round(clip_start * fps))
    while frame_index / fps <= clip_end:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_s = frame_index / fps
        if annotation["start_s"] <= timestamp_s <= annotation["end_s"]:
            current = _interpolate_record(annotation["records"], timestamp_s)
            if current:
                x1, y1, x2, y2 = map(int, current["bbox"]); color = tuple(annotation["color_bgr"])
                center = ((x1 + x2) // 2, (y1 + y2) // 2); trajectory.append(center)
                if len(trajectory) > 30: trajectory.pop(0)
                if len(trajectory) > 1: cv2.polylines(frame, [np.asarray(trajectory, np.int32)], False, color, 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
                frame_points = pose_frames.get(frame_index, {})
                for left, right in POSE_EDGES:
                    if left in frame_points and right in frame_points:
                        cv2.line(frame, frame_points[left], frame_points[right], (255, 110, 30), 2, cv2.LINE_AA)
                for point in frame_points.values():
                    cv2.circle(frame, point, 4, (255, 245, 60), -1, cv2.LINE_AA)
                text = f"{annotation['label']} · det={float(current.get('confidence', 0)):.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
                top = max(0, y1 - th - 12); cv2.rectangle(frame, (x1, top), (min(width, x1 + tw + 10), y1), color, -1)
                cv2.putText(frame, text, (x1 + 5, max(th + 2, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (15, 15, 15), 2, cv2.LINE_AA)
        writer.write(frame); frame_index += 1
    cap.release(); writer.release()
    sidecar.write_text(json.dumps({**annotation, "clip_start_s": clip_start, "clip_end_s": clip_end, "pose_keypoints": str(pose_path) if pose_frames else None}, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
