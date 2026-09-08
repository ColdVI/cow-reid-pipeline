from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from fnmatch import fnmatch
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .overlap import assign_overlap_groups, perceptual_fingerprint
from .timestamping import TimestampSample, sample_offsets, validate_timestamp_samples
from .utils import bool_mask, sha256_file, video_id_from_hash


TIMESTAMP_RE = re.compile(
    r"(?P<month>\d{1,2})-(?P<day>\d{1,2})-(?P<year>\d{4}).*?"
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2})"
)
WEEKDAY_RE = re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b", re.IGNORECASE)
WEEKDAY_NUMBER = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
WHATSAPP_TIMESTAMP_RE = re.compile(
    r"(?P<year>20\d{2})[-_](?P<month>\d{1,2})[-_](?P<day>\d{1,2})"
    r".*?(?:\bat\b|\bsaat\b)[\s_-]*"
    r"(?P<hour>\d{1,2})[.\-_:](?P<minute>\d{2})[.\-_:](?P<second>\d{2})",
    re.IGNORECASE,
)


def classify_session(hour: int | None) -> str:
    if hour is None:
        return "unknown"
    if 5 <= hour < 10:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    if hour >= 20 or hour < 3:
        return "night"
    return "other"


def _parse_ocr_timestamp(text: str) -> datetime | None:
    # Camera overlays are numeric, while Tesseract occasionally emits O/I/l.
    weekday_match = WEEKDAY_RE.search(text)
    normalized = (
        text.replace("—", "-")
        .replace("–", "-")
        .replace("_", "-")
        .replace("O", "0")
        .replace("o", "0")
        .replace("I", "1")
        .replace("l", "1")
        .replace("|", "1")
    )
    match = TIMESTAMP_RE.search(normalized)
    if not match:
        return None
    values = {key: int(value) for key, value in match.groupdict().items()}
    if not 2000 <= values["year"] <= 2100:
        return None
    try:
        parsed = datetime(
            values["year"],
            values["month"],
            values["day"],
            values["hour"],
            values["minute"],
            values["second"],
        )
    except ValueError:
        return None
    if weekday_match:
        expected_weekday = WEEKDAY_NUMBER[weekday_match.group(1).lower()]
        if parsed.weekday() != expected_weekday:
            return None
    return parsed


def parse_filename_timestamp(filename: str) -> datetime | None:
    """Parse common WhatsApp export names without pretending the value came from CCTV OCR."""
    match = WHATSAPP_TIMESTAMP_RE.search(Path(filename).stem)
    if not match:
        return None
    values = {key: int(value) for key, value in match.groupdict().items()}
    try:
        return datetime(
            values["year"],
            values["month"],
            values["day"],
            values["hour"],
            values["minute"],
            values["second"],
        )
    except ValueError:
        return None


def _ocr_timestamp(frame: np.ndarray) -> tuple[str | None, str | None]:
    if frame is None or not shutil.which("tesseract"):
        return None, None
    h, w = frame.shape[:2]
    crops = [
        frame[int(0.015 * h) : max(45, int(0.115 * h)), int(0.005 * w) : max(360, int(0.43 * w))],
        frame[0 : max(45, int(0.15 * h)), 0 : max(420, int(0.66 * w))],
    ]
    variants: list[np.ndarray] = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=5.0, fy=5.0, interpolation=cv2.INTER_CUBIC)
        variants.append(gray)
        for threshold in (100, 130, 160, 190):
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            variants.append(binary)

    temp_path: str | None = None
    best_raw: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temp_path = handle.name
        for variant in variants:
            cv2.imwrite(temp_path, variant)
            result = subprocess.run(
                [
                    "tesseract",
                    temp_path,
                    "stdout",
                    "--psm",
                    "7",
                    "-c",
                    "tessedit_char_whitelist=0123456789-:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            text = " ".join(result.stdout.split())
            if text and (best_raw is None or len(text) > len(best_raw)):
                best_raw = text
            parsed = _parse_ocr_timestamp(text)
            if parsed is not None:
                return parsed.isoformat(sep=" "), text
    except (OSError, subprocess.SubprocessError):
        return None, best_raw
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
    return None, best_raw


def _video_metadata(path: Path, use_ocr: bool, camera_id: str = "unknown") -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"readable": False, "error": "OpenCV could not open video"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    offsets = sample_offsets(duration, 3)
    frame = None
    samples: list[TimestampSample] = []
    for offset_s in offsets:
        cap.set(cv2.CAP_PROP_POS_MSEC, offset_s * 1000.0)
        ok, candidate = cap.read()
        if not ok:
            continue
        if frame is None:
            frame = candidate
        if use_ocr:
            observed_timestamp, observed_raw = _ocr_timestamp(candidate)
            samples.append(TimestampSample(offset_s, datetime.fromisoformat(observed_timestamp) if observed_timestamp else None, observed_raw))
    cap.release()
    brightness = None
    saturation = None
    if frame is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = float(np.mean(hsv[..., 2]))
        saturation = float(np.mean(hsv[..., 1]))
    validated = validate_timestamp_samples(samples, duration)
    timestamp = validated.recording_start.isoformat(sep=" ") if validated.recording_start else None
    recording_end = validated.recording_end.isoformat(sep=" ") if validated.recording_end else None
    hour = validated.recording_start.hour if validated.recording_start else None
    session_type = classify_session(hour)
    date_part = timestamp[:10].replace("-", "") if timestamp else None
    return {
        "readable": True,
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": duration,
        "width": width,
        "height": height,
        "camera_id": camera_id,
        "camera_timestamp": timestamp,
        "recording_timestamp": timestamp,
        "recording_start": timestamp,
        "recording_end": recording_end,
        "timestamp_source": validated.source,
        "timestamp_confidence": validated.confidence,
        "ocr_samples_json": validated.samples_json(),
        "ocr_validation_error": validated.error,
        "downloaded_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" "),
        "session_type": session_type,
        "session_id": f"{camera_id}_{date_part}_{session_type}" if date_part else None,
        "brightness": brightness,
        "saturation": saturation,
    }


def scan_videos(
    videos_dir: str | Path,
    output_csv: str | Path | None = None,
    pattern: str = "*.mp4",
    use_ocr: bool = True,
    chunk_mb: int = 8,
    camera_id: str = "unknown",
    manual_overrides: dict[str, datetime] | None = None,
) -> pd.DataFrame:
    videos_dir = Path(videos_dir).resolve()
    # Camera/NVR exports commonly use upper-case .MP4. Match the configured
    # pattern case-insensitively and keep symlinked inputs valid.
    paths = sorted(
        path
        for path in videos_dir.iterdir()
        if path.is_file() and fnmatch(path.name.lower(), pattern.lower())
    )
    rows: list[dict[str, Any]] = []
    first_for_hash: dict[str, str] = {}
    for path in paths:
        sha256 = sha256_file(path, chunk_mb=chunk_mb)
        video_id = video_id_from_hash(sha256)
        duplicate_of = first_for_hash.get(sha256)
        if duplicate_of is None:
            first_for_hash[sha256] = video_id
        metadata = _video_metadata(path, use_ocr=use_ocr, camera_id=camera_id)
        override = (manual_overrides or {}).get(path.name)
        if override is not None:
            metadata["recording_start"] = override.isoformat(sep=" ")
            metadata["recording_timestamp"] = metadata["recording_start"]
            metadata["recording_end"] = (override + timedelta(seconds=float(metadata.get("duration_s", 0)))).isoformat(sep=" ")
            metadata["timestamp_source"] = "manual"
            metadata["timestamp_confidence"] = 1.0
            metadata["session_type"] = classify_session(override.hour)
            metadata["session_id"] = f"{camera_id}_{override:%Y%m%d}_{metadata['session_type']}"
        if not metadata.get("session_id"):
            metadata["session_id"] = video_id
        rows.append(
            {
                "video_id": video_id,
                "filename": path.name,
                "path": str(path),
                "sha256": sha256,
                "size_bytes": path.stat().st_size,
                "is_duplicate": duplicate_of is not None,
                "duplicate_of": duplicate_of,
                "enabled": duplicate_of is None,
                "perceptual_fingerprint": perceptual_fingerprint(path),
                **metadata,
            }
        )
    frame = pd.DataFrame(rows)
    frame = assign_overlap_groups(frame)
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False)
    return frame


def canonical_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    enabled = bool_mask(frame.get("enabled", pd.Series(True, index=frame.index)), default=True)
    duplicate = bool_mask(frame.get("is_duplicate", pd.Series(False, index=frame.index)), default=False)
    return frame.loc[enabled & ~duplicate].copy()
