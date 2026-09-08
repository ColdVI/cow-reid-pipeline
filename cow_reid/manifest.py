"""Canonical video manifest: reconcile OCR/inventory manifests with human-reviewed
timestamp/session correction exports (e.g. ``exports/current_14_manifest_v1.csv``)
into one manifest that downstream extraction, splitting and worker code can trust.

The file-based inventory pipeline (``inventory.scan_videos``) and the manual
correction exports produced during review currently live as two disconnected
CSVs: the raw manifest keeps whatever OCR produced (including failures), and a
correction export hand-fixes a subset of rows but is never applied back to the
manifest that ``extract``/``train-reid`` actually read. This module closes that
gap and re-derives overlap groups from the corrected timestamps, so duplicate
recordings hidden by an OCR failure (a missing timestamp always looks like a
non-overlapping singleton to ``assign_overlap_groups``) are grouped correctly
once a human has supplied the real time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .overlap import assign_overlap_groups

# Columns a correction export is allowed to override on the raw manifest. Every
# other column (path, sha256, size_bytes, fps, dimensions, ocr artifacts, ...)
# is always preserved from the raw manifest untouched.
CORRECTION_COLUMNS = (
    "recording_start",
    "recording_end",
    "timestamp_source",
    "timestamp_confidence",
    "session_id",
    "session_type",
    "domain",
)


def apply_manifest_corrections(raw: pd.DataFrame, corrections: pd.DataFrame) -> pd.DataFrame:
    """Overlay one human-reviewed correction export onto the raw manifest.

    Only rows whose correction entry marks ``timestamp_source == "manual"``
    are applied; this guards against silently trusting a correction file
    that mixes reviewed and unreviewed rows.
    """
    if "video_id" not in raw.columns:
        raise ValueError("Raw manifest is missing a 'video_id' column")
    if corrections.empty:
        return raw.copy()
    if "video_id" not in corrections.columns:
        raise ValueError("Corrections file is missing a 'video_id' column")

    raw_ids = set(raw["video_id"].astype(str))
    correction_ids = set(corrections["video_id"].astype(str))
    unknown = correction_ids - raw_ids
    if unknown:
        raise ValueError(
            "Corrections reference video_id(s) not present in the raw manifest: "
            f"{sorted(unknown)}"
        )

    result = raw.copy().set_index("video_id", drop=False)
    indexed_corrections = corrections.copy().set_index("video_id", drop=False)
    available_columns = [column for column in CORRECTION_COLUMNS if column in indexed_corrections.columns]
    for column in available_columns:
        if column not in result.columns:
            result[column] = pd.NA

    for video_id, correction_row in indexed_corrections.iterrows():
        if str(correction_row.get("timestamp_source", "")).strip().lower() != "manual":
            continue
        for column in available_columns:
            value = correction_row[column]
            if pd.isna(value):
                continue
            result.at[video_id, column] = value
    return result.reset_index(drop=True)


def build_canonical_manifest(raw: pd.DataFrame, correction_frames: Iterable[pd.DataFrame] = ()) -> pd.DataFrame:
    """Apply correction exports in order, then re-derive overlap groups from the
    corrected timestamps (never trust a correction file's own hand-written
    ``overlap_group_id`` column directly — recomputing keeps one algorithm as the
    single source of truth for grouping)."""
    merged = raw.copy()
    for corrections in correction_frames:
        merged = apply_manifest_corrections(merged, corrections)
    return assign_overlap_groups(merged)


def build_canonical_manifest_files(
    raw_paths: str | Path | Iterable[str | Path],
    correction_paths: Iterable[str | Path],
    output_path: str | Path,
) -> pd.DataFrame:
    """``raw_paths`` may be a single manifest or several to concatenate first
    (e.g. two inventory batches recorded separately but sharing the same
    camera/schema) before overlaying corrections and re-deriving overlap
    groups across the combined set."""
    if isinstance(raw_paths, (str, Path)):
        raw_paths = [raw_paths]
    raw_frames = [pd.read_csv(path) for path in raw_paths]
    if not raw_frames:
        raise ValueError("At least one raw manifest is required.")
    raw = pd.concat(raw_frames, ignore_index=True, sort=False) if len(raw_frames) > 1 else raw_frames[0]
    corrections = [pd.read_csv(path) for path in correction_paths]
    result = build_canonical_manifest(raw, corrections)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
