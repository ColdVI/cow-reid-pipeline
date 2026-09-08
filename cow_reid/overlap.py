from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


def perceptual_fingerprint(path: str | Path, samples: int = 5) -> str:
    """Stable coarse fingerprint for detecting re-encoded/trimmed exports."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return ""
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indexes = np.linspace(0, max(0, frames - 1), samples, dtype=int)
    hashes: list[bytes] = []
    for index in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (16, 16), interpolation=cv2.INTER_AREA)
        bits = np.packbits(gray >= float(gray.mean()))
        hashes.append(bits.tobytes())
    cap.release()
    return hashlib.sha256(b"".join(hashes)).hexdigest() if hashes else ""


def interval_overlap_seconds(left_start: object, left_end: object, right_start: object, right_end: object) -> float:
    values = [pd.to_datetime(value, errors="coerce") for value in (left_start, left_end, right_start, right_end)]
    if any(pd.isna(value) for value in values):
        return 0.0
    return max(0.0, (min(values[1], values[3]) - max(values[0], values[2])).total_seconds())


def assign_overlap_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Group transitive time overlaps per camera; duplicates share a canonical video.

    Rows with a missing/unparseable ``recording_start`` or ``recording_end`` cannot be
    interval-overlap-tested against anything (``interval_overlap_seconds`` always
    returns 0.0 for them), so they would otherwise fall through as a falsely
    confident singleton group even if they are actually the same physical
    recording as another video. Such rows are flagged ``overlap_unverified=True``
    so callers (the split logic in particular) know this row's isolation is an
    artifact of missing timestamp data, not a verified fact, and must never treat
    it as a legitimate held-out day.
    """
    result = frame.copy()
    if result.empty:
        for column in ("overlap_group_id", "canonical_video_id"):
            result[column] = pd.Series(dtype=str)
        result["overlap_unverified"] = pd.Series(dtype=bool)
        return result
    result["overlap_group_id"] = ""
    result["canonical_video_id"] = result["video_id"].astype(str)
    result["overlap_unverified"] = (
        pd.to_datetime(result.get("recording_start"), errors="coerce").isna()
        | pd.to_datetime(result.get("recording_end"), errors="coerce").isna()
    )
    for _, indexes in result.groupby(result.get("camera_id", pd.Series("unknown", index=result.index))).groups.items():
        ordered = sorted(indexes, key=lambda idx: str(result.at[idx, "recording_start"] or ""))
        components: list[list[int]] = []
        for idx in ordered:
            attached: list[int] | None = None
            for component in components:
                if any(
                    interval_overlap_seconds(
                        result.at[idx, "recording_start"], result.at[idx, "recording_end"],
                        result.at[other, "recording_start"], result.at[other, "recording_end"],
                    ) > 0
                    for other in component
                ):
                    attached = component
                    break
            if attached is None:
                components.append([idx])
            else:
                attached.append(idx)
        for number, component in enumerate(components, start=1):
            if not component:
                continue
            canonical = str(result.loc[component].sort_values(["recording_start", "video_id"]).iloc[0]["video_id"])
            group_id = f"ovl_{hashlib.sha1('|'.join(sorted(result.loc[component, 'video_id'].astype(str))).encode()).hexdigest()[:12]}"
            result.loc[component, "overlap_group_id"] = group_id
            result.loc[component, "canonical_video_id"] = canonical
    duplicate_map = result.set_index("video_id")["canonical_video_id"].to_dict()
    for idx, row in result.iterrows():
        duplicate_of = row.get("duplicate_of")
        if pd.notna(duplicate_of) and str(duplicate_of).strip():
            result.at[idx, "canonical_video_id"] = duplicate_map.get(str(duplicate_of), str(duplicate_of))
    return result
