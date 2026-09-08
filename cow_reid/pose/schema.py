from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd


CANONICAL_COLUMNS = [
    "tracklet_id", "frame_idx", "timestamp_s", "keypoint_name", "x_px", "y_px",
    "x_normalized", "y_normalized", "visibility", "model_name", "model_version",
]


@dataclass(frozen=True)
class KeypointFrame:
    tracklet_id: str
    frame_idx: int
    timestamp_s: float
    keypoint_name: str
    x_px: float
    y_px: float
    x_normalized: float
    y_normalized: float
    visibility: float
    model_name: str
    model_version: str


@dataclass
class CanonicalPoseResult:
    tracklet_id: str
    frames: list[KeypointFrame] = field(default_factory=list)
    raw_output_uri: str | None = None
    quality_summary: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if any(frame.tracklet_id != self.tracklet_id for frame in self.frames):
            raise ValueError("Every keypoint row must belong to the result tracklet_id.")
        if any(not 0.0 <= frame.visibility <= 1.0 for frame in self.frames):
            raise ValueError("visibility must be in [0, 1].")

    def to_csv(self, path: str | Path) -> Path:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(frame) for frame in self.frames], columns=CANONICAL_COLUMNS).to_csv(target, index=False)
        return target

