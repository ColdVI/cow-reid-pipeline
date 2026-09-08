from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from .schema import CanonicalPoseResult


class PoseBackend(Protocol):
    name: str
    version: str

    def infer_tracklet(
        self,
        video_path: str,
        start_s: float,
        end_s: float,
        trajectory: Sequence[dict[str, object]],
        output_dir: str | Path,
    ) -> CanonicalPoseResult: ...

