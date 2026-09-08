from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .schema import CanonicalPoseResult, KeypointFrame


class ExternalCommandPoseBackend:
    """Adapter for a keypoint repository that writes one CSV per invocation.

    Command tokens may use {video}, {start}, {end}, {trajectory}, {output}, and
    {checkpoint}. No shell is involved.
    """

    def __init__(self, command: Sequence[str], checkpoint: str, keypoint_map: Mapping[str, str], name: str, version: str):
        self.command = list(command)
        self.checkpoint = checkpoint
        self.keypoint_map = dict(keypoint_map)
        self.name = name
        self.version = version

    def infer_tracklet(self, video_path: str, start_s: float, end_s: float, trajectory: Sequence[dict[str, object]], output_dir: str | Path) -> CanonicalPoseResult:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        trajectory_path = target / "trajectory.json"
        trajectory_path.write_text(json.dumps(list(trajectory)), encoding="utf-8")
        output = target / "raw_keypoints.csv"
        values = {"video": video_path, "start": str(start_s), "end": str(end_s), "trajectory": str(trajectory_path), "output": str(output), "checkpoint": self.checkpoint}
        command = [token.format(**values) for token in self.command]
        subprocess.run(command, check=True)
        raw = pd.read_csv(output)
        required = {"tracklet_id", "frame_idx", "timestamp_s", "keypoint_name", "x_px", "y_px", "visibility"}
        if not required.issubset(raw.columns):
            raise ValueError(f"External pose CSV is missing {sorted(required.difference(raw.columns))}")
        frames = []
        for row in raw.to_dict("records"):
            name = self.keypoint_map.get(str(row["keypoint_name"]), str(row["keypoint_name"]))
            frames.append(KeypointFrame(str(row["tracklet_id"]), int(row["frame_idx"]), float(row["timestamp_s"]), name, float(row["x_px"]), float(row["y_px"]), float(row.get("x_normalized", row["x_px"])), float(row.get("y_normalized", row["y_px"])), float(row["visibility"]), self.name, self.version))
        return CanonicalPoseResult(frames[0].tracklet_id if frames else target.name, frames, str(output))

