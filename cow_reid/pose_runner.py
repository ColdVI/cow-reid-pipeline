from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .health.features import dorsal_arch_score
from .posture import import_posture_scores
from .pose.lameness_dlc import validate_checkpoint_file


DORSAL = ["withers", "thoracic_spine", "dorsal_apex", "lumbar_spine", "sacrum"]


def run_lameness_pose(run_dir: str | Path, tracklet_id: str, repo: str | Path, checkpoint: str | Path, python_executable: str = sys.executable, device: str = "auto") -> dict[str, str | int]:
    run = Path(run_dir).resolve(); tracks = pd.read_csv(run / "tracklets.csv"); selected = tracks.loc[tracks["tracklet_id"].astype(str).eq(str(tracklet_id))]
    if selected.empty: raise ValueError(f"Unknown tracklet: {tracklet_id}")
    checkpoint_path = validate_checkpoint_file(checkpoint)
    row = selected.iloc[0]; metadata = Path(str(row["metadata_path"])); output = run / "pose" / str(tracklet_id) / "pose_lameness"
    command = [python_executable, "-m", "cow_reid.pose.worker", "--repo", str(Path(repo).resolve()), "--video", str(row["source_video"]), "--metadata", str(metadata), "--output", str(output), "--checkpoint", str(checkpoint_path), "--device", device]
    subprocess.run(command, check=True)
    keypoints = pd.read_csv(output / "keypoints.csv"); scores = []
    for frame_idx, group in keypoints.groupby("frame_idx"):
        points = group.set_index("keypoint_name")
        if not set(DORSAL).issubset(points.index): continue
        dorsal = points.loc[DORSAL]
        if (pd.to_numeric(dorsal["visibility"], errors="coerce") < 0.25).any(): continue
        score = dorsal_arch_score(dorsal[["x_px", "y_px"]].to_numpy(float))
        scores.append({"tracklet_id": tracklet_id, "frame_idx": int(frame_idx), "timestamp_s": float(dorsal["timestamp_s"].iloc[0]), "arch_score": score, "model_name": str(dorsal["model_name"].iloc[0]), "model_version": str(dorsal["model_version"].iloc[0])})
    if not scores: raise RuntimeError("Pose completed but no frame passed dorsal quality gates")
    score_path = output / "posture_scores.csv"; pd.DataFrame(scores).to_csv(score_path, index=False); observation = import_posture_scores(run, score_path)
    return {"tracklet_id": str(tracklet_id), "keypoints": str(output / "keypoints.csv"), "posture_scores": str(score_path), "observation": str(observation), "valid_frames": len(scores)}
