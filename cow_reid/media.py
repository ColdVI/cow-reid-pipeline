from __future__ import annotations

from pathlib import Path

import pandas as pd


def evidence_window(start_s: float, end_s: float, video_duration_s: float | None = None, padding_s: float = 2.0) -> tuple[float, float]:
    start = max(0.0, float(start_s) - padding_s)
    end = float(end_s) + padding_s
    if video_duration_s is not None:
        end = min(float(video_duration_s), end)
    return start, max(start, end)


def enrich_tracks_with_recording_time(tracks: pd.DataFrame, run_dir: str | Path) -> pd.DataFrame:
    frame = tracks.copy()
    candidates = [Path(run_dir).resolve() / "video_manifest.csv", Path(__file__).resolve().parent.parent / "data" / "video_manifest.csv"]
    manifests = []
    for path in candidates:
        if path.exists():
            manifest = pd.read_csv(path)
            if "video_id" in manifest:
                manifests.append(manifest)
    if manifests:
        manifest = pd.concat(manifests, ignore_index=True).drop_duplicates("video_id", keep="last")
        columns = [name for name in ("video_id", "recording_start", "recording_timestamp", "camera_id", "session_id", "domain", "duration_s") if name in manifest]
        rename = {name: f"manifest_{name}" for name in columns if name != "video_id"}
        frame = frame.merge(manifest[columns].rename(columns=rename), on="video_id", how="left")
        for name in ("recording_start", "recording_timestamp", "camera_id", "session_id", "domain", "duration_s"):
            source = f"manifest_{name}"
            if source in frame:
                frame[name] = frame[source].combine_first(frame[name]) if name in frame else frame[source]
    if "recording_start" not in frame and "recording_timestamp" in frame:
        frame["recording_start"] = frame["recording_timestamp"]
    return frame

