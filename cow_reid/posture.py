from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .identity import load_identity_labels, normalize_cow_id
from .utils import bool_mask
from .health.baseline import longitudinal_baseline
from .health.alerts import review_statuses


POSTURE_COLUMNS = [
    "tracklet_id",
    "cow_id",
    "video_id",
    "session_id",
    "observed_at",
    "arch_score_mean",
    "arch_score_median",
    "arch_score_p90",
    "arch_score_max",
    "arch_score_std",
    "n_scored_frames",
    "valid_frame_ratio",
    "measurement_uncertainty",
    "model_name",
    "model_version",
    "source_scores",
]

SCORE_ALIASES = ("arch_score", "posture_score", "hunch_score", "score")


def _timestamp_plus_seconds(value: object, seconds: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return ""
    try:
        base = datetime.fromisoformat(str(value))
        return (base + timedelta(seconds=float(seconds))).isoformat(sep=" ", timespec="seconds")
    except (TypeError, ValueError):
        return ""


def load_posture_observations(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir).resolve()
    path = run_dir / "posture_observations.csv"
    if not path.exists():
        return pd.DataFrame(columns=POSTURE_COLUMNS)
    frame = pd.read_csv(path, dtype={"cow_id": str})
    for column in POSTURE_COLUMNS:
        if column not in frame:
            frame[column] = ""
    # Identity assignment can happen after posture inference. Always resolve the
    # current manual identity instead of freezing the cow_id present at import.
    labels = load_identity_labels(run_dir)
    confirmed = labels.loc[bool_mask(labels["confirmed"], default=False)].set_index("tracklet_id")["cow_id"]
    frame["cow_id"] = frame["tracklet_id"].astype(str).map(confirmed).fillna("").map(normalize_cow_id)
    return frame[POSTURE_COLUMNS]


def import_posture_scores(
    run_dir: str | Path,
    scores_csv: str | Path,
    output_csv: str | Path | None = None,
) -> Path:
    """Import per-frame or per-tracklet keypoint posture scores and attach manual identities."""
    run_dir = Path(run_dir).resolve()
    source = Path(scores_csv).resolve()
    scores = pd.read_csv(source)
    if "tracklet_id" not in scores:
        raise ValueError("Posture score CSV needs a tracklet_id column.")
    score_column = next((column for column in SCORE_ALIASES if column in scores), None)
    if score_column is None:
        raise ValueError(f"Posture score CSV needs one score column: {list(SCORE_ALIASES)}")
    total_by_tracklet = scores.groupby("tracklet_id").size()
    scores[score_column] = pd.to_numeric(scores[score_column], errors="coerce")
    scores = scores.dropna(subset=[score_column]).copy()
    if scores.empty:
        raise ValueError("Posture score CSV contains no numeric scores.")

    model_name = str(scores["model_name"].dropna().iloc[0]) if "model_name" in scores and scores["model_name"].notna().any() else "keypoint_model"
    model_version = (
        str(scores["model_version"].dropna().iloc[0])
        if "model_version" in scores and scores["model_version"].notna().any()
        else "unknown"
    )
    grouped = scores.groupby("tracklet_id")[score_column]
    summary = grouped.agg(["mean", "median", "max", "std", "count"]).reset_index()
    p90 = grouped.quantile(0.90).rename("p90").reset_index()
    summary = summary.merge(p90, on="tracklet_id", how="left")
    summary["std"] = summary["std"].fillna(0.0)
    summary["valid_frame_ratio"] = summary.apply(lambda row: float(row["count"] / max(1, total_by_tracklet.get(row["tracklet_id"], row["count"]))), axis=1)
    mad = grouped.apply(lambda values: float(np.median(np.abs(values.to_numpy(float) - np.median(values.to_numpy(float)))))).rename("mad").reset_index()
    summary = summary.merge(mad, on="tracklet_id", how="left")
    summary["measurement_uncertainty"] = 1.4826 * summary["mad"] / np.sqrt(summary["count"].clip(lower=1))

    tracks = pd.read_csv(run_dir / "tracklets.csv")
    tracks = tracks.drop_duplicates("tracklet_id")
    metadata_columns = ["tracklet_id", "video_id", "session_id", "start_s"]
    for column in metadata_columns:
        if column not in tracks:
            tracks[column] = ""
    summary = summary.merge(tracks[metadata_columns], on="tracklet_id", how="left")

    labels = load_identity_labels(run_dir)[["tracklet_id", "cow_id", "confirmed"]]
    summary = summary.merge(labels, on="tracklet_id", how="left")
    summary["cow_id"] = summary["cow_id"].map(normalize_cow_id)

    manifest_path = run_dir / "video_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        timestamp_column = "recording_timestamp" if "recording_timestamp" in manifest else "camera_timestamp"
        if timestamp_column in manifest:
            summary = summary.merge(
                manifest[["video_id", timestamp_column]].drop_duplicates("video_id"),
                on="video_id",
                how="left",
            )
            summary["observed_at"] = [
                _timestamp_plus_seconds(timestamp, start)
                for timestamp, start in zip(summary[timestamp_column], summary["start_s"])
            ]
        else:
            summary["observed_at"] = ""
    else:
        summary["observed_at"] = ""

    imported = pd.DataFrame(
        {
            "tracklet_id": summary["tracklet_id"].astype(str),
            "cow_id": summary["cow_id"].fillna("").astype(str),
            "video_id": summary["video_id"].fillna("").astype(str),
            "session_id": summary["session_id"].fillna("").astype(str),
            "observed_at": summary["observed_at"].fillna("").astype(str),
            "arch_score_mean": summary["mean"].astype(float),
            "arch_score_median": summary["median"].astype(float),
            "arch_score_p90": summary["p90"].astype(float),
            "arch_score_max": summary["max"].astype(float),
            "arch_score_std": summary["std"].astype(float),
            "n_scored_frames": summary["count"].astype(int),
            "valid_frame_ratio": summary["valid_frame_ratio"].astype(float),
            "measurement_uncertainty": summary["measurement_uncertainty"].astype(float),
            "model_name": model_name,
            "model_version": model_version,
            "source_scores": str(source),
        }
    )

    output = Path(output_csv).resolve() if output_csv else run_dir / "posture_observations.csv"
    existing = load_posture_observations(run_dir) if output == run_dir / "posture_observations.csv" else pd.DataFrame(columns=POSTURE_COLUMNS)
    combined = imported.copy() if existing.empty else pd.concat([existing, imported], ignore_index=True)
    # Re-importing one model run is idempotent; an existing raw observation is
    # never silently overwritten.
    combined = combined.drop_duplicates(["tracklet_id", "model_name", "model_version"], keep="first")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    combined[POSTURE_COLUMNS].to_csv(temporary, index=False)
    temporary.replace(output)
    return output


def posture_history(run_dir: str | Path, cow_id: str) -> pd.DataFrame:
    observations = load_posture_observations(run_dir)
    resolved = normalize_cow_id(cow_id)
    if observations.empty:
        return observations
    selected = observations.loc[observations["cow_id"].map(normalize_cow_id).eq(resolved)].copy()
    selected["arch_score_mean"] = pd.to_numeric(selected["arch_score_mean"], errors="coerce")
    selected["observation_order"] = np.arange(1, len(selected) + 1)
    if selected["observed_at"].fillna("").ne("").any():
        selected["_sort_time"] = pd.to_datetime(selected["observed_at"], errors="coerce")
        selected = selected.sort_values(["_sort_time", "session_id", "tracklet_id"], na_position="last")
    else:
        selected = selected.sort_values(["session_id", "tracklet_id"])
    selected["observation_order"] = np.arange(1, len(selected) + 1)
    scores = selected["arch_score_mean"]
    selected["personal_baseline"] = scores.expanding(min_periods=3).median().shift(1)
    selected["delta_from_baseline"] = scores - selected["personal_baseline"]
    return selected.drop(columns=["_sort_time"], errors="ignore").reset_index(drop=True)


def refresh_health_summaries(run_dir: str | Path, output_csv: str | Path | None = None) -> Path:
    """Rebuild cow/model-specific baselines from append-only observations."""
    run_dir = Path(run_dir).resolve()
    observations = load_posture_observations(run_dir)
    rows: list[pd.DataFrame] = []
    for (cow_id, model_name, model_version), group in observations.loc[observations["cow_id"].ne("")].groupby(["cow_id", "model_name", "model_version"]):
        prepared = group.rename(columns={"arch_score_median": "value", "measurement_uncertainty": "uncertainty"}).copy()
        baseline = longitudinal_baseline(prepared, min_days=3)
        if baseline.empty:
            continue
        baseline["status"] = review_statuses(baseline)
        baseline["cow_id"] = cow_id
        baseline["model_name"] = model_name
        baseline["model_version"] = model_version
        rows.append(baseline)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["cow_id", "tracklet_id", "observed_at", "value", "baseline", "mad", "delta", "robust_z", "status", "model_name", "model_version"])
    target = Path(output_csv).resolve() if output_csv else run_dir / "health_summaries.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False)
    return target
