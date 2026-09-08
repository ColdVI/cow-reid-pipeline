from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .embeddings import build_track_embeddings
from .extract import extract_tracklets
from .gallery import identify_run
from .metric_model import infer_checkpoint_kind
from .registry import pending_videos, record_video_error, register_embeddings, register_model, register_tracklets, transition_video


def process_new_videos(
    database_url: str,
    output_root: str | Path,
    config: dict[str, Any],
    checkpoint: str,
    device: str = "auto",
    gallery_path: str | Path | None = None,
    backend: str | None = None,
) -> list[dict[str, object]]:
    """Process each registered video in its own resumable run directory."""
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config["embedding"]["checkpoint"] = checkpoint
    config["embedding"]["device"] = device
    if backend is not None:
        config["embedding"]["backend"] = backend
    resolved_backend = str(config["embedding"].get("backend"))
    checkpoint_kind = infer_checkpoint_kind(checkpoint)
    # Passing --checkpoint alone does NOT select a backend: the default backend
    # (opencows2020) uses a completely different key-remapping loader that will
    # either silently drop every tensor or crash deep inside build_track_embeddings
    # if handed a farm metric_resnet50 checkpoint. Fail fast, before any
    # extraction/DB work happens, with a message that names the fix.
    if resolved_backend == "metric" and checkpoint_kind != "metric_resnet50":
        raise ValueError(
            f"embedding.backend='metric' requires a farm metric_resnet50 checkpoint; "
            f"'{checkpoint}' does not look like one. Pass --backend matching the "
            "checkpoint (opencows2020/resnet18/hist), or point --checkpoint at a "
            "metric_resnet50 checkpoint."
        )
    if resolved_backend != "metric" and checkpoint_kind == "metric_resnet50":
        raise ValueError(
            f"'{checkpoint}' is a farm metric_resnet50 checkpoint but "
            f"embedding.backend='{resolved_backend}'. Pass --backend metric to select "
            "the matching loader."
        )
    model_version = register_model(database_url, checkpoint)
    results: list[dict[str, object]] = []
    queued = []
    for state in ("INGESTED", "TRACKED", "EMBEDDED"):
        queued.extend(pending_videos(database_url, state))
    for video in queued:
        video_id = str(video["video_uuid"])
        state = str(video["status"])
        canonical_uuid = video.get("canonical_video_uuid")
        if canonical_uuid and str(canonical_uuid) != video_id:
            # A duplicate recording of an already-tracked physical session
            # (shared overlap_group_id): never run it through its own independent
            # extract/embed/identify pipeline — that would double-count one
            # physical passage as two, and is exactly the leak this pipeline is
            # meant to avoid. Just link it to its canonical video and stop.
            transition_video(database_url, video_id, state, "DUPLICATE_LINKED")
            results.append({
                "video_uuid": video_id,
                "status": "DUPLICATE_LINKED",
                "canonical_video_uuid": str(canonical_uuid),
            })
            continue
        run_dir = root / video_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = run_dir / "video_manifest.csv"
        row = {
            "video_id": video_id, "path": video["media_uri"], "filename": Path(str(video["media_uri"])).name,
            "session_id": video.get("overlap_group_id") or video_id, "enabled": True, "is_duplicate": False,
            "recording_start": video.get("recording_start"), "recording_end": video.get("recording_end"),
            "recording_timestamp": video.get("recording_start"), "camera_id": video.get("camera_id"),
        }
        pd.DataFrame([row]).to_csv(manifest, index=False)
        try:
            if state == "INGESTED":
                if not (run_dir / "tracklets.csv").exists():
                    extract_tracklets(manifest, run_dir, config)
                register_tracklets(database_url, video_id, run_dir / "tracklets.csv")
                transition_video(database_url, video_id, "INGESTED", "TRACKED")
                state = "TRACKED"
            if state == "TRACKED":
                if not (run_dir / "track_embeddings.npz").exists():
                    build_track_embeddings(run_dir, config)
                register_embeddings(database_url, run_dir / "track_embeddings.npz", model_version)
                transition_video(database_url, video_id, "TRACKED", "EMBEDDED")
                state = "EMBEDDED"
            target = "ID_PENDING"
            if state == "EMBEDDED":
                if gallery_path:
                    identify_run(run_dir, gallery_path, config)
                    summary_path = run_dir / "identity_summary.json"
                    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
                    resolved = int(summary.get("known_predictions", 0)) - int(summary.get("ambiguous_predictions", 0))
                    # "ran the identification step" is not the same as "assigned an
                    # identity" — a run where every tracklet came back UNKNOWN/
                    # AMBIGUOUS must not be recorded as if a match was found.
                    target = "ID_ASSIGNED" if resolved > 0 else "ID_UNRESOLVED"
                transition_video(database_url, video_id, "EMBEDDED", target)
            results.append({"video_uuid": video_id, "status": target, "run": str(run_dir)})
        except Exception as exc:
            # Keep the last successful stage resumable and record the failure.
            record_video_error(database_url, video_id, str(exc))
            results.append({"video_uuid": video_id, "status": "ERROR", "error": str(exc), "run": str(run_dir)})
    return results
