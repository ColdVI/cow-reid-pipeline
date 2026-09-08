"""Regression tests for cow_reid.worker.process_new_videos: the backend/
checkpoint fail-fast validation, skipping reprocessing of duplicate-recording
overlap-group members, and no longer reporting an identification run that
found zero known identities as "ID_ASSIGNED".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cow_reid.config import load_config
from cow_reid.db import connect, init_database
from cow_reid.worker import process_new_videos


def _insert_video(database: str, video_uuid: str, status: str, canonical_video_uuid: str | None = None) -> None:
    with connect(database) as connection:
        connection.execute(
            "INSERT INTO videos(video_uuid,sha256,camera_id,media_uri,status,canonical_video_uuid) "
            "VALUES (?,?,?,?,?,?)",
            (video_uuid, f"hash-{video_uuid}", "cam1", f"{video_uuid}.mp4", status, canonical_video_uuid),
        )


def test_process_new_videos_rejects_metric_checkpoint_with_default_backend(tmp_path: Path):
    database = f"sqlite:///{tmp_path / 'registry.db'}"
    init_database(database)
    _insert_video(database, "v1", "INGESTED")
    checkpoint = tmp_path / "farm_metric_resnet50.pt"
    torch.save({"model_name": "metric_resnet50", "state_dict": {}, "class_names": ["COW_0001"]}, checkpoint)

    config = load_config(None)  # default embedding.backend == "opencows2020"
    with pytest.raises(ValueError, match="backend"):
        process_new_videos(database, tmp_path / "processed", config, str(checkpoint))


def test_process_new_videos_accepts_matching_metric_backend_and_checkpoint(tmp_path: Path):
    database = f"sqlite:///{tmp_path / 'registry.db'}"
    init_database(database)
    # No pending videos, so this only exercises the fail-fast validation path
    # (it must NOT raise once backend and checkpoint agree).
    checkpoint = tmp_path / "farm_metric_resnet50.pt"
    torch.save({"model_name": "metric_resnet50", "state_dict": {}, "class_names": ["COW_0001"]}, checkpoint)
    config = load_config(None)
    results = process_new_videos(database, tmp_path / "processed", config, str(checkpoint), backend="metric")
    assert results == []


def test_process_new_videos_links_duplicate_overlap_group_member_without_reprocessing(tmp_path: Path):
    database = f"sqlite:///{tmp_path / 'registry.db'}"
    init_database(database)
    # The canonical video is already past the resumable states, so only the
    # duplicate is queued -- isolating the duplicate-skip path from needing a
    # real extract/embed pipeline.
    _insert_video(database, "v_canonical", "ID_ASSIGNED")
    _insert_video(database, "v_duplicate", "INGESTED", canonical_video_uuid="v_canonical")

    checkpoint = tmp_path / "opencows.pkl"
    checkpoint.write_bytes(b"not-a-torch-checkpoint")
    config = load_config(None)

    results = process_new_videos(database, tmp_path / "processed", config, str(checkpoint))

    assert len(results) == 1
    assert results[0]["video_uuid"] == "v_duplicate"
    assert results[0]["status"] == "DUPLICATE_LINKED"
    assert results[0]["canonical_video_uuid"] == "v_canonical"
    assert not (tmp_path / "processed" / "v_duplicate").exists()
    with connect(database) as connection:
        row = connection.execute("SELECT status FROM videos WHERE video_uuid='v_duplicate'").fetchone()
        assert row[0] == "DUPLICATE_LINKED"


def test_process_new_videos_reports_unresolved_when_every_prediction_is_unknown(tmp_path: Path):
    database = f"sqlite:///{tmp_path / 'registry.db'}"
    init_database(database)
    _insert_video(database, "v1", "EMBEDDED")

    run_dir = tmp_path / "processed" / "v1"
    run_dir.mkdir(parents=True)
    np.savez_compressed(
        run_dir / "track_embeddings.npz",
        tracklet_ids=np.asarray(["t1"]),
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        session_ids=np.asarray(["s1"]),
        video_ids=np.asarray(["v1"]),
        backend=np.asarray(["test"]),
    )
    gallery_path = tmp_path / "gallery.npz"
    np.savez_compressed(
        gallery_path,
        cow_ids=np.asarray(["cow_a"]),
        prototypes=np.asarray([[0.0, 1.0]], dtype=np.float32),  # orthogonal -> similarity 0, far below threshold
        backend=np.asarray(["test"]),
        checkpoint=np.asarray(["unknown"]),
    )
    checkpoint = tmp_path / "opencows.pkl"
    checkpoint.write_bytes(b"not-a-torch-checkpoint")
    config = load_config(None)

    results = process_new_videos(database, tmp_path / "processed", config, str(checkpoint), gallery_path=str(gallery_path))

    assert len(results) == 1
    assert results[0]["status"] == "ID_UNRESOLVED"
    with connect(database) as connection:
        row = connection.execute("SELECT status FROM videos WHERE video_uuid='v1'").fetchone()
        assert row[0] == "ID_UNRESOLVED"
