from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from cow_reid.db import connect, init_database
from cow_reid.health.baseline import longitudinal_baseline
from cow_reid.health.features import dorsal_arch_score, summarize_feature
from cow_reid.identity_graph import create_cow, find_merge_conflicts, merge_cows, split_cow, undo_identity_operation
from cow_reid.overlap import assign_overlap_groups
from cow_reid.registry import register_embeddings, register_model, register_tracklets
from cow_reid.timestamping import TimestampSample, validate_timestamp_samples


def test_multiframe_timestamp_validation_and_rejection():
    start = datetime(2026, 8, 13, 15, 47)
    samples = [TimestampSample(offset, start + timedelta(seconds=offset), "ocr") for offset in (1, 50, 99)]
    result = validate_timestamp_samples(samples, 100)
    assert result.source == "camera_ocr"
    assert result.recording_start == start
    bad = validate_timestamp_samples([samples[0], TimestampSample(50, start + timedelta(seconds=500), "bad")], 100)
    assert bad.source == "unknown"


def test_overlap_groups_are_camera_and_time_safe():
    frame = pd.DataFrame({
        "video_id": ["a", "b", "c"], "camera_id": ["side", "side", "top"],
        "recording_start": ["2026-08-13 15:47:00", "2026-08-13 15:50:00", "2026-08-13 15:50:00"],
        "recording_end": ["2026-08-13 15:52:00", "2026-08-13 15:55:00", "2026-08-13 15:55:00"],
        "duplicate_of": ["", "", ""],
    })
    result = assign_overlap_groups(frame).set_index("video_id")
    assert result.loc["a", "overlap_group_id"] == result.loc["b", "overlap_group_id"]
    assert result.loc["a", "overlap_group_id"] != result.loc["c", "overlap_group_id"]


def test_arch_feature_and_robust_baseline():
    assert dorsal_arch_score(np.asarray([[0, 0], [1, 1], [2, 0]])) == 0.5
    assert summarize_feature(np.asarray([0.1, 0.2, np.nan]), total_frames=4)["valid_frame_ratio"] == 0.5
    times = pd.date_range("2026-08-01", periods=5, freq="D")
    result = longitudinal_baseline(pd.DataFrame({"observed_at": times, "value": [1, 1, 1, 1, 2]}), min_days=3)
    assert result.iloc[-1]["baseline"] == 1
    assert result.iloc[-1]["delta"] == 1


def test_db_merge_conflict_and_split_are_audited(tmp_path: Path):
    database = f"sqlite:///{tmp_path / 'registry.db'}"
    init_database(database)
    left = create_cow(database, "COW_0022")
    right = create_cow(database, "COW_0023")
    with connect(database) as connection:
        connection.execute("INSERT INTO videos(video_uuid,sha256,camera_id,media_uri) VALUES ('v','h','side','x.mp4')")
        connection.execute("INSERT INTO tracklets(tracklet_uuid,video_uuid,cow_uuid,start_s,end_s) VALUES ('t1','v',?,0,1)", (left,))
        connection.execute("INSERT INTO tracklets(tracklet_uuid,video_uuid,cow_uuid,start_s,end_s) VALUES ('t2','v',?,2,3)", (right,))
        connection.execute("INSERT INTO identity_edges(edge_uuid,left_entity,right_entity,decision) VALUES ('e','t1','t2','different')")
    assert find_merge_conflicts(database, [left, right])
    audit = merge_cows(database, [left, right], left, resolve_conflicts=True)
    new_cow, split_audit = split_cow(database, left, ["t2"], "COW_0024")
    with connect(database) as connection:
        assert connection.execute("SELECT cow_uuid FROM tracklets WHERE tracklet_uuid='t1'").fetchone()[0] == left
        assert connection.execute("SELECT cow_uuid FROM tracklets WHERE tracklet_uuid='t2'").fetchone()[0] == new_cow
        assert connection.execute("SELECT count(*) FROM identity_audit").fetchone()[0] == 2
    undo_identity_operation(database, split_audit)
    undo_identity_operation(database, audit)
    with connect(database) as connection:
        assert connection.execute("SELECT cow_uuid FROM tracklets WHERE tracklet_uuid='t2'").fetchone()[0] == right


def test_registry_model_tracklet_embedding_is_idempotent(tmp_path: Path):
    database = f"sqlite:///{tmp_path / 'registry.db'}"
    init_database(database)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    model_version = register_model(database, checkpoint)
    with connect(database) as connection:
        connection.execute("INSERT INTO videos(video_uuid,sha256,camera_id,media_uri) VALUES ('v','h','side','x.mp4')")
    tracks = tmp_path / "tracklets.csv"
    pd.DataFrame([{"tracklet_id": "t", "start_s": 0, "end_s": 1, "valid": True, "mean_quality": 0.8}]).to_csv(tracks, index=False)
    assert register_tracklets(database, "v", tracks) == 1
    assert register_tracklets(database, "v", tracks) == 0
    embeddings = tmp_path / "track_embeddings.npz"
    np.savez_compressed(embeddings, tracklet_ids=np.asarray(["t"]), embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32))
    assert register_embeddings(database, embeddings, model_version) == 1
    assert register_embeddings(database, embeddings, model_version) == 0
