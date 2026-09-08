import json
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from cow_reid.debug_overlay import _interpolate_record, _load_pose_frames, annotation_sidecar
from cow_reid.media import evidence_window
from cow_reid.pose.lameness_dlc import validate_checkpoint_file
from cow_reid.review_state import record_tracklet_state
from cow_reid.review_bundle import build_review_bundle


def test_evidence_window_adds_padding_and_clamps_to_video():
    assert evidence_window(1.0, 4.0, 5.0, 2.0) == (0.0, 5.0)
    assert evidence_window(8.0, 9.0, 20.0, 2.0) == (6.0, 11.0)


def test_debug_annotation_interpolates_tracklet_box(tmp_path: Path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "records": [
                    {"timestamp_s": 1.0, "bbox": [0, 10, 20, 30], "confidence": 0.8},
                    {"timestamp_s": 3.0, "bbox": [20, 10, 40, 30], "confidence": 1.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    sidecar = annotation_sidecar(
        {"tracklet_id": "t1", "metadata_path": metadata, "start_s": 1, "end_s": 3},
        "COW_0012",
        "PREDICTED",
        0.94,
    )
    middle = _interpolate_record(sidecar["records"], 2.0)
    assert sidecar["label"] == "COW_0012 · PREDICTED · similarity=0.940"
    assert middle is not None
    assert middle["bbox"] == [10.0, 10.0, 30.0, 30.0]
    assert middle["confidence"] == pytest.approx(0.9)


def test_tracklet_review_state_is_latest_value(tmp_path: Path):
    record_tracklet_state(tmp_path, "t1", "unsure")
    record_tracklet_state(tmp_path, "t1", "split_requested", split_at_s=12.5)
    rows = pd.read_csv(tmp_path / "tracklet_reviews.csv")
    assert len(rows) == 1
    assert rows.iloc[0]["state"] == "split_requested"
    assert rows.iloc[0]["split_at_s"] == pytest.approx(12.5)


def test_pose_overlay_uses_only_visible_keypoints(tmp_path: Path):
    path = tmp_path / "keypoints.csv"
    pd.DataFrame(
        [
            {"frame_idx": 7, "keypoint_name": "withers", "x_px": 10.4, "y_px": 20.6, "visibility": 0.9, "model_version": "v1"},
            {"frame_idx": 7, "keypoint_name": "sacrum", "x_px": 30, "y_px": 40, "visibility": 0.1, "model_version": "v1"},
        ]
    ).to_csv(path, index=False)
    frames, token = _load_pose_frames(path)
    assert frames == {7: {"withers": (10, 21)}}
    assert token.endswith(":v1")


def test_git_lfs_pointer_is_rejected_with_actionable_error(tmp_path: Path):
    checkpoint = tmp_path / "snapshot.pt"
    checkpoint.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\n"
        "size 118236749\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Git LFS pointer.*118236749 bytes"):
        validate_checkpoint_file(checkpoint)


def test_unassigned_best_guess_queue_orders_by_confidence_and_skips_confirmed(tmp_path: Path):
    from cow_reid.review_app import _unassigned_best_guess_queue

    pd.DataFrame(
        [
            {"tracklet_id": "t1", "pseudo_id": "", "session_id": "s1", "video_id": "v1", "cow_id": "COW_0001", "confirmed": True, "assignment_source": "manual", "notes": "", "updated_at": ""},
            {"tracklet_id": "t2", "pseudo_id": "", "session_id": "s2", "video_id": "v2", "cow_id": "", "confirmed": False, "assignment_source": "", "notes": "", "updated_at": ""},
            {"tracklet_id": "t3", "pseudo_id": "", "session_id": "s3", "video_id": "v3", "cow_id": "", "confirmed": False, "assignment_source": "", "notes": "", "updated_at": ""},
        ]
    ).to_csv(tmp_path / "labels.csv", index=False)
    pd.DataFrame(
        [
            {"query_tracklet_id": "t1", "rank": 1, "candidate_tracklet_id": "t2", "cosine_similarity": 0.9, "geometry_similarity": 0.8},
            {"query_tracklet_id": "t2", "rank": 1, "candidate_tracklet_id": "t1", "cosine_similarity": 0.9, "geometry_similarity": 0.8},
            {"query_tracklet_id": "t3", "rank": 1, "candidate_tracklet_id": "t1", "cosine_similarity": 0.6, "geometry_similarity": 0.5},
        ]
    ).to_csv(tmp_path / "candidate_pairs.csv", index=False)
    labels = pd.read_csv(tmp_path / "labels.csv")

    queue = _unassigned_best_guess_queue(tmp_path, labels)

    # t1 is already confirmed, so it never appears as a query (left-hand) tracklet.
    assert queue["tracklet_id"].tolist() == ["t2", "t3"]
    assert queue.iloc[0]["cosine_similarity"] >= queue.iloc[1]["cosine_similarity"]


def test_review_bundle_preserves_old_label_and_adds_new_candidate(tmp_path: Path):
    base, new, output = tmp_path / "base", tmp_path / "new", tmp_path / "combined"
    base.mkdir(); new.mkdir()
    pd.DataFrame([{"tracklet_id": "old", "video_id": "v1", "session_id": "s1"}]).to_csv(base / "tracklets.csv", index=False)
    pd.DataFrame([{"tracklet_id": "new", "video_id": "v2", "session_id": "s2"}]).to_csv(new / "tracklets.csv", index=False)
    pd.DataFrame([{"tracklet_id": "old", "frame_idx": 1}, {"tracklet_id": "old", "frame_idx": 2}]).to_csv(base / "frames.csv", index=False)
    pd.DataFrame([{"tracklet_id": "new", "frame_idx": 1}, {"tracklet_id": "new", "frame_idx": 2}]).to_csv(new / "frames.csv", index=False)
    pd.DataFrame([{"tracklet_id": "old", "pseudo_id": "P1", "session_id": "s1", "video_id": "v1", "cow_id": "COW_0001", "confirmed": True}]).to_csv(base / "labels.csv", index=False)
    pd.DataFrame([{"tracklet_id": "new", "session_id": "s2", "pseudo_id": "P1"}]).to_csv(new / "pseudo_id_assignments.csv", index=False)
    pd.DataFrame([{"tracklet_id": "new", "predicted_cow_id": "UNKNOWN", "top_1_cow_id": "COW_0001", "top_1_similarity": 0.8}]).to_csv(new / "identity_predictions.csv", index=False)
    for run, tracklet, session, video in ((base, "old", "s1", "v1"), (new, "new", "s2", "v2")):
        np.savez_compressed(run / "track_embeddings.npz", tracklet_ids=[tracklet], embeddings=np.asarray([[1, 0]], np.float32), session_ids=[session], video_ids=[video], backend=["metric"], checkpoint=["model.pt"])
    summary = build_review_bundle(base, new, output)
    labels = pd.read_csv(output / "labels.csv")
    assert summary["track_embeddings"] == 2
    assert summary["frames"] == 4
    assert labels.loc[labels.tracklet_id.eq("old"), "cow_id"].iloc[0] == "COW_0001"
    assert labels.loc[labels.tracklet_id.eq("new"), "pseudo_id"].iloc[0] == "NEW_20260903_P1"


def test_review_bundle_merges_tracklet_reviews_from_both_runs(tmp_path: Path):
    base, new, output = tmp_path / "base", tmp_path / "new", tmp_path / "combined"
    base.mkdir(); new.mkdir()
    pd.DataFrame([{"tracklet_id": "old", "video_id": "v1", "session_id": "s1"}]).to_csv(base / "tracklets.csv", index=False)
    pd.DataFrame([{"tracklet_id": "new", "video_id": "v2", "session_id": "s2"}]).to_csv(new / "tracklets.csv", index=False)
    pd.DataFrame([{"tracklet_id": "old", "frame_idx": 1}]).to_csv(base / "frames.csv", index=False)
    pd.DataFrame([{"tracklet_id": "new", "frame_idx": 1}]).to_csv(new / "frames.csv", index=False)
    pd.DataFrame([{"tracklet_id": "old", "pseudo_id": "P1", "session_id": "s1", "video_id": "v1", "cow_id": "COW_0001", "confirmed": True}]).to_csv(base / "labels.csv", index=False)
    pd.DataFrame([{"tracklet_id": "new", "session_id": "s2", "pseudo_id": "P1"}]).to_csv(new / "pseudo_id_assignments.csv", index=False)
    for run, tracklet, session, video in ((base, "old", "s1", "v1"), (new, "new", "s2", "v2")):
        np.savez_compressed(run / "track_embeddings.npz", tracklet_ids=[tracklet], embeddings=np.asarray([[1, 0]], np.float32), session_ids=[session], video_ids=[video], backend=["metric"], checkpoint=["model.pt"])
    record_tracklet_state(base, "old", "tracklet_error")
    record_tracklet_state(new, "new", "split_requested", split_at_s=5.0)

    build_review_bundle(base, new, output)

    reviews = pd.read_csv(output / "tracklet_reviews.csv").set_index("tracklet_id")
    # Previously, tracklet_reviews.csv was shutil.copy2'd from `base` only, so
    # `new`'s review ("new" -> split_requested) was silently dropped.
    assert set(reviews.index) == {"old", "new"}
    assert reviews.loc["old", "state"] == "tracklet_error"
    assert reviews.loc["new", "state"] == "split_requested"
