from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cow_reid.audit import audit_labels
from cow_reid.clustering import candidate_pairs, constrained_greedy_cluster
from cow_reid.evaluation import evaluate_embeddings
from cow_reid.gallery import build_gallery
from cow_reid.identity import (
    accept_pair_as_same,
    assign_tracklets,
    auto_confirm_dual_signal,
    identity_candidates,
    identity_catalog,
    load_identity_labels,
    load_pair_reviews,
    next_cow_id,
    normalize_cow_id,
    reject_identity_candidate,
    save_pair_review,
    save_identity_labels,
)
from cow_reid.inventory import _parse_ocr_timestamp, classify_session, parse_filename_timestamp, scan_videos
from cow_reid.posture import import_posture_scores, load_posture_observations
from cow_reid.report import generate_report
from cow_reid.training import _derive_tracklet_split
from cow_reid.utils import bool_mask, torso_box


def test_session_classifier():
    assert classify_session(7) == "morning"
    assert classify_session(15) == "afternoon"
    assert classify_session(0) == "night"


def test_ocr_timestamp_rejects_calendar_mismatch():
    assert _parse_ocr_timestamp("08-25-2026 Tue 07:59:33").isoformat() == "2026-08-25T07:59:33"
    assert _parse_ocr_timestamp("03-20-2026 Thu 15:32:31") is None
    assert _parse_ocr_timestamp("08-28-2626 Fri 07:51:58") is None
    assert _parse_ocr_timestamp("08-28-2026 07:51:58").isoformat() == "2026-08-28T07:51:58"


def test_whatsapp_filename_timestamp_fallback():
    parsed = parse_filename_timestamp("WhatsApp Video 2026-09-02 at 13.21.07 (1).mp4")
    assert parsed.isoformat() == "2026-09-02T13:21:07"
    assert parse_filename_timestamp("WhatsApp Video WA0007.mp4") is None


def test_inventory_matches_uppercase_mp4(tmp_path: Path, monkeypatch):
    video = tmp_path / "CAMERA.MP4"
    video.write_bytes(b"video")
    monkeypatch.setattr("cow_reid.inventory.sha256_file", lambda *_args, **_kwargs: "a" * 64)
    monkeypatch.setattr("cow_reid.inventory.perceptual_fingerprint", lambda *_args, **_kwargs: "fingerprint")
    monkeypatch.setattr(
        "cow_reid.inventory._video_metadata",
        lambda *_args, **_kwargs: {
            "readable": True,
            "camera_id": "side",
            "recording_start": None,
            "recording_end": None,
            "session_id": "session",
        },
    )
    result = scan_videos(tmp_path, use_ocr=False)
    assert result["filename"].tolist() == ["CAMERA.MP4"]


def test_csv_boolean_parser():
    values = pd.Series(["True", "False", "1", "0", None])
    assert bool_mask(values, default=False).tolist() == [True, False, True, False, False]


def test_training_split_is_tracklet_and_session_safe():
    labels = pd.DataFrame(
        {
            "tracklet_id": ["a1", "a2", "b1", "b2"],
            "cow_id": ["cow_a", "cow_a", "cow_b", "cow_b"],
            "confirmed": [True, True, True, True],
        }
    )
    tracks = pd.DataFrame(
        {
            "tracklet_id": ["a1", "a2", "b1", "b2"],
            "session_id": ["s1", "s2", "s1", "s2"],
            "video_id": ["v1", "v2", "v1", "v2"],
        }
    )
    video_split = pd.DataFrame({"video_id": ["v1", "v2"], "split": ["train", "validation"]})
    split = _derive_tracklet_split(labels, tracks, video_split)
    assert set(split.loc[split["video_id"].eq("v1"), "split"]) == {"train"}
    assert set(split.loc[split["video_id"].eq("v2"), "split"]) == {"val"}


def test_torso_box_is_inside_frame():
    box = torso_box([100, 100, 500, 400], 640, 480, "left_to_right")
    assert 0 <= box[0] < box[2] <= 640
    assert 0 <= box[1] < box[3] <= 480


def test_constrained_cluster_respects_session():
    tracklet_ids = np.asarray(["a", "b", "c"])
    embeddings = np.asarray([[1.0, 0.0], [0.999, 0.01], [0.998, 0.02]], dtype=np.float32)
    sessions = np.asarray(["s1", "s1", "s2"])
    result = constrained_greedy_cluster(tracklet_ids, embeddings, sessions, threshold=0.9, same_session_cannot_link=True)
    pseudo = dict(zip(result.tracklet_id, result.pseudo_id))
    assert pseudo["a"] != pseudo["b"]
    assert pseudo["c"] in {pseudo["a"], pseudo["b"]}


def test_candidate_pairs_has_schema_when_only_one_session():
    result = candidate_pairs(
        np.asarray(["a", "b"]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray(["same_session", "same_session"]),
    )
    assert result.empty
    assert result.columns.tolist() == [
        "query_tracklet_id",
        "query_session_id",
        "rank",
        "candidate_tracklet_id",
        "candidate_session_id",
        "cosine_similarity",
        "geometry_similarity",
    ]


def test_candidate_pairs_geometry_similarity_optional():
    tracklet_ids = np.asarray(["a", "b"])
    embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    session_ids = np.asarray(["s1", "s2"])

    without_geometry = candidate_pairs(tracklet_ids, embeddings, session_ids)
    assert without_geometry["geometry_similarity"].isna().all()

    matching_geometry = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    with_geometry = candidate_pairs(
        tracklet_ids, embeddings, session_ids, geometry_embeddings=matching_geometry
    )
    assert with_geometry.loc[0, "geometry_similarity"] == pytest.approx(1.0)

    mirrored_geometry = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mirrored = candidate_pairs(
        tracklet_ids, embeddings, session_ids, geometry_embeddings=mirrored_geometry
    )
    assert mirrored.loc[0, "cosine_similarity"] > 0.9
    assert mirrored.loc[0, "geometry_similarity"] == pytest.approx(0.0)


def test_report_accepts_legacy_empty_candidate_file(tmp_path: Path):
    pd.DataFrame(
        [
            {
                "tracklet_id": "t1",
                "session_id": "s1",
                "valid": True,
                "start_s": 0.0,
                "end_s": 1.0,
                "mean_quality": 0.8,
                "n_best_frames": 3,
                "contact_path": "",
            }
        ]
    ).to_csv(tmp_path / "tracklets.csv", index=False)
    pd.DataFrame(
        [
            {
                "tracklet_id": "t1",
                "session_id": "s1",
                "pseudo_id": "PSEUDO_0001",
                "similarity_to_cluster": 1.0,
                "needs_review": False,
            }
        ]
    ).to_csv(tmp_path / "pseudo_id_assignments.csv", index=False)
    (tmp_path / "candidate_pairs.csv").touch()
    report = generate_report(tmp_path)
    assert report.exists()


def test_gallery_build(tmp_path: Path):
    ids = np.asarray(["t1", "t2", "t3"])
    np.savez_compressed(
        tmp_path / "track_embeddings.npz",
        tracklet_ids=ids,
        embeddings=np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
        session_ids=np.asarray(["s1", "s2", "s1"]),
        video_ids=np.asarray(["v1", "v2", "v1"]),
        backend=np.asarray(["test"]),
    )
    labels = pd.DataFrame({"tracklet_id": ids, "cow_id": ["cow_a", "cow_a", "cow_b"]})
    labels.to_csv(tmp_path / "labels.csv", index=False)
    output = build_gallery(tmp_path, tmp_path / "labels.csv")
    with np.load(output, allow_pickle=False) as gallery:
        assert set(gallery["cow_ids"].tolist()) == {"cow_a", "cow_b"}


def _identity_run(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "tracklet_id": ["t1", "t2", "t3", "t4"],
            "video_id": ["v1", "v2", "v3", "v4"],
            "session_id": ["s1", "s2", "s3", "s4"],
            "valid": [True, True, True, True],
            "source_filename": ["one.mp4", "two.mp4", "three.mp4", "four.mp4"],
            "start_s": [1.0, 2.0, 3.0, 4.0],
            "contact_path": ["", "", "", ""],
        }
    ).to_csv(tmp_path / "tracklets.csv", index=False)
    pd.DataFrame(
        {
            "tracklet_id": ["t1", "t2", "t3", "t4"],
            "pseudo_id": ["P1", "P2", "P3", "P4"],
            "session_id": ["s1", "s2", "s3", "s4"],
        }
    ).to_csv(tmp_path / "pseudo_id_assignments.csv", index=False)
    np.savez_compressed(
        tmp_path / "track_embeddings.npz",
        tracklet_ids=np.asarray(["t1", "t2", "t3", "t4"]),
        embeddings=np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32),
        session_ids=np.asarray(["s1", "s2", "s3", "s4"]),
        video_ids=np.asarray(["v1", "v2", "v3", "v4"]),
        backend=np.asarray(["test"]),
    )


def test_manual_identity_assignment_and_retrieval(tmp_path: Path):
    _identity_run(tmp_path)
    labels = load_identity_labels(tmp_path)
    assert normalize_cow_id("cow1") == "COW_0001"
    assert next_cow_id(labels) == "COW_0001"
    labels, cow_id = assign_tracklets(labels, ["t1", "t2"])
    assert cow_id == "COW_0001"
    save_identity_labels(labels, tmp_path)
    reloaded = load_identity_labels(tmp_path)
    catalog = identity_catalog(reloaded)
    assert catalog.loc[0, "tracklets"] == 2
    candidates = identity_candidates(tmp_path, reloaded, cow_id, top_k=2)
    assert candidates.iloc[0]["tracklet_id"] == "t4"


def test_posture_scores_attach_to_identity(tmp_path: Path):
    _identity_run(tmp_path)
    labels = load_identity_labels(tmp_path)
    labels, _ = assign_tracklets(labels, ["t1", "t2"], "cow1")
    save_identity_labels(labels, tmp_path)
    pd.DataFrame(
        {
            "video_id": ["v1", "v2", "v3", "v4"],
            "recording_timestamp": [
                "2026-09-02 07:00:00",
                "2026-09-02 14:00:00",
                "2026-09-03 07:00:00",
                "2026-09-03 14:00:00",
            ],
        }
    ).to_csv(tmp_path / "video_manifest.csv", index=False)
    scores = tmp_path / "scores.csv"
    pd.DataFrame(
        {
            "tracklet_id": ["t1", "t1", "t2", "t2"],
            "frame_idx": [1, 2, 1, 2],
            "arch_score": [0.2, 0.4, 0.5, 0.7],
            "model_name": ["kp", "kp", "kp", "kp"],
            "model_version": ["v1", "v1", "v1", "v1"],
        }
    ).to_csv(scores, index=False)
    output = import_posture_scores(tmp_path, scores)
    assert output.exists()
    observations = load_posture_observations(tmp_path)
    assert observations["cow_id"].tolist() == ["COW_0001", "COW_0001"]
    assert observations["arch_score_mean"].round(2).tolist() == [0.3, 0.6]
    assert observations.loc[0, "observed_at"] == "2026-09-02 07:00:01"


def test_same_pair_merges_complete_existing_identity_groups(tmp_path: Path):
    _identity_run(tmp_path)
    labels = load_identity_labels(tmp_path)
    labels, _ = assign_tracklets(labels, ["t1", "t2"], "cow22")
    labels, _ = assign_tracklets(labels, ["t3", "t4"], "cow23")
    save_identity_labels(labels, tmp_path)

    merged, target = accept_pair_as_same(tmp_path, labels, "t2", "t3", similarity=0.98)

    assert target == "COW_0022"
    assert set(merged["cow_id"]) == {"COW_0022"}
    assert identity_catalog(merged).to_dict("records") == [
        {"cow_id": "COW_0022", "tracklets": 4, "videos": 4, "sessions": 4}
    ]
    reviews = load_pair_reviews(tmp_path)
    assert reviews.loc[0, "decision"] == "same"
    assert reviews.loc[0, "cow_id"] == "COW_0022"
    assert "COW_0023" in reviews.loc[0, "notes"]


def test_label_audit_quarantines_different_pair_inside_identity(tmp_path: Path):
    _identity_run(tmp_path)
    labels = load_identity_labels(tmp_path)
    labels, _ = assign_tracklets(labels, ["t1", "t2"], "cow1")
    labels, _ = assign_tracklets(labels, ["t3", "t4"], "cow2")
    save_identity_labels(labels, tmp_path)
    save_pair_review(tmp_path, "t1", "t2", "different")
    outputs = audit_labels(tmp_path)
    audit = pd.read_csv(outputs["audit"])
    safe = pd.read_csv(outputs["safe_labels"])
    assert audit.set_index("cow_id").loc["COW_0001", "status"] == "conflict"
    assert not safe.loc[safe["cow_id"].eq("COW_0001"), "training_eligible"].any()
    assert safe.loc[safe["cow_id"].eq("COW_0002"), "training_eligible"].all()


def test_transitive_merge_respects_previous_different_decision(tmp_path: Path):
    _identity_run(tmp_path)
    labels = load_identity_labels(tmp_path)
    labels, _ = assign_tracklets(labels, ["t1", "t2"], "cow22")
    labels, _ = assign_tracklets(labels, ["t3", "t4"], "cow23")
    save_identity_labels(labels, tmp_path)
    save_pair_review(tmp_path, "t1", "t4", "different")
    with pytest.raises(ValueError, match="farklı inek"):
        accept_pair_as_same(tmp_path, labels, "t2", "t3")


def test_auto_confirm_dual_signal_requires_mutual_top1_and_both_thresholds(tmp_path: Path):
    _identity_run(tmp_path)
    labels = load_identity_labels(tmp_path)
    candidate_pairs = pd.DataFrame(
        [
            # t1 <-> t2: mutual top-1, both signals agree -> auto-confirm
            {"query_tracklet_id": "t1", "rank": 1, "candidate_tracklet_id": "t2", "cosine_similarity": 0.97, "geometry_similarity": 0.8},
            {"query_tracklet_id": "t2", "rank": 1, "candidate_tracklet_id": "t1", "cosine_similarity": 0.97, "geometry_similarity": 0.8},
            # t3 -> t4, but t4's real top-1 is t1: not mutual -> leave for manual review
            {"query_tracklet_id": "t3", "rank": 1, "candidate_tracklet_id": "t4", "cosine_similarity": 0.97, "geometry_similarity": 0.8},
            {"query_tracklet_id": "t4", "rank": 1, "candidate_tracklet_id": "t1", "cosine_similarity": 0.9, "geometry_similarity": 0.8},
        ]
    )
    updated, applied = auto_confirm_dual_signal(tmp_path, labels, candidate_pairs)
    assert len(applied) == 1
    assert {applied[0]["left_tracklet_id"], applied[0]["right_tracklet_id"]} == {"t1", "t2"}
    assigned = updated.set_index("tracklet_id")["cow_id"]
    assert assigned["t1"] == assigned["t2"] != ""
    assert assigned["t3"] == "" and assigned["t4"] == ""
    confirmed_rows = updated.loc[updated["tracklet_id"].isin(["t1", "t2"])]
    assert (confirmed_rows["assignment_source"] == "auto_dual_signal").all()


def test_auto_confirm_dual_signal_blocks_on_low_geometry(tmp_path: Path):
    _identity_run(tmp_path)
    labels = load_identity_labels(tmp_path)
    candidate_pairs = pd.DataFrame(
        [
            {"query_tracklet_id": "t1", "rank": 1, "candidate_tracklet_id": "t2", "cosine_similarity": 0.97, "geometry_similarity": 0.1},
            {"query_tracklet_id": "t2", "rank": 1, "candidate_tracklet_id": "t1", "cosine_similarity": 0.97, "geometry_similarity": 0.1},
        ]
    )
    _, applied = auto_confirm_dual_signal(tmp_path, labels, candidate_pairs)
    assert applied == []


def test_auto_confirm_dual_signal_respects_prior_different_decision(tmp_path: Path):
    _identity_run(tmp_path)
    labels = load_identity_labels(tmp_path)
    save_pair_review(tmp_path, "t1", "t2", "different")
    candidate_pairs = pd.DataFrame(
        [
            {"query_tracklet_id": "t1", "rank": 1, "candidate_tracklet_id": "t2", "cosine_similarity": 0.97, "geometry_similarity": 0.8},
            {"query_tracklet_id": "t2", "rank": 1, "candidate_tracklet_id": "t1", "cosine_similarity": 0.97, "geometry_similarity": 0.8},
        ]
    )
    _, applied = auto_confirm_dual_signal(tmp_path, labels, candidate_pairs)
    assert applied == []


def test_auto_confirm_dual_signal_respects_gallery_rejection(tmp_path: Path):
    _identity_run(tmp_path)
    labels = load_identity_labels(tmp_path)
    labels, _ = assign_tracklets(labels, ["t1"], "cow1")
    save_identity_labels(labels, tmp_path)
    reject_identity_candidate(tmp_path, "COW_0001", "t2")
    candidate_pairs = pd.DataFrame(
        [
            {"query_tracklet_id": "t1", "rank": 1, "candidate_tracklet_id": "t2", "cosine_similarity": 0.97, "geometry_similarity": 0.8},
            {"query_tracklet_id": "t2", "rank": 1, "candidate_tracklet_id": "t1", "cosine_similarity": 0.97, "geometry_similarity": 0.8},
        ]
    )
    _, applied = auto_confirm_dual_signal(tmp_path, labels, candidate_pairs)
    assert applied == []


def test_rejected_identity_candidate_is_not_suggested_again(tmp_path: Path):
    _identity_run(tmp_path)
    labels = load_identity_labels(tmp_path)
    labels, _ = assign_tracklets(labels, ["t1", "t2"], "cow1")
    save_identity_labels(labels, tmp_path)
    assert "t4" in identity_candidates(tmp_path, labels, "cow1", top_k=4)["tracklet_id"].tolist()
    reject_identity_candidate(tmp_path, "cow1", "t4")
    assert "t4" not in identity_candidates(tmp_path, labels, "cow1", top_k=4)["tracklet_id"].tolist()


def test_cross_session_retrieval_evaluation(tmp_path: Path):
    _identity_run(tmp_path)
    np.savez_compressed(
        tmp_path / "track_embeddings.npz",
        tracklet_ids=np.asarray(["t1", "t2", "t3", "t4"]),
        embeddings=np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]], dtype=np.float32),
        session_ids=np.asarray(["s1", "s2", "s1", "s2"]),
        video_ids=np.asarray(["v1", "v2", "v3", "v4"]),
        backend=np.asarray(["test"]),
        checkpoint=np.asarray(["test"]),
    )
    labels = load_identity_labels(tmp_path)
    labels, _ = assign_tracklets(labels, ["t1", "t2"], "cow1")
    labels, _ = assign_tracklets(labels, ["t3", "t4"], "cow2")
    save_identity_labels(labels, tmp_path)
    result = evaluate_embeddings(tmp_path, tmp_path / "labels.csv")
    assert result["queries_with_cross_session_positive"] == 4
    assert result["top1"] == 1.0
    assert result["top5"] == 1.0
    assert result["mAP"] == 1.0
