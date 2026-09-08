"""Regression tests for the overlap/date-aware split machinery.

These reproduce, at a small synthetic scale, the exact leak found in the live
repo: two recordings of the same physical milking session (one with a
corrected timestamp, one where OCR originally failed) were treated as two
independent "sessions" by the old lexicographic-session splitter, so a
checkpoint's reported validation accuracy was measured almost entirely
against a duplicate copy of its own training data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cow_reid.audit import inspect_label_consistency
from cow_reid.evaluation import held_out_date_split
from cow_reid.manifest import build_canonical_manifest
from cow_reid.overlap import assign_overlap_groups
from cow_reid.review_state import load_excluded_tracklets, record_tracklet_state
from cow_reid.training import _auto_last_day_split, _derive_tracklet_split


def test_assign_overlap_groups_flags_missing_timestamp_as_unverified():
    manifest = pd.DataFrame(
        {
            "video_id": ["v1", "v2"],
            "camera_id": ["cam1", "cam1"],
            "recording_start": ["2026-08-13 15:47:04", None],
            "recording_end": ["2026-08-13 15:52:04", None],
        }
    )
    grouped = assign_overlap_groups(manifest)
    assert grouped.set_index("video_id").loc["v1", "overlap_unverified"] == False  # noqa: E712
    assert grouped.set_index("video_id").loc["v2", "overlap_unverified"] == True  # noqa: E712
    # Without a real timestamp, v2 cannot be tested for overlap against
    # anything, so it still falls out as its own singleton group -- the flag,
    # not a different grouping, is what tells callers not to trust that.
    assert grouped.set_index("video_id").loc["v1", "overlap_group_id"] != grouped.set_index("video_id").loc["v2", "overlap_group_id"]


def test_build_canonical_manifest_reunites_recording_once_timestamp_is_corrected():
    # This is the exact real-world shape: vid_e2b59d770f3b's OCR timestamp
    # worked, vid_eef4a44f4b99's did not (raw NaN), and a human later supplied
    # the real, overlapping recording_start/recording_end for the second one.
    raw = pd.DataFrame(
        {
            "video_id": ["vid_a", "vid_b"],
            "camera_id": ["cam1", "cam1"],
            "recording_start": ["2026-08-13 15:47:04", None],
            "recording_end": ["2026-08-13 15:52:04", None],
            "timestamp_source": ["camera_ocr", "unknown"],
        }
    )
    corrections = pd.DataFrame(
        {
            "video_id": ["vid_b"],
            "recording_start": ["2026-08-13 15:47:21"],
            "recording_end": ["2026-08-13 15:52:21"],
            "timestamp_source": ["manual"],
            "timestamp_confidence": [1.0],
        }
    )
    canonical = build_canonical_manifest(raw, [corrections])
    by_id = canonical.set_index("video_id")
    assert by_id.loc["vid_a", "overlap_group_id"] == by_id.loc["vid_b", "overlap_group_id"]
    assert not bool(by_id.loc["vid_b", "overlap_unverified"])


def test_build_canonical_manifest_rejects_unknown_video_id_in_corrections():
    raw = pd.DataFrame({"video_id": ["vid_a"], "camera_id": ["cam1"], "recording_start": ["2026-08-13 00:00:00"], "recording_end": ["2026-08-13 00:05:00"]})
    corrections = pd.DataFrame({"video_id": ["vid_typo"], "recording_start": ["2026-08-13 00:00:00"], "recording_end": ["2026-08-13 00:05:00"], "timestamp_source": ["manual"]})
    with pytest.raises(ValueError):
        build_canonical_manifest(raw, [corrections])


def test_held_out_date_split_never_lets_one_overlap_group_cross_splits():
    # Same overlap group, but a recording that straddles midnight so its two
    # members fall on different calendar dates -- the naive per-date
    # assignment below would otherwise put them in different splits.
    manifest = pd.DataFrame(
        {
            "video_id": ["v1", "v2"],
            "camera_id": ["cam1", "cam1"],
            "recording_start": ["2026-08-13 23:58:00", "2026-08-14 00:03:00"],
            "recording_end": ["2026-08-14 00:05:00", "2026-08-14 00:08:00"],
        }
    )
    grouped = assign_overlap_groups(manifest)
    assert grouped["overlap_group_id"].nunique() == 1
    with pytest.raises(ValueError, match="crosses dataset splits"):
        held_out_date_split(grouped, test_dates=[], validation_dates=["2026-08-14"])


def test_held_out_date_split_forces_unverified_rows_to_train():
    manifest = pd.DataFrame(
        {
            "video_id": ["v1", "v2", "v3"],
            "camera_id": ["cam1", "cam1", "cam1"],
            "recording_start": ["2026-08-01 08:00:00", "2026-08-05 08:00:00", None],
            "recording_end": ["2026-08-01 08:05:00", "2026-08-05 08:05:00", None],
        }
    )
    grouped = assign_overlap_groups(manifest)
    result = held_out_date_split(grouped, test_dates=[], validation_dates=["2026-08-05"])
    by_id = result.set_index("video_id")
    assert by_id.loc["v1", "split"] == "train"
    assert by_id.loc["v2", "split"] == "validation"
    assert by_id.loc["v3", "split"] == "train"  # unverified, never eligible as a held-out day


def test_overlap_group_never_split_across_train_and_validation():
    """Reproduces the historical bug shape directly: a cow whose only two
    labelled tracklets come from a duplicate-recorded physical passage must
    never end up with one tracklet in train and the other in validation."""
    manifest = pd.DataFrame(
        {
            "video_id": ["vA", "vB"],
            "camera_id": ["cam1", "cam1"],
            "recording_start": ["2026-08-13 15:47:04", "2026-08-13 15:47:21"],
            "recording_end": ["2026-08-13 15:52:04", "2026-08-13 15:52:21"],
        }
    )
    grouped = assign_overlap_groups(manifest)
    assert grouped["overlap_group_id"].nunique() == 1

    tracks = pd.DataFrame({"tracklet_id": ["a1", "b1"], "session_id": ["vA", "vB"], "video_id": ["vA", "vB"]})
    labels = pd.DataFrame({"tracklet_id": ["a1", "b1"], "cow_id": ["cow_x", "cow_x"], "confirmed": [True, True]})

    video_split = _auto_last_day_split(grouped)
    assert video_split["split"].nunique() == 1  # one calendar day -> one assignment for the whole group

    # A single physical day cannot honestly provide both a train and a
    # validation example for this cow -- refusing to train on it is the safe
    # outcome, not silently manufacturing a validation split from a duplicate
    # of the same passage.
    with pytest.raises(RuntimeError):
        _derive_tracklet_split(labels, tracks, video_split)


def test_derive_tracklet_split_allows_genuinely_different_days():
    manifest = pd.DataFrame(
        {
            "video_id": ["v1", "v2"],
            "camera_id": ["cam1", "cam1"],
            "recording_start": ["2026-08-01 08:00:00", "2026-08-10 08:00:00"],
            "recording_end": ["2026-08-01 08:05:00", "2026-08-10 08:05:00"],
        }
    )
    grouped = assign_overlap_groups(manifest)
    assert grouped["overlap_group_id"].nunique() == 2  # genuinely different passages

    tracks = pd.DataFrame({"tracklet_id": ["a1", "a2"], "session_id": ["v1", "v2"], "video_id": ["v1", "v2"]})
    labels = pd.DataFrame({"tracklet_id": ["a1", "a2"], "cow_id": ["cow_x", "cow_x"], "confirmed": [True, True]})

    video_split = _auto_last_day_split(grouped)  # holds out 2026-08-10 (the later day) as validation
    split = _derive_tracklet_split(labels, tracks, video_split)
    by_track = split.set_index("tracklet_id")
    assert by_track.loc["a1", "split"] == "train"
    assert by_track.loc["a2", "split"] == "val"


def test_open_set_cow_ids_are_excluded_from_train_and_validation():
    manifest = pd.DataFrame(
        {
            "video_id": ["v1", "v2"],
            "camera_id": ["cam1", "cam1"],
            "recording_start": ["2026-08-01 08:00:00", "2026-08-10 08:00:00"],
            "recording_end": ["2026-08-01 08:05:00", "2026-08-10 08:05:00"],
        }
    )
    grouped = assign_overlap_groups(manifest)
    tracks = pd.DataFrame({"tracklet_id": ["a1", "a2"], "session_id": ["v1", "v2"], "video_id": ["v1", "v2"]})
    labels = pd.DataFrame({"tracklet_id": ["a1", "a2"], "cow_id": ["cow_x", "cow_x"], "confirmed": [True, True]})
    video_split = _auto_last_day_split(grouped)

    split = _derive_tracklet_split(labels, tracks, video_split, open_set_cow_ids={"cow_x"})
    assert set(split["split"]) == {"open_set_test"}


def test_flagged_tracklets_are_excluded_from_the_split(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record_tracklet_state(run_dir, "a1", "tracklet_error")
    excluded = load_excluded_tracklets(run_dir)
    assert excluded == {"a1"}

    manifest = pd.DataFrame(
        {
            "video_id": ["v1", "v2"],
            "camera_id": ["cam1", "cam1"],
            "recording_start": ["2026-08-01 08:00:00", "2026-08-10 08:00:00"],
            "recording_end": ["2026-08-01 08:05:00", "2026-08-10 08:05:00"],
        }
    )
    grouped = assign_overlap_groups(manifest)
    tracks = pd.DataFrame({"tracklet_id": ["a1", "a2"], "session_id": ["v1", "v2"], "video_id": ["v1", "v2"]})
    labels = pd.DataFrame({"tracklet_id": ["a1", "a2"], "cow_id": ["cow_x", "cow_x"], "confirmed": [True, True]})
    video_split = _auto_last_day_split(grouped)

    with pytest.raises(RuntimeError):
        # a1 (tracklet_error) is excluded, leaving cow_x with only one
        # tracklet -- correctly refused rather than trained on a flagged track
        _derive_tracklet_split(labels, tracks, video_split, excluded_tracklets=excluded)


def test_audit_downgrades_cow_when_flagged_tracklet_leaves_too_few():
    labels = pd.DataFrame(
        {
            "tracklet_id": ["a1", "a2", "b1", "b2", "b3"],
            "cow_id": ["cow_a", "cow_a", "cow_b", "cow_b", "cow_b"],
            "confirmed": [True, True, True, True, True],
            "session_id": ["s1", "s2", "s1", "s2", "s3"],
        }
    )
    empty_reviews = pd.DataFrame(columns=["left_tracklet_id", "right_tracklet_id", "decision", "updated_at"])

    # Without any flag, both cows have >=2 tracklets and are "safe".
    audit, _, _ = inspect_label_consistency(labels, empty_reviews, excluded_tracklets=set())
    by_cow = audit.set_index("cow_id")
    assert bool(by_cow.loc["cow_a", "training_eligible"])
    assert bool(by_cow.loc["cow_b", "training_eligible"])

    # Flagging a1 leaves cow_a with only one usable tracklet -> not eligible.
    # cow_b still has two usable tracklets after losing b1 -> stays eligible.
    audit, _, _ = inspect_label_consistency(labels, empty_reviews, excluded_tracklets={"a1", "b1"})
    by_cow = audit.set_index("cow_id")
    assert not bool(by_cow.loc["cow_a", "training_eligible"])
    assert by_cow.loc["cow_a", "status"] == "needs_more_tracklets"
    assert bool(by_cow.loc["cow_b", "training_eligible"])
    assert int(by_cow.loc["cow_b", "excluded_flagged_tracklets"]) == 1
