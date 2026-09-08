"""Regression tests for the embedding/gallery version contract and for
tracklet_reviews.csv enforcement at the embedding/gallery layer.

Before this, embedding/gallery compatibility was checked by comparing the
checkpoint *path string* recorded in track_embeddings.npz, so overwriting a
checkpoint file in place (this repo's own retraining workflow) would silently
pass the check even though the embedding spaces were now incompatible. And
tracklet_reviews.csv (human "this tracklet is broken" decisions) was
write-only: nothing filtered a flagged tracklet out of embeddings or the
gallery.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cow_reid.config import load_config
from cow_reid.embeddings import build_track_embeddings
from cow_reid.gallery import build_gallery, identify_run
from cow_reid.review_state import record_tracklet_state


def _write_crop(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(120, 90, 60)).save(path)


def _make_run(run_dir: Path, tracklet_ids: list[str], crop_version: str = "extract_v1_masked_torso") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    track_rows = []
    frame_rows = []
    for index, tracklet_id in enumerate(tracklet_ids):
        crop_path = run_dir / "crops" / f"{tracklet_id}.jpg"
        _write_crop(crop_path)
        track_rows.append(
            {
                "tracklet_id": tracklet_id,
                "session_id": f"s{index}",
                "video_id": f"v{index}",
                "valid": True,
                "contact_path": "",
                "start_s": float(index),
                "crop_version": crop_version,
            }
        )
        frame_rows.append({"tracklet_id": tracklet_id, "rank": 0, "quality": 0.8, "torso_path": str(crop_path)})
    pd.DataFrame(track_rows).to_csv(run_dir / "tracklets.csv", index=False)
    pd.DataFrame(frame_rows).to_csv(run_dir / "frames.csv", index=False)


def test_build_track_embeddings_records_version_contract(tmp_path: Path):
    _make_run(tmp_path, ["t1", "t2"])
    config = load_config(None)
    output, metadata = build_track_embeddings(tmp_path, config, backend_override="hist")
    with np.load(output, allow_pickle=False) as data:
        assert str(data["crop_version"][0]) == "extract_v1_masked_torso"
        assert str(data["preprocessing_profile"][0]) == "hist_v1"
        assert str(data["checkpoint_sha256"][0]) == ""
        assert str(data["embedding_schema_version"][0]).endswith("l2norm_v1")
    assert set(metadata["crop_version"]) == {"extract_v1_masked_torso"}
    assert set(metadata["preprocessing_profile"]) == {"hist_v1"}


def test_build_track_embeddings_excludes_flagged_tracklets(tmp_path: Path):
    _make_run(tmp_path, ["t1", "t2"])
    record_tracklet_state(tmp_path, "t1", "tracklet_error")
    config = load_config(None)
    output, metadata = build_track_embeddings(tmp_path, config, backend_override="hist")
    assert "t1" not in set(metadata["tracklet_id"])
    assert "t2" in set(metadata["tracklet_id"])


def test_build_gallery_excludes_flagged_tracklets_even_if_confirmed(tmp_path: Path):
    _make_run(tmp_path, ["t1", "t2"])
    config = load_config(None)
    build_track_embeddings(tmp_path, config, backend_override="hist")
    record_tracklet_state(tmp_path, "t1", "tracklet_error")
    # No "confirmed" column: exercises build_gallery's plain tracklet_id/cow_id
    # path directly, without entangling audit_labels' separate >=2-tracklets-
    # per-cow training_eligible rule.
    labels = pd.DataFrame({"tracklet_id": ["t1", "t2"], "cow_id": ["cow_a", "cow_b"]})
    labels.to_csv(tmp_path / "labels.csv", index=False)
    output = build_gallery(tmp_path, tmp_path / "labels.csv")
    with np.load(output, allow_pickle=False) as gallery:
        assert set(gallery["cow_ids"].tolist()) == {"cow_b"}


def test_identify_run_rejects_checkpoint_hash_mismatch(tmp_path: Path):
    gallery_dir = tmp_path / "gallery_run"
    query_dir = tmp_path / "query_run"
    for run_dir in (gallery_dir, query_dir):
        run_dir.mkdir()
    ids = np.asarray(["t1", "t2"])
    np.savez_compressed(
        gallery_dir / "gallery_source.npz",
        tracklet_ids=ids,
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        session_ids=np.asarray(["s1", "s2"]),
        video_ids=np.asarray(["v1", "v2"]),
        backend=np.asarray(["metric"]),
        checkpoint=np.asarray(["weights/farm_metric_resnet50.pt"]),
        checkpoint_sha256=np.asarray(["aaa111"]),
        preprocessing_profile=np.asarray(["opencows2020_legacy"]),
        crop_version=np.asarray(["extract_v1_masked_torso"]),
        embedding_schema_version=np.asarray(["128d_l2norm_v1"]),
    )
    labels = pd.DataFrame({"tracklet_id": ["t1", "t2"], "cow_id": ["cow_a", "cow_b"]})
    labels.to_csv(gallery_dir / "labels.csv", index=False)
    # build_gallery reads track_embeddings.npz by convention; point it at our
    # synthetic file by writing it under that name instead.
    (gallery_dir / "gallery_source.npz").rename(gallery_dir / "track_embeddings.npz")
    gallery_path = build_gallery(gallery_dir, gallery_dir / "labels.csv")

    # Same path/backend string, but the checkpoint file was retrained (different
    # content hash) -- the old path-string check would have passed this.
    np.savez_compressed(
        query_dir / "track_embeddings.npz",
        tracklet_ids=np.asarray(["q1"]),
        embeddings=np.asarray([[0.9, 0.1]], dtype=np.float32),
        session_ids=np.asarray(["s3"]),
        video_ids=np.asarray(["v3"]),
        backend=np.asarray(["metric"]),
        checkpoint=np.asarray(["weights/farm_metric_resnet50.pt"]),
        checkpoint_sha256=np.asarray(["bbb222"]),
        preprocessing_profile=np.asarray(["opencows2020_legacy"]),
        crop_version=np.asarray(["extract_v1_masked_torso"]),
        embedding_schema_version=np.asarray(["128d_l2norm_v1"]),
    )
    config = load_config(None)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        identify_run(query_dir, gallery_path, config)


def test_build_metric_checkpoint_profile_includes_stride_layout(tmp_path: Path):
    import torch

    from cow_reid.embeddings import _build_metric_checkpoint
    from cow_reid.metric_model import build_metric_resnet50

    model = build_metric_resnet50(num_classes=2, embedding_dim=8, imagenet=False, stride_layout="opencows")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_name": "metric_resnet50",
            "state_dict": model.state_dict(),
            "class_names": ["COW_0001", "COW_0002"],
            "embedding_dim": 8,
            "input_profile": "opencows2020_legacy",
            "stride_layout": "opencows",
        },
        checkpoint,
    )
    _, dim, profile = _build_metric_checkpoint("cpu", str(checkpoint))
    assert profile == "opencows2020_legacy+stride_opencows"
    assert dim == 8


def test_identify_run_rejects_stride_layout_mismatch_even_with_matching_hash(tmp_path: Path):
    """Independent of checkpoint_sha256: two embedding sets that agree on
    every other version field but disagree on stride layout (folded into
    preprocessing_profile) must still be rejected -- this is the field that
    catches the one mismatch state isn't part of, so a hash coincidence
    can't paper over it."""
    gallery_dir = tmp_path / "gallery_run"
    query_dir = tmp_path / "query_run"
    for run_dir in (gallery_dir, query_dir):
        run_dir.mkdir()
    common_fields = dict(
        backend=np.asarray(["metric"]),
        checkpoint=np.asarray(["weights/farm_metric_resnet50.pt"]),
        checkpoint_sha256=np.asarray(["same-hash-both-sides"]),
        crop_version=np.asarray(["extract_v1_masked_torso"]),
        embedding_schema_version=np.asarray(["128d_l2norm_v1"]),
    )
    np.savez_compressed(
        gallery_dir / "track_embeddings.npz",
        tracklet_ids=np.asarray(["t1", "t2"]),
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        session_ids=np.asarray(["s1", "s2"]),
        video_ids=np.asarray(["v1", "v2"]),
        preprocessing_profile=np.asarray(["opencows2020_legacy+stride_opencows"]),
        **common_fields,
    )
    labels = pd.DataFrame({"tracklet_id": ["t1", "t2"], "cow_id": ["cow_a", "cow_b"]})
    labels.to_csv(gallery_dir / "labels.csv", index=False)
    gallery_path = build_gallery(gallery_dir, gallery_dir / "labels.csv")

    np.savez_compressed(
        query_dir / "track_embeddings.npz",
        tracklet_ids=np.asarray(["q1"]),
        embeddings=np.asarray([[0.9, 0.1]], dtype=np.float32),
        session_ids=np.asarray(["s3"]),
        video_ids=np.asarray(["v3"]),
        preprocessing_profile=np.asarray(["opencows2020_legacy+stride_torchvision"]),
        **common_fields,
    )
    config = load_config(None)
    with pytest.raises(ValueError, match="preprocessing_profile"):
        identify_run(query_dir, gallery_path, config)


def test_identify_run_never_writes_labels_csv(tmp_path: Path):
    """Automatic identification predictions must stay quarantined in
    identity_predictions.csv/identity_summary.json -- they must never be
    merged into labels.csv (the file the gallery and training treat as
    human-confirmed ground truth) without an explicit human action."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    np.savez_compressed(
        run_dir / "track_embeddings.npz",
        tracklet_ids=np.asarray(["q1"]),
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        session_ids=np.asarray(["s1"]),
        video_ids=np.asarray(["v1"]),
        backend=np.asarray(["test"]),
    )
    labels_path = run_dir / "labels.csv"
    original_labels = pd.DataFrame({"tracklet_id": ["other"], "cow_id": ["cow_z"], "confirmed": [True]})
    original_labels.to_csv(labels_path, index=False)
    gallery_path = tmp_path / "gallery.npz"
    np.savez_compressed(
        gallery_path,
        cow_ids=np.asarray(["cow_a"]),
        prototypes=np.asarray([[0.99, 0.01]], dtype=np.float32),
        backend=np.asarray(["test"]),
    )
    config = load_config(None)
    identify_run(run_dir, gallery_path, config)
    assert (run_dir / "identity_predictions.csv").exists()
    after = pd.read_csv(labels_path)
    pd.testing.assert_frame_equal(after, original_labels)
