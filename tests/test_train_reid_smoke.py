"""End-to-end smoke test for train_reid with the new stride_layout/
freeze_backbone options and retrieval-based checkpoint selection. Uses the
real local OpenCows2020 checkpoint (fully offline, no network download) so
this exercises the actual code path the real ablation runs use, not a
network-dependent ImageNet-download path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from cow_reid.config import load_config
from cow_reid.training import train_reid

PRETRAINED = Path(__file__).resolve().parent.parent / "weights" / "opencows2020_softmaxrtl.pkl"


def _write_crop(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color=(100, 120, 140)).save(path)


@pytest.mark.skipif(not PRETRAINED.exists(), reason="OpenCows2020 pretrained checkpoint not downloaded locally")
def test_train_reid_with_opencows_stride_and_frozen_backbone(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    tracklets = []
    frames = []
    labels = []
    for cow_index, cow_id in enumerate(("COW_TEST_A", "COW_TEST_B")):
        for video_index, (video_id, recording_start) in enumerate(
            (("v1", "2026-08-01 08:00:00"), ("v2", "2026-08-02 08:00:00"))
        ):
            tracklet_id = f"{cow_id}_{video_id}"
            crop_path = run_dir / "crops" / f"{tracklet_id}.jpg"
            _write_crop(crop_path)
            tracklets.append({"tracklet_id": tracklet_id, "video_id": video_id, "session_id": video_id})
            frames.append({"tracklet_id": tracklet_id, "torso_path": str(crop_path)})
            labels.append({"tracklet_id": tracklet_id, "cow_id": cow_id, "confirmed": True})

    pd.DataFrame(tracklets).to_csv(run_dir / "tracklets.csv", index=False)
    pd.DataFrame(frames).to_csv(run_dir / "frames.csv", index=False)
    pd.DataFrame(
        [
            {"video_id": "v1", "recording_start": "2026-08-01 08:00:00", "recording_end": "2026-08-01 08:05:00"},
            {"video_id": "v2", "recording_start": "2026-08-02 08:00:00", "recording_end": "2026-08-02 08:05:00"},
        ]
    ).to_csv(run_dir / "video_manifest.csv", index=False)
    labels_path = run_dir / "labels.csv"
    pd.DataFrame(labels).to_csv(labels_path, index=False)

    config = load_config(None)
    config["training"]["epochs"] = 1
    config["training"]["device"] = "cpu"
    config["training"]["stride_layout"] = "opencows"
    config["training"]["freeze_backbone"] = True
    config["training"]["embedding_dim"] = 8
    config["splitting"]["validation_dates"] = ["2026-08-02"]

    output = train_reid(run_dir, labels_path, config, output_path=tmp_path / "checkpoint.pt")

    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["stride_layout"] == "opencows"
    assert payload["freeze_backbone"] is True
    assert payload["best_retrieval_top1"] is not None
    assert payload["best_retrieval_queries"] >= 1

    import json

    summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["stride_layout"] == "opencows"
    assert summary["freeze_backbone"] is True
    assert summary["best_retrieval_top1"] is not None
    assert len(summary["history"]) == 1
    assert "retrieval_top1" in summary["history"][0]
