"""Regression tests for the OpenCows-vs-torchvision ResNet50 bottleneck
stride placement. Tensor shapes are identical between the two layouts (only
the stride attribute differs), so a checkpoint trained with one layout would
silently "load" into the other with zero errors and produce a different,
incompatible forward computation -- these tests pin down both the patch
itself and its round-trip through a saved checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import torch

from cow_reid.metric_model import build_metric_resnet50, load_metric_checkpoint

DOWNSAMPLING_LAYERS = ("layer2", "layer3", "layer4")


def _strides(model) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    return {
        name: (tuple(getattr(model.backbone, name)[0].conv1.stride), tuple(getattr(model.backbone, name)[0].conv2.stride))
        for name in ("layer1", *DOWNSAMPLING_LAYERS)
    }


def test_torchvision_stride_layout_is_the_untouched_default():
    model = build_metric_resnet50(num_classes=2, embedding_dim=8, imagenet=False, stride_layout="torchvision")
    strides = _strides(model)
    assert strides["layer1"] == ((1, 1), (1, 1))
    for name in DOWNSAMPLING_LAYERS:
        assert strides[name] == ((1, 1), (2, 2)), f"{name} should keep stride on conv2 (torchvision default)"


def test_opencows_stride_layout_moves_stride_to_conv1():
    model = build_metric_resnet50(num_classes=2, embedding_dim=8, imagenet=False, stride_layout="opencows")
    strides = _strides(model)
    # layer1 has no spatial downsampling in either layout -- must stay untouched.
    assert strides["layer1"] == ((1, 1), (1, 1))
    for name in DOWNSAMPLING_LAYERS:
        assert strides[name] == ((2, 2), (1, 1)), f"{name} should have stride moved onto conv1 (opencows layout)"


def test_opencows_stride_layout_does_not_change_downsample_shortcut_stride():
    torchvision_model = build_metric_resnet50(num_classes=2, embedding_dim=8, imagenet=False, stride_layout="torchvision")
    opencows_model = build_metric_resnet50(num_classes=2, embedding_dim=8, imagenet=False, stride_layout="opencows")
    for name in DOWNSAMPLING_LAYERS:
        torchvision_shortcut = getattr(torchvision_model.backbone, name)[0].downsample[0].stride
        opencows_shortcut = getattr(opencows_model.backbone, name)[0].downsample[0].stride
        assert tuple(torchvision_shortcut) == tuple(opencows_shortcut) == (2, 2)


def test_invalid_stride_layout_rejected():
    import pytest

    with pytest.raises(ValueError):
        build_metric_resnet50(num_classes=2, embedding_dim=8, imagenet=False, stride_layout="bogus")


def test_load_metric_checkpoint_round_trips_stride_layout(tmp_path: Path):
    model = build_metric_resnet50(num_classes=2, embedding_dim=8, imagenet=False, stride_layout="opencows")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_name": "metric_resnet50",
            "state_dict": model.state_dict(),
            "class_names": ["COW_0001", "COW_0002"],
            "embedding_dim": 8,
            "stride_layout": "opencows",
        },
        checkpoint,
    )
    loaded, payload = load_metric_checkpoint(checkpoint, device="cpu")
    assert payload["stride_layout"] == "opencows"
    assert _strides(loaded) == _strides(model)
    for name in DOWNSAMPLING_LAYERS:
        assert _strides(loaded)[name] == ((2, 2), (1, 1))


def test_load_metric_checkpoint_defaults_to_torchvision_layout_when_absent(tmp_path: Path):
    """A checkpoint saved before stride_layout existed (no such key) must
    reconstruct with the torchvision layout -- the layout every pre-existing
    checkpoint in this repo was actually trained with."""
    model = build_metric_resnet50(num_classes=2, embedding_dim=8, imagenet=False, stride_layout="torchvision")
    checkpoint = tmp_path / "legacy_checkpoint.pt"
    torch.save(
        {
            "model_name": "metric_resnet50",
            "state_dict": model.state_dict(),
            "class_names": ["COW_0001", "COW_0002"],
            "embedding_dim": 8,
        },
        checkpoint,
    )
    loaded, payload = load_metric_checkpoint(checkpoint, device="cpu")
    assert payload.get("stride_layout", "torchvision") == "torchvision"
    for name in DOWNSAMPLING_LAYERS:
        assert _strides(loaded)[name] == ((1, 1), (2, 2))
