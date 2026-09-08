from __future__ import annotations

from pathlib import Path
from typing import Any


def _safe_torch_load(path: str | Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload: {path}")
    return payload


def _apply_opencows_stride_layout(backbone) -> None:
    """Move each downsampling bottleneck's stride from the 3x3 conv (conv2,
    torchvision's default "v1.5" placement) to the 1x1 conv (conv1, the
    placement the public OpenCows2020 checkpoint was trained with). Tensor
    shapes are unaffected either way -- only the stride attribute differs --
    so this must never be applied inconsistently between a checkpoint's
    training run and its later loading; see build_metric_resnet50's
    stride_layout and load_metric_checkpoint.
    """
    for layer in (backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4):
        block = layer[0]
        if tuple(block.conv2.stride) != (1, 1):
            block.conv1.stride, block.conv2.stride = block.conv2.stride, block.conv1.stride


def build_metric_resnet50(
    num_classes: int,
    embedding_dim: int = 128,
    imagenet: bool = True,
    stride_layout: str = "torchvision",
):
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    if stride_layout not in ("torchvision", "opencows"):
        raise ValueError(f"Unsupported stride_layout: {stride_layout!r}")

    class MetricResNet50(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weights = ResNet50_Weights.DEFAULT if imagenet else None
            self.backbone = resnet50(weights=weights)
            if stride_layout == "opencows":
                _apply_opencows_stride_layout(self.backbone)
            feature_dim = int(self.backbone.fc.in_features)
            self.backbone.fc = torch.nn.Identity()
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, 1000),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(1000, embedding_dim),
            )
            self.classifier = torch.nn.Linear(embedding_dim, num_classes)

        def forward(self, images):
            features = self.backbone(images)
            embedding = torch.nn.functional.normalize(self.projection(features), dim=1)
            return embedding, self.classifier(embedding)

        def embed(self, images):
            features = self.backbone(images)
            return torch.nn.functional.normalize(self.projection(features), dim=1)

    return MetricResNet50()


def load_opencows2020_initialization(model, checkpoint_path: str | Path) -> dict[str, int | str]:
    """Map the public OpenCows2020 ResNet-50 Softmax+RTL state into our model."""

    payload = _safe_torch_load(checkpoint_path)
    state = payload.get("model_state", payload.get("state_dict", payload))
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model state: {checkpoint_path}")
    destination = model.state_dict()
    mapped: dict[str, Any] = {}
    skipped = 0
    for key, value in state.items():
        target = None
        if key.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")):
            target = f"backbone.{key}"
        elif key.startswith("fc."):
            target = f"projection.0.{key.removeprefix('fc.')}"
        elif key.startswith("fc_embedding."):
            target = f"projection.2.{key.removeprefix('fc_embedding.')}"
        if target and target in destination and getattr(value, "shape", None) == destination[target].shape:
            mapped[target] = value
        else:
            skipped += 1
    if not mapped:
        raise ValueError(f"No compatible OpenCows2020 tensors found in {checkpoint_path}")
    model.load_state_dict(mapped, strict=False)
    return {"loaded_tensors": len(mapped), "skipped_tensors": skipped, "source": str(checkpoint_path)}


def infer_checkpoint_kind(checkpoint_path: str | Path) -> str:
    """Best-effort identification of a checkpoint's format, used to fail fast on a
    backend/checkpoint mismatch before running the (expensive) extract/embed
    pipeline stages, rather than deep inside a key-remapping loader."""
    try:
        payload = _safe_torch_load(checkpoint_path)
    except Exception:
        return "unknown"
    if payload.get("model_name") == "metric_resnet50":
        return "metric_resnet50"
    return "unknown"


def load_metric_checkpoint(checkpoint_path: str | Path, device: str = "cpu"):
    payload = _safe_torch_load(checkpoint_path)
    if payload.get("model_name") != "metric_resnet50":
        raise ValueError(f"Not a v0.3 metric checkpoint: {checkpoint_path}")
    class_names = [str(value) for value in payload.get("class_names", [])]
    embedding_dim = int(payload.get("embedding_dim", 128))
    stride_layout = str(payload.get("stride_layout", "torchvision"))
    model = build_metric_resnet50(
        max(1, len(class_names)), embedding_dim=embedding_dim, imagenet=False, stride_layout=stride_layout
    )
    model.load_state_dict(payload["state_dict"])
    model.eval().to(device)
    return model, payload


class LetterboxSquare:
    def __init__(self, size: int = 224) -> None:
        self.size = int(size)

    def __call__(self, image):
        from PIL import Image

        source = image.convert("RGB")
        ratio = self.size / max(source.size)
        resized = source.resize(
            (max(1, int(source.width * ratio)), max(1, int(source.height * ratio))),
            Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (self.size, self.size))
        canvas.paste(resized, ((self.size - resized.width) // 2, (self.size - resized.height) // 2))
        return canvas


def build_metric_transform(profile: str, training: bool = False, image_size: int = 224):
    from torchvision import transforms

    augment = []
    if training:
        # No horizontal/vertical flip here: coat-pattern position is the identity
        # signal for this task, and a cow's left/right flank patterns already differ
        # on the same animal (see docs). Training the model to treat a mirrored
        # pattern as "the same" teaches it to confuse animals whose spots are
        # simply transposed (e.g. bottom-left/top-right vs top-left/bottom-right).
        augment = [
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.06),
        ]
    if profile == "opencows2020_legacy":
        return transforms.Compose(
            [
                *augment,
                LetterboxSquare(image_size),
                transforms.PILToTensor(),
                transforms.Lambda(lambda tensor: tensor.float()),
            ]
        )
    return transforms.Compose(
        [
            *augment,
            transforms.Resize((256, 256)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
