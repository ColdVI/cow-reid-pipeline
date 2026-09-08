from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .config import resolve_model_path
from .metric_model import (
    build_metric_resnet50,
    build_metric_transform,
    load_metric_checkpoint,
    load_opencows2020_initialization,
)
from .review_state import load_excluded_tracklets
from .utils import bool_mask, l2_normalize, save_json, sha256_file

# Bump when the L2-normalized embedding convention itself changes (dimension
# aside) -- e.g. a different pooling/normalization scheme -- so old and new
# embedding spaces never silently compare as compatible via gallery.py.
EMBEDDING_SCHEMA_VERSION = "l2norm_v1"


def _resolve_device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _hist_feature(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, (160, 96), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(gray)
    spatial = cv2.resize(gray, (32, 16), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256]).astype(np.float32).reshape(-1)
    histogram /= max(float(histogram.sum()), 1e-6)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    gradient = cv2.resize(magnitude, (16, 8), interpolation=cv2.INTER_AREA).reshape(-1)
    gradient /= max(float(np.linalg.norm(gradient)), 1e-6)
    return l2_normalize(np.concatenate([spatial, histogram, gradient]))


def _pattern_layout_descriptor(path: str | Path, grid: tuple[int, int] = (6, 4)) -> np.ndarray:
    """Coarse, mean-centered light/dark occupancy grid for one crop.

    Deliberately not flip-invariant and not shared-background invariant beyond
    per-image mean centering: this exists only to catch pairs whose coat pattern
    is spatially transposed/mirrored (e.g. spot bottom-left vs top-left) but whose
    deep embedding still rates them as similar because pooling discards layout.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (160, 96), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cols, rows = grid
    cell_h, cell_w = binary.shape[0] // rows, binary.shape[1] // cols
    occupancy = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            cell = binary[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w]
            occupancy[row, col] = float(cell.mean()) if cell.size else 0.0
    flat = occupancy.reshape(-1)
    flat = flat - flat.mean()
    return l2_normalize(flat)


def _batched_infer(model, transform, device: str) -> Callable[[list[str]], np.ndarray]:
    import torch

    def infer(paths: list[str]) -> np.ndarray:
        tensors = []
        for path in paths:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            features = model.embed(batch).detach().cpu().numpy().astype(np.float32)
        return l2_normalize(features)

    return infer


def _build_opencows2020(device: str, checkpoint: str) -> tuple[Callable[[list[str]], np.ndarray], int, str]:
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"OpenCows2020 checkpoint not found: {checkpoint}. Run: cow-reid download-cattle-weights"
        )
    model = build_metric_resnet50(num_classes=1, embedding_dim=128, imagenet=False)
    load_opencows2020_initialization(model, path)
    model.eval().to(device)
    profile = "opencows2020_legacy"
    return _batched_infer(model, build_metric_transform(profile), device), 128, profile


def _build_metric_checkpoint(device: str, checkpoint: str) -> tuple[Callable[[list[str]], np.ndarray], int, str]:
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Fine-tuned metric checkpoint not found: {checkpoint}")
    model, payload = load_metric_checkpoint(path, device=device)
    input_profile = str(payload.get("input_profile", "opencows2020_legacy"))
    stride_layout = str(payload.get("stride_layout", "torchvision"))
    # A checkpoint trained with one stride layout silently "loads" into the
    # other with zero errors (stride isn't part of state_dict) but produces a
    # different, incompatible forward computation -- fold it into the
    # reported profile so gallery.py's version-contract check catches a
    # gallery/query pair built with mismatched stride layouts.
    reported_profile = f"{input_profile}+stride_{stride_layout}"
    return _batched_infer(model, build_metric_transform(input_profile), device), int(payload.get("embedding_dim", 128)), reported_profile


def _build_resnet18(device: str, checkpoint: str | None = None) -> tuple[Callable[[list[str]], np.ndarray], int, str]:
    """Legacy v0.2 ImageNet baseline retained for reproducing an old run."""

    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights if checkpoint is None else None)
    if checkpoint is not None:
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        if payload.get("model_name") != "resnet18" or "state_dict" not in payload:
            raise ValueError(f"Unsupported legacy Re-ID checkpoint: {checkpoint}")
        class_names = payload.get("class_names", [])
        if not class_names:
            raise ValueError(f"Checkpoint has no class_names: {checkpoint}")
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(payload["state_dict"])
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    transform = weights.transforms()

    def infer(paths: list[str]) -> np.ndarray:
        tensors = []
        for path in paths:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            features = model(batch).detach().cpu().numpy().astype(np.float32)
        return l2_normalize(features)

    return infer, 512, "imagenet_resnet18"


def _backup_embedding_artifacts(run_dir: Path) -> Path | None:
    names = ("track_embeddings.npz", "embedding_metadata.csv", "embedding_summary.json")
    existing = [run_dir / name for name in names if (run_dir / name).exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = run_dir / "embedding_backups" / stamp
    suffix = 1
    while target.exists():
        target = run_dir / "embedding_backups" / f"{stamp}_{suffix}"
        suffix += 1
    target.mkdir(parents=True)
    for source in existing:
        shutil.copy2(source, target / source.name)
    return target


def build_track_embeddings(
    run_dir: str | Path,
    config: dict[str, Any],
    backend_override: str | None = None,
) -> tuple[Path, pd.DataFrame]:
    run_dir = Path(run_dir).resolve()
    tracks = pd.read_csv(run_dir / "tracklets.csv")
    frames = pd.read_csv(run_dir / "frames.csv")
    if tracks.empty or frames.empty:
        raise RuntimeError("No extracted tracklet crops are available for embedding.")
    valid_tracks = tracks.loc[bool_mask(tracks["valid"])].copy()
    excluded_tracklets = load_excluded_tracklets(run_dir)
    if excluded_tracklets:
        valid_tracks = valid_tracks.loc[~valid_tracks["tracklet_id"].astype(str).isin(excluded_tracklets)]
    frames = frames.loc[frames["tracklet_id"].isin(valid_tracks["tracklet_id"])].copy()
    embedding_cfg = config["embedding"]
    requested_backend = backend_override or str(embedding_cfg.get("backend", "opencows2020"))
    backend = requested_backend
    device = _resolve_device(str(embedding_cfg.get("device", "auto")))
    batch_size = int(embedding_cfg.get("batch_size", 64))
    deep_infer: Callable[[list[str]], np.ndarray] | None = None
    checkpoint_value = embedding_cfg.get("checkpoint")
    checkpoint = resolve_model_path(str(checkpoint_value), config) if checkpoint_value else None
    allow_fallback = bool(embedding_cfg.get("allow_fallback", False))
    preprocessing_profile = "hist_v1"
    try:
        if backend == "opencows2020":
            if not checkpoint:
                raise FileNotFoundError("OpenCows2020 backend needs a checkpoint. Run: cow-reid download-cattle-weights")
            deep_infer, _, preprocessing_profile = _build_opencows2020(device, checkpoint)
        elif backend == "metric":
            if not checkpoint:
                raise FileNotFoundError("Metric backend needs --checkpoint pointing to a v0.3 trained model.")
            deep_infer, _, preprocessing_profile = _build_metric_checkpoint(device, checkpoint)
        elif backend == "resnet18":
            deep_infer, _, preprocessing_profile = _build_resnet18(device, checkpoint=checkpoint)
        elif backend != "hist":
            raise ValueError(f"Unsupported embedding backend: {backend}")
    except Exception as exc:
        if not allow_fallback:
            raise
        backend = "hist"
        save_json(run_dir / "embedding_fallback.json", {"requested": requested_backend, "fallback": backend, "reason": str(exc)})

    if backend == "hist":
        # The histogram/gradient descriptor never touches any checkpoint --
        # whether reached by direct selection or by fallback -- so recording
        # an unrelated configured checkpoint path/hash here would be a lie a
        # later reader could mistake for provenance.
        checkpoint = None
        preprocessing_profile = "hist_v1"

    checkpoint_sha256 = sha256_file(checkpoint) if checkpoint and Path(checkpoint).exists() else ""

    tracklet_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    geometry_embeddings: list[np.ndarray] = []
    session_ids: list[str] = []
    video_ids: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    ordered_tracks = valid_tracks.sort_values(["session_id", "start_s"]).reset_index(drop=True)
    total_tracks = len(ordered_tracks)
    progress_every = max(1, total_tracks // 10)
    print(f"[embed] backend={backend} device={device} tracklets={total_tracks}", flush=True)
    for track_index, (_, track) in enumerate(ordered_tracks.iterrows(), start=1):
        tracklet_id = str(track["tracklet_id"])
        subset = frames.loc[frames["tracklet_id"] == tracklet_id].sort_values("rank")
        feature_column = "embedding_path" if "embedding_path" in subset.columns else "torso_path"
        paths = subset[feature_column].astype(str).tolist()
        if not paths:
            continue
        geometry_frame_features = np.vstack([_pattern_layout_descriptor(path) for path in paths])
        if backend in {"opencows2020", "metric", "resnet18"}:
            chunks = []
            for offset in range(0, len(paths), batch_size):
                chunks.append(deep_infer(paths[offset : offset + batch_size]))
            frame_features = np.vstack(chunks)
        else:
            frame_features = geometry_frame_features
        qualities = subset["quality"].to_numpy(dtype=np.float32)
        weights = np.maximum(qualities, 0.05)
        pooled = l2_normalize(np.average(frame_features, axis=0, weights=weights))
        # Hand-crafted, flip-sensitive spatial/gradient layout descriptor computed
        # independently of the deep backend, so a mismatch here can flag pairs the
        # deep embedding rates as similar due to a mirrored/transposed coat pattern.
        geometry_pooled = l2_normalize(np.average(geometry_frame_features, axis=0, weights=weights))
        tracklet_ids.append(tracklet_id)
        embeddings.append(pooled)
        geometry_embeddings.append(geometry_pooled)
        session_ids.append(str(track["session_id"]))
        video_ids.append(str(track["video_id"]))
        metadata_rows.append(
            {
                "tracklet_id": tracklet_id,
                "session_id": str(track["session_id"]),
                "video_id": str(track["video_id"]),
                "embedding_backend": backend,
                "embedding_checkpoint": checkpoint or "ImageNet",
                "checkpoint_sha256": checkpoint_sha256,
                "preprocessing_profile": preprocessing_profile,
                "crop_version": str(track.get("crop_version", "unknown")),
                "embedding_dim": int(pooled.shape[0]),
                "embedding_schema_version": f"{int(pooled.shape[0])}d_{EMBEDDING_SCHEMA_VERSION}",
                "frames_used": len(paths),
                "mean_frame_quality": float(np.mean(qualities)),
                "contact_path": track.get("contact_path", ""),
            }
        )
        if track_index % progress_every == 0 or track_index == total_tracks:
            print(f"[embed] {track_index}/{total_tracks}", flush=True)
    if not embeddings:
        raise RuntimeError("No valid track embeddings were produced.")

    # crop_version should be uniform across one extraction run; if the column is
    # missing (an older tracklets.csv) or mixed, fall back to "unknown" rather
    # than guessing -- gallery.py's compatibility check treats "unknown" as
    # "cannot verify," never as "matches."
    crop_versions = {str(row.get("crop_version", "unknown")) for row in metadata_rows}
    crop_version = crop_versions.pop() if len(crop_versions) == 1 else "unknown"
    embedding_dim = int(embeddings[0].shape[0])
    embedding_schema_version = f"{embedding_dim}d_{EMBEDDING_SCHEMA_VERSION}"

    backup = _backup_embedding_artifacts(run_dir)
    output = run_dir / "track_embeddings.npz"
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            tracklet_ids=np.asarray(tracklet_ids, dtype=str),
            embeddings=np.vstack(embeddings).astype(np.float32),
            geometry_embeddings=np.vstack(geometry_embeddings).astype(np.float32),
            session_ids=np.asarray(session_ids, dtype=str),
            video_ids=np.asarray(video_ids, dtype=str),
            backend=np.asarray([backend], dtype=str),
            checkpoint=np.asarray([checkpoint or "ImageNet"], dtype=str),
            checkpoint_sha256=np.asarray([checkpoint_sha256], dtype=str),
            preprocessing_profile=np.asarray([preprocessing_profile], dtype=str),
            crop_version=np.asarray([crop_version], dtype=str),
            embedding_schema_version=np.asarray([embedding_schema_version], dtype=str),
        )
    temporary.replace(output)
    metadata = pd.DataFrame(metadata_rows)
    metadata_tmp = run_dir / "embedding_metadata.csv.tmp"
    metadata.to_csv(metadata_tmp, index=False)
    metadata_tmp.replace(run_dir / "embedding_metadata.csv")
    save_json(
        run_dir / "embedding_summary.json",
        {
            "requested_backend": requested_backend,
            "actual_backend": backend,
            "device": device,
            "track_embeddings": len(embeddings),
            "dimension": int(embeddings[0].shape[0]),
            "checkpoint": checkpoint or "ImageNet",
            "checkpoint_sha256": checkpoint_sha256,
            "preprocessing_profile": preprocessing_profile,
            "crop_version": crop_version,
            "embedding_schema_version": embedding_schema_version,
            "excluded_flagged_tracklets": len(excluded_tracklets),
            "previous_embedding_backup": str(backup) if backup else None,
        },
    )
    return output, metadata


def load_track_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
