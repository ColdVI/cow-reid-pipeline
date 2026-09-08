from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import cv2
import numpy as np

from .quality import frame_is_valid
from .schema import CanonicalPoseResult, KeypointFrame


DORSAL_POINTS = ("withers", "thoracic_spine", "dorsal_apex", "lumbar_spine", "sacrum")


def validate_checkpoint_file(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DLC checkpoint not found: {checkpoint}")
    prefix = checkpoint.read_bytes()[:256]
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        pointer = prefix.decode("utf-8", errors="replace")
        expected = next(
            (line.removeprefix("size ") for line in pointer.splitlines() if line.startswith("size ")),
            "unknown",
        )
        raise RuntimeError(
            f"DLC checkpoint is only a Git LFS pointer ({checkpoint.stat().st_size} bytes); "
            f"download the real model object (expected {expected} bytes): {checkpoint}"
        )
    return checkpoint


def _load_external_module(repo_path: Path) -> ModuleType:
    script = repo_path / "scripts" / "run_custom_cow_inference.py"
    if not script.is_file():
        raise FileNotFoundError(f"lameness-main inference script not found: {script}")
    # Ultralytics initializes a settings file at import time. Keep that incidental
    # state in a writable cache instead of touching the user's global config.
    settings_dir = Path(tempfile.gettempdir()) / "cow-reid-ultralytics"
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_dir))
    spec = importlib.util.spec_from_file_location("cow_reid_lameness_external", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load external lameness module: {script}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _load_custom_pose_model_compat(
    module: ModuleType,
    config_path: Path,
    device: str,
    batch_size: int,
    checkpoint: Path | None = None,
) -> tuple[Any, dict[str, Any], Path]:
    """Load lameness-main weights across supported DeepLabCut 3 runner APIs."""
    if checkpoint is None:
        try:
            return module.load_custom_pose_model(
                config_path, device_str=device, batch_size=batch_size
            )
        except TypeError as exc:
            if "async_mode" not in str(exc):
                raise

    # lameness-main was authored against a DLC 3 prerelease that accepted
    # async_mode=False. DLC 3.0.1 removed that argument; the remaining model
    # construction and checkpoint format are unchanged.
    import yaml
    from deeplabcut.pose_estimation_pytorch.apis.utils import (
        build_transforms,
        resolve_device,
    )
    from deeplabcut.pose_estimation_pytorch.data.postprocessor import (
        build_bottom_up_postprocessor,
    )
    from deeplabcut.pose_estimation_pytorch.data.preprocessor import (
        build_bottom_up_preprocessor,
    )
    from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
    from deeplabcut.pose_estimation_pytorch.runners import PoseInferenceRunner

    snapshots = sorted(config_path.parent.glob("dlc-models-pytorch/**/snapshot-*.pt"))
    best = [path for path in snapshots if "snapshot-best-" in path.name]
    if checkpoint is not None:
        selected = validate_checkpoint_file(checkpoint)
    elif not snapshots:
        raise FileNotFoundError(f"No DLC snapshots found below {config_path.parent}")
    else:
        selected = validate_checkpoint_file(best[-1] if best else snapshots[-1])
    model_cfg_path = selected.parent / "pytorch_config.yaml"
    if not model_cfg_path.is_file():
        matching = [path.parent / "pytorch_config.yaml" for path in snapshots if path.name == selected.name]
        configs = [path for path in matching if path.is_file()] or sorted(config_path.parent.glob("dlc-models-pytorch/**/pytorch_config.yaml"))
        if not configs:
            raise FileNotFoundError("No pytorch_config.yaml found for the DLC checkpoint")
        model_cfg_path = configs[0]
    with model_cfg_path.open(encoding="utf-8") as stream:
        model_cfg = yaml.safe_load(stream)
    resolved_device = resolve_device(model_cfg) if device == "auto" else device
    model = PoseModel.build(model_cfg["model"])
    transform = build_transforms(model_cfg["data"]["inference"])
    preprocessor = build_bottom_up_preprocessor(color_mode="RGB", transform=transform)
    postprocessor = build_bottom_up_postprocessor(
        max_individuals=1,
        num_bodyparts=len(model_cfg["metadata"]["bodyparts"]),
        num_unique_bodyparts=0,
    )
    runner = PoseInferenceRunner(
        model=model,
        snapshot_path=selected,
        device=resolved_device,
        batch_size=batch_size,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        # This user-supplied legacy DLC snapshot contains optimizer metadata and
        # predates PyTorch's weights_only default. It is not an untrusted upload.
        load_weights_only=False,
    )
    return runner, model_cfg, selected


def _interpolate_box(trajectory: Sequence[dict[str, Any]], timestamp_s: float) -> np.ndarray | None:
    records = sorted(trajectory, key=lambda row: float(row["timestamp_s"]))
    if not records: return None
    if timestamp_s <= float(records[0]["timestamp_s"]): return np.asarray(records[0]["bbox"], float)
    if timestamp_s >= float(records[-1]["timestamp_s"]): return np.asarray(records[-1]["bbox"], float)
    for left, right in zip(records, records[1:]):
        a, b = float(left["timestamp_s"]), float(right["timestamp_s"])
        if a <= timestamp_s <= b:
            alpha = (timestamp_s - a) / max(1e-9, b - a)
            return (1 - alpha) * np.asarray(left["bbox"], float) + alpha * np.asarray(right["bbox"], float)
    return None


class LamenessDLCBackend:
    name = "lameness_custom_hrnet_w32"

    def __init__(self, repo_path: str | Path, checkpoint: str | Path | None = None, device: str = "auto", batch_size: int = 16, min_visibility: float = 0.25):
        self.repo_path = Path(repo_path).resolve(); self.checkpoint = Path(checkpoint).resolve() if checkpoint else None
        self.device = device; self.batch_size = batch_size; self.min_visibility = min_visibility
        self._loaded: tuple[Any, dict[str, Any], Path, ModuleType] | None = None
        selected = validate_checkpoint_file(self.checkpoint or self._discover_checkpoint())
        self.version = f"{selected.stem}:{hashlib.sha256(selected.read_bytes()).hexdigest()[:12]}"

    def _discover_checkpoint(self) -> Path:
        snapshots = sorted(self.repo_path.glob("dlc_projects/**/snapshot-best-*.pt")) or sorted(self.repo_path.glob("dlc_projects/**/snapshot-*.pt"))
        if not snapshots: raise FileNotFoundError("No trained DLC checkpoint found in lameness-main")
        return snapshots[-1]

    def _load(self) -> tuple[Any, dict[str, Any], Path, ModuleType]:
        if self._loaded is None:
            module = _load_external_module(self.repo_path)
            configs = sorted(self.repo_path.glob("dlc_projects/*/config.yaml"))
            if not configs: raise FileNotFoundError("No DLC project config found in lameness-main")
            runner, model_cfg, selected = _load_custom_pose_model_compat(
                module, configs[-1], self.device, self.batch_size, self.checkpoint
            )
            selected = Path(selected).resolve()
            if self.checkpoint and selected != self.checkpoint:
                raise ValueError(f"External loader selected {selected}, requested {self.checkpoint}")
            self._loaded = runner, model_cfg, selected, module
        return self._loaded

    def infer_tracklet(self, video_path: str, start_s: float, end_s: float, trajectory: Sequence[dict[str, object]], output_dir: str | Path) -> CanonicalPoseResult:
        runner, model_cfg, _, module = self._load(); bodyparts = list(model_cfg["metadata"]["bodyparts"])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        first_frame = max(0, int(np.floor(start_s * fps))); last_frame = max(first_frame, int(np.ceil(end_s * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        inputs: list[np.ndarray] = []; contexts: list[tuple[int, float, np.ndarray, tuple[int, int, int, int]]] = []
        for frame_idx in range(first_frame, last_frame + 1):
            ok, frame = cap.read()
            if not ok: break
            timestamp = frame_idx / fps; box = _interpolate_box(trajectory, timestamp)
            if box is None: continue
            x1, y1, x2, y2 = box; bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            px1, py1 = max(0, int(x1 - .15 * bw)), max(0, int(y1 - .15 * bh)); px2, py2 = min(width, int(x2 + .15 * bw)), min(height, int(y2 + .15 * bh))
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0: continue
            inputs.append(cv2.cvtColor(cv2.resize(crop, (448, 448)), cv2.COLOR_BGR2RGB)); contexts.append((frame_idx, timestamp, box, (px1, py1, px2, py2)))
        cap.release()
        predictions = []
        for offset in range(0, len(inputs), self.batch_size): predictions.extend(runner.inference(images=inputs[offset:offset + self.batch_size]))
        heading_right = True
        if trajectory: heading_right = float(trajectory[-1]["bbox"][0]) >= float(trajectory[0]["bbox"][0])
        canonical: list[KeypointFrame] = []; invalid = Counter(); valid_frames = 0
        for pred, (frame_idx, timestamp, box, crop_box) in zip(predictions, contexts):
            points = np.asarray(pred["bodyparts"][0], dtype=float).copy(); px1, py1, px2, py2 = crop_box
            points[:, 0] = points[:, 0] * ((px2 - px1) / 448.0) + px1; points[:, 1] = points[:, 1] * ((py2 - py1) / 448.0) + py1
            mapping = {name: idx for idx, name in enumerate(bodyparts)}
            points = module.validate_and_correct_cow_anatomy(points, box, mapping, width, height, heading_right)
            point_map = {name: (float(points[i, 0]), float(points[i, 1]), float(points[i, 2])) for i, name in enumerate(bodyparts)}
            valid, reasons = frame_is_valid(point_map, required=DORSAL_POINTS, min_visibility=self.min_visibility)
            if valid: valid_frames += 1
            else: invalid.update(reasons)
            body_length = max(1.0, float(box[2] - box[0]))
            for index, name in enumerate(bodyparts):
                x, y, visibility = map(float, points[index])
                canonical.append(KeypointFrame("", frame_idx, timestamp, name, x, y, (x - float(box[0])) / body_length, (y - float(box[1])) / body_length, visibility, self.name, self.version))
        output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        tracklet_id = output.parent.name if output.name.startswith("pose_") else output.name
        canonical = [KeypointFrame(tracklet_id, item.frame_idx, item.timestamp_s, item.keypoint_name, item.x_px, item.y_px, item.x_normalized, item.y_normalized, item.visibility, item.model_name, item.model_version) for item in canonical]
        result = CanonicalPoseResult(tracklet_id, canonical, str(output / "keypoints.csv"), {"total_frames": len(contexts), "valid_frames": valid_frames, "valid_frame_ratio": valid_frames / max(1, len(contexts)), "invalid_reasons": dict(invalid), "checkpoint": str(self.checkpoint or self._discover_checkpoint())})
        result.to_csv(output / "keypoints.csv"); (output / "quality.json").write_text(json.dumps(result.quality_summary, indent=2), encoding="utf-8")
        return result
