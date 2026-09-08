from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "inventory": {
        "video_glob": "*.mp4",
        "ocr_timestamp": True,
        "hash_chunk_mb": 8,
    },
    "extract": {
        "backend": "yolo",
        "model": "yolo11n-seg.pt",
        "cow_class": 19,
        "sample_fps": 5.0,
        "imgsz": 768,
        "confidence": 0.16,
        "nms_iou": 0.10,
        "device": "auto",
        "roi": [0.0, 0.22, 1.0, 0.80],
        "gate": [0.18, 0.82],
        "orientation": "left_to_right",
        "best_frames": 12,
        "min_track_samples": 8,
        "min_track_span": 0.16,
        "min_best_frames": 3,
        "min_box_area": 0.010,
        "keep_rejected": False,
        "jpeg_quality": 92,
        "motion": {
            "history": 250,
            "var_threshold": 28,
            "min_area": 0.012,
            "max_distance": 0.22,
            "max_missed": 8,
        },
    },
    "embedding": {
        "backend": "opencows2020",
        "checkpoint": "weights/opencows2020_softmaxrtl.pkl",
        "batch_size": 64,
        "device": "auto",
        "pool": "quality_mean",
        "allow_fallback": False,
    },
    "splitting": {
        "test_dates": [],
        "validation_dates": [],
        "open_set_cow_ids": [],
    },
    "training": {
        "epochs": 30,
        "learning_rate": 0.00001,
        "weight_decay": 0.0001,
        "device": "auto",
        "seed": 42,
        "embedding_dim": 128,
        "pretrained_checkpoint": "weights/opencows2020_softmaxrtl.pkl",
        "require_pretrained": True,
        "input_profile": "opencows2020_legacy",
        "metric_loss": "reciprocal",
        "metric_weight": 0.01,
        "triplet_margin": 0.2,
        "identities_per_batch": 4,
        "instances_per_identity": 4,
        "stride_layout": "torchvision",
        "freeze_backbone": False,
    },
    "matching": {
        "similarity_threshold": 0.95,
        "min_margin": 0.02,
        "top_k": 5,
        "same_session_cannot_link": True,
        "unknown_threshold": 0.95,
    },
}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return cfg
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    _deep_update(cfg, supplied)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def resolve_model_path(model_value: str, config: dict[str, Any]) -> str:
    candidate = Path(model_value)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    config_path = config.get("_config_path")
    if config_path:
        root = Path(config_path).resolve().parent.parent
        rooted = root / candidate
        if rooted.exists():
            return str(rooted)
        bundled = root / "weights" / candidate.name
        if bundled.exists():
            return str(bundled)
    return model_value
