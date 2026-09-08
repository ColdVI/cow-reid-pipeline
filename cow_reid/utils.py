from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd


def sha256_file(path: str | Path, chunk_mb: int = 8) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_mb * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def video_id_from_hash(sha256: str) -> str:
    return f"vid_{sha256[:12]}"


def bool_mask(series: pd.Series, default: bool = False) -> pd.Series:
    """Parse CSV booleans without treating the string 'False' as truthy."""
    truthy = {"true", "1", "yes", "y", "on"}
    falsy = {"false", "0", "no", "n", "off"}

    def convert(value: object) -> bool:
        if pd.isna(value):
            return default
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in truthy:
            return True
        if normalized in falsy:
            return False
        return default

    return series.map(convert).astype(bool)


def l2_normalize(array: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.maximum(norm, 1e-12)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(1e-12, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return float(inter / union)


def clip_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(x1 + 1, min(width, round(x2))))
    y2 = int(max(y1 + 1, min(height, round(y2))))
    return x1, y1, x2, y2


def expand_box(
    box: Sequence[float], width: int, height: int, expand_x: float = 0.06, expand_y: float = 0.08
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, box)
    bw, bh = x2 - x1, y2 - y1
    return clip_box(
        (x1 - bw * expand_x, y1 - bh * expand_y, x2 + bw * expand_x, y2 + bh * expand_y),
        width,
        height,
    )


def torso_box(
    box: Sequence[float], width: int, height: int, orientation: str = "left_to_right"
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, box)
    bw, bh = x2 - x1, y2 - y1
    if orientation == "right_to_left":
        left, right = x1 + 0.22 * bw, x2 - 0.08 * bw
    else:
        left, right = x1 + 0.08 * bw, x2 - 0.22 * bw
    top, bottom = y1 + 0.04 * bh, y1 + 0.76 * bh
    return clip_box((left, top, right, bottom), width, height)


def crop_box(frame: np.ndarray, box: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    return frame[y1:y2, x1:x2].copy()


def sharpness_score(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.clip(math.log1p(value) / math.log(1001.0), 0.0, 1.0))


def encode_jpeg(image: np.ndarray, quality: int = 92) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


def decode_jpeg(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def save_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def make_contact_sheet(images: Iterable[np.ndarray], columns: int = 4, tile_size: tuple[int, int] = (256, 144)) -> np.ndarray:
    items = [img for img in images if img is not None and img.size]
    if not items:
        return np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
    tiles = []
    tw, th = tile_size
    for image in items:
        h, w = image.shape[:2]
        scale = min(tw / max(w, 1), th / max(h, 1))
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
        canvas = np.full((th, tw, 3), 24, dtype=np.uint8)
        y = (th - resized.shape[0]) // 2
        x = (tw - resized.shape[1]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        tiles.append(canvas)
    rows = int(math.ceil(len(tiles) / columns))
    while len(tiles) < rows * columns:
        tiles.append(np.full((th, tw, 3), 24, dtype=np.uint8))
    return np.vstack([np.hstack(tiles[r * columns : (r + 1) * columns]) for r in range(rows)])
