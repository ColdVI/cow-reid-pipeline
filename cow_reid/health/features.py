from __future__ import annotations

import numpy as np


def dorsal_arch_score(points: np.ndarray) -> float:
    """Maximum dorsal distance to the withers/hip chord, normalized by body length."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError("points must be an Nx2 array ordered from withers to hip")
    start, end = points[0], points[-1]
    chord = end - start
    length = float(np.linalg.norm(chord))
    if length <= 1e-9:
        raise ValueError("withers and hip cannot coincide")
    distances = np.abs(chord[0] * (points[:, 1] - start[1]) - chord[1] * (points[:, 0] - start[0])) / length
    return float(np.max(distances) / length)


def summarize_feature(values: np.ndarray, total_frames: int | None = None) -> dict[str, float | int]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("No valid feature values")
    total = max(int(total_frames or clean.size), int(clean.size))
    return {
        "median": float(np.median(clean)), "p90": float(np.percentile(clean, 90)),
        "standard_deviation": float(np.std(clean)), "valid_frame_count": int(clean.size),
        "valid_frame_ratio": float(clean.size / total),
        "measurement_uncertainty": float(1.4826 * np.median(np.abs(clean - np.median(clean))) / np.sqrt(clean.size)),
    }
