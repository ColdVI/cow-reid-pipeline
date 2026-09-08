from __future__ import annotations

from collections.abc import Mapping, Sequence


def frame_is_valid(
    keypoints: Mapping[str, tuple[float, float, float]],
    required: Sequence[str] = ("withers", "spine", "hip"),
    min_visibility: float = 0.5,
    occlusion: float = 0.0,
    tracker_jump: bool = False,
    scale_ratio: float = 1.0,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if any(name not in keypoints or keypoints[name][2] < min_visibility for name in required):
        reasons.append("low_visibility")
    if occlusion > 0.35:
        reasons.append("occlusion")
    if tracker_jump:
        reasons.append("tracker_jump")
    if not 0.65 <= scale_ratio <= 1.55:
        reasons.append("implausible_scale_change")
    present = [keypoints[name] for name in required if name in keypoints]
    if len(present) == len(required):
        xs = [point[0] for point in present]
        if not (all(a <= b for a, b in zip(xs, xs[1:])) or all(a >= b for a, b in zip(xs, xs[1:]))):
            reasons.append("anatomical_order")
    return not reasons, reasons

