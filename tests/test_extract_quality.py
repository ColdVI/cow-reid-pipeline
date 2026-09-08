import numpy as np

from cow_reid.extract import (
    APPEARANCE_CONSISTENCY_FLAG_THRESHOLD,
    _classify_direction,
    _intra_track_appearance_consistency,
)


def test_classify_direction_clear_left_to_right_is_ok() -> None:
    ok, wrong, uncertain = _classify_direction(net_dx=1.0, min_span=0.5, orientation="left_to_right")
    assert ok and not wrong and not uncertain


def test_classify_direction_clear_reversal_is_wrong_not_uncertain() -> None:
    ok, wrong, uncertain = _classify_direction(net_dx=-1.0, min_span=0.5, orientation="left_to_right")
    assert not ok and wrong and not uncertain


def test_classify_direction_near_zero_net_motion_is_uncertain_not_wrong() -> None:
    ok, wrong, uncertain = _classify_direction(net_dx=0.01, min_span=0.5, orientation="left_to_right")
    assert not ok and not wrong and uncertain


def test_classify_direction_right_to_left_orientation_mirrors_logic() -> None:
    ok, wrong, uncertain = _classify_direction(net_dx=-1.0, min_span=0.5, orientation="right_to_left")
    assert ok and not wrong and not uncertain
    ok, wrong, uncertain = _classify_direction(net_dx=1.0, min_span=0.5, orientation="right_to_left")
    assert not ok and wrong and not uncertain


def _solid(color: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def test_single_frame_returns_none() -> None:
    assert _intra_track_appearance_consistency([_solid((200, 200, 200))]) is None


def test_empty_list_returns_none() -> None:
    assert _intra_track_appearance_consistency([]) is None


def test_identical_frames_score_near_one() -> None:
    frames = [_solid((180, 120, 90)) for _ in range(5)]
    score = _intra_track_appearance_consistency(frames)
    assert score is not None
    assert score > 0.999


def test_wildly_different_frames_fall_below_threshold() -> None:
    frames = [_solid((250, 250, 250)), _solid((250, 250, 250)), _solid((5, 5, 5))]
    score = _intra_track_appearance_consistency(frames)
    assert score is not None
    assert score < APPEARANCE_CONSISTENCY_FLAG_THRESHOLD


def test_mild_lighting_drift_stays_above_threshold() -> None:
    rng = np.random.default_rng(0)
    base = rng.integers(60, 200, size=(64, 64, 3), dtype=np.uint8)
    frames = [
        np.clip(base.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        for shift in (0, 8, -8)
    ]
    score = _intra_track_appearance_consistency(frames)
    assert score is not None
    assert score > APPEARANCE_CONSISTENCY_FLAG_THRESHOLD
