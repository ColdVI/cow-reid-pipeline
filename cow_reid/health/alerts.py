from __future__ import annotations

import pandas as pd


def review_statuses(frame: pd.DataFrame, z_threshold: float = 3.0, min_valid_ratio: float = 0.6, max_uncertainty: float = 0.05) -> pd.Series:
    def numeric(name: str, default: float) -> pd.Series:
        source = frame[name] if name in frame else pd.Series(default, index=frame.index)
        return pd.to_numeric(source, errors="coerce").fillna(default)
    valid = numeric("valid_frame_ratio", 0) >= min_valid_ratio
    certain = numeric("uncertainty", float("inf")) <= max_uncertainty
    deviates = numeric("robust_z", 0).abs() >= z_threshold
    recent = (valid & certain & deviates).rolling(3, min_periods=3).sum() >= 2
    return recent.map({True: "REVIEW_NEEDED", False: "OK"})
