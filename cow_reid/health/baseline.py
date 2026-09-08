from __future__ import annotations

import numpy as np
import pandas as pd


def longitudinal_baseline(
    observations: pd.DataFrame,
    value_column: str = "value",
    time_column: str = "observed_at",
    window_days: int = 30,
    min_days: int = 3,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    frame = observations.copy()
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna(subset=[time_column, value_column]).sort_values(time_column).reset_index(drop=True)
    baselines: list[float] = []
    mads: list[float] = []
    day_counts: list[int] = []
    for _, row in frame.iterrows():
        start = row[time_column] - pd.Timedelta(days=window_days)
        history = frame.loc[(frame[time_column] < row[time_column]) & (frame[time_column] >= start)]
        days = history[time_column].dt.date.nunique()
        values = history[value_column].to_numpy(float)
        median = float(np.median(values)) if days >= min_days else np.nan
        mad = float(np.median(np.abs(values - median))) if days >= min_days else np.nan
        baselines.append(median); mads.append(mad); day_counts.append(int(days))
    frame["baseline"] = baselines
    frame["mad"] = mads
    frame["baseline_day_count"] = day_counts
    frame["delta"] = frame[value_column] - frame["baseline"]
    frame["robust_z"] = frame["delta"] / (1.4826 * frame["mad"] + epsilon)
    return frame

