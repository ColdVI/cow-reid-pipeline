from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REVIEW_STATE_COLUMNS = ["tracklet_id", "state", "split_at_s", "notes", "updated_at"]

# Tracklets in these states must never be used as positive training examples or
# gallery prototypes: a flagged tracker error may mix two animals' frames, and an
# un-actioned split request means the boundary between two passages hasn't been
# confirmed yet. Only "valid" (explicitly reviewed and confirmed clean) or an
# unreviewed tracklet (no row at all) stay eligible.
EXCLUDED_TRACKLET_STATES = {"tracklet_error", "split_requested"}


def record_tracklet_state(run_dir: str | Path, tracklet_id: str, state: str, split_at_s: float | None = None, notes: str = "") -> Path:
    allowed = {"unsure", "tracklet_error", "split_requested", "valid"}
    if state not in allowed:
        raise ValueError(f"Invalid tracklet review state: {state}")
    path = Path(run_dir).resolve() / "tracklet_reviews.csv"
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=REVIEW_STATE_COLUMNS)
    for column in REVIEW_STATE_COLUMNS:
        if column not in frame: frame[column] = ""
    frame = frame.loc[~frame["tracklet_id"].astype(str).eq(str(tracklet_id))].copy()
    frame.loc[len(frame)] = {"tracklet_id": str(tracklet_id), "state": state, "split_at_s": split_at_s, "notes": notes, "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    temporary = path.with_suffix(".csv.tmp"); frame[REVIEW_STATE_COLUMNS].to_csv(temporary, index=False); temporary.replace(path)
    return path


def load_excluded_tracklets(run_dir: str | Path) -> set[str]:
    """Tracklet ids a human has flagged as broken/unresolved-split in this run's
    tracklet_reviews.csv. Every consumer that could turn a tracklet into a
    positive training example or a gallery prototype (embedding build, gallery
    build, train/val split) must filter these out. ``record_tracklet_state``
    already keeps at most one row per tracklet_id (it drops any prior row for
    the same id before appending), so no extra "latest wins" logic is needed
    here — every row present is already the current state.
    """
    path = Path(run_dir).resolve() / "tracklet_reviews.csv"
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if "tracklet_id" not in frame.columns or "state" not in frame.columns:
        return set()
    flagged = frame.loc[frame["state"].astype(str).isin(EXCLUDED_TRACKLET_STATES)]
    return set(flagged["tracklet_id"].astype(str))

