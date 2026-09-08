from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .identity import load_pair_reviews, normalize_cow_id
from .review_state import load_excluded_tracklets
from .utils import bool_mask, save_json


IDENTITY_AUDIT_COLUMNS = [
    "cow_id",
    "tracklets",
    "sessions",
    "videos",
    "different_edges_inside_identity",
    "same_edges_across_identities",
    "same_session_duplicates",
    "excluded_flagged_tracklets",
    "status",
    "training_eligible",
]

CONFLICT_COLUMNS = [
    "conflict_type",
    "cow_id",
    "other_cow_id",
    "left_tracklet_id",
    "right_tracklet_id",
    "decision",
    "updated_at",
]


def _confirmed_labels(labels: pd.DataFrame) -> pd.DataFrame:
    frame = labels.copy()
    if "cow_id" not in frame:
        frame["cow_id"] = ""
    frame["cow_id"] = frame["cow_id"].fillna("").map(normalize_cow_id)
    confirmed = bool_mask(frame.get("confirmed", pd.Series(True, index=frame.index)), default=False)
    return frame.loc[confirmed & frame["cow_id"].ne("")].copy()


def inspect_label_consistency(
    labels: pd.DataFrame,
    pair_reviews: pd.DataFrame,
    excluded_tracklets: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return identity audit, direct conflicts and a training-safe label table.

    ``excluded_tracklets`` (typically from ``review_state.load_excluded_tracklets``)
    are tracklets a human has flagged broken/unresolved-split in
    tracklet_reviews.csv: they never count toward a cow_id's "enough
    tracklets to train on" threshold, so a cow whose only two confirmed
    tracklets include a flagged one is correctly reported
    ``needs_more_tracklets`` rather than falsely ``safe``, while a cow with a
    flagged tracklet *and* enough other clean ones stays eligible on those.
    """

    excluded_tracklets = excluded_tracklets or set()
    original = labels.copy()
    assigned = _confirmed_labels(original)
    id_by_track = assigned.drop_duplicates("tracklet_id").set_index("tracklet_id")["cow_id"].to_dict()
    conflict_rows: list[dict[str, Any]] = []
    conflict_ids: set[str] = set()

    for _, review in pair_reviews.iterrows():
        left = str(review.get("left_tracklet_id", ""))
        right = str(review.get("right_tracklet_id", ""))
        decision = str(review.get("decision", ""))
        left_id = normalize_cow_id(id_by_track.get(left, ""))
        right_id = normalize_cow_id(id_by_track.get(right, ""))
        if decision == "different" and left_id and left_id == right_id:
            conflict_ids.add(left_id)
            conflict_rows.append(
                {
                    "conflict_type": "different_pair_inside_identity",
                    "cow_id": left_id,
                    "other_cow_id": "",
                    "left_tracklet_id": left,
                    "right_tracklet_id": right,
                    "decision": decision,
                    "updated_at": str(review.get("updated_at", "")),
                }
            )
        elif decision == "same" and left_id and right_id and left_id != right_id:
            conflict_ids.update((left_id, right_id))
            conflict_rows.append(
                {
                    "conflict_type": "same_pair_across_identities",
                    "cow_id": left_id,
                    "other_cow_id": right_id,
                    "left_tracklet_id": left,
                    "right_tracklet_id": right,
                    "decision": decision,
                    "updated_at": str(review.get("updated_at", "")),
                }
            )

    conflicts = pd.DataFrame(conflict_rows, columns=CONFLICT_COLUMNS)
    audit_rows: list[dict[str, Any]] = []
    for cow_id, group in assigned.groupby("cow_id"):
        members = set(group["tracklet_id"].astype(str))
        cross_count = 0
        if not conflicts.empty:
            cross_count = int(
                (
                    conflicts["conflict_type"].eq("same_pair_across_identities")
                    & (conflicts["cow_id"].eq(cow_id) | conflicts["other_cow_id"].eq(cow_id))
                ).sum()
            )
        different_count = 0
        if not pair_reviews.empty:
            different_count = int(
                (
                    pair_reviews["decision"].astype(str).eq("different")
                    & pair_reviews["left_tracklet_id"].astype(str).isin(members)
                    & pair_reviews["right_tracklet_id"].astype(str).isin(members)
                ).sum()
            )
        session_sizes = group.groupby("session_id").size() if "session_id" in group else pd.Series(dtype=int)
        same_session_duplicates = int((session_sizes > 1).sum())
        usable_members = members - excluded_tracklets
        enough_tracklets = len(usable_members) >= 2
        has_conflict = cow_id in conflict_ids
        if has_conflict:
            status = "conflict"
        elif not enough_tracklets:
            status = "needs_more_tracklets"
        elif same_session_duplicates:
            status = "warning_same_session_duplicates"
        else:
            status = "safe"
        audit_rows.append(
            {
                "cow_id": cow_id,
                "tracklets": int(group["tracklet_id"].nunique()),
                "sessions": int(group["session_id"].nunique()) if "session_id" in group else 0,
                "videos": int(group["video_id"].nunique()) if "video_id" in group else 0,
                "different_edges_inside_identity": different_count,
                "same_edges_across_identities": cross_count,
                "same_session_duplicates": same_session_duplicates,
                "excluded_flagged_tracklets": int(len(members & excluded_tracklets)),
                "status": status,
                "training_eligible": bool(enough_tracklets and not has_conflict),
            }
        )
    audit = pd.DataFrame(audit_rows, columns=IDENTITY_AUDIT_COLUMNS)
    if not audit.empty:
        audit = audit.sort_values(["training_eligible", "status", "cow_id"], ascending=[True, True, True])

    safe = original.copy()
    if "cow_id" not in safe:
        safe["cow_id"] = ""
    safe["cow_id"] = safe["cow_id"].fillna("").map(normalize_cow_id)
    eligible_ids = set(audit.loc[audit["training_eligible"], "cow_id"]) if not audit.empty else set()
    safe["training_eligible"] = safe["cow_id"].isin(eligible_ids) & bool_mask(
        safe.get("confirmed", pd.Series(True, index=safe.index)), default=False
    )
    status_by_id = audit.set_index("cow_id")["status"].to_dict() if not audit.empty else {}
    safe["audit_status"] = safe["cow_id"].map(status_by_id).fillna("unassigned")
    return audit.reset_index(drop=True), conflicts.reset_index(drop=True), safe


def audit_labels(
    run_dir: str | Path,
    labels_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    run_dir = Path(run_dir).resolve()
    labels_path = Path(labels_csv).resolve() if labels_csv else run_dir / "labels.csv"
    target = Path(output_dir).resolve() if output_dir else run_dir
    target.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(labels_path, dtype={"cow_id": str})
    reviews = load_pair_reviews(run_dir)
    excluded_tracklets = load_excluded_tracklets(run_dir)
    audit, conflicts, safe = inspect_label_consistency(labels, reviews, excluded_tracklets)

    audit_path = target / "label_audit.csv"
    conflicts_path = target / "label_conflicts.csv"
    safe_path = target / "labels_training_safe.csv"
    summary_path = target / "label_audit_summary.json"
    audit.to_csv(audit_path, index=False)
    conflicts.to_csv(conflicts_path, index=False)
    safe.to_csv(safe_path, index=False)
    save_json(
        summary_path,
        {
            "confirmed_tracklets": int(_confirmed_labels(labels)["tracklet_id"].nunique()),
            "identities": int(len(audit)),
            "safe_identities": int(audit["training_eligible"].sum()) if not audit.empty else 0,
            "conflicted_identities": int(audit["status"].eq("conflict").sum()) if not audit.empty else 0,
            "direct_conflicts": int(len(conflicts)),
            "training_safe_tracklets": int(safe["training_eligible"].sum()),
            "warning": "Conflicted identities are preserved in labels.csv but excluded from training.",
        },
    )
    return {
        "audit": audit_path,
        "conflicts": conflicts_path,
        "safe_labels": safe_path,
        "summary": summary_path,
    }


def forbidden_different_edges(
    run_dir: str | Path,
    labels: pd.DataFrame,
    tracklet_ids: set[str],
    ignored_pair: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Return cannot-link decisions that would be violated by a proposed union."""

    reviews = load_pair_reviews(run_dir)
    if reviews.empty:
        return reviews
    ignored = tuple(sorted(ignored_pair)) if ignored_pair else None
    mask: list[bool] = []
    for _, row in reviews.iterrows():
        left = str(row["left_tracklet_id"])
        right = str(row["right_tracklet_id"])
        pair = tuple(sorted((left, right)))
        mask.append(
            str(row["decision"]) == "different"
            and left in tracklet_ids
            and right in tracklet_ids
            and pair != ignored
        )
    return reviews.loc[mask].copy()
