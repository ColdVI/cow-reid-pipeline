from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import audit_labels
from .embeddings import load_track_embeddings
from .utils import bool_mask, l2_normalize, save_json


PREDICTION_COLUMNS = [
    "query_tracklet_id", "query_cow_id", "query_session_id",
    "best_candidate_tracklet_id", "best_candidate_cow_id", "best_similarity",
    "correct_rank", "top1_correct", "top5_correct", "average_precision", "eligible_candidates",
]


def held_out_date_split(
    manifest: pd.DataFrame,
    test_dates: list[str],
    validation_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Assign whole recording dates and overlap groups to one leakage-safe split.

    A row with an unverified/unparseable recording date (see
    ``overlap.assign_overlap_groups``'s ``overlap_unverified`` flag, or a
    missing date outright) is always forced to ``"train"`` — an unverified
    date must never become part of a held-out validation/test day, since we
    cannot confirm it doesn't secretly overlap a day already used for
    training. It is still fine to train on it.
    """
    frame = manifest.copy()
    timestamp = frame.get("recording_start", frame.get("recording_timestamp"))
    if timestamp is None:
        raise ValueError("Manifest needs recording_start from OCR/manual correction")
    frame["recording_date"] = pd.to_datetime(timestamp, errors="coerce").dt.date.astype("string")
    tests = set(map(str, test_dates)); validations = set(map(str, validation_dates or []))
    if tests & validations:
        raise ValueError("A recording date cannot be both validation and test")
    known_dates = set(frame["recording_date"].dropna())
    missing = (tests | validations) - known_dates
    if missing:
        raise ValueError(f"No manifest rows found for requested split date(s): {sorted(missing)}")
    frame["split"] = "train"
    frame.loc[frame["recording_date"].isin(validations), "split"] = "validation"
    frame.loc[frame["recording_date"].isin(tests), "split"] = "test"
    unverified = bool_mask(frame.get("overlap_unverified", pd.Series(False, index=frame.index)), default=False)
    frame.loc[unverified.to_numpy() | frame["recording_date"].isna().to_numpy(), "split"] = "train"
    group_column = "overlap_group_id" if "overlap_group_id" in frame else "canonical_video_id"
    if group_column in frame:
        conflicts = frame.loc[frame[group_column].fillna("").ne("")].groupby(group_column)["split"].nunique()
        if (conflicts > 1).any():
            raise ValueError("An overlap/canonical source group crosses dataset splits")
    return frame


def evaluate_embeddings(
    run_dir: str | Path,
    labels_csv: str | Path,
    embeddings_path: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate cross-session, track-level retrieval on conflict-free labels."""

    run_dir = Path(run_dir).resolve()
    audit_outputs = audit_labels(run_dir, labels_csv)
    labels = pd.read_csv(audit_outputs["safe_labels"], dtype={"cow_id": str})
    keep = bool_mask(labels.get("confirmed", pd.Series(True, index=labels.index)), default=False)
    keep &= bool_mask(labels.get("training_eligible", pd.Series(True, index=labels.index)), default=False)
    labels = labels.loc[keep & labels["cow_id"].fillna("").ne("")].drop_duplicates("tracklet_id")

    embedding_file = Path(embeddings_path).resolve() if embeddings_path else run_dir / "track_embeddings.npz"
    data = load_track_embeddings(embedding_file)
    embedded = pd.DataFrame(
        {
            "tracklet_id": data["tracklet_ids"].astype(str),
            "embedding_index": np.arange(len(data["tracklet_ids"])),
            "embedding_session_id": data["session_ids"].astype(str),
        }
    )
    labelled = labels.merge(embedded, on="tracklet_id", how="inner")
    if "session_id" not in labelled:
        labelled["session_id"] = labelled["embedding_session_id"]
    else:
        labelled["session_id"] = labelled["session_id"].fillna(labelled["embedding_session_id"]).astype(str)
    embeddings = l2_normalize(data["embeddings"])
    rows: list[dict[str, Any]] = []

    for _, query in labelled.iterrows():
        query_index = int(query["embedding_index"])
        query_session = str(query["session_id"])
        candidates = labelled.loc[
            labelled["tracklet_id"].astype(str).ne(str(query["tracklet_id"]))
            & labelled["session_id"].astype(str).ne(query_session)
        ].copy()
        positives_exist = candidates["cow_id"].astype(str).eq(str(query["cow_id"])).any()
        if candidates.empty or not positives_exist:
            continue
        candidate_indices = candidates["embedding_index"].to_numpy(dtype=int)
        candidates["similarity"] = embeddings[candidate_indices] @ embeddings[query_index]
        candidates = candidates.sort_values("similarity", ascending=False).reset_index(drop=True)
        relevant = candidates["cow_id"].astype(str).eq(str(query["cow_id"])).to_numpy()
        positive_ranks = np.flatnonzero(relevant) + 1
        precision_at_positive = np.cumsum(relevant)[positive_ranks - 1] / positive_ranks
        correct_rank = int(positive_ranks[0])
        best = candidates.iloc[0]
        rows.append(
            {
                "query_tracklet_id": str(query["tracklet_id"]),
                "query_cow_id": str(query["cow_id"]),
                "query_session_id": query_session,
                "best_candidate_tracklet_id": str(best["tracklet_id"]),
                "best_candidate_cow_id": str(best["cow_id"]),
                "best_similarity": float(best["similarity"]),
                "correct_rank": correct_rank,
                "top1_correct": correct_rank == 1,
                "top5_correct": correct_rank <= 5,
                "average_precision": float(np.mean(precision_at_positive)),
                "eligible_candidates": int(len(candidates)),
            }
        )

    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    predictions_path = run_dir / "reid_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    backend = str(data.get("backend", np.asarray(["unknown"]))[0])
    checkpoint = str(data.get("checkpoint", np.asarray(["unknown"]))[0])
    summary: dict[str, Any] = {
        "protocol": "cross-session leave-one-tracklet-out retrieval",
        "embedding_file": str(embedding_file),
        "embedding_backend": backend,
        "embedding_checkpoint": checkpoint,
        "safe_cow_ids": int(labels["cow_id"].nunique()),
        "safe_labelled_tracklets": int(labels["tracklet_id"].nunique()),
        "queries_with_cross_session_positive": int(len(predictions)),
        "top1": float(predictions["top1_correct"].mean()) if not predictions.empty else None,
        "top5": float(predictions["top5_correct"].mean()) if not predictions.empty else None,
        "mAP": float(predictions["average_precision"].mean()) if not predictions.empty else None,
        "median_correct_rank": float(predictions["correct_rank"].median()) if not predictions.empty else None,
        "predictions": str(predictions_path),
        "warning": (
            "This is a cross-session retrieval check, not a final held-out-day result. "
            "If these labels were used to train the checkpoint, report a separate unseen day before deployment."
        ),
    }
    target = Path(output_json).resolve() if output_json else run_dir / "reid_evaluation.json"
    save_json(target, summary)
    summary["output"] = str(target)
    return summary
