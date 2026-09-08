from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_COLUMNS = [
    "tracklet_id", "pseudo_id", "session_id", "video_id", "cow_id",
    "confirmed", "assignment_source", "notes", "updated_at",
]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _merge_table(
    base: Path,
    new: Path,
    output: Path,
    filename: str,
    deduplicate: bool = True,
) -> int:
    frames = [frame for frame in (_read(base / filename), _read(new / filename)) if not frame.empty]
    if not frames:
        return 0
    merged = pd.concat(frames, ignore_index=True, sort=False)
    key = "tracklet_id" if "tracklet_id" in merged else "video_id" if "video_id" in merged else None
    if deduplicate and key:
        merged = merged.drop_duplicates(key, keep="last")
    merged.to_csv(output / filename, index=False)
    return len(merged)


def _merge_embeddings(base: Path, new: Path, output: Path) -> int:
    parts = []
    for run in (base, new):
        with np.load(run / "track_embeddings.npz", allow_pickle=False) as data:
            parts.append({name: data[name].copy() for name in data.files})
    for field in ("backend", "checkpoint"):
        values = {str(part[field][0]) for part in parts}
        if len(values) != 1:
            raise ValueError(f"Cannot merge runs with different {field} values: {sorted(values)}")
    ids = np.concatenate([part["tracklet_ids"].astype(str) for part in parts])
    if len(set(ids)) != len(ids):
        raise ValueError("Tracklet IDs overlap between review runs")
    np.savez_compressed(
        output / "track_embeddings.npz",
        tracklet_ids=ids,
        embeddings=np.vstack([part["embeddings"] for part in parts]).astype(np.float32),
        session_ids=np.concatenate([part["session_ids"].astype(str) for part in parts]),
        video_ids=np.concatenate([part["video_ids"].astype(str) for part in parts]),
        backend=parts[0]["backend"].astype(str),
        checkpoint=parts[0]["checkpoint"].astype(str),
    )
    return len(ids)


def _new_labels(new: Path) -> pd.DataFrame:
    assignments = pd.read_csv(new / "pseudo_id_assignments.csv", dtype=str)
    tracks = pd.read_csv(new / "tracklets.csv", dtype={"tracklet_id": str, "video_id": str})
    tracks = tracks.drop_duplicates("tracklet_id").set_index("tracklet_id")
    predictions = _read(new / "identity_predictions.csv")
    prediction_lookup = predictions.set_index("tracklet_id") if not predictions.empty else pd.DataFrame()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for _, assignment in assignments.iterrows():
        tracklet_id = str(assignment["tracklet_id"])
        track = tracks.loc[tracklet_id]
        notes = ""
        if not prediction_lookup.empty and tracklet_id in prediction_lookup.index:
            prediction = prediction_lookup.loc[tracklet_id]
            notes = (
                f"model_top1={prediction.get('top_1_cow_id', '')};"
                f"similarity={float(prediction.get('top_1_similarity', 0)):.6f};"
                f"prediction={prediction.get('predicted_cow_id', 'UNKNOWN')}"
            )
        rows.append(
            {
                "tracklet_id": tracklet_id,
                "pseudo_id": f"NEW_20260903_{assignment['pseudo_id']}",
                "session_id": str(track.get("session_id", assignment.get("session_id", ""))),
                "video_id": str(track.get("video_id", "")),
                "cow_id": "",
                "confirmed": False,
                "assignment_source": "new_video_unassigned",
                "notes": notes,
                "updated_at": timestamp,
            }
        )
    return pd.DataFrame(rows, columns=LABEL_COLUMNS)


def build_review_bundle(
    base_run: str | Path,
    new_run: str | Path,
    output_run: str | Path,
) -> dict[str, object]:
    base = Path(base_run).resolve()
    new = Path(new_run).resolve()
    output = Path(output_run).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "labels.csv").exists():
        raise FileExistsError(f"Review bundle already has labels and will not be overwritten: {output}")

    tracklets = _merge_table(base, new, output, "tracklets.csv")
    frames = _merge_table(base, new, output, "frames.csv", deduplicate=False)
    videos = _merge_table(base, new, output, "video_manifest.csv")
    embeddings = _merge_embeddings(base, new, output)

    old_labels = pd.read_csv(base / "labels.csv", dtype={"cow_id": str})
    for column in LABEL_COLUMNS:
        if column not in old_labels:
            old_labels[column] = ""
    labels = pd.concat([old_labels[LABEL_COLUMNS], _new_labels(new)], ignore_index=True)
    labels = labels.drop_duplicates("tracklet_id", keep="first")
    labels.to_csv(output / "labels.csv", index=False)

    _merge_table(base, new, output, "embedding_metadata.csv")
    # tracklet_reviews.csv is genuinely keyed by tracklet_id alone, so the
    # generic concat+dedupe(keep="last") merge is correct here: it combines
    # both runs' human tracklet-quality decisions instead of the previous
    # behavior of copying base's only and silently dropping every review
    # recorded against the new run.
    _merge_table(base, new, output, "tracklet_reviews.csv")
    # pair_reviews.csv/identity_exclusions.csv are keyed by composite
    # (tracklet, tracklet) / (cow_id, tracklet_id) pairs that the generic
    # single-column merge does not handle correctly, so they keep the
    # base-only copy for now.
    for filename in ("pair_reviews.csv", "identity_exclusions.csv"):
        source = base / filename
        if source.is_file():
            shutil.copy2(source, output / filename)
    prediction_source = new / "identity_predictions.csv"
    if prediction_source.is_file():
        shutil.copy2(prediction_source, output / "new_identity_predictions.csv")

    summary = {
        "base_run": str(base),
        "new_run": str(new),
        "output_run": str(output),
        "videos": videos,
        "tracklets_total": tracklets,
        "frames": frames,
        "track_embeddings": embeddings,
        "labels": len(labels),
        "confirmed_labels": int(pd.Series(labels["confirmed"]).astype(str).str.lower().isin(["true", "1"]).sum()),
        "new_unassigned": len(labels) - len(old_labels),
    }
    (output / "bundle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
