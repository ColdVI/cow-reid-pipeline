from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import audit_labels
from .embeddings import load_track_embeddings
from .review_state import load_excluded_tracklets
from .utils import bool_mask, l2_normalize, save_json


def build_gallery(run_dir: str | Path, labels_csv: str | Path, output_path: str | Path | None = None) -> Path:
    run_dir = Path(run_dir).resolve()
    labels = pd.read_csv(labels_csv)
    required = {"tracklet_id", "cow_id"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Labels need columns: {sorted(required)}")
    if "confirmed" in labels.columns:
        audited = audit_labels(run_dir, labels_csv)
        labels = pd.read_csv(audited["safe_labels"], dtype={"cow_id": str})
    labels["cow_id"] = labels["cow_id"].fillna("").astype(str).str.strip()
    labels = labels.loc[labels["cow_id"].ne("")]
    if "confirmed" in labels.columns:
        labels = labels.loc[bool_mask(labels["confirmed"])]
    if "training_eligible" in labels.columns:
        labels = labels.loc[bool_mask(labels["training_eligible"])]
    # A human-flagged broken/unresolved-split tracklet must never become a
    # gallery prototype, even if its labels.csv row is confirmed -- defense in
    # depth alongside build_track_embeddings' own exclusion, since a gallery
    # can be rebuilt from an embeddings file produced before a review flag
    # was recorded.
    excluded_tracklets = load_excluded_tracklets(run_dir)
    if excluded_tracklets:
        labels = labels.loc[~labels["tracklet_id"].astype(str).isin(excluded_tracklets)]
    data = load_track_embeddings(run_dir / "track_embeddings.npz")
    lookup = {str(tracklet_id): embedding for tracklet_id, embedding in zip(data["tracklet_ids"], data["embeddings"])}
    cow_ids: list[str] = []
    prototypes: list[np.ndarray] = []
    counts: list[int] = []
    prototype_tracklet_ids: list[str] = []
    prototype_flanks: list[str] = []
    prototype_views: list[str] = []
    prototype_domains: list[str] = []
    metadata_path = run_dir / "tracklets.csv"
    metadata = pd.read_csv(metadata_path).drop_duplicates("tracklet_id").set_index("tracklet_id") if metadata_path.exists() else pd.DataFrame()
    for cow_id, group in labels.groupby("cow_id"):
        selected = [(str(tracklet_id), lookup[str(tracklet_id)]) for tracklet_id in group["tracklet_id"] if str(tracklet_id) in lookup]
        for tracklet_id, embedding in selected:
            row = metadata.loc[tracklet_id] if not metadata.empty and tracklet_id in metadata.index else pd.Series(dtype=object)
            cow_ids.append(str(cow_id))
            prototypes.append(l2_normalize(embedding))
            counts.append(1)
            prototype_tracklet_ids.append(tracklet_id)
            prototype_flanks.append(str(row.get("flank", "unknown")))
            prototype_views.append(str(row.get("view", row.get("camera_id", "unknown"))))
            prototype_domains.append(str(row.get("domain", "unknown")))
    if not prototypes:
        raise RuntimeError("No labelled tracklets matched the embedding file.")
    output = Path(output_path) if output_path else run_dir / "cow_gallery.npz"
    np.savez_compressed(
        output,
        cow_ids=np.asarray(cow_ids, dtype=str),
        prototypes=np.vstack(prototypes).astype(np.float32),
        tracklet_counts=np.asarray(counts, dtype=np.int32),
        prototype_tracklet_ids=np.asarray(prototype_tracklet_ids, dtype=str),
        flanks=np.asarray(prototype_flanks, dtype=str),
        views=np.asarray(prototype_views, dtype=str),
        domains=np.asarray(prototype_domains, dtype=str),
        backend=np.asarray(data.get("backend", np.asarray(["unknown"])), dtype=str),
        checkpoint=np.asarray(data.get("checkpoint", np.asarray(["unknown"])), dtype=str),
        checkpoint_sha256=np.asarray(data.get("checkpoint_sha256", np.asarray([""])), dtype=str),
        preprocessing_profile=np.asarray(data.get("preprocessing_profile", np.asarray(["unknown"])), dtype=str),
        crop_version=np.asarray(data.get("crop_version", np.asarray(["unknown"])), dtype=str),
        embedding_schema_version=np.asarray(data.get("embedding_schema_version", np.asarray(["unknown"])), dtype=str),
    )
    return output


def identify_run(
    run_dir: str | Path,
    gallery_path: str | Path,
    config: dict[str, Any],
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    run_dir = Path(run_dir).resolve()
    tracks = load_track_embeddings(run_dir / "track_embeddings.npz")
    version_fields = ("backend", "checkpoint", "checkpoint_sha256", "preprocessing_profile", "crop_version", "embedding_schema_version")
    with np.load(gallery_path, allow_pickle=False) as gallery:
        cow_ids = gallery["cow_ids"]
        prototypes = l2_normalize(gallery["prototypes"])
        gallery_fields = {name: str(gallery[name][0]) for name in version_fields if name in gallery.files and len(gallery[name]) > 0}
    track_fields = {name: str(tracks[name][0]) for name in version_fields if name in tracks and len(tracks[name]) > 0}
    # Content-based compatibility: a checkpoint file can be overwritten in place
    # (this repo's own workflow does exactly that) while its path stays the
    # same, so path/backend-name equality alone would silently accept a
    # gallery and a query built from two different embedding spaces. Hash and
    # preprocessing/crop/schema identity are the actual gate; backend/path
    # strings are kept in the files only for human-readable debugging.
    unverifiable = (None, "", "unknown")
    hard_fields = ("checkpoint_sha256", "preprocessing_profile", "crop_version", "embedding_schema_version")
    for field in hard_fields:
        gallery_value = gallery_fields.get(field)
        track_value = track_fields.get(field)
        if gallery_value in unverifiable or track_value in unverifiable:
            # Missing/unknown on either side (e.g. a gallery/run built before
            # this contract existed) -- cannot verify, so fall through to the
            # legacy backend/checkpoint-path check below instead of
            # pretending a match.
            continue
        if gallery_value != track_value:
            raise ValueError(
                f"Gallery and query embeddings disagree on {field} "
                f"(gallery={gallery_value!r}, query={track_value!r}). Re-embed both "
                "the gallery and the query run with the same checkpoint/crop pipeline."
            )
    if not all(gallery_fields.get(field) not in unverifiable and track_fields.get(field) not in unverifiable for field in hard_fields):
        gallery_backend = gallery_fields.get("backend", "unknown")
        gallery_checkpoint = gallery_fields.get("checkpoint", "unknown")
        track_backend = track_fields.get("backend", "unknown")
        track_checkpoint = track_fields.get("checkpoint", "unknown")
        if gallery_backend != "unknown" and track_backend != "unknown" and gallery_backend != track_backend:
            raise ValueError(f"Gallery uses {gallery_backend}, but query run uses {track_backend}. Re-embed both with one checkpoint.")
        if gallery_checkpoint != "unknown" and track_checkpoint != "unknown" and gallery_checkpoint != track_checkpoint:
            raise ValueError("Gallery and query embeddings were generated from different checkpoints.")
    queries = l2_normalize(tracks["embeddings"])
    scores = queries @ prototypes.T
    threshold = float(config["matching"].get("unknown_threshold", 0.95))
    unique_cows = np.asarray(sorted(set(cow_ids.astype(str))), dtype=str)
    top_k = min(int(config["matching"].get("top_k", 5)), len(unique_cows))
    min_margin = float(config["matching"].get("min_margin", 0.02))
    rows: list[dict[str, Any]] = []
    for index, tracklet_id in enumerate(tracks["tracklet_ids"]):
        cow_scores = []
        for cow_id in unique_cows:
            values = np.sort(scores[index, cow_ids.astype(str) == cow_id])[::-1]
            # Best view is primary; a second matching prototype stabilizes the score.
            combined = float(values[0] if len(values) == 1 else 0.7 * values[0] + 0.3 * np.median(values[:3]))
            cow_scores.append(combined)
        ranked = np.argsort(cow_scores)[::-1][:top_k]
        best = int(ranked[0])
        best_score = float(cow_scores[best])
        second_score = float(cow_scores[int(ranked[1])]) if len(ranked) > 1 else -1.0
        ambiguous = best_score >= threshold and best_score - second_score < min_margin
        prediction = "UNKNOWN" if best_score < threshold else ("AMBIGUOUS" if ambiguous else str(unique_cows[best]))
        row: dict[str, Any] = {
            "tracklet_id": str(tracklet_id),
            "session_id": str(tracks["session_ids"][index]),
            "predicted_cow_id": prediction,
            "confidence_similarity": best_score,
            "is_unknown": best_score < threshold,
            "is_ambiguous": ambiguous,
            "top2_margin": best_score - second_score,
        }
        for rank, candidate_index in enumerate(ranked, start=1):
            row[f"top_{rank}_cow_id"] = str(unique_cows[candidate_index])
            row[f"top_{rank}_similarity"] = float(cow_scores[candidate_index])
        rows.append(row)
    result = pd.DataFrame(rows)
    output = Path(output_csv) if output_csv else run_dir / "identity_predictions.csv"
    result.to_csv(output, index=False)
    save_json(
        run_dir / "identity_summary.json",
        {
            "queries": len(result),
            "known_predictions": int((~result["is_unknown"]).sum()),
            "unknown_predictions": int(result["is_unknown"].sum()),
            "ambiguous_predictions": int(result["is_ambiguous"].sum()),
            "threshold": threshold,
            "min_margin": min_margin,
        },
    )
    return result
