from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .embeddings import load_track_embeddings
from .utils import l2_normalize, save_json


def constrained_greedy_cluster(
    tracklet_ids: np.ndarray,
    embeddings: np.ndarray,
    session_ids: np.ndarray,
    threshold: float,
    same_session_cannot_link: bool = True,
    min_margin: float = 0.0,
) -> pd.DataFrame:
    embeddings = l2_normalize(embeddings)
    clusters: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for index, (tracklet_id, embedding, session_id) in enumerate(zip(tracklet_ids, embeddings, session_ids)):
        best_cluster = None
        best_similarity = -1.0
        eligible_similarities: list[float] = []
        for cluster_index, cluster in enumerate(clusters):
            if same_session_cannot_link and str(session_id) in cluster["sessions"]:
                continue
            similarity = float(np.dot(embedding, cluster["prototype"]))
            eligible_similarities.append(similarity)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_index
        ranked_similarities = sorted(eligible_similarities, reverse=True)
        second_best = ranked_similarities[1] if len(ranked_similarities) > 1 else -1.0
        margin = best_similarity - second_best if second_best >= 0 else float("inf")
        create_new = best_cluster is None or best_similarity < threshold or margin < min_margin
        if create_new:
            cluster_id = len(clusters)
            clusters.append(
                {
                    "prototype": embedding.copy(),
                    "members": [index],
                    "sessions": {str(session_id)},
                }
            )
            similarity_to_cluster = 1.0
            assignment_type = "seed"
        else:
            cluster_id = best_cluster
            cluster = clusters[cluster_id]
            cluster["members"].append(index)
            cluster["sessions"].add(str(session_id))
            cluster["prototype"] = l2_normalize(
                np.mean(embeddings[np.asarray(cluster["members"], dtype=int)], axis=0)
            )
            similarity_to_cluster = best_similarity
            assignment_type = "matched"
        assignments.append(
            {
                "tracklet_id": str(tracklet_id),
                "session_id": str(session_id),
                "pseudo_id": f"PSEUDO_{cluster_id + 1:04d}",
                "similarity_to_cluster": similarity_to_cluster,
                "candidate_similarity": best_similarity if best_cluster is not None else np.nan,
                "candidate_margin": margin if np.isfinite(margin) else np.nan,
                "assignment_type": assignment_type,
                "needs_review": assignment_type == "matched"
                and (similarity_to_cluster < min(0.99, threshold + 0.02) or margin < min_margin + 0.02),
            }
        )
    return pd.DataFrame(assignments)


def candidate_pairs(
    tracklet_ids: np.ndarray,
    embeddings: np.ndarray,
    session_ids: np.ndarray,
    top_k: int = 5,
    geometry_embeddings: np.ndarray | None = None,
) -> pd.DataFrame:
    columns = [
        "query_tracklet_id",
        "query_session_id",
        "rank",
        "candidate_tracklet_id",
        "candidate_session_id",
        "cosine_similarity",
        "geometry_similarity",
    ]
    embeddings = l2_normalize(embeddings)
    similarity = embeddings @ embeddings.T
    geometry_similarity = None
    if geometry_embeddings is not None:
        geometry_normed = l2_normalize(geometry_embeddings)
        geometry_similarity = geometry_normed @ geometry_normed.T
    rows: list[dict[str, Any]] = []
    for query_index, query_id in enumerate(tracklet_ids):
        allowed = np.asarray(session_ids != session_ids[query_index])
        allowed[query_index] = False
        candidate_indices = np.flatnonzero(allowed)
        if not len(candidate_indices):
            continue
        ranked = candidate_indices[np.argsort(similarity[query_index, candidate_indices])[::-1]][:top_k]
        for rank, candidate_index in enumerate(ranked, start=1):
            rows.append(
                {
                    "query_tracklet_id": str(query_id),
                    "query_session_id": str(session_ids[query_index]),
                    "rank": rank,
                    "candidate_tracklet_id": str(tracklet_ids[candidate_index]),
                    "candidate_session_id": str(session_ids[candidate_index]),
                    "cosine_similarity": float(similarity[query_index, candidate_index]),
                    "geometry_similarity": (
                        float(geometry_similarity[query_index, candidate_index])
                        if geometry_similarity is not None
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def cluster_run(run_dir: str | Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = Path(run_dir).resolve()
    data = load_track_embeddings(run_dir / "track_embeddings.npz")
    matching = config["matching"]
    assignments = constrained_greedy_cluster(
        data["tracklet_ids"],
        data["embeddings"],
        data["session_ids"],
        threshold=float(matching.get("similarity_threshold", 0.95)),
        same_session_cannot_link=bool(matching.get("same_session_cannot_link", True)),
        min_margin=float(matching.get("min_margin", 0.0)),
    )
    pairs = candidate_pairs(
        data["tracklet_ids"],
        data["embeddings"],
        data["session_ids"],
        top_k=int(matching.get("top_k", 5)),
        geometry_embeddings=data.get("geometry_embeddings"),
    )
    assignments.to_csv(run_dir / "pseudo_id_assignments.csv", index=False)
    pairs.to_csv(run_dir / "candidate_pairs.csv", index=False)
    sizes = assignments.groupby("pseudo_id").size() if not assignments.empty else pd.Series(dtype=int)
    save_json(
        run_dir / "matching_summary.json",
        {
            "tracklets": int(len(assignments)),
            "pseudo_id_clusters": int(assignments["pseudo_id"].nunique()) if not assignments.empty else 0,
            "multi_session_clusters": int((sizes > 1).sum()),
            "similarity_threshold": float(matching.get("similarity_threshold", 0.95)),
            "min_margin": float(matching.get("min_margin", 0.0)),
            "warning": "Pseudo IDs are appearance-based hypotheses until a human, ear tag, or RFID confirms them.",
        },
    )
    return assignments, pairs
