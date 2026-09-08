from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .embeddings import load_track_embeddings
from .utils import bool_mask, l2_normalize


LABEL_COLUMNS = [
    "tracklet_id",
    "pseudo_id",
    "session_id",
    "video_id",
    "cow_id",
    "confirmed",
    "assignment_source",
    "notes",
    "updated_at",
]

PAIR_REVIEW_COLUMNS = [
    "left_tracklet_id",
    "right_tracklet_id",
    "decision",
    "cow_id",
    "similarity",
    "notes",
    "updated_at",
]

IDENTITY_EXCLUSION_COLUMNS = ["cow_id", "tracklet_id", "reason", "updated_at"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _backup_csv(path: Path, category: str) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = path.parent / "label_backups" / f"{category}.{stamp}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    return target


def normalize_cow_id(value: object) -> str:
    """Normalize friendly inputs such as cow1 to the stable COW_0001 form."""
    text = "" if pd.isna(value) else str(value).strip()
    match = re.fullmatch(r"cow[\s_-]*0*(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"COW_{int(match.group(1)):04d}"
    return text


def next_cow_id(labels: pd.DataFrame) -> str:
    used: list[int] = []
    if "cow_id" in labels:
        for value in labels["cow_id"].fillna("").astype(str):
            match = re.fullmatch(r"COW_(\d+)", value.strip(), flags=re.IGNORECASE)
            if match:
                used.append(int(match.group(1)))
    return f"COW_{max(used, default=0) + 1:04d}"


def _cow_id_sort_key(cow_id: str) -> tuple[int, int | str]:
    normalized = normalize_cow_id(cow_id)
    match = re.fullmatch(r"COW_(\d+)", normalized, flags=re.IGNORECASE)
    return (0, int(match.group(1))) if match else (1, normalized)


def load_identity_labels(run_dir: str | Path) -> pd.DataFrame:
    """Load one canonical identity row per valid tracklet, including legacy labels."""
    run_dir = Path(run_dir).resolve()
    tracks = pd.read_csv(run_dir / "tracklets.csv")
    if "valid" in tracks:
        tracks = tracks.loc[bool_mask(tracks["valid"], default=False)].copy()
    for column in ("video_id", "session_id"):
        if column not in tracks:
            tracks[column] = ""
    base = tracks[["tracklet_id", "session_id", "video_id"]].drop_duplicates("tracklet_id").copy()

    assignments_path = run_dir / "pseudo_id_assignments.csv"
    if assignments_path.exists():
        assignments = pd.read_csv(assignments_path)
        if {"tracklet_id", "pseudo_id"}.issubset(assignments.columns):
            base = base.merge(
                assignments[["tracklet_id", "pseudo_id"]].drop_duplicates("tracklet_id"),
                on="tracklet_id",
                how="left",
            )
    if "pseudo_id" not in base:
        base["pseudo_id"] = ""

    base["cow_id"] = ""
    base["confirmed"] = False
    base["assignment_source"] = ""
    base["notes"] = ""
    base["updated_at"] = ""

    existing_path = run_dir / "labels.csv"
    if not existing_path.exists() and (run_dir / "labels_template.csv").exists():
        existing_path = run_dir / "labels_template.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path, dtype={"cow_id": str})
        if "tracklet_id" in existing:
            existing = existing.drop_duplicates("tracklet_id", keep="last").set_index("tracklet_id")
            indexed = base.set_index("tracklet_id")
            for column in ("cow_id", "confirmed", "assignment_source", "notes", "updated_at"):
                if column in existing:
                    values = existing[column].reindex(indexed.index)
                    indexed[column] = values.where(values.notna(), indexed[column])
            base = indexed.reset_index()

    base["cow_id"] = base["cow_id"].map(normalize_cow_id)
    base["confirmed"] = bool_mask(base["confirmed"], default=False) & base["cow_id"].ne("")
    for column in LABEL_COLUMNS:
        if column not in base:
            base[column] = ""
    return base[LABEL_COLUMNS].sort_values(["cow_id", "session_id", "tracklet_id"]).reset_index(drop=True)


def save_identity_labels(labels: pd.DataFrame, run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).resolve()
    frame = labels.copy()
    if frame["tracklet_id"].duplicated().any():
        raise ValueError("Each tracklet must have exactly one identity row.")
    frame["cow_id"] = frame["cow_id"].map(normalize_cow_id)
    frame["confirmed"] = bool_mask(frame["confirmed"], default=False) & frame["cow_id"].ne("")
    for column in LABEL_COLUMNS:
        if column not in frame:
            frame[column] = ""
    path = run_dir / "labels.csv"
    _backup_csv(path, "labels")
    _atomic_csv(frame[LABEL_COLUMNS], path)
    return path


def assign_tracklets(
    labels: pd.DataFrame,
    tracklet_ids: Iterable[str],
    cow_id: str | None = None,
    source: str = "manual",
    notes: str = "",
) -> tuple[pd.DataFrame, str]:
    frame = labels.copy()
    selected = {str(value) for value in tracklet_ids}
    unknown = selected.difference(frame["tracklet_id"].astype(str))
    if unknown:
        raise ValueError(f"Unknown tracklet IDs: {sorted(unknown)}")
    resolved = normalize_cow_id(cow_id) if cow_id else next_cow_id(frame)
    if not resolved:
        raise ValueError("Cow_ID cannot be empty.")
    mask = frame["tracklet_id"].astype(str).isin(selected)
    frame.loc[mask, "cow_id"] = resolved
    frame.loc[mask, "confirmed"] = True
    frame.loc[mask, "assignment_source"] = source
    frame.loc[mask, "notes"] = notes
    frame.loc[mask, "updated_at"] = _now()
    return frame, resolved


def unassign_tracklets(labels: pd.DataFrame, tracklet_ids: Iterable[str]) -> pd.DataFrame:
    frame = labels.copy()
    selected = {str(value) for value in tracklet_ids}
    mask = frame["tracklet_id"].astype(str).isin(selected)
    frame.loc[mask, ["cow_id", "assignment_source", "notes", "updated_at"]] = ""
    frame.loc[mask, "confirmed"] = False
    return frame


def rename_identity(labels: pd.DataFrame, old_cow_id: str, new_cow_id: str) -> pd.DataFrame:
    frame = labels.copy()
    old_value = normalize_cow_id(old_cow_id)
    new_value = normalize_cow_id(new_cow_id)
    if not new_value:
        raise ValueError("New Cow_ID cannot be empty.")
    existing = set(frame["cow_id"].fillna("").map(normalize_cow_id))
    if new_value != old_value and new_value in existing:
        raise ValueError(f"{new_value} already exists. Reassign tracklets explicitly if you intend to merge identities.")
    mask = frame["cow_id"].fillna("").map(normalize_cow_id).eq(old_value)
    if not mask.any():
        raise ValueError(f"Identity not found: {old_value}")
    frame.loc[mask, "cow_id"] = new_value
    frame.loc[mask, "assignment_source"] = "identity_rename"
    frame.loc[mask, "updated_at"] = _now()
    return frame


def merge_identities(
    labels: pd.DataFrame,
    cow_ids: Iterable[str],
    target_cow_id: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Merge complete identity groups, never only the two visible pair members."""
    frame = labels.copy()
    sources = {normalize_cow_id(value) for value in cow_ids if normalize_cow_id(value)}
    if not sources:
        raise ValueError("No identity groups were supplied for merging.")
    resolved = normalize_cow_id(target_cow_id) if target_cow_id else sorted(sources, key=_cow_id_sort_key)[0]
    if not resolved:
        raise ValueError("Merge target Cow_ID cannot be empty.")

    assigned_ids = frame["cow_id"].fillna("").map(normalize_cow_id)
    missing = sources.difference(set(assigned_ids))
    if missing:
        raise ValueError(f"Identity groups not found: {sorted(missing)}")
    mask = assigned_ids.isin(sources)
    frame.loc[mask, "cow_id"] = resolved
    frame.loc[mask, "confirmed"] = True
    frame.loc[mask, "assignment_source"] = "identity_merge"
    frame.loc[mask, "updated_at"] = _now()
    return frame, resolved


def identity_catalog(labels: pd.DataFrame) -> pd.DataFrame:
    assigned = labels.loc[bool_mask(labels["confirmed"], default=False) & labels["cow_id"].fillna("").ne("")]
    columns = ["cow_id", "tracklets", "videos", "sessions"]
    if assigned.empty:
        return pd.DataFrame(columns=columns)
    return (
        assigned.groupby("cow_id", as_index=False)
        .agg(
            tracklets=("tracklet_id", "nunique"),
            videos=("video_id", "nunique"),
            sessions=("session_id", "nunique"),
        )
        .sort_values("cow_id")
        .reset_index(drop=True)
    )


def identity_candidates(
    run_dir: str | Path,
    labels: pd.DataFrame,
    cow_id: str,
    top_k: int = 20,
    exclude_assigned: bool = True,
    exclude_seen_sessions: bool = True,
) -> pd.DataFrame:
    """Rank tracklets against the mean embedding of an existing manual identity."""
    run_dir = Path(run_dir).resolve()
    resolved = normalize_cow_id(cow_id)
    data = load_track_embeddings(run_dir / "track_embeddings.npz")
    tracklet_ids = data["tracklet_ids"].astype(str)
    embeddings = l2_normalize(data["embeddings"])
    index = {tracklet_id: offset for offset, tracklet_id in enumerate(tracklet_ids)}
    owned = labels.loc[
        bool_mask(labels["confirmed"], default=False) & labels["cow_id"].astype(str).eq(resolved)
    ]
    owned_indices = [index[value] for value in owned["tracklet_id"].astype(str) if value in index]
    if not owned_indices:
        raise ValueError(f"{resolved} has no embedded tracklets.")
    prototype = l2_normalize(np.mean(embeddings[np.asarray(owned_indices, dtype=int)], axis=0))
    scores = embeddings @ prototype

    tracks = pd.read_csv(run_dir / "tracklets.csv").drop_duplicates("tracklet_id").set_index("tracklet_id")
    label_lookup = labels.drop_duplicates("tracklet_id").set_index("tracklet_id")
    owned_tracklets = set(owned["tracklet_id"].astype(str))
    owned_sessions = set(owned["session_id"].astype(str))
    exclusions = load_identity_exclusions(run_dir)
    excluded_tracklets = set(exclusions.loc[exclusions["cow_id"].eq(resolved), "tracklet_id"].astype(str))
    rows: list[dict[str, object]] = []
    for offset in np.argsort(scores)[::-1]:
        tracklet_id = str(tracklet_ids[offset])
        if tracklet_id in owned_tracklets or tracklet_id not in label_lookup.index:
            continue
        if tracklet_id in excluded_tracklets:
            continue
        label = label_lookup.loc[tracklet_id]
        current_cow_id = normalize_cow_id(label.get("cow_id", ""))
        confirmed = bool_mask(pd.Series([label.get("confirmed", False)]), default=False).iloc[0]
        if exclude_assigned and confirmed and current_cow_id:
            continue
        session_id = str(label.get("session_id", ""))
        if exclude_seen_sessions and session_id in owned_sessions:
            continue
        track = tracks.loc[tracklet_id] if tracklet_id in tracks.index else pd.Series(dtype=object)
        rows.append(
            {
                "tracklet_id": tracklet_id,
                "similarity": float(scores[offset]),
                "session_id": session_id,
                "video_id": str(label.get("video_id", "")),
                "source_filename": str(track.get("source_filename", "")),
                "start_s": float(track.get("start_s", 0.0)),
                "contact_path": str(track.get("contact_path", "")),
                "current_cow_id": current_cow_id,
            }
        )
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def load_pair_reviews(run_dir: str | Path) -> pd.DataFrame:
    path = Path(run_dir).resolve() / "pair_reviews.csv"
    if not path.exists():
        return pd.DataFrame(columns=PAIR_REVIEW_COLUMNS)
    frame = pd.read_csv(path, dtype={"cow_id": str})
    for column in PAIR_REVIEW_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[PAIR_REVIEW_COLUMNS]


def load_identity_exclusions(run_dir: str | Path) -> pd.DataFrame:
    path = Path(run_dir).resolve() / "identity_exclusions.csv"
    if not path.exists():
        return pd.DataFrame(columns=IDENTITY_EXCLUSION_COLUMNS)
    frame = pd.read_csv(path, dtype={"cow_id": str, "tracklet_id": str})
    for column in IDENTITY_EXCLUSION_COLUMNS:
        if column not in frame:
            frame[column] = ""
    frame["cow_id"] = frame["cow_id"].map(normalize_cow_id)
    return frame[IDENTITY_EXCLUSION_COLUMNS]


def reject_identity_candidate(
    run_dir: str | Path,
    cow_id: str,
    tracklet_id: str,
    reason: str = "human_rejected_identity_candidate",
) -> Path:
    path = Path(run_dir).resolve() / "identity_exclusions.csv"
    frame = load_identity_exclusions(run_dir)
    resolved = normalize_cow_id(cow_id)
    tracklet = str(tracklet_id)
    mask = frame["cow_id"].eq(resolved) & frame["tracklet_id"].astype(str).eq(tracklet)
    frame = frame.loc[~mask].copy()
    frame.loc[len(frame)] = {"cow_id": resolved, "tracklet_id": tracklet, "reason": reason, "updated_at": _now()}
    _atomic_csv(frame[IDENTITY_EXCLUSION_COLUMNS], path)
    return path


def _different_constraints_within(
    run_dir: str | Path,
    tracklet_ids: set[str],
    ignored_pair: tuple[str, str] | None = None,
) -> pd.DataFrame:
    reviews = load_pair_reviews(run_dir)
    if reviews.empty:
        return reviews
    ignored = tuple(sorted(ignored_pair)) if ignored_pair else None
    keep = []
    for _, row in reviews.iterrows():
        left = str(row["left_tracklet_id"])
        right = str(row["right_tracklet_id"])
        keep.append(
            str(row["decision"]) == "different"
            and left in tracklet_ids
            and right in tracklet_ids
            and tuple(sorted((left, right))) != ignored
        )
    return reviews.loc[keep].copy()


def assign_tracklets_checked(
    run_dir: str | Path,
    labels: pd.DataFrame,
    tracklet_ids: Iterable[str],
    cow_id: str | None = None,
    source: str = "manual",
    notes: str = "",
) -> tuple[pd.DataFrame, str]:
    selected = {str(value) for value in tracklet_ids}
    resolved = normalize_cow_id(cow_id) if cow_id else next_cow_id(labels)
    current_members = set(
        labels.loc[
            bool_mask(labels["confirmed"], default=False)
            & labels["cow_id"].fillna("").map(normalize_cow_id).eq(resolved),
            "tracklet_id",
        ].astype(str)
    )
    violations = _different_constraints_within(run_dir, selected | current_members)
    if not violations.empty:
        first = violations.iloc[0]
        raise ValueError(
            "Bu atama daha önce verdiğin 'farklı inek' kararıyla çelişiyor: "
            f"{first['left_tracklet_id']} ↔ {first['right_tracklet_id']}. "
            "Önce Çelişki denetimi sekmesinde kararı düzelt."
        )
    updated, resolved = assign_tracklets(labels, selected, resolved, source=source, notes=notes)
    exclusions = load_identity_exclusions(run_dir)
    if not exclusions.empty:
        exclusions = exclusions.loc[
            ~(exclusions["cow_id"].eq(resolved) & exclusions["tracklet_id"].astype(str).isin(selected))
        ]
        _atomic_csv(exclusions[IDENTITY_EXCLUSION_COLUMNS], Path(run_dir).resolve() / "identity_exclusions.csv")
    return updated, resolved


def record_pair_as_different(
    run_dir: str | Path,
    labels: pd.DataFrame,
    left_tracklet_id: str,
    right_tracklet_id: str,
    similarity: float | None = None,
) -> Path:
    lookup = labels.drop_duplicates("tracklet_id").set_index("tracklet_id")
    left, right = str(left_tracklet_id), str(right_tracklet_id)
    left_id = normalize_cow_id(lookup.loc[left, "cow_id"]) if left in lookup.index else ""
    right_id = normalize_cow_id(lookup.loc[right, "cow_id"]) if right in lookup.index else ""
    if left_id and left_id == right_id:
        raise ValueError(
            f"İki geçiş şu anda {left_id} altında. 'Farklı' kaydetmeden önce yanlış olan geçişi "
            "Çelişki denetimi veya İnek galerisi sekmesinden kimlikten çıkar."
        )
    return save_pair_review(run_dir, left, right, "different", similarity=similarity)


def save_pair_review(
    run_dir: str | Path,
    left_tracklet_id: str,
    right_tracklet_id: str,
    decision: str,
    cow_id: str = "",
    similarity: float | None = None,
    notes: str = "",
) -> Path:
    if decision not in {"same", "different", "unsure"}:
        raise ValueError("Pair decision must be 'same', 'different' or 'unsure'.")
    path = Path(run_dir).resolve() / "pair_reviews.csv"
    frame = load_pair_reviews(run_dir)
    left, right = _canonical_pair(left_tracklet_id, right_tracklet_id)
    key = frame["left_tracklet_id"].astype(str).eq(left) & frame["right_tracklet_id"].astype(str).eq(right)
    frame = frame.loc[~key].copy()
    frame.loc[len(frame)] = {
        "left_tracklet_id": left,
        "right_tracklet_id": right,
        "decision": decision,
        "cow_id": normalize_cow_id(cow_id),
        "similarity": np.nan if similarity is None else float(similarity),
        "notes": notes,
        "updated_at": _now(),
    }
    _backup_csv(path, "pair_reviews")
    _atomic_csv(frame[PAIR_REVIEW_COLUMNS], path)
    return path


def accept_pair_as_same(
    run_dir: str | Path,
    labels: pd.DataFrame,
    left_tracklet_id: str,
    right_tracklet_id: str,
    requested_cow_id: str | None = None,
    similarity: float | None = None,
    source: str = "pair_review",
) -> tuple[pd.DataFrame, str]:
    pair_ids = [str(left_tracklet_id), str(right_tracklet_id)]
    existing = labels.loc[
        labels["tracklet_id"].astype(str).isin(pair_ids)
        & bool_mask(labels["confirmed"], default=False)
        & labels["cow_id"].fillna("").ne("")
    ]["cow_id"].map(normalize_cow_id).unique().tolist()
    resolved = (
        normalize_cow_id(requested_cow_id)
        if requested_cow_id
        else (sorted(existing, key=_cow_id_sort_key)[0] if existing else next_cow_id(labels))
    )
    updated = labels.copy()
    all_assigned = set(updated["cow_id"].fillna("").map(normalize_cow_id))
    merge_sources = set(existing)
    if resolved in all_assigned:
        merge_sources.add(resolved)
    union_members = set(pair_ids)
    if merge_sources:
        union_members.update(
            updated.loc[
                updated["cow_id"].fillna("").map(normalize_cow_id).isin(merge_sources),
                "tracklet_id",
            ].astype(str)
        )
    violations = _different_constraints_within(run_dir, union_members, ignored_pair=(pair_ids[0], pair_ids[1]))
    if not violations.empty:
        first = violations.iloc[0]
        raise ValueError(
            "Bu birleştirme kimlik grubundaki eski bir 'farklı inek' kararıyla çelişiyor: "
            f"{first['left_tracklet_id']} ↔ {first['right_tracklet_id']}. "
            "Önce Çelişki denetimi sekmesinden grubu düzelt."
        )
    if len(merge_sources) > 1 or (merge_sources and resolved not in merge_sources):
        updated, resolved = merge_identities(updated, merge_sources, resolved)
    updated, resolved = assign_tracklets(updated, pair_ids, resolved, source=source)
    save_identity_labels(updated, run_dir)
    merge_note = ""
    merged_away = sorted(set(existing).difference({resolved}), key=_cow_id_sort_key)
    if merged_away:
        merge_note = f"Merged {', '.join(merged_away)} into {resolved}"
    save_pair_review(run_dir, pair_ids[0], pair_ids[1], "same", resolved, similarity, notes=merge_note)
    return updated, resolved


def auto_confirm_dual_signal(
    run_dir: str | Path,
    labels: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    cosine_threshold: float = 0.95,
    geometry_threshold: float = 0.5,
    source: str = "auto_dual_signal",
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Auto-confirm only pairs where two independent signals agree.

    Requires mutual top-1 (each tracklet is the other's single best candidate,
    not just a one-directional best guess), appearance cosine_similarity above
    the same threshold already validated for human-reviewed matches, AND the
    flip-insensitive geometry_similarity signal agreeing. Any pair that would
    conflict with a prior human 'different' decision is skipped, not forced.
    Intentionally conservative: this is meant to clear the small, unambiguous
    tier only, not to replace human review of the rest of the backlog.
    """
    updated = labels.copy()
    top1 = candidate_pairs.loc[candidate_pairs["rank"] == 1].set_index("query_tracklet_id")
    reviews = load_pair_reviews(run_dir)
    different_pairs = {
        _canonical_pair(row["left_tracklet_id"], row["right_tracklet_id"])
        for _, row in reviews.iterrows()
        if str(row["decision"]) == "different"
    }
    excluded_from_cow = {
        (str(row["cow_id"]), str(row["tracklet_id"])) for _, row in load_identity_exclusions(run_dir).iterrows()
    }
    applied: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for query_id, row in top1.iterrows():
        candidate_id = str(row["candidate_tracklet_id"])
        cosine = row["cosine_similarity"]
        geometry = row.get("geometry_similarity")
        if pd.isna(cosine) or pd.isna(geometry):
            continue
        if cosine < cosine_threshold or geometry < geometry_threshold:
            continue
        if candidate_id not in top1.index or str(top1.loc[candidate_id, "candidate_tracklet_id"]) != str(query_id):
            continue  # not a mutual top-1 match
        pair_key = tuple(sorted((str(query_id), candidate_id)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        if pair_key in different_pairs:
            continue  # a human directly reviewed this exact pair as different; never auto-override that
        left_cow = updated.set_index("tracklet_id")["cow_id"].get(pair_key[0], "")
        right_cow = updated.set_index("tracklet_id")["cow_id"].get(pair_key[1], "")
        if pd.notna(left_cow) and pd.notna(right_cow) and normalize_cow_id(left_cow) == normalize_cow_id(right_cow) and left_cow:
            continue  # already the same identity, nothing to do
        if (normalize_cow_id(right_cow), pair_key[0]) in excluded_from_cow or (
            normalize_cow_id(left_cow),
            pair_key[1],
        ) in excluded_from_cow:
            continue  # a human clicked "Bu inek değil" on this exact candidate/identity pairing
        try:
            updated, resolved = accept_pair_as_same(
                run_dir, updated, pair_key[0], pair_key[1], similarity=float(cosine), source=source
            )
        except ValueError:
            continue  # conflicts with a prior human 'different' decision; leave for manual review
        applied.append(
            {
                "left_tracklet_id": pair_key[0],
                "right_tracklet_id": pair_key[1],
                "cow_id": resolved,
                "cosine_similarity": float(cosine),
                "geometry_similarity": float(geometry),
            }
        )
    return updated, applied
