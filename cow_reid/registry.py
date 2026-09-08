from __future__ import annotations

import json
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np

from .db import connect, init_database
from .inventory import scan_videos
from .overlap import assign_overlap_groups
from .utils import sha256_file
from .utils import bool_mask


def ingest_directory(
    videos: str | Path,
    camera_id: str,
    database_url: str,
    use_ocr: bool = True,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    init_database(database_url)
    manifest = scan_videos(videos, manifest_path, use_ocr=use_ocr, camera_id=camera_id)
    inserted = duplicate = 0
    with connect(database_url) as connection:
        for row in manifest.to_dict("records"):
            exists = connection.execute("SELECT video_uuid FROM videos WHERE sha256 = ?", (row["sha256"],)).fetchone()
            if exists:
                duplicate += 1
                continue
            connection.execute(
                """INSERT INTO videos(
                  video_uuid, sha256, perceptual_fingerprint, camera_id, recording_start, recording_end,
                  timestamp_source, timestamp_confidence, ocr_samples_json, canonical_video_uuid,
                  overlap_group_id, media_uri, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'INGESTED')""",
                (
                    row["video_id"], row["sha256"], row.get("perceptual_fingerprint"), camera_id,
                    _none(row.get("recording_start")), _none(row.get("recording_end")),
                    row.get("timestamp_source") or "unknown", float(row.get("timestamp_confidence") or 0),
                    row.get("ocr_samples_json") or "[]", row.get("canonical_video_id"),
                    row.get("overlap_group_id"), row["path"],
                ),
            )
            inserted += 1
    return {"discovered": len(manifest), "inserted": inserted, "duplicates": duplicate}


def _none(value: object) -> object:
    return None if value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip() else value


def pending_videos(database_url: str, status: str = "INGESTED") -> list[dict[str, object]]:
    with connect(database_url) as connection:
        return [dict(row) for row in connection.execute("SELECT * FROM videos WHERE status = ? ORDER BY created_at", (status,))]


def transition_video(database_url: str, video_uuid: str, expected: str, target: str, error: str | None = None) -> bool:
    with connect(database_url) as connection:
        cursor = connection.execute(
            "UPDATE videos SET status = ?, last_error = ?, attempt_count = attempt_count + 1 WHERE video_uuid = ? AND status = ?",
            (target, error, video_uuid, expected),
        )
        return cursor.rowcount == 1


def record_video_error(database_url: str, video_uuid: str, error: str) -> None:
    with connect(database_url) as connection:
        connection.execute(
            "UPDATE videos SET last_error=?, attempt_count=attempt_count+1 WHERE video_uuid=?",
            (str(error)[:4000], video_uuid),
        )


def register_model(database_url: str, checkpoint: str | Path, model_name: str = "reid", code_commit: str = "unknown", config_hash: str = "unknown") -> str:
    init_database(database_url)
    path = Path(checkpoint).resolve()
    digest = sha256_file(path)
    model_version = f"{model_name}:{digest[:16]}"
    with connect(database_url) as connection:
        connection.execute(
            "INSERT INTO model_registry(model_version,model_name,checkpoint_sha256,code_commit,config_hash) VALUES (?,?,?,?,?) ON CONFLICT(model_version) DO NOTHING",
            (model_version, model_name, digest, code_commit, config_hash),
        )
    return model_version


def register_tracklets(database_url: str, video_uuid: str, tracklets_csv: str | Path) -> int:
    frame = pd.read_csv(tracklets_csv)
    if "video_id" in frame:
        frame = frame.loc[frame["video_id"].astype(str).eq(str(video_uuid))].copy()
    inserted = 0
    with connect(database_url) as connection:
        for row in frame.to_dict("records"):
            metadata = {key: value for key, value in row.items() if key not in {"tracklet_id", "start_s", "end_s", "direction", "flank", "mean_quality"} and not (isinstance(value, float) and pd.isna(value))}
            quality = row.get("mean_quality", row.get("quality_score", 0))
            quality = 0.0 if pd.isna(quality) else float(quality)
            is_valid = bool_mask(pd.Series([row.get("valid", True)]), default=True).iloc[0]
            cursor = connection.execute(
                """INSERT INTO tracklets(tracklet_uuid,video_uuid,start_s,end_s,direction,flank,quality_score,review_state,metadata_json)
                VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(tracklet_uuid) DO NOTHING""",
                (str(row["tracklet_id"]), video_uuid, float(row.get("start_s", 0)), float(row.get("end_s", 0)), _none(row.get("direction")), _none(row.get("flank")), quality, "pending" if is_valid else "rejected", json.dumps(metadata, default=str)),
            )
            inserted += max(0, cursor.rowcount)
    return inserted


def register_embeddings(database_url: str, embeddings_path: str | Path, model_version: str) -> int:
    with np.load(embeddings_path, allow_pickle=False) as data:
        tracklet_ids = data["tracklet_ids"].astype(str)
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    inserted = 0
    with connect(database_url) as connection:
        for tracklet_id, embedding in zip(tracklet_ids, embeddings):
            embedding_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{tracklet_id}:{model_version}"))
            if connection.postgres:
                vector = "[" + ",".join(map(str, embedding.tolist())) + "]"
                cursor = connection.execute(
                    "INSERT INTO reid_embeddings(embedding_uuid,tracklet_uuid,model_version,embedding,dimensions) VALUES (?,?,?,?,?) ON CONFLICT(tracklet_uuid,model_version) DO NOTHING",
                    (embedding_uuid, tracklet_id, model_version, vector, len(embedding)),
                )
            else:
                cursor = connection.execute(
                    "INSERT INTO reid_embeddings(embedding_uuid,tracklet_uuid,model_version,embedding,dimensions) VALUES (?,?,?,?,?) ON CONFLICT(tracklet_uuid,model_version) DO NOTHING",
                    (embedding_uuid, tracklet_id, model_version, embedding.tobytes(), len(embedding)),
                )
            inserted += max(0, cursor.rowcount)
    return inserted


def import_timestamp_overrides(database_url: str, manifest_path: str | Path) -> dict[str, int]:
    overrides = pd.read_csv(manifest_path, dtype={"video_id": str})
    required = {"video_id", "recording_start", "recording_end"}
    if not required.issubset(overrides.columns):
        raise ValueError(f"Override manifest needs columns: {sorted(required)}")
    updated = 0
    with connect(database_url) as connection:
        for row in overrides.to_dict("records"):
            cursor = connection.execute(
                """UPDATE videos SET recording_start=?, recording_end=?, timestamp_source='manual',
                timestamp_confidence=1.0 WHERE video_uuid=?""",
                (row["recording_start"], row["recording_end"], row["video_id"]),
            )
            updated += max(0, cursor.rowcount)
        videos = pd.DataFrame([dict(row) for row in connection.execute(
            "SELECT video_uuid AS video_id,camera_id,recording_start,recording_end,sha256 FROM videos"
        )])
        videos["duplicate_of"] = ""
        grouped = assign_overlap_groups(videos)
        for row in grouped.to_dict("records"):
            connection.execute(
                "UPDATE videos SET overlap_group_id=?, canonical_video_uuid=? WHERE video_uuid=?",
                (row["overlap_group_id"], row["canonical_video_id"], row["video_id"]),
            )
    return {"updated": updated, "overlap_groups": int(grouped["overlap_group_id"].nunique())}


def import_existing_run(database_url: str, run_dir: str | Path, checkpoint: str | Path) -> dict[str, int | str]:
    run = Path(run_dir).resolve()
    tracks_path = run / "tracklets.csv"
    embeddings_path = run / "track_embeddings.npz"
    if not tracks_path.exists() or not embeddings_path.exists():
        raise ValueError("Run needs tracklets.csv and track_embeddings.npz")
    tracks = pd.read_csv(tracks_path)
    model_version = register_model(database_url, checkpoint)
    inserted_tracks = sum(register_tracklets(database_url, video_id, tracks_path) for video_id in tracks["video_id"].astype(str).unique())
    inserted_embeddings = register_embeddings(database_url, embeddings_path, model_version)
    identity_counts = import_existing_identities(database_url, run)
    return {"tracklets": inserted_tracks, "embeddings": inserted_embeddings, "model_version": model_version, **identity_counts}


def import_existing_identities(database_url: str, run_dir: str | Path) -> dict[str, int]:
    run = Path(run_dir).resolve()
    labels_path = run / "labels.csv"
    if not labels_path.exists():
        return {"cows": 0, "identity_assignments": 0, "identity_edges": 0}
    labels = pd.read_csv(labels_path, dtype={"cow_id": str})
    if "confirmed" in labels:
        labels = labels.loc[bool_mask(labels["confirmed"], default=False)]
    labels = labels.loc[labels["cow_id"].fillna("").astype(str).str.strip().ne("")].copy()
    conflict_ids: set[str] = set()
    audit_path = run / "label_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path, dtype={"cow_id": str})
        conflict_ids = set(audit.loc[audit.get("status", "").eq("conflict"), "cow_id"].astype(str)) if "status" in audit else set()
    created = assignments = edges = 0
    with connect(database_url) as connection:
        dates = {row["video_uuid"]: str(row["recording_start"] or "")[:10] for row in connection.execute("SELECT video_uuid,recording_start FROM videos")}
        for cow_id, group in labels.groupby("cow_id"):
            cow_id = str(cow_id)
            group_dates = {dates.get(str(video_id), "") for video_id in group.get("video_id", pd.Series(dtype=str))} - {""}
            state = "ambiguous" if cow_id in conflict_ids else ("stable_visual" if len(group_dates) >= 3 else ("visual_confirmed" if len(group_dates) >= 2 else "provisional"))
            candidate_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"cow-reid:{cow_id}"))
            cursor = connection.execute("INSERT INTO cows(cow_uuid,display_id,identity_state) VALUES (?,?,?) ON CONFLICT(display_id) DO NOTHING", (candidate_uuid, cow_id, state))
            created += max(0, cursor.rowcount)
            resolved = connection.execute("SELECT cow_uuid FROM cows WHERE display_id=?", (cow_id,)).fetchone()["cow_uuid"]
            for tracklet_id in group["tracklet_id"].astype(str):
                cursor = connection.execute("UPDATE tracklets SET cow_uuid=? WHERE tracklet_uuid=?", (resolved, tracklet_id))
                assignments += max(0, cursor.rowcount)
        reviews_path = run / "pair_reviews.csv"
        if reviews_path.exists():
            reviews = pd.read_csv(reviews_path)
            for row in reviews.to_dict("records"):
                left, right = sorted((str(row["left_tracklet_id"]), str(row["right_tracklet_id"])))
                edge_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"cow-reid-edge:{left}:{right}:{row.get('decision')}"))
                cursor = connection.execute(
                    "INSERT INTO identity_edges(edge_uuid,left_entity,right_entity,decision,reviewer,similarity) VALUES (?,?,?,?,?,?) ON CONFLICT(edge_uuid) DO NOTHING",
                    (edge_uuid, left, right, str(row.get("decision")), "legacy_import", _none(row.get("similarity"))),
                )
                edges += max(0, cursor.rowcount)
    return {"cows": created, "identity_assignments": assignments, "identity_edges": edges}
