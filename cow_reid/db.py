from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import Any, Iterator


SQLITE_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS model_registry (
  model_version TEXT PRIMARY KEY, model_name TEXT NOT NULL, checkpoint_sha256 TEXT,
  code_commit TEXT, config_hash TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS cows (
  cow_uuid TEXT PRIMARY KEY, display_id TEXT NOT NULL UNIQUE,
  identity_state TEXT NOT NULL DEFAULT 'provisional', external_id TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, retired_at TEXT
);
CREATE TABLE IF NOT EXISTS cow_aliases (
  alias TEXT PRIMARY KEY, cow_uuid TEXT NOT NULL REFERENCES cows(cow_uuid),
  valid_from TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, valid_to TEXT, reason TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
  session_uuid TEXT PRIMARY KEY, camera_id TEXT NOT NULL, recording_date TEXT NOT NULL,
  session_type TEXT, domain TEXT, start_at TEXT, end_at TEXT,
  UNIQUE(camera_id, recording_date, session_type, domain)
);
CREATE TABLE IF NOT EXISTS videos (
  video_uuid TEXT PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE, perceptual_fingerprint TEXT,
  camera_id TEXT NOT NULL, recording_start TEXT, recording_end TEXT,
  timestamp_source TEXT NOT NULL DEFAULT 'unknown', timestamp_confidence REAL NOT NULL DEFAULT 0,
  ocr_samples_json TEXT, canonical_video_uuid TEXT, overlap_group_id TEXT, media_uri TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'INGESTED', attempt_count INTEGER NOT NULL DEFAULT 0,
  last_error TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS tracklets (
  tracklet_uuid TEXT PRIMARY KEY, video_uuid TEXT NOT NULL REFERENCES videos(video_uuid),
  session_uuid TEXT REFERENCES sessions(session_uuid), cow_uuid TEXT REFERENCES cows(cow_uuid),
  start_s REAL NOT NULL, end_s REAL NOT NULL, direction TEXT, flank TEXT,
  quality_score REAL, review_state TEXT, metadata_json TEXT
);
CREATE TABLE IF NOT EXISTS reid_embeddings (
  embedding_uuid TEXT PRIMARY KEY, tracklet_uuid TEXT NOT NULL REFERENCES tracklets(tracklet_uuid),
  model_version TEXT NOT NULL REFERENCES model_registry(model_version), embedding BLOB NOT NULL,
  dimensions INTEGER NOT NULL, quality_score REAL, UNIQUE(tracklet_uuid, model_version)
);
CREATE TABLE IF NOT EXISTS identity_edges (
  edge_uuid TEXT PRIMARY KEY, left_entity TEXT NOT NULL, right_entity TEXT NOT NULL,
  decision TEXT NOT NULL, reviewer TEXT, similarity REAL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  superseded_by TEXT
);
CREATE TABLE IF NOT EXISTS identity_audit (
  audit_uuid TEXT PRIMARY KEY, operation TEXT NOT NULL, payload_json TEXT NOT NULL,
  inverse_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, undone_at TEXT
);
CREATE TABLE IF NOT EXISTS keypoint_runs (
  run_uuid TEXT PRIMARY KEY, tracklet_uuid TEXT NOT NULL REFERENCES tracklets(tracklet_uuid),
  model_version TEXT NOT NULL, status TEXT NOT NULL, raw_output_uri TEXT,
  quality_summary_json TEXT, attempt_count INTEGER NOT NULL DEFAULT 0, last_error TEXT,
  UNIQUE(tracklet_uuid, model_version)
);
CREATE TABLE IF NOT EXISTS posture_observations (
  observation_uuid TEXT PRIMARY KEY, tracklet_uuid TEXT NOT NULL REFERENCES tracklets(tracklet_uuid),
  feature_name TEXT NOT NULL, value REAL NOT NULL, uncertainty REAL,
  valid_frame_ratio REAL, model_version TEXT NOT NULL, observed_at TEXT NOT NULL,
  UNIQUE(tracklet_uuid, feature_name, model_version)
);
CREATE TABLE IF NOT EXISTS health_summaries (
  cow_uuid TEXT NOT NULL REFERENCES cows(cow_uuid), period_start TEXT NOT NULL, period_end TEXT NOT NULL,
  feature_name TEXT NOT NULL, median_value REAL, baseline REAL, delta REAL, robust_z REAL,
  status TEXT NOT NULL, algorithm_version TEXT NOT NULL,
  PRIMARY KEY(cow_uuid, period_end, feature_name, algorithm_version)
);
"""


def sqlite_path(database_url: str) -> Path:
    if database_url == "sqlite:///:memory:":
        return Path(":memory:")
    prefix = "sqlite:///"
    if database_url.startswith(prefix):
        return Path(database_url[len(prefix):]).expanduser().resolve()
    if "://" not in database_url:
        return Path(database_url).expanduser().resolve()
    raise ValueError("Not a SQLite database URL")


class _ConnectionProxy:
    def __init__(self, connection: Any, postgres: bool = False):
        self.connection = connection
        self.postgres = postgres

    def execute(self, sql: str, parameters: tuple[object, ...] = ()):
        return self.connection.execute(sql.replace("?", "%s") if self.postgres else sql, parameters)

    def executescript(self, sql: str) -> None:
        if self.postgres:
            self.connection.execute(sql)
        else:
            self.connection.executescript(sql)


@contextlib.contextmanager
def connect(database_url: str) -> Iterator[_ConnectionProxy]:
    postgres = database_url.startswith(("postgresql://", "postgres://"))
    if postgres:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("PostgreSQL URLs require: pip install -e '.[db]'") from exc
        connection = psycopg.connect(database_url, row_factory=dict_row)
    else:
        path = sqlite_path(database_url)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
    proxy = _ConnectionProxy(connection, postgres)
    try:
        yield proxy
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_database(database_url: str) -> None:
    with connect(database_url) as connection:
        if database_url.startswith(("postgresql://", "postgres://")):
            migration = Path(__file__).resolve().parent.parent / "migrations" / "postgresql.sql"
            connection.executescript(migration.read_text(encoding="utf-8"))
        else:
            connection.executescript(SQLITE_SCHEMA)
