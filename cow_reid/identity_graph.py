from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Iterable

from .db import connect, init_database


IDENTITY_STATES = {"provisional", "visual_confirmed", "stable_visual", "external_linked", "ambiguous", "retired"}


@dataclass(frozen=True)
class IdentityConflict:
    edge_uuid: str
    left_entity: str
    right_entity: str
    decision: str


def create_cow(database_url: str, display_id: str, identity_state: str = "provisional", external_id: str | None = None) -> str:
    if identity_state not in IDENTITY_STATES:
        raise ValueError(f"Invalid identity state: {identity_state}")
    init_database(database_url)
    cow_uuid = str(uuid.uuid4())
    with connect(database_url) as connection:
        connection.execute("INSERT INTO cows(cow_uuid,display_id,identity_state,external_id) VALUES (?,?,?,?)", (cow_uuid, display_id, identity_state, external_id))
    return cow_uuid


def find_merge_conflicts(database_url: str, cow_uuids: Iterable[str]) -> list[IdentityConflict]:
    ids = set(map(str, cow_uuids))
    if not ids:
        return []
    with connect(database_url) as connection:
        tracklets = {row["tracklet_uuid"] for row in connection.execute(f"SELECT tracklet_uuid FROM tracklets WHERE cow_uuid IN ({','.join('?' for _ in ids)})", tuple(ids))}
        rows = connection.execute("SELECT * FROM identity_edges WHERE decision='different' AND superseded_by IS NULL").fetchall()
    return [IdentityConflict(row["edge_uuid"], row["left_entity"], row["right_entity"], row["decision"]) for row in rows if row["left_entity"] in tracklets and row["right_entity"] in tracklets]


def merge_cows(database_url: str, source_cow_uuids: Iterable[str], target_cow_uuid: str, reviewer: str = "manual", resolve_conflicts: bool = False) -> str:
    sources = set(map(str, source_cow_uuids)); sources.add(str(target_cow_uuid))
    conflicts = find_merge_conflicts(database_url, sources)
    if conflicts and not resolve_conflicts:
        pairs = ", ".join(f"{item.left_entity}↔{item.right_entity}" for item in conflicts)
        raise ValueError(f"Merge conflicts with active different edges: {pairs}")
    audit_uuid = str(uuid.uuid4())
    with connect(database_url) as connection:
        cows = [dict(row) for row in connection.execute(f"SELECT * FROM cows WHERE cow_uuid IN ({','.join('?' for _ in sources)})", tuple(sources))]
        if len(cows) != len(sources):
            raise ValueError("One or more cow UUIDs do not exist")
        moved = [dict(row) for row in connection.execute(f"SELECT tracklet_uuid,cow_uuid FROM tracklets WHERE cow_uuid IN ({','.join('?' for _ in sources)})", tuple(sources))]
        aliases = []
        for cow in cows:
            if cow["cow_uuid"] == target_cow_uuid:
                continue
            aliases.append(cow["display_id"])
            connection.execute("INSERT INTO cow_aliases(alias,cow_uuid,reason) VALUES (?,?,?) ON CONFLICT(alias) DO UPDATE SET cow_uuid=excluded.cow_uuid, valid_from=CURRENT_TIMESTAMP, valid_to=NULL, reason=excluded.reason", (cow["display_id"], target_cow_uuid, f"merge:{audit_uuid}"))
            connection.execute("UPDATE tracklets SET cow_uuid=? WHERE cow_uuid=?", (target_cow_uuid, cow["cow_uuid"]))
            connection.execute("UPDATE cows SET identity_state='retired', retired_at=CURRENT_TIMESTAMP WHERE cow_uuid=?", (cow["cow_uuid"],))
        if conflicts and resolve_conflicts:
            for conflict in conflicts:
                connection.execute("UPDATE identity_edges SET superseded_by=? WHERE edge_uuid=?", (audit_uuid, conflict.edge_uuid))
        payload = {"sources": sorted(sources), "target": target_cow_uuid, "aliases": aliases, "reviewer": reviewer}
        inverse = {"tracklets": moved, "retired": [cow for cow in cows if cow["cow_uuid"] != target_cow_uuid], "conflicts": [item.edge_uuid for item in conflicts]}
        connection.execute("INSERT INTO identity_audit(audit_uuid,operation,payload_json,inverse_json) VALUES (?,?,?,?)", (audit_uuid, "merge", json.dumps(payload), json.dumps(inverse)))
    return audit_uuid


def split_cow(database_url: str, source_cow_uuid: str, tracklet_uuids: Iterable[str], new_display_id: str, reviewer: str = "manual") -> tuple[str, str]:
    selected = sorted(set(map(str, tracklet_uuids)))
    if not selected:
        raise ValueError("A split needs at least one tracklet")
    new_uuid = create_cow(database_url, new_display_id, "provisional")
    audit_uuid = str(uuid.uuid4())
    with connect(database_url) as connection:
        placeholders = ",".join("?" for _ in selected)
        cursor = connection.execute(f"UPDATE tracklets SET cow_uuid=? WHERE cow_uuid=? AND tracklet_uuid IN ({placeholders})", (new_uuid, source_cow_uuid, *selected))
        if cursor.rowcount != len(selected):
            raise ValueError("Some split tracklets do not belong to the source cow")
        payload = {"source": source_cow_uuid, "new": new_uuid, "tracklets": selected, "reviewer": reviewer}
        inverse = {"tracklets": selected, "restore_to": source_cow_uuid, "delete_cow": new_uuid}
        connection.execute("INSERT INTO identity_audit(audit_uuid,operation,payload_json,inverse_json) VALUES (?,?,?,?)", (audit_uuid, "split", json.dumps(payload), json.dumps(inverse)))
    return new_uuid, audit_uuid


def undo_identity_operation(database_url: str, audit_uuid: str) -> None:
    """Apply the recorded inverse without erasing the original audit event."""
    with connect(database_url) as connection:
        audit = connection.execute("SELECT * FROM identity_audit WHERE audit_uuid=?", (audit_uuid,)).fetchone()
        if audit is None:
            raise ValueError(f"Unknown audit operation: {audit_uuid}")
        if audit["undone_at"]:
            raise ValueError("Identity operation was already undone")
        inverse = json.loads(audit["inverse_json"])
        if audit["operation"] == "merge":
            for item in inverse["tracklets"]:
                connection.execute("UPDATE tracklets SET cow_uuid=? WHERE tracklet_uuid=?", (item["cow_uuid"], item["tracklet_uuid"]))
            for cow in inverse["retired"]:
                connection.execute("UPDATE cows SET identity_state=?, retired_at=? WHERE cow_uuid=?", (cow["identity_state"], cow["retired_at"], cow["cow_uuid"]))
                connection.execute("UPDATE cow_aliases SET valid_to=CURRENT_TIMESTAMP WHERE cow_uuid=? AND reason=?", (json.loads(audit["payload_json"])["target"], f"merge:{audit_uuid}"))
            for edge_uuid in inverse.get("conflicts", []):
                connection.execute("UPDATE identity_edges SET superseded_by=NULL WHERE edge_uuid=? AND superseded_by=?", (edge_uuid, audit_uuid))
        elif audit["operation"] == "split":
            for tracklet_uuid in inverse["tracklets"]:
                connection.execute("UPDATE tracklets SET cow_uuid=? WHERE tracklet_uuid=?", (inverse["restore_to"], tracklet_uuid))
            connection.execute("UPDATE cows SET identity_state='retired', retired_at=CURRENT_TIMESTAMP WHERE cow_uuid=?", (inverse["delete_cow"],))
        else:
            raise ValueError(f"Unsupported identity operation: {audit['operation']}")
        connection.execute("UPDATE identity_audit SET undone_at=CURRENT_TIMESTAMP WHERE audit_uuid=?", (audit_uuid,))
