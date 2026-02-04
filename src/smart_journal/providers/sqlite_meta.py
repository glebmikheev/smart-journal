from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from smart_journal.contracts import BlobRef


class SQLiteMetaStore:
    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        options = options or {}
        raw_db_path = options.get("path", ":memory:")
        db_path_value = str(raw_db_path)
        self._db_path = Path(db_path_value) if db_path_value != ":memory:" else None
        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(db_path_value)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._initialize_schema()

    def close(self) -> None:
        self._connection.close()

    def __del__(self) -> None:
        try:
            self.close()
        except sqlite3.Error:
            pass

    def provider_id(self) -> str:
        return "sqlite"

    def version(self) -> str:
        return "0.2.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "transactions": True,
            "fulltext_backend": "none",
            "durable": True,
            "schema_version": 1,
        }

    def begin_transaction(self) -> None:
        if not self._connection.in_transaction:
            self._connection.execute("BEGIN")

    def create_graph(self, title: str) -> str:
        graph_id = str(uuid4())
        timestamp = _utc_now()
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO graphs(graph_id, title, created_at, updated_at, deleted_at)
                VALUES (?, ?, ?, ?, NULL)
                """,
                (graph_id, title, timestamp, timestamp),
            )
        return graph_id

    def get_graph(
        self, graph_id: str, *, include_deleted: bool = False
    ) -> Mapping[str, Any] | None:
        query = "SELECT * FROM graphs WHERE graph_id = ?"
        params: tuple[Any, ...] = (graph_id,)
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        row = self._connection.execute(query, params).fetchone()
        return _row_to_dict(row) if row is not None else None

    def create_node(self, graph_id: str, title: str, body: str = "") -> str:
        self._require_live_graph(graph_id)
        node_id = str(uuid4())
        timestamp = _utc_now()
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO nodes(
                    node_id,
                    graph_id,
                    title,
                    body,
                    created_at,
                    updated_at,
                    deleted_at,
                    current_revision_id
                )
                VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (node_id, graph_id, title, body, timestamp, timestamp),
            )
            revision_id = self._insert_revision(
                node_id=node_id,
                revision_no=1,
                title=title,
                body=body,
                comment="create",
            )
            self._connection.execute(
                "UPDATE nodes SET current_revision_id = ? WHERE node_id = ?",
                (revision_id, node_id),
            )
        return node_id

    def update_node(
        self, node_id: str, *, title: str | None = None, body: str | None = None
    ) -> None:
        current = self.get_node(node_id)
        if current is None:
            raise KeyError(f"Node not found or deleted: {node_id}")
        next_title = str(title if title is not None else current["title"])
        next_body = str(body if body is not None else current["body"])
        timestamp = _utc_now()
        next_revision_no = int(self._next_revision_no(node_id))
        with self._connection:
            self._connection.execute(
                """
                UPDATE nodes
                SET title = ?, body = ?, updated_at = ?
                WHERE node_id = ?
                """,
                (next_title, next_body, timestamp, node_id),
            )
            revision_id = self._insert_revision(
                node_id=node_id,
                revision_no=next_revision_no,
                title=next_title,
                body=next_body,
                comment="update",
            )
            self._connection.execute(
                "UPDATE nodes SET current_revision_id = ? WHERE node_id = ?",
                (revision_id, node_id),
            )

    def delete_node(self, node_id: str, *, soft_delete: bool = True) -> None:
        if soft_delete:
            timestamp = _utc_now()
            with self._connection:
                self._connection.execute(
                    """
                    UPDATE nodes
                    SET deleted_at = ?, updated_at = ?
                    WHERE node_id = ? AND deleted_at IS NULL
                    """,
                    (timestamp, timestamp, node_id),
                )
            return

        with self._connection:
            self._connection.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,))

    def list_nodes(
        self, graph_id: str, *, include_deleted: bool = False
    ) -> list[Mapping[str, Any]]:
        query = "SELECT * FROM nodes WHERE graph_id = ?"
        params: tuple[Any, ...] = (graph_id,)
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        query += " ORDER BY created_at ASC"
        rows = self._connection.execute(query, params).fetchall()
        payload: list[Mapping[str, Any]] = [_row_to_dict(row) for row in rows]
        return payload

    def get_node(self, node_id: str, *, include_deleted: bool = False) -> Mapping[str, Any] | None:
        query = "SELECT * FROM nodes WHERE node_id = ?"
        params: tuple[Any, ...] = (node_id,)
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        row = self._connection.execute(query, params).fetchone()
        return _row_to_dict(row) if row is not None else None

    def list_revisions(self, node_id: str) -> list[Mapping[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT revision_id, node_id, revision_no, title, body, created_at, comment
            FROM revisions
            WHERE node_id = ?
            ORDER BY revision_no ASC
            """,
            (node_id,),
        ).fetchall()
        payload: list[Mapping[str, Any]] = [_row_to_dict(row) for row in rows]
        return payload

    def attach_content_item(
        self,
        node_id: str,
        blob_ref: BlobRef,
        *,
        mime_type: str | None = None,
        filename: str | None = None,
    ) -> str:
        self._require_live_node(node_id)
        content_item_id = str(uuid4())
        timestamp = _utc_now()
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO content_items(
                    content_item_id,
                    node_id,
                    blob_scheme,
                    blob_key,
                    blob_hash,
                    blob_size,
                    blob_version,
                    mime_type,
                    filename,
                    created_at,
                    deleted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    content_item_id,
                    node_id,
                    blob_ref.scheme,
                    blob_ref.key,
                    blob_ref.hash,
                    blob_ref.size,
                    blob_ref.version,
                    mime_type,
                    filename,
                    timestamp,
                ),
            )
        return content_item_id

    def list_content_items(
        self, node_id: str, *, include_deleted: bool = False
    ) -> list[Mapping[str, Any]]:
        query = "SELECT * FROM content_items WHERE node_id = ?"
        params: tuple[Any, ...] = (node_id,)
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        query += " ORDER BY created_at ASC"
        rows = self._connection.execute(query, params).fetchall()
        payload: list[Mapping[str, Any]] = [_row_to_dict(row) for row in rows]
        return payload

    def detach_content_item(self, content_item_id: str, *, soft_delete: bool = True) -> None:
        if soft_delete:
            with self._connection:
                self._connection.execute(
                    """
                    UPDATE content_items
                    SET deleted_at = ?
                    WHERE content_item_id = ? AND deleted_at IS NULL
                    """,
                    (_utc_now(), content_item_id),
                )
            return

        with self._connection:
            self._connection.execute(
                "DELETE FROM content_items WHERE content_item_id = ?",
                (content_item_id,),
            )

    def _initialize_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_info(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS graphs(
                graph_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            );

            CREATE TABLE IF NOT EXISTS nodes(
                node_id TEXT PRIMARY KEY,
                graph_id TEXT NOT NULL REFERENCES graphs(graph_id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                body TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT,
                current_revision_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_graph_id ON nodes(graph_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_deleted_at ON nodes(deleted_at);

            CREATE TABLE IF NOT EXISTS revisions(
                revision_id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                revision_no INTEGER NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                created_at TEXT NOT NULL,
                comment TEXT,
                UNIQUE(node_id, revision_no)
            );
            CREATE INDEX IF NOT EXISTS idx_revisions_node_id ON revisions(node_id);

            CREATE TABLE IF NOT EXISTS content_items(
                content_item_id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                blob_scheme TEXT NOT NULL,
                blob_key TEXT NOT NULL,
                blob_hash TEXT NOT NULL,
                blob_size INTEGER NOT NULL,
                blob_version TEXT,
                mime_type TEXT,
                filename TEXT,
                created_at TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_content_items_node_id ON content_items(node_id);
            CREATE INDEX IF NOT EXISTS idx_content_items_deleted_at ON content_items(deleted_at);

            CREATE TABLE IF NOT EXISTS tags(
                tag_id TEXT PRIMARY KEY,
                graph_id TEXT NOT NULL REFERENCES graphs(graph_id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                deleted_at TEXT,
                UNIQUE(graph_id, name)
            );

            CREATE TABLE IF NOT EXISTS node_tags(
                node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                tag_id TEXT NOT NULL REFERENCES tags(tag_id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                PRIMARY KEY(node_id, tag_id)
            );

            CREATE TABLE IF NOT EXISTS "groups"(
                group_id TEXT PRIMARY KEY,
                graph_id TEXT NOT NULL REFERENCES graphs(graph_id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                deleted_at TEXT,
                UNIQUE(graph_id, name)
            );

            CREATE TABLE IF NOT EXISTS node_groups(
                node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                group_id TEXT NOT NULL REFERENCES "groups"(group_id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                PRIMARY KEY(node_id, group_id)
            );

            CREATE TABLE IF NOT EXISTS edges(
                edge_id TEXT PRIMARY KEY,
                graph_id TEXT NOT NULL REFERENCES graphs(graph_id) ON DELETE CASCADE,
                from_node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                to_node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                edge_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                weight REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_edges_graph_id ON edges(graph_id);
            CREATE INDEX IF NOT EXISTS idx_edges_from_node_id ON edges(from_node_id);
            CREATE INDEX IF NOT EXISTS idx_edges_to_node_id ON edges(to_node_id);
            CREATE INDEX IF NOT EXISTS idx_edges_status ON edges(status);
            """
        )
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO schema_info(key, value)
                VALUES ('schema_version', '1')
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """
            )

    def _insert_revision(
        self,
        *,
        node_id: str,
        revision_no: int,
        title: str,
        body: str,
        comment: str,
    ) -> str:
        revision_id = str(uuid4())
        self._connection.execute(
            """
            INSERT INTO revisions(
                revision_id,
                node_id,
                revision_no,
                title,
                body,
                created_at,
                comment
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (revision_id, node_id, revision_no, title, body, _utc_now(), comment),
        )
        return revision_id

    def _next_revision_no(self, node_id: str) -> int:
        row = self._connection.execute(
            (
                "SELECT COALESCE(MAX(revision_no), 0) AS max_revision_no "
                "FROM revisions WHERE node_id = ?"
            ),
            (node_id,),
        ).fetchone()
        if row is None:
            return 1
        return int(row["max_revision_no"]) + 1

    def _require_live_graph(self, graph_id: str) -> None:
        graph = self.get_graph(graph_id)
        if graph is None:
            raise KeyError(f"Graph not found or deleted: {graph_id}")

    def _require_live_node(self, node_id: str) -> None:
        node = self.get_node(node_id)
        if node is None:
            raise KeyError(f"Node not found or deleted: {node_id}")


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {str(key): row[key] for key in row.keys()}


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")
