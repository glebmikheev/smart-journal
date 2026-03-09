from __future__ import annotations

import json
import sqlite3
import struct
from collections.abc import Mapping, Sequence
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
        # FastAPI runs sync endpoints in a threadpool, so sqlite calls may come
        # from threads other than the one that created the connection.
        self._connection = sqlite3.connect(db_path_value, check_same_thread=False)
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
        return "0.6.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "transactions": True,
            "fulltext_backend": "fts5",
            "durable": True,
            "schema_version": 5,
            "supports_scope_filters": True,
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

    def list_graphs(self, *, include_deleted: bool = False) -> list[Mapping[str, Any]]:
        query = "SELECT * FROM graphs"
        if not include_deleted:
            query += " WHERE deleted_at IS NULL"
        query += " ORDER BY created_at ASC"
        rows = self._connection.execute(query).fetchall()
        return [_row_to_dict(row) for row in rows]

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
            self._refresh_node_search_index(node_id)
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
            self._refresh_node_search_index(node_id)

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
                self._refresh_node_search_index(node_id)
            return

        with self._connection:
            self._connection.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,))
            self._refresh_node_search_index(node_id)

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
                    extraction_status,
                    extracted_text,
                    extracted_metadata_json,
                    extraction_error,
                    created_at,
                    deleted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', '', '{}', NULL, ?, NULL)
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
            self._refresh_node_search_index(node_id)
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
        payload: list[Mapping[str, Any]] = [_content_item_row_to_dict(row) for row in rows]
        return payload

    def get_content_item(
        self, content_item_id: str, *, include_deleted: bool = False
    ) -> Mapping[str, Any] | None:
        query = "SELECT * FROM content_items WHERE content_item_id = ?"
        params: tuple[Any, ...] = (content_item_id,)
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        row = self._connection.execute(query, params).fetchone()
        return _content_item_row_to_dict(row) if row is not None else None

    def detach_content_item(self, content_item_id: str, *, soft_delete: bool = True) -> None:
        row = self._connection.execute(
            "SELECT node_id FROM content_items WHERE content_item_id = ?",
            (content_item_id,),
        ).fetchone()
        if row is None:
            return
        node_id = str(row["node_id"])

        if soft_delete:
            timestamp = _utc_now()
            with self._connection:
                self._connection.execute(
                    """
                    UPDATE content_items
                    SET deleted_at = ?
                    WHERE content_item_id = ? AND deleted_at IS NULL
                    """,
                    (timestamp, content_item_id),
                )
                self._connection.execute(
                    """
                    UPDATE chunks
                    SET deleted_at = ?
                    WHERE content_item_id = ? AND deleted_at IS NULL
                    """,
                    (timestamp, content_item_id),
                )
                self._refresh_node_search_index(node_id)
            return

        with self._connection:
            self._connection.execute(
                "DELETE FROM chunks WHERE content_item_id = ?",
                (content_item_id,),
            )
            self._connection.execute(
                "DELETE FROM content_items WHERE content_item_id = ?",
                (content_item_id,),
            )
            self._refresh_node_search_index(node_id)

    def set_content_item_extraction(
        self,
        content_item_id: str,
        *,
        status: str,
        extracted_text: str | None = None,
        metadata: Mapping[str, str] | None = None,
        error: str | None = None,
    ) -> None:
        row = self._connection.execute(
            """
            SELECT node_id, extracted_text, extracted_metadata_json
            FROM content_items
            WHERE content_item_id = ? AND deleted_at IS NULL
            """,
            (content_item_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Content item not found or deleted: {content_item_id}")

        next_text = str(row["extracted_text"]) if extracted_text is None else extracted_text
        next_metadata_json = (
            str(row["extracted_metadata_json"])
            if metadata is None
            else json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True)
        )
        with self._connection:
            self._connection.execute(
                """
                UPDATE content_items
                SET extraction_status = ?,
                    extracted_text = ?,
                    extracted_metadata_json = ?,
                    extraction_error = ?
                WHERE content_item_id = ?
                """,
                (status, next_text, next_metadata_json, error, content_item_id),
            )
            self._refresh_node_search_index(str(row["node_id"]))

    def replace_content_item_chunks(
        self,
        content_item_id: str,
        chunks: Sequence[Mapping[str, str | int]],
    ) -> list[str]:
        row = self._connection.execute(
            """
            SELECT node_id
            FROM content_items
            WHERE content_item_id = ? AND deleted_at IS NULL
            """,
            (content_item_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Content item not found or deleted: {content_item_id}")
        node_id = str(row["node_id"])
        created_ids: list[str] = []

        with self._connection:
            self._connection.execute(
                "DELETE FROM chunks WHERE content_item_id = ?",
                (content_item_id,),
            )
            for chunk in chunks:
                chunk_id = str(uuid4())
                created_ids.append(chunk_id)
                self._connection.execute(
                    """
                    INSERT INTO chunks(
                        chunk_id,
                        content_item_id,
                        node_id,
                        chunk_index,
                        text,
                        checksum,
                        created_at,
                        deleted_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        chunk_id,
                        content_item_id,
                        node_id,
                        int(chunk["chunk_index"]),
                        str(chunk["text"]),
                        str(chunk["checksum"]),
                        _utc_now(),
                    ),
                )
        return created_ids

    def list_chunks(
        self, content_item_id: str, *, include_deleted: bool = False
    ) -> list[Mapping[str, Any]]:
        query = "SELECT * FROM chunks WHERE content_item_id = ?"
        params: tuple[Any, ...] = (content_item_id,)
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        query += " ORDER BY chunk_index ASC"
        rows = self._connection.execute(query, params).fetchall()
        payload: list[Mapping[str, Any]] = [_row_to_dict(row) for row in rows]
        return payload

    def upsert_chunk_embeddings(self, embeddings: Sequence[Mapping[str, Any]]) -> None:
        if not embeddings:
            return
        with self._connection:
            for embedding in embeddings:
                chunk_id = str(embedding["chunk_id"])
                model_id = str(embedding["model_id"])
                metric = str(embedding.get("metric", "cosine"))
                dim = int(embedding["dim"])

                chunk_row = self._connection.execute(
                    """
                    SELECT checksum
                    FROM chunks
                    WHERE chunk_id = ? AND deleted_at IS NULL
                    """,
                    (chunk_id,),
                ).fetchone()
                if chunk_row is None:
                    raise KeyError(f"Chunk not found or deleted: {chunk_id}")
                chunk_checksum = str(chunk_row["checksum"])

                checksum_raw = embedding.get("checksum")
                checksum = chunk_checksum if checksum_raw is None else str(checksum_raw)
                if checksum != chunk_checksum:
                    raise ValueError(
                        "Embedding checksum does not match chunk checksum "
                        f"for chunk_id={chunk_id}."
                    )

                vector_raw = embedding.get("vector")
                vector = _coerce_vector(vector_raw)
                if len(vector) != dim:
                    raise ValueError(
                        f"Vector dim mismatch for chunk_id={chunk_id}: "
                        f"expected {dim}, got {len(vector)}."
                    )
                vector_blob = _vector_to_blob(vector)

                self._connection.execute(
                    """
                    INSERT INTO embeddings(
                        chunk_id,
                        model_id,
                        dim,
                        metric,
                        vector_blob,
                        checksum,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id, model_id) DO UPDATE SET
                        dim = excluded.dim,
                        metric = excluded.metric,
                        vector_blob = excluded.vector_blob,
                        checksum = excluded.checksum,
                        created_at = excluded.created_at
                    """,
                    (
                        chunk_id,
                        model_id,
                        dim,
                        metric,
                        vector_blob,
                        checksum,
                        _utc_now(),
                    ),
                )

    def list_chunk_embeddings(
        self,
        content_item_id: str,
        *,
        model_id: str | None = None,
    ) -> list[Mapping[str, Any]]:
        query = """
            SELECT
                e.chunk_id,
                e.model_id,
                e.dim,
                e.metric,
                e.vector_blob,
                e.checksum,
                e.created_at,
                c.chunk_index
            FROM embeddings e
            INNER JOIN chunks c ON c.chunk_id = e.chunk_id
            WHERE c.content_item_id = ?
              AND c.deleted_at IS NULL
        """
        params: list[Any] = [content_item_id]
        if model_id is not None:
            query += "\nAND e.model_id = ?"
            params.append(model_id)
        query += "\nORDER BY c.chunk_index ASC, e.model_id ASC"

        rows = self._connection.execute(query, tuple(params)).fetchall()
        payload: list[Mapping[str, Any]] = []
        for row in rows:
            item = _row_to_dict(row)
            item["dim"] = int(item["dim"])
            item["vector"] = _blob_to_vector(bytes(row["vector_blob"]), int(item["dim"]))
            item.pop("vector_blob", None)
            item.pop("chunk_index", None)
            payload.append(item)
        return payload

    def get_chunk_embedding(
        self,
        chunk_id: str,
        model_id: str,
    ) -> Mapping[str, Any] | None:
        row = self._connection.execute(
            """
            SELECT chunk_id, model_id, dim, metric, vector_blob, checksum, created_at
            FROM embeddings
            WHERE chunk_id = ? AND model_id = ?
            """,
            (chunk_id, model_id),
        ).fetchone()
        if row is None:
            return None
        item = _row_to_dict(row)
        item["dim"] = int(item["dim"])
        item["vector"] = _blob_to_vector(bytes(row["vector_blob"]), int(item["dim"]))
        item.pop("vector_blob", None)
        return item

    def enqueue_vector_index_ops(
        self,
        ops: Sequence[Mapping[str, str]],
    ) -> list[str]:
        if not ops:
            return []
        created_ids: list[str] = []
        with self._connection:
            for op in ops:
                op_type = str(op["op_type"])
                if op_type not in {"upsert", "delete"}:
                    raise ValueError(f"Unsupported vector index op_type: {op_type}")
                op_id = str(uuid4())
                created_ids.append(op_id)
                self._connection.execute(
                    """
                    INSERT INTO vector_index_ops(
                        op_id,
                        op_type,
                        chunk_id,
                        model_id,
                        status,
                        created_at,
                        applied_at
                    )
                    VALUES (?, ?, ?, ?, 'pending', ?, NULL)
                    """,
                    (
                        op_id,
                        op_type,
                        str(op["chunk_id"]),
                        str(op["model_id"]),
                        _utc_now(),
                    ),
                )
        return created_ids

    def list_vector_index_ops(
        self,
        *,
        status: str = "pending",
        model_id: str | None = None,
        limit: int = 1000,
    ) -> list[Mapping[str, Any]]:
        if limit <= 0:
            return []
        query = """
            SELECT op_id, op_type, chunk_id, model_id, status, created_at, applied_at
            FROM vector_index_ops
            WHERE status = ?
        """
        params: list[Any] = [status]
        if model_id is not None:
            query += "\nAND model_id = ?"
            params.append(model_id)
        query += "\nORDER BY created_at ASC, op_id ASC LIMIT ?"
        params.append(limit)

        rows = self._connection.execute(query, tuple(params)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def mark_vector_index_ops_applied(self, op_ids: Sequence[str]) -> None:
        unique_ids = list(dict.fromkeys(str(op_id) for op_id in op_ids if str(op_id)))
        if not unique_ids:
            return

        placeholders = ", ".join("?" for _ in unique_ids)
        with self._connection:
            self._connection.execute(
                f"""
                UPDATE vector_index_ops
                SET status = 'applied',
                    applied_at = ?
                WHERE op_id IN ({placeholders})
                """,
                (_utc_now(), *unique_ids),
            )

    def create_tag(self, graph_id: str, name: str) -> str:
        self._require_live_graph(graph_id)
        tag_id = str(uuid4())
        with self._connection:
            try:
                self._connection.execute(
                    """
                    INSERT INTO tags(tag_id, graph_id, name, created_at, deleted_at)
                    VALUES (?, ?, ?, ?, NULL)
                    """,
                    (tag_id, graph_id, name, _utc_now()),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(f"Tag already exists in graph '{graph_id}': {name}") from error
        return tag_id

    def list_tags(self, graph_id: str, *, include_deleted: bool = False) -> list[Mapping[str, Any]]:
        query = "SELECT * FROM tags WHERE graph_id = ?"
        params: tuple[Any, ...] = (graph_id,)
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        query += " ORDER BY name ASC"
        rows = self._connection.execute(query, params).fetchall()
        payload: list[Mapping[str, Any]] = [_row_to_dict(row) for row in rows]
        return payload

    def delete_tag(self, tag_id: str, *, soft_delete: bool = True) -> None:
        if soft_delete:
            with self._connection:
                self._connection.execute(
                    "UPDATE tags SET deleted_at = ? WHERE tag_id = ? AND deleted_at IS NULL",
                    (_utc_now(), tag_id),
                )
            return

        with self._connection:
            self._connection.execute("DELETE FROM tags WHERE tag_id = ?", (tag_id,))

    def add_node_tag(self, node_id: str, tag_id: str) -> None:
        node = self.get_node(node_id)
        if node is None:
            raise KeyError(f"Node not found or deleted: {node_id}")
        tag = self._get_live_tag(tag_id)
        if str(node["graph_id"]) != str(tag["graph_id"]):
            raise ValueError("Node and tag must belong to the same graph.")
        with self._connection:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO node_tags(node_id, tag_id, created_at)
                VALUES (?, ?, ?)
                """,
                (node_id, tag_id, _utc_now()),
            )

    def remove_node_tag(self, node_id: str, tag_id: str) -> None:
        with self._connection:
            self._connection.execute(
                "DELETE FROM node_tags WHERE node_id = ? AND tag_id = ?",
                (node_id, tag_id),
            )

    def list_node_tags(self, node_id: str) -> list[Mapping[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT t.*
            FROM tags t
            INNER JOIN node_tags nt ON nt.tag_id = t.tag_id
            WHERE nt.node_id = ? AND t.deleted_at IS NULL
            ORDER BY t.name ASC
            """,
            (node_id,),
        ).fetchall()
        payload: list[Mapping[str, Any]] = [_row_to_dict(row) for row in rows]
        return payload

    def create_group(self, graph_id: str, name: str) -> str:
        self._require_live_graph(graph_id)
        group_id = str(uuid4())
        with self._connection:
            try:
                self._connection.execute(
                    """
                    INSERT INTO "groups"(group_id, graph_id, name, created_at, deleted_at)
                    VALUES (?, ?, ?, ?, NULL)
                    """,
                    (group_id, graph_id, name, _utc_now()),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(f"Group already exists in graph '{graph_id}': {name}") from error
        return group_id

    def list_groups(
        self, graph_id: str, *, include_deleted: bool = False
    ) -> list[Mapping[str, Any]]:
        query = 'SELECT * FROM "groups" WHERE graph_id = ?'
        params: tuple[Any, ...] = (graph_id,)
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        query += " ORDER BY name ASC"
        rows = self._connection.execute(query, params).fetchall()
        payload: list[Mapping[str, Any]] = [_row_to_dict(row) for row in rows]
        return payload

    def delete_group(self, group_id: str, *, soft_delete: bool = True) -> None:
        if soft_delete:
            with self._connection:
                self._connection.execute(
                    'UPDATE "groups" SET deleted_at = ? WHERE group_id = ? AND deleted_at IS NULL',
                    (_utc_now(), group_id),
                )
            return

        with self._connection:
            self._connection.execute('DELETE FROM "groups" WHERE group_id = ?', (group_id,))

    def add_node_to_group(self, node_id: str, group_id: str) -> None:
        node = self.get_node(node_id)
        if node is None:
            raise KeyError(f"Node not found or deleted: {node_id}")
        group = self._get_live_group(group_id)
        if str(node["graph_id"]) != str(group["graph_id"]):
            raise ValueError("Node and group must belong to the same graph.")
        with self._connection:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO node_groups(node_id, group_id, created_at)
                VALUES (?, ?, ?)
                """,
                (node_id, group_id, _utc_now()),
            )

    def remove_node_from_group(self, node_id: str, group_id: str) -> None:
        with self._connection:
            self._connection.execute(
                "DELETE FROM node_groups WHERE node_id = ? AND group_id = ?",
                (node_id, group_id),
            )

    def list_node_groups(self, node_id: str) -> list[Mapping[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT g.*
            FROM "groups" g
            INNER JOIN node_groups ng ON ng.group_id = g.group_id
            WHERE ng.node_id = ? AND g.deleted_at IS NULL
            ORDER BY g.name ASC
            """,
            (node_id,),
        ).fetchall()
        payload: list[Mapping[str, Any]] = [_row_to_dict(row) for row in rows]
        return payload

    def search_fulltext(
        self,
        query: str,
        *,
        graph_id: str | None = None,
        group_id: str | None = None,
        tag_ids: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[Mapping[str, Any]]:
        fts_query = _build_fts_query(query)
        if not fts_query or limit <= 0:
            return []

        sql_parts: list[str] = [
            "SELECT n.*, bm25(node_search_fts) AS score",
            "FROM node_search_fts",
            "INNER JOIN nodes n ON n.node_id = node_search_fts.node_id",
            "WHERE node_search_fts MATCH ?",
            "AND n.deleted_at IS NULL",
        ]
        params: list[Any] = [fts_query]

        if graph_id is not None:
            sql_parts.append("AND n.graph_id = ?")
            params.append(graph_id)
        if group_id is not None:
            sql_parts.append(
                """
                AND EXISTS (
                    SELECT 1
                    FROM node_groups ng
                    INNER JOIN "groups" g ON g.group_id = ng.group_id
                    WHERE ng.node_id = n.node_id
                      AND ng.group_id = ?
                      AND g.deleted_at IS NULL
                )
                """
            )
            params.append(group_id)
        for tag_id in dict.fromkeys(tag_ids or []):
            sql_parts.append(
                """
                AND EXISTS (
                    SELECT 1
                    FROM node_tags nt
                    INNER JOIN tags t ON t.tag_id = nt.tag_id
                    WHERE nt.node_id = n.node_id
                      AND nt.tag_id = ?
                      AND t.deleted_at IS NULL
                )
                """
            )
            params.append(tag_id)

        sql_parts.append("ORDER BY score ASC, n.updated_at DESC LIMIT ?")
        params.append(limit)

        try:
            rows = self._connection.execute("\n".join(sql_parts), tuple(params)).fetchall()
        except sqlite3.OperationalError:
            return []

        payload: list[Mapping[str, Any]] = []
        for row in rows:
            item = _row_to_dict(row)
            raw_score = row["score"]
            item["score"] = float(raw_score) if raw_score is not None else 0.0
            payload.append(item)
        return payload

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
                extraction_status TEXT NOT NULL DEFAULT 'pending',
                extracted_text TEXT NOT NULL DEFAULT '',
                extracted_metadata_json TEXT NOT NULL DEFAULT '{}',
                extraction_error TEXT,
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
            CREATE INDEX IF NOT EXISTS idx_tags_graph_id ON tags(graph_id);

            CREATE TABLE IF NOT EXISTS node_tags(
                node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                tag_id TEXT NOT NULL REFERENCES tags(tag_id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                PRIMARY KEY(node_id, tag_id)
            );
            CREATE INDEX IF NOT EXISTS idx_node_tags_node_id ON node_tags(node_id);
            CREATE INDEX IF NOT EXISTS idx_node_tags_tag_id ON node_tags(tag_id);

            CREATE TABLE IF NOT EXISTS "groups"(
                group_id TEXT PRIMARY KEY,
                graph_id TEXT NOT NULL REFERENCES graphs(graph_id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                deleted_at TEXT,
                UNIQUE(graph_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_groups_graph_id ON "groups"(graph_id);

            CREATE TABLE IF NOT EXISTS node_groups(
                node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                group_id TEXT NOT NULL REFERENCES "groups"(group_id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                PRIMARY KEY(node_id, group_id)
            );
            CREATE INDEX IF NOT EXISTS idx_node_groups_node_id ON node_groups(node_id);
            CREATE INDEX IF NOT EXISTS idx_node_groups_group_id ON node_groups(group_id);

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

            CREATE TABLE IF NOT EXISTS chunks(
                chunk_id TEXT PRIMARY KEY,
                content_item_id TEXT NOT NULL
                    REFERENCES content_items(content_item_id) ON DELETE CASCADE,
                node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                checksum TEXT NOT NULL,
                created_at TEXT NOT NULL,
                deleted_at TEXT,
                UNIQUE(content_item_id, chunk_index)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_content_item_id ON chunks(content_item_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_node_id ON chunks(node_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_deleted_at ON chunks(deleted_at);

            CREATE TABLE IF NOT EXISTS embeddings(
                chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                model_id TEXT NOT NULL,
                dim INTEGER NOT NULL,
                metric TEXT NOT NULL,
                vector_blob BLOB NOT NULL,
                checksum TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(chunk_id, model_id)
            );
            CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);
            CREATE INDEX IF NOT EXISTS idx_embeddings_model_checksum
                ON embeddings(model_id, checksum);

            CREATE TABLE IF NOT EXISTS vector_index_ops(
                op_id TEXT PRIMARY KEY,
                op_type TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                applied_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_vector_index_ops_status
                ON vector_index_ops(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_vector_index_ops_model_status
                ON vector_index_ops(model_id, status, created_at);
            CREATE INDEX IF NOT EXISTS idx_vector_index_ops_chunk_id
                ON vector_index_ops(chunk_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS node_search_fts USING fts5(
                node_id UNINDEXED,
                title,
                body,
                content_text
            );
            """
        )
        self._ensure_content_item_columns()
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO schema_info(key, value)
                VALUES ('schema_version', '5')
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """
            )
        self._refresh_all_search_indexes()

    def _ensure_content_item_columns(self) -> None:
        rows = self._connection.execute("PRAGMA table_info(content_items)").fetchall()
        existing_columns = {str(row["name"]) for row in rows}
        required_columns = {
            "extraction_status": (
                "ALTER TABLE content_items ADD COLUMN extraction_status "
                "TEXT NOT NULL DEFAULT 'pending'"
            ),
            "extracted_text": (
                "ALTER TABLE content_items ADD COLUMN extracted_text "
                "TEXT NOT NULL DEFAULT ''"
            ),
            "extracted_metadata_json": (
                "ALTER TABLE content_items ADD COLUMN extracted_metadata_json "
                "TEXT NOT NULL DEFAULT '{}'"
            ),
            "extraction_error": "ALTER TABLE content_items ADD COLUMN extraction_error TEXT",
        }
        for column_name, statement in required_columns.items():
            if column_name in existing_columns:
                continue
            with self._connection:
                self._connection.execute(statement)

    def _refresh_node_search_index(self, node_id: str) -> None:
        self._connection.execute("DELETE FROM node_search_fts WHERE node_id = ?", (node_id,))

        node_row = self._connection.execute(
            "SELECT node_id, title, body, deleted_at FROM nodes WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        if node_row is None or node_row["deleted_at"] is not None:
            return

        content_row = self._connection.execute(
            """
            SELECT COALESCE(GROUP_CONCAT(extracted_text, ' '), '') AS content_text
            FROM content_items
            WHERE node_id = ?
              AND deleted_at IS NULL
              AND extraction_status = 'done'
            """,
            (node_id,),
        ).fetchone()
        content_text = ""
        if content_row is not None and content_row["content_text"] is not None:
            content_text = str(content_row["content_text"])

        self._connection.execute(
            """
            INSERT INTO node_search_fts(node_id, title, body, content_text)
            VALUES (?, ?, ?, ?)
            """,
            (node_id, str(node_row["title"]), str(node_row["body"]), content_text),
        )

    def _refresh_all_search_indexes(self) -> None:
        self._connection.execute("DELETE FROM node_search_fts")
        rows = self._connection.execute("SELECT node_id FROM nodes").fetchall()
        for row in rows:
            self._refresh_node_search_index(str(row["node_id"]))

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

    def _get_live_tag(self, tag_id: str) -> Mapping[str, Any]:
        row = self._connection.execute(
            "SELECT * FROM tags WHERE tag_id = ? AND deleted_at IS NULL",
            (tag_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Tag not found or deleted: {tag_id}")
        return _row_to_dict(row)

    def _get_live_group(self, group_id: str) -> Mapping[str, Any]:
        row = self._connection.execute(
            'SELECT * FROM "groups" WHERE group_id = ? AND deleted_at IS NULL',
            (group_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Group not found or deleted: {group_id}")
        return _row_to_dict(row)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {str(key): row[key] for key in row.keys()}


def _content_item_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    payload = _row_to_dict(row)
    raw_metadata = payload.get("extracted_metadata_json")
    parsed_metadata: Mapping[str, str] = {}
    if isinstance(raw_metadata, str) and raw_metadata:
        try:
            loaded = json.loads(raw_metadata)
            if isinstance(loaded, dict):
                parsed_metadata = {
                    str(key): str(value)
                    for key, value in loaded.items()
                }
        except json.JSONDecodeError:
            parsed_metadata = {}
    payload["extracted_metadata"] = parsed_metadata
    return payload


def _coerce_vector(raw: Any) -> list[float]:
    if isinstance(raw, Sequence) and not isinstance(raw, str | bytes | bytearray):
        return [float(value) for value in raw]
    raise TypeError("Embedding vector must be a sequence of floats.")


def _vector_to_blob(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _blob_to_vector(blob: bytes, dim: int) -> list[float]:
    expected_size = dim * struct.calcsize("<f")
    if len(blob) != expected_size:
        raise ValueError(
            f"Invalid vector blob size: expected {expected_size} bytes, got {len(blob)}."
        )
    return [float(value) for value in struct.unpack(f"<{dim}f", blob)]


def _build_fts_query(raw_query: str) -> str:
    terms = [term.strip() for term in raw_query.split() if term.strip()]
    if not terms:
        return ""
    escaped = [term.replace('"', '""') for term in terms]
    quoted = [f'"{term}"' for term in escaped]
    return " AND ".join(quoted)


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")
