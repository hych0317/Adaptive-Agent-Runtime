"""SQLite persistence for application-owned personal knowledge data."""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Iterator
from uuid import NAMESPACE_URL, UUID, uuid5

from applications.personal_knowledge.errors import (
    KnowledgeConflictError,
    KnowledgeInvariantError,
    KnowledgeNotFoundError,
)
from applications.personal_knowledge.models import (
    Category,
    CategoryChangeResult,
    CategoryChangeType,
    CategoryStatus,
    ConfirmedCategoryChange,
    ConfirmedKnowledgeChange,
    KnowledgeChangeType,
    KnowledgeDocument,
    KnowledgeEntry,
    KnowledgeSearchHit,
    KnowledgeStatus,
    ReviewArchive,
    SourceRecord,
    SourceSubscription,
    SourceTextArtifact,
    SearchScope,
)


_SCHEMA_VERSION = 3
_KNOWLEDGE_ENTRY_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "adaptive-agent-runtime/personal-knowledge/entry",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_schema (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    source_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_artifacts (
    artifact_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    artifact_json TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES sources(source_id)
);
CREATE INDEX IF NOT EXISTS idx_source_artifacts_source
    ON source_artifacts(source_id);

CREATE TABLE IF NOT EXISTS review_archives (
    review_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,
    review_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS categories (
    category_id TEXT PRIMARY KEY,
    normalized_name TEXT NOT NULL UNIQUE,
    revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    category_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS category_confirmations (
    confirmation_id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    result_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_snapshots (
    entry_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    PRIMARY KEY (entry_id, revision)
);
CREATE TABLE IF NOT EXISTS knowledge_current (
    entry_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,
    category_id TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_knowledge_current_category
    ON knowledge_current(category_id);
CREATE TABLE IF NOT EXISTS knowledge_confirmations (
    confirmation_id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    entry_id TEXT NOT NULL,
    revision INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    entry_id UNINDEXED,
    title,
    body,
    tags,
    tokenize='trigram'
);
CREATE VIRTUAL TABLE IF NOT EXISTS archive_fts USING fts5(
    artifact_id UNINDEXED,
    source_id UNINDEXED,
    title,
    body,
    tokenize='trigram'
);
CREATE TABLE IF NOT EXISTS source_subscriptions (
    subscription_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,
    enabled INTEGER NOT NULL,
    next_poll_at TEXT NOT NULL,
    subscription_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_subscriptions_due
    ON source_subscriptions(enabled, next_poll_at);
"""


def _model_json(value: object) -> str:
    model_dump = getattr(value, "model_dump", None)
    if model_dump is None:
        raise TypeError("persistence values must be Pydantic models")
    return json.dumps(
        model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(value: object) -> str:
    return sha256(_model_json(value).encode("utf-8")).hexdigest()


def _normalized_category_name(value: str) -> str:
    return value.strip().casefold()


class KnowledgeSQLiteDatabase:
    """One locked SQLite connection for the application domain database."""

    def __init__(self, path: str | Path) -> None:
        raw_path = str(path)
        if raw_path != ":memory:":
            resolved = Path(path).expanduser().resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            raw_path = str(resolved)
        self.path = raw_path
        self._lock = RLock()
        self._connection = sqlite3.connect(
            raw_path,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            if raw_path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.executescript(_SCHEMA)
            row = self._connection.execute(
                "SELECT version FROM knowledge_schema WHERE singleton = 1"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO knowledge_schema(singleton, version) VALUES (1, ?)",
                    (_SCHEMA_VERSION,),
                )
            elif int(row["version"]) == 1:
                self._connection.execute(
                    "UPDATE knowledge_schema SET version = ? WHERE singleton = 1",
                    (_SCHEMA_VERSION,),
                )
                self._backfill_search_indexes()
            elif int(row["version"]) == 2:
                self._connection.execute(
                    "UPDATE knowledge_schema SET version = ? WHERE singleton = 1",
                    (_SCHEMA_VERSION,),
                )
            elif int(row["version"]) != _SCHEMA_VERSION:
                self._connection.close()
                raise KnowledgeInvariantError(
                    f"unsupported knowledge schema version {row['version']}"
                )

    def _backfill_search_indexes(self) -> None:
        self._connection.execute("DELETE FROM knowledge_fts")
        self._connection.execute("DELETE FROM archive_fts")
        rows = self._connection.execute(
            "SELECT snapshots.snapshot_json FROM knowledge_current AS current "
            "JOIN knowledge_snapshots AS snapshots "
            "ON snapshots.entry_id = current.entry_id "
            "AND snapshots.revision = current.revision "
            "WHERE current.status = ?",
            (KnowledgeStatus.ACTIVE.value,),
        ).fetchall()
        for row in rows:
            entry = KnowledgeEntry.model_validate_json(row["snapshot_json"])
            self._connection.execute(
                "INSERT INTO knowledge_fts(entry_id, title, body, tags) "
                "VALUES (?, ?, ?, ?)",
                (
                    str(entry.entry_id),
                    entry.document.title,
                    entry.document.body,
                    " ".join(entry.document.tags),
                ),
            )
        artifacts = self._connection.execute(
            "SELECT artifact_json FROM source_artifacts"
        ).fetchall()
        for row in artifacts:
            artifact = SourceTextArtifact.model_validate_json(row["artifact_json"])
            source_row = self._connection.execute(
                "SELECT source_json FROM sources WHERE source_id = ?",
                (str(artifact.source_id),),
            ).fetchone()
            source = (
                SourceRecord.model_validate_json(source_row["source_json"])
                if source_row is not None
                else None
            )
            self._connection.execute(
                "INSERT INTO archive_fts(artifact_id, source_id, title, body) "
                "VALUES (?, ?, ?, ?)",
                (
                    str(artifact.artifact_id),
                    str(artifact.source_id),
                    source.title if source and source.title else "来源存档",
                    artifact.text,
                ),
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                yield cursor
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
            finally:
                cursor.close()

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cursor = self._connection.cursor()
            try:
                yield cursor
            finally:
                cursor.close()

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class SQLitePersonalKnowledgeStore:
    """Application store enforcing confirmation, history, and trash rules."""

    def __init__(self, path: str | Path) -> None:
        self._database = KnowledgeSQLiteDatabase(path)

    def close(self) -> None:
        self._database.close()

    def __enter__(self) -> SQLitePersonalKnowledgeStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    async def save_source(self, source: SourceRecord) -> None:
        payload = _model_json(source)
        with self._database.transaction() as cursor:
            existing = cursor.execute(
                "SELECT source_json FROM sources WHERE source_id = ?",
                (str(source.source_id),),
            ).fetchone()
            if existing is not None:
                if existing["source_json"] == payload:
                    return
                raise KnowledgeConflictError(
                    "source id was already used with different content"
                )
            cursor.execute(
                "INSERT INTO sources(source_id, source_json) VALUES (?, ?)",
                (str(source.source_id), payload),
            )

    async def load_source(self, source_id: UUID) -> SourceRecord | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT source_json FROM sources WHERE source_id = ?",
                (str(source_id),),
            ).fetchone()
        return (
            SourceRecord.model_validate_json(row["source_json"])
            if row is not None
            else None
        )

    async def save_artifact(self, artifact: SourceTextArtifact) -> None:
        payload = _model_json(artifact)
        with self._database.transaction() as cursor:
            source = cursor.execute(
                "SELECT 1 FROM sources WHERE source_id = ?",
                (str(artifact.source_id),),
            ).fetchone()
            if source is None:
                raise KnowledgeNotFoundError(
                    f"source '{artifact.source_id}' was not found"
                )
            existing = cursor.execute(
                "SELECT artifact_json FROM source_artifacts WHERE artifact_id = ?",
                (str(artifact.artifact_id),),
            ).fetchone()
            if existing is not None:
                if existing["artifact_json"] == payload:
                    return
                raise KnowledgeConflictError(
                    "artifact id was already used with different content"
                )
            cursor.execute(
                "INSERT INTO source_artifacts"
                "(artifact_id, source_id, content_hash, artifact_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    str(artifact.artifact_id),
                    str(artifact.source_id),
                    artifact.content_hash,
                    payload,
                ),
            )
            self._index_artifact(cursor, artifact)

    async def load_artifact(
        self,
        artifact_id: UUID,
    ) -> SourceTextArtifact | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT artifact_json FROM source_artifacts WHERE artifact_id = ?",
                (str(artifact_id),),
            ).fetchone()
        return (
            SourceTextArtifact.model_validate_json(row["artifact_json"])
            if row is not None
            else None
        )

    async def save_review(
        self,
        review: ReviewArchive,
        *,
        expected_revision: int | None,
    ) -> None:
        with self._database.transaction() as cursor:
            self._validate_review_references(cursor, review)
            current = cursor.execute(
                "SELECT revision, review_json FROM review_archives "
                "WHERE review_id = ?",
                (str(review.review_id),),
            ).fetchone()
            payload = _model_json(review)
            if current is None:
                if expected_revision is not None or review.revision != 0:
                    raise KnowledgeConflictError(
                        "new review must start at revision 0"
                    )
                cursor.execute(
                    "INSERT INTO review_archives(review_id, revision, review_json) "
                    "VALUES (?, ?, ?)",
                    (str(review.review_id), review.revision, payload),
                )
                return
            if current["review_json"] == payload:
                return
            if (
                expected_revision is None
                or int(current["revision"]) != expected_revision
                or review.revision != expected_revision + 1
            ):
                raise KnowledgeConflictError(
                    "review write is based on a stale revision"
                )
            cursor.execute(
                "UPDATE review_archives SET revision = ?, review_json = ? "
                "WHERE review_id = ?",
                (review.revision, payload, str(review.review_id)),
            )

    async def load_review(self, review_id: UUID) -> ReviewArchive | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT review_json FROM review_archives WHERE review_id = ?",
                (str(review_id),),
            ).fetchone()
        return (
            ReviewArchive.model_validate_json(row["review_json"])
            if row is not None
            else None
        )

    async def list_reviews(
        self,
        *,
        include_finalized: bool = False,
    ) -> tuple[ReviewArchive, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT review_json FROM review_archives ORDER BY rowid DESC"
            ).fetchall()
        reviews = tuple(
            ReviewArchive.model_validate_json(row["review_json"]) for row in rows
        )
        if include_finalized:
            return reviews
        return tuple(item for item in reviews if item.final_confirmation_id is None)

    async def apply_category_change(
        self,
        change: ConfirmedCategoryChange,
    ) -> CategoryChangeResult:
        fingerprint = _fingerprint(change)
        with self._database.transaction() as cursor:
            replay = cursor.execute(
                "SELECT fingerprint, result_json FROM category_confirmations "
                "WHERE confirmation_id = ?",
                (str(change.confirmation_id),),
            ).fetchone()
            if replay is not None:
                if replay["fingerprint"] != fingerprint:
                    raise KnowledgeConflictError(
                        "category confirmation id was reused with another payload"
                    )
                return CategoryChangeResult.model_validate_json(
                    replay["result_json"]
                )

            if change.change_type is CategoryChangeType.CREATE:
                result = self._create_category(cursor, change)
            else:
                current = self._load_category_cursor(cursor, change.category_id)
                if current is None:
                    raise KnowledgeNotFoundError(
                        f"category '{change.category_id}' was not found"
                    )
                if current.revision != change.expected_revision:
                    raise KnowledgeConflictError(
                        "category change is based on a stale revision"
                    )
                if current.status is not CategoryStatus.ACTIVE:
                    raise KnowledgeInvariantError(
                        "only an active category can be changed"
                    )
                if change.change_type is CategoryChangeType.RENAME:
                    result = self._rename_category(cursor, current, change)
                elif change.change_type is CategoryChangeType.RETIRE:
                    result = self._retire_category(cursor, current, change)
                else:
                    result = self._merge_category(cursor, current, change)

            cursor.execute(
                "INSERT INTO category_confirmations"
                "(confirmation_id, fingerprint, result_json) VALUES (?, ?, ?)",
                (
                    str(change.confirmation_id),
                    fingerprint,
                    _model_json(result),
                ),
            )
            return result

    async def load_category(self, category_id: UUID) -> Category | None:
        with self._database.reader() as cursor:
            return self._load_category_cursor(cursor, category_id)

    async def list_categories(
        self,
        *,
        include_retired: bool = False,
    ) -> tuple[Category, ...]:
        sql = "SELECT category_json FROM categories"
        parameters: tuple[str, ...] = ()
        if not include_retired:
            sql += " WHERE status = ?"
            parameters = (CategoryStatus.ACTIVE.value,)
        sql += " ORDER BY normalized_name"
        with self._database.reader() as cursor:
            rows = cursor.execute(sql, parameters).fetchall()
        return tuple(
            Category.model_validate_json(row["category_json"]) for row in rows
        )

    async def apply_knowledge_change(
        self,
        change: ConfirmedKnowledgeChange,
    ) -> KnowledgeEntry:
        fingerprint = _fingerprint(change)
        with self._database.transaction() as cursor:
            replay = cursor.execute(
                "SELECT fingerprint, entry_id, revision "
                "FROM knowledge_confirmations WHERE confirmation_id = ?",
                (str(change.confirmation_id),),
            ).fetchone()
            if replay is not None:
                if replay["fingerprint"] != fingerprint:
                    raise KnowledgeConflictError(
                        "knowledge confirmation id was reused with another payload"
                    )
                entry = self._load_entry_revision_cursor(
                    cursor,
                    UUID(replay["entry_id"]),
                    int(replay["revision"]),
                )
                if entry is None:
                    entry = self._load_entry_cursor(
                        cursor,
                        UUID(replay["entry_id"]),
                    )
                if entry is None:
                    raise KnowledgeInvariantError("confirmed knowledge is missing")
                return entry

            if change.change_type is KnowledgeChangeType.PUBLISH:
                entry_id = uuid5(
                    _KNOWLEDGE_ENTRY_NAMESPACE,
                    str(change.confirmation_id),
                )
                if self._load_entry_cursor(cursor, entry_id) is not None:
                    raise KnowledgeConflictError("knowledge entry already exists")
                document = self._validated_document(cursor, change)
                entry = KnowledgeEntry(
                    entry_id=entry_id,
                    document=document,
                    last_confirmation_id=change.confirmation_id,
                    created_at=change.confirmed_at,
                    updated_at=change.confirmed_at,
                )
            else:
                target_id = change.target_entry_id
                if target_id is None:
                    raise KnowledgeInvariantError("knowledge change has no target")
                current = self._load_entry_cursor(cursor, target_id)
                if current is None:
                    raise KnowledgeNotFoundError(
                        f"knowledge entry '{target_id}' was not found"
                    )
                if current.revision != change.expected_revision:
                    raise KnowledgeConflictError(
                        "knowledge change is based on a stale revision"
                    )
                entry = self._evolve_entry(cursor, current, change)

            self._write_entry_cursor(cursor, entry)
            cursor.execute(
                "INSERT INTO knowledge_confirmations"
                "(confirmation_id, fingerprint, entry_id, revision) "
                "VALUES (?, ?, ?, ?)",
                (
                    str(change.confirmation_id),
                    fingerprint,
                    str(entry.entry_id),
                    entry.revision,
                ),
            )
            if change.review_id is not None:
                self._finalize_review_cursor(cursor, change.review_id, change)
            return entry

    async def load_entry(self, entry_id: UUID) -> KnowledgeEntry | None:
        with self._database.reader() as cursor:
            return self._load_entry_cursor(cursor, entry_id)

    async def list_entries(
        self,
        *,
        include_trashed: bool = False,
    ) -> tuple[KnowledgeEntry, ...]:
        sql = (
            "SELECT snapshots.snapshot_json FROM knowledge_current AS current "
            "JOIN knowledge_snapshots AS snapshots "
            "ON snapshots.entry_id = current.entry_id "
            "AND snapshots.revision = current.revision"
        )
        parameters: tuple[str, ...] = ()
        if not include_trashed:
            sql += " WHERE current.status = ?"
            parameters = (KnowledgeStatus.ACTIVE.value,)
        sql += " ORDER BY snapshots.entry_id"
        with self._database.reader() as cursor:
            rows = cursor.execute(sql, parameters).fetchall()
        return tuple(
            KnowledgeEntry.model_validate_json(row["snapshot_json"])
            for row in rows
        )

    async def history_for(self, entry_id: UUID) -> tuple[KnowledgeEntry, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT snapshot_json FROM knowledge_snapshots "
                "WHERE entry_id = ? ORDER BY revision",
                (str(entry_id),),
            ).fetchall()
        return tuple(
            KnowledgeEntry.model_validate_json(row["snapshot_json"])
            for row in rows
        )

    async def search(
        self,
        query: str,
        *,
        include_archive: bool = False,
        limit: int = 10,
    ) -> tuple[KnowledgeSearchHit, ...]:
        normalized = query.strip()
        if not normalized:
            return ()
        bounded_limit = max(1, min(limit, 100))
        with self._database.reader() as cursor:
            knowledge_rows = cursor.execute(
                "SELECT entry_id, title, "
                "snippet(knowledge_fts, 2, '[', ']', ' … ', 24) AS excerpt, "
                "bm25(knowledge_fts) AS rank "
                "FROM knowledge_fts WHERE knowledge_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (self._fts_query(normalized), bounded_limit),
            ).fetchall()
            if not knowledge_rows:
                like = f"%{normalized}%"
                knowledge_rows = cursor.execute(
                    "SELECT entry_id, title, substr(body, 1, 180) AS excerpt, "
                    "0.0 AS rank FROM knowledge_fts "
                    "WHERE title LIKE ? OR body LIKE ? OR tags LIKE ? LIMIT ?",
                    (like, like, like, bounded_limit),
                ).fetchall()
            archive_rows: list[sqlite3.Row] = []
            if include_archive:
                archive_rows = cursor.execute(
                    "SELECT artifact_id, source_id, title, "
                    "snippet(archive_fts, 3, '[', ']', ' … ', 24) AS excerpt, "
                    "bm25(archive_fts) AS rank "
                    "FROM archive_fts WHERE archive_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (self._fts_query(normalized), bounded_limit),
                ).fetchall()
                if not archive_rows:
                    like = f"%{normalized}%"
                    archive_rows = cursor.execute(
                        "SELECT artifact_id, source_id, title, "
                        "substr(body, 1, 180) AS excerpt, 0.0 AS rank "
                        "FROM archive_fts WHERE title LIKE ? OR body LIKE ? LIMIT ?",
                        (like, like, bounded_limit),
                    ).fetchall()
            hits = [
                KnowledgeSearchHit(
                    scope=SearchScope.KNOWLEDGE,
                    entry_id=UUID(row["entry_id"]),
                    title=row["title"],
                    snippet=row["excerpt"] or row["title"],
                    score=-float(row["rank"]),
                )
                for row in knowledge_rows
            ]
            for row in archive_rows:
                source = self._load_source_cursor(cursor, UUID(row["source_id"]))
                hits.append(
                    KnowledgeSearchHit(
                        scope=SearchScope.ARCHIVE,
                        source_id=UUID(row["source_id"]),
                        artifact_id=UUID(row["artifact_id"]),
                        title=row["title"] or "来源存档",
                        snippet=row["excerpt"] or row["title"] or "来源存档",
                        score=-float(row["rank"]),
                        source_url=(source.canonical_url if source else None),
                    )
                )
        return tuple(sorted(hits, key=lambda item: item.score, reverse=True)[:bounded_limit])

    async def save_subscription(
        self,
        subscription: SourceSubscription,
        *,
        expected_revision: int | None,
    ) -> None:
        payload = _model_json(subscription)
        with self._database.transaction() as cursor:
            row = cursor.execute(
                "SELECT revision, subscription_json FROM source_subscriptions "
                "WHERE subscription_id = ?",
                (str(subscription.subscription_id),),
            ).fetchone()
            if row is None:
                if expected_revision is not None or subscription.revision != 0:
                    raise KnowledgeConflictError(
                        "new subscription must start at revision 0"
                    )
                cursor.execute(
                    "INSERT INTO source_subscriptions"
                    "(subscription_id, revision, enabled, next_poll_at, subscription_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        str(subscription.subscription_id),
                        subscription.revision,
                        int(subscription.enabled),
                        subscription.next_poll_at.isoformat(),
                        payload,
                    ),
                )
                return
            if row["subscription_json"] == payload:
                return
            if (
                expected_revision is None
                or int(row["revision"]) != expected_revision
                or subscription.revision != expected_revision + 1
            ):
                raise KnowledgeConflictError(
                    "subscription update is based on a stale revision"
                )
            cursor.execute(
                "UPDATE source_subscriptions SET revision = ?, enabled = ?, "
                "next_poll_at = ?, subscription_json = ? WHERE subscription_id = ?",
                (
                    subscription.revision,
                    int(subscription.enabled),
                    subscription.next_poll_at.isoformat(),
                    payload,
                    str(subscription.subscription_id),
                ),
            )

    async def list_subscriptions(
        self,
        *,
        enabled_only: bool = False,
    ) -> tuple[SourceSubscription, ...]:
        sql = "SELECT subscription_json FROM source_subscriptions"
        parameters: tuple[int, ...] = ()
        if enabled_only:
            sql += " WHERE enabled = ?"
            parameters = (1,)
        sql += " ORDER BY next_poll_at, subscription_id"
        with self._database.reader() as cursor:
            rows = cursor.execute(sql, parameters).fetchall()
        return tuple(
            SourceSubscription.model_validate_json(row["subscription_json"])
            for row in rows
        )

    @staticmethod
    def _fts_query(query: str) -> str:
        tokens: list[str] = []
        for run in re.findall(r"[\w\u3400-\u9fff]+", query, flags=re.UNICODE):
            if len(run) <= 3:
                tokens.append(run)
            else:
                tokens.extend(run[index : index + 3] for index in range(len(run) - 2))
        unique = tuple(dict.fromkeys(item for item in tokens if len(item) >= 3))
        if not unique:
            return '"' + query.replace('"', '""') + '"'
        return " OR ".join('"' + item.replace('"', '""') + '"' for item in unique[:24])

    @staticmethod
    def _load_source_cursor(
        cursor: sqlite3.Cursor,
        source_id: UUID,
    ) -> SourceRecord | None:
        row = cursor.execute(
            "SELECT source_json FROM sources WHERE source_id = ?",
            (str(source_id),),
        ).fetchone()
        return SourceRecord.model_validate_json(row["source_json"]) if row else None

    @staticmethod
    def _index_artifact(
        cursor: sqlite3.Cursor | sqlite3.Connection,
        artifact: SourceTextArtifact,
    ) -> None:
        source_row = cursor.execute(
            "SELECT source_json FROM sources WHERE source_id = ?",
            (str(artifact.source_id),),
        ).fetchone()
        source = (
            SourceRecord.model_validate_json(source_row["source_json"])
            if source_row is not None
            else None
        )
        cursor.execute(
            "DELETE FROM archive_fts WHERE artifact_id = ?",
            (str(artifact.artifact_id),),
        )
        cursor.execute(
            "INSERT INTO archive_fts(artifact_id, source_id, title, body) "
            "VALUES (?, ?, ?, ?)",
            (
                str(artifact.artifact_id),
                str(artifact.source_id),
                source.title if source and source.title else "来源存档",
                artifact.text,
            ),
        )

    @staticmethod
    def _index_entry(
        cursor: sqlite3.Cursor | sqlite3.Connection,
        entry: KnowledgeEntry,
    ) -> None:
        cursor.execute(
            "DELETE FROM knowledge_fts WHERE entry_id = ?",
            (str(entry.entry_id),),
        )
        if entry.status is KnowledgeStatus.ACTIVE:
            cursor.execute(
                "INSERT INTO knowledge_fts(entry_id, title, body, tags) "
                "VALUES (?, ?, ?, ?)",
                (
                    str(entry.entry_id),
                    entry.document.title,
                    entry.document.body,
                    " ".join(entry.document.tags),
                ),
            )

    @staticmethod
    def _validate_review_references(
        cursor: sqlite3.Cursor,
        review: ReviewArchive,
    ) -> None:
        for source_id in review.source_ids:
            row = cursor.execute(
                "SELECT 1 FROM sources WHERE source_id = ?",
                (str(source_id),),
            ).fetchone()
            if row is None:
                raise KnowledgeNotFoundError(
                    f"source '{source_id}' was not found"
                )
        for artifact_id in review.artifact_ids:
            row = cursor.execute(
                "SELECT 1 FROM source_artifacts WHERE artifact_id = ?",
                (str(artifact_id),),
            ).fetchone()
            if row is None:
                raise KnowledgeNotFoundError(
                    f"artifact '{artifact_id}' was not found"
                )

    @staticmethod
    def _load_category_cursor(
        cursor: sqlite3.Cursor,
        category_id: UUID,
    ) -> Category | None:
        row = cursor.execute(
            "SELECT category_json FROM categories WHERE category_id = ?",
            (str(category_id),),
        ).fetchone()
        return (
            Category.model_validate_json(row["category_json"])
            if row is not None
            else None
        )

    def _create_category(
        self,
        cursor: sqlite3.Cursor,
        change: ConfirmedCategoryChange,
    ) -> CategoryChangeResult:
        if self._load_category_cursor(cursor, change.category_id) is not None:
            raise KnowledgeConflictError("category already exists")
        if change.name is None:
            raise KnowledgeInvariantError("category creation has no name")
        category = Category(
            category_id=change.category_id,
            name=change.name.strip(),
            created_at=change.confirmed_at,
            updated_at=change.confirmed_at,
        )
        self._insert_category_cursor(cursor, category)
        return CategoryChangeResult(category=category)

    def _rename_category(
        self,
        cursor: sqlite3.Cursor,
        current: Category,
        change: ConfirmedCategoryChange,
    ) -> CategoryChangeResult:
        if change.name is None:
            raise KnowledgeInvariantError("category rename has no name")
        updated = current.model_copy(
            update={
                "name": change.name.strip(),
                "revision": current.revision + 1,
                "updated_at": change.confirmed_at,
            }
        )
        self._update_category_cursor(cursor, updated)
        return CategoryChangeResult(category=updated)

    def _retire_category(
        self,
        cursor: sqlite3.Cursor,
        current: Category,
        change: ConfirmedCategoryChange,
    ) -> CategoryChangeResult:
        in_use = cursor.execute(
            "SELECT 1 FROM knowledge_current WHERE category_id = ? LIMIT 1",
            (str(current.category_id),),
        ).fetchone()
        if in_use is not None:
            raise KnowledgeInvariantError(
                "a category with knowledge entries must be merged, not retired"
            )
        updated = current.model_copy(
            update={
                "status": CategoryStatus.RETIRED,
                "revision": current.revision + 1,
                "updated_at": change.confirmed_at,
            }
        )
        self._update_category_cursor(cursor, updated)
        return CategoryChangeResult(category=updated)

    def _merge_category(
        self,
        cursor: sqlite3.Cursor,
        current: Category,
        change: ConfirmedCategoryChange,
    ) -> CategoryChangeResult:
        target_id = change.target_category_id
        if target_id is None:
            raise KnowledgeInvariantError("category merge has no target")
        target = self._load_category_cursor(cursor, target_id)
        if target is None:
            raise KnowledgeNotFoundError(
                f"category '{target_id}' was not found"
            )
        if target.status is not CategoryStatus.ACTIVE:
            raise KnowledgeInvariantError("merge target must be active")
        rows = cursor.execute(
            "SELECT entry_id FROM knowledge_current WHERE category_id = ? "
            "ORDER BY entry_id",
            (str(current.category_id),),
        ).fetchall()
        affected: list[UUID] = []
        for row in rows:
            entry_id = UUID(row["entry_id"])
            entry = self._load_entry_cursor(cursor, entry_id)
            if entry is None:
                raise KnowledgeInvariantError("knowledge current snapshot is missing")
            document = entry.document.model_copy(
                update={"category_id": target.category_id}
            )
            updated_entry = entry.model_copy(
                update={
                    "document": document,
                    "revision": entry.revision + 1,
                    "last_confirmation_id": change.confirmation_id,
                    "updated_at": change.confirmed_at,
                }
            )
            self._write_entry_cursor(cursor, updated_entry)
            affected.append(entry_id)
        retired = current.model_copy(
            update={
                "status": CategoryStatus.RETIRED,
                "merged_into": target.category_id,
                "revision": current.revision + 1,
                "updated_at": change.confirmed_at,
            }
        )
        self._update_category_cursor(cursor, retired)
        return CategoryChangeResult(
            category=retired,
            affected_entry_ids=tuple(affected),
        )

    @staticmethod
    def _insert_category_cursor(
        cursor: sqlite3.Cursor,
        category: Category,
    ) -> None:
        try:
            cursor.execute(
                "INSERT INTO categories"
                "(category_id, normalized_name, revision, status, category_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(category.category_id),
                    _normalized_category_name(category.name),
                    category.revision,
                    category.status.value,
                    _model_json(category),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise KnowledgeConflictError(
                "category names must be unique"
            ) from exc

    @staticmethod
    def _update_category_cursor(
        cursor: sqlite3.Cursor,
        category: Category,
    ) -> None:
        try:
            cursor.execute(
                "UPDATE categories SET normalized_name = ?, revision = ?, "
                "status = ?, category_json = ? WHERE category_id = ?",
                (
                    _normalized_category_name(category.name),
                    category.revision,
                    category.status.value,
                    _model_json(category),
                    str(category.category_id),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise KnowledgeConflictError(
                "category names must be unique"
            ) from exc

    def _validated_document(
        self,
        cursor: sqlite3.Cursor,
        change: ConfirmedKnowledgeChange,
    ) -> KnowledgeDocument:
        document = change.document
        review_id = change.review_id
        if document is None or review_id is None:
            raise KnowledgeInvariantError("confirmed knowledge lacks review content")
        category = self._load_category_cursor(cursor, document.category_id)
        if category is None or category.status is not CategoryStatus.ACTIVE:
            raise KnowledgeInvariantError("knowledge requires an active category")
        review = cursor.execute(
            "SELECT 1 FROM review_archives WHERE review_id = ?",
            (str(review_id),),
        ).fetchone()
        if review is None:
            raise KnowledgeNotFoundError(f"review '{review_id}' was not found")
        for citation in document.citations:
            source = cursor.execute(
                "SELECT 1 FROM sources WHERE source_id = ?",
                (str(citation.source_id),),
            ).fetchone()
            if source is None:
                raise KnowledgeNotFoundError(
                    f"citation source '{citation.source_id}' was not found"
                )
            if citation.artifact_id is not None:
                artifact = cursor.execute(
                    "SELECT source_id FROM source_artifacts WHERE artifact_id = ?",
                    (str(citation.artifact_id),),
                ).fetchone()
                if artifact is None:
                    raise KnowledgeNotFoundError(
                        f"citation artifact '{citation.artifact_id}' was not found"
                    )
                if artifact["source_id"] != str(citation.source_id):
                    raise KnowledgeInvariantError(
                        "citation artifact belongs to another source"
                    )
        return document

    def _evolve_entry(
        self,
        cursor: sqlite3.Cursor,
        current: KnowledgeEntry,
        change: ConfirmedKnowledgeChange,
    ) -> KnowledgeEntry:
        common = {
            "revision": current.revision + 1,
            "last_confirmation_id": change.confirmation_id,
            "updated_at": change.confirmed_at,
        }
        if change.change_type is KnowledgeChangeType.UPDATE:
            if current.status is not KnowledgeStatus.ACTIVE:
                raise KnowledgeInvariantError("trashed knowledge cannot be updated")
            document = self._validated_document(cursor, change)
            return current.model_copy(update={**common, "document": document})
        if change.change_type is KnowledgeChangeType.DELETE:
            if current.status is not KnowledgeStatus.ACTIVE:
                raise KnowledgeInvariantError("only active knowledge can be deleted")
            return current.model_copy(
                update={
                    **common,
                    "status": KnowledgeStatus.TRASHED,
                    "deleted_at": change.confirmed_at,
                }
            )
        if current.status is not KnowledgeStatus.TRASHED:
            raise KnowledgeInvariantError("only trashed knowledge can be restored")
        return current.model_copy(
            update={
                **common,
                "status": KnowledgeStatus.ACTIVE,
                "deleted_at": None,
            }
        )

    @staticmethod
    def _load_entry_revision_cursor(
        cursor: sqlite3.Cursor,
        entry_id: UUID,
        revision: int,
    ) -> KnowledgeEntry | None:
        row = cursor.execute(
            "SELECT snapshot_json FROM knowledge_snapshots "
            "WHERE entry_id = ? AND revision = ?",
            (str(entry_id), revision),
        ).fetchone()
        return (
            KnowledgeEntry.model_validate_json(row["snapshot_json"])
            if row is not None
            else None
        )

    def _load_entry_cursor(
        self,
        cursor: sqlite3.Cursor,
        entry_id: UUID,
    ) -> KnowledgeEntry | None:
        row = cursor.execute(
            "SELECT revision FROM knowledge_current WHERE entry_id = ?",
            (str(entry_id),),
        ).fetchone()
        if row is None:
            return None
        return self._load_entry_revision_cursor(
            cursor,
            entry_id,
            int(row["revision"]),
        )

    def _write_entry_cursor(
        self,
        cursor: sqlite3.Cursor,
        entry: KnowledgeEntry,
    ) -> None:
        cursor.execute(
            "INSERT INTO knowledge_snapshots"
            "(entry_id, revision, snapshot_json) VALUES (?, ?, ?)",
            (str(entry.entry_id), entry.revision, _model_json(entry)),
        )
        cursor.execute(
            "INSERT INTO knowledge_current(entry_id, revision, category_id, status) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(entry_id) DO UPDATE SET "
            "revision = excluded.revision, category_id = excluded.category_id, "
            "status = excluded.status",
            (
                str(entry.entry_id),
                entry.revision,
                str(entry.document.category_id),
                entry.status.value,
            ),
        )
        cursor.execute(
            "DELETE FROM knowledge_snapshots WHERE entry_id = ? AND revision < ?",
            (str(entry.entry_id), max(0, entry.revision - 2)),
        )
        self._index_entry(cursor, entry)

    @staticmethod
    def _finalize_review_cursor(
        cursor: sqlite3.Cursor,
        review_id: UUID,
        change: ConfirmedKnowledgeChange,
    ) -> None:
        row = cursor.execute(
            "SELECT review_json FROM review_archives WHERE review_id = ?",
            (str(review_id),),
        ).fetchone()
        if row is None:
            raise KnowledgeNotFoundError(f"review '{review_id}' was not found")
        review = ReviewArchive.model_validate_json(row["review_json"])
        if (
            review.final_confirmation_id is not None
            and review.final_confirmation_id != change.confirmation_id
        ):
            raise KnowledgeInvariantError(
                "review was already finalized by another confirmation"
            )
        finalized = review.model_copy(
            update={
                "final_confirmation_id": change.confirmation_id,
                "revision": review.revision + 1,
                "updated_at": max(review.updated_at, change.confirmed_at),
            }
        )
        cursor.execute(
            "UPDATE review_archives SET revision = ?, review_json = ? "
            "WHERE review_id = ?",
            (finalized.revision, _model_json(finalized), str(review_id)),
        )
