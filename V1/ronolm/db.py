import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ronolm.embedding_client import (
    EMBEDDING_MODEL_NAME,
    EmbeddingError,
    build_memory_embedding_text,
    get_embedding,
)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ronolm.sqlite"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (thread_id) REFERENCES threads(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            source_message_id TEXT NOT NULL,
            type TEXT NOT NULL,
            subject TEXT NOT NULL,
            content TEXT NOT NULL,
            lifetime TEXT NOT NULL,
            importance REAL NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (thread_id) REFERENCES threads(id),
            FOREIGN KEY (source_message_id) REFERENCES messages(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_embeddings (
            memory_id TEXT PRIMARY KEY,
            embedding_model TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (memory_id) REFERENCES memories(id)
        )
        """
    )

    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS topic_units (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            global_summary TEXT,
            global_state_json TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS topic_episodes (
            id TEXT PRIMARY KEY,
            topic_unit_id TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT,
            current_state_json TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            ended_at TEXT,
            FOREIGN KEY (topic_unit_id) REFERENCES topic_units(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS episode_threads (
            episode_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            local_summary TEXT,
            message_count INTEGER NOT NULL DEFAULT 0,
            relevance_score REAL NOT NULL DEFAULT 1.0,
            first_message_id TEXT,
            last_message_id TEXT,
            first_activity_at TEXT,
            last_activity_at TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            PRIMARY KEY (episode_id, thread_id),
            FOREIGN KEY (episode_id) REFERENCES topic_episodes(id) ON DELETE CASCADE,
            FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE,
            FOREIGN KEY (first_message_id) REFERENCES messages(id) ON DELETE SET NULL,
            FOREIGN KEY (last_message_id) REFERENCES messages(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS message_episodes (
            message_id TEXT NOT NULL,
            episode_id TEXT NOT NULL,
            relevance_score REAL NOT NULL,
            contribution_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (message_id, episode_id),
            FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
            FOREIGN KEY (episode_id) REFERENCES topic_episodes(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS semantic_embeddings (
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            vector_json TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (entity_type, entity_id, embedding_model)
        );

        CREATE INDEX IF NOT EXISTS idx_topic_units_status_updated
            ON topic_units(status, updated_at);
        CREATE INDEX IF NOT EXISTS idx_topic_episodes_topic_status
            ON topic_episodes(topic_unit_id, status, updated_at);
        CREATE INDEX IF NOT EXISTS idx_episode_threads_thread_activity
            ON episode_threads(thread_id, last_activity_at);
        CREATE INDEX IF NOT EXISTS idx_message_episodes_episode
            ON message_episodes(episode_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_semantic_embeddings_type
            ON semantic_embeddings(entity_type);
        """
    )

    conn.commit()
    conn.close()


def create_thread(title: str = "Default thread") -> str:
    thread_id = f"thread_{uuid.uuid4().hex}"
    timestamp = now_iso()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO threads (id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (thread_id, title, timestamp, timestamp),
    )

    conn.commit()
    conn.close()

    return thread_id


def get_latest_thread_id() -> str | None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id FROM threads
        ORDER BY updated_at DESC
        LIMIT 1
        """
    )

    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    return row["id"]


def get_or_create_thread() -> str:
    thread_id = get_latest_thread_id()

    if thread_id is not None:
        return thread_id

    return create_thread()


def add_message(thread_id: str, role: str, content: str) -> str:
    if role not in {"user", "assistant"}:
        raise ValueError(f"Invalid role: {role}")

    message_id = f"msg_{uuid.uuid4().hex}"
    timestamp = now_iso()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO messages (id, thread_id, role, content, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (message_id, thread_id, role, content, timestamp),
    )

    cur.execute(
        """
        UPDATE threads
        SET updated_at = ?
        WHERE id = ?
        """,
        (timestamp, thread_id),
    )

    conn.commit()
    conn.close()

    return message_id


def list_messages(thread_id: str) -> list[sqlite3.Row]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, role, content, created_at
        FROM messages
        WHERE thread_id = ?
        ORDER BY created_at ASC
        """,
        (thread_id,),
    )

    rows = cur.fetchall()
    conn.close()

    return rows


def get_message(message_id: str) -> sqlite3.Row | None:
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT * FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
    finally:
        conn.close()



def add_memory(
    thread_id: str,
    source_message_id: str,
    memory_type: str,
    subject: str,
    content: str,
    lifetime: str,
    importance: float,
    confidence: float,
) -> str:
    memory_id = f"mem_{uuid.uuid4().hex}"
    timestamp = now_iso()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id FROM memories
        WHERE type = ?
          AND subject = ?
          AND content = ?
          AND status = 'active'
        LIMIT 1
        """,
        (memory_type, subject, content),
    )

    existing = cur.fetchone()
    if existing is not None:
        conn.close()
        memory_id = existing["id"]
        _try_ensure_memory_embedding(
            {
                "id": memory_id,
                "type": memory_type,
                "subject": subject,
                "content": content,
                "lifetime": lifetime,
            }
        )
        return memory_id

    cur.execute(
        """
        INSERT INTO memories (
            id,
            thread_id,
            source_message_id,
            type,
            subject,
            content,
            lifetime,
            importance,
            confidence,
            status,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            memory_id,
            thread_id,
            source_message_id,
            memory_type,
            subject,
            content,
            lifetime,
            importance,
            confidence,
            "active",
            timestamp,
            timestamp,
        ),
    )

    conn.commit()
    conn.close()

    _try_ensure_memory_embedding(
        {
            "id": memory_id,
            "type": memory_type,
            "subject": subject,
            "content": content,
            "lifetime": lifetime,
        }
    )

    return memory_id


def list_memories() -> list[sqlite3.Row]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, type, subject, content, lifetime, importance, confidence, created_at
        FROM memories
        WHERE status = 'active'
        ORDER BY created_at ASC
        """
    )

    rows = cur.fetchall()
    conn.close()

    return rows


def add_memory_embedding(
    memory_id: str, embedding_model: str, vector: list[float]
) -> None:
    timestamp = now_iso()
    vector_json = json.dumps([float(value) for value in vector])

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO memory_embeddings (
            memory_id, embedding_model, vector_json, created_at
        )
        VALUES (?, ?, ?, ?)
        ON CONFLICT(memory_id) DO UPDATE SET
            embedding_model = excluded.embedding_model,
            vector_json = excluded.vector_json,
            created_at = excluded.created_at
        """,
        (memory_id, embedding_model, vector_json, timestamp),
    )
    conn.commit()
    conn.close()


def get_memory_embedding(memory_id: str) -> list[float] | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT vector_json FROM memory_embeddings WHERE memory_id = ?",
        (memory_id,),
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    return [float(value) for value in json.loads(row["vector_json"])]


def list_memories_with_embeddings() -> list[dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            m.id,
            m.type,
            m.subject,
            m.content,
            m.lifetime,
            m.importance,
            m.confidence,
            m.created_at,
            e.embedding_model,
            e.vector_json
        FROM memories AS m
        LEFT JOIN memory_embeddings AS e ON e.memory_id = m.id
        WHERE m.status = 'active'
        ORDER BY m.created_at ASC
        """
    )
    rows = cur.fetchall()
    conn.close()

    memories: list[dict[str, Any]] = []
    for row in rows:
        memory = dict(row)
        vector_json = memory.pop("vector_json")
        memory["embedding"] = (
            [float(value) for value in json.loads(vector_json)]
            if vector_json is not None
            else None
        )
        memories.append(memory)

    return memories


def ensure_memory_embedding(memory: dict[str, Any]) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT embedding_model
        FROM memory_embeddings
        WHERE memory_id = ?
        """,
        (memory["id"],),
    )
    row = cur.fetchone()
    conn.close()

    if row is not None and row["embedding_model"] == EMBEDDING_MODEL_NAME:
        return False

    vector = get_embedding(build_memory_embedding_text(memory))
    add_memory_embedding(memory["id"], EMBEDDING_MODEL_NAME, vector)
    return True


def _try_ensure_memory_embedding(memory: dict[str, Any]) -> None:
    try:
        ensure_memory_embedding(memory)
    except EmbeddingError:
        # Memory storage remains usable if the optional local model is unavailable.
        pass


# Step 9 topic repository ----------------------------------------------------

TOPIC_UNIT_STATUSES = {"active", "dormant", "closed"}
TOPIC_EPISODE_STATUSES = {"active", "dormant", "completed", "abandoned"}
EPISODE_THREAD_STATUSES = {"active", "dormant", "closed"}
CONTRIBUTION_TYPES = {
    "context",
    "decision",
    "question",
    "answer",
    "next_action",
    "correction",
    "transition",
    "evidence",
    "other",
}
SEMANTIC_ENTITY_TYPES = {"topic_unit", "topic_episode", "episode_thread"}


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _validate_status(value: str, allowed: set[str], label: str) -> str:
    if value not in allowed:
        raise ValueError(f"Invalid {label} status: {value}")
    return value


def _json_text(value: Any | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = json.loads(value)
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def create_topic_unit(
    title: str,
    description: str | None = None,
    global_summary: str | None = None,
    global_state: Any | None = None,
    status: str = "active",
    topic_unit_id: str | None = None,
    global_state_json: Any | None = None,
) -> str:
    title = title.strip()
    if not title:
        raise ValueError("Topic Unit title cannot be empty")
    if global_state is not None and global_state_json is not None:
        raise ValueError("Provide global_state or global_state_json, not both")
    if global_state_json is not None:
        global_state = global_state_json
    _validate_status(status, TOPIC_UNIT_STATUSES, "Topic Unit")
    existing = find_topic_unit_by_title(title)
    if existing is not None:
        return str(existing["id"])

    entity_id = topic_unit_id or f"topic_{uuid.uuid4().hex}"
    timestamp = now_iso()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO topic_units (
                id, title, description, global_summary, global_state_json,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity_id,
                title,
                description,
                global_summary,
                _json_text(global_state),
                status,
                timestamp,
                timestamp,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return entity_id


def update_topic_unit(topic_unit_id: str, **fields: Any) -> bool:
    allowed = {
        "title",
        "description",
        "global_summary",
        "global_state_json",
        "global_state",
        "status",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Unknown Topic Unit fields: {sorted(unknown)}")
    if "title" in fields:
        fields["title"] = str(fields["title"]).strip()
        if not fields["title"]:
            raise ValueError("Topic Unit title cannot be empty")
    if "status" in fields:
        _validate_status(str(fields["status"]), TOPIC_UNIT_STATUSES, "Topic Unit")
    if "global_state" in fields:
        fields["global_state_json"] = fields.pop("global_state")
    if "global_state_json" in fields:
        fields["global_state_json"] = _json_text(fields["global_state_json"])
    if not fields:
        return False

    fields["updated_at"] = now_iso()
    assignments = ", ".join(f"{name} = ?" for name in fields)
    conn = get_connection()
    try:
        cur = conn.execute(
            f"UPDATE topic_units SET {assignments} WHERE id = ?",
            (*fields.values(), topic_unit_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_topic_unit(topic_unit_id: str) -> sqlite3.Row | None:
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT * FROM topic_units WHERE id = ?", (topic_unit_id,)
        ).fetchone()
    finally:
        conn.close()


def list_topic_units(status: str | None = None) -> list[sqlite3.Row]:
    conn = get_connection()
    try:
        if status is None:
            return conn.execute(
                "SELECT * FROM topic_units ORDER BY updated_at DESC, title"
            ).fetchall()
        _validate_status(status, TOPIC_UNIT_STATUSES, "Topic Unit")
        return conn.execute(
            """
            SELECT * FROM topic_units
            WHERE status = ?
            ORDER BY updated_at DESC, title
            """,
            (status,),
        ).fetchall()
    finally:
        conn.close()


def find_topic_unit_by_title(title: str) -> sqlite3.Row | None:
    conn = get_connection()
    try:
        return conn.execute(
            """
            SELECT * FROM topic_units
            WHERE lower(trim(title)) = lower(trim(?))
            LIMIT 1
            """,
            (title,),
        ).fetchone()
    finally:
        conn.close()


def create_topic_episode(
    topic_unit_id: str,
    title: str,
    summary: str | None = None,
    current_state: Any | None = None,
    status: str = "active",
    started_at: str | None = None,
    ended_at: str | None = None,
    episode_id: str | None = None,
    current_state_json: Any | None = None,
) -> str:
    title = title.strip()
    if not title:
        raise ValueError("Topic Episode title cannot be empty")
    if current_state is not None and current_state_json is not None:
        raise ValueError("Provide current_state or current_state_json, not both")
    if current_state_json is not None:
        current_state = current_state_json
    _validate_status(status, TOPIC_EPISODE_STATUSES, "Topic Episode")
    conn = get_connection()
    try:
        existing = conn.execute(
            """
            SELECT id FROM topic_episodes
            WHERE topic_unit_id = ?
              AND lower(trim(title)) = lower(trim(?))
            LIMIT 1
            """,
            (topic_unit_id, title),
        ).fetchone()
        if existing is not None:
            return str(existing["id"])

        entity_id = episode_id or f"episode_{uuid.uuid4().hex}"
        timestamp = now_iso()
        conn.execute(
            """
            INSERT INTO topic_episodes (
                id, topic_unit_id, title, summary, current_state_json,
                status, started_at, updated_at, ended_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity_id,
                topic_unit_id,
                title,
                summary,
                _json_text(current_state),
                status,
                started_at or timestamp,
                timestamp,
                ended_at,
            ),
        )
        conn.commit()
        return entity_id
    finally:
        conn.close()


def update_topic_episode(episode_id: str, **fields: Any) -> bool:
    allowed = {
        "title",
        "summary",
        "current_state_json",
        "current_state",
        "status",
        "ended_at",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Unknown Topic Episode fields: {sorted(unknown)}")
    if "title" in fields:
        fields["title"] = str(fields["title"]).strip()
        if not fields["title"]:
            raise ValueError("Topic Episode title cannot be empty")
    if "status" in fields:
        _validate_status(str(fields["status"]), TOPIC_EPISODE_STATUSES, "Topic Episode")
    if "current_state" in fields:
        fields["current_state_json"] = fields.pop("current_state")
    if "current_state_json" in fields:
        fields["current_state_json"] = _json_text(fields["current_state_json"])
    if not fields:
        return False

    fields["updated_at"] = now_iso()
    assignments = ", ".join(f"{name} = ?" for name in fields)
    conn = get_connection()
    try:
        cur = conn.execute(
            f"UPDATE topic_episodes SET {assignments} WHERE id = ?",
            (*fields.values(), episode_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_topic_episode(episode_id: str) -> sqlite3.Row | None:
    conn = get_connection()
    try:
        return conn.execute(
            """
            SELECT e.*, u.title AS topic_title
            FROM topic_episodes AS e
            JOIN topic_units AS u ON u.id = e.topic_unit_id
            WHERE e.id = ?
            """,
            (episode_id,),
        ).fetchone()
    finally:
        conn.close()


def list_topic_episodes(topic_unit_id: str | None = None) -> list[sqlite3.Row]:
    conn = get_connection()
    try:
        if topic_unit_id is None:
            return conn.execute(
                """
                SELECT e.*, u.title AS topic_title
                FROM topic_episodes AS e
                JOIN topic_units AS u ON u.id = e.topic_unit_id
                ORDER BY e.updated_at DESC, e.title
                """
            ).fetchall()
        return conn.execute(
            """
            SELECT e.*, u.title AS topic_title
            FROM topic_episodes AS e
            JOIN topic_units AS u ON u.id = e.topic_unit_id
            WHERE e.topic_unit_id = ?
            ORDER BY e.updated_at DESC, e.title
            """,
            (topic_unit_id,),
        ).fetchall()
    finally:
        conn.close()


def list_active_topic_episodes(
    topic_unit_id: str | None = None,
) -> list[sqlite3.Row]:
    conn = get_connection()
    try:
        query = """
            SELECT e.*, u.title AS topic_title
            FROM topic_episodes AS e
            JOIN topic_units AS u ON u.id = e.topic_unit_id
            WHERE e.status = 'active'
        """
        params: tuple[str, ...] = ()
        if topic_unit_id is not None:
            query += " AND e.topic_unit_id = ?"
            params = (topic_unit_id,)
        query += " ORDER BY e.updated_at DESC, e.title"
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def _refresh_episode_thread(
    conn: sqlite3.Connection, episode_id: str, thread_id: str
) -> None:
    stats = conn.execute(
        """
        SELECT
            COUNT(*) AS message_count,
            (
                SELECT me2.message_id
                FROM message_episodes AS me2
                JOIN messages AS m2 ON m2.id = me2.message_id
                WHERE me2.episode_id = ? AND m2.thread_id = ?
                ORDER BY m2.created_at ASC, m2.id ASC LIMIT 1
            ) AS first_message_id,
            (
                SELECT me3.message_id
                FROM message_episodes AS me3
                JOIN messages AS m3 ON m3.id = me3.message_id
                WHERE me3.episode_id = ? AND m3.thread_id = ?
                ORDER BY m3.created_at DESC, m3.id DESC LIMIT 1
            ) AS last_message_id,
            MIN(m.created_at) AS first_activity_at,
            MAX(m.created_at) AS last_activity_at,
            MAX(me.relevance_score) AS relevance_score
        FROM message_episodes AS me
        JOIN messages AS m ON m.id = me.message_id
        WHERE me.episode_id = ? AND m.thread_id = ?
        """,
        (
            episode_id,
            thread_id,
            episode_id,
            thread_id,
            episode_id,
            thread_id,
        ),
    ).fetchone()
    if stats is None or stats["message_count"] == 0:
        conn.execute(
            "DELETE FROM episode_threads WHERE episode_id = ? AND thread_id = ?",
            (episode_id, thread_id),
        )
        return
    conn.execute(
        """
        UPDATE episode_threads
        SET message_count = ?, first_message_id = ?, last_message_id = ?,
            first_activity_at = ?, last_activity_at = ?, relevance_score = ?
        WHERE episode_id = ? AND thread_id = ?
        """,
        (
            stats["message_count"],
            stats["first_message_id"],
            stats["last_message_id"],
            stats["first_activity_at"],
            stats["last_activity_at"],
            _clamp_score(stats["relevance_score"]),
            episode_id,
            thread_id,
        ),
    )


def link_message_to_episode(
    message_id: str,
    episode_id: str,
    relevance_score: float,
    contribution_type: str = "context",
) -> None:
    if contribution_type not in CONTRIBUTION_TYPES:
        raise ValueError(f"Invalid contribution type: {contribution_type}")
    score = _clamp_score(relevance_score)
    conn = get_connection()
    try:
        conn.execute("BEGIN")
        message = conn.execute(
            "SELECT thread_id, created_at FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if message is None:
            raise ValueError(f"Unknown message: {message_id}")
        if conn.execute(
            "SELECT 1 FROM topic_episodes WHERE id = ?", (episode_id,)
        ).fetchone() is None:
            raise ValueError(f"Unknown Topic Episode: {episode_id}")

        conn.execute(
            """
            INSERT INTO message_episodes (
                message_id, episode_id, relevance_score,
                contribution_type, created_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(message_id, episode_id) DO UPDATE SET
                relevance_score = excluded.relevance_score,
                contribution_type = excluded.contribution_type
            """,
            (message_id, episode_id, score, contribution_type, now_iso()),
        )
        conn.execute(
            """
            INSERT INTO episode_threads (
                episode_id, thread_id, relevance_score, first_message_id,
                last_message_id, first_activity_at, last_activity_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(episode_id, thread_id) DO NOTHING
            """,
            (
                episode_id,
                message["thread_id"],
                score,
                message_id,
                message_id,
                message["created_at"],
                message["created_at"],
            ),
        )
        _refresh_episode_thread(conn, episode_id, str(message["thread_id"]))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def unlink_message_from_episode(message_id: str, episode_id: str) -> bool:
    conn = get_connection()
    try:
        conn.execute("BEGIN")
        message = conn.execute(
            "SELECT thread_id FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        cur = conn.execute(
            "DELETE FROM message_episodes WHERE message_id = ? AND episode_id = ?",
            (message_id, episode_id),
        )
        if message is not None:
            _refresh_episode_thread(conn, episode_id, str(message["thread_id"]))
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_message_episodes(message_id: str) -> list[sqlite3.Row]:
    conn = get_connection()
    try:
        return conn.execute(
            """
            SELECT me.*, e.title AS episode_title, e.topic_unit_id,
                   u.title AS topic_title
            FROM message_episodes AS me
            JOIN topic_episodes AS e ON e.id = me.episode_id
            JOIN topic_units AS u ON u.id = e.topic_unit_id
            WHERE me.message_id = ?
            ORDER BY me.relevance_score DESC
            """,
            (message_id,),
        ).fetchall()
    finally:
        conn.close()


def list_episode_messages(
    episode_id: str, limit: int | None = None
) -> list[sqlite3.Row]:
    query = """
        SELECT m.*, me.relevance_score, me.contribution_type, t.title AS thread_title
        FROM message_episodes AS me
        JOIN messages AS m ON m.id = me.message_id
        JOIN threads AS t ON t.id = m.thread_id
        WHERE me.episode_id = ?
        ORDER BY m.created_at DESC, m.id DESC
    """
    params: list[Any] = [episode_id]
    if limit is not None:
        query += " LIMIT ?"
        params.append(max(0, int(limit)))
    conn = get_connection()
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def upsert_episode_thread(
    episode_id: str,
    thread_id: str,
    local_summary: str | None = None,
    relevance_score: float = 1.0,
    status: str = "active",
    first_message_id: str | None = None,
    last_message_id: str | None = None,
    first_activity_at: str | None = None,
    last_activity_at: str | None = None,
    message_count: int | None = None,
) -> None:
    _validate_status(status, EPISODE_THREAD_STATUSES, "Episode Thread")
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO episode_threads (
                episode_id, thread_id, local_summary, message_count, relevance_score,
                first_message_id, last_message_id, first_activity_at,
                last_activity_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(episode_id, thread_id) DO UPDATE SET
                local_summary = COALESCE(excluded.local_summary, local_summary),
                message_count = CASE
                    WHEN ? THEN excluded.message_count ELSE message_count
                END,
                relevance_score = excluded.relevance_score,
                first_message_id = COALESCE(first_message_id, excluded.first_message_id),
                last_message_id = COALESCE(excluded.last_message_id, last_message_id),
                first_activity_at = COALESCE(first_activity_at, excluded.first_activity_at),
                last_activity_at = COALESCE(
                    excluded.last_activity_at, last_activity_at
                ),
                status = excluded.status
            """,
            (
                episode_id,
                thread_id,
                local_summary,
                max(0, int(message_count or 0)),
                _clamp_score(relevance_score),
                first_message_id,
                last_message_id,
                first_activity_at,
                last_activity_at,
                status,
                message_count is not None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_episode_thread(
    episode_id: str, thread_id: str
) -> sqlite3.Row | None:
    conn = get_connection()
    try:
        return conn.execute(
            """
            SELECT et.*, t.title AS thread_title
            FROM episode_threads AS et
            JOIN threads AS t ON t.id = et.thread_id
            WHERE et.episode_id = ? AND et.thread_id = ?
            """,
            (episode_id, thread_id),
        ).fetchone()
    finally:
        conn.close()


def list_episode_threads(episode_id: str) -> list[sqlite3.Row]:
    conn = get_connection()
    try:
        return conn.execute(
            """
            SELECT et.*, t.title AS thread_title
            FROM episode_threads AS et
            JOIN threads AS t ON t.id = et.thread_id
            WHERE et.episode_id = ?
            ORDER BY et.last_activity_at DESC
            """,
            (episode_id,),
        ).fetchall()
    finally:
        conn.close()


def list_thread_episodes(thread_id: str) -> list[sqlite3.Row]:
    conn = get_connection()
    try:
        return conn.execute(
            """
            SELECT et.*, e.title AS episode_title, e.summary AS episode_summary,
                   e.current_state_json, e.topic_unit_id,
                   u.title AS topic_title, u.global_summary
            FROM episode_threads AS et
            JOIN topic_episodes AS e ON e.id = et.episode_id
            JOIN topic_units AS u ON u.id = e.topic_unit_id
            WHERE et.thread_id = ?
            ORDER BY et.last_activity_at DESC
            """,
            (thread_id,),
        ).fetchall()
    finally:
        conn.close()


def upsert_semantic_embedding(
    entity_type: str,
    entity_id: str,
    embedding_model: str,
    vector: list[float],
    source_hash: str,
) -> None:
    if entity_type not in SEMANTIC_ENTITY_TYPES:
        raise ValueError(f"Invalid semantic entity type: {entity_type}")
    if not vector:
        raise ValueError("Embedding vector cannot be empty")
    timestamp = now_iso()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO semantic_embeddings (
                entity_type, entity_id, embedding_model, dimensions,
                vector_json, source_hash, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entity_type, entity_id, embedding_model) DO UPDATE SET
                dimensions = excluded.dimensions,
                vector_json = excluded.vector_json,
                source_hash = excluded.source_hash,
                updated_at = excluded.updated_at
            """,
            (
                entity_type,
                entity_id,
                embedding_model,
                len(vector),
                json.dumps([float(value) for value in vector]),
                source_hash,
                timestamp,
                timestamp,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_semantic_embedding(
    entity_type: str,
    entity_id: str,
    model: str = EMBEDDING_MODEL_NAME,
) -> dict[str, Any] | None:
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT * FROM semantic_embeddings
            WHERE entity_type = ? AND entity_id = ? AND embedding_model = ?
            """,
            (entity_type, entity_id, model),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    result = dict(row)
    result["vector"] = [
        float(value) for value in json.loads(result.pop("vector_json"))
    ]
    return result


def list_semantic_embeddings(entity_type: str) -> list[dict[str, Any]]:
    if entity_type not in SEMANTIC_ENTITY_TYPES:
        raise ValueError(f"Invalid semantic entity type: {entity_type}")
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM semantic_embeddings
            WHERE entity_type = ?
            ORDER BY updated_at DESC
            """,
            (entity_type,),
        ).fetchall()
    finally:
        conn.close()
    results: list[dict[str, Any]] = []
    for row in rows:
        result = dict(row)
        result["vector"] = [
            float(value) for value in json.loads(result.pop("vector_json"))
        ]
        results.append(result)
    return results


def delete_semantic_embedding(
    entity_type: str, entity_id: str, model: str | None = None
) -> bool:
    conn = get_connection()
    try:
        if model is None:
            cur = conn.execute(
                """
                DELETE FROM semantic_embeddings
                WHERE entity_type = ? AND entity_id = ?
                """,
                (entity_type, entity_id),
            )
        else:
            cur = conn.execute(
                """
                DELETE FROM semantic_embeddings
                WHERE entity_type = ? AND entity_id = ? AND embedding_model = ?
                """,
                (entity_type, entity_id, model),
            )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def embedding_is_stale(
    entity_type: str, entity_id: str, model: str, source_hash: str
) -> bool:
    embedding = get_semantic_embedding(entity_type, entity_id, model)
    return embedding is None or embedding["source_hash"] != source_hash
