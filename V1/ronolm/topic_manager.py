import json
from collections.abc import Mapping
from typing import Any

from ronolm import db
from ronolm.embedding_client import (
    EMBEDDING_MODEL_NAME,
    EmbeddingError,
    build_episode_thread_embedding_text,
    build_topic_episode_embedding_text,
    build_topic_unit_embedding_text,
    embedding_source_hash,
    get_embedding,
)


TOPIC_STATE_KEYS = {
    "decisions",
    "open_questions",
    "next_actions",
    "entities",
    "tags",
}
EPISODE_STATE_KEYS = TOPIC_STATE_KEYS | {"current_step"}
MAX_TITLE_LENGTH = 120
MAX_SUMMARY_LENGTH = 1200
MAX_STATE_ITEMS = 20


def _clean_text(value: Any, *, maximum: int, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise ValueError("Required text is missing")
        return None
    text = " ".join(str(value).split()).strip()
    if not text and required:
        raise ValueError("Required text is empty")
    return text[:maximum] or None


def validate_topic_state(value: Any, *, episode: bool = False) -> dict[str, Any]:
    if value is None:
        raw: Mapping[str, Any] = {}
    elif isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, Mapping):
            raise ValueError("Structured state must be a JSON object")
        raw = parsed
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise ValueError("Structured state must be an object")

    allowed = EPISODE_STATE_KEYS if episode else TOPIC_STATE_KEYS
    result: dict[str, Any] = {}
    if episode:
        result["current_step"] = _clean_text(
            raw.get("current_step"), maximum=160
        )
    for key in sorted(allowed - {"current_step"}):
        items = raw.get(key, [])
        if not isinstance(items, list):
            raise ValueError(f"State field {key!r} must be a list")
        cleaned = []
        for item in items[:MAX_STATE_ITEMS]:
            text = _clean_text(item, maximum=240)
            if text and text not in cleaned:
                cleaned.append(text)
        result[key] = cleaned
    return result


def _ensure_embedding(entity_type: str, entity_id: str, text: str) -> bool:
    source_hash = embedding_source_hash(text)
    if not db.embedding_is_stale(
        entity_type, entity_id, EMBEDDING_MODEL_NAME, source_hash
    ):
        return False
    vector = get_embedding(text)
    db.upsert_semantic_embedding(
        entity_type,
        entity_id,
        EMBEDDING_MODEL_NAME,
        vector,
        source_hash,
    )
    return True


def ensure_topic_unit_embedding(topic_unit_id: str) -> bool:
    row = db.get_topic_unit(topic_unit_id)
    if row is None:
        raise ValueError(f"Unknown Topic Unit: {topic_unit_id}")
    return _ensure_embedding(
        "topic_unit", topic_unit_id, build_topic_unit_embedding_text(dict(row))
    )


def ensure_topic_episode_embedding(episode_id: str) -> bool:
    row = db.get_topic_episode(episode_id)
    if row is None:
        raise ValueError(f"Unknown Topic Episode: {episode_id}")
    return _ensure_embedding(
        "topic_episode", episode_id, build_topic_episode_embedding_text(dict(row))
    )


def episode_thread_embedding_id(episode_id: str, thread_id: str) -> str:
    return f"{episode_id}:{thread_id}"


def ensure_episode_thread_embedding(episode_id: str, thread_id: str) -> bool:
    row = db.get_episode_thread(episode_id, thread_id)
    episode = db.get_topic_episode(episode_id)
    if row is None or episode is None:
        raise ValueError(f"Unknown Episode Thread: {episode_id}/{thread_id}")
    content = dict(row)
    content["episode_title"] = episode["title"]
    content["topic_title"] = episode["topic_title"]
    return _ensure_embedding(
        "episode_thread",
        episode_thread_embedding_id(episode_id, thread_id),
        build_episode_thread_embedding_text(content),
    )


def _best_effort_embedding(callback: Any, *args: str) -> None:
    try:
        callback(*args)
    except (EmbeddingError, ValueError):
        # Topic state and routing remain usable with keyword scoring if the local
        # sentence embedding model is unavailable.
        pass


def create_topic_unit(
    title: str,
    description: str | None = None,
    global_summary: str | None = None,
    global_state: Any | None = None,
    status: str = "active",
    global_state_json: Any | None = None,
) -> str:
    clean_title = _clean_text(title, maximum=MAX_TITLE_LENGTH, required=True)
    assert clean_title is not None
    if global_state is not None and global_state_json is not None:
        raise ValueError("Provide global_state or global_state_json, not both")
    state = validate_topic_state(
        global_state if global_state_json is None else global_state_json
    )
    topic_id = db.create_topic_unit(
        clean_title,
        _clean_text(description, maximum=600),
        _clean_text(global_summary, maximum=MAX_SUMMARY_LENGTH),
        state,
        status,
    )
    _best_effort_embedding(ensure_topic_unit_embedding, topic_id)
    return topic_id


def update_topic_unit(topic_unit_id: str, **fields: Any) -> bool:
    if "global_state" in fields:
        fields["global_state"] = validate_topic_state(fields["global_state"])
    if "global_state_json" in fields:
        fields["global_state_json"] = validate_topic_state(
            fields["global_state_json"]
        )
    if "title" in fields:
        fields["title"] = _clean_text(
            fields["title"], maximum=MAX_TITLE_LENGTH, required=True
        )
    for name, maximum in (("description", 600), ("global_summary", MAX_SUMMARY_LENGTH)):
        if name in fields:
            fields[name] = _clean_text(fields[name], maximum=maximum)
    changed = db.update_topic_unit(topic_unit_id, **fields)
    if changed:
        _best_effort_embedding(ensure_topic_unit_embedding, topic_unit_id)
    return changed


def create_topic_episode(
    topic_unit_id: str,
    title: str,
    summary: str | None = None,
    current_state: Any | None = None,
    status: str = "active",
    current_state_json: Any | None = None,
) -> str:
    clean_title = _clean_text(title, maximum=MAX_TITLE_LENGTH, required=True)
    assert clean_title is not None
    if current_state is not None and current_state_json is not None:
        raise ValueError("Provide current_state or current_state_json, not both")
    state = validate_topic_state(
        current_state if current_state_json is None else current_state_json,
        episode=True,
    )
    episode_id = db.create_topic_episode(
        topic_unit_id,
        clean_title,
        _clean_text(summary, maximum=MAX_SUMMARY_LENGTH),
        state,
        status,
    )
    _best_effort_embedding(ensure_topic_episode_embedding, episode_id)
    return episode_id


def update_topic_episode(episode_id: str, **fields: Any) -> bool:
    if "current_state" in fields:
        fields["current_state"] = validate_topic_state(
            fields["current_state"], episode=True
        )
    if "current_state_json" in fields:
        fields["current_state_json"] = validate_topic_state(
            fields["current_state_json"], episode=True
        )
    if "title" in fields:
        fields["title"] = _clean_text(
            fields["title"], maximum=MAX_TITLE_LENGTH, required=True
        )
    if "summary" in fields:
        fields["summary"] = _clean_text(
            fields["summary"], maximum=MAX_SUMMARY_LENGTH
        )
    changed = db.update_topic_episode(episode_id, **fields)
    if changed:
        _best_effort_embedding(ensure_topic_episode_embedding, episode_id)
    return changed


def link_message_to_episode(
    message_id: str,
    episode_id: str,
    relevance_score: float,
    contribution_type: str = "context",
) -> None:
    db.link_message_to_episode(
        message_id, episode_id, relevance_score, contribution_type
    )
    message = db.get_message(message_id)
    if message is not None:
        _best_effort_embedding(
            ensure_episode_thread_embedding, episode_id, str(message["thread_id"])
        )


def update_episode_thread_summary(
    episode_id: str, thread_id: str, local_summary: str
) -> None:
    summary = _clean_text(local_summary, maximum=MAX_SUMMARY_LENGTH)
    current = db.get_episode_thread(episode_id, thread_id)
    if current is None:
        raise ValueError(f"Unknown Episode Thread: {episode_id}/{thread_id}")
    db.upsert_episode_thread(
        episode_id,
        thread_id,
        local_summary=summary,
        relevance_score=float(current["relevance_score"]),
        status=str(current["status"]),
    )
    _best_effort_embedding(ensure_episode_thread_embedding, episode_id, thread_id)
