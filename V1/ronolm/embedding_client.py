import math
import json
import hashlib
import importlib.util
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(RuntimeError):
    pass


def embedding_dependency_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


@lru_cache(maxsize=1)
def _get_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(
            "sentence-transformers is required for semantic memory retrieval"
        ) from exc

    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as exc:
        raise EmbeddingError(f"Could not load embedding model: {exc}") from exc


def get_embedding(text: str) -> list[float]:
    text = text.strip()
    if not text:
        raise ValueError("Embedding text cannot be empty")

    try:
        vector = _get_model().encode(text, normalize_embeddings=True)
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Could not generate embedding: {exc}") from exc

    return [float(value) for value in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(value * value for value in a))
    magnitude_b = math.sqrt(sum(value * value for value in b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return max(-1.0, min(1.0, dot_product / (magnitude_a * magnitude_b)))


def build_memory_embedding_text(memory: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            f"type: {memory['type']}",
            f"subject: {memory['subject']}",
            f"content: {memory['content']}",
            f"lifetime: {memory['lifetime']}",
        )
    )


def _structured_state(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def _stable_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def build_topic_unit_embedding_text(topic: Mapping[str, Any]) -> str:
    state = _structured_state(
        topic.get("global_state", topic.get("global_state_json"))
    )
    return "\n".join(
        (
            f"topic title: {_stable_value(topic.get('title'))}",
            f"description: {_stable_value(topic.get('description'))}",
            f"global summary: {_stable_value(topic.get('global_summary'))}",
            f"decisions: {_stable_value(state.get('decisions'))}",
            f"entities: {_stable_value(state.get('entities'))}",
            f"tags: {_stable_value(state.get('tags'))}",
        )
    )


def build_topic_episode_embedding_text(episode: Mapping[str, Any]) -> str:
    state = _structured_state(
        episode.get("current_state", episode.get("current_state_json"))
    )
    return "\n".join(
        (
            f"topic: {_stable_value(episode.get('topic_title'))}",
            f"episode: {_stable_value(episode.get('title'))}",
            f"summary: {_stable_value(episode.get('summary'))}",
            f"current step: {_stable_value(state.get('current_step'))}",
            f"decisions: {_stable_value(state.get('decisions'))}",
            f"open questions: {_stable_value(state.get('open_questions'))}",
            f"next actions: {_stable_value(state.get('next_actions'))}",
            f"entities: {_stable_value(state.get('entities'))}",
            f"tags: {_stable_value(state.get('tags'))}",
        )
    )


def build_episode_thread_embedding_text(episode_thread: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            f"topic: {_stable_value(episode_thread.get('topic_title'))}",
            f"episode: {_stable_value(episode_thread.get('episode_title'))}",
            f"thread title: {_stable_value(episode_thread.get('thread_title'))}",
            f"local summary: {_stable_value(episode_thread.get('local_summary'))}",
            f"first activity: {_stable_value(episode_thread.get('first_activity_at'))}",
            f"last activity: {_stable_value(episode_thread.get('last_activity_at'))}",
        )
    )


def embedding_source_hash(canonical_text: str) -> str:
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
