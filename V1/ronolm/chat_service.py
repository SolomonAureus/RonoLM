from collections.abc import Callable, Mapping
from typing import Any

from ronolm.db import add_memory, add_message, list_messages
from ronolm.llm_maf import LLMError, generate_reply
from ronolm.memory_extractor import (
    MemoryExtractionError,
    extract_memories_from_user_message,
)
from ronolm.memory_judge import MemoryJudgementError, judge_memory_candidates
from ronolm.memory_retriever import retrieve_relevant_memories
from ronolm.thread_retriever import retrieve_topic_context
from ronolm.topic_manager import link_message_to_episode
from ronolm.topic_router import route_message
from ronolm.topic_summarizer import update_episodes_after_exchange


RECENT_MESSAGE_LIMIT = 12
StatusCallback = Callable[[str], None]
ResponseCallback = Callable[[str], None]


def _emit(callback: StatusCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def process_user_message(
    thread_id: str,
    user_text: str,
    input_mode: str = "text",
    metadata: Mapping[str, Any] | None = None,
    *,
    status_callback: StatusCallback | None = None,
    response_callback: ResponseCallback | None = None,
) -> str:
    """Process one canonical text turn through RonoLM's complete chat pipeline.

    ``input_mode`` and ``metadata`` are adapter context. The current messages
    schema has no metadata column, so they are intentionally not persisted.
    The canonical stored user content is always ``user_text``.
    """

    clean_text = user_text.strip()
    if not clean_text:
        raise ValueError("User message cannot be empty.")
    if not input_mode.strip():
        raise ValueError("input_mode cannot be empty.")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None.")

    user_message_id = add_message(thread_id, "user", clean_text)
    recent_messages = list_messages(thread_id)[-RECENT_MESSAGE_LIMIT:]
    relevant_memories = retrieve_relevant_memories(clean_text, limit=8)

    try:
        routing = route_message(
            clean_text,
            thread_id=thread_id,
            message_id=user_message_id,
            store=True,
        )
    except (ValueError, RuntimeError) as exc:
        _emit(status_callback, f"RonoLM topic router error: {exc}")
        routing = {"topic_episodes": []}

    topic_context = retrieve_topic_context(
        clean_text, current_thread_id=thread_id
    )
    _emit(
        status_callback,
        f"RonoLM: retrieved {len(relevant_memories)} relevant memory item(s).",
    )
    _emit(
        status_callback,
        f"RonoLM: retrieved {len(topic_context)} relevant Topic Unit(s).",
    )
    _emit(status_callback, "RonoLM: thinking...")

    reply_generated = True
    try:
        assistant_reply = generate_reply(
            recent_messages, relevant_memories, topic_context
        )
        if not assistant_reply.strip():
            raise LLMError("Ollama returned an empty response.")
    except LLMError as exc:
        assistant_reply = f"[LLM error] {exc}"
        reply_generated = False

    assistant_message_id = add_message(
        thread_id, "assistant", assistant_reply
    )
    if response_callback is not None:
        response_callback(assistant_reply)
    affected_episode_ids = [
        str(episode["id"]) for episode in routing["topic_episodes"]
    ]
    if reply_generated:
        for episode in routing["topic_episodes"]:
            link_message_to_episode(
                assistant_message_id,
                str(episode["id"]),
                float(episode["final_score"]),
                "answer",
            )
        update_episodes_after_exchange(
            affected_episode_ids,
            thread_id,
            clean_text,
            assistant_reply,
        )

    try:
        candidate_memories = extract_memories_from_user_message(clean_text)
    except MemoryExtractionError as exc:
        _emit(status_callback, f"RonoLM memory extractor error: {exc}")
        return assistant_reply

    try:
        approved_memories = judge_memory_candidates(
            clean_text, candidate_memories
        )
    except MemoryJudgementError as exc:
        _emit(status_callback, f"RonoLM memory judge error: {exc}")
        return assistant_reply

    for memory in approved_memories:
        add_memory(
            thread_id=thread_id,
            source_message_id=user_message_id,
            memory_type=memory["type"],
            subject=memory["subject"],
            content=memory["content"],
            lifetime=memory["lifetime"],
            importance=memory["importance"],
            confidence=memory["confidence"],
        )

    if candidate_memories:
        _emit(
            status_callback,
            "RonoLM: extracted "
            f"{len(candidate_memories)} candidate memory item(s), "
            f"approved {len(approved_memories)}.",
        )
    if approved_memories:
        _emit(
            status_callback,
            f"RonoLM: stored {len(approved_memories)} memory item(s).",
        )

    return assistant_reply
