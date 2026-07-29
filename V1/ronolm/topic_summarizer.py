import json
import logging
import re
from typing import Any

from ronolm import db
from ronolm.llm_maf import LLMError, call_ollama
from ronolm.topic_manager import (
    update_episode_thread_summary,
    update_topic_episode,
    update_topic_unit,
    validate_topic_state,
)


LOGGER = logging.getLogger(__name__)
MEANINGFUL_CHANGE_PATTERN = re.compile(
    r"\b(?:decid(?:e|ed|ing)|decision|next action|next step|todo|to-do|"
    r"implement|finish|begin|start|defer(?:red)?|deadline|blocked|complete(?:d)?|"
    r"switch(?:ed)?|instead|open question|step\s*\d+)\b",
    re.IGNORECASE,
)


def should_update_episode(
    user_message: str, *, linked_message_count: int = 0
) -> bool:
    return bool(MEANINGFUL_CHANGE_PATTERN.search(user_message)) or (
        linked_message_count > 0 and linked_message_count % 4 == 0
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Topic summarizer output must be a JSON object")
    return parsed


def _prompt(
    episode: dict[str, Any],
    thread: dict[str, Any],
    user_message: str,
    assistant_reply: str,
) -> str:
    return f"""Update a concise conversation-state snapshot for one Topic Episode.
Return only one valid JSON object with exactly these fields:
{{
  "episode_summary": "concise snapshot",
  "episode_state": {{
    "current_step": null,
    "decisions": [],
    "open_questions": [],
    "next_actions": [],
    "entities": [],
    "tags": []
  }},
  "thread_local_summary": "concise state in this physical thread",
  "meaningful_global_change": false,
  "topic_global_summary": null,
  "topic_global_state": {{
    "decisions": [],
    "open_questions": [],
    "next_actions": [],
    "entities": [],
    "tags": []
  }}
}}

Rules:
- Preserve still-current useful state from the existing snapshot.
- Keep summaries below 120 words; they are state snapshots, not transcripts.
- The user message is evidence of user facts and decisions.
- The assistant reply may summarize shared work state but is not independent
  evidence for a fact about the user.
- Set meaningful_global_change true only for a durable project-wide change.
- Do not invent IDs, people, decisions, deadlines, or actions.

Topic: {episode.get('topic_title')}
Episode: {episode.get('title')}
Existing episode summary: {episode.get('summary') or ''}
Existing episode state JSON: {episode.get('current_state_json') or '{}'}
Existing thread-local summary: {thread.get('local_summary') or ''}
USER_MESSAGE: {user_message}
ASSISTANT_REPLY: {assistant_reply}
"""


def update_episode_after_exchange(
    episode_id: str,
    thread_id: str,
    user_message: str,
    assistant_reply: str,
) -> bool:
    episode_row = db.get_topic_episode(episode_id)
    thread_row = db.get_episode_thread(episode_id, thread_id)
    if episode_row is None or thread_row is None:
        return False
    if not should_update_episode(
        user_message, linked_message_count=int(thread_row["message_count"])
    ):
        return False

    episode = dict(episode_row)
    thread = dict(thread_row)
    try:
        raw = call_ollama(
            [
                {
                    "role": "system",
                    "content": (
                        "You maintain RonoLM topic state. Follow the JSON schema "
                        "and grounding rules exactly."
                    ),
                },
                {
                    "role": "user",
                    "content": _prompt(
                        episode, thread, user_message, assistant_reply
                    ),
                },
            ]
        )
        update = _parse_json_object(raw)
        episode_summary = str(update["episode_summary"]).strip()
        local_summary = str(update["thread_local_summary"]).strip()
        episode_state = validate_topic_state(
            update["episode_state"], episode=True
        )
        meaningful_global = update["meaningful_global_change"]
        if not isinstance(meaningful_global, bool):
            raise ValueError("meaningful_global_change must be a boolean")
        if not episode_summary or not local_summary:
            raise ValueError("Summaries cannot be empty")

        update_topic_episode(
            episode_id,
            summary=episode_summary,
            current_state=episode_state,
        )
        update_episode_thread_summary(episode_id, thread_id, local_summary)
        if meaningful_global:
            global_summary_raw = update.get("topic_global_summary")
            global_summary = (
                str(global_summary_raw).strip()
                if global_summary_raw is not None
                else None
            )
            global_state = validate_topic_state(update["topic_global_state"])
            update_topic_unit(
                str(episode["topic_unit_id"]),
                global_summary=global_summary,
                global_state=global_state,
            )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, LLMError) as exc:
        LOGGER.warning("Skipping invalid Topic Episode update for %s: %s", episode_id, exc)
        return False
    return True


def update_episodes_after_exchange(
    episode_ids: list[str],
    thread_id: str,
    user_message: str,
    assistant_reply: str,
) -> int:
    updated = 0
    for episode_id in dict.fromkeys(episode_ids):
        if update_episode_after_exchange(
            episode_id, thread_id, user_message, assistant_reply
        ):
            updated += 1
    return updated
