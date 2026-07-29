import json
import re
import urllib.error
import urllib.request
from typing import Any

from ronolm.llm_maf import MODEL_NAME, OLLAMA_URL


VALID_TYPES = {
    "profile_fact",
    "relationship",
    "preference",
    "project",
    "goal",
    "event",
    "habit",
    "conversation_thread",
}

VALID_LIFETIMES = {
    "transient",
    "episodic",
    "long_term",
    "permanent",
}


MIN_EXTRACTOR_CONFIDENCE = 0.65


EXPLICIT_MEMORY_REQUEST_PATTERN = re.compile(
    r"\b(?:remember|do not forget|don't forget|keep in mind|note|save this)"
    r"(?:\s+that\b|\s*[:,])",
    flags=re.IGNORECASE,
)

RECALL_QUESTION_PATTERN = re.compile(
    r"^\s*(?:do|does|did)\s+you\s+remember\b",
    flags=re.IGNORECASE,
)

CONTEXT_DEPENDENT_START_PATTERN = re.compile(
    r"^\s*(?:it|that|this|they|he|she)\b",
    flags=re.IGNORECASE,
)

QUESTION_START_PATTERN = re.compile(
    r"^\s*(?:(?:what|who|when|where|why|how)\b|"
    r"(?:do|does|did|can|could|would|should|will|is|are|am|was|were|"
    r"have|has|had)\s+(?:i|you|we|my|your|our|the|this|that)\b|"
    r"tell\s+me\b|remind\s+me\b)",
    flags=re.IGNORECASE,
)

NON_FACT_CONTENT_PATTERNS = (
    re.compile(
        r"\b(?:is|are|was|were)\s+not\s+(?:explicitly\s+)?"
        r"(?:stated|provided|mentioned|known)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:cannot|can't|could not|couldn't)\s+be\s+determined\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:unknown|insufficient information|no information (?:was )?provided)\b",
        flags=re.IGNORECASE,
    ),
)


MEMORY_EXTRACTOR_SYSTEM_PROMPT = """You are RonoLM's memory candidate extraction module.

Your job is to propose candidate memories from the user's latest message.
You are not the final authority on whether a memory should be stored.

Rules:
- Output only valid JSON.
- Extract memories only from the USER_MESSAGE.
- Do not treat the assistant reply as factual evidence.
- Propose memories only for information explicitly stated by the user.
- A question asking for information is not evidence for the answer.
- Never create a memory that says information is unknown, missing, not provided, or not explicitly stated.
- Prefer stable facts, preferences, projects, goals, habits, important events, relationships, and ongoing conversation threads.
- Avoid proposing candidates for random small talk, one-off questions, or temporary emotions unless they affect ongoing work.
- Be especially cautious with sexual, romantic, medical, political, religious, or highly private information.
- If nothing useful can be proposed, return an empty memories list.

Examples:
- USER_MESSAGE: "My name is Mrinal Dhami"
  Extract: "User's name is Mrinal Dhami."
- USER_MESSAGE: "I love to eat ice cream"
  Extract: "User likes eating ice cream."
- USER_MESSAGE: "What is my name?"
  Return: {"memories": []}
- USER_MESSAGE: "What do I like?"
  Return: {"memories": []}

Memory types:
- profile_fact: name, college, work, role, identity-level project context
- relationship: friends, family, boss, teammates, professors
- preference: likes, dislikes, style preferences, constraints
- project: named projects the user is building or involved in
- goal: long-term or medium-term objectives
- event: dated or time-bounded events
- habit: repeated behavior or routine
- conversation_thread: unfinished work or topic to continue later

Lifetime classes:
- transient: useful for today or a few days
- episodic: useful for a later follow-up, then may expire
- long_term: useful for months or years
- permanent: core identity or extremely stable user information

Return JSON in this exact shape:
{
  "memories": [
    {
      "should_store": true,
      "type": "project",
      "subject": "RonoLM",
      "content": "User is building RonoLM, a memory-first conversational AI.",
      "lifetime": "long_term",
      "importance": 0.9,
      "confidence": 0.95
    }
  ]
}
"""


class MemoryExtractionError(RuntimeError):
    pass


def is_explicit_memory_request(user_message: str) -> bool:
    return EXPLICIT_MEMORY_REQUEST_PATTERN.search(user_message) is not None


def is_question_only_message(user_message: str) -> bool:
    text = user_message.strip()

    if not text:
        return False

    if RECALL_QUESTION_PATTERN.search(text):
        return True

    if is_explicit_memory_request(text):
        return False

    if not QUESTION_START_PATTERN.search(text):
        return False

    if "?" in text and text.split("?", 1)[1].strip():
        return False

    return True


def is_context_dependent_message(user_message: str) -> bool:
    text = user_message.strip()

    if not text or is_explicit_memory_request(text):
        return False

    return CONTEXT_DEPENDENT_START_PATTERN.search(text) is not None


def is_non_fact_memory_content(content: str) -> bool:
    return any(pattern.search(content) for pattern in NON_FACT_CONTENT_PATTERNS)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise MemoryExtractionError(f"No JSON object found in response: {text}")

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise MemoryExtractionError(f"Invalid JSON response: {text}") from exc

    if not isinstance(parsed, dict):
        raise MemoryExtractionError(f"Expected JSON object, got: {type(parsed)}")

    return parsed


def _call_ollama_json(messages: list[dict[str, str]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise MemoryExtractionError(
            "Could not connect to Ollama during memory extraction."
        ) from exc
    except TimeoutError as exc:
        raise MemoryExtractionError("Ollama memory extraction request timed out.") from exc

    try:
        content = response_data["message"]["content"].strip()
    except KeyError as exc:
        raise MemoryExtractionError(f"Unexpected Ollama response: {response_data}") from exc

    if not content:
        raise MemoryExtractionError(f"Ollama returned empty memory output: {response_data}")

    return _extract_json_object(content)


def _validate_memory(raw: dict[str, Any]) -> dict[str, Any] | None:
    if raw.get("should_store") is not True:
        return None

    memory_type = str(raw.get("type", "")).strip()
    lifetime = str(raw.get("lifetime", "")).strip()

    if memory_type not in VALID_TYPES:
        return None

    if lifetime not in VALID_LIFETIMES:
        return None

    subject = str(raw.get("subject", "")).strip()
    content = str(raw.get("content", "")).strip()

    if not subject or not content:
        return None

    if is_non_fact_memory_content(content):
        return None

    if (
        memory_type == "profile_fact"
        and "name" in subject.lower()
        and not re.search(r"\b(?:user's|the user's)\s+name\s+is\b", content, re.I)
    ):
        content = f"User's name is {content.rstrip('.')}."

    try:
        importance = float(raw.get("importance", 0.5))
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        return None

    if not 0.0 <= importance <= 1.0:
        return None

    if not 0.0 <= confidence <= 1.0:
        return None

    if confidence < MIN_EXTRACTOR_CONFIDENCE:
        return None

    return {
        "should_store": True,
        "type": memory_type,
        "subject": subject,
        "content": content,
        "lifetime": lifetime,
        "importance": importance,
        "confidence": confidence,
    }


def extract_memories_from_user_message(user_message: str) -> list[dict[str, Any]]:
    if is_question_only_message(user_message) or is_context_dependent_message(
        user_message
    ):
        return []

    messages = [
        {
            "role": "system",
            "content": MEMORY_EXTRACTOR_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"USER_MESSAGE:\n{user_message}",
        },
    ]

    parsed = _call_ollama_json(messages)

    raw_memories = parsed.get("memories", [])
    if not isinstance(raw_memories, list):
        raise MemoryExtractionError(f"Expected memories list, got: {parsed}")

    validated: list[dict[str, Any]] = []

    for raw in raw_memories:
        if not isinstance(raw, dict):
            continue

        memory = _validate_memory(raw)
        if memory is not None:
            validated.append(memory)

    return validated
