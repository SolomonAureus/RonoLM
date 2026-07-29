import json
import re
import urllib.error
import urllib.request
from typing import Any

from ronolm.llm_maf import MODEL_NAME, OLLAMA_URL
from ronolm.memory_extractor import (
    MIN_EXTRACTOR_CONFIDENCE,
    VALID_LIFETIMES,
    VALID_TYPES,
    is_context_dependent_message,
    is_explicit_memory_request,
    is_non_fact_memory_content,
    is_question_only_message,
)


MIN_JUDGE_CONFIDENCE = 0.75
ALLOWED_DECISIONS = {"store", "skip"}

NAME_FACT_PATTERN = re.compile(
    r"\bmy\s+(?:full\s+)?name\s+is\s+(?P<value>[^.!?\n]+)",
    flags=re.IGNORECASE,
)

PREFERENCE_FACT_PATTERN = re.compile(
    r"\bi\s+(?:really\s+|usually\s+)?"
    r"(?P<verb>like|love|enjoy|prefer|dislike|hate)\s+(?P<value>[^.!?\n]+)",
    flags=re.IGNORECASE,
)

POSITIVE_PREFERENCE_PATTERN = re.compile(
    r"\b(?:like|likes|love|loves|enjoy|enjoys|prefer|prefers|favorite)\b",
    flags=re.IGNORECASE,
)

NEGATIVE_PREFERENCE_PATTERN = re.compile(
    r"\b(?:dislike|dislikes|hate|hates|does not like|doesn't like)\b",
    flags=re.IGNORECASE,
)

SENSITIVE_MEMORY_PATTERN = re.compile(
    r"\b(?:sexual|sex life|attracted\s+to|make\s+out|romantic|intimate|"
    r"pregnan(?:t|cy)|diagnos(?:is|ed)|medical|medication|"
    r"disease|disorder|therapy|therapist|politic(?:al|s)|election|vote|voting|"
    r"religion|religious|christian|hindu|muslim|jewish|buddhist|home address|"
    r"phone number|email address|bank account|credit card|social security)\b",
    flags=re.IGNORECASE,
)

GROUNDING_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "is",
    "it",
    "my",
    "of",
    "the",
    "to",
    "user",
}


MEMORY_JUDGE_SYSTEM_PROMPT = """You are RonoLM's memory judge.

Your job is to decide whether a proposed candidate memory should be stored.
Be precise: approve clear, useful user facts and reject unsupported or unsuitable claims.

Evaluate the CANDIDATE_MEMORY only against the original USER_MESSAGE.
Do not use assistant replies, assumptions, interpretations, or outside knowledge as evidence.

Store only if:
- The information is explicitly stated by the user.
- It is useful beyond the current conversation.
- It helps future personalization.
- It concerns stable profile facts, projects, goals, preferences, habits, relationships, important events, or unfinished work.
- It would make future conversations more continuous or personalized.

Important approval guidance:
- A clearly stated name is a stable profile fact and should normally be stored.
- An ordinary clearly stated like, dislike, or preference should normally be stored.
- Names and ordinary preferences do not require a "remember this" request.
- Do not reject a clear profile fact or preference as small talk merely because the message is short.

Skip if:
- It is small talk, greeting, thanks, or casual filler.
- It is just a one-off question.
- It is only relevant to the current moment.
- It is an assistant interpretation, refusal, or assumption.
- It is uncertain, inferred, exaggerated, or not directly stated.
- It is sensitive personal information and the user did not explicitly ask to remember it.
- It is sexual, romantic, medical, political, religious, or highly private information without an explicit remember-this style request.
- It would be creepy or unnecessary to bring up later.

If uncertain, skip.

Examples:
- USER_MESSAGE: "My name is Mrinal Dhami"
  CANDIDATE: "User's name is Mrinal Dhami."
  Decision: {"decision": "store", "reason": "stable_profile_fact", "confidence": 0.99}
- USER_MESSAGE: "I love to eat ice cream"
  CANDIDATE: "User likes eating ice cream."
  Decision: {"decision": "store", "reason": "explicit_preference", "confidence": 0.97}
- USER_MESSAGE: "What is my name?"
  CANDIDATE: "User's name is not explicitly stated."
  Decision: {"decision": "skip", "reason": "question_not_memory", "confidence": 0.99}
- If the candidate is not supported by USER_MESSAGE, use reason "not_a_user_fact".

Output only valid JSON in this exact shape:
{
  "decision": "store",
  "reason": "stable_profile_fact",
  "confidence": 0.96
}

Allowed decisions:
- store
- skip
"""


class MemoryJudgementError(RuntimeError):
    pass


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
        raise MemoryJudgementError(f"No JSON object found in response: {text}")

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise MemoryJudgementError(f"Invalid JSON response: {text}") from exc

    if not isinstance(parsed, dict):
        raise MemoryJudgementError(f"Expected JSON object, got: {type(parsed)}")

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
        raise MemoryJudgementError(
            "Could not connect to Ollama during memory judgement."
        ) from exc
    except TimeoutError as exc:
        raise MemoryJudgementError("Ollama memory judgement request timed out.") from exc

    try:
        content = response_data["message"]["content"].strip()
    except KeyError as exc:
        raise MemoryJudgementError(f"Unexpected Ollama response: {response_data}") from exc

    if not content:
        raise MemoryJudgementError(f"Ollama returned empty judgement output: {response_data}")

    return _extract_json_object(content)


def _is_valid_candidate(candidate: dict[str, Any]) -> bool:
    memory_type = str(candidate.get("type", "")).strip()
    lifetime = str(candidate.get("lifetime", "")).strip()
    subject = str(candidate.get("subject", "")).strip()
    content = str(candidate.get("content", "")).strip()

    if memory_type not in VALID_TYPES:
        return False

    if lifetime not in VALID_LIFETIMES:
        return False

    if not subject or not content:
        return False

    try:
        importance = float(candidate.get("importance"))
        confidence = float(candidate.get("confidence"))
    except (TypeError, ValueError):
        return False

    if not 0.0 <= importance <= 1.0:
        return False

    if not 0.0 <= confidence <= 1.0:
        return False

    return confidence >= MIN_EXTRACTOR_CONFIDENCE


def _normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": str(candidate["type"]).strip(),
        "subject": str(candidate["subject"]).strip(),
        "content": str(candidate["content"]).strip(),
        "lifetime": str(candidate["lifetime"]).strip(),
        "importance": float(candidate["importance"]),
        "confidence": float(candidate["confidence"]),
    }


def _grounding_tokens(text: str) -> set[str]:
    tokens: set[str] = set()

    for token in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        if token in GROUNDING_STOPWORDS:
            continue

        if token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("es") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]

        tokens.add(token)

    return tokens


def _candidate_contains_fact_value(
    value: str,
    candidate: dict[str, Any],
    minimum_overlap: float = 1.0,
) -> bool:
    value_tokens = _grounding_tokens(value)
    candidate_tokens = _grounding_tokens(str(candidate["content"]))

    overlap = len(value_tokens & candidate_tokens)
    required_overlap = max(1, int(len(value_tokens) * minimum_overlap + 0.999))

    return bool(value_tokens) and overlap >= required_overlap


def _clear_grounded_fact_judgement(
    user_message: str, candidate: dict[str, Any]
) -> dict[str, Any] | None:
    candidate_text = f"{candidate['subject']} {candidate['content']}"

    if SENSITIVE_MEMORY_PATTERN.search(f"{user_message} {candidate_text}"):
        return None

    name_match = NAME_FACT_PATTERN.search(user_message)
    subject = str(candidate["subject"]).strip().lower()

    if (
        candidate["type"] == "profile_fact"
        and name_match is not None
        and "name" in subject
        and _candidate_contains_fact_value(name_match.group("value"), candidate)
    ):
        return {
            "decision": "store",
            "reason": "stable_profile_fact",
            "confidence": 0.99,
        }

    preference_match = PREFERENCE_FACT_PATTERN.search(user_message)

    if candidate["type"] == "preference" and preference_match is not None:
        verb = preference_match.group("verb").lower()
        content = str(candidate["content"])
        expected_polarity = (
            NEGATIVE_PREFERENCE_PATTERN
            if verb in {"dislike", "hate"}
            else POSITIVE_PREFERENCE_PATTERN
        )

        if expected_polarity.search(content) and _candidate_contains_fact_value(
            preference_match.group("value"), candidate, minimum_overlap=0.6
        ):
            return {
                "decision": "store",
                "reason": "explicit_preference",
                "confidence": 0.97,
            }

    return None


def _validate_judgement(raw: dict[str, Any]) -> dict[str, Any]:
    decision = str(raw.get("decision", "")).strip().lower()
    reason = str(raw.get("reason", "")).strip()

    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError):
        raise MemoryJudgementError(f"Invalid judgement confidence: {raw}")

    if decision not in ALLOWED_DECISIONS:
        raise MemoryJudgementError(f"Invalid judgement decision: {raw}")

    if not reason:
        raise MemoryJudgementError(f"Missing judgement reason: {raw}")

    if not 0.0 <= confidence <= 1.0:
        raise MemoryJudgementError(f"Judgement confidence out of range: {raw}")

    return {
        "decision": decision,
        "reason": reason,
        "confidence": confidence,
    }


def judge_memory_candidate(user_message: str, candidate: dict[str, Any]) -> dict[str, Any]:
    if not _is_valid_candidate(candidate):
        return {
            "decision": "skip",
            "reason": "invalid_candidate",
            "confidence": 1.0,
        }

    if is_question_only_message(user_message):
        return {
            "decision": "skip",
            "reason": "question_not_memory",
            "confidence": 1.0,
        }

    if is_context_dependent_message(user_message):
        return {
            "decision": "skip",
            "reason": "context_dependent_statement",
            "confidence": 1.0,
        }

    if is_non_fact_memory_content(str(candidate["content"])):
        return {
            "decision": "skip",
            "reason": "not_a_user_fact",
            "confidence": 1.0,
        }

    candidate_text = f"{candidate['subject']} {candidate['content']}"
    if (
        SENSITIVE_MEMORY_PATTERN.search(f"{user_message} {candidate_text}")
        and not is_explicit_memory_request(user_message)
    ):
        return {
            "decision": "skip",
            "reason": "sensitive_without_explicit_request",
            "confidence": 1.0,
        }

    clear_fact_judgement = _clear_grounded_fact_judgement(user_message, candidate)
    if clear_fact_judgement is not None:
        return clear_fact_judgement

    messages = [
        {
            "role": "system",
            "content": MEMORY_JUDGE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "USER_MESSAGE:\n"
                f"{user_message}\n\n"
                "CANDIDATE_MEMORY:\n"
                f"{json.dumps(_normalize_candidate(candidate), ensure_ascii=True)}"
            ),
        },
    ]

    parsed = _call_ollama_json(messages)
    return _validate_judgement(parsed)


def judge_memory_candidates(
    user_message: str, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    approved: list[dict[str, Any]] = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        judgement = judge_memory_candidate(user_message, candidate)


        if (
            judgement["decision"] == "store"
            and judgement["confidence"] >= MIN_JUDGE_CONFIDENCE
        ):
            approved.append(_normalize_candidate(candidate))

    return approved
