import math
import re
from datetime import datetime, timezone
from typing import Any

from ronolm import db
from ronolm.embedding_client import (
    EMBEDDING_MODEL_NAME,
    EmbeddingError,
    build_topic_episode_embedding_text,
    build_topic_unit_embedding_text,
    cosine_similarity,
    get_embedding,
)
from ronolm.topic_manager import (
    create_topic_episode,
    create_topic_unit,
    ensure_topic_episode_embedding,
    ensure_topic_unit_embedding,
    link_message_to_episode,
)


TOPIC_SEMANTIC_WEIGHT = 0.60
TOPIC_KEYWORD_WEIGHT = 0.20
TOPIC_RECENCY_WEIGHT = 0.10
TOPIC_ACTIVE_CONTEXT_WEIGHT = 0.10

EPISODE_SEMANTIC_WEIGHT = 0.55
EPISODE_KEYWORD_WEIGHT = 0.15
EPISODE_RECENCY_WEIGHT = 0.15
EPISODE_ACTIVE_STATUS_WEIGHT = 0.10
EPISODE_SAME_THREAD_WEIGHT = 0.05

TOPIC_MATCH_THRESHOLD = 0.43
EPISODE_MATCH_THRESHOLD = 0.40
RULE_ONLY_TOPIC_THRESHOLD = 0.15
RULE_ONLY_EPISODE_THRESHOLD = 0.25
FOLLOW_UP_TOPIC_THRESHOLD = 0.18
FOLLOW_UP_EPISODE_THRESHOLD = 0.20
MIN_SEMANTIC_ADMISSION_SCORE = 0.45
ACTIVE_CONTEXT_COHORT_SECONDS = 5.0
MAX_TOPIC_MATCHES = 4
MAX_EPISODE_MATCHES_PER_TOPIC = 4

STOP_WORDS = {
    "a",
    "about",
    "also",
    "an",
    "and",
    "are",
    "at",
    "be",
    "did",
    "do",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "the",
    "then",
    "this",
    "to",
    "was",
    "we",
    "what",
    "where",
    "will",
    "with",
    "work",
}

SMALL_TALK_PATTERN = re.compile(
    r"^\s*(?:hi|hello|hey|thanks|thank you|good morning|good evening|bye)[.! ]*$",
    re.IGNORECASE,
)
DEICTIC_FOLLOW_UP_PATTERN = re.compile(
    r"^\s*(?:"
    r"how\s+(?:will|would|does|did|can|could)\s+(?:that|this|it)\b|"
    r"why\s*(?:\?|$)|"
    r"why\s+(?:is|does|did|would)\s+(?:that|this|it)\b|"
    r"what(?:'s| is)\s+next\b|"
    r"what\s+about\s+(?:that|this|it)\b|"
    r"where\s+did\s+we\s+leave\s+off\b|"
    r"continue(?:\s+(?:that|this|it))?\s*[?.!]*$|"
    r"and\s+then\b|then\s+what\b"
    r")",
    re.IGNORECASE,
)

KNOWN_TOPIC_PATTERNS = (
    (
        re.compile(r"\bronolm\b|\bstep\s*(?:8|9)\b", re.IGNORECASE),
        "RonoLM Development",
    ),
    (
        re.compile(r"\b(?:ml|machine learning)\s+task\s*phase\b|\btaskphase\b", re.I),
        "Software/ML Taskphase",
    ),
    (
        re.compile(r"\b(?:goldman(?:\s+sachs)?\s+)?(?:cv|resume|résumé)\b", re.I),
        "Goldman Sachs CV",
    ),
    (re.compile(r"\bdelcon\b|\bdelcon\s+paper\b", re.IGNORECASE), "DELCON Paper"),
)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    }


def _keyword_score(query: str, candidate: str) -> float:
    query_tokens = _tokens(query)
    candidate_tokens = _tokens(candidate)
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = query_tokens & candidate_tokens
    if not overlap:
        return 0.0
    coverage = len(overlap) / len(candidate_tokens)
    query_coverage = len(overlap) / len(query_tokens)
    return min(1.0, coverage * 0.7 + query_coverage * 0.3)


def _recency_score(timestamp: Any) -> float:
    if not timestamp:
        return 0.0
    try:
        value = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.0
    days = max(0.0, (datetime.now(timezone.utc) - value).total_seconds() / 86400)
    return math.exp(-days / 45.0)


def _semantic_embedding(entity_type: str, entity_id: str) -> list[float] | None:
    stored = db.get_semantic_embedding(
        entity_type, entity_id, EMBEDDING_MODEL_NAME
    )
    if stored is not None:
        return stored["vector"]
    try:
        if entity_type == "topic_unit":
            ensure_topic_unit_embedding(entity_id)
        else:
            ensure_topic_episode_embedding(entity_id)
    except (EmbeddingError, ValueError):
        return None
    stored = db.get_semantic_embedding(
        entity_type, entity_id, EMBEDDING_MODEL_NAME
    )
    return None if stored is None else stored["vector"]


def _query_embedding(text: str) -> list[float] | None:
    try:
        return get_embedding(text)
    except (EmbeddingError, ValueError):
        return None


def _recent_active_rows(thread_id: str | None) -> list[dict[str, Any]]:
    if not thread_id:
        return []
    rows = [
        dict(row)
        for row in db.list_thread_episodes(thread_id)
        if row["status"] == "active" and row["last_activity_at"]
    ]
    if not rows:
        return []
    try:
        latest = max(
            datetime.fromisoformat(
                str(row["last_activity_at"]).replace("Z", "+00:00")
            )
            for row in rows
        )
    except ValueError:
        return rows[:1]
    recent: list[dict[str, Any]] = []
    for row in rows:
        try:
            activity = datetime.fromisoformat(
                str(row["last_activity_at"]).replace("Z", "+00:00")
            )
        except ValueError:
            continue
        if abs((latest - activity).total_seconds()) <= ACTIVE_CONTEXT_COHORT_SECONDS:
            recent.append(row)
    return recent


def _active_topic_ids(thread_id: str | None) -> set[str]:
    return {
        str(row["topic_unit_id"]) for row in _recent_active_rows(thread_id)
    }


def _active_episode_ids(thread_id: str | None) -> set[str]:
    return {str(row["episode_id"]) for row in _recent_active_rows(thread_id)}


def _has_admission_evidence(
    scores: dict[str, float],
    *,
    explicit_match: bool,
    contextual_follow_up: bool,
    context_score_name: str,
) -> bool:
    return (
        explicit_match
        or scores["keyword_score"] > 0.0
        or scores["semantic_score"] >= MIN_SEMANTIC_ADMISSION_SCORE
        or (
            contextual_follow_up
            and scores.get(context_score_name, 0.0) > 0.0
        )
    )


def _topic_candidate_text(row: dict[str, Any]) -> str:
    episode_titles = " ".join(
        str(episode["title"])
        for episode in db.list_topic_episodes(str(row["id"]))
    )
    return f"{build_topic_unit_embedding_text(row)}\nepisodes: {episode_titles}"


def _episode_candidate_text(row: dict[str, Any]) -> str:
    return build_topic_episode_embedding_text(row)


def _score_topic(
    row: dict[str, Any],
    query: str,
    query_vector: list[float] | None,
    active_topic_ids: set[str],
) -> dict[str, float]:
    candidate_vector = (
        _semantic_embedding("topic_unit", str(row["id"]))
        if query_vector is not None
        else None
    )
    semantic = (
        max(0.0, cosine_similarity(query_vector, candidate_vector))
        if query_vector is not None and candidate_vector is not None
        else 0.0
    )
    scores = {
        "semantic_score": semantic,
        "keyword_score": _keyword_score(query, _topic_candidate_text(row)),
        "recency_score": _recency_score(row.get("updated_at")),
        "active_context_score": 1.0 if row["id"] in active_topic_ids else 0.0,
    }
    scores["final_score"] = (
        scores["semantic_score"] * TOPIC_SEMANTIC_WEIGHT
        + scores["keyword_score"] * TOPIC_KEYWORD_WEIGHT
        + scores["recency_score"] * TOPIC_RECENCY_WEIGHT
        + scores["active_context_score"] * TOPIC_ACTIVE_CONTEXT_WEIGHT
    )
    return scores


def _score_episode(
    row: dict[str, Any],
    query: str,
    query_vector: list[float] | None,
    thread_episode_ids: set[str],
) -> dict[str, float]:
    candidate_vector = (
        _semantic_embedding("topic_episode", str(row["id"]))
        if query_vector is not None
        else None
    )
    semantic = (
        max(0.0, cosine_similarity(query_vector, candidate_vector))
        if query_vector is not None and candidate_vector is not None
        else 0.0
    )
    scores = {
        "semantic_score": semantic,
        "keyword_score": _keyword_score(query, _episode_candidate_text(row)),
        "recency_score": _recency_score(row.get("updated_at")),
        "active_status_score": 1.0 if row["status"] == "active" else 0.0,
        "same_thread_score": 1.0 if row["id"] in thread_episode_ids else 0.0,
    }
    scores["final_score"] = (
        scores["semantic_score"] * EPISODE_SEMANTIC_WEIGHT
        + scores["keyword_score"] * EPISODE_KEYWORD_WEIGHT
        + scores["recency_score"] * EPISODE_RECENCY_WEIGHT
        + scores["active_status_score"] * EPISODE_ACTIVE_STATUS_WEIGHT
        + scores["same_thread_score"] * EPISODE_SAME_THREAD_WEIGHT
    )
    return scores


def _inferred_topic_titles(text: str) -> list[str]:
    titles: list[str] = []
    for pattern, title in KNOWN_TOPIC_PATTERNS:
        if pattern.search(text) and title not in titles:
            titles.append(title)
    return titles


def _topic_specific_text(text: str, topic_title: str, topic_count: int) -> str:
    if topic_count <= 1:
        return text
    patterns = {
        "RonoLM Development": re.compile(
            r"\bronolm\b|\bstep\s*(?:8|9)\b|\bembedding\b|"
            r"\b(?:cross[- ]thread|thread recall|topic recall)\b",
            re.I,
        ),
        "Software/ML Taskphase": re.compile(
            r"\btaskphase\b|\b(?:ml|machine learning)\b|\bdeadline\b", re.I
        ),
        "Goldman Sachs CV": re.compile(
            r"\bgoldman\b|\b(?:cv|resume|résumé)\b", re.I
        ),
        "DELCON Paper": re.compile(r"\bdelcon\b|\bpaper\b", re.I),
    }
    pattern = patterns.get(topic_title)
    if pattern is None:
        return text
    segments = re.split(r"\s*(?:,|;|\bthen\b|\balso\b|\band\b)\s*", text, flags=re.I)
    selected = [segment for segment in segments if pattern.search(segment)]
    return " ".join(selected) if selected else text


def _fallback_topic_title(text: str) -> str | None:
    if text.rstrip().endswith("?") or not re.search(
        r"\b(?:build|building|implement|implementing|project|working on|"
        r"migration|paper|deadline|workstream)\b",
        text,
        re.I,
    ):
        return None
    match = re.search(
        r"\b(?:called|named|working on|project|implementing|building)\s+"
        r"([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,3})",
        text,
    )
    if match is None:
        return None
    title = " ".join(match.group(1).split())
    return title if len(title) >= 3 else None


def _inferred_episode_titles(text: str, topic_title: str) -> list[str]:
    titles: list[str] = []
    if re.search(r"\b(?:step\s*8|hybrid|embedding)\b", text, re.I):
        titles.append("Hybrid Embedding Retrieval")
    if re.search(
        r"\b(?:step\s*9|cross[- ]thread|thread recall|topic recall)\b", text, re.I
    ):
        titles.append("Cross-Thread Topic Recall")
    if re.search(r"\bdeadline\b", text, re.I):
        titles.append("Deadline Planning")
    if not titles:
        if "CV" in topic_title:
            titles.append("CV Work")
        elif "Paper" in topic_title:
            titles.append("Paper Work")
        else:
            titles.append("General Work")
    return titles


def _contribution_type(text: str) -> str:
    if re.search(r"\b(?:decide|decided|decision|should be|will use)\b", text, re.I):
        return "decision"
    if re.search(r"\b(?:next|then|todo|to-do|implement|finish|begin|start)\b", text, re.I):
        return "next_action"
    if text.rstrip().endswith("?"):
        return "question"
    if re.search(r"\b(?:correct|instead|not .* but)\b", text, re.I):
        return "correction"
    return "context"


def route_message(
    text: str,
    *,
    thread_id: str | None = None,
    message_id: str | None = None,
    store: bool = False,
) -> dict[str, Any]:
    query = text.strip()
    result: dict[str, Any] = {
        "topic_units": [],
        "topic_episodes": [],
        "create_topic_units": [],
        "create_topic_episodes": [],
    }
    if not query or SMALL_TALK_PATTERN.match(query):
        return result
    if store and message_id is None:
        raise ValueError("message_id is required when store=True")

    query_vector = _query_embedding(query)
    active_topic_ids = _active_topic_ids(thread_id)
    active_episode_ids = _active_episode_ids(thread_id)
    follow_up = bool(DEICTIC_FOLLOW_UP_PATTERN.match(query)) and bool(
        active_topic_ids
    )
    boosted_topic_ids = active_topic_ids if follow_up else set()
    boosted_episode_ids = active_episode_ids if follow_up else set()
    inferred_titles = _inferred_topic_titles(query)
    explicitly_inferred_topics = {title.casefold() for title in inferred_titles}
    topic_threshold = (
        FOLLOW_UP_TOPIC_THRESHOLD
        if follow_up
        else (
            TOPIC_MATCH_THRESHOLD
            if query_vector is not None
            else RULE_ONLY_TOPIC_THRESHOLD
        )
    )

    scored_topics: list[dict[str, Any]] = []
    for stored in db.list_topic_units():
        row = dict(stored)
        if row["status"] == "closed":
            continue
        scores = _score_topic(row, query, query_vector, boosted_topic_ids)
        admitted = _has_admission_evidence(
            scores,
            explicit_match=str(row["title"]).casefold()
            in explicitly_inferred_topics,
            contextual_follow_up=follow_up,
            context_score_name="active_context_score",
        )
        if admitted and scores["final_score"] >= topic_threshold:
            scored_topics.append({**row, **scores})
    scored_topics.sort(key=lambda item: item["final_score"], reverse=True)
    scored_topics = scored_topics[:MAX_TOPIC_MATCHES]

    if not scored_topics and not inferred_titles:
        fallback_title = _fallback_topic_title(query)
        if fallback_title is not None:
            inferred_titles.append(fallback_title)
    matched_normalized = {str(row["title"]).casefold() for row in scored_topics}
    for title in inferred_titles:
        existing = db.find_topic_unit_by_title(title)
        if existing is not None and title.casefold() not in matched_normalized:
            row = dict(existing)
            scores = _score_topic(row, query, query_vector, boosted_topic_ids)
            scored_topics.append({**row, **scores})
            matched_normalized.add(title.casefold())
        elif existing is None:
            result["create_topic_units"].append({"title": title})

    if store:
        for proposed in result["create_topic_units"]:
            topic_id = create_topic_unit(
                proposed["title"],
                description=f"Conversation topic for {proposed['title']}.",
            )
            row = dict(db.get_topic_unit(topic_id) or {})
            scores = _score_topic(row, query, query_vector, boosted_topic_ids)
            scored_topics.append({**row, **scores, "created": True})

    deduplicated_topics = {
        str(item["id"]): item for item in scored_topics if item.get("id")
    }
    scored_topics = sorted(
        deduplicated_topics.values(),
        key=lambda item: item["final_score"],
        reverse=True,
    )[:MAX_TOPIC_MATCHES]
    result["topic_units"] = scored_topics

    episode_threshold = (
        FOLLOW_UP_EPISODE_THRESHOLD
        if follow_up
        else (
            EPISODE_MATCH_THRESHOLD
            if query_vector is not None
            else RULE_ONLY_EPISODE_THRESHOLD
        )
    )
    for topic in scored_topics:
        topic_query = _topic_specific_text(
            query, str(topic["title"]), len(scored_topics)
        )
        episode_query_vector = (
            query_vector
            if topic_query == query
            else _query_embedding(topic_query)
        )
        episode_matches: list[dict[str, Any]] = []
        episodes = [
            dict(row) for row in db.list_topic_episodes(str(topic["id"]))
        ]
        inferred_episodes = _inferred_episode_titles(
            topic_query, str(topic["title"])
        )
        explicit_episode_titles = (
            set()
            if inferred_episodes == ["General Work"]
            else {title.casefold() for title in inferred_episodes}
        )
        for episode in episodes:
            if episode["status"] in {"completed", "abandoned"}:
                continue
            scores = _score_episode(
                episode, topic_query, episode_query_vector, boosted_episode_ids
            )
            admitted = _has_admission_evidence(
                scores,
                explicit_match=str(episode["title"]).casefold()
                in explicit_episode_titles,
                contextual_follow_up=follow_up,
                context_score_name="same_thread_score",
            )
            if admitted and scores["final_score"] >= episode_threshold:
                episode_matches.append({**episode, **scores})

        if inferred_episodes == ["General Work"] and episode_matches:
            inferred_episodes = []
        existing_titles = {str(row["title"]).casefold() for row in episodes}
        matched_titles = {str(row["title"]).casefold() for row in episode_matches}
        for title in inferred_episodes:
            if title.casefold() in matched_titles:
                continue
            existing = next(
                (
                    episode
                    for episode in episodes
                    if str(episode["title"]).casefold() == title.casefold()
                ),
                None,
            )
            if existing is not None:
                scores = _score_episode(
                    existing, topic_query, episode_query_vector, boosted_episode_ids
                )
                episode_matches.append({**existing, **scores})
            elif title.casefold() not in existing_titles:
                proposal = {
                    "topic_unit_id": topic["id"],
                    "topic_title": topic["title"],
                    "title": title,
                }
                result["create_topic_episodes"].append(proposal)
                if store:
                    episode_id = create_topic_episode(
                        str(topic["id"]),
                        title,
                        summary=f"Work concerning {title}.",
                    )
                    created = dict(db.get_topic_episode(episode_id) or {})
                    scores = _score_episode(
                        created,
                        topic_query,
                        episode_query_vector,
                        boosted_episode_ids,
                    )
                    episode_matches.append(
                        {**created, **scores, "created": True}
                    )

        episode_matches.sort(
            key=lambda item: item["final_score"], reverse=True
        )
        result["topic_episodes"].extend(
            episode_matches[:MAX_EPISODE_MATCHES_PER_TOPIC]
        )

    if store and message_id is not None:
        contribution = _contribution_type(query)
        for episode in result["topic_episodes"]:
            link_message_to_episode(
                message_id,
                str(episode["id"]),
                max(0.5, float(episode["final_score"])),
                contribution,
            )
    return result
