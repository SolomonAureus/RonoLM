from dataclasses import dataclass
from datetime import datetime

@dataclass
class Message:
    id: str
    thread_id: str
    role: str
    content: str
    created_at: datetime

@dataclass
class Thread:
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass
class TopicUnit:
    id: str
    title: str
    description: str | None
    global_summary: str | None
    global_state_json: str | None
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass
class TopicEpisode:
    id: str
    topic_unit_id: str
    title: str
    summary: str | None
    current_state_json: str | None
    status: str
    started_at: datetime
    updated_at: datetime
    ended_at: datetime | None


@dataclass
class EpisodeThread:
    episode_id: str
    thread_id: str
    local_summary: str | None
    message_count: int
    relevance_score: float
    first_message_id: str | None
    last_message_id: str | None
    first_activity_at: datetime | None
    last_activity_at: datetime | None
    status: str


@dataclass
class MessageEpisode:
    message_id: str
    episode_id: str
    relevance_score: float
    contribution_type: str
    created_at: datetime


@dataclass
class SemanticEmbedding:
    entity_type: str
    entity_id: str
    embedding_model: str
    dimensions: int
    vector_json: str
    source_hash: str
    created_at: datetime
    updated_at: datetime
