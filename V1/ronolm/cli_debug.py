import json
from typing import Any

from ronolm import db
from ronolm.thread_retriever import retrieve_topic_context
from ronolm.topic_router import route_message


def _state(value: Any) -> str:
    if not value:
        return "{}"
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return str(value)
    return json.dumps(parsed, indent=2, sort_keys=True)


def _print_routing(result: dict[str, Any]) -> None:
    print("\nMatched Topic Units:")
    if not result["topic_units"]:
        print("- none")
    for topic in result["topic_units"]:
        print(
            f"- {topic['title']} ({topic['id']}): "
            f"final={topic['final_score']:.3f} "
            f"semantic={topic['semantic_score']:.3f} "
            f"keyword={topic['keyword_score']:.3f} "
            f"recency={topic['recency_score']:.3f} "
            f"active={topic['active_context_score']:.3f}"
        )
    print("Matched Topic Episodes:")
    if not result["topic_episodes"]:
        print("- none")
    for episode in result["topic_episodes"]:
        print(
            f"- {episode['topic_title']} / {episode['title']} ({episode['id']}): "
            f"final={episode['final_score']:.3f} "
            f"semantic={episode['semantic_score']:.3f} "
            f"keyword={episode['keyword_score']:.3f} "
            f"recency={episode['recency_score']:.3f} "
            f"active={episode['active_status_score']:.3f} "
            f"same_thread={episode['same_thread_score']:.3f}"
        )
    new_units = [item["title"] for item in result["create_topic_units"]]
    new_episodes = [
        f"{item['topic_title']} / {item['title']}"
        for item in result["create_topic_episodes"]
    ]
    print(f"Would create Topic Units: {new_units or 'none'}")
    print(f"Would create Topic Episodes: {new_episodes or 'none'}")
    print()


def _print_retrieval(context: list[dict[str, Any]]) -> None:
    if not context:
        print("RonoLM: No topic context would be injected.")
        return
    print("\nTopic context that would be injected:")
    for topic in context:
        print(
            f"- Topic Unit: {topic['title']} ({topic['id']}), "
            f"score={topic['score']:.3f}"
        )
        for episode in topic["episodes"]:
            print(
                f"  - Episode: {episode['title']} ({episode['id']}), "
                f"score={episode['score']:.3f}"
            )
            for thread in episode["episode_threads"]:
                print(
                    f"    - Thread: {thread['thread_title']} "
                    f"({thread['thread_id']}), score={thread['final_score']:.3f}, "
                    f"messages={thread['message_count']}"
                )
            for message in episode["supporting_messages"]:
                content = " ".join(str(message["content"]).split())[:160]
                print(
                    f"    - Support: {message['role']} "
                    f"({message['thread_title']}): {content}"
                )
    print()


def handle_debug_command(command: str, current_thread_id: str) -> bool:
    name, _, argument = command.partition(" ")
    argument = argument.strip()
    if name == "/topics":
        rows = db.list_topic_units()
        if not rows:
            print("RonoLM: No Topic Units stored.")
        for row in rows:
            print(
                f"- {row['title']} ({row['id']}), "
                f"status={row['status']}, updated={row['updated_at']}"
            )
        return True
    if name == "/episodes":
        rows = db.list_topic_episodes(argument or None)
        if not rows:
            print("RonoLM: No Topic Episodes stored.")
        for row in rows:
            print(
                f"- {row['topic_title']} / {row['title']} ({row['id']}), "
                f"status={row['status']}, updated={row['updated_at']}"
            )
        return True
    if name == "/topic":
        if not argument:
            print("RonoLM: Usage: /topic <topic_id>")
            return True
        row = db.get_topic_unit(argument)
        if row is None:
            print("RonoLM: Topic Unit not found.")
            return True
        print(
            f"\nTopic Unit: {row['title']} ({row['id']})\n"
            f"Status: {row['status']}\n"
            f"Description: {row['description'] or ''}\n"
            f"Global summary: {row['global_summary'] or ''}\n"
            f"Global state:\n{_state(row['global_state_json'])}\n"
        )
        return True
    if name == "/episode":
        if not argument:
            print("RonoLM: Usage: /episode <episode_id>")
            return True
        row = db.get_topic_episode(argument)
        if row is None:
            print("RonoLM: Topic Episode not found.")
            return True
        print(
            f"\nEpisode: {row['topic_title']} / {row['title']} ({row['id']})\n"
            f"Status: {row['status']}\n"
            f"Summary: {row['summary'] or ''}\n"
            f"Current state:\n{_state(row['current_state_json'])}\n"
            "Associated threads:"
        )
        threads = db.list_episode_threads(argument)
        for thread in threads:
            print(
                f"- {thread['thread_title']} ({thread['thread_id']}), "
                f"messages={thread['message_count']}, "
                f"status={thread['status']}, "
                f"last_activity={thread['last_activity_at']}"
            )
        print()
        return True
    if name == "/thread-topics":
        thread_id = argument or current_thread_id
        rows = db.list_thread_episodes(thread_id)
        if not rows:
            print("RonoLM: No Topic Episodes linked to this thread.")
        for row in rows:
            print(
                f"- {row['topic_title']} / {row['episode_title']} "
                f"({row['episode_id']}), status={row['status']}, "
                f"messages={row['message_count']}, "
                f"last_activity={row['last_activity_at']}"
            )
        return True
    if name == "/route":
        if not argument:
            print("RonoLM: Usage: /route <text>")
            return True
        _print_routing(
            route_message(argument, thread_id=current_thread_id, store=False)
        )
        return True
    if name == "/thread-retrieve":
        if not argument:
            print("RonoLM: Usage: /thread-retrieve <text>")
            return True
        _print_retrieval(
            retrieve_topic_context(argument, current_thread_id=current_thread_id)
        )
        return True
    return False
