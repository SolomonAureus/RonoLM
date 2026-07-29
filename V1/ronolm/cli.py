from ronolm.db import (
    get_or_create_thread,
    init_db,
    list_memories,
    list_messages,
)
from ronolm.chat_service import process_user_message
from ronolm.cli_debug import handle_debug_command
from ronolm.embedding_client import embedding_dependency_available
from ronolm.memory_retriever import retrieve_relevant_memories


def print_recent_context(thread_id: str, limit: int = 5) -> None:
    messages = list_messages(thread_id)

    if not messages:
        return

    print("\nRecent stored messages:")
    for msg in messages[-limit:]:
        role = "You" if msg["role"] == "user" else "RonoLM"
        print(f"{role}: {msg['content']}")
    print()


def main() -> None:
    init_db()
    thread_id = get_or_create_thread()

    print("RonoLM Step 9 local LLM chat")
    print("Type 'exit' to quit.")
    if not embedding_dependency_available():
        print(
            "RonoLM: warning: sentence-transformers is unavailable; "
            "using strict rule-only retrieval."
        )
    print_recent_context(thread_id)

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if user_input.startswith("/") and handle_debug_command(
            user_input, thread_id
        ):
            continue

        if user_input.lower() == "/memories":
            print_memories()
            continue

        if user_input.lower().startswith("/retrieve "):
            query = user_input[len("/retrieve "):].strip()

            if not query:
                print("RonoLM: Usage: /retrieve <query>")
                continue

            retrieved = retrieve_relevant_memories(query, limit=8)

            if not retrieved:
                print("RonoLM: No relevant memories retrieved.")
                continue

            print("\nRetrieved memories:")
            for memory in retrieved:
                print(
                    f"- [{memory['type']}] {memory['content']} "
                    f"final={memory['final_score']:.2f} "
                    f"embedding={memory['embedding_score']:.2f} "
                    f"keyword={memory['keyword_score']:.2f} "
                    f"type={memory['type_score']:.2f}"
                )
            print()
            continue

        process_user_message(
            thread_id,
            user_input,
            status_callback=lambda message: print(message, flush=True),
            response_callback=lambda response: print(
                f"RonoLM: {response}", flush=True
            ),
        )


def print_memories() -> None:
    memories = list_memories()

    if not memories:
        print("RonoLM: No memories stored yet.")
        return

    print("\nStored memories:")
    for memory in memories:
        print(
            f"- [{memory['type']}] {memory['subject']}: {memory['content']} "
            f"(lifetime={memory['lifetime']}, "
            f"importance={memory['importance']:.2f}, "
            f"confidence={memory['confidence']:.2f})"
        )
    print()


if __name__ == "__main__":
    main()
