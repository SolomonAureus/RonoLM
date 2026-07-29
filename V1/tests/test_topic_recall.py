import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ronolm.db as db
from ronolm.embedding_client import EmbeddingError
from ronolm.thread_retriever import (
    format_topic_context_for_prompt,
    retrieve_topic_context,
)
from ronolm.topic_manager import (
    create_topic_episode,
    create_topic_unit,
    link_message_to_episode,
    update_topic_episode,
    update_topic_unit,
)
from ronolm.topic_router import route_message


class TopicRecallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "ronolm.sqlite"
        self.patchers = [
            patch.object(db, "DB_PATH", self.database_path),
            patch(
                "ronolm.topic_manager.get_embedding",
                side_effect=EmbeddingError("embedding unavailable in test"),
            ),
            patch(
                "ronolm.topic_router.get_embedding",
                side_effect=EmbeddingError("embedding unavailable in test"),
            ),
            patch(
                "ronolm.thread_retriever.get_embedding",
                side_effect=EmbeddingError("embedding unavailable in test"),
            ),
        ]
        for patcher in self.patchers:
            patcher.start()
        db.init_db()

    def tearDown(self) -> None:
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temp_dir.cleanup()

    def _route_stored(self, thread_id: str, text: str) -> tuple[str, dict]:
        message_id = db.add_message(thread_id, "user", text)
        result = route_message(
            text, thread_id=thread_id, message_id=message_id, store=True
        )
        return message_id, result

    def test_schema_is_idempotent_and_foreign_keys_are_enabled(self) -> None:
        db.init_db()
        conn = db.get_connection()
        try:
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        finally:
            conn.close()
        self.assertTrue(
            {
                "topic_units",
                "topic_episodes",
                "episode_threads",
                "message_episodes",
                "semantic_embeddings",
            }.issubset(tables)
        )
        self.assertEqual(foreign_keys, 1)

    def test_one_topic_episode_spans_multiple_threads(self) -> None:
        thread_a = db.create_thread("Thread A")
        thread_b = db.create_thread("Thread B")
        message_a, first = self._route_stored(
            thread_a,
            "We are implementing hybrid memory retrieval for RonoLM.",
        )
        message_b, second = self._route_stored(
            thread_b, "Continue the embedding retrieval work."
        )

        topics = db.list_topic_units()
        episodes = db.list_topic_episodes(str(topics[0]["id"]))
        links = db.list_episode_threads(str(episodes[0]["id"]))
        self.assertEqual(len(topics), 1)
        self.assertEqual(len(episodes), 1)
        self.assertEqual({row["thread_id"] for row in links}, {thread_a, thread_b})
        self.assertEqual(len(db.list_message_episodes(message_a)), 1)
        self.assertEqual(len(db.list_message_episodes(message_b)), 1)
        self.assertTrue(first["topic_episodes"])
        self.assertTrue(second["topic_episodes"])

    def test_multiple_topics_inside_one_thread_are_not_merged(self) -> None:
        thread_id = db.create_thread("Mixed work")
        self._route_stored(thread_id, "Continue RonoLM thread recall.")
        self._route_stored(
            thread_id, "Also reduce the ML taskphase deadline."
        )

        titles = {row["title"] for row in db.list_topic_units()}
        thread_topics = {row["topic_title"] for row in db.list_thread_episodes(thread_id)}
        self.assertIn("RonoLM Development", titles)
        self.assertIn("Software/ML Taskphase", titles)
        self.assertFalse(any("and Taskphase" in title for title in titles))
        self.assertTrue(
            {"RonoLM Development", "Software/ML Taskphase"}.issubset(thread_topics)
        )

    def test_one_message_can_contribute_to_multiple_episodes(self) -> None:
        thread_id = db.create_thread("RonoLM work")
        message_id, _ = self._route_stored(
            thread_id,
            "Finish Step 8 embeddings, then start Step 9 thread recall.",
        )

        links = db.list_message_episodes(message_id)
        self.assertEqual(
            {row["episode_title"] for row in links},
            {"Hybrid Embedding Retrieval", "Cross-Thread Topic Recall"},
        )

    def test_multi_topic_message_does_not_cross_product_episodes(self) -> None:
        thread_id = db.create_thread("Multi-label")
        message_id, _ = self._route_stored(
            thread_id,
            "Step 9 for RonoLM should be thread recall, "
            "and reduce the ML taskphase deadline.",
        )

        links = {
            (row["topic_title"], row["episode_title"])
            for row in db.list_message_episodes(message_id)
        }
        self.assertEqual(
            links,
            {
                ("RonoLM Development", "Cross-Thread Topic Recall"),
                ("Software/ML Taskphase", "Deadline Planning"),
            },
        )

    def test_cross_thread_decision_is_retrieved(self) -> None:
        old_thread = db.create_thread("Old RonoLM thread")
        topic_id = create_topic_unit(
            "RonoLM Development",
            global_summary="Cloud support was deferred to v1.3.",
            global_state={"decisions": ["Defer cloud support to v1.3."]},
        )
        episode_id = create_topic_episode(
            topic_id,
            "Cross-Thread Topic Recall",
            summary="Local Step 9 is the current focus; cloud switching is deferred.",
            current_state={
                "current_step": "Step 9",
                "decisions": ["Cloud switching is deferred to v1.3."],
            },
        )
        old_message = db.add_message(
            old_thread,
            "user",
            "Cloud switching is deferred to v1.3. Current focus is local Step 9.",
        )
        link_message_to_episode(old_message, episode_id, 1.0, "decision")
        new_thread = db.create_thread("New thread")

        context = retrieve_topic_context(
            "What did we decide about cloud support?",
            current_thread_id=new_thread,
        )
        prompt = format_topic_context_for_prompt(context)
        self.assertIn("RonoLM Development", prompt)
        self.assertIn("deferred to v1.3", prompt)
        self.assertNotIn("No relevant historical", prompt)

    def test_continue_work_retrieves_current_step_and_next_action(self) -> None:
        old_thread = db.create_thread("Old work")
        topic_id = create_topic_unit("RonoLM Development")
        episode_id = create_topic_episode(
            topic_id,
            "Cross-Thread Topic Recall",
            summary="Step 9 cross-thread recall is in progress.",
            current_state={
                "current_step": "Step 9",
                "next_actions": ["Implement topic_manager.py."],
            },
        )
        old_message = db.add_message(
            old_thread, "user", "Next implement topic_manager.py."
        )
        link_message_to_episode(old_message, episode_id, 1.0, "next_action")
        new_thread = db.create_thread("Continuation")

        context = retrieve_topic_context(
            "Continue RonoLM.", current_thread_id=new_thread
        )
        prompt = format_topic_context_for_prompt(context)
        self.assertIn("Current step: Step 9", prompt)
        self.assertIn("Implement topic_manager.py.", prompt)

    def test_irrelevant_topics_are_suppressed(self) -> None:
        for title in (
            "RonoLM Development",
            "Software/ML Taskphase",
            "Goldman Sachs CV",
            "DELCON Paper",
        ):
            topic_id = create_topic_unit(title)
            create_topic_episode(topic_id, "General Work")
        thread_id = db.create_thread("Technical question")

        context = retrieve_topic_context(
            "Explain SQLite indexes.", current_thread_id=thread_id
        )
        self.assertEqual(context, [])

    def test_personal_preference_question_does_not_inherit_active_topic(
        self,
    ) -> None:
        thread_id = db.create_thread("Mixed conversation")
        topic_id = create_topic_unit("RonoLM Development")
        episode_id = create_topic_episode(topic_id, "General Work")
        message_id = db.add_message(
            thread_id, "user", "I am working on RonoLM."
        )
        link_message_to_episode(message_id, episode_id, 1.0, "context")

        result = route_message(
            "what do i like do u remember?",
            thread_id=thread_id,
            store=False,
        )

        self.assertEqual(result["topic_units"], [])
        self.assertEqual(result["topic_episodes"], [])
        self.assertEqual(result["create_topic_units"], [])
        self.assertEqual(result["create_topic_episodes"], [])

    def test_deictic_follow_up_still_uses_most_recent_active_topic(self) -> None:
        thread_id = db.create_thread("Follow-up")
        topic_id = create_topic_unit("RonoLM Development")
        episode_id = create_topic_episode(
            topic_id, "Cross-Thread Topic Recall"
        )
        message_id = db.add_message(
            thread_id, "user", "Step 9 should be thread recall."
        )
        link_message_to_episode(message_id, episode_id, 1.0, "decision")

        result = route_message(
            "How will that work?", thread_id=thread_id, store=False
        )

        self.assertEqual(
            [row["title"] for row in result["topic_units"]],
            ["RonoLM Development"],
        )
        self.assertEqual(
            [row["title"] for row in result["topic_episodes"]],
            ["Cross-Thread Topic Recall"],
        )

    def test_reprocessing_a_message_is_idempotent(self) -> None:
        thread_id = db.create_thread("Idempotency")
        message_id = db.add_message(
            thread_id, "user", "Continue RonoLM thread recall."
        )
        for _ in range(2):
            route_message(
                "Continue RonoLM thread recall.",
                thread_id=thread_id,
                message_id=message_id,
                store=True,
            )

        self.assertEqual(len(db.list_topic_units()), 1)
        self.assertEqual(len(db.list_topic_episodes()), 1)
        self.assertEqual(len(db.list_message_episodes(message_id)), 1)
        episode_id = str(db.list_topic_episodes()[0]["id"])
        episode_thread = db.get_episode_thread(episode_id, thread_id)
        self.assertIsNotNone(episode_thread)
        self.assertEqual(episode_thread["message_count"], 1)

    def test_validation_clamps_scores_and_rejects_invalid_values(self) -> None:
        thread_id = db.create_thread("Validation")
        topic_id = create_topic_unit("RonoLM Development")
        episode_id = create_topic_episode(topic_id, "Validation")
        message_id = db.add_message(thread_id, "user", "Validate this.")

        db.link_message_to_episode(message_id, episode_id, 5.0, "evidence")
        link = db.list_message_episodes(message_id)[0]
        self.assertEqual(link["relevance_score"], 1.0)
        with self.assertRaises(ValueError):
            db.link_message_to_episode(message_id, episode_id, 1.0, "invented")
        with self.assertRaises(ValueError):
            update_topic_unit(topic_id, status="invented")
        with self.assertRaises(ValueError):
            update_topic_episode(episode_id, current_state={"decisions": "no"})

    def test_semantic_embedding_staleness_uses_source_hash(self) -> None:
        db.upsert_semantic_embedding(
            "topic_unit", "topic_1", "test-model", [0.1, 0.2], "hash-a"
        )
        self.assertFalse(
            db.embedding_is_stale(
                "topic_unit", "topic_1", "test-model", "hash-a"
            )
        )
        self.assertTrue(
            db.embedding_is_stale(
                "topic_unit", "topic_1", "test-model", "hash-b"
            )
        )
        stored = db.get_semantic_embedding(
            "topic_unit", "topic_1", "test-model"
        )
        self.assertEqual(stored["dimensions"], 2)
        self.assertEqual(stored["vector"], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
