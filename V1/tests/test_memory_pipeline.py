import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ronolm.db as db
import ronolm.embedding_client as embedding_client
import ronolm.memory_retriever as memory_retriever
from ronolm.memory_extractor import (
    _validate_memory,
    extract_memories_from_user_message,
    is_question_only_message,
)
from ronolm.memory_judge import judge_memory_candidate, judge_memory_candidates
from ronolm.memory_retriever import (
    format_memories_for_prompt,
    retrieve_relevant_memories,
)


EMBEDDING_MODEL_NAME = embedding_client.EMBEDDING_MODEL_NAME


NAME_CANDIDATE = {
    "should_store": True,
    "type": "profile_fact",
    "subject": "name",
    "content": "User's name is Mrinal Dhami.",
    "lifetime": "permanent",
    "importance": 0.9,
    "confidence": 0.95,
}

BARE_NAME_CANDIDATE = {
    **NAME_CANDIDATE,
    "subject": "User's name",
    "content": "Mrinal Dhami",
    "importance": 1.0,
    "confidence": 1.0,
}

PREFERENCE_CANDIDATE = {
    "should_store": True,
    "type": "preference",
    "subject": "ice cream",
    "content": "User likes ice cream.",
    "lifetime": "long_term",
    "importance": 0.75,
    "confidence": 0.93,
}


class MemoryExtractorTests(unittest.TestCase):
    def test_question_only_message_does_not_call_model(self) -> None:
        with patch(
            "ronolm.memory_extractor._call_ollama_json",
            side_effect=AssertionError("question should not reach the extractor model"),
        ):
            self.assertEqual(extract_memories_from_user_message("What is my name?"), [])

    def test_typo_question_is_still_question_only(self) -> None:
        self.assertTrue(is_question_only_message("what is my na,e?"))

    def test_explicit_memory_request_is_not_filtered_as_question(self) -> None:
        self.assertFalse(is_question_only_message("Can you remember that I like tea?"))

    def test_recall_question_is_filtered(self) -> None:
        self.assertTrue(is_question_only_message("Do you remember my name?"))

    def test_question_followed_by_a_fact_is_not_filtered_wholesale(self) -> None:
        self.assertFalse(
            is_question_only_message("What should I order? I usually prefer vegetarian food.")
        )

    def test_valid_name_candidate_survives_extractor_validation(self) -> None:
        with patch(
            "ronolm.memory_extractor._call_ollama_json",
            return_value={"memories": [BARE_NAME_CANDIDATE]},
        ):
            memories = extract_memories_from_user_message("My name is Mrinal Dhami")

        self.assertEqual(memories[0]["content"], "User's name is Mrinal Dhami.")

    def test_context_dependent_follow_up_does_not_create_memory(self) -> None:
        with patch(
            "ronolm.memory_extractor._call_ollama_json",
            side_effect=AssertionError(
                "context-dependent follow-up should not reach the extractor model"
            ),
        ):
            memories = extract_memories_from_user_message("it would be consensual")

        self.assertEqual(memories, [])

    def test_absence_statement_is_not_a_valid_memory(self) -> None:
        candidate = {
            **NAME_CANDIDATE,
            "content": "User's name is not explicitly stated.",
            "lifetime": "transient",
        }

        self.assertIsNone(_validate_memory(candidate))

    def test_should_store_must_be_a_json_boolean(self) -> None:
        self.assertIsNone(_validate_memory({**NAME_CANDIDATE, "should_store": "false"}))


class MemoryJudgeTests(unittest.TestCase):
    def test_question_candidate_gets_accurate_reason_without_model(self) -> None:
        with patch(
            "ronolm.memory_judge._call_ollama_json",
            side_effect=AssertionError("question should not reach the judge model"),
        ):
            judgement = judge_memory_candidate("What is my name?", NAME_CANDIDATE)

        self.assertEqual(judgement["decision"], "skip")
        self.assertEqual(judgement["reason"], "question_not_memory")

    def test_absence_candidate_gets_not_a_user_fact_reason(self) -> None:
        candidate = {
            **NAME_CANDIDATE,
            "content": "User's name is not explicitly stated.",
        }

        with patch(
            "ronolm.memory_judge._call_ollama_json",
            side_effect=AssertionError("non-fact should not reach the judge model"),
        ):
            judgement = judge_memory_candidate("My name is Mrinal Dhami", candidate)

        self.assertEqual(judgement["decision"], "skip")
        self.assertEqual(judgement["reason"], "not_a_user_fact")

    def test_grounded_name_is_approved_without_model_judge(self) -> None:
        with patch(
            "ronolm.memory_judge._call_ollama_json",
            side_effect=AssertionError("clear name fact should not need the model judge"),
        ):
            approved = judge_memory_candidates("My name is Mrinal Dhami", [NAME_CANDIDATE])

        self.assertEqual(len(approved), 1)
        self.assertEqual(approved[0]["content"], "User's name is Mrinal Dhami.")

    def test_grounded_preference_is_approved_without_model_judge(self) -> None:
        with patch(
            "ronolm.memory_judge._call_ollama_json",
            side_effect=AssertionError("clear preference should not need the model judge"),
        ):
            judgement = judge_memory_candidate(
                "I love to eat ice cream", PREFERENCE_CANDIDATE
            )

        self.assertEqual(judgement["decision"], "store")
        self.assertEqual(judgement["reason"], "explicit_preference")

    def test_grounded_preference_approval_does_not_depend_on_lifetime_label(self) -> None:
        candidate = {**PREFERENCE_CANDIDATE, "lifetime": "transient"}

        with patch(
            "ronolm.memory_judge._call_ollama_json",
            side_effect=AssertionError("clear preference should not need the model judge"),
        ):
            judgement = judge_memory_candidate("I love ice cream", candidate)

        self.assertEqual(judgement["decision"], "store")

    def test_preference_with_wrong_value_is_not_automatically_approved(self) -> None:
        wrong_candidate = {
            **PREFERENCE_CANDIDATE,
            "subject": "ice cream",
            "content": "User likes drinking coffee.",
        }
        model_judgement = {
            "decision": "skip",
            "reason": "not_a_user_fact",
            "confidence": 0.99,
        }

        with patch(
            "ronolm.memory_judge._call_ollama_json",
            return_value=model_judgement,
        ) as model_call:
            judgement = judge_memory_candidate(
                "I love to eat ice cream", wrong_candidate
            )

        model_call.assert_called_once()
        self.assertEqual(judgement["decision"], "skip")

    def test_sensitive_attraction_candidate_gets_accurate_skip_reason(self) -> None:
        candidate = {
            "should_store": True,
            "type": "relationship",
            "subject": "Latina women",
            "content": "User is attracted to Latina women.",
            "lifetime": "transient",
            "importance": 0.8,
            "confidence": 0.9,
        }

        with patch(
            "ronolm.memory_judge._call_ollama_json",
            side_effect=AssertionError("sensitive candidate should be handled by policy"),
        ):
            judgement = judge_memory_candidate(
                "I also like Latina women", candidate
            )

        self.assertEqual(judgement["decision"], "skip")
        self.assertEqual(judgement["reason"], "sensitive_without_explicit_request")


class MemoryRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.legacy_name_row = {
            "id": "mem_name",
            "type": "profile_fact",
            "subject": "User's name",
            "content": "Mrinal Dhami",
            "lifetime": "permanent",
            "importance": 1.0,
            "confidence": 1.0,
            "created_at": "2026-07-14T00:00:00+00:00",
        }

    def test_quality_alone_does_not_make_memory_relevant(self) -> None:
        with patch(
            "ronolm.memory_retriever.list_memories_with_embeddings",
            return_value=[self.legacy_name_row],
        ):
            retrieved = retrieve_relevant_memories("why?")

        self.assertEqual(retrieved, [])

    def test_name_query_retrieves_legacy_bare_content_memory(self) -> None:
        with patch(
            "ronolm.memory_retriever.list_memories_with_embeddings",
            return_value=[self.legacy_name_row],
        ):
            retrieved = retrieve_relevant_memories("what is my name?")

        self.assertEqual(len(retrieved), 1)

    def test_prompt_includes_subject_and_content(self) -> None:
        prompt = format_memories_for_prompt([self.legacy_name_row])

        self.assertIn("subject=User's name", prompt)
        self.assertIn("content=User's name is Mrinal Dhami.", prompt)

    def test_semantic_project_query_retrieves_project_memory(self) -> None:
        project = {
            "id": "mem_project",
            "type": "project",
            "subject": "RonoLM",
            "content": "User is building RonoLM, a memory-first conversational AI.",
            "lifetime": "long_term",
            "importance": 0.9,
            "confidence": 0.95,
            "created_at": "2026-07-14T00:00:00+00:00",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding": [1.0, 0.0],
        }
        unrelated = {
            **self.legacy_name_row,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding": [0.0, 1.0],
        }

        with (
            patch(
                "ronolm.memory_retriever.list_memories_with_embeddings",
                return_value=[unrelated, project],
            ),
            patch("ronolm.memory_retriever.get_embedding", return_value=[1.0, 0.0]),
        ):
            retrieved = retrieve_relevant_memories(
                "What personal AI project am I working on?"
            )

        self.assertEqual([memory["id"] for memory in retrieved], ["mem_project"])
        self.assertGreater(retrieved[0]["embedding_score"], 0.9)

    def test_semantic_preference_filters_an_unrelated_preference(self) -> None:
        ice_cream = {
            "id": "mem_ice_cream",
            "type": "preference",
            "subject": "ice cream",
            "content": "User likes ice cream.",
            "lifetime": "long_term",
            "importance": 0.75,
            "confidence": 0.93,
            "created_at": "2026-07-14T00:00:00+00:00",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding": [1.0, 0.0],
        }
        mushrooms = {
            **ice_cream,
            "id": "mem_mushrooms",
            "subject": "mushrooms",
            "content": "User dislikes mushrooms.",
            "embedding": [0.0, 1.0],
        }

        with (
            patch(
                "ronolm.memory_retriever.list_memories_with_embeddings",
                return_value=[mushrooms, ice_cream],
            ),
            patch("ronolm.memory_retriever.get_embedding", return_value=[1.0, 0.0]),
        ):
            retrieved = retrieve_relevant_memories("Suggest a dessert I might enjoy")

        self.assertEqual([memory["id"] for memory in retrieved], ["mem_ice_cream"])

    def test_dinner_query_retrieves_negative_food_preference(self) -> None:
        base_memory = {
            "id": "mem_ice_cream",
            "type": "preference",
            "subject": "ice cream",
            "content": "User likes ice cream.",
            "lifetime": "long_term",
            "importance": 0.75,
            "confidence": 0.93,
            "created_at": "2026-07-14T00:00:00+00:00",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding": [1.0, 0.0],
        }
        mushrooms = {
            **base_memory,
            "id": "mem_mushrooms",
            "subject": "mushrooms",
            "content": "User dislikes mushrooms.",
            "embedding": [0.0, 1.0],
        }

        with (
            patch(
                "ronolm.memory_retriever.list_memories_with_embeddings",
                return_value=[base_memory, mushrooms],
            ),
            patch("ronolm.memory_retriever.get_embedding", return_value=[0.0, 1.0]),
        ):
            retrieved = retrieve_relevant_memories("Suggest dinner")

        self.assertEqual([memory["id"] for memory in retrieved], ["mem_mushrooms"])

    def test_unrelated_query_suppresses_personal_memories_with_embeddings(self) -> None:
        memory = {
            **self.legacy_name_row,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding": [0.0, 1.0],
        }

        with (
            patch(
                "ronolm.memory_retriever.list_memories_with_embeddings",
                return_value=[memory],
            ),
            patch("ronolm.memory_retriever.get_embedding", return_value=[1.0, 0.0]),
        ):
            retrieved = retrieve_relevant_memories("Explain SQLite")

        self.assertEqual(retrieved, [])

    def test_colloquial_project_recall_works_without_embeddings(self) -> None:
        project = {
            "id": "mem_project",
            "type": "project",
            "subject": "RonoLM",
            "content": "User is building RonoLM, a memory-first conversational AI.",
            "lifetime": "long_term",
            "importance": 0.9,
            "confidence": 0.95,
            "created_at": "2026-07-14T00:00:00+00:00",
        }
        with (
            patch(
                "ronolm.memory_retriever.list_memories_with_embeddings",
                return_value=[project],
            ),
            patch(
                "ronolm.memory_retriever.get_embedding",
                side_effect=embedding_client.EmbeddingError("unavailable"),
            ),
        ):
            retrieved = retrieve_relevant_memories(
                "Do u remember any project i am working on?"
            )

        self.assertEqual([memory["id"] for memory in retrieved], ["mem_project"])
        self.assertEqual(retrieved[0]["type_score"], 1.0)

    def test_generic_preference_recall_finds_transient_like_without_embeddings(
        self,
    ) -> None:
        preference = {
            "id": "mem_ice_cream",
            "type": "preference",
            "subject": "Ice cream",
            "content": "User likes eating ice cream.",
            "lifetime": "transient",
            "importance": 0.8,
            "confidence": 0.9,
            "created_at": "2026-07-27T00:00:00+00:00",
        }
        with (
            patch(
                "ronolm.memory_retriever.list_memories_with_embeddings",
                return_value=[preference],
            ),
            patch(
                "ronolm.memory_retriever.get_embedding",
                side_effect=embedding_client.EmbeddingError("unavailable"),
            ),
        ):
            retrieved = retrieve_relevant_memories(
                "what do i like do u remember?"
            )

        self.assertEqual([memory["id"] for memory in retrieved], ["mem_ice_cream"])
        self.assertGreater(retrieved[0]["keyword_score"], 0.0)
        self.assertGreaterEqual(
            retrieved[0]["final_score"],
            memory_retriever.RULE_ONLY_MIN_RETRIEVAL_SCORE,
        )

    def test_typed_recall_does_not_leak_other_memory_types(self) -> None:
        project = {
            "id": "mem_project",
            "type": "project",
            "subject": "RonoLM",
            "content": "User is building RonoLM.",
            "lifetime": "long_term",
            "importance": 0.9,
            "confidence": 0.95,
            "created_at": "2026-07-14T00:00:00+00:00",
        }
        with (
            patch(
                "ronolm.memory_retriever.list_memories_with_embeddings",
                return_value=[project],
            ),
            patch(
                "ronolm.memory_retriever.get_embedding",
                side_effect=embedding_client.EmbeddingError("unavailable"),
            ),
        ):
            retrieved = retrieve_relevant_memories("what do i like?")

        self.assertEqual(retrieved, [])

    def test_identical_memories_are_not_duplicated_in_prompt_retrieval(
        self,
    ) -> None:
        first = {
            "id": "mem_old",
            "type": "preference",
            "subject": "ice cream",
            "content": "User likes eating ice cream.",
            "lifetime": "transient",
            "importance": 0.8,
            "confidence": 0.9,
            "created_at": "2026-07-14T00:00:00+00:00",
        }
        second = {
            **first,
            "id": "mem_new",
            "subject": "Ice cream",
            "created_at": "2026-07-27T00:00:00+00:00",
        }
        with (
            patch(
                "ronolm.memory_retriever.list_memories_with_embeddings",
                return_value=[first, second],
            ),
            patch(
                "ronolm.memory_retriever.get_embedding",
                side_effect=embedding_client.EmbeddingError("unavailable"),
            ),
        ):
            retrieved = retrieve_relevant_memories("what do i like?")

        self.assertEqual([memory["id"] for memory in retrieved], ["mem_new"])


class EmbeddingClientTests(unittest.TestCase):
    def test_embedding_text_contains_retrieval_metadata(self) -> None:
        text = embedding_client.build_memory_embedding_text(PREFERENCE_CANDIDATE)

        self.assertIn("type: preference", text)
        self.assertIn("subject: ice cream", text)
        self.assertIn("content: User likes ice cream.", text)
        self.assertIn("lifetime: long_term", text)

    def test_cosine_similarity(self) -> None:
        self.assertAlmostEqual(
            embedding_client.cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0
        )
        self.assertAlmostEqual(
            embedding_client.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0
        )


class MemoryPipelineIntegrationTests(unittest.TestCase):
    def test_default_database_path_is_independent_of_working_directory(self) -> None:
        self.assertTrue(db.DB_PATH.is_absolute())
        self.assertEqual(db.DB_PATH.name, "ronolm.sqlite")
        self.assertEqual(db.DB_PATH.parent.name, "data")

    def test_name_memory_is_stored_and_retrieved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "ronolm.sqlite"

            with (
                patch.object(db, "DB_PATH", database_path),
                patch.object(
                    memory_retriever,
                    "list_memories_with_embeddings",
                    db.list_memories_with_embeddings,
                ),
                patch(
                    "ronolm.memory_extractor._call_ollama_json",
                    return_value={"memories": [BARE_NAME_CANDIDATE]},
                ),
                patch(
                    "ronolm.memory_judge._call_ollama_json",
                    side_effect=AssertionError(
                        "clear name fact should not need the model judge"
                    ),
                ),
            ):
                db.init_db()
                thread_id = db.create_thread()
                message_id = db.add_message(
                    thread_id, "user", "My name is Mrinal Dhami"
                )
                candidates = extract_memories_from_user_message(
                    "My name is Mrinal Dhami"
                )
                approved = judge_memory_candidates(
                    "My name is Mrinal Dhami", candidates
                )

                for memory in approved:
                    db.add_memory(
                        thread_id=thread_id,
                        source_message_id=message_id,
                        memory_type=memory["type"],
                        subject=memory["subject"],
                        content=memory["content"],
                        lifetime=memory["lifetime"],
                        importance=memory["importance"],
                        confidence=memory["confidence"],
                    )

                retrieved = memory_retriever.retrieve_relevant_memories(
                    "What is my name?"
                )

        self.assertEqual(len(approved), 1)
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0]["content"], "User's name is Mrinal Dhami.")

    def test_add_memory_stores_embedding_and_does_not_duplicate_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "ronolm.sqlite"

            with (
                patch.object(db, "DB_PATH", database_path),
                patch.object(db, "get_embedding", return_value=[0.25, 0.75]) as embed,
            ):
                db.init_db()
                thread_id = db.create_thread()
                message_id = db.add_message(thread_id, "user", "I like ice cream")

                memory_id = db.add_memory(
                    thread_id=thread_id,
                    source_message_id=message_id,
                    memory_type="preference",
                    subject="ice cream",
                    content="User likes ice cream.",
                    lifetime="long_term",
                    importance=0.75,
                    confidence=0.93,
                )
                duplicate_id = db.add_memory(
                    thread_id=thread_id,
                    source_message_id=message_id,
                    memory_type="preference",
                    subject="ice cream",
                    content="User likes ice cream.",
                    lifetime="long_term",
                    importance=0.75,
                    confidence=0.93,
                )
                stored_vector = db.get_memory_embedding(memory_id)
                memories = db.list_memories_with_embeddings()

        self.assertEqual(duplicate_id, memory_id)
        self.assertEqual(stored_vector, [0.25, 0.75])
        self.assertEqual(memories[0]["embedding_model"], EMBEDDING_MODEL_NAME)
        self.assertEqual(memories[0]["embedding"], [0.25, 0.75])
        embed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
