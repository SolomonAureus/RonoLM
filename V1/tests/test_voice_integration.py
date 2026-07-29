import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ronolm.db as db
from ronolm.audio.recorder import RecordedAudio
from ronolm.audio.stt import _resolved_runtime
from ronolm.audio.tts import TTSError, speak_text, validate_piper_config
from ronolm.audio.voice_session import VoiceSession
from ronolm.chat_service import process_user_message
from ronolm.config import ConfigurationError, load_voice_config
from ronolm.embedding_client import EmbeddingError
from ronolm.voice_cli import _run_loop


class VoiceConfigTests(unittest.TestCase):
    def test_voice_defaults_use_local_providers(self) -> None:
        config = load_voice_config({})

        self.assertTrue(config.enabled)
        self.assertEqual(config.stt_provider, "faster_whisper")
        self.assertEqual(config.stt_model, "small")
        self.assertEqual(config.tts_provider, "piper")
        self.assertTrue(config.delete_temp_audio)
        self.assertEqual(config.tts_max_chars, 1200)

    def test_invalid_boolean_has_clear_error(self) -> None:
        with self.assertRaisesRegex(
            ConfigurationError, "VOICE_ENABLED must be true or false"
        ):
            load_voice_config({"VOICE_ENABLED": "sometimes"})

    def test_cpu_auto_compute_type_resolves_to_int8(self) -> None:
        config = load_voice_config(
            {"STT_DEVICE": "cpu", "STT_COMPUTE_TYPE": "auto"}
        )

        self.assertEqual(_resolved_runtime(config), ("cpu", "int8"))

    def test_missing_piper_model_names_required_variable(self) -> None:
        with self.assertRaisesRegex(TTSError, "PIPER_MODEL_PATH"):
            validate_piper_config(load_voice_config({}))


class ChatServiceIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "ronolm.sqlite"
        self.patchers = [
            patch.object(db, "DB_PATH", self.database_path),
            patch(
                "ronolm.topic_manager.get_embedding",
                side_effect=EmbeddingError("unavailable in test"),
            ),
            patch(
                "ronolm.topic_router.get_embedding",
                side_effect=EmbeddingError("unavailable in test"),
            ),
            patch(
                "ronolm.thread_retriever.get_embedding",
                side_effect=EmbeddingError("unavailable in test"),
            ),
            patch(
                "ronolm.chat_service.retrieve_relevant_memories",
                return_value=[],
            ),
            patch(
                "ronolm.chat_service.generate_reply",
                return_value="Shared pipeline response.",
            ),
            patch(
                "ronolm.chat_service.extract_memories_from_user_message",
                return_value=[],
            ),
            patch(
                "ronolm.chat_service.judge_memory_candidates",
                return_value=[],
            ),
        ]
        for patcher in self.patchers:
            patcher.start()
        db.init_db()

    def tearDown(self) -> None:
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temp_dir.cleanup()

    def test_voice_text_uses_complete_shared_message_and_topic_pipeline(
        self,
    ) -> None:
        thread_id = db.create_thread("Voice test")

        response = process_user_message(
            thread_id,
            "Continue RonoLM thread recall.",
            input_mode="voice",
            metadata={"stt_provider": "faster_whisper"},
        )

        self.assertEqual(response, "Shared pipeline response.")
        messages = db.list_messages(thread_id)
        self.assertEqual(
            [(row["role"], row["content"]) for row in messages],
            [
                ("user", "Continue RonoLM thread recall."),
                ("assistant", "Shared pipeline response."),
            ],
        )
        self.assertEqual(len(db.list_message_episodes(messages[0]["id"])), 1)
        self.assertEqual(len(db.list_message_episodes(messages[1]["id"])), 1)


class VoiceAdapterTests(unittest.TestCase):
    def test_voice_cli_commands_delegate_to_session(self) -> None:
        session = unittest.mock.Mock()
        session.config = load_voice_config({})
        session.thread_id = "thread_commands"
        session.unmute.return_value = True

        with patch(
            "builtins.input",
            side_effect=[
                "",
                "/text",
                "typed turn",
                "/mute",
                "/unmute",
                "/repeat",
                "/debug",
                "/quit",
            ],
        ):
            _run_loop(session)

        session.process_voice_turn.assert_called_once_with()
        session.process_text_turn.assert_called_once_with("typed turn")
        session.mute.assert_called_once_with()
        session.unmute.assert_called_once_with()
        session.repeat.assert_called_once_with()

    def test_voice_session_routes_transcript_and_deletes_input_audio(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "input.wav"
            audio_path.write_bytes(b"test audio")
            config = load_voice_config(
                {
                    "TTS_OUTPUT_DIR": directory,
                    "DELETE_TEMP_AUDIO": "true",
                }
            )
            transcriber = unittest.mock.Mock()
            transcriber.transcribe.return_value = {
                "text": "hello from voice",
                "language": "en",
                "duration": 1.25,
                "segments": [],
            }
            session = VoiceSession(
                "thread_test",
                config,
                tts_enabled=False,
                transcriber=transcriber,
            )

            with (
                patch(
                    "ronolm.audio.voice_session.record_audio",
                    return_value=RecordedAudio(
                        audio_path, 1.25, 16_000, 1
                    ),
                ),
                patch(
                    "ronolm.audio.voice_session.process_user_message",
                    return_value="voice response",
                ) as process,
            ):
                response = session.process_voice_turn()

            self.assertEqual(response, "voice response")
            self.assertFalse(audio_path.exists())
            process.assert_called_once()
            arguments = process.call_args.kwargs
            self.assertEqual(arguments["user_text"], "hello from voice")
            self.assertEqual(arguments["input_mode"], "voice")
            self.assertEqual(arguments["metadata"]["language"], "en")

    def test_long_tts_text_is_truncated_before_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "generated.wav"
            generated.write_bytes(b"wav")
            config = load_voice_config(
                {
                    "TTS_OUTPUT_DIR": directory,
                    "TTS_MAX_CHARS": "10",
                    "DELETE_TEMP_AUDIO": "true",
                }
            )
            with (
                patch(
                    "ronolm.audio.tts.synthesize_to_wav",
                    return_value=str(generated),
                ) as synthesize,
                patch("ronolm.audio.tts.play_wav"),
            ):
                speak_text("abcdefghij-more text", config=config)

            spoken = synthesize.call_args.args[0]
            self.assertEqual(
                spoken, "abcdefghij Response truncated for speech."
            )
            self.assertFalse(generated.exists())


if __name__ == "__main__":
    unittest.main()
