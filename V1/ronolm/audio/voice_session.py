from pathlib import Path
from typing import Any

from ronolm.audio.recorder import (
    MIN_USABLE_DURATION_SECONDS,
    AudioRecordingError,
    record_audio,
)
from ronolm.audio.stt import FasterWhisperTranscriber, STTError, get_transcriber
from ronolm.audio.tts import TTSError, speak_text, validate_piper_config
from ronolm.chat_service import process_user_message
from ronolm.config import VoiceConfig


class VoiceSession:
    def __init__(
        self,
        thread_id: str,
        config: VoiceConfig,
        *,
        tts_enabled: bool = True,
        transcriber: FasterWhisperTranscriber | None = None,
    ) -> None:
        self.thread_id = thread_id
        self.config = config
        self.tts_enabled = tts_enabled
        self.muted = not tts_enabled
        self.last_assistant_response: str | None = None

        if tts_enabled:
            validate_piper_config(config)
        self.transcriber = transcriber or get_transcriber(config)

    def _delete_input_audio(self, path: Path) -> None:
        if self.config.delete_temp_audio:
            path.unlink(missing_ok=True)

    def _speak_response(self, response: str) -> None:
        if self.muted or not self.tts_enabled:
            return
        try:
            speak_text(response, config=self.config)
        except TTSError as exc:
            print(f"Voice output error: {exc}")

    def process_voice_turn(self) -> str | None:
        recording = None
        try:
            recording = record_audio(self.config.tts_output_dir)
            if recording.duration < MIN_USABLE_DURATION_SECONDS:
                print("No usable speech detected.")
                return None
            transcription = self.transcriber.transcribe(recording.path)
            transcript = str(transcription.get("text") or "").strip()
            if not transcript:
                print("No usable speech detected.")
                return None

            print(f"\nYou said:\n{transcript}\n")
            metadata: dict[str, Any] = {
                "stt_provider": "faster_whisper",
                "stt_model": self.config.stt_model,
                "language": transcription.get("language"),
                "duration": transcription.get("duration"),
            }
            try:
                response = process_user_message(
                    thread_id=self.thread_id,
                    user_text=transcript,
                    input_mode="voice",
                    metadata=metadata,
                )
            except Exception as exc:
                print(f"RonoLM processing error: {exc}")
                return None
            self.last_assistant_response = response
            print(f"RonoLM:\n{response}\n")
            self._speak_response(response)
            return response
        except (AudioRecordingError, STTError, ValueError) as exc:
            print(f"Voice input error: {exc}")
            return None
        finally:
            if recording is not None:
                self._delete_input_audio(recording.path)

    def process_text_turn(self, text: str) -> str | None:
        clean_text = text.strip()
        if not clean_text:
            return None
        try:
            response = process_user_message(
                thread_id=self.thread_id,
                user_text=clean_text,
                input_mode="text",
            )
        except Exception as exc:
            print(f"RonoLM processing error: {exc}")
            return None
        self.last_assistant_response = response
        print(f"RonoLM:\n{response}\n")
        self._speak_response(response)
        return response

    def mute(self) -> None:
        self.muted = True

    def unmute(self) -> bool:
        if not self.tts_enabled:
            return False
        self.muted = False
        return True

    def repeat(self) -> bool:
        if self.last_assistant_response is None:
            print("There is no assistant response to repeat.")
            return False
        if self.muted or not self.tts_enabled:
            print("TTS playback is muted.")
            return False
        self._speak_response(self.last_assistant_response)
        return True
