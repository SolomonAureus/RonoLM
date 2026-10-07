import argparse

from ronolm.audio.recorder import AudioRecordingError, get_input_device_info
from ronolm.audio.stt import STTError
from ronolm.audio.tts import TTSError
from ronolm.audio.voice_session import VoiceSession
from ronolm.config import ConfigurationError, VoiceConfig, load_voice_config
from ronolm.db import get_or_create_thread, init_db


def _print_debug(session: VoiceSession) -> None:
    print(f"THREAD_ID={session.thread_id}")
    for name, value in session.config.debug_values().items():
        print(f"{name}={value}")
    if session.microphone_info is not None:
        print(f"MICROPHONE={session.microphone_info.name}")


def _run_loop(session: VoiceSession) -> None:
    while True:
        command = input(
            "\nPress Enter to speak or type a command: "
        ).strip().casefold()
        if command == "":
            session.process_voice_turn()
        elif command in {"/quit", "/exit"}:
            return
        elif command == "/text":
            typed_text = input("You: ")
            session.process_text_turn(typed_text)
        elif command == "/mute":
            session.mute()
            print("TTS playback muted.")
        elif command == "/unmute":
            if session.unmute():
                print("TTS playback enabled.")
            else:
                print("TTS is disabled for this session.")
        elif command == "/repeat":
            session.repeat()
        elif command == "/mic":
            session.print_microphone_status()
        elif command == "/debug":
            _print_debug(session)
        else:
            print(
                "Unknown command. Use /text, /mute, /unmute, "
                "/repeat, /mic, /debug, or /quit."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RonoLM local push-to-talk voice interface."
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Transcribe and process speech without audio playback.",
    )
    arguments = parser.parse_args()

    try:
        config = load_voice_config()
        if not config.enabled:
            parser.exit(
                1,
                "Voice is disabled. Set VOICE_ENABLED=true to start it.\n",
            )
        init_db()
        thread_id = get_or_create_thread()
        microphone_info = get_input_device_info()
        session = VoiceSession(
            thread_id,
            config,
            tts_enabled=not arguments.no_tts,
            microphone_info=microphone_info,
        )
    except (
        AudioRecordingError,
        ConfigurationError,
        STTError,
        TTSError,
    ) as exc:
        parser.exit(1, f"Voice startup error: {exc}\n")

    try:
        _run_loop(session)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting voice session.")


if __name__ == "__main__":
    main()
