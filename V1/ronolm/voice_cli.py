import argparse

from ronolm.audio.stt import STTError
from ronolm.audio.tts import TTSError
from ronolm.audio.voice_session import VoiceSession
from ronolm.config import ConfigurationError, VoiceConfig, load_voice_config
from ronolm.db import get_or_create_thread, init_db


def _print_debug(config: VoiceConfig, thread_id: str) -> None:
    print(f"THREAD_ID={thread_id}")
    for name, value in config.debug_values().items():
        print(f"{name}={value}")


def _run_loop(session: VoiceSession) -> None:
    print("Press Enter to speak.")
    print("Type /quit to exit.")
    print("Type /text to switch to typed input for one turn.")

    while True:
        command = input("\nvoice> ").strip().casefold()
        if command == "":
            session.process_voice_turn()
        elif command == "/quit":
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
        elif command == "/debug":
            _print_debug(session.config, session.thread_id)
        else:
            print(
                "Unknown command. Use /text, /mute, /unmute, "
                "/repeat, /debug, or /quit."
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
        print("Initializing local speech recognition...", flush=True)
        session = VoiceSession(
            thread_id,
            config,
            tts_enabled=not arguments.no_tts,
        )
    except (ConfigurationError, STTError, TTSError) as exc:
        parser.exit(1, f"Voice startup error: {exc}\n")

    print(f"RonoLM voice session ready. Thread: {thread_id}")
    if arguments.no_tts:
        print("TTS playback is disabled (--no-tts).")
    try:
        _run_loop(session)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting voice session.")


if __name__ == "__main__":
    main()
