import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ronolm.audio.playback import AudioPlaybackError, play_wav
from ronolm.config import VoiceConfig, load_voice_config


class TTSError(RuntimeError):
    pass


def validate_piper_config(config: VoiceConfig | None = None) -> str:
    active_config = config or load_voice_config()
    if active_config.tts_provider != "piper":
        raise TTSError("Only TTS_PROVIDER=piper is supported.")
    if active_config.piper_model_path is None:
        raise TTSError(
            "PIPER_MODEL_PATH is not set. Set it to a local Piper "
            ".onnx voice model before starting voice output."
        )
    if not active_config.piper_model_path.is_file():
        raise TTSError(
            f"PIPER_MODEL_PATH does not exist: "
            f"{active_config.piper_model_path}"
        )
    if (
        active_config.piper_config_path is not None
        and not active_config.piper_config_path.is_file()
    ):
        raise TTSError(
            f"PIPER_CONFIG_PATH does not exist: "
            f"{active_config.piper_config_path}"
        )
    configured_executable = active_config.piper_executable
    if not configured_executable:
        raise TTSError("PIPER_EXECUTABLE cannot be empty.")
    executable_path = Path(configured_executable).expanduser()
    if executable_path.is_absolute() or executable_path.parent != Path("."):
        executable = (
            str(executable_path.resolve())
            if executable_path.is_file()
            and os.access(executable_path, os.X_OK)
            else None
        )
    else:
        executable = shutil.which(configured_executable)
    if executable is None:
        raise TTSError(
            f"Piper executable {configured_executable!r} was not found. "
            "Set PIPER_EXECUTABLE to its absolute path."
        )
    return executable


def synthesize_to_wav(
    text: str,
    output_path: str | None = None,
    *,
    config: VoiceConfig | None = None,
) -> str:
    clean_text = text.strip()
    if not clean_text:
        raise TTSError("Cannot synthesize empty text.")

    active_config = config or load_voice_config()
    executable = validate_piper_config(active_config)
    temporary = output_path is None
    if temporary:
        active_config.tts_output_dir.mkdir(parents=True, exist_ok=True)
        descriptor, raw_path = tempfile.mkstemp(
            prefix="ronolm-output-",
            suffix=".wav",
            dir=active_config.tts_output_dir,
        )
        os.close(descriptor)
        target = Path(raw_path)
    else:
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

    command = [
        executable,
        "--model",
        str(active_config.piper_model_path),
    ]
    if active_config.piper_config_path is not None:
        command.extend(
            ["--config", str(active_config.piper_config_path)]
        )
    command.extend(["--output_file", str(target)])

    try:
        completed = subprocess.run(
            command,
            input=clean_text,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        if temporary:
            target.unlink(missing_ok=True)
        raise TTSError(f"Piper synthesis failed: {exc}") from exc
    if completed.returncode != 0:
        if temporary:
            target.unlink(missing_ok=True)
        detail = completed.stderr.strip() or "unknown Piper error"
        raise TTSError(f"Piper synthesis failed: {detail}")
    if not target.is_file() or target.stat().st_size == 0:
        if temporary:
            target.unlink(missing_ok=True)
        raise TTSError("Piper completed without producing a usable WAV file.")
    return str(target)


def speak_text(
    text: str, *, config: VoiceConfig | None = None
) -> None:
    active_config = config or load_voice_config()
    clean_text = text.strip()
    if not clean_text:
        raise TTSError("Cannot speak an empty response.")
    speech_text = clean_text[: active_config.tts_max_chars]
    if len(clean_text) > active_config.tts_max_chars:
        speech_text += " Response truncated for speech."

    audio_path = synthesize_to_wav(speech_text, config=active_config)
    try:
        play_wav(audio_path)
    except AudioPlaybackError as exc:
        raise TTSError(str(exc)) from exc
    finally:
        if active_config.delete_temp_audio:
            Path(audio_path).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize and play text using local Piper TTS."
    )
    parser.add_argument("text")
    parser.add_argument(
        "--output",
        help="Generate a WAV at this path instead of playing it.",
    )
    arguments = parser.parse_args()
    try:
        if arguments.output:
            generated = synthesize_to_wav(
                arguments.text, arguments.output
            )
            print(generated)
        else:
            speak_text(arguments.text)
    except TTSError as exc:
        parser.exit(1, f"TTS error: {exc}\n")


if __name__ == "__main__":
    main()
