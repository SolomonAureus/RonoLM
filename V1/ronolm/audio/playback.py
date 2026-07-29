import shutil
import subprocess
import wave
from pathlib import Path


class AudioPlaybackError(RuntimeError):
    pass


def _play_with_sounddevice(path: Path) -> None:
    import numpy as np
    import sounddevice as sd

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    dtype_by_width = {1: np.uint8, 2: np.int16, 4: np.int32}
    dtype = dtype_by_width.get(sample_width)
    if dtype is None:
        raise AudioPlaybackError(
            f"Unsupported WAV sample width: {sample_width} bytes."
        )
    audio = np.frombuffer(frames, dtype=dtype)
    if sample_width == 1:
        audio = audio.astype(np.int16) - 128
    if channels > 1:
        audio = audio.reshape((-1, channels))
    sd.play(audio, sample_rate)
    sd.wait()


def _system_player_command(path: Path) -> list[str] | None:
    candidates = (
        ("aplay", [str(path)]),
        ("paplay", [str(path)]),
        (
            "ffplay",
            ["-nodisp", "-autoexit", "-loglevel", "error", str(path)],
        ),
        ("play", ["-q", str(path)]),
    )
    for executable, arguments in candidates:
        resolved = shutil.which(executable)
        if resolved:
            return [resolved, *arguments]
    return None


def play_wav(audio_path: str | Path) -> None:
    path = Path(audio_path)
    if not path.is_file():
        raise AudioPlaybackError(f"WAV file does not exist: {path}")

    sounddevice_error: Exception | None = None
    try:
        _play_with_sounddevice(path)
        return
    except Exception as exc:
        sounddevice_error = exc

    command = _system_player_command(path)
    if command is None:
        detail = f" sounddevice error: {sounddevice_error}" if sounddevice_error else ""
        raise AudioPlaybackError(
            "Audio playback failed and no supported system player was found."
            + detail
        )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioPlaybackError(f"Audio playback failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown player error"
        raise AudioPlaybackError(f"Audio playback failed: {detail}")
