import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1
MIN_USABLE_DURATION_SECONDS = 0.25


class AudioRecordingError(RuntimeError):
    pass


@dataclass(frozen=True)
class RecordedAudio:
    path: Path
    duration: float
    sample_rate: int
    channels: int


def record_audio(
    output_dir: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> RecordedAudio:
    """Record push-to-talk microphone audio until the user presses Enter."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero.")
    if channels != 1:
        raise ValueError("RonoLM voice input currently supports mono audio only.")

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise AudioRecordingError(
            "Microphone recording requires the optional 'sounddevice' "
            "dependency. Install RonoLM with: pip install -e '.[voice]'"
        ) from exc

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix="ronolm-input-", suffix=".wav", dir=destination
    )
    os.close(descriptor)
    path = Path(raw_path)
    chunks: list[bytes] = []
    stream_warnings: list[str] = []

    def callback(
        indata: Any, frames: int, time_info: Any, status: Any
    ) -> None:
        del frames, time_info
        if status:
            stream_warnings.append(str(status))
        chunks.append(indata.copy().tobytes())

    try:
        print("Recording... press Enter to stop.")
        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            callback=callback,
        ):
            input()
    except KeyboardInterrupt:
        path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise AudioRecordingError(
            f"Could not record microphone audio: {exc}"
        ) from exc

    audio_bytes = b"".join(chunks)
    duration = len(audio_bytes) / (sample_rate * channels * 2)
    try:
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
    except (OSError, wave.Error) as exc:
        path.unlink(missing_ok=True)
        raise AudioRecordingError(
            f"Could not write recorded WAV file: {exc}"
        ) from exc

    if stream_warnings:
        print(f"Audio warning: {stream_warnings[-1]}")
    return RecordedAudio(path, duration, sample_rate, channels)
