import math
import os
import sys
import tempfile
import threading
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
    peak_level: float = 0.0


@dataclass(frozen=True)
class InputDeviceInfo:
    name: str
    max_input_channels: int
    default_sample_rate: float


def _load_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise AudioRecordingError(
            "Microphone recording requires the optional 'sounddevice' "
            "dependency. Install RonoLM with: pip install -e '.[voice]'"
        ) from exc
    except OSError as exc:
        raise AudioRecordingError(
            "The sounddevice package is installed, but the native PortAudio "
            "library is missing. On Pop!_OS/Ubuntu install it with: "
            "sudo apt install libportaudio2"
        ) from exc
    return sd


def get_input_device_info(
    *, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> InputDeviceInfo:
    """Verify and describe the default microphone without recording audio."""

    sd = _load_sounddevice()
    try:
        raw = sd.query_devices(kind="input")
        sd.check_input_settings(
            device=None,
            channels=DEFAULT_CHANNELS,
            dtype="int16",
            samplerate=sample_rate,
        )
    except Exception as exc:
        raise AudioRecordingError(
            f"No usable default microphone was found: {exc}"
        ) from exc
    return InputDeviceInfo(
        name=str(raw["name"]),
        max_input_channels=int(raw["max_input_channels"]),
        default_sample_rate=float(raw["default_samplerate"]),
    )


def _format_level_meter(level: float, width: int = 24) -> str:
    """Convert normalized RMS audio into a terminal-friendly decibel meter."""

    safe_level = max(0.0, min(1.0, float(level)))
    decibels = 20.0 * math.log10(max(safe_level, 1e-6))
    visible = max(0.0, min(1.0, (decibels + 60.0) / 60.0))
    filled = round(visible * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {visible * 100:3.0f}%"


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

    sd = _load_sounddevice()
    try:
        import numpy as np
    except ImportError as exc:
        raise AudioRecordingError(
            "Microphone level display requires NumPy."
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
    level_lock = threading.Lock()
    level_state = {"current": 0.0, "peak": 0.0}
    meter_stop = threading.Event()

    def callback(
        indata: Any, frames: int, time_info: Any, status: Any
    ) -> None:
        del frames, time_info
        if status:
            stream_warnings.append(str(status))
        chunks.append(indata.copy().tobytes())
        samples = indata.astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(samples)))) / 32768.0
        with level_lock:
            level_state["current"] = rms
            level_state["peak"] = max(level_state["peak"], rms)

    def display_meter() -> None:
        while not meter_stop.wait(0.08):
            with level_lock:
                level = level_state["current"]
            sys.stdout.write(
                "\r[LISTENING] Mic "
                f"{_format_level_meter(level)}  Press Enter to stop"
            )
            sys.stdout.flush()

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            callback=callback,
        ):
            meter_thread = None
            if sys.stdout.isatty():
                meter_thread = threading.Thread(
                    target=display_meter,
                    name="ronolm-microphone-meter",
                    daemon=True,
                )
                meter_thread.start()
            else:
                print("[LISTENING] Microphone active. Press Enter to stop.")
            input()
            meter_stop.set()
            if meter_thread is not None:
                meter_thread.join(timeout=0.5)
                sys.stdout.write("\r" + " " * 90 + "\r")
                sys.stdout.flush()
    except KeyboardInterrupt:
        meter_stop.set()
        path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        meter_stop.set()
        path.unlink(missing_ok=True)
        raise AudioRecordingError(
            f"Could not record microphone audio: {exc}"
        ) from exc

    audio_bytes = b"".join(chunks)
    duration = len(audio_bytes) / (sample_rate * channels * 2)
    with level_lock:
        peak_level = level_state["peak"]
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
    return RecordedAudio(
        path, duration, sample_rate, channels, peak_level
    )
