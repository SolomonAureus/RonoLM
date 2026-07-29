import argparse
import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

from ronolm.config import VoiceConfig, load_voice_config


class STTError(RuntimeError):
    pass


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except (ImportError, RuntimeError):
        return False


def _resolved_runtime(config: VoiceConfig) -> tuple[str, str]:
    device = config.stt_device
    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    if device not in {"cuda", "cpu"}:
        raise STTError(
            f"STT_DEVICE must be auto, cuda, or cpu; got {device!r}."
        )

    compute_type = config.stt_compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


class FasterWhisperTranscriber:
    def __init__(
        self, model_name: str, device: str, compute_type: str
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise STTError(
                "Local speech recognition requires 'faster-whisper'. "
                "Install RonoLM with: pip install -e '.[voice]'"
            ) from exc

        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        try:
            self._model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
        except Exception as exc:
            if device != "cuda":
                raise STTError(
                    f"Could not initialize faster-whisper model "
                    f"{model_name!r} on {device}: {exc}"
                ) from exc
            warnings.warn(
                "CUDA faster-whisper initialization failed; falling back "
                f"to CPU int8. Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                self._model = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                )
            except Exception as cpu_exc:
                raise STTError(
                    "Could not initialize faster-whisper on CUDA or CPU: "
                    f"{cpu_exc}"
                ) from cpu_exc
            self.device = "cpu"
            self.compute_type = "int8"

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        path = Path(audio_path)
        if not path.is_file():
            raise STTError(f"Audio file does not exist: {path}")
        try:
            raw_segments, info = self._model.transcribe(
                str(path),
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            segments = []
            text_parts = []
            for segment in raw_segments:
                text = str(segment.text).strip()
                if not text:
                    continue
                text_parts.append(text)
                segments.append(
                    {
                        "id": getattr(segment, "id", None),
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": text,
                        "avg_logprob": getattr(segment, "avg_logprob", None),
                        "no_speech_prob": getattr(
                            segment, "no_speech_prob", None
                        ),
                    }
                )
        except Exception as exc:
            raise STTError(f"Speech transcription failed: {exc}") from exc

        duration = getattr(info, "duration", None)
        if duration is None:
            duration = max(
                (float(segment["end"]) for segment in segments), default=0.0
            )
        return {
            "text": " ".join(text_parts).strip(),
            "language": getattr(info, "language", None),
            "duration": float(duration),
            "segments": segments,
        }


@lru_cache(maxsize=4)
def _cached_transcriber(
    model_name: str, device: str, compute_type: str
) -> FasterWhisperTranscriber:
    return FasterWhisperTranscriber(model_name, device, compute_type)


def get_transcriber(
    config: VoiceConfig | None = None,
) -> FasterWhisperTranscriber:
    active_config = config or load_voice_config()
    if active_config.stt_provider != "faster_whisper":
        raise STTError(
            "Only STT_PROVIDER=faster_whisper is supported."
        )
    device, compute_type = _resolved_runtime(active_config)
    return _cached_transcriber(
        active_config.stt_model, device, compute_type
    )


def transcribe_audio(
    audio_path: str, config: VoiceConfig | None = None
) -> dict[str, Any]:
    return get_transcriber(config).transcribe(audio_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a local WAV file with faster-whisper."
    )
    parser.add_argument("audio_path")
    arguments = parser.parse_args()
    try:
        result = transcribe_audio(arguments.audio_path)
    except STTError as exc:
        parser.exit(1, f"STT error: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
