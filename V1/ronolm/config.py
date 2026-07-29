import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigurationError(RuntimeError):
    pass


def _boolean(name: str, raw: str) -> bool:
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(
        f"{name} must be true or false, got {raw!r}."
    )


def _optional_path(raw: str | None) -> Path | None:
    if raw is None or not raw.strip():
        return None
    return Path(raw).expanduser().resolve()


def _project_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class VoiceConfig:
    enabled: bool
    stt_provider: str
    stt_model: str
    stt_device: str
    stt_compute_type: str
    piper_executable: str
    piper_model_path: Path | None
    piper_config_path: Path | None
    tts_provider: str
    tts_output_dir: Path
    delete_temp_audio: bool
    tts_max_chars: int

    def debug_values(self) -> dict[str, str | bool | int | None]:
        return {
            "VOICE_ENABLED": self.enabled,
            "STT_PROVIDER": self.stt_provider,
            "STT_MODEL": self.stt_model,
            "STT_DEVICE": self.stt_device,
            "STT_COMPUTE_TYPE": self.stt_compute_type,
            "PIPER_EXECUTABLE": self.piper_executable,
            "PIPER_MODEL_PATH": (
                str(self.piper_model_path) if self.piper_model_path else None
            ),
            "PIPER_CONFIG_PATH": (
                str(self.piper_config_path) if self.piper_config_path else None
            ),
            "TTS_PROVIDER": self.tts_provider,
            "TTS_OUTPUT_DIR": str(self.tts_output_dir),
            "DELETE_TEMP_AUDIO": self.delete_temp_audio,
            "TTS_MAX_CHARS": self.tts_max_chars,
        }


def load_voice_config(
    environ: Mapping[str, str] | None = None,
) -> VoiceConfig:
    values = os.environ if environ is None else environ
    max_chars_raw = values.get("TTS_MAX_CHARS", "1200")
    try:
        max_chars = int(max_chars_raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"TTS_MAX_CHARS must be an integer, got {max_chars_raw!r}."
        ) from exc
    if max_chars <= 0:
        raise ConfigurationError("TTS_MAX_CHARS must be greater than zero.")

    return VoiceConfig(
        enabled=_boolean(
            "VOICE_ENABLED", values.get("VOICE_ENABLED", "true")
        ),
        stt_provider=values.get(
            "STT_PROVIDER", "faster_whisper"
        ).strip(),
        stt_model=values.get("STT_MODEL", "small").strip(),
        stt_device=values.get("STT_DEVICE", "auto").strip().casefold(),
        stt_compute_type=values.get(
            "STT_COMPUTE_TYPE", "auto"
        ).strip().casefold(),
        piper_executable=values.get(
            "PIPER_EXECUTABLE", "piper"
        ).strip(),
        piper_model_path=_optional_path(values.get("PIPER_MODEL_PATH")),
        piper_config_path=_optional_path(values.get("PIPER_CONFIG_PATH")),
        tts_provider=values.get("TTS_PROVIDER", "piper").strip().casefold(),
        tts_output_dir=_project_path(
            values.get("TTS_OUTPUT_DIR", "tmp/audio")
        ),
        delete_temp_audio=_boolean(
            "DELETE_TEMP_AUDIO",
            values.get("DELETE_TEMP_AUDIO", "true"),
        ),
        tts_max_chars=max_chars,
    )
