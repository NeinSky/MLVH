"""Настройки MLVH: ключ берём из .env, остальное — из config.json."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

from app.personas import (
    DEFAULT_PERSONA_ID,
    ELVI_PROMPT,
    Persona,
    builtin_personas,
    persona_by_id,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.json"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

AVAILABLE_MODELS = ("deepseek-flash", "deepseek-v4-pro")
DEFAULT_VOICE = "ru-RU-DmitryNeural"
FALLBACK_VOICES = ("ru-RU-DmitryNeural", "ru-RU-SvetlanaNeural")

# Конфига ещё нет — работаем от характера по умолчанию (Элви).
DEFAULT_SYSTEM_PROMPT = ELVI_PROMPT


def _known_values(cls: type, data: dict) -> dict:
    """Оставляем только известные dataclass-поля, чтобы старый конфиг не ломал запуск."""
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in data.items() if key in names}


@dataclass
class ModelSettings:
    """Параметры запроса к DeepSeek."""

    name: str = "deepseek-flash"
    temperature: float = 0.7
    max_tokens: int = 2000
    thinking: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "ModelSettings":
        return cls(**_known_values(cls, data))


@dataclass
class TTSSettings:
    """Параметры голоса: всё, что можно покрутить в интерфейсе."""

    engine: str = "edge"
    voice: str = DEFAULT_VOICE
    rate: int = 0            # темп, -50..+50 %
    volume: int = 0          # громкость синтеза, -50..+50 %
    pitch: int = 0           # тон, -50..+50 Гц
    playback_volume: float = 0.9
    max_speak_chars: int = 1500
    enabled: bool = True
    # --- локальный движок Chatterbox ---
    chatterbox_reference: str = ""     # путь к образцу голоса (.mp3/.wav)
    chatterbox_exaggeration: float = 0.5   # выразительность 0..1
    chatterbox_cfg: float = 0.5            # верность образцу
    chatterbox_temperature: float = 0.7    # ниже — стабильнее произношение
    chatterbox_repetition_penalty: float = 1.3  # выше — меньше зацикливаний
    chatterbox_watermark: bool = False     # водяной знак PerTh (только для сохранения)
    chatterbox_fp16: bool = False          # на 3070 Ti FP16 оказался в 30 раз медленнее
    # --- локальный движок Silero ---
    silero_speaker: str = "baya"           # baya, kseniya, xenia, aidar, eugene
    silero_sample_rate: int = 48000
    silero_accent: bool = True             # автоматические ударения
    silero_yo: bool = True                 # буква «ё»

    @classmethod
    def from_dict(cls, data: dict) -> "TTSSettings":
        return cls(**_known_values(cls, data))


ASR_MODES = ("off", "push", "wake")
ASR_MODE_TITLES = {
    "off": "Без микрофона",
    "push": "По нажатию",
    "wake": "Слушать микрофон",
}


@dataclass
class ASRSettings:
    """Распознавание речи: режим, движок, горячая клавиша."""

    mode: str = "push"                  # off / push / wake
    engine: str = "whisper"             # whisper / vosk
    model: str = "small"                # tiny, base, small, medium, large-v3
    device: str = "cuda"                # cuda / cpu
    compute_type: str = "int8_float16"
    language: str = "ru"
    hotkey: str = "Ctrl+Space"
    global_hotkey: bool = False         # ловить клавишу, даже когда окно не в фокусе
    auto_send: bool = False             # отправлять сразу после распознавания
    wake_word: str = "элви"             # спецслово для режима «Слушать микрофон»
    input_device: str = ""              # пусто — устройство по умолчанию
    max_seconds: int = 30               # предел одной записи
    mute_while_speaking: bool = True    # не слушать, пока MLVH говорит

    @classmethod
    def from_dict(cls, data: dict) -> "ASRSettings":
        settings = cls(**_known_values(cls, data))
        if settings.mode not in ASR_MODES:
            print(
                f"[config] Неизвестный режим микрофона «{settings.mode}» — "
                "включён режим «По нажатию»."
            )
            settings.mode = "push"
        settings.max_seconds = max(3, min(120, int(settings.max_seconds or 30)))
        return settings


@dataclass
class Settings:
    """Все настройки приложения."""

    model: ModelSettings = None
    tts: TTSSettings = None
    asr: ASRSettings = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    history_turns: int = 10
    personas: list[Persona] = None
    active_persona: str = DEFAULT_PERSONA_ID

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = ModelSettings()
        if self.tts is None:
            self.tts = TTSSettings()
        if self.asr is None:
            self.asr = ASRSettings()
        if not self.personas:
            # Конфига нет: характеры встроенные, поэтому сразу берём пресет
            # характера по умолчанию — иначе промпт и голос разъедутся.
            self.personas = builtin_personas()
            default = persona_by_id(self.personas, self.active_persona)
            if default is not None:
                self.apply_persona(default)

    def persona(self) -> Persona | None:
        """Активный характер (или None, если id не найден)."""
        return persona_by_id(self.personas, self.active_persona)

    def apply_persona(self, persona: Persona) -> None:
        """Переключаемся на характер: промпт и голос берём из него."""
        self.active_persona = persona.id
        self.system_prompt = persona.prompt
        self.tts.voice = persona.voice
        self.tts.rate = persona.rate
        self.tts.volume = persona.volume
        self.tts.pitch = persona.pitch
        self.tts.max_speak_chars = persona.max_speak_chars

    def save_persona(self, persona: Persona) -> None:
        """Обновляем характер в списке (или добавляем, если он новый)."""
        for index, existing in enumerate(self.personas):
            if existing.id == persona.id:
                self.personas[index] = persona
                break
        else:
            self.personas.append(persona)
        if self.active_persona == persona.id:
            self.apply_persona(persona)

    def reset_persona(self, persona_id: str) -> Persona:
        """Возвращаем встроенный характер на место изменённого."""
        original = next(
            (item for item in builtin_personas() if item.id == persona_id),
            None,
        )
        if original is None:
            raise ValueError(f"Характер «{persona_id}» не встроенный: сброс невозможен")
        self.save_persona(original)
        return original

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        model = ModelSettings.from_dict(data.get("model") or {})
        tts = TTSSettings.from_dict(data.get("tts") or {})
        asr = ASRSettings.from_dict(data.get("asr") or {})
        personas = _read_personas(data.get("personas"))
        system_prompt = data.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        active = data.get("active_persona")

        if not active:
            # Конфиг от версии без характеров: сохраняем текущий голос как свой характер.
            custom = Persona(
                id="custom",
                name="Мой голос",
                emoji="🎛",
                prompt=system_prompt,
                voice=tts.voice,
                rate=tts.rate,
                volume=tts.volume,
                pitch=tts.pitch,
                max_speak_chars=tts.max_speak_chars,
                greeting="",
            )
            personas = [item for item in personas if item.id != "custom"] + [custom]
            active = "custom"

        return cls(
            model=model,
            tts=tts,
            asr=asr,
            system_prompt=system_prompt,
            history_turns=int(data.get("history_turns") or 10),
            personas=personas,
            active_persona=active,
        )

    def to_dict(self) -> dict:
        return {
            "model": {
                "name": self.model.name,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "thinking": self.model.thinking,
            },
            "tts": {
                "engine": self.tts.engine,
                "voice": self.tts.voice,
                "rate": self.tts.rate,
                "volume": self.tts.volume,
                "pitch": self.tts.pitch,
                "playback_volume": self.tts.playback_volume,
                "max_speak_chars": self.tts.max_speak_chars,
                "enabled": self.tts.enabled,
                "chatterbox_reference": self.tts.chatterbox_reference,
                "chatterbox_exaggeration": self.tts.chatterbox_exaggeration,
                "chatterbox_cfg": self.tts.chatterbox_cfg,
                "chatterbox_temperature": self.tts.chatterbox_temperature,
                "chatterbox_repetition_penalty": self.tts.chatterbox_repetition_penalty,
                "chatterbox_watermark": self.tts.chatterbox_watermark,
                "chatterbox_fp16": self.tts.chatterbox_fp16,
                "silero_speaker": self.tts.silero_speaker,
                "silero_sample_rate": self.tts.silero_sample_rate,
                "silero_accent": self.tts.silero_accent,
                "silero_yo": self.tts.silero_yo,
            },
            "asr": {
                "mode": self.asr.mode,
                "engine": self.asr.engine,
                "model": self.asr.model,
                "device": self.asr.device,
                "compute_type": self.asr.compute_type,
                "language": self.asr.language,
                "hotkey": self.asr.hotkey,
                "global_hotkey": self.asr.global_hotkey,
                "auto_send": self.asr.auto_send,
                "wake_word": self.asr.wake_word,
                "input_device": self.asr.input_device,
                "max_seconds": self.asr.max_seconds,
                "mute_while_speaking": self.asr.mute_while_speaking,
            },
            "system_prompt": self.system_prompt,
            "history_turns": self.history_turns,
            "active_persona": self.active_persona,
            "personas": [persona.to_dict() for persona in self.personas],
        }


def _read_personas(raw: object) -> list[Persona]:
    """Характеры из конфига; битые записи пропускаем, недостающие добираем встроенными."""
    personas: list[Persona] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                personas.append(Persona.from_dict(item))
            except (TypeError, ValueError) as exc:
                print(f"[config] Характер пропущен: {exc}")

    known = {persona.id for persona in personas}
    for builtin in builtin_personas():
        if builtin.id not in known:
            personas.append(builtin)
    return personas



def load_settings() -> Settings:
    """Читаем config.json; если файла нет или он битый — берём значения по умолчанию."""
    if not CONFIG_PATH.exists():
        return Settings()
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[config] Не удалось прочитать {CONFIG_PATH}: {exc}. Взяты настройки по умолчанию.")
        return Settings()
    return Settings.from_dict(raw)


def save_settings(settings: Settings) -> None:
    """Сохраняем настройки рядом с проектом."""
    CONFIG_PATH.write_text(
        json.dumps(settings.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_api_key() -> str:
    """Ключ DeepSeek из окружения (.env подхватывается при импорте)."""
    return (os.getenv("DEEPSEEK_API_KEY") or "").strip()


def save_api_key(key: str) -> None:
    """Записываем ключ в .env, не трогая остальные строки."""
    key = key.strip()
    lines: list[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

    for index, line in enumerate(lines):
        if line.strip().startswith("DEEPSEEK_API_KEY="):
            lines[index] = f"DEEPSEEK_API_KEY={key}"
            break
    else:
        lines.append(f"DEEPSEEK_API_KEY={key}")

    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ["DEEPSEEK_API_KEY"] = key


def mask_key(key: str) -> str:
    """Ключ в логах и подписях показываем частично."""
    if not key:
        return "не задан"
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}…{key[-4:]}"
