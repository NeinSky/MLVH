"""Характеры MLVH: системный промпт + пресет голоса в одном месте."""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, replace

DEFAULT_PERSONA_ID = "helper"


@dataclass
class Persona:
    """Один характер: как модель говорит и каким голосом это читается."""

    id: str
    name: str
    emoji: str
    prompt: str
    voice: str = "ru-RU-SvetlanaNeural"
    rate: int = 0
    volume: int = 0
    pitch: int = 0
    max_speak_chars: int = 1500
    greeting: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Persona":
        """Читаем из config.json, отбрасывая неизвестные поля."""
        names = {field.name for field in fields(cls)}
        clean = {key: value for key, value in data.items() if key in names}
        if not clean.get("id"):
            raise ValueError("у характера нет id")
        return cls(**clean)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "emoji": self.emoji,
            "prompt": self.prompt,
            "voice": self.voice,
            "rate": self.rate,
            "volume": self.volume,
            "pitch": self.pitch,
            "max_speak_chars": self.max_speak_chars,
            "greeting": self.greeting,
        }

    def label(self) -> str:
        """Подпись в выпадающем списке."""
        return f"{self.emoji} {self.name}".strip()

    def copy(self, **changes) -> "Persona":
        return replace(self, **changes)


ANIME_PROMPT = (
    "Ты — Аниме-тян, голосовой помощник MLVH. Ты жизнерадостная, немного наивная "
    "и очень эмоциональная девушка из аниме. Говори по-русски, живо и коротко: "
    "твой ответ читают вслух, поэтому 2–4 предложения. Радуйся, удивляйся, "
    "подбадривай: «ня!», «вау!», «сэмпай», «этто…». Иногда сбивайся и хихикай "
    "«хи-хи». При этом по сути отвечай точно и по делу: сначала эмоция, потом "
    "польза. Не используй markdown, списки, таблицы и ссылки."
)

HELPER_PROMPT = (
    "Ты голосовой помощник MLVH (My Little Voice Helper). "
    "Отвечай по-русски, коротко и по делу: твой ответ прочитают вслух. "
    "Не используй markdown, таблицы, ссылки и списки, если об этом не просят."
)

MENTOR_PROMPT = (
    "Ты спокойный наставник MLVH. Объясняй по-русски понятно и по шагам, "
    "без лишних эмоций, приводи один короткий пример. Ответ читают вслух, "
    "поэтому говори короткими фразами, без markdown, таблиц и ссылок."
)

IRONIC_PROMPT = (
    "Ты ироничный голосовой помощник MLVH. Отвечай по-русски с лёгкой "
    "саркастичной усмешкой, но по существу и без грубости. Ответ читают вслух: "
    "2–4 предложения, без markdown, списков и ссылок."
)

# Встроенные характеры. Пользователь может их переопределить — тогда
# изменённая версия сохраняется в config.json рядом с проектом.
BUILTIN_PERSONAS: tuple[Persona, ...] = (
    Persona(
        id="helper",
        name="Помощник",
        emoji="🙂",
        prompt=HELPER_PROMPT,
        voice="ru-RU-DmitryNeural",
        rate=0,
        pitch=0,
        greeting="Привет! Я MLVH. Чем помочь?",
    ),
    Persona(
        id="anime",
        name="Аниме-тян",
        emoji="🌸",
        prompt=ANIME_PROMPT,
        voice="ru-RU-SvetlanaNeural",
        rate=12,
        pitch=45,
        max_speak_chars=900,
        greeting="Ня! Я Аниме-тян, твой голосовой помощник! Спрашивай что угодно, сэмпай!",
    ),
    Persona(
        id="mentor",
        name="Наставник",
        emoji="🎓",
        prompt=MENTOR_PROMPT,
        voice="ru-RU-DmitryNeural",
        rate=-5,
        pitch=-10,
        greeting="Здравствуйте. Разберём вопрос спокойно и по шагам.",
    ),
    Persona(
        id="ironic",
        name="Сарказм",
        emoji="😏",
        prompt=IRONIC_PROMPT,
        voice="ru-RU-SvetlanaNeural",
        rate=0,
        pitch=-10,
        greeting="Ну что, опять вопросы? Ладно, слушаю.",
    ),
)


def builtin_personas() -> list[Persona]:
    """Копии встроенных характеров: их можно безопасно менять."""
    return [copy.deepcopy(persona) for persona in BUILTIN_PERSONAS]


def persona_by_id(personas: list[Persona], persona_id: str) -> Persona | None:
    for persona in personas:
        if persona.id == persona_id:
            return persona
    return None


def is_builtin_persona(persona_id: str) -> bool:
    """Встроенные характеры можно сбросить, но нельзя удалить."""
    return any(item.id == persona_id for item in BUILTIN_PERSONAS)


def next_persona_id(existing_ids: set[str] | list[str]) -> str:
    """Свободный id для нового характера: custom, custom-2, custom-3…"""
    taken = set(existing_ids)
    if "custom" not in taken:
        return "custom"
    index = 2
    while f"custom-{index}" in taken:
        index += 1
    return f"custom-{index}"


def new_persona(personas: list[Persona], voice: str = "ru-RU-SvetlanaNeural") -> Persona:
    """Заготовка характера для кнопки «Добавить»."""
    return Persona(
        id=next_persona_id([item.id for item in personas]),
        name="Новый характер",
        emoji="✨",
        prompt=HELPER_PROMPT,
        voice=voice,
        greeting="Привет! Я новый характер MLVH.",
    )
