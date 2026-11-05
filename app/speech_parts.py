"""Разбиение готового текста на части по языку и озвучке.

Кусок ответа бывает смешанным: «Курс Python — это язык программирования».
Локальные движки латиницу вырезают, поэтому текст режется по письменности,
а английские фрагменты решаются по режиму `tts.english_mode`:

| режим     | короткая вставка (1 слово)  | фраза из 2+ слов              |
|-----------|-----------------------------|-------------------------------|
| translit  | русскими буквами            | русскими буквами              |
| edge      | русскими буквами            | голосом Edge                  |
| silero_en | русскими буквами            | локальной v3_en               |
| auto      | русскими буквами (до 3 слов) | английским голосом            |
| off       | как было до 0.8 (слово теряется) | то же                     |

Одно короткое слово голос не меняет: смена тембра посреди русской фразы
раздражает сильнее лёгкого акцента в «пайтоне».

Порядок поиска произношения: словарь (`app/speech_terms.py`) → правила
транслита. Списанное по буквам и имена файлов вслух не читаем.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import ENGLISH_MODES, TTSSettings
from app.speech_terms import dictionary
from app.text_norm import (
    TOKEN,
    has_latin,
    is_probable_code,
    split_by_script,
    transliterate_word,
)

SHORT_CHARS = 24        # auto: вставка до 24 символов остаётся в русском голосе
SHORT_WORDS = 3         # и до трёх слов: «pull request», «Node.js», «GPT-4»
SINGLE_CHARS = 12       # edge/silero_en: одно короткое слово тоже по-русски


@dataclass
class SpeechPart:
    """Кусок текста и язык, которым его надо прочитать."""

    lang: str          # ru / en
    text: str


def resolve_mode(settings: TTSSettings) -> str:
    mode = str(getattr(settings, "english_mode", "auto") or "auto").strip().lower()
    return mode if mode in ENGLISH_MODES else "auto"


def build_parts(text: str, settings: TTSSettings) -> list[SpeechPart]:
    """Текст → части: русские читаем как есть, английские — по режиму."""
    if not text or not text.strip():
        return []
    mode = resolve_mode(settings)
    if mode == "off" or not has_latin(text):
        return [SpeechPart("ru", text)]

    parts: list[SpeechPart] = []
    pending = ""            # знаки, отставшие от английского фрагмента
    for lang, chunk in split_by_script(text):
        if lang != "en":
            _append(parts, "ru", pending + chunk)
            pending = ""
            continue
        if _route(chunk, mode) == "ru":
            # Вставка внутри русской фразы: переписываем кириллицей, голос
            # не меняется — «Python» звучит как «пайтон» голосом Ксении.
            _append(parts, "ru", pending + transliterate_english(chunk))
            pending = ""
        else:
            body, tail = _split_tail(chunk)
            _append(parts, "ru", pending)
            _append(parts, "en", body)
            pending = tail
    _append(parts, "ru", pending)
    return [
        SpeechPart(part.lang, re.sub(r"[ \t]{2,}", " ", part.text))
        for part in parts
        if part.text.strip()
    ]


def transliterate_english(text: str) -> str:
    """Английский фрагмент русскими буквами: словарь, потом правила.

    Пробелы по краям сохраняем: фрагмент вставляется в русскую фразу, и без
    них «Python — это» превратилось бы в «пайтон —это».
    """
    if not text:
        return ""
    core = text.strip()
    if not core:
        return text
    lead = text[: len(text) - len(text.lstrip())]
    tail = text[len(text.rstrip()):]

    terms = dictionary()
    prepared = terms.apply_phrases(core)

    def replace(match) -> str:
        token = match.group(0)
        known = terms.lookup(token)
        if known:
            return known
        if is_probable_code(token):
            return " "
        return transliterate_word(token)

    result = TOKEN.sub(replace, prepared)
    # Выкинутые имена файлов оставляют лишние пробелы и висячие запятые.
    result = re.sub(r"\s+([,.;:!?])", r"\1", result)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return lead + result + tail if result else lead + tail


def _split_tail(chunk: str) -> tuple[str, str]:
    """Отделяем только тире и пробелы: Edge не должен читать «Python —»."""
    body = chunk.rstrip(" \t—–-")
    return body, chunk[len(body):]


def english_words(text: str) -> list[str]:
    """Незнакомые термины в тексте — то, что стоит выучить."""
    return dictionary().missing(text)


def _route(chunk: str, mode: str) -> str:
    """Куда отдать английский фрагмент: в русский движок или английский."""
    if mode == "translit":
        return "ru"

    words = [match.group(0) for match in TOKEN.finditer(chunk)]
    body = " ".join(words)

    if mode in ("edge", "silero_en"):
        # Одно короткое слово посреди русской фразы: смена голоса на слове
        # звучит хуже, чем «пайтон» голосом Ксении. Длинное — в английский.
        if len(words) <= 1 and len(body) <= SINGLE_CHARS:
            return "ru"
        return "en"

    # auto: короткая вставка дешевле и приятнее в русском голосе.
    if len(words) <= SHORT_WORDS and len(body) <= SHORT_CHARS:
        return "ru"
    return "en"


def _append(parts: list[SpeechPart], lang: str, text: str) -> None:
    """Склеиваем соседние части одного языка, чтобы не звать движок зря."""
    if not text:
        return
    if parts and parts[-1].lang == lang:
        parts[-1].text = parts[-1].text + text
    else:
        parts.append(SpeechPart(lang, text))
