"""Словарь произношения английских терминов: поставляемый + выученный.

Локальные движки знают только кириллицу, поэтому латинские слова приходится
переписывать русскими буквами. Словарь отвечает на вопрос «как это звучит».

Два слоя:

- `app/data/speech_terms.json` — поставляемый, лежит в git и обновляется
  вместе с проектом;
- `speech_terms.local.json` рядом с `config.json` — то, что выучила модель,
  и ручные правки. Файл в `.gitignore`: у каждого свой.

Локальный слой перекрывает поставляемый: ручная правка сильнее машинной.

Поиск идёт по n-граммам от длинных к коротким, поэтому «pull request»
находится раньше, чем «pull». Индекс — обычный `dict` в памяти: на 5000
записей поиск занимает ~0,05 мкс, тогда как SQLite на том же объёме — ~43 мкс.
База данных здесь не нужна: словарь читается один раз при старте.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import BASE_DIR
from app.text_norm import is_probable_code, latin_tokens

CORE_PATH = BASE_DIR / "app" / "data" / "speech_terms.json"
LOCAL_PATH = BASE_DIR / "speech_terms.local.json"

SCHEMA = 1
MAX_ENTRIES = 5000        # дальше словарь только чистим, чтобы файл не пух
SAVE_DELAY = 1.5          # секунд: копим правки, пишем пачкой
MAX_PHRASE_WORDS = 3      # самые длинные фразы: «pull request», «user experience»
MIN_LENGTH = 1
MAX_LENGTH = 60

log = logging.getLogger(__name__)

_LATIN = re.compile(r"[A-Za-z]")
_CYRILLIC_ONLY = re.compile(r"^[а-яё]+(?:[ -][а-яё]+)*$")


def normalize_key(text: str) -> str:
    """Ключ для поиска: без регистра, разделители схлопнуты в пробел.

    «Node.js», «node js» и «node-js» дают один и тот же ключ.
    """
    cleaned = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", " ", text or "")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def is_valid_pronunciation(value: str) -> bool:
    """Проверяем ответ модели: только кириллица, без латиницы и мусора.

    Модель иногда возвращает перевод («ядро» вместо «кубернетес») — это ещё
    допустимо, а вот латиница или пустая строка означают, что ответ негоден.
    """
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not MIN_LENGTH <= len(text) <= MAX_LENGTH:
        return False
    if _LATIN.search(text):
        return False
    return bool(_CYRILLIC_ONLY.match(text))


@dataclass
class Entry:
    """Одно произношение: «kubernetes» → «кубернетес»."""

    ru: str
    source: str = "core"          # core / llm / user
    words: int = 1
    added: str = ""


@dataclass
class DictionaryStats:
    core: int = 0
    local: int = 0
    rejected: int = 0
    phrases: int = 0

    def line(self) -> str:
        return (
            f"словарь: {self.core} поставляемых, {self.local} своих, "
            f"фраз: {self.phrases}, отказов: {self.rejected}"
        )


@dataclass
class TermDictionary:
    """Поиск произношения с двумя слоями и отложенной записью."""

    core_path: Path = CORE_PATH
    local_path: Path = LOCAL_PATH

    _entries: dict[str, Entry] = field(default_factory=dict, init=False)
    _phrases: dict[str, Entry] = field(default_factory=dict, init=False)
    _rejected: dict[str, float] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _loaded: bool = field(default=False, init=False)
    _dirty: bool = field(default=False, init=False)
    _saved_at: float = field(default=0.0, init=False)
    _phrase_re: "re.Pattern[str] | None" = field(default=None, init=False)
    _phrase_count: int = field(default=-1, init=False)

    # --- загрузка --------------------------------------------------------
    def load(self) -> "TermDictionary":
        """Читаем оба слоя. Битый файл не должен ломать озвучку."""
        with self._lock:
            if self._loaded:
                return self
            self._entries.clear()
            self._phrases.clear()
            self._rejected.clear()
            self._phrase_re = None
            self._phrase_count = -1

            for path, source in ((self.core_path, "core"), (self.local_path, "local")):
                data = self._read(path)
                if data is not None:
                    self._apply(data, source)
            self._loaded = True
            return self

    @staticmethod
    def _read(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Словарь %s не прочитан: %s", path, exc)
            return None

    def _apply(self, data: dict, source: str) -> None:
        entries = data.get("entries") or {}
        if isinstance(entries, dict):
            for raw_key, raw_value in entries.items():
                key = normalize_key(str(raw_key))
                if not key:
                    continue
                if isinstance(raw_value, dict):
                    ru = str(raw_value.get("ru") or "")
                    origin = str(raw_value.get("src") or source)
                    added = str(raw_value.get("at") or "")
                else:
                    ru = str(raw_value or "")
                    origin = source
                    added = ""
                if not is_valid_pronunciation(ru):
                    log.warning("Словарь: пропускаю «%s» — негодное произношение", raw_key)
                    continue
                self._store(key, Entry(ru=ru.strip(), source=origin, added=added))

        rejected = data.get("rejected") or []
        if isinstance(rejected, dict):
            rejected = list(rejected)
        for word in rejected:
            key = normalize_key(str(word))
            if key:
                self._rejected[key] = 0.0

    def _store(self, key: str, entry: Entry) -> bool:
        entry.words = key.count(" ") + 1
        if entry.words > MAX_PHRASE_WORDS:
            log.warning("Словарь: «%s» длиннее %d слов — не сохраняю", key, MAX_PHRASE_WORDS)
            return False
        self._entries[key] = entry
        if entry.words > 1:
            self._phrases[key] = entry
        else:
            self._phrases.pop(key, None)
        return True

    # --- поиск -----------------------------------------------------------
    def lookup(self, word: str) -> str | None:
        """Произношение одного слова или фразы (без n-граммного перебора)."""
        self.load()
        entry = self._entries.get(normalize_key(word))
        return entry.ru if entry else None

    def get(self, key: str) -> Entry | None:
        self.load()
        return self._entries.get(normalize_key(key))

    def is_known(self, word: str) -> bool:
        return self.get(word) is not None

    def is_rejected(self, word: str) -> bool:
        self.load()
        return normalize_key(word) in self._rejected

    def missing(self, text: str) -> list[str]:
        """Незнакомые термины из текста — кандидаты на обучение."""
        self.load()
        found: list[str] = []
        seen: set[str] = set()
        for token in latin_tokens(text):
            key = normalize_key(token)
            if not key or key in seen:
                continue
            if key in self._entries or key in self._rejected:
                continue
            if is_probable_code(token):
                continue
            seen.add(key)
            found.append(token)
        return found

    def apply_phrases(self, text: str) -> str:
        """Заменяем известные фразы целиком: «pull request» → «пулл реквест».

        Иначе фраза распадётся на два слова: «pull» и «request» по отдельности
        звучат хуже, а «Node.js» вообще перестанет находиться.
        """
        self.load()
        if not text:
            return text
        pattern = self._phrase_pattern()
        if pattern is None:
            return text

        def replace(match: "re.Match[str]") -> str:
            entry = self._phrases.get(normalize_key(match.group(0)))
            return entry.ru if entry else match.group(0)

        return pattern.sub(replace, text)

    def _phrase_pattern(self) -> "re.Pattern[str] | None":
        with self._lock:
            if self._phrase_re is not None and self._phrase_count == len(self._phrases):
                return self._phrase_re
            keys = sorted(self._phrases, key=len, reverse=True)
            pattern = None
            if keys:
                parts = []
                for key in keys:
                    body = r"[\s._\-]+".join(re.escape(word) for word in key.split())
                    parts.append(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])")
                pattern = re.compile("|".join(parts), re.IGNORECASE)
            self._phrase_re = pattern
            self._phrase_count = len(self._phrases)
            return pattern

    # --- запись ----------------------------------------------------------
    def remember(self, word: str, pronunciation: str, source: str = "llm") -> bool:
        """Запоминаем произношение. Негодные ответы модели не сохраняем."""
        if not is_valid_pronunciation(pronunciation):
            return False
        key = normalize_key(word)
        if not key:
            return False

        self.load()
        with self._lock:
            existing = self._entries.get(key)
            value = pronunciation.strip()
            if existing is not None and existing.ru == value and existing.source != "core":
                return False
            if existing is not None and existing.source == "user":
                return False          # ручную правку не перетираем
            if existing is None and len(self._entries) >= MAX_ENTRIES:
                log.warning("Словарь заполнен (%d) — «%s» не сохранён", MAX_ENTRIES, word)
                return False
            stored = self._store(
                key, Entry(ru=value, source=source, added=time.strftime("%Y-%m-%d"))
            )
            if not stored:
                return False
            self._rejected.pop(key, None)
            self._dirty = True
        self.flush()
        return True

    def reject(self, word: str) -> None:
        """Помечаем слово как «учили, но не вышло» — второй раз не спрашиваем."""
        key = normalize_key(word)
        if not key:
            return
        self.load()
        with self._lock:
            self._rejected[key] = time.time()
            self._dirty = True
        self.flush()

    def forget(self, word: str) -> bool:
        """Убираем слово из локального слоя, чтобы вернуться к поставляемому."""
        key = normalize_key(word)
        self.load()
        with self._lock:
            removed = self._entries.pop(key, None) is not None
            self._phrases.pop(key, None)
            if self._rejected.pop(key, None) is not None:
                removed = True
            self._dirty = self._dirty or removed
        self.flush(force=True)
        return removed

    def flush(self, force: bool = False) -> None:
        """Пишем локальный слой: с задержкой, атомарно, в том же потоке."""
        with self._lock:
            if not self._dirty:
                return
            if not force and time.monotonic() - self._saved_at < SAVE_DELAY:
                return
            payload = {
                "schema": SCHEMA,
                "entries": {
                    key: {"ru": entry.ru, "src": entry.source, "at": entry.added}
                    for key, entry in self._entries.items()
                    if entry.source != "core"
                },
                "rejected": sorted(self._rejected),
            }
            self._dirty = False
            self._saved_at = time.monotonic()

        try:
            data = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True)
            tmp = self.local_path.with_suffix(".tmp")
            tmp.write_text(data, encoding="utf-8")
            tmp.replace(self.local_path)
        except OSError as exc:
            log.warning("Не удалось сохранить словарь %s: %s", self.local_path, exc)

    # --- для инструментов и интерфейса -----------------------------------
    def stats(self) -> DictionaryStats:
        self.load()
        local = sum(1 for entry in self._entries.values() if entry.source != "core")
        return DictionaryStats(
            core=len(self._entries) - local,
            local=local,
            rejected=len(self._rejected),
            phrases=len(self._phrases),
        )

    def entries(self) -> dict[str, Entry]:
        self.load()
        return dict(self._entries)


_TERMS: TermDictionary | None = None


def dictionary() -> TermDictionary:
    """Общий словарь на процесс: файлы читаются один раз."""
    global _TERMS
    if _TERMS is None:
        _TERMS = TermDictionary().load()
    return _TERMS

