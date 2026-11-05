"""Обучение произношению незнакомых терминов через провайдера модели.

Правила транслита дают 8 точных совпадений из 20 IT-терминов — на незнакомых
словах этого мало. Поэтому модель добирает хвост: спрашиваем батчем «как это
звучит по-русски» и складываем ответ в локальный словарь.

Про токены и ожидание:

- обучение включается только явным флагом `tts.learn_terms` — это запросы к
  провайдеру за деньги пользователя;
- ожидание ограничено `tts.learn_timeout` (30 с по умолчанию): не успели —
  читаем по правилам, слово доучится в следующий раз;
- после `tts.learn_fail_limit` неудач подряд обучение выключается до
  перезапуска, иначе при лежащей сети каждый ответ стоил бы полминуты тишины;
- `thinking` выключаем явно: `deepseek-flash` по умолчанию рассуждает и на
  коротком запросе возвращает пустой ответ за 13 с вместо JSON за 1 с.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading

from app.config import ModelSettings, TTSSettings, get_api_key
from app.llm_client import ChatClient, LLMError
from app.speech_terms import dictionary, is_valid_pronunciation, normalize_key

BATCH_SIZE = 50          # слов в одном запросе: 51 слово — 610 токенов на выход
BATCH_TOKENS = 1600

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты помогаешь синтезу речи читать английские технические термины русским "
    "голосом. Для каждого слова дай произношение русскими буквами так, как его "
    "говорит русскоязычный инженер вслух: Python → пайтон, GitHub → гитхаб, "
    "Kubernetes → кубернетес, gRPC → джи ар пи си. Только кириллица, без "
    "латиницы и без пояснений. Ответ — один JSON-объект вида "
    '{"слово": "произношение"} на все слова сразу.'
)

_FENCE = re.compile(r"^\s*```(?:json)?|```\s*$", re.MULTILINE)


def parse_answer(raw: str) -> dict[str, str]:
    """Разбираем ответ модели: JSON в тексте, ключи нормализованы."""
    text = _FENCE.sub("", raw or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Модель не вернула JSON с произношением")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ответ модели не разобрать: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Модель вернула не объект")

    result: dict[str, str] = {}
    for key, value in data.items():
        normalized = normalize_key(str(key))
        if normalized and isinstance(value, str):
            result[normalized] = value.strip()
    return result


class TermLearner:
    """Фоновое обучение: очередь слов, один рабочий поток, предохранитель."""

    def __init__(
        self,
        model: ModelSettings,
        settings: TTSSettings,
        on_note=None,
    ) -> None:
        self._model = model
        self._settings = settings
        self._on_note = on_note
        self._queue: queue.Queue[list[str]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._pending: set[str] = set()
        self._failures = 0
        self._disabled = False

    # --- состояние -------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """Флаг в настройках и предохранитель после серии неудач."""
        return bool(self._settings.learn_terms) and not self._disabled

    def _note(self, message: str) -> None:
        if self._on_note is not None:
            try:
                self._on_note(message)
            except Exception:
                pass

    # --- постановка задачи -----------------------------------------------
    def request(self, words: list[str]) -> None:
        """Ставим слова в очередь. Синтез не ждёт — ждём только через `wait`."""
        if not words or not self.enabled:
            return

        known = dictionary()
        with self._lock:
            fresh: list[str] = []
            for word in words:
                key = normalize_key(word)
                if not key or key in self._pending:
                    continue
                if known.is_known(key) or known.is_rejected(key):
                    continue
                self._pending.add(key)
                fresh.append(word)
            if not fresh:
                return
            self._queue.put(fresh)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="mlvh-terms", daemon=True
                )
                self._thread.start()

    def wait(self, timeout: float | None = None) -> None:
        """Ждём, пока текущий батч закончится, но не дольше `timeout`."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    # --- работа ----------------------------------------------------------
    def _run(self) -> None:
        while True:
            try:
                words = self._queue.get_nowait()
            except queue.Empty:
                return
            if not self.enabled:
                self._release(words)
                return
            try:
                self._learn(words)
            except Exception as exc:          # поток не должен умирать молча
                log.warning("Обучение терминов упало: %s", exc)
            finally:
                self._release(words)

    def _release(self, words: list[str]) -> None:
        with self._lock:
            for word in words:
                self._pending.discard(normalize_key(word))

    def _learn(self, words: list[str]) -> None:
        for start in range(0, len(words), BATCH_SIZE):
            batch = words[start:start + BATCH_SIZE]
            try:
                answer = self._ask(batch)
            except (LLMError, ValueError) as exc:
                self._fail(str(exc))
                return

            known = dictionary()
            learned: list[str] = []
            for word in batch:
                value = answer.get(normalize_key(word))
                if value is None:
                    continue
                if is_valid_pronunciation(value):
                    if known.remember(word, value, source="llm"):
                        learned.append(f"{word} → {value}")
                else:
                    # Латиница или перевод с латиницей: второй раз не спрашиваем.
                    known.reject(word)
                    log.info("Термин «%s» выучен негодно: %r", word, value)

            self._failures = 0
            if learned:
                log.info("Выучил термины: %s", ", ".join(learned))
                tail = "…" if len(learned) > 6 else ""
                self._note("Выучил: " + ", ".join(learned[:6]) + tail)

        # Пишем сразу: иначе последние слова батча ждали бы дебаунса и могли
        # потеряться при выходе.
        dictionary().flush(force=True)

    def _fail(self, message: str) -> None:
        self._failures += 1
        limit = max(1, int(self._settings.learn_fail_limit))
        log.warning("Обучение терминов: %s (неудача %d из %d)", message, self._failures, limit)
        if self._failures >= limit:
            self._disabled = True
            self._note(
                "Обучение терминов выключено до перезапуска: провайдер не отвечает. "
                "Незнакомые слова читаю по правилам."
            )

    def _ask(self, words: list[str]) -> dict[str, str]:
        """Один запрос: список слов → {ключ: произношение}."""
        client = ChatClient(
            api_key=get_api_key(self._model),
            model=self._model.name,
            temperature=0.0,
            max_tokens=BATCH_TOKENS,
            thinking=False,
            timeout=max(5.0, float(self._settings.learn_timeout)),
            base_url=self._model.endpoint(),
            thinking_style=self._model.thinking_style,
            key_env=self._model.key_env(),
            max_retries=0,      # иначе таймаут повторяется и превращается в двойной
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ", ".join(words)},
        ]
        parts: list[str] = []
        for event in client.stream_chat(messages):
            if event.kind == "content":
                parts.append(event.text)
        return parse_answer("".join(parts))

