"""Фоновые задачи: запрос к DeepSeek, список голосов, озвучка, распознавание.

Qt-интерфейс нельзя блокировать сетью, синтезом и моделями, поэтому всё тяжёлое
живёт в QThread.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from app.asr.base import ASREngine, ASRError
from app.config import ASRSettings, TTSSettings
from app.deepseek_client import DeepSeekClient, DeepSeekError
from app.tts import TTSError, TTSService


class ChatWorker(QThread):
    """Стримит ответ модели и отдаёт куски текста в интерфейс."""

    chunk = Signal(str)
    reasoning = Signal(str)
    note = Signal(str)
    finished_ok = Signal(str, dict)
    failed = Signal(str)

    def __init__(self, client: DeepSeekClient, messages: list[dict], parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._messages = messages
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            text, usage, finish = self._attempt(self._client)

            # Рассуждения могут съесть весь бюджет токенов: тогда content пуст.
            # Повторяем без режима рассуждений — он дороже по токенам.
            if not text.strip() and not self._cancel.is_set():
                self.note.emit("Пустой ответ — повторяю без рассуждений…")
                retry = self._client.clone(
                    thinking=False,
                    max_tokens=max(self._client.max_tokens, 1500),
                )
                text, retry_usage, finish = self._attempt(retry)
                usage = retry_usage or usage

            if not text.strip():
                if self._cancel.is_set():
                    return
                self.failed.emit(
                    "Модель вернула пустой ответ. Скорее всего, лимит токенов ушёл на "
                    "рассуждения — увеличьте «max_tokens» или выключите «Режим рассуждений»."
                )
                return

            if finish == "length":
                usage["truncated"] = True
            self.finished_ok.emit(text, usage)
        except DeepSeekError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # на всякий случай: поток не должен умирать молча
            self.failed.emit(f"Непредвиденная ошибка: {exc}")

    def _attempt(self, client: DeepSeekClient) -> tuple[str, dict, str | None]:
        """Один запрос к модели: собираем текст, статистику и причину остановки."""
        parts: list[str] = []
        usage: dict = {}
        finish: str | None = None
        for event in client.stream_chat(self._messages):
            if self._cancel.is_set():
                break
            if event.kind == "content":
                parts.append(event.text)
                self.chunk.emit(event.text)
            elif event.kind == "reasoning":
                self.reasoning.emit(event.text)
            elif event.kind == "finish":
                finish = event.finish_reason
            elif event.kind == "usage" and event.usage:
                usage = event.usage
        return "".join(parts), usage, finish


class PrepareWorker(QThread):
    """Готовит движок распознавания заранее: грузит модель без запроса."""

    ready = Signal(str)
    failed = Signal(str)

    def __init__(self, engine: ASREngine, settings: ASRSettings, parent=None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._settings = settings

    def run(self) -> None:
        try:
            self._engine.prepare(self._settings)
            device = getattr(self._engine, "device_name", "")
            self.ready.emit(device or "готово")
        except ASRError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Модель не загрузилась: {exc}")


class TranscribeWorker(QThread):
    """Распознавание в фоне: при первом запуске грузится модель."""

    note = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, engine: ASREngine, samples, settings: ASRSettings, parent=None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._samples = samples
        self._settings = settings

    def run(self) -> None:
        try:
            self.note.emit("Распознаю…")
            self._engine.prepare(self._settings)
            text = self._engine.transcribe(self._samples, self._settings)
            self.done.emit(text)
        except ASRError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # поток не должен умирать молча
            self.failed.emit(f"Ошибка распознавания: {exc}")


class PlayFileWorker(QThread):
    """Проигрывание готового файла: проверка записи с микрофона."""

    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, tts: TTSService, path: Path, volume: float = 0.9, parent=None) -> None:
        super().__init__(parent)
        self._tts = tts
        self._path = path
        self._volume = volume

    def run(self) -> None:
        try:
            self._tts.play(self._path, self._volume)
            self.finished_ok.emit()
        except TTSError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Ошибка воспроизведения: {exc}")


class VoiceListWorker(QThread):
    """Загружает список голосов Edge, чтобы не подвешивать окно."""

    loaded = Signal(list)
    failed = Signal(str)

    def __init__(self, locale_prefix: str = "", parent=None) -> None:
        super().__init__(parent)
        self._locale_prefix = locale_prefix

    def run(self) -> None:
        try:
            self.loaded.emit(TTSService.list_voices(self._locale_prefix))
        except TTSError as exc:
            self.failed.emit(str(exc))


class SpeechSignals(QObject):
    """Мост из потоков озвучки в GUI: сигналы Qt доставляются в главный поток."""

    done = Signal()
    error = Signal(str)
    note = Signal(str)
    prepare_done = Signal(str)
    prepare_failed = Signal(str)


class SpeakWorker(QThread):
    """Синтез и проигрывание одного ответа."""

    started = Signal()
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, tts: TTSService, text: str, settings: TTSSettings, parent=None) -> None:
        super().__init__(parent)
        self._tts = tts
        self._text = text
        self._settings = settings

    def run(self) -> None:
        try:
            self.started.emit()
            self._tts.speak(self._text, self._settings)
            self.finished_ok.emit()
        except TTSError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Ошибка озвучки: {exc}")
