"""Фоновые задачи: запрос к модели, список голосов, озвучка, распознавание.

Qt-интерфейс нельзя блокировать сетью, синтезом и моделями, поэтому всё тяжёлое
живёт в QThread.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from app.agent import (
    AgentLimits,
    AgentRunner,
    McpLimits,
    McpServer,
    McpSession,
    ToolContext,
)
from app.asr.base import ASREngine, ASRError
from app.config import ASRSettings, TTSSettings
from app.llm_client import ChatClient, LLMError
from app.llm_probe import ProbeError, probe_models
from app.tts import TTSError, TTSService


class ChatWorker(QThread):
    """Стримит ответ модели и отдаёт куски текста в интерфейс."""

    chunk = Signal(str)
    reasoning = Signal(str)
    note = Signal(str)
    finished_ok = Signal(str, dict)
    failed = Signal(str)

    def __init__(self, client: ChatClient, messages: list[dict], parent=None) -> None:
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
        except LLMError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # на всякий случай: поток не должен умирать молча
            self.failed.emit(f"Непредвиденная ошибка: {exc}")

    def _attempt(self, client: ChatClient) -> tuple[str, dict, str | None]:
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


class AgentWorker(QThread):
    """Агентский цикл: стримит итоговый ответ и сообщает о вызовах инструментов."""

    chunk = Signal(str)
    reasoning = Signal(str)
    note = Signal(str)
    tool = Signal(str)
    finished_ok = Signal(str, dict)
    failed = Signal(str)

    def __init__(
        self,
        client: ChatClient,
        messages: list[dict],
        context: ToolContext,
        limits: AgentLimits,
        *,
        context_tokens: int,
        max_tokens: int,
        close_mcp: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._messages = messages
        self._context = context
        self._limits = limits
        self._context_tokens = context_tokens
        self._max_tokens = max_tokens
        self._close_mcp = close_mcp
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            runner = AgentRunner(
                self._client,
                self._messages,
                self._context,
                self._limits,
                context_tokens=self._context_tokens,
                max_tokens=self._max_tokens,
                cancel=self._cancel,
                on_content=self.chunk.emit,
                on_reasoning=self.reasoning.emit,
                on_note=self.note.emit,
                on_tool=self.tool.emit,
            )
            reply = runner.run()
        except LLMError as exc:
            if not self._cancel.is_set():
                self.failed.emit(str(exc))
            return
        except Exception as exc:  # поток не должен умирать молча
            if not self._cancel.is_set():
                self.failed.emit(f"Ошибка агента: {exc}")
            return
        finally:
            self._close_context()

        if reply is None or self._cancel.is_set():
            return
        if not reply.text.strip():
            self.failed.emit(
                "Модель не дала ответа. Попробуйте переформулировать вопрос "
                "или уменьшить объём чтения."
            )
            return
        self.finished_ok.emit(reply.text, reply.usage)

    def _close_context(self) -> None:
        """После ответа закрываем интернет; MCP — если он не живёт в окне.

        Тёплая MCP-сессия принадлежит окну: её закрывает оно само, когда
        меняются настройки серверов или MLVH выходит.
        """
        resources = [self._context.web]
        if self._close_mcp:
            resources.append(self._context.mcp)
        for resource in resources:
            if resource is None:
                continue
            try:
                resource.close()
            except Exception:  # закрытие не должно портить ответ
                pass


class McpProbeWorker(QThread):
    """Проверка одного MCP-сервера: запуск, рукопожатие, список инструментов."""

    done = Signal(str, list)
    failed = Signal(str, str)

    def __init__(self, server: dict, limits: McpLimits, parent=None) -> None:
        super().__init__(parent)
        self._server = dict(server) if isinstance(server, dict) else {}
        self._limits = limits

    def run(self) -> None:
        server = McpServer.from_dict(self._server)
        if server is None:
            self.failed.emit(
                str(self._server.get("id") or "MCP"), "нужны id и command"
            )
            return

        session = McpSession([server], self._limits, log_path=None)
        try:
            names = [
                item["function"]["name"]
                for item in session.schemas(self.isInterruptionRequested)
            ]
            if session.problems:
                self.failed.emit(server.id, "; ".join(session.problems))
            elif not names:
                self.failed.emit(server.id, "сервер не вернул ни одного инструмента")
            else:
                self.done.emit(server.id, names)
        except Exception as exc:  # поток не должен умирать молча
            self.failed.emit(server.id, str(exc))
        finally:
            session.close()


class ProviderProbeWorker(QThread):
    """Проверка провайдера модели: GET /models и список доступных алиасов."""

    done = Signal(str, list)
    failed = Signal(str)

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: float = 8.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout

    def run(self) -> None:
        try:
            url, models = probe_models(self._base_url, self._api_key, self._timeout)
        except ProbeError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # поток не должен умирать молча
            self.failed.emit(f"Проверка не удалась: {exc}")
        else:
            self.done.emit(url, models)


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
