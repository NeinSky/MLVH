"""Агентский цикл: инструменты, лимиты и защита от зацикливания."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Callable

from app.agent.context import context_budget, trim_messages
from app.agent.policy import AgentLimits, Workspace
from app.agent.tools import TOOL_SCHEMAS, call_tool
from app.deepseek_client import DeepSeekClient, DeepSeekError, ToolCall

STOP_TEXT = (
    "Агент остановился: исчерпан лимит шагов или объёма данных. "
    "Уточните вопрос или уменьшите объём чтения."
)


@dataclass
class AgentReply:
    """Итог одного вопроса: текст, статистика и причина остановки."""

    text: str
    usage: dict
    rounds: int = 0
    tool_calls: int = 0
    stopped: str = "ok"


@dataclass
class _Round:
    text: str
    tool_calls: list[ToolCall]
    usage: dict
    finish: str | None


class AgentRunner:
    """Один вопрос → несколько шагов с инструментами → итоговый ответ.

    Все защиты продублированы: число шагов, повторные вызовы, общий счётчик
    вызовов, объём данных, время и отсутствие прогресса. При срабатывании
    делается один финальный запрос без инструментов, чтобы пользователь
    всё-таки получил ответ.
    """

    def __init__(
        self,
        client: DeepSeekClient,
        messages: list[dict],
        workspace: Workspace,
        limits: AgentLimits,
        *,
        context_tokens: int,
        max_tokens: int,
        cancel: threading.Event | None = None,
        on_content: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        on_note: Callable[[str], None] | None = None,
        on_tool: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._messages = [dict(message) for message in messages]
        self._workspace = workspace
        self._limits = limits
        self._budget = context_budget(context_tokens, max_tokens)
        self._cancel = cancel or threading.Event()
        self._clock = clock
        self._on_content = on_content or (lambda _text: None)
        self._on_reasoning = on_reasoning or (lambda _text: None)
        self._on_note = on_note or (lambda _text: None)
        self._on_tool = on_tool or (lambda _text: None)
        self._content_parts: list[str] = []

    # --- мелочи ----------------------------------------------------------
    def _note(self, text: str) -> None:
        self._on_note(text)

    def _partial_text(self) -> str:
        return "".join(self._content_parts).strip()

    def _limit_reason(
        self,
        rounds: int,
        calls: int,
        tool_chars: int,
        no_progress: int,
    ) -> tuple[str, str]:
        """(код, человеческая причина) или ("", "")."""
        if no_progress >= 2:
            return "stuck", "отсутствия прогресса"
        if rounds >= self._limits.max_rounds:
            return "rounds", "шагов"
        if calls >= self._limits.max_tool_calls:
            return "calls", "вызовов"
        if tool_chars >= self._limits.max_tool_chars:
            return "data", "объёма данных"
        return "", ""

    def _final_messages(self) -> list[dict]:
        """Тот же диалог, но с требованием ответить без новых инструментов."""
        messages = [dict(message) for message in self._messages]
        note = (
            "\n\nВАЖНО: лимит шагов с инструментами исчерпан. "
            "Ответь пользователю по уже прочитанному, без новых вызовов."
        )
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = str(messages[0].get("content") or "") + note
        else:
            messages.insert(0, {"role": "system", "content": note.strip()})
        return trim_messages(messages, self._budget).messages

    def _current_messages(self) -> list[dict]:
        """История + уже прочитанное, обрезанные под окно модели."""
        return trim_messages(self._messages, self._budget).messages

    @staticmethod
    def _add_usage(total: dict, part: dict) -> None:
        if not part:
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total[key] = int(total.get(key, 0)) + int(part.get(key, 0) or 0)
        # Для предупреждения о заполнении окна нужен максимум одного запроса,
        # а не сумма по всем шагам агента.
        total["max_prompt_tokens"] = max(
            int(total.get("max_prompt_tokens", 0) or 0),
            int(part.get("prompt_tokens", 0) or 0),
        )

    def _finish(
        self,
        text: str,
        usage: dict,
        rounds: int,
        calls: int,
        stopped: str,
    ) -> AgentReply:
        total = dict(usage)
        total["rounds"] = rounds
        total["tool_calls"] = calls
        total["stopped"] = stopped
        return AgentReply(
            text=text,
            usage=total,
            rounds=rounds,
            tool_calls=calls,
            stopped=stopped,
        )

    # --- основной цикл ---------------------------------------------------
    def run(self) -> AgentReply | None:
        """Возвращает AgentReply или None, если пользователь остановил запрос."""
        started = self._clock()
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        rounds = 0
        calls = 0
        tool_chars = 0
        duplicates: dict[str, int] = {}
        no_progress = 0
        forced = False
        retried_without_thinking = False

        while True:
            if self._cancel.is_set():
                return None
            if self._clock() - started > self._limits.max_seconds:
                self._note("Лимит времени исчерпан — останавливаюсь.")
                return self._finish(
                    self._partial_text() or STOP_TEXT, usage, rounds, calls, "time"
                )

            code, human = self._limit_reason(rounds, calls, tool_chars, no_progress)
            if code:
                if forced:
                    return self._finish(
                        self._partial_text() or STOP_TEXT, usage, rounds, calls, code
                    )
                forced = True
                self._note(f"Лимит {human}: отвечаю по уже прочитанному.")
                try:
                    final = self._request(self._final_messages(), None)
                except DeepSeekError:
                    return self._finish(
                        self._partial_text() or STOP_TEXT, usage, rounds, calls, code
                    )
                rounds += 1
                self._add_usage(usage, final.usage)
                text = final.text.strip() or self._partial_text() or STOP_TEXT
                return self._finish(text, usage, rounds, calls, code)

            try:
                result = self._request(self._current_messages(), TOOL_SCHEMAS)
            except DeepSeekError as exc:
                # Первый запрос не принял tools (например, прокси) — отвечаем без файлов.
                if calls == 0:
                    self._note(f"Инструменты недоступны ({exc}); отвечаю без файлов.")
                    result = self._request(self._current_messages(), None)
                else:
                    raise
            rounds += 1
            self._add_usage(usage, result.usage)
            if self._cancel.is_set():
                return None
            if result.finish == "length":
                usage["truncated"] = True

            if not result.tool_calls:
                text = result.text.strip()
                if not text and self._client.thinking and not retried_without_thinking:
                    retried_without_thinking = True
                    self._note("Пустой ответ — повторяю без рассуждений…")
                    try:
                        retry_client = self._client.clone(
                            thinking=False,
                            max_tokens=max(self._client.max_tokens, 1500),
                        )
                        result = self._request(
                            self._current_messages(),
                            TOOL_SCHEMAS,
                            client=retry_client,
                        )
                        rounds += 1
                        self._add_usage(usage, result.usage)
                        if result.finish == "length":
                            usage["truncated"] = True
                    except DeepSeekError:
                        pass
                    if self._cancel.is_set():
                        return None
                    text = result.text.strip()
                if not result.tool_calls:
                    if not text:
                        return self._finish(
                            self._partial_text() or STOP_TEXT,
                            usage,
                            rounds,
                            calls,
                            "empty",
                        )
                    return self._finish(text, usage, rounds, calls, "ok")

            # Модель попросила инструменты: assistant-сообщение добавляем целиком.
            self._messages.append(_assistant_message(result))
            executed = 0
            for call in result.tool_calls:
                if self._cancel.is_set():
                    return None
                calls += 1
                key = _call_key(call)
                duplicates[key] = duplicates.get(key, 0) + 1

                if duplicates[key] > self._limits.duplicate_limit:
                    content = (
                        "Ошибка: этот вызов уже повторялся. Смени подход "
                        "или ответь по уже прочитанному."
                    )
                    summary = f"повторный вызов {call.name} отклонён"
                elif calls > self._limits.max_tool_calls:
                    content = "Ошибка: исчерпан лимит вызовов инструментов."
                    summary = "лимит вызовов"
                elif tool_chars >= self._limits.max_tool_chars:
                    content = "Ошибка: исчерпан лимит объёма данных."
                    summary = "лимит данных"
                else:
                    tool_result = call_tool(
                        call.name,
                        call.arguments,
                        self._workspace,
                        self._cancel.is_set,
                    )
                    content = tool_result.content
                    summary = tool_result.summary
                    if tool_result.ok:
                        executed += 1
                    remaining = self._limits.max_tool_chars - tool_chars
                    if len(content) > remaining:
                        content = (
                            content[: max(0, remaining)]
                            + "\n[…обрезано: лимит данных]"
                        )
                    tool_chars += len(content)

                # На каждый tool_call обязателен ответ tool с тем же id.
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": content,
                    }
                )
                self._on_tool(summary)
                if self._cancel.is_set():
                    return None

            no_progress = 0 if executed else no_progress + 1

    # --- один запрос к модели --------------------------------------------
    def _request(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        client: DeepSeekClient | None = None,
    ) -> _Round:
        client = client or self._client
        parts: list[str] = []
        calls: list[ToolCall] = []
        usage: dict = {}
        finish: str | None = None

        for event in client.stream_chat(messages, tools=tools):
            if self._cancel.is_set():
                break
            if event.kind == "content":
                parts.append(event.text)
                self._content_parts.append(event.text)
                self._on_content(event.text)
            elif event.kind == "reasoning":
                self._on_reasoning(event.text)
            elif event.kind == "tool_call" and event.tool_call is not None:
                calls.append(event.tool_call)
            elif event.kind == "usage" and event.usage:
                usage = event.usage
            elif event.kind == "finish":
                finish = event.finish_reason

        return _Round("".join(parts), calls, usage, finish)


def _assistant_message(result: _Round) -> dict:
    """Сообщение assistant в том виде, в каком его ждёт API."""
    message: dict = {"role": "assistant", "content": result.text or None}
    if result.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in result.tool_calls
        ]
    return message


def _call_key(call: ToolCall) -> str:
    """Ключ для поиска повторов: имя + канонизированные аргументы."""
    try:
        args = json.loads(call.arguments or "{}")
        canonical = json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, ValueError):
        canonical = call.arguments or ""
    return f"{call.name}:{canonical}"
