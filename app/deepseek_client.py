"""Тонкая обёртка над DeepSeek API (совместим с официальным SDK openai)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import openai
from openai import Client

BASE_URL = "https://api.deepseek.com"

log = logging.getLogger(__name__)


class DeepSeekError(RuntimeError):
    """Ошибка с текстом, который не стыдно показать пользователю."""


@dataclass
class ToolCall:
    """Запрошенный моделью вызов инструмента."""

    id: str
    name: str
    arguments: str


@dataclass
class StreamEvent:
    """Один кусочек ответа: текст, рассуждение, инструмент, статистика или стоп."""

    kind: str  # content | reasoning | tool_call | usage | finish
    text: str = ""
    usage: dict | None = None
    finish_reason: str | None = None
    tool_call: ToolCall | None = None


class ToolCallAccumulator:
    """Склеивает фрагменты `delta.tool_calls`: они приходят по частям, по index."""

    def __init__(self) -> None:
        self._items: dict[int, dict] = {}

    def feed(self, deltas) -> None:
        for delta in deltas or []:
            index = int(getattr(delta, "index", 0) or 0)
            item = self._items.setdefault(
                index, {"id": "", "name": "", "arguments": ""}
            )
            call_id = getattr(delta, "id", None)
            if call_id:
                item["id"] = str(call_id)
            function = getattr(delta, "function", None)
            if function is None:
                continue
            name = getattr(function, "name", None)
            if name:
                item["name"] += str(name)
            arguments = getattr(function, "arguments", None)
            if arguments:
                item["arguments"] += str(arguments)

    def finish(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index in sorted(self._items):
            item = self._items[index]
            calls.append(
                ToolCall(
                    id=item["id"] or f"call_{index}",
                    name=item["name"],
                    arguments=item["arguments"] or "{}",
                )
            )
        return calls


class DeepSeekClient:
    """Клиент чата DeepSeek. Держит один HTTP-клиент на один диалог."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-flash",
        temperature: float = 0.7,
        max_tokens: int = 1200,
        thinking: bool = False,
        timeout: float = 90.0,
    ) -> None:
        if not api_key:
            raise DeepSeekError(
                "Не задан ключ DeepSeek. Нажмите «Ввести ключ» в окне MLVH "
                "или положите DEEPSEEK_API_KEY в файл .env."
            )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self._api_key = api_key
        self._client = Client(
            api_key=api_key,
            base_url=BASE_URL,
            timeout=timeout,
            max_retries=1,
        )

    def clone(self, *, thinking: bool | None = None, max_tokens: int | None = None) -> "DeepSeekClient":
        """Копия клиента с другими параметрами — для повторного запроса."""
        return DeepSeekClient(
            api_key=self._api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            thinking=self.thinking if thinking is None else thinking,
        )

    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> Iterator[StreamEvent]:
        """Стримим ответ. Ошибки приводим к DeepSeekError.

        Если передан `tools`, модель может попросить инструменты: в конце
        потока отдаём готовые `tool_call`-события.
        """
        params: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            params["tools"] = tools
        if self.thinking:
            # Включаем режим рассуждений (deepseek-flash / deepseek-v4-pro).
            params["extra_body"] = {"thinking": {"type": "enabled"}}

        accumulator = ToolCallAccumulator()
        try:
            stream = self._client.chat.completions.create(**params)
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    yield StreamEvent(kind="usage", usage=_usage_to_dict(chunk.usage))
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    yield StreamEvent(kind="finish", finish_reason=choice.finish_reason)
                delta = choice.delta
                if delta is None:
                    continue
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield StreamEvent(kind="reasoning", text=reasoning)
                tool_deltas = getattr(delta, "tool_calls", None)
                if tool_deltas:
                    accumulator.feed(tool_deltas)
                if delta.content:
                    yield StreamEvent(kind="content", text=delta.content)
            for call in accumulator.finish():
                yield StreamEvent(kind="tool_call", tool_call=call)
        except openai.AuthenticationError as exc:
            raise DeepSeekError(
                "DeepSeek отклонил ключ (401). Проверьте DEEPSEEK_API_KEY в .env."
            ) from exc
        except openai.PermissionDeniedError as exc:
            raise DeepSeekError("У ключа нет доступа к этой модели (403).") from exc
        except openai.RateLimitError as exc:
            raise DeepSeekError("Слишком много запросов к DeepSeek (429). Подождите немного.") from exc
        except openai.APITimeoutError as exc:
            raise DeepSeekError("DeepSeek не ответил вовремя. Попробуйте ещё раз.") from exc
        except openai.APIConnectionError as exc:
            raise DeepSeekError(
                "Нет связи с api.deepseek.com. Проверьте интернет, VPN или прокси."
            ) from exc
        except openai.APIStatusError as exc:
            if exc.status_code == 402:
                raise DeepSeekError(
                    "На балансе DeepSeek закончились средства (402). "
                    "Пополните счёт на platform.deepseek.com."
                ) from exc
            raise DeepSeekError(f"DeepSeek вернул ошибку {exc.status_code}: {exc.message}") from exc
        except openai.APIError as exc:
            raise DeepSeekError(f"Ошибка DeepSeek API: {exc}") from exc
        except Exception as exc:  # сеть может упасть как угодно, UI не должен падать
            log.exception("Неожиданная ошибка при запросе к DeepSeek")
            raise DeepSeekError(f"Не удалось получить ответ: {exc}") from exc


def _usage_to_dict(usage: object) -> dict:
    """Приводим объект usage к обычному словарю."""
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
