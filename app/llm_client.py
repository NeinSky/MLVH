"""Клиент модели: OpenAI-совместимый API (DeepSeek по умолчанию).

Стриминг, инструменты (tools), рассуждения и понятные тексты ошибок.
Адрес API задаётся `base_url`: пусто — DeepSeek, иначе OpenRouter, Groq,
Ollama, LM Studio, vLLM или корпоративный шлюз с тем же протоколом.
"""

from __future__ import annotations

import importlib
import logging
import urllib.parse
from dataclasses import dataclass
from typing import Iterator

import openai
from openai import Client

from app.config import is_local_host

BASE_URL = "https://api.deepseek.com"
DEFAULT_KEY_ENV = "DEEPSEEK_API_KEY"

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Ошибка с текстом, который не стыдно показать пользователю.

    `status` — HTTP-код, если ошибку вернул провайдер (401, 402, 404, 400…).
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


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


class ChatClient:
    """Клиент чата. Держит один HTTP-клиент на один диалог.

    Рассуждения DeepSeek включает через `extra_body={"thinking": ...}`. Этот
    параметр понимает только DeepSeek, остальные провайдеры отвечают на него
    400, поэтому `thinking_style` решает, отправлять ли его:
    `auto` — только DeepSeek, `none` — никогда, `deepseek` — всегда.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-flash",
        temperature: float = 0.7,
        max_tokens: int = 1200,
        thinking: bool = False,
        timeout: float = 90.0,
        base_url: str = "",
        thinking_style: str = "auto",
        key_env: str = "",
    ) -> None:
        self.base_url = (base_url or "").strip().rstrip("/") or BASE_URL
        self.thinking_style = (thinking_style or "auto").strip().casefold()
        self.key_env = key_env or self._default_key_env()
        self.model = str(model or "").strip()
        self.temperature = _clamp_float(temperature, 0.0, 2.0, 0.7)
        self.max_tokens = max(1, _clamp_int(max_tokens, 1200))
        self.thinking = thinking
        self._api_key = api_key
        self._timeout = timeout
        # Некоторые локальные серверы не знают stream_options — выключаем после 400.
        self._stream_options = True

        if not self.model:
            raise LLMError(
                "Не задан алиас модели. Укажите его в окне «Ввести ключ» "
                "или в поле model.name файла config.json."
            )
        if not api_key:
            raise LLMError(
                f"Не задан ключ {self.label}. Нажмите «Ввести ключ» в окне MLVH "
                f"или положите {self.key_env} в файл .env."
            )
        self._client = Client(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=1,
        )

    # --- провайдер -------------------------------------------------------
    @property
    def host(self) -> str:
        """Хост API без схемы и пути: «api.deepseek.com», «openrouter.ai»."""
        return urllib.parse.urlsplit(self.base_url).netloc or self.base_url

    @property
    def is_deepseek(self) -> bool:
        host = self.host.casefold()
        return host == "deepseek.com" or host.endswith(".deepseek.com")

    @property
    def label(self) -> str:
        """Имя провайдера для текстов: DeepSeek или хост."""
        return "DeepSeek" if self.is_deepseek else self.host

    @property
    def is_local(self) -> bool:
        """Локальный сервер (Ollama, LM Studio, vLLM): системный прокси не нужен."""
        return is_local_host(self.host)

    def _request_client(self) -> tuple[Client, Client | None]:
        """HTTP-клиент на один запрос: локальному серверу отключаем прокси.

        Иначе прокси из системы (он же виноват в 503) перехватывает запрос
        к 127.0.0.1. Возвращаем клиент и его же, если его нужно закрыть.
        """
        if not self.is_local:
            return self._client, None

        http = _http_library().Client(timeout=self._timeout, trust_env=False)
        client = Client(
            api_key=self._api_key,
            base_url=self.base_url,
            http_client=http,
            max_retries=1,
        )
        return client, client

    def _default_key_env(self) -> str:
        return DEFAULT_KEY_ENV if self.is_deepseek else "OPENAI_API_KEY"

    def thinking_payload(self) -> dict | None:
        """`extra_body` для режима рассуждений; None — не отправлять."""
        if not self.thinking:
            return None
        if self.thinking_style == "none":
            return None
        if self.thinking_style == "deepseek" or self.is_deepseek:
            return {"thinking": {"type": "enabled"}}
        return None

    def clone(self, *, thinking: bool | None = None, max_tokens: int | None = None) -> "ChatClient":
        """Копия клиента с другими параметрами — для повторного запроса."""
        return ChatClient(
            api_key=self._api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            thinking=self.thinking if thinking is None else thinking,
            timeout=self._timeout,
            base_url=self.base_url,
            thinking_style=self.thinking_style,
            key_env=self.key_env,
        )

    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> Iterator[StreamEvent]:
        """Стримим ответ. Ошибки приводим к LLMError.

        Если передан `tools`, модель может попросить инструменты: в конце
        потока отдаём готовые `tool_call`-события.
        """
        params = self._params(messages, tools)
        emitted = False
        try:
            for event in self._stream(params):
                emitted = True
                yield event
        except LLMError as exc:
            # Провайдер не знает stream_options (обычно локальный сервер):
            # повторяем один раз без него, если ещё ничего не отдали.
            if emitted or exc.status != 400 or "stream_options" not in params:
                raise
            log.info("%s отверг stream_options, повторяю без него", self.label)
            self._stream_options = False
            params.pop("stream_options", None)
            for event in self._stream(params):
                yield event

    def _params(self, messages: list[dict], tools: list[dict] | None) -> dict:
        params: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            params["tools"] = tools
        thinking = self.thinking_payload()
        if thinking is not None:
            params["extra_body"] = thinking
        if self._stream_options:
            # Без этого флага часть провайдеров не присылает usage вовсе.
            params["stream_options"] = {"include_usage": True}
        return params

    def _stream(self, params: dict) -> Iterator[StreamEvent]:
        accumulator = ToolCallAccumulator()
        client, owned = self._request_client()
        try:
            stream = client.chat.completions.create(**params)
            for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage:
                    yield StreamEvent(kind="usage", usage=_usage_to_dict(usage))
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                finish = getattr(choice, "finish_reason", None)
                if finish:
                    yield StreamEvent(kind="finish", finish_reason=str(finish))
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                reasoning = _reasoning_text(delta)
                if reasoning:
                    yield StreamEvent(kind="reasoning", text=reasoning)
                tool_deltas = getattr(delta, "tool_calls", None)
                if tool_deltas:
                    accumulator.feed(tool_deltas)
                content = getattr(delta, "content", None)
                if content:
                    yield StreamEvent(kind="content", text=str(content))
            for call in accumulator.finish():
                yield StreamEvent(kind="tool_call", tool_call=call)
        except openai.AuthenticationError as exc:
            raise LLMError(
                f"{self.label} отклонил ключ (401). Проверьте {self.key_env} в .env.",
                401,
            ) from exc
        except openai.PermissionDeniedError as exc:
            raise LLMError(
                f"У ключа нет доступа к модели «{self.model}» ({self.label}, 403).",
                403,
            ) from exc
        except openai.NotFoundError as exc:
            raise LLMError(
                f"Модель «{self.model}» не найдена у {self.label} (404). "
                "Проверьте алиас модели и адрес API.",
                404,
            ) from exc
        except openai.RateLimitError as exc:
            raise LLMError(
                f"Слишком много запросов к {self.label} (429). Подождите немного.",
                429,
            ) from exc
        except openai.APITimeoutError as exc:
            raise LLMError(f"{self.label} не ответил вовремя. Попробуйте ещё раз.") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(
                f"Нет связи с {self.host}. Проверьте интернет, VPN или прокси."
            ) from exc
        except openai.BadRequestError as exc:
            raise LLMError(
                f"{self.label} отклонил запрос (400): {_api_message(exc)}",
                400,
            ) from exc
        except openai.APIStatusError as exc:
            if exc.status_code == 402:
                hint = (
                    " Пополните счёт на platform.deepseek.com."
                    if self.is_deepseek
                    else ""
                )
                raise LLMError(
                    f"На балансе {self.label} закончились средства (402).{hint}",
                    402,
                ) from exc
            raise LLMError(
                f"{self.label} вернул ошибку {exc.status_code}: {_api_message(exc)}",
                exc.status_code,
            ) from exc
        except openai.APIError as exc:
            raise LLMError(f"Ошибка {self.label} API: {exc}") from exc
        except Exception as exc:  # сеть может упасть как угодно, UI не должен падать
            log.exception("Неожиданная ошибка при запросе к %s", self.label)
            raise LLMError(f"Не удалось получить ответ: {exc}") from exc
        finally:
            if owned is not None:
                try:
                    owned.close()
                except Exception:  # закрытие не должно прятать исходную ошибку
                    pass


# До 0.7.0 клиент умел только DeepSeek — старые имена оставляем рабочими.
DeepSeekClient = ChatClient
DeepSeekError = LLMError


def _http_library():
    """HTTP-библиотека установленного openai: `httpx` в 1.x, `httpx2` в 3.x.

    Отдельного `httpx` может не быть вовсе — в CI ставится только сам openai
    со своими зависимостями, а с 3.0 он собран на `httpx2`. Поэтому модуль
    спрашиваем у SDK, а не импортируем по имени: иначе `import httpx` падает
    с ModuleNotFoundError ещё до первого запроса.
    """
    try:
        from openai import _base_client
    except ImportError:  # перепакованный SDK без внутренних модулей
        _base_client = None
    if _base_client is not None:
        for name in ("httpx", "httpx2"):
            module = getattr(_base_client, name, None)
            if module is not None and hasattr(module, "Client"):
                return module

    for name in ("httpx", "httpx2"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(module, "Client"):
            return module
    raise LLMError(
        "У пакета openai нет HTTP-клиента (httpx2/httpx). "
        "Переустановите его: pip install --force-reinstall openai"
    )


def _reasoning_text(delta: object) -> str:
    """Рассуждения: у DeepSeek это `reasoning_content`, у других — `reasoning`."""
    for name in ("reasoning_content", "reasoning"):
        value = getattr(delta, name, None)
        if value:
            return str(value)
    extra = getattr(delta, "model_extra", None)
    if isinstance(extra, dict):
        for name in ("reasoning_content", "reasoning"):
            value = extra.get(name)
            if value:
                return str(value)
    return ""


def _api_message(exc: object) -> str:
    """Текст ошибки от провайдера, а не «объект по такому-то адресу»."""
    message = str(getattr(exc, "message", "") or "").strip()
    return message or str(exc)


def _clamp_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _clamp_float(value: object, minimum: float, maximum: float, default: float) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _usage_to_dict(usage: object) -> dict:
    """Приводим объект usage к обычному словарю. Нет поля — считаем нулём."""
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
