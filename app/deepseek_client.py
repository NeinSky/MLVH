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
class StreamEvent:
    """Один кусочек ответа: текст, рассуждение, статистика токенов или причина остановки."""

    kind: str  # content | reasoning | usage | finish
    text: str = ""
    usage: dict | None = None
    finish_reason: str | None = None


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

    def stream_chat(self, messages: list[dict]) -> Iterator[StreamEvent]:
        """Стримим ответ. Ошибки приводим к DeepSeekError."""
        params: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.thinking:
            # Включаем режим рассуждений (deepseek-flash / deepseek-v4-pro).
            params["extra_body"] = {"thinking": {"type": "enabled"}}

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
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield StreamEvent(kind="reasoning", text=reasoning)
                if delta.content:
                    yield StreamEvent(kind="content", text=delta.content)
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
