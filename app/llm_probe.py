"""Проверка подключения к провайдеру модели: GET /models, без Qt и openai.

Нужна кнопке «Проверить подключение» в окне и проверкам в tools/.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from app.config import DEEPSEEK_BASE_URL, is_local_host

DEFAULT_TIMEOUT = 8.0


class ProbeError(RuntimeError):
    """Проверка не удалась: текст понятен пользователю."""


def models_urls(base_url: str) -> list[str]:
    """Куда стучаться за списком моделей: /models и, если надо, /v1/models.

    Пользователь может указать адрес с префиксом (`.../api/v1`) или без него
    (`https://api.deepseek.com`), поэтому пробуем оба варианта.
    """
    base = (base_url or "").strip().rstrip("/") or DEEPSEEK_BASE_URL
    urls = [f"{base}/models"]
    if not base.endswith("/v1"):
        urls.append(f"{base}/v1/models")
    return urls


def probe_models(
    base_url: str,
    api_key: str = "",
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[str, list[str]]:
    """GET /models → (рабочий адрес, список алиасов). Ошибки — ProbeError."""
    problems: list[str] = []
    for url in models_urls(base_url):
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        # Локальному серверу прокси из системы только мешает: он отвечает 503.
        local = is_local_host(_host(url))
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({} if local else None)
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                raw = response.read(200_000)
        except urllib.error.HTTPError as exc:
            detail = _error_detail(exc)
            if exc.code in (401, 403):
                raise ProbeError(
                    f"Ключ отклонён ({exc.code}). Проверьте ключ и адрес API."
                ) from exc
            problems.append(f"{url}: {exc.code} {detail}".strip())
            continue
        except urllib.error.URLError as exc:
            raise ProbeError(f"Нет связи с {_host(url)}: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            raise ProbeError(f"{_host(url)} не ответил вовремя: {exc}") from exc

        models = _parse_models(raw)
        if models is None:
            problems.append(f"{url}: ответ не похож на список моделей")
            continue
        return url, models
    raise ProbeError(
        "Не удалось получить список моделей. " + "; ".join(problems)
    )


def _parse_models(raw: bytes) -> list[str] | None:
    """Разбор ответа /models: {"data": [{"id": ...}]} или просто список."""
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return None
    if isinstance(data, dict):
        items = data.get("data")
    elif isinstance(data, list):
        items = data
    else:
        return None
    if not isinstance(items, list):
        return None

    names: list[str] = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("id") or item.get("name") or "").strip()
        else:
            name = str(item).strip()
        if name and name not in names:
            names.append(name)
    return names


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """Текст ошибки от провайдера, если он его прислал."""
    try:
        raw = exc.read(4000)
    except Exception:  # тело может быть уже прочитано или пустым
        return str(exc.reason or "")
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return raw.decode("utf-8", "replace").strip()[:200]
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or "").strip()[:200]
        if isinstance(error, str):
            return error.strip()[:200]
    return str(exc.reason or "")


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc or url