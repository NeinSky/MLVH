"""Инструменты агента: только чтение. Схемы — в формате OpenAI function calling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from app.agent.policy import ToolError, Workspace


@dataclass
class ToolResult:
    """Результат вызова: content уходит модели, summary — в чат пользователю."""

    content: str
    summary: str
    ok: bool = True


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in ("1", "true", "да", "yes")
    return default


def _path_arg(args: dict) -> str:
    value = args.get("path", ".")
    if not isinstance(value, str):
        raise ToolError("path должен быть строкой")
    return value


def _short(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _schema(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "Показать содержимое папки: файлы и подпапки с размерами. "
                "Начни с этого, если не знаешь точный путь."
            ),
            "parameters": _schema(
                {
                    "path": {
                        "type": "string",
                        "description": (
                            "Папка: абсолютный путь или путь от рабочей папки. "
                            "По умолчанию корень."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Сколько элементов показать (1–500).",
                        "minimum": 1,
                        "maximum": 500,
                    },
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Прочитать текстовый файл постранично, с номерами строк. "
                "Для продолжения передай start_line. Большие файлы читай частями."
            ),
            "parameters": _schema(
                {
                    "path": {
                        "type": "string",
                        "description": "Файл: абсолютный путь или путь от рабочей папки.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "С какой строки начать (нумерация с 1).",
                        "minimum": 1,
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Сколько строк вернуть за раз (1–2000).",
                        "minimum": 1,
                        "maximum": 2000,
                    },
                },
                ["path"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": (
                "Найти подстроку в текстовых файлах папки. Возвращает путь, "
                "номер строки и фрагмент. Ищет буквально, без регулярных выражений."
            ),
            "parameters": _schema(
                {
                    "query": {
                        "type": "string",
                        "description": "Что искать (до 200 символов).",
                    },
                    "path": {
                        "type": "string",
                        "description": "Где искать: папка или файл. По умолчанию корень.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Сколько совпадений вернуть (1–50).",
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Учитывать регистр (по умолчанию нет).",
                    },
                },
                ["query"],
            ),
        },
    },
]


def _list_dir(
    workspace: Workspace,
    args: dict,
    should_stop: Callable[[], bool] | None,
) -> ToolResult:
    path = _path_arg(args)
    limit = _as_int(args.get("limit"), 200)
    content = workspace.list_dir(path, limit)
    return ToolResult(content=content, summary=_short(f"смотрю «{path}»"))


def _read_file(
    workspace: Workspace,
    args: dict,
    should_stop: Callable[[], bool] | None,
) -> ToolResult:
    path = _path_arg(args)
    start = max(1, _as_int(args.get("start_line"), 1))
    max_lines = max(
        1,
        _as_int(args.get("max_lines"), workspace.limits.max_read_lines),
    )
    content = workspace.read_file(path, start, max_lines)
    return ToolResult(
        content=content,
        summary=_short(f"читаю «{path}», строки с {start}"),
    )


def _search_text(
    workspace: Workspace,
    args: dict,
    should_stop: Callable[[], bool] | None,
) -> ToolResult:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query должен быть непустой строкой")
    path = _path_arg(args)
    max_results = _as_int(args.get("max_results"), 30)
    case_sensitive = _as_bool(args.get("case_sensitive"), False)
    content = workspace.search_text(
        query.strip(),
        path,
        max_results,
        case_sensitive,
        should_stop,
    )
    return ToolResult(
        content=content,
        summary=_short(f"ищу «{query.strip()}» в «{path}»"),
    )


_HANDLERS: dict[str, Callable] = {
    "list_dir": _list_dir,
    "read_file": _read_file,
    "search_text": _search_text,
}


def tool_names() -> tuple[str, ...]:
    return tuple(_HANDLERS)


def call_tool(
    name: str,
    arguments: str | None,
    workspace: Workspace,
    should_stop: Callable[[], bool] | None = None,
) -> ToolResult:
    """Разбираем аргументы и вызываем инструмент. Ошибки — текстом для модели."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return ToolResult(
            content=f"Ошибка: неизвестный инструмент «{name}». Доступны: "
            + ", ".join(_HANDLERS),
            summary=_short(f"неизвестный инструмент «{name}»"),
            ok=False,
        )

    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        return ToolResult(
            content=f"Ошибка: не удалось разобрать аргументы JSON: {exc}",
            summary=_short(f"{name}: ошибка аргументов"),
            ok=False,
        )
    if not isinstance(args, dict):
        return ToolResult(
            content="Ошибка: аргументы инструмента должны быть JSON-объектом.",
            summary=_short(f"{name}: аргументы не объект"),
            ok=False,
        )

    try:
        return handler(workspace, args, should_stop)
    except ToolError as exc:
        return ToolResult(
            content=f"Ошибка: {exc}",
            summary=_short(f"{name}: отказ — {exc}"),
            ok=False,
        )
    except Exception as exc:  # инструмент не должен ронять агентский цикл
        return ToolResult(
            content=f"Ошибка инструмента: {exc}",
            summary=_short(f"{name}: сбой — {exc}"),
            ok=False,
        )
