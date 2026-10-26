"""Инструменты агента: файлы и интернет, только чтение. Схемы — OpenAI function calling."""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from app.agent.policy import ToolError, Workspace

if TYPE_CHECKING:  # web.py — только stdlib, но импорт держим ленивым
    from app.agent.web import WebSession

CAPABILITY_FILES = "files"
CAPABILITY_WEB = "web"


@dataclass
class ToolResult:
    """Результат вызова: content уходит модели, summary — в чат пользователю."""

    content: str
    summary: str
    ok: bool = True


@dataclass
class ToolContext:
    """Что доступно инструментам: файлы, интернет и признак отмены."""

    workspace: Workspace | None = None
    web: "WebSession | None" = None
    should_stop: Callable[[], bool] | None = None


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
    {
        "type": "function",
        "function": {
            "name": "count_lines",
            "description": (
                "Посчитать строки в файлах с указанными расширениями "
                "(например, .py и .toml). Возвращает только статистику, "
                "без содержимого файлов."
            ),
            "parameters": _schema(
                {
                    "path": {
                        "type": "string",
                        "description": "Папка или файл. По умолчанию корень.",
                    },
                    "extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Расширения, например [\".py\", \".toml\"].",
                    },
                    "max_files": {
                        "type": "integer",
                        "description": "Сколько подходящих файлов обработать (1–5000).",
                        "minimum": 1,
                        "maximum": 5000,
                    },
                },
                ["extensions"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Найти страницы в интернете через DuckDuckGo. Возвращает заголовки, "
                "адреса и короткие описания. Начни с поиска, затем открой 1–3 "
                "самых подходящих ссылки через open_page."
            ),
            "parameters": _schema(
                {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос, лучше короткий и конкретный.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Сколько результатов вернуть (1–10).",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                ["query"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_page",
            "description": (
                "Открыть страницу по адресу и вернуть её текст, заголовок и ссылки. "
                "Только http и https: файлы и локальная сеть закрыты. "
                "Страницы на JavaScript отрисовываются в Chrome."
            ),
            "parameters": _schema(
                {
                    "url": {
                        "type": "string",
                        "description": "Полный адрес страницы, например https://example.com/news.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Сколько символов текста вернуть (1000–100000).",
                        "minimum": 1000,
                        "maximum": 100000,
                    },
                },
                ["url"],
            ),
        },
    },
]


def _list_dir(ctx: ToolContext, args: dict) -> ToolResult:
    workspace = _require_workspace(ctx)
    path = _path_arg(args)
    limit = _as_int(args.get("limit"), 200)
    content = workspace.list_dir(path, limit)
    return ToolResult(content=content, summary=_short(f"смотрю «{path}»"))


def _read_file(ctx: ToolContext, args: dict) -> ToolResult:
    workspace = _require_workspace(ctx)
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


def _search_text(ctx: ToolContext, args: dict) -> ToolResult:
    workspace = _require_workspace(ctx)
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
        ctx.should_stop,
    )
    return ToolResult(
        content=content,
        summary=_short(f"ищу «{query.strip()}» в «{path}»"),
    )


def _count_lines(ctx: ToolContext, args: dict) -> ToolResult:
    workspace = _require_workspace(ctx)
    path = _path_arg(args)
    extensions = args.get("extensions")
    if isinstance(extensions, str):
        extensions = [
            part for part in extensions.replace(",", " ").split() if part
        ]
    if not isinstance(extensions, list):
        raise ToolError(
            "extensions должен быть списком, например [\".py\", \".toml\"]"
        )
    max_files = _as_int(args.get("max_files"), 2000)
    content = workspace.count_lines(path, extensions, max_files, ctx.should_stop)
    pretty = ", ".join(str(item) for item in extensions)
    return ToolResult(
        content=content,
        summary=_short(f"считаю строки {pretty} в «{path}»"),
    )


def _search_web(ctx: ToolContext, args: dict) -> ToolResult:
    web = _require_web(ctx)
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query должен быть непустой строкой")
    max_results = _as_int(args.get("max_results"), 0)
    content = web.search(query.strip(), max_results, ctx.should_stop)
    return ToolResult(
        content=content,
        summary=_short(f"ищу в интернете «{query.strip()}»"),
    )


def _open_page(ctx: ToolContext, args: dict) -> ToolResult:
    web = _require_web(ctx)
    url = args.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ToolError("url должен быть непустой строкой")
    url = url.strip()
    max_chars = _as_int(args.get("max_chars"), 0)
    content = web.open_page(url, max_chars, ctx.should_stop)
    host = urllib.parse.urlsplit(
        url if "://" in url else "https://" + url
    ).hostname or url
    return ToolResult(content=content, summary=_short(f"открываю «{host}»"))



def _require_workspace(ctx: ToolContext) -> Workspace:
    if ctx.workspace is None:
        raise ToolError("чтение файлов не разрешено")
    return ctx.workspace


def _require_web(ctx: ToolContext) -> "WebSession":
    if ctx.web is None:
        raise ToolError("доступ в интернет не разрешён")
    return ctx.web


_HANDLERS: dict[str, Callable[[ToolContext, dict], ToolResult]] = {
    "list_dir": _list_dir,
    "read_file": _read_file,
    "search_text": _search_text,
    "count_lines": _count_lines,
    "search_web": _search_web,
    "open_page": _open_page,
}

_CAPABILITIES: dict[str, str] = {
    "list_dir": CAPABILITY_FILES,
    "read_file": CAPABILITY_FILES,
    "search_text": CAPABILITY_FILES,
    "count_lines": CAPABILITY_FILES,
    "search_web": CAPABILITY_WEB,
    "open_page": CAPABILITY_WEB,
}

_SCHEMAS: dict[str, dict] = {
    item["function"]["name"]: item for item in TOOL_SCHEMAS
}


def tool_schemas(*, files: bool = True, web: bool = False) -> list[dict]:
    """Схемы только разрешённых возможностей: модели не предлагаем лишнего."""
    allowed: set[str] = set()
    if files:
        allowed.add(CAPABILITY_FILES)
    if web:
        allowed.add(CAPABILITY_WEB)
    return [
        schema
        for name, schema in _SCHEMAS.items()
        if _CAPABILITIES.get(name) in allowed
    ]


def tool_names(*, files: bool = True, web: bool = False) -> tuple[str, ...]:
    return tuple(
        item["function"]["name"] for item in tool_schemas(files=files, web=web)
    )


def call_tool(
    name: str,
    arguments: str | None,
    context: ToolContext | Workspace | None,
    should_stop: Callable[[], bool] | None = None,
) -> ToolResult:
    """Разбираем аргументы и вызываем инструмент. Ошибки — текстом для модели."""
    if isinstance(context, ToolContext):
        ctx = context
    else:
        ctx = ToolContext(workspace=context)
    if should_stop is not None:
        ctx.should_stop = should_stop

    handler = _HANDLERS.get(name)
    if handler is None:
        available = tool_names(
            files=ctx.workspace is not None, web=ctx.web is not None
        )
        return ToolResult(
            content=f"Ошибка: неизвестный инструмент «{name}». Доступны: "
            + ", ".join(available or ("— нет —",)),
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
        return handler(ctx, args)
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
