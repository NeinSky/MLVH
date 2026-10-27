"""Фейковый MCP-сервер для проверок MLVH: только stdlib, без сети.

Запуск (так его и запускает клиент):
    python tools/fake_mcp_server.py

Умеет ровно то, что нужно проверкам: initialize, tools/list с двумя
страницами, tools/call с текстом, ошибкой, картинкой, падением и сном.
"""

from __future__ import annotations

import json
import os
import sys
import time

# MCP требует UTF-8 в stdio, а в Windows консоль по умолчанию в другой кодировке.
for stream in (sys.stdin, sys.stdout):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

TOOLS = [
    {
        "name": "echo",
        "description": "Вернуть текст обратно",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "fail",
        "description": "Всегда возвращает ошибку",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "crash",
        "description": "Роняет процесс",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "slow",
        "description": "Спит заданное число секунд",
        "inputSchema": {
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
        },
    },
    {
        "name": "image",
        "description": "Возвращает картинку",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "page",
        "description": "Инструмент со второй страницы tools/list",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def text_result(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict):
            continue

        method = message.get("method")
        message_id = message.get("id")

        if method == "initialize":
            params = message.get("params") or {}
            send(
                {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "protocolVersion": params.get(
                            "protocolVersion", "2025-06-18"
                        ),
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "mlvh-fake", "version": "1.0"},
                    },
                }
            )
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            params = message.get("params") or {}
            if params.get("cursor") == "page2":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": message_id,
                        "result": {"tools": [t for t in TOOLS if t["name"] == "page"]},
                    }
                )
            else:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": message_id,
                        "result": {
                            "tools": [t for t in TOOLS if t["name"] != "page"],
                            "nextCursor": "page2",
                        },
                    }
                )
        elif method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name == "echo":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": message_id,
                        "result": text_result(str(arguments.get("text", ""))),
                    }
                )
            elif name == "fail":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": message_id,
                        "result": text_result("не получилось", True),
                    }
                )
            elif name == "crash":
                os._exit(3)
            elif name == "slow":
                try:
                    seconds = float(arguments.get("seconds", 1))
                except (TypeError, ValueError):
                    seconds = 1.0
                time.sleep(max(0.0, seconds))
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": message_id,
                        "result": text_result("проснулся"),
                    }
                )
            elif name == "image":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": message_id,
                        "result": {
                            "content": [
                                {
                                    "type": "image",
                                    "data": "iVBORw0KGgo=",
                                    "mimeType": "image/png",
                                },
                                {"type": "text", "text": "картинка выше"},
                            ],
                            "isError": False,
                        },
                    }
                )
            elif name == "page":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": message_id,
                        "result": text_result("вторая страница"),
                    }
                )
            else:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": message_id,
                        "error": {
                            "code": -32602,
                            "message": f"Unknown tool: {name}",
                        },
                    }
                )
        elif message_id is not None:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "error": {"code": -32601, "message": "Method not found"},
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
