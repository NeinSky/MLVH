"""Проверка MCP-клиента MLVH: фейковый сервер офлайн или серверы из конфига.

Запуск:
    .venv\\Scripts\\python.exe tools\\check_mcp.py
    .venv\\Scripts\\python.exe tools\\check_mcp.py --config
    .venv\\Scripts\\python.exe tools\\check_mcp.py --config --call mcp_demo_echo --args '{"text": "привет"}'
    .venv\\Scripts\\python.exe tools\\check_mcp.py --preset chrome-devtools
    .venv\\Scripts\\python.exe tools\\check_mcp.py --agent-live
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}".rstrip())
        FAILED.append(name)


def _fake_server():
    from app.agent.mcp import McpServer

    script = BASE_DIR / "tools" / "fake_mcp_server.py"
    return McpServer(
        id="demo",
        command=sys.executable,
        args=["-u", str(script)],
        timeout=0,
    )


def _offline() -> int:
    from app.agent.mcp import McpLimits, McpServer, McpSession
    from app.agent.policy import ToolError

    print("MCP: фейковый сервер, офлайн")
    session = McpSession(
        [_fake_server()],
        McpLimits(timeout=3, start_timeout=20),
        log_path=None,
    )
    try:
        schemas = session.schemas()
        names = [item["function"]["name"] for item in schemas]
        check("шесть инструментов со второй страницы", len(names) == 6, str(names))
        check(
            "имена с префиксом mcp_<id>_",
            all(name.startswith("mcp_demo_") for name in names),
            str(names),
        )
        check("описание с именем сервера", "[demo]" in schemas[0]["function"]["description"])

        echo = session.call("mcp_demo_echo", {"text": "сорок два"})
        check("echo вернул текст", echo.ok and "сорок два" in echo.content, echo.content)
        check("подпись вызова для чата", "MCP demo" in echo.summary, echo.summary)

        failed = session.call("mcp_demo_fail", {})
        check(
            "isError → ошибка",
            not failed.ok and failed.content.startswith("Ошибка:"),
            failed.content,
        )

        image = session.call("mcp_demo_image", {})
        check(
            "картинка — плейсхолдер без base64",
            image.ok
            and "изображение" in image.content
            and "iVBORw0KGgo" not in image.content,
            image.content,
        )

        page = session.call("mcp_demo_page", {})
        check("инструмент со второй страницы", "вторая страница" in page.content, page.content)

        try:
            session.call("mcp_demo_nope", {})
            check("неизвестный инструмент — ошибка", False)
        except ToolError:
            check("неизвестный инструмент — ошибка", True)

        crash = session.call("mcp_demo_crash", {})
        check("падение сервера — ошибка", not crash.ok, crash.content)
        again = session.call("mcp_demo_echo", {"text": "после падения"})
        check(
            "после падения сервер перезапущен",
            again.ok and "после падения" in again.content,
            again.content,
        )

        stopped = McpSession(
            [_fake_server()],
            McpLimits(timeout=3, start_timeout=20),
            log_path=None,
        )
        try:
            stopped.schemas()
            try:
                stopped.call("mcp_demo_echo", {"text": "x"}, should_stop=lambda: True)
                check("отмена при вызове", False)
            except ToolError:
                check("отмена при вызове", True)
        finally:
            stopped.close()
    finally:
        session.close()

    print("MCP: таймаут и битая команда")
    slow = McpSession(
        [_fake_server()],
        McpLimits(timeout=0.5, start_timeout=20),
        log_path=None,
    )
    try:
        slow.schemas()
        timed_out = slow.call("mcp_demo_slow", {"seconds": 2})
        check(
            "таймаут вызова",
            not timed_out.ok and "не ответил" in timed_out.content,
            timed_out.content,
        )
    finally:
        slow.close()

    broken = McpSession(
        [
            McpServer.from_dict(
                {"id": "broken", "command": "mlvh-no-such-command-xyz"}
            )
        ],
        McpLimits(timeout=2, start_timeout=5),
        log_path=None,
    )
    try:
        check("битая команда не даёт инструментов", broken.schemas() == [])
        check("причина записана", bool(broken.problems), str(broken.problems))
    finally:
        broken.close()

    print()
    if FAILED:
        print(f"Провалено: {len(FAILED)} — {', '.join(FAILED)}")
        return 1
    print("Офлайн-проверка MCP пройдена.")
    return 0


def _preset(name: str):
    from app.agent.mcp import McpServer

    lowered = name.strip().casefold()
    if lowered in ("demo", "fake"):
        return _fake_server()
    if lowered in ("chrome-devtools", "chrome"):
        return McpServer(
            id="chrome-devtools",
            command="npx",
            args=["-y", "chrome-devtools-mcp@latest"],
            timeout=180,
        )
    raise SystemExit(f"Неизвестный пресет «{name}». Есть: demo, chrome-devtools.")


def _config_mode(args) -> int:
    from app.agent.mcp import McpLimits, McpSession, mcp_servers_from_settings
    from app.config import load_settings

    settings = load_settings()
    if args.preset:
        servers = [_preset(args.preset)]
        print(f"Пресет: {args.preset}")
    else:
        servers = mcp_servers_from_settings(settings.agent)
        off = [server.id for server in servers if not server.enabled]
        if off:
            print("Выключены в config.json: " + ", ".join(off))

    if not servers:
        print(
            "В config.json нет MCP-серверов (agent.mcp_servers). "
            "Добавьте сервер в окне MLVH или впишите в config.json."
        )
        return 0

    limits = McpLimits.from_settings(settings.agent)
    if args.preset:
        # Первый запуск npx скачивает пакет — даём проверке больше времени.
        limits = McpLimits(
            timeout=max(limits.timeout, 120),
            start_timeout=max(limits.start_timeout, 300),
        )
    session = McpSession(servers, limits)
    names: list[str] = []
    try:
        names = [item["function"]["name"] for item in session.schemas()]
        if names:
            print("Инструменты:")
            for name in names:
                print(f"  {name}")
        else:
            print("Серверы не дали инструментов.")
        for problem in session.problems:
            print(f"  проблема: {problem}")

        if args.call:
            try:
                arguments = json.loads(args.arguments or "{}")
            except json.JSONDecodeError as exc:
                print(f"Аргументы не JSON: {exc}")
                return 1
            if not isinstance(arguments, dict):
                print("Аргументы должны быть JSON-объектом.")
                return 1
            try:
                result = session.call(args.call, arguments)
            except Exception as exc:
                print(f"Вызов не удался: {exc}")
                return 1
            print()
            print(("ok   " if result.ok else "FAIL ") + result.summary)
            print(result.content)
            if not result.ok:
                FAILED.append(args.call)
    finally:
        session.close()

    print()
    if FAILED:
        print(f"Провалено: {len(FAILED)} — {', '.join(FAILED)}")
        return 1
    if session.problems and not names:
        return 1
    print("Проверка MCP-серверов завершена.")
    return 0


def _agent_live(args) -> int:
    from app.agent import (
        AgentLimits,
        AgentRunner,
        ToolContext,
        build_agent_prompt,
        build_messages,
    )
    from app.agent.mcp import McpLimits, McpSession
    from app.config import get_api_key, load_settings
    from app.deepseek_client import DeepSeekClient, DeepSeekError

    key = get_api_key()
    if not key:
        print("Нужен DEEPSEEK_API_KEY в .env")
        return 1

    settings = load_settings()
    limits = AgentLimits.from_settings(settings.agent)
    model = settings.model
    max_tokens = max(model.max_tokens, 2000) if model.thinking else model.max_tokens

    session = McpSession(
        [_fake_server()],
        McpLimits(timeout=10, start_timeout=30),
        log_path=None,
    )
    try:
        context = ToolContext(mcp=session)
        prompt = build_agent_prompt(
            settings.system_prompt, None, limits, None, session.server_ids
        )
        question = (
            args.question
            or "Вызови инструмент mcp_demo_echo с текстом «сорок два» "
            "и напиши, что он вернул."
        )
        messages = build_messages(
            prompt,
            [{"role": "user", "content": question}],
            model.context_tokens,
            max_tokens,
        ).messages

        client = DeepSeekClient(
            api_key=key,
            model=model.name,
            temperature=model.temperature,
            max_tokens=max_tokens,
            thinking=model.thinking,
        )

        def on_tool(summary: str) -> None:
            print(f"  [инструмент] {summary}", flush=True)

        runner = AgentRunner(
            client,
            messages,
            context,
            limits,
            context_tokens=model.context_tokens,
            max_tokens=max_tokens,
            on_content=lambda chunk: print(chunk, end="", flush=True),
            on_note=lambda text: print(f"\n  [агент] {text}", flush=True),
            on_tool=on_tool,
        )
        try:
            reply = runner.run()
        except DeepSeekError as exc:
            print(f"\nОшибка DeepSeek: {exc}")
            return 1
    finally:
        session.close()

    print()
    if reply is None:
        print("Агент остановлен без ответа.")
        return 1
    print(f"Шагов: {reply.rounds}, вызовов: {reply.tool_calls}, причина: {reply.stopped}")
    check("модель вызвала MCP-инструмент", reply.tool_calls > 0, str(reply.tool_calls))
    check(
        "ответ содержит результат инструмента",
        "сорок два" in reply.text.casefold(),
        reply.text,
    )
    print()
    if FAILED:
        print(f"Провалено: {len(FAILED)} — {', '.join(FAILED)}")
        return 1
    print("Живая проверка MCP через DeepSeek пройдена.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка MCP-клиента MLVH")
    parser.add_argument(
        "--config",
        action="store_true",
        help="поднять серверы из config.json (agent.mcp_servers)",
    )
    parser.add_argument(
        "--preset",
        metavar="NAME",
        help="готовый сервер вместо config.json: demo или chrome-devtools",
    )
    parser.add_argument("--call", metavar="TOOL", help="вызвать инструмент по имени")
    parser.add_argument(
        "--args",
        dest="arguments",
        metavar="JSON",
        default="{}",
        help="аргументы для --call, JSON-объект",
    )
    parser.add_argument(
        "--agent-live",
        action="store_true",
        help="живой DeepSeek + фейковый MCP-сервер",
    )
    parser.add_argument("--question", metavar="TEXT", help="вопрос для --agent-live")
    args = parser.parse_args()

    if args.agent_live:
        return _agent_live(args)
    if args.config or args.preset:
        return _config_mode(args)
    return _offline()


if __name__ == "__main__":
    raise SystemExit(main())
