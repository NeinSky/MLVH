"""Проверка агента MLVH: офлайн на фейковой модели или вживую через DeepSeek.

Запуск:
    .venv\\Scripts\\python.exe tools\\check_agent.py
    .venv\\Scripts\\python.exe tools\\check_agent.py --live
    .venv\\Scripts\\python.exe tools\\check_agent.py --live --folder D:\\docs
"""

from __future__ import annotations

import argparse
import sys
import tempfile
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


class ScriptedClient:
    """Фейковая модель: отдаёт заранее заданные раунды, без сети."""

    model = "scripted"
    max_tokens = 1000
    thinking = False

    def __init__(self, script: list[list]) -> None:
        self.script = list(script)
        self.calls = 0
        self.seen: list[list[dict]] = []

    def stream_chat(self, messages, tools=None):
        self.calls += 1
        self.seen.append([dict(message) for message in messages])
        step = self.script.pop(0) if self.script else []
        if not step:
            step = [_content("Готово.")]
        for event in step:
            yield event

    def clone(self, **_kwargs):
        return self


def _content(text: str):
    from app.deepseek_client import StreamEvent

    return StreamEvent(kind="content", text=text)


def _tool(call_id: str, name: str, arguments: str):
    from app.deepseek_client import StreamEvent, ToolCall

    return StreamEvent(kind="tool_call", tool_call=ToolCall(call_id, name, arguments))


def _offline() -> int:
    from app.agent import AgentLimits, AgentRunner, ToolError, Workspace

    print("Агент: офлайн на фейковой модели")
    with tempfile.TemporaryDirectory(prefix="mlvh_agent_") as tmp:
        base = Path(tmp)
        root = base / "root"
        root.mkdir()
        (root / "sample.txt").write_text(
            "Первая строка.\nПривет, это sample.\n", encoding="utf-8"
        )
        (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
        (root / "binary.dat").write_bytes(b"\x00\x01\x02")
        (base / "outside.txt").write_text("снаружи\n", encoding="utf-8")

        limits = AgentLimits(
            max_rounds=10,
            duplicate_limit=2,
            max_tool_calls=20,
            max_seconds=60,
            max_file_bytes=100_000,
            max_read_lines=100,
            max_tool_chars=100_000,
        )
        workspace = Workspace([root], limits)

        print("политика доступа:")
        try:
            text = workspace.read_file("sample.txt")
        except ToolError as exc:
            check("файл внутри папки", False, str(exc))
        else:
            check("файл внутри папки", "Привет" in text, text[:120])

        for name, call in (
            ("выход через ..", lambda: workspace.read_file("../outside.txt")),
            ("абсолютный путь наружу", lambda: workspace.read_file(str(base / "outside.txt"))),
            ("секрет .env", lambda: workspace.read_file(".env")),
            ("двоичный файл", lambda: workspace.read_file("binary.dat")),
        ):
            try:
                call()
            except ToolError:
                check(name, True)
            else:
                check(name, False, "отказа не было")

        listing = workspace.list_dir(".", 50)
        check("list_dir скрывает .env", ".env" not in listing, listing[:160])
        check("list_dir показывает sample.txt", "sample.txt" in listing, listing[:160])

        from app.agent import call_tool

        counted = call_tool(
            "count_lines", '{"extensions": [".txt"]}', workspace
        )
        check(
            "count_lines считает .txt",
            counted.ok and "строк 2" in counted.content,
            counted.content,
        )

        # --- нормальный цикл: инструмент, затем ответ ---
        print("агентский цикл:")
        script = [
            [_tool("call_1", "read_file", '{"path": "sample.txt", "start_line": 1}')],
            [_content("В sample.txt написано: привет.")],
        ]
        client = ScriptedClient(script)
        summaries: list[str] = []
        reply = AgentRunner(
            client,
            [
                {"role": "system", "content": "Ты тестовый агент."},
                {"role": "user", "content": "Что в sample.txt?"},
            ],
            workspace,
            limits,
            context_tokens=100_000,
            max_tokens=1000,
            on_tool=summaries.append,
        ).run()
        check("ответ получен", reply is not None and bool(reply.text.strip()))
        check(
            "ответ по файлу",
            reply is not None and "привет" in reply.text.casefold(),
            reply.text if reply else "None",
        )
        check(
            "был один вызов",
            reply is not None and reply.tool_calls == 1,
            str(reply.tool_calls if reply else "None"),
        )
        check("строка инструмента в чат", any("читаю" in item for item in summaries),
              str(summaries))
        second = client.seen[-1] if client.seen else []
        tool_messages = [message for message in second if message.get("role") == "tool"]
        check(
            "результат ушёл модели",
            bool(tool_messages) and "Привет" in (tool_messages[0].get("content") or ""),
            str(tool_messages)[:160],
        )

        # --- зацикливание: одинаковый вызов много раз ---
        print("защита от зацикливания:")
        loop_call = _tool("call_loop", "read_file", '{"path": "sample.txt"}')
        loop_script = [[loop_call] for _ in range(4)] + [
            [_content("Отвечаю по прочитанному.")]
        ]
        loop_client = ScriptedClient(loop_script)
        loop_reply = AgentRunner(
            loop_client,
            [
                {"role": "system", "content": "Ты тестовый агент."},
                {"role": "user", "content": "Читай одно и то же."},
            ],
            workspace,
            limits,
            context_tokens=100_000,
            max_tokens=1000,
        ).run()
        check(
            "цикл остановлен",
            loop_reply is not None and loop_reply.stopped == "stuck",
            loop_reply.stopped if loop_reply else "None",
        )
        check(
            "повторов не больше лимита",
            loop_reply is not None
            and loop_reply.tool_calls <= limits.duplicate_limit + 2,
            str(loop_reply.tool_calls if loop_reply else "None"),
        )
        check(
            "финальный ответ всё равно есть",
            loop_reply is not None and "прочитанному" in loop_reply.text,
            loop_reply.text if loop_reply else "None",
        )

    print()
    if FAILED:
        print(f"Провалено: {len(FAILED)} — {', '.join(FAILED)}")
        return 1
    print("Офлайн-проверка агента пройдена.")
    return 0


def _live(args) -> int:
    from app.agent import AgentLimits, AgentRunner, Workspace, build_agent_prompt
    from app.config import get_api_key, load_settings
    from app.deepseek_client import DeepSeekClient, DeepSeekError

    print("Агент: живая проверка через DeepSeek")
    key = get_api_key()
    if not key:
        print("  FAIL нет ключа DeepSeek (.env)")
        return 1

    settings = load_settings()
    model = settings.model
    max_tokens = model.max_tokens
    if model.thinking:
        max_tokens = max(max_tokens, 2000)

    with tempfile.TemporaryDirectory(prefix="mlvh_live_") as tmp:
        if args.folder:
            folder = Path(args.folder).expanduser()
        else:
            folder = Path(tmp)
            (folder / "sample.txt").write_text(
                "Меня зовут MLVH.\nПроект начат в 2026 году.\nСекрет: сорок два.\n",
                encoding="utf-8",
            )
        question = args.question or (
            "Что находится в этой папке? Перечисли файлы и коротко опиши."
            if args.folder
            else "Прочитай sample.txt и скажи, что за секрет."
        )

        limits = AgentLimits.from_settings(settings.agent)
        workspace = Workspace([folder], limits)
        if not workspace:
            for raw, why in workspace.skipped:
                print(f"  FAIL папка {raw}: {why}")
            return 1

        system = build_agent_prompt(settings.system_prompt, workspace, limits)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        try:
            client = DeepSeekClient(
                api_key=key,
                model=model.name,
                temperature=model.temperature,
                max_tokens=max_tokens,
                thinking=model.thinking,
            )
        except DeepSeekError as exc:
            print(f"  FAIL {exc}")
            return 1

        def on_tool(summary: str) -> None:
            print(f"  [инструмент] {summary}", flush=True)

        def on_note(message: str) -> None:
            print(f"  [агент] {message}", flush=True)

        streamed = False

        def on_content(chunk: str) -> None:
            nonlocal streamed
            streamed = True
            print(chunk, end="", flush=True)

        print(f"Вопрос: {question}")
        runner = AgentRunner(
            client,
            messages,
            workspace,
            limits,
            context_tokens=model.context_tokens,
            max_tokens=max_tokens,
            on_content=on_content,
            on_tool=on_tool,
            on_note=on_note,
        )
        try:
            reply = runner.run()
        except DeepSeekError as exc:
            print(f"\n  FAIL {exc}")
            return 1
        print()
        if reply is None:
            print("  FAIL агент остановлен без ответа")
            return 1
        if not streamed and reply.text:
            print(reply.text)

        print(
            f"Шагов: {reply.rounds}, вызовов: {reply.tool_calls}, "
            f"причина: {reply.stopped}"
        )
        check("агент вызвал инструмент", reply.tool_calls > 0, str(reply.tool_calls))
        check("ответ не пустой", bool(reply.text.strip()), reply.text)
        if not args.folder:
            check(
                "ответ содержит секрет",
                "сорок два" in reply.text.casefold(),
                reply.text,
            )

    print()
    if FAILED:
        print(f"Провалено: {len(FAILED)} — {', '.join(FAILED)}")
        return 1
    print("Живая проверка агента пройдена.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка агента MLVH")
    parser.add_argument("--live", action="store_true", help="живой запрос к DeepSeek")
    parser.add_argument(
        "--folder",
        metavar="PATH",
        help="папка для --live (по умолчанию временная с sample.txt)",
    )
    parser.add_argument("--question", metavar="TEXT", help="вопрос для --live")
    args = parser.parse_args()
    return _live(args) if args.live else _offline()


if __name__ == "__main__":
    raise SystemExit(main())
