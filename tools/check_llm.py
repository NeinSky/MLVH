"""Проверка клиента модели MLVH: фейковый OpenAI-совместимый сервер, без ключей.

Запуск:
    .venv\\Scripts\\python.exe tools\\check_llm.py
    .venv\\Scripts\\python.exe tools\\check_llm.py --live
    .venv\\Scripts\\python.exe tools\\check_llm.py --live --base-url https://openrouter.ai/api/v1 --model openai/gpt-4o-mini

Офлайн-часть поднимает свой http.server на 127.0.0.1 с тем же протоколом, что
у настоящего провайдера: /v1/models, /v1/chat/completions, SSE, tools,
рассуждения, usage и коды ошибок.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAILED: list[str] = []
API_KEY = "test-key"
MODELS = (
    "demo",
    "reason",
    "reason-native",
    "reason-tools",
    "reason-empty",
    "strict-reasoning",
    "tools",
    "no-stream-options",
    "no-usage",
    "reject",
    "missing",
)
SEEN: list[dict] = []
LOCK = threading.Lock()

# Одна схема инструмента: проверяем, что она переживает второй запрос агента.
_TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "прочитать файл",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}]


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}".rstrip())
        FAILED.append(name)


def _chunk(model: str, delta: dict, finish: str | None = None) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _chunks(model: str, payload: dict | None = None) -> list[dict]:
    """Что фейковый провайдер пришлёт для этого алиаса."""
    payload = payload or {}
    with_tools = bool(payload.get("tools"))
    messages = payload.get("messages") or []
    asked = [
        message
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and message.get("tool_calls")
    ]
    if model == "reason-empty" and with_tools:
        # Как настоящий DeepSeek в errors/2026-11-17: на первом шаге — только
        # рассуждения (текста нет), на повторе без рассуждений — инструмент,
        # после ответа инструмента — итог.
        thinking_on = (payload.get("thinking") or {}).get("type") == "enabled"
        if not asked:
            if thinking_on:
                return [_chunk(model, {"reasoning_content": "сначала подумаю"})]
            return [
                _chunk(model, {"tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                }]}),
                _chunk(model, {}, "tool_calls"),
            ]
        return [_chunk(model, {"content": "Там две строки."})]
    if with_tools or model in ("tools", "reason-tools"):
        if model in ("reason-tools", "strict-reasoning"):
            # Режим рассуждений: думаем, потом просим инструмент.
            return [
                _chunk(model, {"reasoning_content": "думы"}),
                _chunk(model, {"tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                }]}),
                _chunk(model, {}, "tool_calls"),
            ]
        return [
            _chunk(model, {"tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":'},
            }]}),
            _chunk(model, {"tool_calls": [{
                "index": 0,
                "function": {"arguments": ' "a.txt"}'},
            }]}),
            _chunk(model, {}, "tool_calls"),
        ]
    if model in ("reason", "strict-reasoning"):
        return [
            _chunk(model, {"reasoning_content": "думы"}),
            _chunk(model, {"content": "Ответ"}),
        ]
    if model == "reason-native":
        return [
            _chunk(model, {"reasoning": "размышляю"}),
            _chunk(model, {"content": "Ответ"}),
        ]
    return [
        _chunk(model, {"content": "Привет"}),
        _chunk(model, {"content": "!"}),
        _chunk(model, {}, "stop"),
    ]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # noqa: D401 — тишина в консоли
        return

    def do_GET(self) -> None:  # noqa: N802 — имя задаёт http.server
        path = self.path.split("?", 1)[0]
        if not path.endswith("/models"):
            self._json(404, {"error": {"message": "not found"}})
            return
        if not self._authorized():
            return
        self._json(200, {"object": "list", "data": [{"id": name} for name in MODELS]})

    def do_POST(self) -> None:  # noqa: N802 — имя задаёт http.server
        path = self.path.split("?", 1)[0]
        if not path.endswith("/chat/completions"):
            self._json(404, {"error": {"message": "not found"}})
            return
        if not self._authorized():
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._json(400, {"error": {"message": "bad json"}})
            return
        with LOCK:
            SEEN.append(payload)

        model = str(payload.get("model") or "")
        if model == "missing":
            self._json(404, {"error": {"message": "model not found"}})
            return
        if model == "reject":
            # 400 не про stream_options: повторять запрос без него не нужно.
            self._json(400, {"error": {"message": "unknown field: temperature"}})
            return
        if model == "no-stream-options" and "stream_options" in payload:
            self._json(400, {"error": {"message": "unknown field: stream_options"}})
            return
        # strict-reasoning и reason-empty изображают DeepSeek: thinking
        # и рассуждения они понимают.
        deepseek_like = model in ("strict-reasoning", "reason-empty")
        if not deepseek_like and payload.get("thinking"):
            self._json(400, {"error": {"message": "unknown field: thinking"}})
            return
        if deepseek_like and self._missing_reasoning(payload):
            self._json(400, {
                "error": {
                    "message": "The `reasoning_content` in the thinking mode must be "
                               "passed back to the API.",
                    "type": "invalid_request_error",
                }
            })
            return

        self._stream(model, payload)

    @staticmethod
    def _missing_reasoning(payload: dict) -> bool:
        """Как настоящий DeepSeek: в режиме рассуждений с `tools` история
        обязана нести рассуждения.

        Проверяем каждый assistant, который просил инструменты: без
        `reasoning_content` он отвечает 400 на следующий же запрос. Без
        рассуждений (`thinking` не `enabled`) поле не нужно.
        """
        if not payload.get("tools"):
            return False
        if (payload.get("thinking") or {}).get("type") != "enabled":
            return False
        for message in payload.get("messages") or []:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            if message.get("tool_calls") and not message.get("reasoning_content"):
                return True
        return False

    def _authorized(self) -> bool:
        if self.headers.get("Authorization") == f"Bearer {API_KEY}":
            return True
        self._json(401, {"error": {"message": "invalid api key"}})
        return False

    def _json(self, code: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, model: str, payload: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        for chunk in _chunks(model, payload):
            self._event(chunk)
        if model != "no-usage":
            self._event({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 2,
                    "total_tokens": 9,
                },
            })
        self.wfile.write(b"data: [DONE]\n\n")

    def _event(self, payload: dict) -> None:
        line = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        self.wfile.write(line.encode("utf-8"))


def _collect(client, messages: list[dict], tools: list[dict] | None = None):
    """Один запрос: текст, рассуждения, вызовы инструментов и статистика."""
    text: list[str] = []
    reasoning: list[str] = []
    calls: list = []
    usage: dict = {}
    for event in client.stream_chat(messages, tools=tools):
        if event.kind == "content":
            text.append(event.text)
        elif event.kind == "reasoning":
            reasoning.append(event.text)
        elif event.kind == "tool_call" and event.tool_call is not None:
            calls.append(event.tool_call)
        elif event.kind == "usage" and event.usage:
            usage = event.usage
    return "".join(text), "".join(reasoning), calls, usage


def _failure(func):
    """Выполняем запрос и возвращаем ошибку LLMError (или None, если её нет)."""
    from app.llm_client import LLMError

    try:
        func()
    except LLMError as exc:
        return exc
    return None


def _offline() -> int:
    from app.llm_client import ChatClient, LLMError
    from app.llm_probe import ProbeError, probe_models

    print("Модель: фейковый OpenAI-совместимый сервер")
    from app.log_setup import setup_logging

    setup_logging()  # журнал в памяти: в консоль INFO не идёт
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = f"127.0.0.1:{server.server_port}"
    base = f"http://{host}/v1"
    messages = [{"role": "user", "content": "привет"}]

    def client(model: str = "demo", key: str = API_KEY, **kwargs) -> ChatClient:
        return ChatClient(api_key=key, model=model, base_url=base, **kwargs)

    try:
        print("обычный ответ:")
        text, _, _, usage = _collect(client(), messages)
        check("текст собран", text == "Привет!", text)
        check("usage получен", usage.get("total_tokens") == 9, str(usage))
        check("хост в подписи провайдера", client().label == host, client().label)

        print("рассуждения:")
        _, reasoning, _, _ = _collect(client("reason"), messages)
        check("reasoning_content разобран", reasoning == "думы", reasoning)
        _, reasoning, _, _ = _collect(client("reason-native"), messages)
        check("reasoning разобран", reasoning == "размышляю", reasoning)

        print("инструменты:")
        tools = [
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}},
            }
        ]
        _, _, calls, _ = _collect(client("tools"), messages, tools)
        check("tool_call собран", len(calls) == 1 and calls[0].name == "read_file", str(calls))
        check(
            "аргументы склеены",
            bool(calls) and calls[0].arguments == '{"path": "a.txt"}',
            calls[0].arguments if calls else "",
        )
        with LOCK:
            sent = SEEN[-1]
        check("схемы ушли провайдеру", bool(sent.get("tools")), str(sent)[:160])

        print("ошибки:")
        error = _failure(lambda: _collect(client(key="wrong"), messages))
        check("401 → LLMError", error is not None and error.status == 401, str(error))
        check(
            "401 называет переменную с ключом",
            error is not None and "OPENAI_API_KEY" in str(error),
            str(error),
        )
        error = _failure(lambda: _collect(client("missing"), messages))
        check(
            "404 → модель не найдена",
            error is not None and error.status == 404 and "не найдена" in str(error),
            str(error),
        )

        print("режим рассуждений:")
        auto_client = client(thinking=True)
        check("чужому провайдеру thinking не отправляем",
              auto_client.thinking_payload() is None)
        _collect(auto_client, messages)
        with LOCK:
            sent = SEEN[-1]
        check("в запросе нет thinking", "thinking" not in sent, str(sent)[:160])

        strict = client(thinking=True, thinking_style="deepseek")
        error = _failure(lambda: _collect(strict, messages))
        check(
            "style deepseek → 400 с текстом провайдера",
            error is not None and error.status == 400 and "thinking" in str(error),
            str(error),
        )

        none_client = client(thinking=True, thinking_style="none")
        _collect(none_client, messages)
        with LOCK:
            sent = SEEN[-1]
        check("style none не отправляет thinking", "thinking" not in sent, str(sent)[:160])

        print("stream_options:")
        text, _, _, _ = _collect(client("no-stream-options"), messages)
        check("откат без stream_options сработал", text == "Привет!", text)
        with LOCK:
            attempts = [item for item in SEEN if item.get("model") == "no-stream-options"]
        check("попыток было две", len(attempts) >= 2, str(len(attempts)))
        check(
            "во второй попытке без stream_options",
            bool(attempts) and "stream_options" not in attempts[-1],
            str(attempts[-1].keys()) if attempts else "",
        )

        print("чужой 400 не повторяем:")
        with LOCK:
            before = len(SEEN)
        error = _failure(lambda: _collect(client("reject"), messages))
        check("400 → LLMError", error is not None and error.status == 400, str(error))
        with LOCK:
            attempts = SEEN[before:]
        check("запрос отправлен один раз", len(attempts) == 1, str(len(attempts)))
        check(
            "stream_options не выключен из-за чужого 400",
            bool(attempts) and "stream_options" in attempts[0],
            str(attempts[0].keys()) if attempts else "",
        )

        print("usage может отсутствовать:")
        text, _, _, usage = _collect(client("no-usage"), messages)
        check("ответ без usage не ломается", text == "Привет!" and usage == {},
              f"{text} / {usage}")

        print("рассуждения вместе с инструментами (DeepSeek):")
        from app.agent.runner import _assistant_message, _Round

        check(
            "без thinking reasoning_content не уходит",
            "reasoning_content" not in _assistant_message(_Round("", [], {}, None, "думы")),
        )
        turned = _assistant_message(_Round("", [], {}, None, "думы"), thinking=True)
        check("с thinking reasoning_content возвращается",
              turned.get("reasoning_content") == "думы", str(turned)[:120])
        check("пустые рассуждения не добавляются",
              "reasoning_content" not in _assistant_message(_Round("", [], {}, None), True))

        from app.agent.runner import _Round, _assistant_message
        from app.llm_client import LLMError

        think_client = ChatClient(
            api_key=API_KEY, model="strict-reasoning", base_url=base,
            thinking=True, thinking_style="deepseek",
        )
        text, reasoning, calls, _ = _collect(think_client, messages, tools=_TOOLS)
        check("стрим отдал рассуждения", reasoning == "думы", reasoning)
        check("стрим отдал вызов инструмента", len(calls) == 1, str(calls))
        follow = messages + [
            _assistant_message(
                _Round(text, calls, {}, "tool_calls", "думы"), thinking=True
            ),
            {"role": "tool", "tool_call_id": calls[0].id if calls else "call_1",
             "content": "содержимое"},
        ]
        error = _failure(lambda: _collect(think_client, follow, tools=_TOOLS))
        check("второй запрос с tools принят (нет 400 про reasoning)",
              error is None, str(error))
        with LOCK:
            sent = SEEN[-1]
        assistants = [
            message for message in sent.get("messages", [])
            if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        check("в истории есть reasoning_content",
              bool(assistants) and "reasoning_content" in assistants[-1],
              str(assistants)[:160])

        no_reasoning = messages + [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "да"},
        ]
        error = _failure(lambda: _collect(think_client, no_reasoning, tools=_TOOLS))
        check("без поля провайдер отвечает 400 (как в жизни)",
              isinstance(error, LLMError) and error.status == 400, str(error))
        check("текст ошибки — про reasoning_content",
              "reasoning_content" in str(error), str(error))
        from app.log_setup import current_log_path, flush_error_log

        log_path = flush_error_log()
        body = ""
        if log_path is not None and log_path.exists():
            body = log_path.read_text(encoding="utf-8")
            log_path.unlink()
        check("отказ попал в журнал",
              "отклонил запрос" in body and "strict-reasoning" in body,
              body[-200:] if body else "журнала нет")
        check("журнал не содержит текста диалога", "привет" not in body,
              body[-200:] if body else "")
        check("путь журнала известен", current_log_path() is not None)

        print("пустой ответ в режиме рассуждений (как в errors/2026-11-17):")
        from app.agent.policy import AgentLimits, Workspace
        from app.agent.runner import AgentRunner

        agent_client = ChatClient(
            api_key=API_KEY, model="reason-empty", base_url=base,
            thinking=True, thinking_style="deepseek",
        )
        with tempfile.TemporaryDirectory(prefix="mlvh_llm_") as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text(
                "строка один\nстрока два\n", encoding="utf-8"
            )
            limits = AgentLimits(max_rounds=6)
            try:
                reply = AgentRunner(
                    agent_client,
                    [
                        {"role": "system", "content": "тест"},
                        {"role": "user", "content": "что в a.txt?"},
                    ],
                    Workspace([root], limits),
                    limits,
                    context_tokens=100_000,
                    max_tokens=1000,
                ).run()
            except LLMError as exc:
                reply = None
                print(f"  ошибка агента: {exc}")
        check("после пустого ответа агент дочитал файл",
              reply is not None and "две строки" in reply.text,
              reply.text if reply else "None")
        with LOCK:
            modes = [
                (item.get("thinking") or {}).get("type")
                for item in SEEN
                if item.get("model") == "reason-empty"
            ]
        check("после шага без рассуждений thinking выключен",
              bool(modes) and modes[-1] == "disabled", str(modes))

        print("по умолчанию DeepSeek:")
        default = ChatClient(api_key="k", model="deepseek-flash")
        check("адрес DeepSeek", default.base_url == "https://api.deepseek.com",
              default.base_url)
        check("подпись DeepSeek", default.label == "DeepSeek", default.label)
        check("ключ из DEEPSEEK_API_KEY", default.key_env == "DEEPSEEK_API_KEY",
              default.key_env)
        check(
            "DeepSeek получает thinking",
            ChatClient(api_key="k", thinking=True).thinking_payload()
            == {"thinking": {"type": "enabled"}},
        )

        print("проверка /models:")
        try:
            url, models = probe_models(base, API_KEY)
        except ProbeError as exc:
            check("список моделей получен", False, str(exc))
        else:
            check("список моделей получен", "demo" in models, str(models))
            check("адрес /models верный", url.endswith("/models"), url)
        try:
            probe_models(base, "wrong")
        except ProbeError as exc:
            check("401 в проверке подключения", "401" in str(exc), str(exc))
        else:
            check("401 в проверке подключения", False)
    finally:
        server.shutdown()
        server.server_close()
    return 0


def _live(base_url: str, alias: str) -> int:
    from app.config import get_api_key, load_settings, normalize_base_url
    from app.llm_client import ChatClient, LLMError
    from app.llm_probe import ProbeError, probe_models

    settings = load_settings()
    model = settings.model
    if base_url:
        model.base_url = normalize_base_url(base_url)
    if alias:
        model.name = alias.strip()

    print(f"Модель: живой провайдер {model.provider_name()}, алиас «{model.name}»")
    key = get_api_key(model)
    if not key:
        print(f"  FAIL нет {model.key_env()} в .env")
        return 1

    try:
        url, models = probe_models(model.endpoint(), key)
    except ProbeError as exc:
        print(f"  FAIL проверка /models: {exc}")
        return 1
    print(f"  {url}: моделей {len(models)}")
    check("список моделей получен", bool(models), str(models[:5]))

    # Как в приложении: рассуждения тоже тратят токены, бюджет поднимаем.
    max_tokens = model.max_tokens
    if model.thinking:
        max_tokens = max(max_tokens, 2000)

    try:
        client = ChatClient(
            api_key=key,
            model=model.name,
            temperature=model.temperature,
            max_tokens=max_tokens,
            thinking=model.thinking,
            base_url=model.endpoint(),
            thinking_style=model.thinking_style,
            key_env=model.key_env(),
        )
        text, _, _, usage = _collect(
            client, [{"role": "user", "content": "Ответь одним словом: проверка."}]
        )
    except LLMError as exc:
        print(f"  FAIL {exc}")
        return 1

    check("ответ получен", bool(text.strip()), text)
    print(f"  ответ: {text.strip()[:200]}")
    if usage:
        print(f"  токены: {usage}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка клиента модели MLVH")
    parser.add_argument(
        "--live",
        action="store_true",
        help="живой запрос к провайдеру из config.json",
    )
    parser.add_argument(
        "--base-url",
        metavar="URL",
        help="для --live: другой адрес API (например https://openrouter.ai/api/v1)",
    )
    parser.add_argument("--model", metavar="ALIAS", help="для --live: другой алиас модели")
    args = parser.parse_args()

    if args.live or args.base_url:
        _live(args.base_url or "", args.model or "")
    else:
        _offline()

    print()
    if FAILED:
        print(f"Провалено: {len(FAILED)} — {', '.join(FAILED)}")
        return 1
    print("Проверка модели пройдена.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())