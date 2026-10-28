"""MCP-клиент на стандартной библиотеке: JSON-RPC 2.0 через stdio.

Протокол — https://modelcontextprotocol.io/specification/2025-06-18:
сообщения идут по одному JSON на строку (без Content-Length), UTF-8;
stdout сервера — только сообщения протокола, stderr — его логи.

Нам нужна одна возможность MCP — инструменты (tools):
`initialize` → `tools/list` → `tools/call`. Prompts, resources, sampling
и HTTP-транспорт не поддерживаются: для агента MLVH они не нужны.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, NoReturn, Sequence

from app import __version__ as MLVH_VERSION
from app.agent.policy import ToolError

PROTOCOL_VERSION = "2025-06-18"
CLIENT_NAME = "MLVH"
CLIENT_VERSION = MLVH_VERSION

# Путь по умолчанию: <корень проекта>/logs/mcp.log. В тестах передаём None.
DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "mcp.log"

# Жёсткие пределы: настройками не обходятся.
MAX_LINE_BYTES = 20 * 1024 * 1024     # одна строка ответа сервера
MAX_SCHEMA_CHARS = 20_000             # одна схема инструмента в контекст
STDERR_TAIL = 40                      # строк stderr держим в памяти
LOG_LIMIT_BYTES = 1_000_000           # дальше файл лога начинается заново
CLOSE_WAIT_SECONDS = 2.0


def _short(text: str, limit: int = 180) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _slug(value: str, fallback: str) -> str:
    """id сервера для префикса инструмента: только строчные [a-z0-9_-]."""
    text = re.sub(r"[^a-z0-9_-]+", "_", str(value).strip().casefold())
    text = text.strip("_-")
    return text or fallback


def _tool_part(value: str) -> str:
    """Имя инструмента сервера: сохраняем регистр, но чистим символы."""
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    return text.strip("_") or "tool"


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


@dataclass
class McpServer:
    """Описание одного MCP-сервера из config.json."""

    id: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    enabled: bool = True
    timeout: float = 0.0          # 0 — общий таймаут сессии

    @classmethod
    def from_dict(cls, data: object) -> "McpServer | None":
        """Битая запись не роняет запуск: возвращаем None и работаем дальше."""
        if not isinstance(data, dict):
            return None

        server_id = str(data.get("id") or "").strip()
        command = str(data.get("command") or "").strip()
        if not server_id or not command:
            return None

        raw_args = data.get("args")
        if isinstance(raw_args, str):
            args = raw_args.split()
        elif isinstance(raw_args, (list, tuple)):
            args = [str(item) for item in raw_args]
        else:
            args = []

        env: dict[str, str] = {}
        raw_env = data.get("env")
        if isinstance(raw_env, dict):
            for key, value in raw_env.items():
                key_text = str(key).strip()
                if key_text:
                    env[key_text] = str(value)

        raw_enabled = data.get("enabled", True)
        if isinstance(raw_enabled, str):
            enabled = raw_enabled.strip().casefold() in ("1", "true", "да", "yes")
        else:
            enabled = bool(raw_enabled)

        return cls(
            id=server_id,
            command=command,
            args=args,
            env=env,
            cwd=str(data.get("cwd") or "").strip(),
            enabled=enabled,
            timeout=max(0.0, min(600.0, _as_float(data.get("timeout"), 0.0))),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "command": self.command,
            "args": list(self.args),
            "env": dict(self.env),
            "cwd": self.cwd,
            "enabled": self.enabled,
            "timeout": int(self.timeout) if self.timeout else 0,
        }

    def label(self) -> str:
        """Строка для списка в окне: «id — command args»."""
        command = " ".join([self.command, *self.args]).strip()
        return _short(f"{self.id} — {command}", 100)


@dataclass
class McpLimits:
    """Пределы MCP на один вопрос агента."""

    timeout: float = 60.0          # один tools/call
    start_timeout: float = 120.0   # запуск + initialize + tools/list
    max_tools: int = 128           # схем инструментов на всю сессию
    max_restarts: int = 1          # перезапусков одного сервера на вопрос
    max_result_chars: int = 200_000

    @classmethod
    def from_settings(cls, settings: object) -> "McpLimits":
        """Строим из AgentSettings, не импортируя config (чтобы не было цикла)."""
        timeout = _as_float(getattr(settings, "mcp_timeout", 60), 60.0)
        start = _as_float(getattr(settings, "mcp_start_timeout", 120), 120.0)
        return cls(
            timeout=max(5.0, min(600.0, timeout)),
            start_timeout=max(5.0, min(600.0, start)),
        )


@dataclass
class McpResult:
    """Результат вызова MCP-инструмента: content — модели, summary — в чат."""

    content: str
    summary: str
    ok: bool = True


class McpError(RuntimeError):
    """Сбой MCP-сервера или протокола; текст уходит модели как ошибка."""


_LOG_LOCK = threading.Lock()


def _creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _build_argv(command: str, args: Iterable[str]) -> list[str]:
    """Команда запуска. На Windows batch-файлы запускаем через cmd.exe.

    Путь к `.cmd` передаём отдельным аргументом и не добавляем своих кавычек:
    `list2cmdline` уже экранирует пробелы, а лишние кавычки ломают разбор
    (`The network path was not found`).
    """
    argv = [str(command), *[str(arg) for arg in args]]
    resolved = shutil.which(argv[0])
    if resolved:
        argv[0] = resolved
    if os.name == "nt" and argv[0].casefold().endswith((".cmd", ".bat")):
        comspec = os.environ.get("COMSPEC") or "cmd.exe"
        argv = [comspec, "/d", "/c", *argv]
    return argv


def _kill_tree(process: subprocess.Popen) -> None:
    """Убиваем всё дерево: npx запускает node, и одного kill мало."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                timeout=10,
                creationflags=_creation_flags(),
            )
            return
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            return
        except (OSError, ProcessLookupError):
            pass
    try:
        process.kill()
    except OSError:
        pass


class McpClient:
    """Один MCP-сервер: процесс, JSON-RPC по stdio и фоновые читатели.

    Ошибки протокола (JSON-RPC `error`) сервер не ломают. Падение процесса
    и таймаут помечают клиента `broken`: сессия перезапустит его перед
    следующим вызовом, но **не повторит** уже отправленный вызов.
    """

    def __init__(
        self,
        server: McpServer,
        limits: McpLimits | None = None,
        *,
        log_path: Path | str | None = DEFAULT_LOG_PATH,
    ) -> None:
        self.server = server
        self.limits = limits or McpLimits()
        self._log_path = log_path
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._write_lock = threading.Lock()
        self._lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._next_id = 1
        self._broken = False
        self._initialized = False
        self._protocol = PROTOCOL_VERSION
        self._server_info: dict = {}
        self._stderr: deque[str] = deque(maxlen=STDERR_TAIL)

    # --- состояние -------------------------------------------------------
    @property
    def broken(self) -> bool:
        return self._broken

    @property
    def alive(self) -> bool:
        """Процесс ещё работает: `broken` выставляется только при сбое запроса."""
        process = self._process
        return process is not None and process.poll() is None

    @property
    def protocol(self) -> str:
        return self._protocol

    @property
    def server_info(self) -> dict:
        return dict(self._server_info)

    @property
    def stderr_tail(self) -> list[str]:
        with self._lock:
            return list(self._stderr)

    # --- запуск и остановка ----------------------------------------------
    def start(self, should_stop: Callable[[], bool] | None = None) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                return
            # Процесс уже умер: закрываем пайпы и потоки, чтобы не течь.
            self._shutdown()

        self._broken = False
        self._initialized = False
        with self._lock:
            self._stderr.clear()
        self._drain_queue()

        argv = _build_argv(self.server.command, self.server.args)
        env = os.environ.copy()
        # Ключ DeepSeek сторонним программам не передаём.
        env.pop("DEEPSEEK_API_KEY", None)
        env.update(self.server.env)

        cwd = self.server.cwd or None
        if cwd and not Path(cwd).is_dir():
            raise McpError(
                f"MCP-сервер «{self.server.id}»: рабочая папка не найдена: {cwd}"
            )

        try:
            self._process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=env,
                creationflags=_creation_flags(),
                start_new_session=(os.name != "nt"),
            )
        except OSError as exc:
            self._process = None
            raise McpError(
                f"MCP-сервер «{self.server.id}» не запустился ({argv[0]}): {exc}"
            ) from exc

        self._reader = threading.Thread(
            target=self._read_stdout,
            name=f"mcp-{self.server.id}-out",
            daemon=True,
        )
        self._stderr_reader = threading.Thread(
            target=self._read_stderr,
            name=f"mcp-{self.server.id}-err",
            daemon=True,
        )
        self._reader.start()
        self._stderr_reader.start()

        try:
            result = self._request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
                },
                timeout=self.limits.start_timeout,
                should_stop=should_stop,
            )
        except Exception:
            self._shutdown()
            raise

        if isinstance(result, dict):
            self._protocol = str(result.get("protocolVersion") or PROTOCOL_VERSION)
            info = result.get("serverInfo")
            self._server_info = info if isinstance(info, dict) else {}

        self._notify("notifications/initialized")
        self._initialized = True
        self._log(
            "готов: "
            f"{self._server_info.get('name', '?')} "
            f"{self._server_info.get('version', '')}, протокол {self._protocol}"
        )

    def stop(self) -> None:
        self._shutdown()

    def restart(self, should_stop: Callable[[], bool] | None = None) -> None:
        self._shutdown()
        self._broken = False
        self.start(should_stop)

    def _shutdown(self) -> None:
        """Сериализуем закрытие: его могут звать GUI, рабочий поток и reader."""
        with self._stop_lock:
            self._shutdown_unlocked()

    def _shutdown_unlocked(self) -> None:
        """Закрываем stdin, ждём выхода, при необходимости убиваем дерево."""
        self._initialized = False
        process = self._process
        self._process = None
        if process is not None:
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                process.wait(timeout=CLOSE_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                _kill_tree(process)
                try:
                    process.wait(timeout=CLOSE_WAIT_SECONDS)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            except OSError:
                pass
            for stream in (process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except (OSError, ValueError):
                    pass

        current = threading.current_thread()
        for thread in (self._reader, self._stderr_reader):
            if thread is not None and thread is not current:
                thread.join(timeout=CLOSE_WAIT_SECONDS)
        self._reader = None
        self._stderr_reader = None
        self._drain_queue()

    # --- протокол ---------------------------------------------------------
    def list_tools(self, should_stop: Callable[[], bool] | None = None) -> list[dict]:
        if not self._initialized:
            raise McpError(f"MCP-сервер «{self.server.id}» не инициализирован")
        tools: list[dict] = []
        cursor = ""
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = self._request(
                "tools/list",
                params,
                timeout=self.limits.start_timeout,
                should_stop=should_stop,
            )
            page = result.get("tools")
            if isinstance(page, list):
                tools.extend(item for item in page if isinstance(item, dict))
            next_cursor = result.get("nextCursor")
            if (
                not isinstance(next_cursor, str)
                or not next_cursor
                or next_cursor == cursor
            ):
                break
            cursor = next_cursor
            if len(tools) > 1000:
                self._log("tools/list: больше 1000 инструментов, останавливаюсь")
                break
        return tools

    def call_tool(
        self,
        name: str,
        arguments: dict | None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict:
        if not self._initialized:
            raise McpError(f"MCP-сервер «{self.server.id}» не инициализирован")
        timeout = self.server.timeout or self.limits.timeout
        return self._request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            timeout=timeout,
            should_stop=should_stop,
        )

    def _request(
        self,
        method: str,
        params: dict | None,
        *,
        timeout: float,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict:
        if self._process is None:
            raise McpError(f"MCP-сервер «{self.server.id}» не запущен")

        with self._lock:
            request_id = self._next_id
            self._next_id += 1

        payload: dict = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)

        deadline = time.monotonic() + max(0.1, timeout)
        while True:
            if should_stop is not None and should_stop():
                self._notify(
                    "notifications/cancelled",
                    {"requestId": request_id, "reason": "client cancelled"},
                )
                raise ToolError("остановлено")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._fail(f"не ответил на «{method}» за {timeout:.0f} с")

            try:
                kind, value = self._queue.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue

            if kind == "eof":
                self._fail(f"процесс завершился во время «{method}»")
            if kind == "fatal":
                self._fail(str(value))
            if kind != "response" or not isinstance(value, dict):
                continue
            if value.get("id") != request_id:
                self._log(f"лишний ответ id={value.get('id')!r} на «{method}»")
                continue

            if "error" in value:
                error = value.get("error")
                if isinstance(error, dict):
                    message = str(error.get("message") or "ошибка")
                    code = error.get("code")
                    suffix = f" (код {code})" if code is not None else ""
                    raise McpError(f"«{method}»: {message}{suffix}")
                raise McpError(f"«{method}»: {error}")

            result = value.get("result")
            return result if isinstance(result, dict) else {}

    def _send(self, payload: dict) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise McpError(f"MCP-сервер «{self.server.id}» не запущен")
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        error: Exception | None = None
        with self._write_lock:
            try:
                process.stdin.write(line.encode("utf-8"))
                process.stdin.flush()
            except (OSError, ValueError) as exc:
                error = exc
        if error is not None:
            self._fail(f"не удалось записать в stdin: {error}")

    def _notify(self, method: str, params: dict | None = None) -> None:
        payload: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        try:
            self._send(payload)
        except (McpError, OSError, ValueError) as exc:
            self._log(f"нотификация {method} не отправлена: {exc}")

    def _reply(self, request_id: object, result: dict) -> None:
        try:
            self._send({"jsonrpc": "2.0", "id": request_id, "result": result})
        except (McpError, OSError, ValueError) as exc:
            self._log(f"ответ на запрос сервера не отправлен: {exc}")

    def _reply_error(self, request_id: object, code: int, message: str) -> None:
        try:
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": code, "message": message},
                }
            )
        except (McpError, OSError, ValueError) as exc:
            self._log(f"ошибка на запрос сервера не отправлена: {exc}")

    # --- фоновые читатели -------------------------------------------------
    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            self._queue.put(("eof", None))
            return
        try:
            for raw in iter(process.stdout.readline, b""):
                line = raw.strip()
                if not line:
                    continue
                if len(line) > MAX_LINE_BYTES:
                    self._queue.put(("fatal", "ответ сервера больше 20 МБ"))
                    return
                try:
                    message = json.loads(line.decode("utf-8", "replace"))
                except json.JSONDecodeError:
                    self._log(f"stdout (не JSON): {line[:300]!r}")
                    continue
                self._dispatch(message)
        except (OSError, ValueError):
            pass
        finally:
            self._queue.put(("eof", None))

    def _dispatch(self, message: object) -> None:
        if isinstance(message, list):
            for item in message:
                self._dispatch(item)
            return
        if not isinstance(message, dict):
            self._log(f"stdout: неожиданное сообщение {str(message)[:200]}")
            return

        method = message.get("method")
        if method:
            if "id" in message:
                self._answer_server(message)
            else:
                self._log(f"нотификация {method}")
            return

        if "id" in message:
            self._queue.put(("response", message))
            return
        self._log(f"stdout: сообщение без method и id: {str(message)[:200]}")

    def _answer_server(self, message: dict) -> None:
        """Сервер тоже может спрашивать: отвечаем, чтобы он не завис."""
        method = str(message.get("method") or "")
        request_id = message.get("id")
        if method == "ping":
            self._reply(request_id, {})
        elif method == "roots/list":
            self._reply(request_id, {"roots": []})
        else:
            self._reply_error(request_id, -32601, f"метод не поддерживается: {method}")

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            for raw in iter(process.stderr.readline, b""):
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                with self._lock:
                    self._stderr.append(line)
                self._log(f"stderr: {line}")
        except (OSError, ValueError):
            pass

    # --- мелочи -----------------------------------------------------------
    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _fail(self, reason: str) -> "NoReturn":
        """Сервер сломан: запоминаем причину, освобождаем ресурсы и сообщаем."""
        self._broken = True
        tail = [line for line in self.stderr_tail if line][-3:]
        message = f"MCP-сервер «{self.server.id}»: {reason}"
        if tail:
            message += " (stderr: " + " | ".join(_short(line, 200) for line in tail) + ")"
        self._log(reason)
        self._shutdown()
        raise McpError(message)

    def _log(self, text: str) -> None:
        if not self._log_path:
            return
        try:
            path = Path(self._log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with _LOG_LOCK:
                if path.exists() and path.stat().st_size > LOG_LIMIT_BYTES:
                    path.write_text("--- лог начат заново ---\n", encoding="utf-8")
                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(f"[{stamp}] {self.server.id}: {text}\n")
        except (OSError, ValueError):
            pass


def _block_text(block: dict) -> str:
    """Один блок content → текст. Картинки и байты в контекст не отдаём."""
    kind = str(block.get("type") or "")
    if kind == "text":
        return str(block.get("text") or "")
    if kind == "image":
        data = block.get("data")
        size = len(data) * 3 // 4 if isinstance(data, str) else 0
        mime = str(block.get("mimeType") or "image")
        return f"[изображение {mime}, ~{size // 1024} КБ — не показываю]"
    if kind == "audio":
        mime = str(block.get("mimeType") or "audio")
        return f"[аудио {mime} — не показываю]"
    if kind == "resource":
        resource = block.get("resource")
        if isinstance(resource, dict):
            uri = str(resource.get("uri") or "?")
            if isinstance(resource.get("text"), str):
                return f"[ресурс {uri}]\n{resource['text']}"
            blob = resource.get("blob")
            size = len(blob) * 3 // 4 if isinstance(blob, str) else 0
            return f"[ресурс {uri}, ~{size // 1024} КБ — не показываю]"
        return "[ресурс]"
    if kind == "resource_link":
        return f"[ссылка на ресурс {block.get('uri') or '?'}]"
    return json.dumps(block, ensure_ascii=False)[:1000]


def _result_text(result: dict) -> str:
    parts: list[str] = []
    blocks = result.get("content")
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            text = _block_text(block)
            if text.strip():
                parts.append(text)
    if not parts and result.get("structuredContent") is not None:
        parts.append(
            json.dumps(result["structuredContent"], ensure_ascii=False, indent=2)
        )
    return "\n\n".join(parts).strip()


def _unique_prefix(base: str, used: set[str]) -> str:
    prefix = f"mcp_{base}"
    if prefix not in used:
        used.add(prefix)
        return prefix
    counter = 2
    while f"{prefix}_{counter}" in used:
        counter += 1
    prefix = f"{prefix}_{counter}"
    used.add(prefix)
    return prefix


def _unique_tool_name(prefix: str, tool_name: str, used: set[str]) -> str:
    """OpenAI-совместимое имя ≤64 символов, уникальное в пределах сессии."""
    base = f"{prefix}_{_tool_part(tool_name)}"
    if len(base) > 64:
        base = base[:64]
    name = base
    counter = 2
    while name in used:
        suffix = f"_{counter}"
        name = base[: 64 - len(suffix)] + suffix
        counter += 1
    used.add(name)
    return name


class McpSession:
    """Серверы MCP: схемы, вызовы и закрытие.

    Серверы запускаются лениво при первом `schemas()` — это происходит
    в рабочем потоке агента, а не в интерфейсе. Один сбой одного сервера
    не мешает остальным: причина попадает в `problems`.

    Сессия может жить дольше одного вопроса: тогда владелец зовёт
    `begin_question()` перед каждым вопросом и `close()` на выходе.
    `AgentRunner` делает это сам; `--ask` просто закрывает сессию в конце.
    """

    def __init__(
        self,
        servers: Sequence[McpServer] | None = None,
        limits: McpLimits | None = None,
        *,
        log_path: Path | str | None = DEFAULT_LOG_PATH,
        client_factory: Callable[..., McpClient] | None = None,
    ) -> None:
        self.servers = [
            server for server in (servers or []) if isinstance(server, McpServer)
        ]
        self.limits = limits or McpLimits()
        self.problems: list[str] = []
        self._log_path = log_path
        self._factory = client_factory or McpClient
        self._clients: list[tuple[McpClient, str]] = []
        self._map: dict[str, tuple[McpClient, str]] = {}
        self._restarts: dict[int, int] = {}
        self._schemas: list[dict] | None = None
        self._closed = False

    # --- границы вопросов -------------------------------------------------
    def begin_question(self, should_stop: Callable[[], bool] | None = None) -> None:
        """Новый вопрос: причины сбоев и бюджет перезапусков — заново.

        Серверы, умершие между вопросами, поднимаем сразу: иначе первый
        же вызов инструмента упал бы вхолостую. Если поднять не удалось,
        причина попадёт в `problems`, а остальные серверы продолжат работу.
        """
        self.problems.clear()
        if self._closed:
            return
        for client, prefix in list(self._clients):
            self._restarts[id(client)] = self.limits.max_restarts
            if client.alive and not client.broken:
                continue
            try:
                client.restart(should_stop)
                self._refresh_client(client, prefix)
            except ToolError:
                raise
            except Exception as exc:
                self.problems.append(
                    f"«{client.server.id}»: не поднялся к новому вопросу: {exc}"
                )

    # --- что видит агент --------------------------------------------------
    @property
    def server_ids(self) -> list[str]:
        return [server.id for server in self.servers if server.enabled]

    def owns(self, name: str) -> bool:
        return name in self._map

    def names(self) -> tuple[str, ...]:
        return tuple(self._map)

    def schemas(
        self, should_stop: Callable[[], bool] | None = None
    ) -> list[dict]:
        """Схемы инструментов; при первом вызове запускает серверы."""
        if self._closed:
            return []
        if self._schemas is not None:
            return list(self._schemas)

        self._schemas = []
        used_prefixes: set[str] = set()
        used_names: set[str] = set()

        for index, server in enumerate(self.servers):
            if not server.enabled:
                continue
            if len(self._map) >= self.limits.max_tools:
                self.problems.append(
                    f"«{server.id}»: пропущен — лимит {self.limits.max_tools} инструментов"
                )
                break

            client: McpClient | None = None
            try:
                client = self._factory(server, self.limits, log_path=self._log_path)
                client.start(should_stop)
                tools = client.list_tools(should_stop)
            except ToolError:
                # Пользователь остановил запрос: всё закрываем, инструментов нет.
                if client is not None:
                    try:
                        client.stop()
                    except Exception:
                        pass
                self.close()
                return []
            except Exception as exc:
                if client is not None:
                    try:
                        client.stop()
                    except Exception:
                        pass
                self.problems.append(f"«{server.id}»: {exc}")
                continue

            base = _slug(server.id, f"server{index + 1}")
            prefix = _unique_prefix(base, used_prefixes)
            self._clients.append((client, prefix))
            self._restarts[id(client)] = self.limits.max_restarts
            self._register_tools(client, prefix, tools, used_names)

        return list(self._schemas)

    def _register_tools(
        self,
        client: McpClient,
        prefix: str,
        tools: list[dict],
        used_names: set[str],
    ) -> None:
        assert self._schemas is not None
        for tool in tools:
            if len(self._map) >= self.limits.max_tools:
                self.problems.append(
                    f"«{client.server.id}»: показаны не все инструменты — "
                    f"лимит {self.limits.max_tools}"
                )
                break
            tool_name = str(tool.get("name") or "").strip()
            if not tool_name:
                continue

            public = _unique_tool_name(prefix, tool_name, used_names)
            description = _short(
                str(tool.get("description") or tool.get("title") or ""), 2000
            )
            parameters = tool.get("inputSchema")
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                parameters = {"type": "object", "properties": {}}
            schema = {
                "type": "function",
                "function": {
                    "name": public,
                    "description": f"[{client.server.id}] {description}".strip(),
                    "parameters": parameters,
                },
            }
            if len(json.dumps(schema, ensure_ascii=False)) > MAX_SCHEMA_CHARS:
                self.problems.append(
                    f"«{client.server.id}»/{tool_name}: схема больше "
                    f"{MAX_SCHEMA_CHARS} символов — пропущена"
                )
                continue

            self._map[public] = (client, tool_name)
            self._schemas.append(schema)

    # --- вызов ------------------------------------------------------------
    def call(
        self,
        name: str,
        arguments: dict | None,
        should_stop: Callable[[], bool] | None = None,
    ) -> McpResult:
        entry = self._map.get(name)
        if entry is None:
            raise ToolError(f"неизвестный MCP-инструмент «{name}»")
        client, tool_name = entry

        if client.broken:
            left = self._restarts.get(id(client), 0)
            if left <= 0:
                return McpResult(
                    content=(
                        f"Ошибка: MCP-сервер «{client.server.id}» недоступен "
                        "после сбоя. Вызов не выполнен."
                    ),
                    summary=_short(f"MCP {client.server.id}: сервер недоступен"),
                    ok=False,
                )
            self._restarts[id(client)] = left - 1
            prefix = self._prefix_for(client)
            try:
                client.restart(should_stop)
                if prefix is not None:
                    self._refresh_client(client, prefix)
            except ToolError:
                raise
            except Exception as exc:
                return McpResult(
                    content=(
                        f"Ошибка: MCP-сервер «{client.server.id}» "
                        f"не перезапустился: {exc}"
                    ),
                    summary=_short(f"MCP {client.server.id}: перезапуск не удался"),
                    ok=False,
                )

            entry = self._map.get(name)
            if entry is None:
                return McpResult(
                    content=(
                        f"Ошибка: после перезапуска у сервера "
                        f"«{client.server.id}» нет инструмента «{tool_name}»."
                    ),
                    summary=_short(f"MCP {client.server.id}: инструмент пропал"),
                    ok=False,
                )
            client, tool_name = entry

        try:
            result = client.call_tool(tool_name, arguments, should_stop)
        except ToolError:
            raise
        except McpError as exc:
            return McpResult(
                content=f"Ошибка: {exc}",
                summary=_short(f"MCP {client.server.id}: сбой — {exc}"),
                ok=False,
            )
        except Exception as exc:  # чужой сервер не должен ронять агентский цикл
            return McpResult(
                content=f"Ошибка MCP-сервера: {exc}",
                summary=_short(f"MCP {client.server.id}: сбой — {exc}"),
                ok=False,
            )

        text = _result_text(result)
        if len(text) > self.limits.max_result_chars:
            text = text[: self.limits.max_result_chars] + "\n[…обрезано: результат MCP]"
        is_error = bool(result.get("isError"))
        if is_error:
            text = text or "(сервер не объяснил ошибку)"
            return McpResult(
                content=f"Ошибка: {text}",
                summary=_short(f"MCP {client.server.id}: {tool_name} — ошибка"),
                ok=False,
            )
        return McpResult(
            content=text or "(пустой результат)",
            summary=_short(f"MCP {client.server.id}: {tool_name}"),
            ok=True,
        )

    def _prefix_for(self, client: McpClient) -> str | None:
        for item_client, prefix in self._clients:
            if item_client is client:
                return prefix
        return None

    def _refresh_client(self, client: McpClient, prefix: str) -> None:
        """После перезапуска список инструментов мог измениться."""
        try:
            tools = client.list_tools()
        except Exception as exc:
            self.problems.append(
                f"«{client.server.id}»: список инструментов не обновился: {exc}"
            )
            return

        stale = {
            name for name, (owner, _tool) in self._map.items() if owner is client
        }
        for name in stale:
            self._map.pop(name, None)
        if self._schemas is not None:
            self._schemas = [
                schema
                for schema in self._schemas
                if schema.get("function", {}).get("name") not in stale
            ]
        used = set(self._map)
        self._register_tools(client, prefix, tools, used)

    def close(self) -> None:
        """Закрываем все процессы; повторный вызов безопасен."""
        self._closed = True
        for client, _prefix in self._clients:
            try:
                client.stop()
            except Exception:
                pass
        self._clients.clear()
        self._map.clear()
        self._restarts.clear()
        self._schemas = None


def mcp_servers_from_settings(settings: object) -> list[McpServer]:
    """Серверы из AgentSettings: битые записи пропускаем, id не повторяем."""
    raw = getattr(settings, "mcp_servers", [])
    if not isinstance(raw, list):
        return []
    servers: list[McpServer] = []
    seen: set[str] = set()
    for item in raw:
        server = McpServer.from_dict(item)
        if server is None:
            continue
        key = server.id.casefold()
        if key in seen:
            continue
        seen.add(key)
        servers.append(server)
    return servers
