"""Журнал ошибок сессии: `errors/гггг-ММ-дд_чч-мм-сс.log`.

Пишем не всё подряд, а только то, что нужно для разбора: ординарные записи
(запросы к модели, инструменты) копятся в памяти, а файл создаётся в момент
первой ошибки — и в него попадает весь накопленный контекст сессии. Сессия
без ошибок не оставляет ни одного файла, пустых логов не бывает.

Каталог в `.gitignore`: это рабочий журнал, а не данные проекта.
"""

from __future__ import annotations

import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ERROR_DIR = ROOT / "errors"

# Сколько записей держим в памяти до первой ошибки. Один шаг агента — это
# 2–3 записи, так что хватает на длинный диалог.
BUFFER_RECORDS = 2000
ROOT_LEVEL = logging.INFO
FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
HANDLER_NAME = "mlvh-errors-buffer"


def _state() -> dict:
    """Состояние журнала: одно на процесс, сколько бы копий модуля ни было."""
    return globals().setdefault(
        "_STATE",
        {"created": False, "path": None, "file": None, "dir": None},
    )


def _drop_console_handlers() -> None:
    """В консоли ошибку и так видно: дублируем только то, что идёт в файл.

    Без этого `--ask` и проверки в tools печатали бы в stderr каждую строку
    журнала, включая обычные INFO о запросах. В GUI консоли нет вовсе,
    поэтому `lastResort` тоже выключаем: в файл он не пишет.

    Заодно гасим INFO сторонних библиотек (`httpx`): каждый HTTP-запрос
    дублирует нашу же строку и забивает журнал. Их предупреждения остаются.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            continue
        root.removeHandler(handler)
    logging.lastResort = None
    for name in ("httpx", "httpx2", "httpcore", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _make_handler(path: Path) -> logging.Handler:
    """Файл, который открывается только при первой записи (delay=True)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8", delay=True)
    handler.setLevel(ROOT_LEVEL)
    handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    return handler


def _write(records: list[logging.LogRecord], path: Path) -> bool:
    """Дописываем `records` в файл; при неудаче записи вернём False."""
    try:
        handler = _make_handler(path)
    except OSError:
        return False
    try:
        for record in records:
            handler.handle(record)
        if not records:
            path.touch()
    finally:
        handler.close()
    return True


class _FileTarget(logging.handlers.MemoryHandler):
    """Буфер журнала: INFO копит в памяти, WARNING+ сразу открывает файл."""

    def __init__(self) -> None:
        super().__init__(
            capacity=BUFFER_RECORDS,
            flushLevel=logging.WARNING,
            target=None,
        )
        self.name = HANDLER_NAME
        self.setLevel(ROOT_LEVEL)

    def emit(self, record: logging.LogRecord) -> None:
        state = _state()
        if record.levelno < logging.WARNING:
            super().emit(record)   # в память
            return
        if not state["created"]:
            # Первая ошибка: файл по её времени и весь накопленный контекст.
            with self.lock:
                records = list(self.buffer) + [record]
                self.buffer.clear()
            flush_error_log(records)
            return
        path = state["file"]
        if path is not None:
            _write([record], path)



def setup_logging(error_dir: Path | None = None) -> Path | None:
    """Включает буфер журнала. Возвращает каталог ошибок (None — не вышло)."""
    if logging.getHandlerByName(HANDLER_NAME) is not None:
        return _state()["dir"]

    directory = Path(error_dir) if error_dir is not None else ERROR_DIR
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    buffer = _FileTarget()
    root = logging.getLogger()
    root.setLevel(min(root.level, ROOT_LEVEL) if root.level else ROOT_LEVEL)
    _drop_console_handlers()
    root.addHandler(buffer)

    _state().update({"created": False, "path": None, "file": None, "dir": directory})
    # atexit здесь не нужен: выход без ошибки не должен создавать пустой файл.
    return directory


def flush_error_log(
    pending: list[logging.LogRecord] | None = None,
) -> Path | None:
    """Создаёт файл (время первой ошибки) и сбрасывает в него буфер.

    `pending` — записи, которые уже забрал обработчик (первая ошибка).
    Повторный вызов дописывает в тот же файл: одна сессия — один журнал.
    Ошибки записи гасим: журнал не должен ронять программу.
    """
    buffer = logging.getHandlerByName(HANDLER_NAME)
    state = _state()

    if pending is None:
        records: list[logging.LogRecord] = []
        if buffer is not None:
            with buffer.lock:
                records = list(buffer.buffer)
                buffer.buffer.clear()
    else:
        records = list(pending)

    if not state["created"]:
        directory = state["dir"] or ERROR_DIR
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = directory / f"{stamp}.log"
        if not _write(records, path):
            if buffer is not None:
                buffer.buffer.extend(records)   # не потеряли: сбросим позже
            return None
        state.update({"created": True, "path": path, "file": path})
        return path

    path = state["file"]
    if path is None or not _write(records, path):
        if buffer is not None:
            buffer.buffer.extend(records)
        return None
    return path


def current_log_path() -> Path | None:
    """Файл ошибок этой сессии; None — ошибок ещё не было."""
    return _state()["path"]


def error_log_hint() -> str:
    """Короткая подсказка для окна и консоли: где смотреть подробности."""
    path = current_log_path()
    return f"Подробности: {path}" if path is not None else ""


def discard_empty_log() -> None:
    """Убирает пустой журнал: ошибку пользователю уже показали в чате."""
    state = _state()
    path = state["path"]
    state.update({"created": False, "path": None, "file": None})
    if path is None:
        return
    try:
        if path.exists() and path.stat().st_size == 0:
            path.unlink()
    except OSError:
        pass


def reset_for_tests() -> None:
    """Снимает обработчик с root — только для проверок в одном процессе."""
    buffer = logging.getHandlerByName(HANDLER_NAME)
    if buffer is not None:
        logging.getLogger().removeHandler(buffer)
    _state().update({"created": False, "path": None, "file": None, "dir": None})

