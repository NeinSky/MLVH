"""Политика доступа агента: только чтение и только в разрешённых папках.

Все пути сначала разрешаются через `Path.resolve()` (раскрываются `..` и
симлинки), а затем проверяются на принадлежность одному из корней. Ошибка —
это отказ, а не «попробуем и посмотрим».
"""

from __future__ import annotations

import fnmatch
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

# --- жёсткие пределы, которые нельзя обойти настройками --------------------
BINARY_SNIFF_BYTES = 8192
HARD_FILE_BYTES = 50 * 1024 * 1024          # больше 50 МБ не открываем вообще
SEARCH_MAX_FILES = 2000
SEARCH_MAX_BYTES = 20 * 1024 * 1024
SEARCH_LINE_CHARS = 240

# Секреты не читаем, даже если папка разрешена явно.
SECRET_FILES = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_ed25519",
    ".git-credentials",
    "credentials",
    "credentials.*",
    "secrets.*",
    "*.kdbx",
)
SECRET_DIRS = (".ssh", ".gnupg")

# Служебные каталоги не обходим при поиске: там гигабайты и мусор.
SKIP_DIRS = (
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "models",
    "wheels",
    ".idea",
    ".vscode",
)

# Корни этих каталогов нельзя отдать агенту целиком.
SYSTEM_ENV = ("WINDIR", "ProgramFiles", "ProgramFiles(x86)", "ProgramData")


class ToolError(RuntimeError):
    """Ошибка инструмента: текст увидит модель, поэтому он по-русски и короткий."""


@dataclass(frozen=True)
class AgentLimits:
    """Снимок лимитов агента: настройки могут поменяться, запрос — нет."""

    max_rounds: int = 6
    duplicate_limit: int = 2
    max_tool_calls: int = 12
    max_seconds: float = 120.0
    max_file_bytes: int = 262_144
    max_read_lines: int = 400
    max_tool_chars: int = 400_000

    @classmethod
    def from_settings(cls, settings: object) -> "AgentLimits":
        """Строим из AgentSettings, не импортируя config (чтобы не было цикла)."""
        return cls(
            max_rounds=int(getattr(settings, "max_rounds", 6)),
            duplicate_limit=int(getattr(settings, "duplicate_limit", 2)),
            max_tool_calls=int(getattr(settings, "max_tool_calls", 12)),
            max_seconds=float(getattr(settings, "max_seconds", 120)),
            max_file_bytes=int(getattr(settings, "max_file_bytes", 262_144)),
            max_read_lines=int(getattr(settings, "max_read_lines", 400)),
            max_tool_chars=int(getattr(settings, "max_tool_chars", 400_000)),
        )


def validate_root(raw: str | Path) -> tuple[Path | None, str]:
    """Проверяем папку, которую пользователь хочет открыть агенту.

    Возвращаем (разрешённый путь, "") или (None, причина отказа).
    """
    text = str(raw).strip()
    if not text:
        return None, "пустой путь"

    lowered = text.casefold()
    if os.name == "nt" and (lowered.startswith("\\\\?\\") or lowered.startswith("\\\\.\\")):
        return None, "служебные пути Windows не поддерживаются"

    try:
        path = Path(text).expanduser()
        resolved = path.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"путь не разрешается: {exc}"

    if not resolved.exists():
        return None, "папка не найдена"
    if not resolved.is_dir():
        return None, "это не папка"
    if resolved.parent == resolved:
        return None, "корень диска нельзя открыть агенту — выберите папку внутри"

    if os.name == "nt":
        for name in SYSTEM_ENV:
            value = os.environ.get(name)
            if not value:
                continue
            try:
                system = Path(value).resolve()
            except (OSError, RuntimeError):
                continue
            if resolved == system or resolved.is_relative_to(system):
                return None, f"системная папка {system} закрыта"
    return resolved, ""


def is_secret_name(name: str) -> bool:
    """Файл похож на секрет: его не читаем и не показываем."""
    lowered = name.casefold()
    return any(fnmatch.fnmatchcase(lowered, pattern) for pattern in SECRET_FILES)


def is_secret_dir(name: str) -> bool:
    return name.casefold() in SECRET_DIRS


def is_skipped_dir(name: str) -> bool:
    return name.casefold() in SKIP_DIRS


def _looks_binary(sample: bytes) -> bool:
    return b"\x00" in sample


def _detect_encoding(sample: bytes) -> str:
    """UTF-8, затем cp1251 для старых русских файлов, в крайнем случае latin-1."""
    if _looks_binary(sample):
        raise ToolError("файл выглядит двоичным — читать его нельзя")
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            sample.decode(encoding)
        except UnicodeDecodeError:
            continue
        return encoding
    return "latin-1"


def _human_size(size: int) -> str:
    if size < 1024:
        return f"{size} Б"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} КБ"
    return f"{size / (1024 * 1024):.1f} МБ"


class Workspace:
    """Разрешённые папки и доступные в них операции чтения."""

    def __init__(
        self,
        roots: Iterable[str | Path],
        limits: AgentLimits | None = None,
    ) -> None:
        self.limits = limits or AgentLimits()
        self._roots: list[Path] = []
        self.skipped: list[tuple[str, str]] = []
        seen: set[str] = set()

        for raw in roots:
            resolved, error = validate_root(raw)
            if resolved is None:
                self.skipped.append((str(raw), error))
                continue
            key = os.path.normcase(str(resolved))
            if key in seen:
                continue
            seen.add(key)
            self._roots.append(resolved)

    def __bool__(self) -> bool:
        return bool(self._roots)

    @property
    def roots(self) -> tuple[Path, ...]:
        return tuple(self._roots)

    def default_root(self) -> Path:
        if not self._roots:
            raise ToolError("нет доступных рабочих папок")
        return self._roots[0]

    # --- проверка путей --------------------------------------------------
    def _check_inside(self, target: Path) -> Path:
        for root in self._roots:
            if target == root or target.is_relative_to(root):
                return target
        raise ToolError("путь вне разрешённых рабочих папок")

    @staticmethod
    def _reject_special(target: Path) -> None:
        """NTFS-потоки (`file.txt:stream`) и прочие двоеточия мимо диска."""
        if os.name != "nt":
            return
        text = str(target)
        rest = text[2:] if len(text) > 1 and text[1] == ":" else text
        if ":" in rest:
            raise ToolError("потоки NTFS и служебные имена не поддерживаются")

    def resolve(self, raw: object, *, must_exist: bool = False) -> Path:
        """Приводим аргумент модели к безопасному абсолютному пути."""
        text = str(raw or "").strip()
        if not text:
            text = "."
        if "\x00" in text:
            raise ToolError("недопустимый путь")
        if os.name == "nt" and (text.startswith("\\\\?\\") or text.startswith("\\\\.\\")):
            raise ToolError("служебные пути Windows не поддерживаются")

        try:
            candidate = Path(text).expanduser()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ToolError(f"некорректный путь: {exc}") from exc
        if candidate.is_absolute():
            try:
                target = candidate.resolve()
            except (OSError, RuntimeError, ValueError) as exc:
                raise ToolError(f"путь не разрешается: {exc}") from exc
            self._reject_special(target)
            self._check_inside(target)
            if must_exist and not target.exists():
                raise ToolError(f"файл не найден: {target}")
            return target

        # Относительный путь ищем во всех корнях: удобно, когда папок несколько.
        for root in self._roots:
            try:
                target = (root / candidate).resolve()
            except (OSError, RuntimeError, ValueError):
                continue
            try:
                self._check_inside(target)
            except ToolError:
                continue
            if target.exists():
                self._reject_special(target)
                return target

        target = (self.default_root() / candidate).resolve()
        self._reject_special(target)
        self._check_inside(target)
        if must_exist:
            raise ToolError(f"файл не найден: {target}")
        return target

    def _ensure_file(self, target: Path) -> str:
        """Проверяем обычный файл и возвращаем его кодировку."""
        if is_secret_name(target.name):
            raise ToolError("этот файл закрыт: в нём могут быть секреты")
        if not target.exists():
            raise ToolError(f"файл не найден: {target}")
        try:
            info = target.stat()
        except OSError as exc:
            raise ToolError(f"не удалось получить сведения о файле: {exc}") from exc
        if not stat.S_ISREG(info.st_mode):
            raise ToolError("это не обычный файл")
        if info.st_size > HARD_FILE_BYTES:
            raise ToolError("файл слишком большой (больше 50 МБ)")

        try:
            with target.open("rb") as handle:
                sample = handle.read(BINARY_SNIFF_BYTES)
        except OSError as exc:
            raise ToolError(f"не удалось прочитать файл: {exc}") from exc
        return _detect_encoding(sample)

    # --- операции чтения -------------------------------------------------
    def list_dir(self, raw_path: object = ".", limit: int = 200) -> str:
        """Содержимое папки: тип и размер, без секретов и внешних ссылок."""
        target = self.resolve(raw_path, must_exist=True)
        if is_secret_dir(target.name):
            raise ToolError("эта папка закрыта: в ней могут быть секреты")
        if not target.is_dir():
            raise ToolError("это не папка")
        limit = max(1, min(500, int(limit or 200)))

        try:
            entries = list(os.scandir(target))
        except OSError as exc:
            raise ToolError(f"не удалось прочитать папку: {exc}") from exc

        def sort_key(entry: os.DirEntry) -> tuple[int, str]:
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                is_dir = False
            return (0 if is_dir else 1, entry.name.casefold())

        entries.sort(key=sort_key)

        lines = [f"Папка: {target}"]
        shown = 0
        total = 0
        hidden = 0
        outside = 0

        for entry in entries:
            name = entry.name
            try:
                is_link = entry.is_symlink()
            except OSError:
                is_link = False
            if is_link:
                try:
                    self._check_inside(Path(entry.path).resolve())
                except (OSError, RuntimeError, ValueError, ToolError):
                    outside += 1
                    continue

            if is_secret_name(name) or is_secret_dir(name):
                hidden += 1
                continue

            try:
                if entry.is_dir(follow_symlinks=True):
                    kind = "папка"
                    extra = ""
                else:
                    info = entry.stat(follow_symlinks=True)
                    if not stat.S_ISREG(info.st_mode):
                        continue
                    kind = "файл"
                    extra = f" — {_human_size(info.st_size)}"
            except OSError as exc:
                kind = "?"
                extra = f" — ошибка: {exc}"

            total += 1
            if shown < limit:
                lines.append(f"[{kind}] {name}{extra}")
                shown += 1

        lines.append(f"Элементов: {total} (показано {shown})")
        if total > shown:
            lines.append(f"…ещё {total - shown}; сузьте папку или увеличьте limit.")
        if hidden:
            lines.append(f"Скрыто служебных файлов: {hidden}.")
        if outside:
            lines.append(f"Скрыто ссылок за пределы папки: {outside}.")
        if not total:
            lines.append("Папка пуста.")
        return "\n".join(lines)

    def read_file(
        self,
        raw_path: object,
        start_line: int = 1,
        max_lines: int | None = None,
    ) -> str:
        """Текст файла постранично, с номерами строк."""
        target = self.resolve(raw_path, must_exist=True)
        encoding = self._ensure_file(target)
        start = max(1, int(start_line or 1))
        limit = max(
            1,
            min(int(self.limits.max_read_lines), int(max_lines or self.limits.max_read_lines)),
        )
        max_bytes = int(self.limits.max_file_bytes)

        lines: list[str] = []
        used = 0
        more = False
        try:
            with target.open("r", encoding=encoding, errors="replace", newline="") as handle:
                for number, line in enumerate(handle, 1):
                    if number < start:
                        continue
                    if len(lines) >= limit:
                        more = True
                        break
                    size = len(line.encode(encoding, errors="replace"))
                    if used + size > max_bytes:
                        more = True
                        break
                    lines.append(f"{number}: {line.rstrip()}")
                    used += size
        except OSError as exc:
            raise ToolError(f"не удалось прочитать файл: {exc}") from exc

        header = [f"Файл: {target}", f"Кодировка: {encoding}"]
        if not lines:
            header.append("С указанной строки данных нет (конец файла).")
            return "\n".join(header)
        last = start + len(lines) - 1
        header.append(
            f"Строки: {start}–{last}"
            + (" (файл продолжается)" if more else " (конец файла)")
        )
        return "\n".join(header + lines)

    def _ensure_searchable(self, path: Path) -> str | None:
        try:
            info = path.stat()
        except OSError:
            return None
        if not stat.S_ISREG(info.st_mode) or info.st_size > HARD_FILE_BYTES:
            return None
        try:
            with path.open("rb") as handle:
                sample = handle.read(BINARY_SNIFF_BYTES)
        except OSError:
            return None
        if _looks_binary(sample):
            return None
        return _detect_encoding(sample)

    def _iter_files(self, target: Path) -> Iterator[Path]:
        """Файлы папки для поиска: без служебных каталогов и секретов."""
        if target.is_file():
            if not is_secret_name(target.name):
                yield target
            return

        for dirpath, dirnames, filenames in os.walk(target, followlinks=False):
            base = Path(dirpath)
            dirnames[:] = sorted(
                name
                for name in dirnames
                if not is_skipped_dir(name)
                and not is_secret_dir(name)
                and not is_secret_name(name)
            )
            for name in sorted(filenames):
                if is_secret_name(name):
                    continue
                path = base / name
                try:
                    if path.is_symlink():
                        self._check_inside(path.resolve())
                    if not path.is_file():
                        continue
                except (OSError, RuntimeError, ValueError, ToolError):
                    continue
                yield path

    def search_text(
        self,
        query: object,
        raw_path: object = ".",
        max_results: int = 30,
        case_sensitive: bool = False,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        """Простой поиск подстроки по файлам (без regex — никакого ReDoS)."""
        text = str(query or "").strip()
        if not text:
            raise ToolError("пустой запрос")
        if len(text) > 200:
            raise ToolError("запрос слишком длинный (до 200 символов)")

        target = self.resolve(raw_path, must_exist=True)
        if is_secret_dir(target.name):
            raise ToolError("эта папка закрыта: в ней могут быть секреты")

        limit = max(1, min(50, int(max_results or 30)))
        needle = text if case_sensitive else text.casefold()
        matches: list[str] = []
        files_scanned = 0
        bytes_scanned = 0
        skipped = 0
        truncated = False

        for path in self._iter_files(target):
            if should_stop is not None and should_stop():
                truncated = True
                break
            if files_scanned >= SEARCH_MAX_FILES or bytes_scanned >= SEARCH_MAX_BYTES:
                truncated = True
                break

            encoding = self._ensure_searchable(path)
            if encoding is None:
                skipped += 1
                continue
            try:
                bytes_scanned += path.stat().st_size
            except OSError:
                skipped += 1
                continue
            files_scanned += 1

            try:
                with path.open("r", encoding=encoding, errors="replace") as handle:
                    for number, line in enumerate(handle, 1):
                        haystack = line if case_sensitive else line.casefold()
                        if needle not in haystack:
                            continue
                        snippet = line.strip()
                        if len(snippet) > SEARCH_LINE_CHARS:
                            snippet = snippet[:SEARCH_LINE_CHARS] + "…"
                        matches.append(f"{path}:{number}: {snippet}")
                        if len(matches) >= limit:
                            truncated = True
                            break
            except (OSError, UnicodeError):
                skipped += 1
                continue
            if len(matches) >= limit:
                break

        header = [
            f"Поиск: «{text}» в {target}",
            f"Файлов просмотрено: {files_scanned}, совпадений: {len(matches)}",
        ]
        if truncated:
            header.append("Достигнут лимит — сузьте папку или запрос.")
        if skipped:
            header.append(f"Пропущено двоичных/нечитаемых файлов: {skipped}.")
        if not matches:
            header.append("Ничего не найдено.")
        return "\n".join(header + matches)
