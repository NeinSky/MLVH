"""Интернет для агента: поиск и чтение страниц, Chrome — только рендерер.

Без API-ключей и без внешних пакетов: поиск идёт через HTML-выдачу
DuckDuckGo, страницы качаются стандартным `urllib`, HTML превращается в
текст своим парсером. Если без JavaScript на странице нет текста, один раз
запускается установленный Chrome в headless-режиме (`--dump-dom`).

Правила те же, что и у файлов: только чтение, только http/https, без
локальной сети и `file://` — иначе браузер стал бы обходом файловой
политики. Ошибки — короткий русский текст для модели (`ToolError`).
"""

from __future__ import annotations

import gzip
import ipaddress
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass, replace
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable

from app.agent.policy import ToolError


class NetworkError(ToolError):
    """Сеть не ответила: имеет смысл попробовать другой путь (прокси/напрямую)."""

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36"
)
SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
SEARCH_MAX_QUERY = 300
MAX_URL_CHARS = 2048
MAX_PAGE_BYTES = 2 * 1024 * 1024      # больше 2 МБ страницы не читаем
MAX_PAGE_TEXT = 400_000              # столько текста держим до обрезки
MAX_LINKS = 20
LINK_LABEL_CHARS = 120
LOCAL_SUFFIXES = (
    ".local",
    ".internal",
    ".lan",
    ".home",
    ".intranet",
    ".localhost",
)
TEXT_CONTENT_TYPES = (
    "text/",
    "application/xhtml+xml",
    "application/xml",
    "application/json",
)
# Для этих доменов в режиме auto сначала идём напрямую: локальный прокси
# часто ломает TLS именно на российских сайтах.
RU_HOST_SUFFIXES = (".ru", ".su", ".xn--p1ai", ".by", ".kz")
VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


@dataclass(frozen=True)
class WebLimits:
    """Снимок интернет-настроек на один вопрос агента."""

    max_searches: int = 5
    max_pages: int = 10
    max_results: int = 8
    max_chars: int = 20_000
    timeout: float = 15.0
    render: str = "auto"          # auto / always / never
    chrome_path: str = ""         # пусто — ищем Chrome сами
    proxy: str = "auto"           # auto / system / direct / http://host:port
    allow_local: bool = False     # только для проверок на локальном сервере

    @classmethod
    def from_settings(cls, settings: object) -> "WebLimits":
        """Строим из AgentSettings, не импортируя config (чтобы не было цикла)."""

        def as_int(name: str, default: int) -> int:
            try:
                return int(getattr(settings, name, default))
            except (TypeError, ValueError):
                return default

        def as_float(name: str, default: float) -> float:
            try:
                return float(getattr(settings, name, default))
            except (TypeError, ValueError):
                return default

        render = str(getattr(settings, "web_render", "auto") or "auto").casefold()
        if render not in ("auto", "always", "never"):
            render = "auto"
        return cls(
            max_searches=max(1, as_int("web_max_searches", 5)),
            max_pages=max(1, as_int("web_max_pages", 10)),
            max_results=max(1, as_int("web_max_results", 8)),
            max_chars=max(2000, as_int("web_max_chars", 20_000)),
            timeout=max(5.0, as_float("web_timeout", 15.0)),
            render=render,
            chrome_path=str(getattr(settings, "chrome_path", "") or "").strip(),
            proxy=_normalize_proxy(getattr(settings, "web_proxy", "auto")),
            allow_local=bool(getattr(settings, "web_allow_local", False)),
        )


# --- адреса -----------------------------------------------------------------
def _host_looks_local(host: str) -> bool:
    """Проверка без DNS: localhost, служебные суффиксы, литеральные IP."""
    host = (host or "").rstrip(".").casefold()
    if not host:
        return True
    if host in ("localhost", "localhost.localdomain"):
        return True
    if host.endswith(LOCAL_SUFFIXES):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return "." not in host        # имя без домена — почти всегда интранет
    return not address.is_global


def _host_is_local(host: str) -> bool:
    """То же плюс проверка, куда на самом деле разрешается имя."""
    if _host_looks_local(host):
        return True
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError):
        return False                  # не разрешается — пусть Chrome скажет своё
    for info in infos:
        try:
            address = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if not address.is_global:
            return True
    return False

def validate_url(raw: str, *, allow_local: bool = False, resolve: bool = True) -> str:
    """Нормализуем адрес и отказываем всему, что ведёт не в интернет."""
    text = str(raw or "").strip()
    if not text:
        raise ToolError("пустой адрес")
    if len(text) > MAX_URL_CHARS:
        raise ToolError("слишком длинный адрес")
    if any(ord(char) < 32 for char in text):
        raise ToolError("в адресе есть управляющие символы")
    if "://" not in text:
        text = "https://" + text.lstrip("/")

    try:
        parts = urllib.parse.urlsplit(text)
        port = parts.port
    except ValueError as exc:
        raise ToolError(f"адрес не разбирается: {exc}") from exc

    scheme = parts.scheme.casefold()
    if scheme not in ("http", "https"):
        raise ToolError(
            "открывать можно только http и https: файлы и служебные адреса закрыты"
        )
    if not parts.hostname:
        raise ToolError("в адресе нет имени сайта")
    if parts.username or parts.password:
        raise ToolError("адреса с логином и паролем не открываю")

    host = parts.hostname.rstrip(".").casefold()
    if not allow_local:
        local = _host_is_local(host) if resolve else _host_looks_local(host)
        if local:
            raise ToolError(
                "локальная сеть закрыта: через браузер нельзя обходить доступ к файлам"
            )

    if host.isascii():
        ascii_host = host
    else:
        try:
            ascii_host = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ToolError("такое имя сайта не поддерживается") from exc

    netloc = f"[{ascii_host}]" if ":" in ascii_host else ascii_host
    if port:
        netloc = f"{netloc}:{port}"
    return urllib.parse.urlunsplit(
        (scheme, netloc, parts.path or "/", parts.query, "")
    )


# --- HTML → текст -----------------------------------------------------------
def _clean_line(text: str) -> str:
    """Схлопываем пробелы, убираем невидимые символы."""
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    return " ".join(text.split())


def _normalize_lines(parts: Iterable[str]) -> str:
    """Собираем текст с абзацами: пустых строк не больше одной подряд."""
    raw = "".join(parts).replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for line in raw.split("\n"):
        line = _clean_line(line)
        if line:
            lines.append(line)
        elif lines and lines[-1] != "":
            lines.append("")
    return "\n".join(lines).strip()


class _TextParser(HTMLParser):
    """Достаём заголовок, текст и ссылки, выбрасывая скрипты и разметку."""

    SKIP_TAGS = frozenset(
        {
            "script",
            "style",
            "noscript",
            "svg",
            "canvas",
            "iframe",
            "object",
            "embed",
            "template",
            "head",
            "nav",
            "footer",
            "aside",
            "form",
            "button",
            "select",
            "textarea",
        }
    )
    BLOCK_TAGS = frozenset(
        {
            "address",
            "article",
            "blockquote",
            "br",
            "dd",
            "div",
            "dl",
            "dt",
            "figcaption",
            "figure",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "hr",
            "li",
            "main",
            "ol",
            "p",
            "pre",
            "section",
            "table",
            "tbody",
            "td",
            "tfoot",
            "th",
            "thead",
            "tr",
            "ul",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.base_href = ""
        self._skip = 0
        self._in_title = False
        self._link_href: str | None = None
        self._link_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.casefold()
        values = dict(attrs)
        if tag == "title":
            self._in_title = True
            return
        if tag == "base" and not self.base_href:
            self.base_href = (values.get("href") or "").strip()
            return
        if tag in self.SKIP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "a":
            self._finish_link()
            self._link_href = values.get("href")
            self._link_parts = []
        elif tag == "img":
            alt = _clean_line(values.get("alt") or "")
            if alt:
                self.text_parts.append(f" {alt} ")
        if tag in self.BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in VOID_TAGS:
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in self.SKIP_TAGS:
            if self._skip:
                self._skip -= 1
            return
        if self._skip:
            return
        if tag == "a":
            self._finish_link()
        if tag in self.BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._skip:
            return
        self.text_parts.append(data)
        if self._link_href is not None:
            self._link_parts.append(data)

    def _finish_link(self) -> None:
        href = self._link_href
        label = _clean_line(" ".join(self._link_parts))
        self._link_href = None
        self._link_parts = []
        if href and label:
            self.links.append((href, label))


@dataclass(frozen=True)
class Page:
    """Разобранная страница: то, что уходит модели, и полный текст для обрезки."""

    title: str
    text: str
    links: list[tuple[str, str]]
    truncated: bool = False


def _dedupe_links(
    links: Iterable[tuple[str, str]], base_url: str
) -> list[tuple[str, str]]:
    """Абсолютные http(s)-ссылки без повторов и локальной сети."""
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for href, label in links:
        try:
            absolute = urllib.parse.urljoin(base_url, href)
            parts = urllib.parse.urlsplit(absolute)
        except ValueError:
            continue
        if parts.scheme not in ("http", "https") or not parts.hostname:
            continue
        if _host_looks_local(parts.hostname):
            continue
        clean = urllib.parse.urlunsplit(
            (parts.scheme, parts.netloc, parts.path or "/", parts.query, "")
        )
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append((clean, label[:LINK_LABEL_CHARS]))
        if len(result) >= MAX_LINKS:
            break
    return result


def extract_page(html: str, base_url: str) -> Page:
    """HTML → заголовок, текст и ссылки. Битый HTML не должен ронять цикл."""
    parser = _TextParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # html.parser иногда спотыкается на мусоре
        pass
    base = parser.base_href or base_url
    title = _clean_line(" ".join(parser.title_parts))
    text = _normalize_lines(parser.text_parts)
    truncated = len(text) > MAX_PAGE_TEXT
    return Page(
        title=title[:300],
        text=text[:MAX_PAGE_TEXT],
        links=_dedupe_links(parser.links, base),
        truncated=truncated,
    )


# --- загрузка ---------------------------------------------------------------
@dataclass(frozen=True)
class FetchResult:
    """Ответ сети в одном виде: и для поиска, и для страниц."""

    status: int
    url: str
    content_type: str
    body: bytes


def _is_text_content(content_type: str) -> bool:
    lowered = (content_type or "").casefold()
    if not lowered:
        return True                    # сервер не сказал — попробуем разобрать
    return any(item in lowered for item in TEXT_CONTENT_TYPES)


def _read_limited(response, limit: int, should_stop: Callable[[], bool] | None) -> bytes:
    """Читаем не больше limit байт и уважаем отмену."""
    chunks: list[bytes] = []
    total = 0
    while True:
        if should_stop is not None and should_stop():
            raise ToolError("остановлено")
        chunk = response.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            chunks.append(chunk[: limit - (total - len(chunk))])
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _decompress(body: bytes, encoding: str) -> bytes:
    encoding = (encoding or "").strip().casefold()
    if not encoding or encoding == "identity":
        return body
    try:
        if "gzip" in encoding:
            return gzip.decompress(body)
        if "deflate" in encoding:
            try:
                return zlib.decompress(body)
            except zlib.error:
                return zlib.decompress(body, -zlib.MAX_WBITS)
    except (OSError, EOFError, zlib.error) as exc:
        raise ToolError(f"не удалось распаковать ответ: {exc}") from exc
    return body


_CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?([\w.:-]+)", re.IGNORECASE)
_META_CHARSET_RE = re.compile(
    rb"<meta[^>]+charset\s*=\s*[\"']?([\w.:-]+)", re.IGNORECASE
)


def decode_body(body: bytes, content_type: str = "") -> str:
    """Кодировка: заголовок → <meta charset> → utf-8 → cp1251."""
    candidates: list[str] = []
    match = _CHARSET_RE.search(content_type or "")
    if match:
        candidates.append(match.group(1))
    match = _META_CHARSET_RE.search(body[:4096])
    if match:
        try:
            candidates.append(match.group(1).decode("ascii"))
        except UnicodeDecodeError:
            pass
    candidates.extend(["utf-8", "cp1251", "latin-1"])

    seen: set[str] = set()
    for name in candidates:
        name = name.strip().casefold()
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            return body.decode(name)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", errors="replace")


class _GuardedRedirects(urllib.request.HTTPRedirectHandler):
    """Проверяем каждый редирект: иначе страница уведёт на localhost/file."""

    def __init__(self, allow_local: bool) -> None:
        self._allow_local = allow_local

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl, allow_local=self._allow_local)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _normalize_proxy(value: object) -> str:
    """auto / system / direct или адрес прокси: приводим к одному виду."""
    text = str(value or "auto").strip()
    lowered = text.casefold()
    if lowered in ("", "auto"):
        return "auto"
    if lowered in ("system", "direct"):
        return lowered
    if "://" not in text:
        text = "http://" + text
    return text


@lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    """certifi + корневые сертификаты Windows — как у браузера.

    Проверка сертификата остаётся включённой: мы лишь добавляем доверенные
    корни системы (их же использует Chrome), а не отключаем проверку.
    """
    context = ssl.create_default_context()
    if os.name != "nt" or not hasattr(ssl, "enum_certificates"):
        return context
    pems: list[str] = []
    for store in ("ROOT", "CA"):
        try:
            certificates = ssl.enum_certificates(store)
        except (OSError, AttributeError):
            continue
        for der, encoding, _trust in certificates:
            if encoding != "x509_asn":
                continue
            try:
                pems.append(ssl.DER_cert_to_PEM_cert(der))
            except ValueError:
                continue
    if pems:
        try:
            context.load_verify_locations(cadata="".join(pems))
        except ssl.SSLError:
            for pem in pems:
                try:
                    context.load_verify_locations(cadata=pem)
                except ssl.SSLError:
                    pass
    return context


def system_proxy() -> str:
    """Прокси из настроек Windows/окружения (как его видит urllib)."""
    try:
        proxies = urllib.request.getproxies()
    except (OSError, ValueError):
        return ""
    return str(proxies.get("https") or proxies.get("http") or "").strip()


def proxy_attempts(
    url: str, setting: str, *, system: str | None = None
) -> list[str | None]:
    """Порядок попыток: None — напрямую, строка — через этот прокси."""
    setting = _normalize_proxy(setting)
    if system is None:
        system = system_proxy()
    host = (urllib.parse.urlsplit(url).hostname or "").casefold()

    if _host_looks_local(host):
        return [None]              # проверочный сервер — всегда напрямую
    if setting == "direct":
        return [None]
    if setting == "system":
        return [system] if system else [None]
    if setting == "auto":
        if not system:
            return [None]
        return [None, system] if host.endswith(RU_HOST_SUFFIXES) else [system, None]
    return [setting]

def _fetch_with_proxy(
    url: str,
    proxy_url: str | None,
    *,
    timeout: float,
    allow_local: bool,
    should_stop: Callable[[], bool] | None,
) -> FetchResult:
    handlers: list = [
        urllib.request.ProxyHandler(
            {} if not proxy_url else {"http": proxy_url, "https": proxy_url}
        ),
        urllib.request.HTTPSHandler(context=_ssl_context()),
        _GuardedRedirects(allow_local),
    ]
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
        },
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if not _is_text_content(content_type):
                raise ToolError("это не текстовая страница — открывать не буду")
            body = _read_limited(response, MAX_PAGE_BYTES, should_stop)
            body = _decompress(body, response.headers.get("Content-Encoding", ""))
            # Защита от «zip-бомбы»: больше 10 МБ распакованного не держим.
            if len(body) > MAX_PAGE_BYTES * 5:
                body = body[: MAX_PAGE_BYTES * 5]
            return FetchResult(
                status=int(getattr(response, "status", 200) or 200),
                url=response.geturl(),
                content_type=content_type,
                body=body,
            )
    except ToolError:
        raise
    except urllib.error.HTTPError as exc:
        raise ToolError(f"сайт ответил ошибкой {exc.code}") from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, ssl.SSLCertVerificationError):
            detail = getattr(reason, "verify_message", "") or str(reason)
            raise NetworkError(
                f"сертификат сайта не прошёл проверку: {detail}"
            ) from exc
        raise NetworkError(f"не удалось подключиться: {reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise NetworkError("сайт не ответил вовремя") from exc
    except (OSError, ValueError) as exc:
        raise NetworkError(f"сетевая ошибка: {exc}") from exc


def _default_fetch(
    url: str,
    *,
    timeout: float,
    allow_local: bool,
    should_stop: Callable[[], bool] | None = None,
    proxy: str = "auto",
    routes: dict[str, str] | None = None,
) -> FetchResult:
    """HTTP-запрос с выбором пути: напрямую или через системный прокси.

    В режиме auto российские домены идут напрямую, остальные — через прокси;
    если путь не сработал, пробуем второй. Удачный путь запоминаем на сессию.
    """
    host = (urllib.parse.urlsplit(url).hostname or "").casefold()
    attempts: list[str | None] = []
    for item in proxy_attempts(url, proxy):
        if item not in attempts:
            attempts.append(item)

    cached = (routes or {}).get(host)
    if cached:
        preferred = None if cached == "direct" else cached
        if preferred in attempts:
            attempts.remove(preferred)
        attempts.insert(0, preferred)

    errors: list[str] = []
    for candidate in attempts:
        try:
            result = _fetch_with_proxy(
                url,
                candidate,
                timeout=timeout,
                allow_local=allow_local,
                should_stop=should_stop,
            )
        except NetworkError as exc:
            errors.append(str(exc))
            if should_stop is not None and should_stop():
                raise ToolError("остановлено") from exc
            continue
        if routes is not None:
            routes[host] = candidate or "direct"
        return result

    if len(set(errors)) == 1:
        raise NetworkError(errors[0])
    raise NetworkError(
        f"{errors[0]} (пробовал и через прокси, и напрямую)"
        if errors
        else "не удалось подключиться"
    )



# --- поиск ------------------------------------------------------------------
@dataclass(frozen=True)
class SearchItem:
    """Одна ссылка из выдачи."""

    title: str
    url: str
    snippet: str


class _SearchParser(HTMLParser):
    """Разбираем HTML-выдачу DuckDuckGo: заголовок, ссылка, сниппет."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[dict] = []
        self._current: dict | None = None
        self._capture: str | None = None
        self._depth = 0
        self._ad_tag: str | None = None
        self._ad_nesting = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.casefold()
        values = dict(attrs)
        classes = (values.get("class") or "").casefold().split()

        if self._ad_tag is None and any(
            item.startswith("result--ad") for item in classes
        ):
            self._ad_tag = tag
            self._ad_nesting = 1
            return
        if self._ad_tag is not None:
            if tag not in VOID_TAGS:
                self._ad_nesting += 1
            return

        if self._capture is not None:
            if tag not in VOID_TAGS:
                self._depth += 1
            return

        if tag == "a" and any(item.startswith("result__a") for item in classes):
            self._finish_item()
            self._current = {
                "title": [],
                "url": values.get("href") or "",
                "snippet": [],
            }
            self._capture = "title"
            self._depth = 1
            return

        if (
            self._current is not None
            and any(item.startswith("result__snippet") for item in classes)
        ):
            self._capture = "snippet"
            self._depth = 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in VOID_TAGS:
            return
        if self._ad_tag is not None:
            self._ad_nesting -= 1
            if self._ad_nesting <= 0:
                self._ad_tag = None
                self._ad_nesting = 0
            return
        if self._capture is not None:
            self._depth -= 1
            if self._depth <= 0:
                self._capture = None
                self._depth = 0

    def handle_data(self, data: str) -> None:
        if self._capture is None or self._current is None:
            return
        self._current[self._capture].append(data)

    def close(self) -> None:
        super().close()
        self._finish_item()

    def _finish_item(self) -> None:
        item = self._current
        self._current = None
        if not item or not item.get("url"):
            return
        title = _clean_line(" ".join(item["title"]))
        snippet = _clean_line(" ".join(item["snippet"]))
        if not title:
            return
        self.items.append({"title": title, "url": item["url"], "snippet": snippet})


def _search_url(query: str) -> str:
    params = urllib.parse.urlencode({"q": query, "kl": "ru-ru"})
    return f"{SEARCH_ENDPOINT}?{params}"


def _decode_result_url(href: str) -> str:
    """DuckDuckGo заворачивает ссылки в /l/?uddg=… — разворачиваем обратно."""
    if not href:
        return ""
    href = href.strip()
    if href.startswith("//"):
        href = "https:" + href
    try:
        absolute = urllib.parse.urljoin("https://duckduckgo.com/", href)
        parts = urllib.parse.urlsplit(absolute)
    except ValueError:
        return ""
    host = (parts.hostname or "").casefold()
    if host.endswith("duckduckgo.com"):
        target = urllib.parse.parse_qs(parts.query).get("uddg", [""])[0]
        if target:
            return target
        if parts.path.startswith("/y.js"):
            return ""                 # реклама
        if parts.path in ("/html/", "/lite/"):
            return ""
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, parts.query, "")
    )


def parse_search(html: str) -> list[SearchItem]:
    """HTML выдачи → список результатов без рекламы и локальных адресов."""
    parser = _SearchParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass

    items: list[SearchItem] = []
    seen: set[str] = set()
    for raw in parser.items:
        url = _decode_result_url(raw["url"])
        try:
            parts = urllib.parse.urlsplit(url)
        except ValueError:
            continue
        if parts.scheme not in ("http", "https") or not parts.hostname:
            continue
        if _host_looks_local(parts.hostname):
            continue
        key = url.casefold()
        if key in seen:
            continue
        seen.add(key)
        title = (raw["title"] or parts.hostname).strip()
        items.append(
            SearchItem(
                title=title[:200],
                url=url,
                snippet=(raw["snippet"] or "")[:300],
            )
        )
    return items


# --- Chrome как запасной рендерер -------------------------------------------
def find_chrome(explicit: str = "") -> str | None:
    """Ищем chrome.exe: настройка → env → стандартные места → PATH."""
    candidates: list[str] = []
    explicit = (explicit or "").strip()
    if explicit:
        candidates.append(explicit)
    env = (os.environ.get("MLVH_CHROME_PATH") or "").strip()
    if env:
        candidates.append(env)

    if os.name == "nt":
        for base in (
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LOCALAPPDATA"),
        ):
            if base:
                candidates.append(
                    str(Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe")
                )
    elif sys.platform == "darwin":
        candidates.append(
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        )

    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            candidates.append(found)

    for item in candidates:
        try:
            path = Path(item).expanduser()
        except (OSError, ValueError):
            continue
        if path.is_file():
            return str(path)
    return None


def probe_chrome(explicit: str = "") -> tuple[str, str]:
    """(путь, "") или ("", подсказка) — для интерфейса и проверок."""
    path = find_chrome(explicit)
    if path:
        return path, ""
    return "", (
        "Google Chrome не найден: поиск и обычные страницы работают, "
        "но страницы на JavaScript останутся пустыми. Укажите путь в "
        "config.json (agent.chrome_path)."
    )


def render_with_chrome(
    url: str,
    *,
    chrome_path: str = "",
    timeout: float = 15.0,
    wait_ms: int = 6000,
    proxy: str | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    """Один запуск headless Chrome: отдать готовый DOM после JavaScript.

    `proxy`: None — как в системе, "" — без прокси, URL — через этот прокси.
    """
    exe = find_chrome(chrome_path)
    if not exe:
        raise ToolError("Chrome не найден — страницу с JavaScript не отрисовать")

    profile = Path(tempfile.mkdtemp(prefix="mlvh_chrome_"))
    out_path = profile / "page.html"
    args = [
        exe,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--mute-audio",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-pings",
        "--disable-extensions",
        "--disable-sync",
        "--disable-background-networking",
        "--disable-component-update",
        f"--user-data-dir={profile / 'profile'}",
        f"--virtual-time-budget={int(wait_ms)}",
        *(
            ["--no-proxy-server"]
            if proxy == ""
            else ([f"--proxy-server={proxy}"] if proxy else [])
        ),
        "--dump-dom",
        url,
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    deadline = time.monotonic() + max(10.0, timeout + 10.0)
    try:
        with open(out_path, "wb") as sink:
            try:
                process = subprocess.Popen(
                    args,
                    stdout=sink,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
            except OSError as exc:
                raise ToolError(f"не удалось запустить Chrome: {exc}") from exc

            try:
                while process.poll() is None:
                    if should_stop is not None and should_stop():
                        raise ToolError("остановлено")
                    if time.monotonic() > deadline:
                        raise ToolError("Chrome не ответил вовремя")
                    time.sleep(0.1)
            finally:
                if process.poll() is None:
                    process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        html = out_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ToolError(f"Chrome не смог отрисовать страницу: {exc}") from exc
    finally:
        shutil.rmtree(profile, ignore_errors=True)

    if not html.strip():
        raise ToolError("Chrome вернул пустую страницу")
    return html


# --- сессия -----------------------------------------------------------------
def _clamp(value: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))


class WebSession:
    """Поиск и чтение страниц на один вопрос агента: общие счётчики и лимиты.

    Сеть и рендеринг можно подменить в тестах (`fetch` и `render`), поэтому
    проверки идут без интернета и без Chrome.
    """

    def __init__(
        self,
        limits: WebLimits | None = None,
        *,
        fetch: Callable[[str], FetchResult] | None = None,
        render: Callable[[str], str] | None = None,
    ) -> None:
        self.limits = limits or WebLimits()
        self._fetch_override = fetch
        self._render_override = render
        self._searches = 0
        self._pages = 0
        self._search_cache: dict[str, str] = {}
        self._routes: dict[str, str] = {}

    @property
    def searches_used(self) -> int:
        return self._searches

    @property
    def pages_used(self) -> int:
        return self._pages

    def search(
        self,
        query: str,
        max_results: int = 0,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        """Поиск в интернете: список «заголовок — адрес — сниппет»."""
        text = _clean_line(str(query or ""))
        if not text:
            raise ToolError("пустой поисковый запрос")
        text = text[:SEARCH_MAX_QUERY]
        limit = _clamp(max_results or self.limits.max_results, 1, 10)

        key = f"{text.casefold()}|{limit}"
        cached = self._search_cache.get(key)
        if cached is not None:
            return cached

        if self._searches >= self.limits.max_searches:
            raise ToolError(
                f"лимит поисков на один вопрос ({self.limits.max_searches}). "
                "Ответь по уже найденному или открой подходящую ссылку."
            )
        self._searches += 1

        result = self._fetch_once(_search_url(text), should_stop)
        if result.status >= 400:
            raise ToolError(f"поисковик ответил ошибкой {result.status}")
        html = decode_body(result.body, result.content_type)
        items = parse_search(html)[:limit]
        if not items:
            raise ToolError(
                "поисковик не вернул результатов — уточните запрос "
                "или попробуйте позже (возможно, сработала защита от роботов)"
            )

        lines = [f"Поиск: «{text}» (DuckDuckGo), результатов: {len(items)}"]
        for index, item in enumerate(items, 1):
            lines.append(f"{index}. {item.title}")
            lines.append(f"   {item.url}")
            if item.snippet:
                lines.append(f"   {item.snippet}")
        content = "\n".join(lines)
        self._search_cache[key] = content
        return content

    def open_page(
        self,
        url: str,
        max_chars: int = 0,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        """Открыть страницу и вернуть текст, ссылки и источник."""
        target = validate_url(url, allow_local=self.limits.allow_local)
        limit = _clamp(
            max_chars or self.limits.max_chars, 1000, self.limits.max_chars * 5
        )

        if self._pages >= self.limits.max_pages:
            raise ToolError(
                f"лимит страниц на один вопрос ({self.limits.max_pages}). "
                "Ответь по уже прочитанному."
            )
        self._pages += 1

        used_chrome_fetch = False
        try:
            result = self._fetch_once(target, should_stop)
        except NetworkError as exc:
            # HTTP-клиент не смог (прокси/TLS) — последний шанс: Chrome.
            html = self._render_once(target, should_stop)
            if not html:
                raise NetworkError(
                    f"{exc}. Chrome тоже не открыл страницу — проверьте "
                    "agent.web_proxy (auto/system/direct) или откройте сайт вручную."
                ) from exc
            result = FetchResult(
                200, target, "text/html; charset=utf-8", html.encode("utf-8")
            )
            used_chrome_fetch = True

        if result.status >= 400:
            raise ToolError(f"сайт ответил ошибкой {result.status}")

        final_url = validate_url(
            result.url or target, allow_local=self.limits.allow_local
        )
        html = decode_body(result.body, result.content_type)
        page = extract_page(html, final_url)

        rendered = used_chrome_fetch
        render_attempted = used_chrome_fetch
        needs_render = (not used_chrome_fetch) and (
            self.limits.render == "always"
            or (
                self.limits.render == "auto"
                and len(page.text) < 200
                and len(html) > 3000
            )
        )
        if needs_render and self.limits.render != "never":
            render_attempted = True
            rendered_html = self._render_once(final_url, should_stop)
            if rendered_html:
                rendered_page = extract_page(rendered_html, final_url)
                if len(rendered_page.text) > len(page.text):
                    page = replace(
                        page,
                        title=rendered_page.title or page.title,
                        text=rendered_page.text,
                        links=rendered_page.links or page.links,
                    )
                    rendered = True

        text = page.text.strip()
        truncated = len(text) > limit
        lines = [
            f"Заголовок: {page.title or '(без заголовка)'}",
            f"URL: {final_url}",
        ]
        if used_chrome_fetch:
            lines.append(
                "Страница открыта через Chrome: HTTP-клиент не смог подключиться "
                "(прокси/TLS)."
            )
        elif rendered:
            lines.append("Страница отрисована в Chrome: JavaScript выполнен.")
        lines.append("")
        lines.append("Текст страницы:")
        lines.append(text[:limit] if text else "(текста на странице не нашлось)")
        if truncated:
            lines.append(
                f"\n[…обрезано: показаны первые {limit} символов из {len(text)}]"
            )
        if page.links:
            lines.append("")
            lines.append("Ссылки на странице:")
            for href, label in page.links[:15]:
                lines.append(f"- {label} — {href}")
        if not text and not page.links:
            lines.append(
                "\nПохоже, страница пустая или без JavaScript. Попробуй другой источник."
            )
            if render_attempted and not rendered:
                lines.append(
                    "Chrome не выполнил JavaScript — проверьте, установлен ли он "
                    "(agent.chrome_path в config.json)."
                )
        return "\n".join(lines)

    def close(self) -> None:
        """Сейчас закрывать нечего (urllib без сессии), метод — для единообразия."""

    def _fetch_once(
        self, url: str, should_stop: Callable[[], bool] | None
    ) -> FetchResult:
        if self._fetch_override is not None:
            return self._fetch_override(url)
        return _default_fetch(
            url,
            timeout=self.limits.timeout,
            allow_local=self.limits.allow_local,
            should_stop=should_stop,
            proxy=self.limits.proxy,
            routes=self._routes,
        )

    def _chrome_proxy(self, url: str) -> str | None:
        """Какой путь дать Chrome: "" — напрямую, URL — прокси, None — системный."""
        host = (urllib.parse.urlsplit(url).hostname or "").casefold()
        cached = self._routes.get(host)
        if cached:
            return "" if cached == "direct" else cached
        attempts = proxy_attempts(url, self.limits.proxy)
        first = attempts[0] if attempts else None
        return "" if first is None else first

    def _render_once(self, url: str, should_stop: Callable[[], bool] | None) -> str:
        if self._render_override is not None:
            return self._render_override(url)
        try:
            return render_with_chrome(
                url,
                chrome_path=self.limits.chrome_path,
                timeout=self.limits.timeout,
                proxy=self._chrome_proxy(url),
                should_stop=should_stop,
            )
        except ToolError:
            return ""                  # без Chrome остаётся текст из HTML
