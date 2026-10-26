"""Проверка интернета MLVH: локальный сервер, настоящий Chrome, без ключей.

Запуск:
    .venv\\Scripts\\python.exe tools\\check_web.py
    .venv\\Scripts\\python.exe tools\\check_web.py --no-chrome
    .venv\\Scripts\\python.exe tools\\check_web.py --live
    .venv\\Scripts\\python.exe tools\\check_web.py --url https://example.com

Локальная проверка не ходит в интернет: страницы отдаёт свой http.server на
127.0.0.1, а поиск — подменённый адрес DuckDuckGo с такой же разметкой.
Chrome нужен только для страницы, которая наполняется скриптом.
"""

from __future__ import annotations

import argparse
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


INDEX_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Проверка MLVH</title></head>
<body>
<h1>Заголовок страницы</h1>
<p>Первый абзац с текстом для проверки чтения.</p>
<p>Второй абзац: <b>жирный</b> фрагмент и <a href="/second">вторая страница</a>.</p>
<a href="https://example.com/external">Внешняя ссылка</a>
<script>document.title = "Не должно попасть";</script>
</body></html>
"""

SECOND_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Вторая</title></head>
<body><p>Это вторая страница локального сервера.</p></body></html>
"""

JS_PAGE = (
    """<!doctype html><html><head><meta charset="utf-8"><title>Скрипт</title></head>
<body><div id="root"></div>
<script>
setTimeout(function () {
  document.getElementById('root').innerHTML =
    '<p>Отрисовано скриптом: сорок два.</p>';
}, 300);
</script>
<!-- padding 
"""
    + "x" * 4000
    + " --></body></html>"
)

SEARCH_PAGE = """
<div class="result results_links"><h2 class="result__title">
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fnews%3Fid%3D1">Новость дня</a>
</h2><a class="result__snippet" href="#">Коротко о главном.</a></div>
<div class="result result--ad"><a class="result__a" href="https://ads.example.org/x">Реклама</a></div>
<div class="result results_links"><a class="result__a" href="https://example.com/next">Вторая</a>
<div class="result__snippet">Описание второй.</div></div>
"""


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 — имя задаёт http.server
        path = self.path.split("?", 1)[0]
        if path == "/second":
            body, content_type = SECOND_PAGE, "text/html; charset=utf-8"
        elif path == "/js":
            body, content_type = JS_PAGE, "text/html; charset=utf-8"
        elif path == "/search":
            body, content_type = SEARCH_PAGE, "text/html; charset=utf-8"
        else:
            body, content_type = INDEX_PAGE, "text/html; charset=utf-8"
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


def _start_server() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return server, f"http://127.0.0.1:{port}"


def _offline(use_chrome: bool = True) -> int:
    from app.agent.policy import ToolError
    from app.agent import web as web_module
    from app.agent.web import WebLimits, WebSession, probe_chrome, validate_url

    print("Интернет: локальный сервер, без сети")
    server, base = _start_server()
    original_endpoint = web_module.SEARCH_ENDPOINT
    web_module.SEARCH_ENDPOINT = f"{base}/search"
    try:
        session = WebSession(
            WebLimits(max_searches=1, max_pages=3, max_chars=20_000, allow_local=True)
        )

        page = session.open_page(base + "/", 5000)
        check("страница открыта", "Текст страницы" in page and base in page, page[:200])
        check("заголовок из <title>", "Проверка MLVH" in page, page[:200])
        check("текст абзаца", "Первый абзац с текстом" in page, page[:300])
        check(
            "внешняя ссылка в результате",
            "https://example.com/external" in page,
            page[:400],
        )

        second = session.open_page(base + "/second", 2000)
        check(
            "вторая страница",
            "вторая страница локального сервера" in second,
            second[:200],
        )

        chrome, hint = probe_chrome()
        print(f"  Chrome: {chrome or 'не найден'}")
        if use_chrome and chrome:
            js_page = session.open_page(base + "/js", 4000)
            check(
                "Chrome выполнил скрипт",
                "Отрисовано скриптом: сорок два" in js_page,
                js_page[:300],
            )
            check(
                "пометка об отрисовке",
                "отрисована в Chrome" in js_page,
                js_page[:300],
            )
        elif use_chrome:
            print("  пропуск: " + hint)

        if not (use_chrome and chrome):
            # Третью страницу добавляем сами: лимит проверяется одинаково
            # и с Chrome, и без него.
            session.open_page(base + "/second", 1000)

        try:
            session.open_page(base + "/", 2000)
        except ToolError:
            check("лимит страниц сработал", True)
        else:
            check("лимит страниц сработал", False)

        try:
            validate_url(base + "/")
        except ToolError:
            check("без allow_local локальный адрес закрыт", True)
        else:
            check("без allow_local локальный адрес закрыт", False)

        try:
            validate_url("file:///C:/Windows/win.ini", allow_local=True)
        except ToolError:
            check("file:// закрыт даже при allow_local", True)
        else:
            check("file:// закрыт даже при allow_local", False)

        found = session.search("тест")
        check("поиск вернул результаты", "результатов: 2" in found, found[:300])
        check("реклама отброшена", "Реклама" not in found, found[:300])
        check(
            "ссылка uddg раскодирована",
            "https://example.com/news?id=1" in found,
            found[:300],
        )
    finally:
        web_module.SEARCH_ENDPOINT = original_endpoint
        server.shutdown()
        server.server_close()
    return 0


def _live(url: str, query: str) -> int:
    from app.agent.policy import ToolError
    from app.agent.web import WebLimits, WebSession, probe_chrome

    print("Интернет: живая сеть")
    chrome, hint = probe_chrome()
    print(f"  Chrome: {chrome or 'не найден'}")
    if not chrome:
        print("  " + hint)

    session = WebSession(WebLimits(max_searches=1, max_pages=3))
    try:
        page = session.open_page(url, 8000)
    except ToolError as exc:
        check(f"страница {url}", False, str(exc))
        return 1
    check("страница получена", "Текст страницы" in page, page[:200])
    check("источник указан", f"URL: {url}" in page, page[:200])
    print("  первые строки:")
    for line in page.splitlines()[:6]:
        print(f"    {line}")

    try:
        found = session.search(query)
    except ToolError as exc:
        print(f"  пропуск: поиск недоступен — {exc}")
    else:
        check(
            "поиск вернул ссылки",
            "Поиск:" in found and "http" in found,
            found[:300],
        )
    return 0



def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка интернета MLVH")
    parser.add_argument(
        "--no-chrome",
        action="store_true",
        help="не проверять отрисовку JavaScript в Chrome",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="живая сеть: поиск и страница в интернете",
    )
    parser.add_argument("--url", metavar="URL", help="живая проверка конкретной страницы")
    parser.add_argument(
        "--query",
        metavar="TEXT",
        default="Python",
        help="запрос для поиска в режиме --live",
    )
    args = parser.parse_args()

    if args.live or args.url:
        _live(args.url or "https://example.com", args.query)
    else:
        _offline(use_chrome=not args.no_chrome)

    print()
    if FAILED:
        print(f"Провалено: {len(FAILED)} — {', '.join(FAILED)}")
        return 1
    print("Проверка интернета пройдена.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
