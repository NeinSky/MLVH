"""Генерация поставляемого словаря произношения через провайдера модели.

    python tools/build_terms.py --fill 800
    python tools/build_terms.py --words "GitOps,ArgoCD,Backstage"
    python tools/build_terms.py --fill 600 --dry-run

Словарь `app/data/speech_terms.json` лежит в git и обновляется вместе с
проектом: скрипт дописывает в него только новые термины и сортирует ключи.
Уже имеющиеся произношения не меняются — правьте их руками.

Стоимость: ~50 слов за запрос, около 600 токенов на ответ. На 400 новых
терминов — 8 запросов.
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

CORE_PATH = BASE_DIR / "app" / "data" / "speech_terms.json"
BATCH = 40

SYSTEM_PROMPT = (
    "Ты составляешь словарь произношения английских технических терминов "
    "русскими буквами для синтеза речи. Для каждого термина дай произношение "
    "так, как его произносит русскоязычный инженер вслух: Python → пайтон, "
    "Kubernetes → кубернетес, gRPC → джи ар пи си, PostgreSQL → постгрес. "
    "Только кириллица, без латиницы и пояснений, без переводов вроде «ядро». "
    'Ответ — один JSON-объект {"термин": "произношение"}.'
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Дополнить словарь произношения")
    parser.add_argument("--fill", type=int, metavar="N", help="довести словарь до N записей")
    parser.add_argument("--words", metavar="СПИСОК", help="конкретные термины через запятую")
    parser.add_argument("--dry-run", action="store_true", help="не записывать файл")
    args = parser.parse_args()

    if not args.fill and not args.words:
        parser.print_help()
        return 1

    from app.config import get_api_key, load_settings
    from app.llm_client import ChatClient, LLMError
    from app.speech_learn import parse_answer
    from app.speech_terms import is_valid_pronunciation, normalize_key

    data = json.loads(CORE_PATH.read_text(encoding="utf-8"))
    entries: dict[str, str] = dict(data.get("entries") or {})
    before = len(entries)

    settings = load_settings()
    model = settings.model
    client = ChatClient(
        api_key=get_api_key(model),
        model=model.name,
        temperature=0.2,
        max_tokens=1600,
        thinking=False,
        timeout=60.0,
        base_url=model.endpoint(),
        thinking_style=model.thinking_style,
        key_env=model.key_env(),
        max_retries=0,
    )

    pending: list[str] = []
    if args.words:
        pending = [item.strip() for item in args.words.split(",") if item.strip()]

    rounds = 0
    while True:
        if not pending:
            if not args.fill or len(entries) >= args.fill:
                break
            rounds += 1
            if rounds > 40:
                print("достаточно запросов: модель не даёт новых терминов")
                break
            wanted = min(BATCH, max(1, args.fill - len(entries)))
            sample = ", ".join(sorted(entries)[:80])
            pending = []      # слова предложит сама модель
            prompt = (
                f"Дай {wanted} популярных IT-терминов на латинице, которых нет в этом "
                f"списке: {sample}.\n"
                "Только термины, которые реально встречаются в ответах про программирование."
            )
        else:
            prompt = "Дай произношение для терминов: " + ", ".join(pending)

        try:
            parts: list[str] = []
            for event in client.stream_chat(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}]
            ):
                if event.kind == "content":
                    parts.append(event.text)
            answer = parse_answer("".join(parts))
        except (LLMError, ValueError) as exc:
            print(f"ошибка запроса: {exc}")
            return 1

        added = 0
        for key, value in answer.items():
            if key in entries or not is_valid_pronunciation(value):
                continue
            if not any(char.isalpha() and char.isascii() for char in key):
                continue
            entries[normalize_key(key)] = value.strip()
            added += 1
            if args.fill and len(entries) >= args.fill:
                break
        print(f"запрос {rounds or '-'}: получено {len(answer)}, новых {added}, всего {len(entries)}")
        pending = []

    if len(entries) == before:
        print("новых терминов нет")
        return 0

    if args.dry_run:
        for key in sorted(entries)[:20]:
            print(f"  {key} → {entries[key]}")
        print(f"\nбыло {before}, стало {len(entries)} (файл не тронут)")
        return 0

    payload = {"schema": data.get("schema", 1), "entries": dict(sorted(entries.items()))}
    CORE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"записано: было {before}, стало {len(entries)} → {CORE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
