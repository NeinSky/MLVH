"""Просмотр и правка словаря произношения.

    python tools/terms.py --stats
    python tools/terms.py --missing "Открой Kubernetes и Terraform"
    python tools/terms.py --add "kubernetes=кубернетес"
    python tools/terms.py --forget kubernetes
    python tools/terms.py --export terms.txt

Правки уходят в `speech_terms.local.json` (он в .gitignore), поставляемый
словарь `app/data/speech_terms.json` не меняется.
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Словарь произношения MLVH")
    parser.add_argument("--stats", action="store_true", help="сколько записей в каждом слое")
    parser.add_argument("--missing", metavar="ТЕКСТ", help="незнакомые термины в тексте")
    parser.add_argument("--add", metavar="СЛОВО=ПРОИЗНОШЕНИЕ", help="добавить своё")
    parser.add_argument("--forget", metavar="СЛОВО", help="удалить из своего слоя")
    parser.add_argument("--export", metavar="ФАЙЛ", help="выгрузить всё в текстовый файл")
    args = parser.parse_args()

    from app.speech_terms import dictionary, is_valid_pronunciation

    terms = dictionary()

    if args.stats:
        print(terms.stats().line())
        local = [key for key, entry in terms.entries().items() if entry.source != "core"]
        if local:
            print("свои записи:", ", ".join(sorted(local)[:40]))
        return 0

    if args.missing is not None:
        words = terms.missing(args.missing)
        if not words:
            print("незнакомых терминов нет")
            return 0
        for word in words:
            print(word)
        print(f"\nвсего: {len(words)}")
        return 0

    if args.add:
        if "=" not in args.add:
            print("нужно так: --add \"kubernetes=кубернетес\"")
            return 1
        word, value = args.add.split("=", 1)
        word = word.strip()
        value = value.strip()
        if not is_valid_pronunciation(value):
            print("произношение должно быть на кириллице, без латиницы")
            return 1
        if terms.remember(word, value, source="user"):
            print(f"добавлено: {word} → {value}")
            return 0
        print("не добавлено: такое же значение уже есть или слово не изменилось")
        return 1

    if args.forget:
        if terms.forget(args.forget):
            print(f"удалено из своего слоя: {args.forget}")
            return 0
        print("в своём слое такого слова нет")
        return 1

    if args.export:
        target = Path(args.export)
        lines = [
            f"{key}={entry.ru}\t{entry.source}"
            for key, entry in sorted(terms.entries().items())
        ]
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"выгружено {len(lines)} записей в {target}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
