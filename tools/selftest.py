"""Самопроверка MLVH без GPU, моделей и интернета.

Запуск:  .venv\\Scripts\\python.exe tools\\selftest.py

Проверяет то, что легко сломать: нормализацию текста, нарезку на фразы,
реестр движков, настройки и разбор аудио. Используется и в CI.
"""

from __future__ import annotations

import json
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


def test_text_norm() -> None:
    print("text_norm:")
    from app.text_norm import normalize

    check("проценты", normalize("15% скидка") == "пятнадцать процентов скидка",
          normalize("15% скидка"))
    check("километры", normalize("5 км") == "пять километров", normalize("5 км"))
    check("год в падеже",
          normalize("в 2024 году") == "в две тысячи двадцать четвёртом году",
          normalize("в 2024 году"))
    check("согласование тысяч",
          normalize("21 тысяча") == "двадцать одна тысяча", normalize("21 тысяча"))
    check("сокращения", "так далее" in normalize("и т.д."), normalize("и т.д."))


def test_sentence_buffer() -> None:
    print("SentenceBuffer:")
    from app.tts import SentenceBuffer

    buffer = SentenceBuffer(min_first=20, min_next=40, max_chunk=80)
    chunks = buffer.push("Первое предложение здесь. Второе предложение тоже есть.")
    chunks += buffer.flush()

    check("фразы нарезаны", len(chunks) >= 2, str(chunks))
    check("нет пустых", all(chunk.strip() for chunk in chunks), str(chunks))
    check("порядок сохранён",
          "Первое" in chunks[0] and "Второе" in chunks[-1], str(chunks))

    code_buffer = SentenceBuffer(min_first=20, min_next=40, max_chunk=80)
    parts = code_buffer.push("Текст до кода. ```python print(1) ``` Текст после кода.")
    parts += code_buffer.flush()
    check("код выброшен", not any("print" in part for part in parts), str(parts))


def test_engines() -> None:
    print("движки:")
    from app.engines.registry import DEFAULT_ENGINE, engine_infos, get_engine

    infos = engine_infos()
    ids = [info.id for info in infos]
    check("есть Edge", "edge" in ids, str(ids))
    check("неизвестный id → Edge", get_engine("nope").id == DEFAULT_ENGINE)
    for info in infos:
        check(f"{info.id}: есть описание", bool(info.title), info.title)


def test_settings() -> None:
    print("настройки:")
    from app.config import Settings
    from app.personas import (
        DEFAULT_PERSONA_ID,
        builtin_personas,
        is_builtin_persona,
    )

    settings = Settings()
    check("движок по умолчанию", settings.tts.engine == "edge", settings.tts.engine)
    check("характеры на месте", len(settings.personas) == len(builtin_personas()))
    check("активный характер", settings.active_persona == DEFAULT_PERSONA_ID)

    default_persona = settings.persona()
    check("id по умолчанию встроенный", is_builtin_persona(DEFAULT_PERSONA_ID),
          DEFAULT_PERSONA_ID)
    check("характер по умолчанию — Элви",
          default_persona is not None and default_persona.name == "Элви",
          default_persona.name if default_persona else "нет")
    check("пресет по умолчанию применён",
          default_persona is not None
          and settings.tts.voice == default_persona.voice
          and settings.tts.rate == default_persona.rate,
          settings.tts.voice)

    data = settings.to_dict()
    check("сериализуется в JSON", isinstance(json.dumps(data), str))

    restored = Settings.from_dict(data)
    check("round-trip сохраняет движок", restored.tts.engine == settings.tts.engine)
    check("round-trip сохраняет характеры",
          [p.id for p in restored.personas] == [p.id for p in settings.personas])

    # Старый конфиг без характеров превращается в «Мой голос».
    legacy = Settings.from_dict({"tts": {"voice": "ru-RU-SvetlanaNeural", "rate": 7}})
    check("миграция старого конфига", legacy.active_persona == "custom",
          legacy.active_persona)
    check("миграция сохраняет голос",
          legacy.tts.voice == "ru-RU-SvetlanaNeural", legacy.tts.voice)


def test_audio_helpers() -> None:
    print("аудио:")
    from app.tts import audio_seconds, expected_seconds

    check("ожидаемая длительность растёт с текстом",
          expected_seconds("а" * 100) > expected_seconds("а" * 10))
    check("не-wav даёт 0", audio_seconds(Path("nope.mp3")) == 0.0)
    check("битый wav даёт 0",
          audio_seconds(Path(tempfile.gettempdir()) / "mlvh_missing.wav") == 0.0)


def test_example_config() -> None:
    print("config.example.json:")
    path = BASE_DIR / "config.example.json"
    check("файл существует", path.exists(), str(path))
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    check("есть tts", "tts" in data)
    check("есть характеры", bool(data.get("personas")))
    ids = [item.get("id") for item in data.get("personas") or [] if isinstance(item, dict)]
    check("активный характер есть в списке", data.get("active_persona") in ids,
          str(data.get("active_persona")))
    check("нет ключа", "sk-" not in path.read_text(encoding="utf-8"))


def test_version() -> None:
    print("версия:")
    import re

    from app import __version__

    check("формат X.Y.Z", bool(re.fullmatch(r"\d+\.\d+\.\d+", __version__)), __version__)

    path = BASE_DIR / "CHANGELOG.md"
    check("CHANGELOG.md есть", path.exists(), str(path))
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    versions = re.findall(r"^##\s+(\d+\.\d+\.\d+)", text, flags=re.MULTILINE)
    check("в changelog есть версии", bool(versions), str(versions))
    if not versions:
        return

    check("верхняя запись — текущая версия", versions[0] == __version__,
          f"{versions[0]} != {__version__}")
    check("версии не повторяются", len(versions) == len(set(versions)), str(versions))
    check("новые версии сверху",
          versions == sorted(versions, key=_version_key, reverse=True), str(versions))


def _version_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def main() -> int:
    print(f"Python {sys.version.split()[0]} | {BASE_DIR}")
    print("")
    for test in (
        test_text_norm,
        test_sentence_buffer,
        test_engines,
        test_settings,
        test_version,
        test_audio_helpers,
        test_example_config,
    ):
        try:
            test()
        except Exception as exc:  # тест не должен падать молча
            print(f"  FAIL {test.__name__}: {type(exc).__name__}: {exc}")
            FAILED.append(test.__name__)
        print("")

    if FAILED:
        print(f"Провалено: {len(FAILED)} — {', '.join(FAILED)}")
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
