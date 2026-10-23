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


def test_asr_settings() -> None:
    print("распознавание: настройки")
    from app.config import ASRSettings, Settings

    settings = Settings()
    check("режим по умолчанию — по нажатию", settings.asr.mode == "push", settings.asr.mode)
    check("движок по умолчанию — whisper", settings.asr.engine == "whisper", settings.asr.engine)
    check("клавиша по умолчанию", settings.asr.hotkey == "Ctrl+Space", settings.asr.hotkey)
    check("глобальные клавиши выключены", settings.asr.global_hotkey is False)

    restored = Settings.from_dict(settings.to_dict())
    check("round-trip сохраняет микрофон",
          restored.asr.mode == settings.asr.mode
          and restored.asr.hotkey == settings.asr.hotkey
          and restored.asr.model == settings.asr.model)

    legacy = Settings.from_dict({"tts": {"voice": "ru-RU-SvetlanaNeural"}})
    check("старый конфиг без asr", legacy.asr.mode == "push", legacy.asr.mode)

    broken = ASRSettings.from_dict({"mode": "чепуха", "max_seconds": 1})
    check("неизвестный режим → push", broken.mode == "push", broken.mode)
    check("предел записи зажат", broken.max_seconds == 3, str(broken.max_seconds))


def test_hotkeys() -> None:
    print("горячие клавиши:")
    from app.asr.hotkeys import normalize, parse, to_vk

    hotkey = parse("ctrl + space")
    check("разбор Ctrl+Space", hotkey is not None and hotkey.text() == "Ctrl+Space",
          hotkey.text() if hotkey else "None")

    mixed = parse("alt+ctrl+space")
    check("порядок модификаторов", mixed is not None and mixed.text() == "Ctrl+Alt+Space",
          mixed.text() if mixed else "None")

    check("без модификатора нельзя", parse("space") is None)
    check("неизвестная клавиша нельзя", parse("Ctrl+НетТакой") is None)
    check("нормализация", normalize("CTRL+space") == "Ctrl+Space", normalize("CTRL+space"))
    check("VK пробела", to_vk("Space") == 0x20, str(to_vk("Space")))
    check("VK буквы", to_vk("A") == 0x41, str(to_vk("A")))
    check("VK F5", to_vk("F5") == 0x74, str(to_vk("F5")))
    check("неизвестная клавиша", to_vk("НетТакой") is None)


def test_asr_audio() -> None:
    print("аудио распознавания:")
    try:
        import numpy as np
    except ImportError:
        print("  numpy нет — пропуск")
        return

    from app.asr import audio

    rate = 48000
    tone = np.sin(2 * np.pi * 440 * np.arange(rate // 10) / rate).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1).reshape(-1)
    mono = audio.to_mono_16k(stereo, rate, 2)
    check("48k стерео → 16k моно", abs(len(mono) - 1600) <= 2, str(len(mono)))
    check("амплитуда сохранилась", 0.9 < audio.peak(mono) <= 1.0, f"{audio.peak(mono):.3f}")

    silence = np.zeros(audio.TARGET_RATE, dtype=np.float32)
    check("тишина даёт 0", audio.rms(silence) == 0.0, str(audio.rms(silence)))
    check("level в диапазоне 0..1", 0.0 <= audio.level(0.02) <= 1.0, str(audio.level(0.02)))

    padded = np.concatenate([silence, tone[:1600], silence])
    trimmed = audio.trim_silence(padded)
    check("тишина обрезана", 0 < trimmed.size < padded.size,
          f"{trimmed.size} из {padded.size}")

    path = Path(tempfile.gettempdir()) / "mlvh_selftest.wav"
    audio.write_wav(path, tone[:1600], audio.TARGET_RATE)
    read_back, read_rate = audio.read_wav(path)
    check("wav читается обратно",
          read_rate == audio.TARGET_RATE and read_back.size == 1600,
          f"{read_rate} Гц, {read_back.size} отсчётов")


def test_asr_registry() -> None:
    print("движки распознавания:")
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("  numpy нет — пропуск")
        return

    from app.asr.registry import DEFAULT_ASR_ENGINE, asr_engine_infos, get_asr_engine

    infos = asr_engine_infos()
    ids = [info.id for info in infos]
    check("есть whisper", "whisper" in ids, str(ids))
    check("есть vosk", "vosk" in ids, str(ids))
    check("неизвестный id → whisper", get_asr_engine("nope").id == DEFAULT_ASR_ENGINE)
    for info in infos:
        check(f"{info.id}: есть описание", bool(info.title), info.title)


def test_wake_word() -> None:
    print("спецслово:")
    from app.asr.wake import levenshtein, strip_wake_word

    check("Левенштейн: одна замена", levenshtein("элви", "элвис") == 1,
          str(levenshtein("элви", "элвис")))

    cases = (
        ("Элви, в каком году было восстание декабристов?",
         "в каком году было восстание декабристов?"),
        ("элвис сколько времени", "сколько времени"),
        ("эй элвин расскажи анекдот", "расскажи анекдот"),
        ("эл ви что такое фотосинтез", "что такое фотосинтез"),
        ("Элли, какая сегодня погода?", "какая сегодня погода?"),
    )
    for text, expected in cases:
        actual = strip_wake_word(text, "элви")
        check(f"срезано: {text[:24]!r}", actual == expected, f"{actual!r} != {expected!r}")

    check("только спецслово", strip_wake_word("Элви.", "элви") == "",
          repr(strip_wake_word("Элви.", "элви")))
    check("обычная фраза не срабатывает",
          strip_wake_word("Привет, как дела?", "элви") is None)
    check("«эльвира» не срабатывает",
          strip_wake_word("Эльвира пришла домой.", "элви") is None)
    check("другое спецслово",
          strip_wake_word("Помощник, сколько времени?", "помощник") == "сколько времени?",
          repr(strip_wake_word("Помощник, сколько времени?", "помощник")))


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
        test_asr_settings,
        test_hotkeys,
        test_asr_audio,
        test_asr_registry,
        test_wake_word,
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
