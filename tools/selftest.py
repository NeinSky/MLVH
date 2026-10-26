"""Самопроверка MLVH без GPU, моделей и интернета.

Запуск:  .venv\\Scripts\\python.exe tools\\selftest.py

Проверяет то, что легко сломать: нормализацию текста, нарезку на фразы,
реестр движков, настройки и разбор аудио. Используется и в CI.
"""

from __future__ import annotations

import json
import subprocess
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
    check("окно модели по умолчанию — 1M",
          settings.model.context_tokens == 1_000_000,
          str(settings.model.context_tokens))
    check("агент выключен по умолчанию", settings.agent.read_enabled is False)
    check("рабочих папок нет", settings.agent.roots == [], str(settings.agent.roots))
    check("шагов по умолчанию — 20", settings.agent.max_rounds == 20,
          str(settings.agent.max_rounds))
    check("вызовов по умолчанию — 100", settings.agent.max_tool_calls == 100,
          str(settings.agent.max_tool_calls))
    check("время по умолчанию — 300 с", settings.agent.max_seconds == 300,
          str(settings.agent.max_seconds))
    check("объём данных по умолчанию — 800k",
          settings.agent.max_tool_chars == 800_000,
          str(settings.agent.max_tool_chars))
    check("history_turns больше не пишется", "history_turns" not in data)

    restored = Settings.from_dict(data)
    check("round-trip сохраняет движок", restored.tts.engine == settings.tts.engine)
    check("round-trip сохраняет характеры",
          [p.id for p in restored.personas] == [p.id for p in settings.personas])
    check("round-trip сохраняет окно и лимиты",
          restored.model.context_tokens == settings.model.context_tokens
          and restored.agent.max_rounds == settings.agent.max_rounds)

    broken = Settings.from_dict(
        {"agent": {"max_rounds": 99, "max_seconds": 1, "roots": "D:\\x"}}
    )
    check("верхний предел шагов не зажимается", broken.agent.max_rounds == 99,
          str(broken.agent.max_rounds))
    check("произвольное число шагов сохраняется",
          Settings.from_dict({"agent": {"max_rounds": 1000}}).agent.max_rounds == 1000)
    check("произвольное число вызовов сохраняется",
          Settings.from_dict({"agent": {"max_tool_calls": 500}}).agent.max_tool_calls == 500)
    check("предел времени зажат", broken.agent.max_seconds == 10,
          str(broken.agent.max_seconds))
    check("кривые roots игнорируются", broken.agent.roots == [],
          str(broken.agent.roots))

    broken_text = Settings.from_dict(
        {
            "model": {"context_tokens": "abc"},
            "agent": {"max_rounds": "abc", "read_enabled": "true"},
        }
    )
    check("битое окно → 1M", broken_text.model.context_tokens == 1_000_000,
          str(broken_text.model.context_tokens))
    check("битый лимит → 20", broken_text.agent.max_rounds == 20,
          str(broken_text.agent.max_rounds))
    check("строка true → включено", broken_text.agent.read_enabled is True)

    migrated = Settings.from_dict(
        {
            "agent": {
                "max_rounds": 12,
                "max_tool_calls": 12,
                "max_seconds": 120,
                "max_tool_chars": 400_000,
            }
        }
    )
    check("старые лимиты 0.3.0 подняты",
          migrated.agent.max_rounds == 20
          and migrated.agent.max_tool_calls == 100
          and migrated.agent.max_seconds == 300
          and migrated.agent.max_tool_chars == 800_000,
          str(migrated.agent))
    check("версия лимитов записана", migrated.agent.limits_version == 2,
          str(migrated.agent.limits_version))

    explicit = Settings.from_dict(
        {"agent": {"limits_version": 2, "max_rounds": 12, "max_tool_calls": 12}}
    )
    check("при limits_version=2 явные 12 не трогаем",
          explicit.agent.max_rounds == 12
          and explicit.agent.max_tool_calls == 12,
          str(explicit.agent))

    # Старый конфиг без характеров превращается в «Мой голос».
    legacy = Settings.from_dict({"tts": {"voice": "ru-RU-SvetlanaNeural", "rate": 7}})
    check("миграция старого конфига", legacy.active_persona == "custom",
          legacy.active_persona)
    check("миграция сохраняет голос",
          legacy.tts.voice == "ru-RU-SvetlanaNeural", legacy.tts.voice)
    check("агент старых конфигов выключен", legacy.agent.read_enabled is False)


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
    check("есть агент", "agent" in data)
    check("окно модели 1M",
          (data.get("model") or {}).get("context_tokens") == 1_000_000,
          str((data.get("model") or {}).get("context_tokens")))
    agent = data.get("agent") or {}
    check("шагов в примере — 20", agent.get("max_rounds") == 20,
          str(agent.get("max_rounds")))
    check("вызовов в примере — 100", agent.get("max_tool_calls") == 100,
          str(agent.get("max_tool_calls")))
    check("версия лимитов в примере", agent.get("limits_version") == 2,
          str(agent.get("limits_version")))
    check("history_turns убран", "history_turns" not in data)
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
    from app.asr.wakeword import levenshtein, strip_wake_word

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


def test_agent_policy() -> None:
    print("агент: политика доступа")
    from app.agent.policy import AgentLimits, ToolError, Workspace, validate_root

    with tempfile.TemporaryDirectory(prefix="mlvh_policy_") as tmp:
        base = Path(tmp)
        root = base / "root"
        root.mkdir()
        (root / "ok.txt").write_text("внутри\n", encoding="utf-8")
        (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
        (root / "bin.dat").write_bytes(b"\x00\x01")
        (root / "lines.txt").write_text("1\n2\n3\n4\n5\n", encoding="utf-8")
        (base / "outside.txt").write_text("снаружи\n", encoding="utf-8")

        workspace = Workspace([root], AgentLimits())
        check("корень принят", bool(workspace), str(workspace.skipped))
        check("несуществующая папка пропущена",
              not Workspace([base / "nope"]))
        drive_root = Path.cwd().anchor
        check("корень диска отклонён", validate_root(drive_root)[0] is None,
              str(validate_root(drive_root)))

        try:
            text = workspace.read_file("ok.txt")
        except ToolError as exc:
            check("чтение внутри папки", False, str(exc))
        else:
            check("чтение внутри папки", "внутри" in text, text)

        for name, call in (
            ("выход через ..", lambda: workspace.read_file("../outside.txt")),
            ("абсолютный путь наружу",
             lambda: workspace.read_file(str(base / "outside.txt"))),
            ("секрет .env", lambda: workspace.read_file(".env")),
            ("двоичный файл", lambda: workspace.read_file("bin.dat")),
            ("несуществующий файл", lambda: workspace.read_file("nope.txt")),
        ):
            try:
                call()
            except ToolError:
                check(name, True)
            else:
                check(name, False, "отказа не было")

        chunk = workspace.read_file("lines.txt", 2, 2)
        check("постраничное чтение",
              "2: 2" in chunk and "3: 3" in chunk and "4: 4" not in chunk,
              chunk.replace("\n", " | "))
        check("пометка о продолжении", "продолжается" in chunk, chunk)

        listing = workspace.list_dir(".", 50)
        check("секрет скрыт в списке", ".env" not in listing, listing)
        check("файл виден в списке", "ok.txt" in listing, listing)

        found = workspace.search_text("внутри", ".", 10)
        check("поиск находит текст", "ok.txt" in found, found)
        secret = workspace.search_text("SECRET", ".", 10)
        check("поиск не видит секреты", "Ничего не найдено" in secret, secret)

        link = root / "link.txt"
        try:
            link.symlink_to(base / "outside.txt")
        except (OSError, NotImplementedError) as exc:
            print(f"  пропуск: симлинки недоступны ({exc})")
        else:
            try:
                workspace.read_file("link.txt")
            except ToolError:
                check("симлинк наружу отклонён", True)
            else:
                check("симлинк наружу отклонён", False, "прочитался")


def test_agent_tools() -> None:
    print("агент: инструменты")
    from app.agent.policy import AgentLimits, Workspace
    from app.agent.tools import TOOL_SCHEMAS, call_tool

    names = [item["function"]["name"] for item in TOOL_SCHEMAS]
    check("три инструмента",
          names == ["list_dir", "read_file", "search_text"], str(names))
    check("у всех есть описание",
          all(item["function"].get("description") for item in TOOL_SCHEMAS))

    with tempfile.TemporaryDirectory(prefix="mlvh_tools_") as tmp:
        root = Path(tmp)
        (root / "a.txt").write_text("Привет, мир\n", encoding="utf-8")
        workspace = Workspace([root], AgentLimits())

        result = call_tool("read_file", '{"path": "a.txt"}', workspace)
        check("read_file через диспетчер",
              result.ok and "Привет" in result.content, result.content)
        check("краткое описание для чата", "читаю" in result.summary, result.summary)

        bad_json = call_tool("read_file", "{oops", workspace)
        check("битый JSON — ошибка",
              not bad_json.ok and "JSON" in bad_json.content, bad_json.content)

        unknown = call_tool("write_file", "{}", workspace)
        check("неизвестный инструмент",
              not unknown.ok and "неизвестный" in unknown.content, unknown.content)

        outside = call_tool("read_file", json.dumps({"path": "../x"}), workspace)
        check("инструмент не выпускает наружу", not outside.ok, outside.content)

        query = call_tool("search_text", json.dumps({"query": "мир"}), workspace)
        check("search_text через диспетчер",
              query.ok and "a.txt" in query.content, query.content)


def test_agent_context() -> None:
    print("агент: контекст")
    from app.agent.context import (
        build_messages,
        context_budget,
        estimate_tokens,
        trim_messages,
    )

    check("оценка растёт",
          estimate_tokens("привет") > 0
          and estimate_tokens("привет" * 10) > estimate_tokens("привет"))
    check("бюджет меньше окна",
          context_budget(1_000_000, 2000) < 1_000_000)
    check("бюджет не уходит в минус",
          context_budget(32_000, 30_000) >= 4096)

    system = {"role": "system", "content": "системный промпт"}
    history: list[dict] = []
    for index in range(20):
        history.append({"role": "user", "content": f"вопрос {index} " + "а" * 500})
        history.append({"role": "assistant", "content": f"ответ {index} " + "б" * 500})
    history.append({"role": "user", "content": "последний вопрос"})

    result = trim_messages([system, *history], 1500)
    check("системный промпт сохранён",
          result.messages[0]["role"] == "system")
    check("последний вопрос сохранён",
          result.messages[-1]["content"] == "последний вопрос",
          str(result.messages[-1])[:120])
    check("старое выброшено", result.dropped > 0, str(result.dropped))

    messages = [
        system,
        {"role": "user", "content": "что в файле?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "данные " + "х" * 4000},
        {"role": "user", "content": "и ещё?"},
    ]

    intact = trim_messages(messages, 3000)
    roles = [message["role"] for message in intact.messages]
    check("группа tool не разорвана",
          "tool" in roles
          and roles[roles.index("tool") - 1] == "assistant",
          str(roles))

    dropped = trim_messages(messages, 800)
    roles = [message["role"] for message in dropped.messages]
    check("tool без assistant не остаётся",
          not (roles and roles[0] == "tool"), str(roles))
    check("вопрос не потерян",
          dropped.messages[-1]["content"] == "и ещё?", str(roles))

    huge = trim_messages(
        [system, {"role": "user", "content": "а" * 100_000}],
        1000,
    )
    check("огромный вопрос обрезан, но не потерян",
          huge.truncated and huge.messages[-1]["role"] == "user",
          str(huge.truncated))

    built = build_messages(
        "система",
        [{"role": "user", "content": "привет"}],
        1_000_000,
        2000,
    )
    check("build_messages добавляет system",
          built.messages[0]["content"] == "система")


def test_tool_call_accumulator() -> None:
    print("агент: склейка tool_calls")
    from types import SimpleNamespace as NS

    from app.deepseek_client import ToolCallAccumulator

    acc = ToolCallAccumulator()
    acc.feed([NS(index=0, id="call_1", function=NS(name="read_file", arguments='{"pa'))])
    acc.feed([NS(index=0, id=None, function=NS(name=None, arguments='th": "a.txt"}'))])
    calls = acc.finish()
    check("один вызов", len(calls) == 1, str(len(calls)))
    check("id собран", bool(calls) and calls[0].id == "call_1",
          calls[0].id if calls else "нет")
    check("аргументы собраны",
          bool(calls) and calls[0].arguments == '{"path": "a.txt"}',
          calls[0].arguments if calls else "нет")

    parallel = ToolCallAccumulator()
    parallel.feed(
        [
            NS(index=0, id="a", function=NS(name="list_dir", arguments="{}")),
            NS(index=1, id="b", function=NS(name="read_file", arguments='{"path":"x"}')),
        ]
    )
    two = parallel.finish()
    check("два параллельных вызова",
          len(two) == 2 and two[0].name == "list_dir" and two[1].name == "read_file",
          str(two))

    empty = ToolCallAccumulator()
    empty.feed([NS(index=0, id=None, function=NS(name="search_text", arguments=None))])
    synth = empty.finish()
    check("id синтезирован",
          bool(synth) and synth[0].id == "call_0",
          synth[0].id if synth else "нет")
    check("пустые аргументы — объект",
          bool(synth) and synth[0].arguments == "{}",
          synth[0].arguments if synth else "нет")


def test_agent_runner() -> None:
    print("агент: цикл и защита от зацикливания")
    import threading

    from app.agent.context import context_budget, estimate_message_tokens
    from app.agent.policy import AgentLimits, Workspace
    from app.agent.runner import AgentRunner
    from app.deepseek_client import StreamEvent, ToolCall

    class FakeClient:
        model = "fake"
        max_tokens = 1000
        thinking = False

        def __init__(self, script, tick=None):
            self.script = list(script)
            self.calls = 0
            self.seen = []
            self.tick = tick

        def stream_chat(self, messages, tools=None):
            self.calls += 1
            self.seen.append([dict(message) for message in messages])
            if self.tick is not None:
                self.tick()
            step = self.script.pop(0) if self.script else []
            if not step:
                step = [StreamEvent(kind="content", text="Готово.")]
            for event in step:
                yield event

        def clone(self, **_kwargs):
            return self

    with tempfile.TemporaryDirectory(prefix="mlvh_runner_") as tmp:
        root = Path(tmp)
        (root / "a.txt").write_text("строка один\nстрока два\n", encoding="utf-8")
        limits = AgentLimits(
            max_rounds=4,
            duplicate_limit=2,
            max_tool_calls=10,
            max_seconds=60,
            max_file_bytes=100_000,
            max_read_lines=50,
            max_tool_chars=100_000,
        )
        workspace = Workspace([root], limits)
        base = [
            {"role": "system", "content": "тест"},
            {"role": "user", "content": "что в a.txt?"},
        ]

        client = FakeClient(
            [
                [
                    StreamEvent(
                        kind="usage",
                        usage={
                            "prompt_tokens": 10,
                            "completion_tokens": 1,
                            "total_tokens": 11,
                        },
                    ),
                    StreamEvent(
                        kind="tool_call",
                        tool_call=ToolCall("1", "read_file", '{"path": "a.txt"}'),
                    ),
                ],
                [
                    StreamEvent(
                        kind="usage",
                        usage={
                            "prompt_tokens": 20,
                            "completion_tokens": 5,
                            "total_tokens": 25,
                        },
                    ),
                    StreamEvent(kind="content", text="Там две строки."),
                ],
            ]
        )
        summaries: list[str] = []
        reply = AgentRunner(
            client,
            base,
            workspace,
            limits,
            context_tokens=100_000,
            max_tokens=1000,
            on_tool=summaries.append,
        ).run()
        check("ответ после инструмента",
              reply is not None and "две строки" in reply.text,
              reply.text if reply else "None")
        check("один вызов инструмента",
              reply is not None and reply.tool_calls == 1,
              str(reply.tool_calls if reply else "None"))
        check("подпись вызова в чат",
              bool(summaries) and "читаю" in summaries[0], str(summaries))
        tool_messages = [
            message for message in client.seen[-1] if message.get("role") == "tool"
        ]
        check("tool-ответ с тем же id",
              bool(tool_messages)
              and tool_messages[0].get("tool_call_id") == "1",
              str(tool_messages)[:160])
        check("вход суммируется, максимум учтён",
              reply is not None
              and reply.usage.get("prompt_tokens") == 30
              and reply.usage.get("max_prompt_tokens") == 20,
              str(reply.usage if reply else "None"))

        same = ToolCall("x", "read_file", '{"path": "a.txt"}')
        dup_script = [
            [StreamEvent(kind="tool_call", tool_call=same)] for _ in range(4)
        ] + [[StreamEvent(kind="content", text="Итог.")]]
        dup_reply = AgentRunner(
            FakeClient(dup_script),
            base,
            workspace,
            limits,
            context_tokens=100_000,
            max_tokens=1000,
        ).run()
        check("дубликаты остановлены",
              dup_reply is not None and dup_reply.stopped == "stuck",
              dup_reply.stopped if dup_reply else "None")
        check("повторов не больше лимита",
              dup_reply is not None
              and dup_reply.tool_calls <= limits.duplicate_limit + 2,
              str(dup_reply.tool_calls if dup_reply else "None"))
        check("финальный ответ при дубликатах",
              dup_reply is not None and "Итог" in dup_reply.text,
              dup_reply.text if dup_reply else "None")

        round_script = [
            [
                StreamEvent(
                    kind="tool_call",
                    tool_call=ToolCall(
                        f"c{index}",
                        "read_file",
                        json.dumps({"path": "a.txt", "start_line": index + 1}),
                    ),
                )
            ]
            for index in range(8)
        ]
        round_script.insert(4, [StreamEvent(kind="content", text="Хватит.")])
        round_reply = AgentRunner(
            FakeClient(round_script),
            base,
            workspace,
            limits,
            context_tokens=100_000,
            max_tokens=1000,
        ).run()
        check("лимит шагов сработал",
              round_reply is not None and round_reply.stopped == "rounds",
              round_reply.stopped if round_reply else "None")
        check("шагов не больше лимита",
              round_reply is not None
              and round_reply.rounds <= limits.max_rounds + 1,
              str(round_reply.rounds if round_reply else "None"))
        check("по прочитанному всё равно отвечает",
              round_reply is not None and "Хватит" in round_reply.text,
              round_reply.text if round_reply else "None")

        cancel = threading.Event()
        cancel.set()
        cancelled_client = FakeClient([])
        cancelled = AgentRunner(
            cancelled_client,
            base,
            workspace,
            limits,
            context_tokens=100_000,
            max_tokens=1000,
            cancel=cancel,
        ).run()
        check("отмена возвращает None", cancelled is None)
        check("отмена не делает запросов", cancelled_client.calls == 0,
              str(cancelled_client.calls))

        now = [0.0]

        def clock() -> float:
            return now[0]

        def tick() -> None:
            now[0] += 100

        time_client = FakeClient(
            [[StreamEvent(kind="tool_call", tool_call=ToolCall("t", "list_dir", "{}"))]],
            tick=tick,
        )
        time_reply = AgentRunner(
            time_client,
            base,
            workspace,
            limits,
            context_tokens=100_000,
            max_tokens=1000,
            clock=clock,
        ).run()
        check("лимит времени сработал",
              time_reply is not None and time_reply.stopped == "time",
              time_reply.stopped if time_reply else "None")

        bad_reply = AgentRunner(
            FakeClient(
                [
                    [
                        StreamEvent(
                            kind="tool_call",
                            tool_call=ToolCall(
                                "bad", "read_file", '{"path": "../secret"}'
                            ),
                        )
                    ],
                    [StreamEvent(kind="content", text="Не нашёл.")],
                ]
            ),
            base,
            workspace,
            limits,
            context_tokens=100_000,
            max_tokens=1000,
        ).run()
        check("ошибка инструмента не роняет цикл",
              bad_reply is not None and "Не нашёл" in bad_reply.text,
              bad_reply.text if bad_reply else "None")

        empty_reply = AgentRunner(
            FakeClient([[StreamEvent(kind="content", text="")]]),
            base,
            workspace,
            limits,
            context_tokens=100_000,
            max_tokens=1000,
        ).run()
        check("пустой ответ даёт понятный текст",
              empty_reply is not None and bool(empty_reply.text.strip()),
              empty_reply.text if empty_reply else "None")

        # Большой результат чтения не должен раздувать следующий запрос:
        # перед каждым раундом история обрезается под окно модели.
        (root / "big.txt").write_text(
            "\n".join(["а" * 300 for _ in range(50)]), encoding="utf-8"
        )
        budget = context_budget(6000, 1000)
        big_client = FakeClient(
            [
                [
                    StreamEvent(
                        kind="tool_call",
                        tool_call=ToolCall("big", "read_file", '{"path": "big.txt"}'),
                    )
                ],
                [StreamEvent(kind="content", text="Готово.")],
            ]
        )
        big_reply = AgentRunner(
            big_client,
            base,
            workspace,
            limits,
            context_tokens=6000,
            max_tokens=1000,
        ).run()
        last_tokens = sum(
            estimate_message_tokens(message) for message in big_client.seen[-1]
        )
        check("результат инструмента обрезается под окно",
              big_reply is not None and last_tokens <= budget,
              f"{last_tokens} > {budget}")


def test_core_without_qt() -> None:
    """В CI PySide6 не ставится: ядро не должно его тянуть даже косвенно."""
    print("ядро без Qt:")
    code = (
        "import sys;"
        "import app.config, app.personas, app.text_norm;"
        "import app.asr.audio, app.asr.base, app.asr.hotkeys;"
        "import app.asr.wakeword, app.asr.registry;"
        "import app.agent, app.agent.policy, app.agent.context;"
        "import app.agent.tools, app.agent.runner;"
        "print('PySide6' in sys.modules)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        check("ядро импортируется без Qt", False, str(exc))
        return

    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-1:]
        check("ядро импортируется без Qt", False, " ".join(tail))
        return

    check("ядро не тянет PySide6", result.stdout.strip() == "False", result.stdout.strip())


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
        test_agent_policy,
        test_agent_tools,
        test_agent_context,
        test_tool_call_accumulator,
        test_agent_runner,
        test_core_without_qt,
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
