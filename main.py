"""Точка входа MLVH: окно, а также режимы для проверки голоса без интерфейса."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from app import __version__
from app.config import Settings, get_api_key, load_settings
from app.personas import persona_by_id


def _fix_console() -> None:
    """Русский текст в консоли Windows без «кракозябр»."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    _fix_console()
    parser = argparse.ArgumentParser(description="MLVH — My Little Voice Helper")
    parser.add_argument(
        "--version",
        action="version",
        version=f"MLVH {__version__}",
        help="показать версию и выйти",
    )
    parser.add_argument(
        "--list-voices",
        nargs="?",
        const="ru",
        metavar="PREFIX",
        help="показать голоса Edge (например ru) и выйти",
    )
    parser.add_argument("--say", metavar="TEXT", help="озвучить текст и выйти, без окна")
    parser.add_argument("--voice", help="голос для --say")
    parser.add_argument("--rate", type=int, help="темп для --say, от -50 до 50")
    parser.add_argument("--no-voice", action="store_true", help="запустить окно без озвучки")
    parser.add_argument(
        "--persona",
        metavar="ID",
        help="характер: elvi (по умолчанию), helper, anime, mentor, ironic или свой из config.json",
    )
    parser.add_argument("--list-personas", action="store_true", help="показать характеры и выйти")
    parser.add_argument(
        "--mic-test",
        nargs="?",
        const=3.0,
        type=float,
        metavar="SECONDS",
        help="записать с микрофона и показать уровень (по умолчанию 3 с)",
    )
    parser.add_argument(
        "--transcribe",
        metavar="FILE",
        help="распознать wav-файл и выйти",
    )
    parser.add_argument(
        "--ask",
        metavar="TEXT",
        help="задать один вопрос агенту в консоли (без окна и озвучки)",
    )
    parser.add_argument(
        "--read-folder",
        action="append",
        default=[],
        metavar="PATH",
        help="разрешить агенту чтение папки для --ask (можно несколько раз)",
    )
    parser.add_argument(
        "--agent-rounds",
        type=int,
        metavar="N",
        help="предел шагов агента для --ask (по умолчанию 20)",
    )
    args = parser.parse_args()

    settings = load_settings()
    if args.no_voice:
        settings.tts.enabled = False

    if args.mic_test:
        return _mic_test(args.mic_test)
    if args.transcribe:
        return _transcribe_file(args.transcribe, settings)
    if args.ask:
        return _ask(args.ask, settings, args)

    if args.list_personas:
        return _list_personas(settings)
    if args.persona:
        persona = persona_by_id(settings.personas, args.persona)
        if persona is None:
            print(f"Нет характера «{args.persona}». Доступные:")
            return _list_personas(settings)
        settings.apply_persona(persona)

    if args.list_voices is not None:
        return _list_voices(args.list_voices)
    if args.say:
        return _say(args.say, settings, args)
    return _run_gui(settings)


def _list_personas(settings: Settings) -> int:
    print("Характеры:")
    for persona in settings.personas:
        mark = "*" if persona.id == settings.active_persona else " "
        print(
            f" {mark} {persona.id:<8} {persona.name:<12} {persona.voice:<24} "
            f"темп {persona.rate:+d}%  тон {persona.pitch:+d} Гц"
        )
    return 0



def _list_voices(prefix: str) -> int:
    from app.tts import TTSError, TTSService, voice_label

    try:
        voices = TTSService.list_voices(prefix)
    except TTSError as exc:
        print(f"Ошибка: {exc}")
        return 1

    print(f"Голосов: {len(voices)}")
    for voice in voices:
        short_name = voice.get("ShortName", "")
        print(f"  {short_name:<36} {voice_label(voice)}")
    return 0


def _say(text: str, settings: Settings, args: argparse.Namespace) -> int:
    from app.tts import TTSError, TTSService

    tts_settings = settings.tts
    if args.voice:
        tts_settings.voice = args.voice
    if args.rate is not None:
        tts_settings.rate = max(-50, min(50, args.rate))

    try:
        TTSService().speak(text, tts_settings)
    except TTSError as exc:
        print(f"Ошибка: {exc}")
        return 1
    return 0


def _mic_test(seconds: float) -> int:
    """Проверка микрофона без окна: устройства, уровень, файл для прослушивания."""
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

    from app.asr import audio
    from app.asr.recorder import MicRecorder

    QCoreApplication.instance() or QCoreApplication(sys.argv)

    devices = MicRecorder.devices()
    if not devices:
        print("Микрофон не найден.")
        return 1
    print("Устройства ввода:")
    for name in devices:
        print(f"  - {name}")

    recorder = MicRecorder()
    loop = QEventLoop()
    errors: list[str] = []
    recorder.failed.connect(lambda message: (errors.append(message), loop.quit()))

    if not recorder.start():
        print(f"Ошибка: {errors[-1] if errors else 'не удалось открыть микрофон'}")
        return 1

    print(f"Запись {seconds:.0f} с — говорите…")
    recorder.begin_capture()
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec()

    samples = recorder.end_capture()
    rate = recorder.rate
    recorder.stop()

    if samples.size == 0:
        print("Ничего не записалось.")
        return 1

    data = audio.to_mono_16k(samples, rate, 1)
    print(
        f"Записано: {data.size / audio.TARGET_RATE:.2f} с, "
        f"пик {audio.peak(data):.3f}, RMS {audio.rms(data):.3f}"
    )
    path = Path(tempfile.gettempdir()) / "mlvh_mic_test.wav"
    audio.write_wav(path, data)
    print(f"Файл: {path}")

    if audio.rms(data) < 0.004:
        print("Похоже на тишину: проверьте устройство и разрешение Windows.")
        return 1
    return 0


def _transcribe_file(path: str, settings: Settings) -> int:
    """Распознаём готовый wav: удобно проверять движок без микрофона."""
    from app.asr import audio
    from app.asr.registry import get_asr_engine

    source = Path(path)
    if not source.exists():
        print(f"Файл не найден: {source}")
        return 1

    engine = get_asr_engine(settings.asr.engine)
    info = engine.probe()
    if not info.available:
        print(f"{info.title} недоступен: {info.reason or info.hint}")
        return 1

    try:
        samples, rate = audio.read_wav(source)
    except (OSError, ValueError) as exc:
        print(f"Не удалось прочитать wav: {exc}")
        return 1

    data = audio.to_mono_16k(samples, rate, 1)
    print(f"Распознаю {data.size / audio.TARGET_RATE:.1f} с через {info.title}…")
    try:
        text = engine.transcribe(data, settings.asr)
    except Exception as exc:
        print(f"Ошибка: {exc}")
        return 1

    print(text or "(пусто)")
    return 0


def _ask(text: str, settings: Settings, args) -> int:
    """Один вопрос агенту в консоли: проверка чтения файлов без окна."""
    from dataclasses import replace

    from app.agent import (
        AgentLimits,
        AgentRunner,
        Workspace,
        build_agent_prompt,
        build_messages,
    )
    from app.deepseek_client import DeepSeekClient, DeepSeekError

    key = get_api_key()
    if not key:
        print("Не задан ключ DeepSeek: положите DEEPSEEK_API_KEY в .env.")
        return 1

    model = settings.model
    max_tokens = model.max_tokens
    if model.thinking:
        max_tokens = max(max_tokens, 2000)

    limits = AgentLimits.from_settings(settings.agent)
    if args.agent_rounds:
        limits = replace(limits, max_rounds=max(1, args.agent_rounds))

    roots = list(args.read_folder or [])
    if not roots and settings.agent.read_enabled:
        roots = list(settings.agent.roots)

    workspace = Workspace(roots, limits) if roots else None
    if workspace is not None and not workspace:
        for raw, why in workspace.skipped:
            print(f"Папка пропущена: {raw} — {why}")
        workspace = None

    system_prompt = settings.system_prompt
    if workspace is not None:
        system_prompt = build_agent_prompt(system_prompt, workspace, limits)

    trimmed = build_messages(
        system_prompt,
        [{"role": "user", "content": text}],
        model.context_tokens,
        max_tokens,
    )

    try:
        client = DeepSeekClient(
            api_key=key,
            model=model.name,
            temperature=model.temperature,
            max_tokens=max_tokens,
            thinking=model.thinking,
        )
    except DeepSeekError as exc:
        print(f"Ошибка: {exc}")
        return 1

    if workspace is None:
        try:
            for event in client.stream_chat(trimmed.messages):
                if event.kind == "content":
                    print(event.text, end="", flush=True)
        except DeepSeekError as exc:
            print(f"\nОшибка: {exc}")
            return 1
        print()
        return 0

    def on_tool(summary: str) -> None:
        print(f"[агент] {summary}", flush=True)

    def on_note(message: str) -> None:
        print(f"[агент] {message}", flush=True)

    streamed = False

    def on_content(chunk: str) -> None:
        nonlocal streamed
        streamed = True
        print(chunk, end="", flush=True)

    runner = AgentRunner(
        client,
        trimmed.messages,
        workspace,
        limits,
        context_tokens=model.context_tokens,
        max_tokens=max_tokens,
        on_content=on_content,
        on_note=on_note,
        on_tool=on_tool,
    )
    try:
        reply = runner.run()
    except DeepSeekError as exc:
        print(f"\nОшибка: {exc}")
        return 1

    if reply is None:
        print("\nОстановлено.")
        return 1
    if not streamed and reply.text:
        print(reply.text, end="")
    print()
    print(
        f"Шагов: {reply.rounds}, вызовов: {reply.tool_calls}, "
        f"причина остановки: {reply.stopped}"
    )
    return 0 if reply.text.strip() else 1


def _run_gui(settings: Settings) -> int:
    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("MLVH")
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")

    window = MainWindow(settings)
    window.show()

    if not get_api_key():
        print("Ключ DeepSeek не задан: нажмите «Ввести ключ» в окне MLVH.")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
