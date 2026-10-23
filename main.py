"""Точка входа MLVH: окно, а также режимы для проверки голоса без интерфейса."""

from __future__ import annotations

import argparse
import sys

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
    args = parser.parse_args()

    settings = load_settings()
    if args.no_voice:
        settings.tts.enabled = False

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


def _run_gui(settings: Settings) -> int:
    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("MLVH")
    app.setStyle("Fusion")

    window = MainWindow(settings)
    window.show()

    if not get_api_key():
        print("Ключ DeepSeek не задан: нажмите «Ввести ключ» в окне MLVH.")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
