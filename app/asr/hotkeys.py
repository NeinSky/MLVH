"""Разбор горячих клавиш: без Qt и WinAPI, чтобы проверялось в CI."""

from __future__ import annotations

import re
from dataclasses import dataclass

MODIFIER_ORDER = ("Ctrl", "Alt", "Shift", "Meta")

_ALIASES = {
    "ctrl": "Ctrl",
    "control": "Ctrl",
    "alt": "Alt",
    "shift": "Shift",
    "meta": "Meta",
    "win": "Meta",
    "cmd": "Meta",
    "super": "Meta",
}

# Виртуальные коды Windows: нужны для RegisterHotKey.
_SPECIAL_VK = {
    "Space": 0x20,
    "Enter": 0x0D,
    "Tab": 0x09,
    "Esc": 0x1B,
    "Backspace": 0x08,
    "Delete": 0x2E,
    "Insert": 0x2D,
    "Home": 0x24,
    "End": 0x23,
    "PageUp": 0x21,
    "PageDown": 0x22,
    "Left": 0x25,
    "Up": 0x26,
    "Right": 0x27,
    "Down": 0x28,
    "Pause": 0x13,
    "CapsLock": 0x14,
}

_MODIFIER_BITS = {"Alt": 0x0001, "Ctrl": 0x0002, "Shift": 0x0004, "Meta": 0x0008}


@dataclass(frozen=True)
class Hotkey:
    """Разобранная комбинация: модификаторы в каноническом порядке и клавиша."""

    modifiers: tuple[str, ...]
    key: str

    def text(self) -> str:
        return "+".join((*self.modifiers, self.key))

    def vk(self) -> int | None:
        return to_vk(self.key)

    def modifiers_bits(self) -> int:
        bits = 0
        for name in self.modifiers:
            bits |= _MODIFIER_BITS.get(name, 0)
        return bits


def to_vk(key: str) -> int | None:
    """Виртуальный код клавиши Windows (или None, если неизвестна)."""
    if len(key) == 1 and key.isalnum():
        return ord(key.upper())
    if key in _SPECIAL_VK:
        return _SPECIAL_VK[key]
    match = re.fullmatch(r"F([1-9]|1[0-9]|2[0-4])", key)
    if match:
        return 0x70 + int(match.group(1)) - 1
    return None


def _normalize_key(part: str) -> str | None:
    """«space» → «Space», «f5» → «F5», «а» → «А»."""
    if len(part) == 1:
        return part.upper()
    for name in _SPECIAL_VK:
        if part.lower() == name.lower():
            return name
    match = re.fullmatch(r"[fF]([1-9]|1[0-9]|2[0-4])", part)
    if match:
        return f"F{int(match.group(1))}"
    return None


def parse(text: str) -> Hotkey | None:
    """«ctrl+space» → Hotkey(("Ctrl",), "Space"). Мусор → None."""
    if not text or not text.strip():
        return None

    parts = [part.strip() for part in text.split("+") if part.strip()]
    if len(parts) < 2:
        return None

    modifiers: list[str] = []
    for part in parts[:-1]:
        canonical = _ALIASES.get(part.lower())
        if canonical is None or canonical in modifiers:
            return None
        modifiers.append(canonical)

    key = _normalize_key(parts[-1])
    if key is None:
        return None

    ordered = tuple(name for name in MODIFIER_ORDER if name in modifiers)
    return Hotkey(ordered, key)


def normalize(text: str) -> str:
    """Канонический вид для сравнения с Qt: «ctrl + space» → «Ctrl+Space»."""
    hotkey = parse(text)
    return hotkey.text() if hotkey is not None else ""
