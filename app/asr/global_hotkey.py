"""Глобальные горячие клавиши Windows через RegisterHotKey — без сторонних пакетов.

Глобально Windows отдаёт только нажатие, без отпускания, поэтому в этом режиме
запись работает переключением: нажали — пишем, нажали ещё раз — стоп.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Signal
from PySide6.QtWidgets import QApplication

from app.asr.hotkeys import Hotkey, to_vk

WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000
ERROR_HOTKEY_ALREADY_REGISTERED = 1409
HOTKEY_ID = 0x4D4C  # «ML»


class _HotkeyFilter(QAbstractNativeEventFilter):
    """Ловим WM_HOTKEY в очереди сообщений потока Qt."""

    def __init__(self, callback, hotkey_id: int) -> None:
        super().__init__()
        self._callback = callback
        self._id = hotkey_id

    def nativeEventFilter(self, event_type, message):
        if event_type in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
            try:
                msg = ctypes.wintypes.MSG.from_address(int(message))
            except Exception:
                return False, 0
            if msg.message == WM_HOTKEY and int(msg.wParam) == self._id:
                self._callback()
        return False, 0


class GlobalHotkey(QObject):
    """Одна глобальная комбинация на приложение."""

    pressed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._registered = False
        self._filter: _HotkeyFilter | None = None

    @property
    def registered(self) -> bool:
        return self._registered

    def register(self, hotkey: Hotkey | None) -> tuple[bool, str]:
        """Регистрируем комбинацию. Возвращаем (успех, причина отказа)."""
        self.unregister()

        if hotkey is None:
            return False, "горячая клавиша не разобрана"
        if sys.platform != "win32":
            return False, "глобальные клавиши поддерживаются только в Windows"

        vk = to_vk(hotkey.key)
        if vk is None:
            return False, f"клавиша «{hotkey.key}» не поддерживается глобально"

        app = QApplication.instance()
        if app is None:
            return False, "нет запущенного приложения Qt"

        if self._filter is None:
            self._filter = _HotkeyFilter(self.pressed.emit, HOTKEY_ID)
            app.installNativeEventFilter(self._filter)

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        if not user32.RegisterHotKey(None, HOTKEY_ID, hotkey.modifiers_bits() | MOD_NOREPEAT, vk):
            code = ctypes.get_last_error()
            if code == ERROR_HOTKEY_ALREADY_REGISTERED:
                return False, "комбинацию уже заняла другая программа"
            return False, f"Windows не разрешила регистрацию (код {code})"

        self._registered = True
        return True, ""

    def unregister(self) -> None:
        if not self._registered:
            return
        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.UnregisterHotKey(None, HOTKEY_ID)
        except Exception:
            pass
        self._registered = False
