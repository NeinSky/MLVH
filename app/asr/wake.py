"""Спецслово: фоновый слушатель на Vosk.

Сравнение слов живёт в `app.asr.wakeword` — там нет Qt, поэтому логика
проверяется в CI. Здесь только поток и работа с аудио.
"""

from __future__ import annotations

import json
import queue
import threading
import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from app.asr import audio
from app.asr.audio import TARGET_RATE
from app.asr.vosk_engine import model_path
from app.asr.wakeword import strip_wake_word
from app.config import ASRSettings

MAX_PHRASE_SECONDS = 20


class WakeWorker(QThread):
    """Слушает микрофон и ищет спецслово. Аудио приходит через `feed()`."""

    ready = Signal()
    heard = Signal()                  # спецслово услышано, ждём сам вопрос
    detected = Signal(object, str)    # (аудио фразы, текст без спецслова)
    failed = Signal(str)

    def __init__(self, settings: ASRSettings, parent=None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._word = settings.wake_word or "элви"
        self._queue: queue.Queue = queue.Queue(maxsize=200)
        self._stop = threading.Event()

    def feed(self, samples: np.ndarray) -> None:
        """Вызывается из главного потока: только кладём кусок в очередь."""
        if self._stop.is_set() or samples.size == 0:
            return
        try:
            self._queue.put_nowait(samples)
        except queue.Full:
            pass

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def run(self) -> None:
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel

            SetLogLevel(-1)
            model = Model(str(model_path()))
            recognizer = KaldiRecognizer(model, TARGET_RATE)
        except Exception as exc:
            self.failed.emit(f"Vosk не готов: {exc}")
            return

        self.ready.emit()
        buffer: list[np.ndarray] = []
        buffered = 0
        armed = False
        last_voice = time.monotonic()
        armed_until = 0.0
        silence = 1.0

        while not self._stop.is_set():
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                chunk = None

            now = time.monotonic()
            if chunk is not None:
                buffer.append(chunk)
                buffered += len(chunk)
                while buffered > MAX_PHRASE_SECONDS * TARGET_RATE and buffer:
                    buffered -= len(buffer.pop(0))
                if audio.rms(chunk) > 0.008:
                    last_voice = now

                pcm = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                try:
                    final = recognizer.AcceptWaveform(pcm)
                except Exception as exc:
                    self.failed.emit(f"Vosk: {exc}")
                    return

                if final:
                    text = str(json.loads(recognizer.Result()).get("text", "")).strip()
                    rest = strip_wake_word(text, self._word)
                    if rest is None and armed:
                        # Vosk мог проглотить имя — фраза всё равно наша.
                        rest = text
                    if rest:
                        self.detected.emit(self._take(buffer), rest)
                        buffer, buffered = [], 0
                        armed = False
                        continue
                    if rest == "":
                        # Услышали только спецслово — ждём вопрос отдельно.
                        buffer, buffered = [], 0
                        if not armed:
                            armed = True
                            armed_until = now + 6.0
                            self.heard.emit()
                        continue
                elif not armed:
                    partial = str(
                        json.loads(recognizer.PartialResult()).get("partial", "")
                    )
                    if strip_wake_word(partial, self._word) is not None:
                        armed = True
                        armed_until = now + 6.0
                        self.heard.emit()

            # Долгая тишина закрывает сегмент: проверяем, было ли спецслово.
            if now - last_voice > silence and buffered:
                text = str(json.loads(recognizer.FinalResult()).get("text", "")).strip()
                phrase = self._take(buffer)
                buffer, buffered = [], 0
                recognizer = KaldiRecognizer(model, TARGET_RATE)

                was_armed = armed
                armed = False
                rest = strip_wake_word(text, self._word)
                if rest is None and was_armed:
                    # Имя потерялось при распознавании — считаем вопросом весь текст.
                    rest = text

                if rest:
                    self.detected.emit(phrase, rest)
                elif rest == "" and not was_armed:
                    # Сказали только «Элви» — ждём вопрос отдельной фразой.
                    armed = True
                    armed_until = now + 6.0
                    self.heard.emit()
                continue

            if armed and now > armed_until:
                armed = False
                buffer, buffered = [], 0
                recognizer = KaldiRecognizer(model, TARGET_RATE)

    @staticmethod
    def _take(buffer: list[np.ndarray]) -> np.ndarray:
        if not buffer:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(buffer)


