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

from app.asr.audio import TARGET_RATE
from app.asr.wakeword import strip_wake_word
from app.config import ASRSettings

MAX_PHRASE_SECONDS = 20
ARM_TIMEOUT = 6.0        # сколько ждём вопрос после одного «Элви»
SILENCE_SECONDS = 1.0    # текст перестал меняться — фраза закончилась
QUIET_SEGMENT = 2.5      # текста нет вовсе: всё равно разбираем буфер


class WakeWorker(QThread):
    """Слушает микрофон и ищет спецслово. Аудио приходит через `feed()`."""

    ready = Signal()
    heard = Signal()                  # спецслово услышано, ждём сам вопрос
    timeout = Signal()                # ждали вопрос, но так и не дождались
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
            from vosk import KaldiRecognizer, SetLogLevel

            from app.asr.registry import get_asr_engine

            SetLogLevel(-1)
            # Модель берём общую: прогрев уже мог её загрузить, а вторая
            # параллельная загрузка того же каталога подвешивала Vosk.
            model = get_asr_engine("vosk").load_model()
            recognizer = KaldiRecognizer(model, TARGET_RATE)
        except Exception as exc:
            self.failed.emit(f"Vosk не готов: {exc}")
            return

        self.ready.emit()
        buffer: list[np.ndarray] = []
        buffered = 0
        armed = False
        armed_until = 0.0
        partial_text = ""
        # Фразу закрываем по затиханию распознанного текста, а не по громкости:
        # фоновый шум держал «последний голос» свежим, и фраза не закрывалась.
        last_change = time.monotonic()
        segment_start = last_change

        while not self._stop.is_set():
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                chunk = None

            now = time.monotonic()
            if chunk is not None:
                if not buffer:
                    segment_start = now
                buffer.append(chunk)
                buffered += len(chunk)
                while buffered > MAX_PHRASE_SECONDS * TARGET_RATE and buffer:
                    buffered -= len(buffer.pop(0))

                pcm = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                try:
                    final = recognizer.AcceptWaveform(pcm)
                except Exception as exc:
                    self.failed.emit(f"Vosk: {exc}")
                    return

                if final:
                    text = str(json.loads(recognizer.Result()).get("text", "")).strip()
                    partial_text = ""
                    last_change = now
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
                            armed_until = now + ARM_TIMEOUT
                            self.heard.emit()
                        continue
                else:
                    partial = str(
                        json.loads(recognizer.PartialResult()).get("partial", "")
                    )
                    if partial != partial_text:
                        partial_text = partial
                        last_change = now
                    if not armed and strip_wake_word(partial, self._word) is not None:
                        armed = True
                        armed_until = now + ARM_TIMEOUT
                        self.heard.emit()

            # Фраза закончилась: текст стабилен (или его нет вовсе, но буфер
            # уже набрался). Разбираем накопленное и начинаем заново.
            quiet = now - last_change > SILENCE_SECONDS and bool(partial_text)
            empty = now - segment_start > QUIET_SEGMENT and not partial_text
            if buffered and (quiet or empty):
                text = str(json.loads(recognizer.FinalResult()).get("text", "")).strip()
                phrase = self._take(buffer)
                buffer, buffered = [], 0
                partial_text = ""
                last_change = now
                segment_start = now
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
                    armed_until = now + ARM_TIMEOUT
                    self.heard.emit()
                elif was_armed:
                    # Пустой сегмент во время ожидания вопроса: не сбрасываем
                    # сразу, даём договорить. Таймаут решит armed_until.
                    armed = True
                continue

            if armed and now > armed_until:
                # Ждали вопрос и не дождались: сообщаем интерфейсу, иначе
                # индикатор навсегда остаётся в состоянии «слушаю вопрос».
                armed = False
                buffer, buffered = [], 0
                partial_text = ""
                last_change = now
                recognizer = KaldiRecognizer(model, TARGET_RATE)
                self.timeout.emit()

    @staticmethod
    def _take(buffer: list[np.ndarray]) -> np.ndarray:
        if not buffer:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(buffer)


