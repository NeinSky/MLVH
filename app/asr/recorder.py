"""Запись с микрофона через QtMultimedia: живой уровень и буфер отсчётов."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices

from app.asr import audio

_BYTES_PER_SAMPLE = {
    QAudioFormat.SampleFormat.UInt8: 1,
    QAudioFormat.SampleFormat.Int16: 2,
    QAudioFormat.SampleFormat.Int32: 4,
    QAudioFormat.SampleFormat.Float: 4,
}


class MicRecorder(QObject):
    """Один поток с микрофона: уровень отдаём всегда, отсчёты — по команде.

    `start()` открывает устройство, `begin_capture()` начинает копить буфер,
    `end_capture()` возвращает записанное (float32 моно на частоте устройства).
    """

    level = Signal(float)
    failed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._source: QAudioSource | None = None
        self._io = None
        self._chunks: list[np.ndarray] = []
        self._capturing = False
        self._rate = audio.TARGET_RATE
        self._channels = 1
        self._sample_format = QAudioFormat.SampleFormat.Int16
        self._device_name = ""

        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._pull)

    # --- устройства ------------------------------------------------------
    @staticmethod
    def devices() -> list[str]:
        return [device.description() for device in QMediaDevices.audioInputs()]

    @staticmethod
    def _pick_device(name: str):
        inputs = QMediaDevices.audioInputs()
        for device in inputs:
            if name and device.description() == name:
                return device
        default = QMediaDevices.defaultAudioInput()
        if default is not None and not default.isNull():
            return default
        return inputs[0] if inputs else None

    # --- состояние -------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return self._source is not None

    @property
    def rate(self) -> int:
        return self._rate

    @property
    def device_name(self) -> str:
        return self._device_name

    def start(self, device_name: str = "") -> bool:
        """Открываем микрофон. Уровень начинает идти сразу."""
        if self._source is not None:
            return True

        device = self._pick_device(device_name)
        if device is None or device.isNull():
            self.failed.emit("Микрофон не найден")
            return False

        fmt = QAudioFormat()
        fmt.setSampleRate(audio.TARGET_RATE)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if not device.isFormatSupported(fmt):
            # Устройство не умеет 16 кГц — берём то, что умеет, и пересэмплируем.
            fmt = device.preferredFormat()

        self._rate = fmt.sampleRate() or audio.TARGET_RATE
        self._channels = max(1, fmt.channelCount())
        self._sample_format = fmt.sampleFormat()

        source = QAudioSource(device, fmt)
        io = source.start()
        if io is None:
            self.failed.emit(f"Не удалось открыть микрофон «{device.description()}»")
            return False

        source.setVolume(1.0)
        self._source = source
        self._io = io
        self._device_name = device.description()
        self._chunks = []
        self._capturing = False
        self._timer.start()
        return True

    def stop(self) -> None:
        self._timer.stop()
        self._capturing = False
        source = self._source
        self._source = None
        self._io = None
        if source is not None:
            try:
                source.stop()
            except Exception:
                pass

    # --- запись ----------------------------------------------------------
    def begin_capture(self) -> None:
        self._chunks = []
        self._capturing = True

    def end_capture(self) -> np.ndarray:
        self._capturing = False
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        data = np.concatenate(self._chunks).astype(np.float32, copy=False)
        self._chunks = []
        return data

    @property
    def captured_seconds(self) -> float:
        if not self._chunks or not self._rate:
            return 0.0
        return sum(len(chunk) for chunk in self._chunks) / float(self._rate)

    # --- чтение потока ---------------------------------------------------
    def _pull(self) -> None:
        if self._io is None:
            return
        try:
            data = bytes(self._io.readAll())
        except Exception as exc:
            self.failed.emit(f"Микрофон: {exc}")
            self.stop()
            return

        if not data:
            return

        samples = self._convert(data)
        if samples.size == 0:
            return
        if self._capturing:
            self._chunks.append(samples)
        self.level.emit(audio.level(audio.rms(samples)))

    def _convert(self, data: bytes) -> np.ndarray:
        width = _BYTES_PER_SAMPLE.get(self._sample_format, 2)
        data = data[: len(data) - len(data) % width]
        if not data:
            return np.empty(0, dtype=np.float32)

        if self._sample_format == QAudioFormat.SampleFormat.Int16:
            samples = np.frombuffer(data, dtype=np.int16)
        elif self._sample_format == QAudioFormat.SampleFormat.Int32:
            samples = np.frombuffer(data, dtype=np.int32)
        elif self._sample_format == QAudioFormat.SampleFormat.Float:
            samples = np.frombuffer(data, dtype=np.float32)
        elif self._sample_format == QAudioFormat.SampleFormat.UInt8:
            samples = np.frombuffer(data, dtype=np.uint8)
        else:
            return np.empty(0, dtype=np.float32)

        converted = audio.as_float(samples)
        if self._channels > 1:
            converted = audio.to_mono(converted, self._channels)
        return converted
