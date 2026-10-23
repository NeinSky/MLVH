"""Аудио для распознавания: приведение к 16 кГц моно, уровень, обрезка тишины.

Модуль `audioop` удалён в Python 3.13, поэтому вся арифметика — на numpy.
Формат на входе любой (int16/int32/float), на выходе всегда float32 моно.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

TARGET_RATE = 16000


def as_float(samples: np.ndarray) -> np.ndarray:
    """Целые отсчёты приводим к float32 в диапазоне −1…1."""
    if samples.dtype == np.int16:
        return samples.astype(np.float32) / 32768.0
    if samples.dtype == np.int32:
        return samples.astype(np.float32) / 2147483648.0
    if samples.dtype == np.uint8:
        return (samples.astype(np.float32) - 128.0) / 128.0
    return samples.astype(np.float32, copy=False)


def to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    """Стерео (или больше) сводим в моно усреднением каналов."""
    if channels <= 1:
        return samples
    usable = (len(samples) // channels) * channels
    if usable == 0:
        return samples[:0]
    return samples[:usable].reshape(-1, channels).mean(axis=1)


def resample(samples: np.ndarray, src_rate: int, dst_rate: int = TARGET_RATE) -> np.ndarray:
    """Пересэмплирование: для кратных частот — усреднение, иначе линейная интерполяция."""
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)

    if src_rate > dst_rate and src_rate % dst_rate == 0:
        factor = src_rate // dst_rate
        usable = (len(samples) // factor) * factor
        if usable == 0:
            return samples[:0]
        return samples[:usable].reshape(-1, factor).mean(axis=1).astype(np.float32, copy=False)

    duration = len(samples) / float(src_rate)
    count = max(1, int(round(duration * dst_rate)))
    old = np.linspace(0.0, duration, num=len(samples), endpoint=False, dtype=np.float32)
    new = np.linspace(0.0, duration, num=count, endpoint=False, dtype=np.float32)
    return np.interp(new, old, samples).astype(np.float32, copy=False)


def to_mono_16k(samples: np.ndarray, rate: int, channels: int = 1) -> np.ndarray:
    """Единый формат для всех движков распознавания."""
    return resample(to_mono(as_float(samples), channels), int(rate), TARGET_RATE)


def rms(samples: np.ndarray) -> float:
    """Среднеквадратичная громкость: 0 — тишина, ~0.1 — обычная речь."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def peak(samples: np.ndarray) -> float:
    """Максимальная амплитуда: удобно для проверки «клиппинга»."""
    if samples.size == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def level(rms_value: float) -> float:
    """Громкость для индикатора: корень, чтобы тихая речь была заметна."""
    return float(min(1.0, (max(0.0, rms_value) ** 0.5) * 3.0))


def trim_silence(
    samples: np.ndarray,
    rate: int = TARGET_RATE,
    threshold: float = 0.008,
    keep_ms: int = 200,
) -> np.ndarray:
    """Обрезаем тишину по краям, оставляя запас keep_ms с каждой стороны."""
    if samples.size == 0:
        return samples

    step = max(1, int(rate * 0.01))  # окна по 10 мс
    frames = len(samples) // step
    if frames == 0:
        return samples

    loud = np.max(np.abs(samples[: frames * step]).reshape(frames, step), axis=1) >= threshold
    if not loud.any():
        return samples[:0]

    first = int(np.argmax(loud))
    last = frames - int(np.argmax(loud[::-1]))
    keep = int(rate * keep_ms / 1000)
    start = max(0, first * step - keep)
    end = min(len(samples), last * step + keep)
    return samples[start:end]


def write_wav(path: Path, samples: np.ndarray, rate: int = TARGET_RATE) -> Path:
    """Пишем int16-моно wav: для прослушивания и сохранения проверок."""
    data = np.clip(as_float(samples), -1.0, 1.0)
    pcm = (data * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(pcm.tobytes())
    return path


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Читаем wav в float32 моно.

    Обычный 16-битный PCM читает стандартный модуль `wave`; float32-wav
    (его пишет, например, torchaudio) ему недоступен — тогда берём soundfile.
    """
    try:
        return _read_pcm16(path)
    except Exception:
        pass
    return _read_soundfile(path)


def _read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    if width != 2:
        raise ValueError(f"не 16-битный wav: {width * 8} бит")
    data = np.frombuffer(raw, dtype=np.int16)
    return to_mono(as_float(data), channels), rate


def _read_soundfile(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile
    except ImportError as exc:
        raise ValueError(
            "не удалось прочитать wav: для float-формата нужен пакет soundfile"
        ) from exc

    try:
        data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"не удалось прочитать wav: {exc}") from exc
    return to_mono(data, int(data.shape[1])), int(rate)
