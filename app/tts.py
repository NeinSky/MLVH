"""Синтез речи (движки) и воспроизведение через pygame."""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import edge_tts

from app.config import TTSSettings
from app.engines.base import TTSError
from app.engines.registry import get_engine
from app.speech_parts import SpeechPart, build_parts, english_words, resolve_mode, transliterate_english
from app.text_norm import has_cyrillic, has_latin, normalize as normalize_ru, split_by_script

TMP_DIR = Path(tempfile.gettempdir()) / "mlvh_tts"
CHUNK_CHARS = 400

log = logging.getLogger(__name__)

__all__ = [
    "TTSError",
    "TTSService",
    "SpeechPipeline",
    "SentenceBuffer",
    "clean_for_speech",
    "split_into_chunks",
    "voice_label",
]


def audio_seconds(path: Path) -> float:
    """Длительность wav в секундах (0, если формат не wav или файл битый)."""
    if path.suffix.lower() != ".wav":
        return 0.0

    try:
        import wave

        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / (handle.getframerate() or 1)
    except Exception:
        pass

    # torchaudio пишет float32-WAV, который модуль wave не понимает.
    try:
        import soundfile

        return float(soundfile.info(str(path)).duration)
    except Exception:
        return 0.0


def expected_seconds(text: str, chars_per_second: float = 14.0) -> float:
    """Сколько примерно должна длиться речь: ~14 символов в секунду.

    Английский голос говорит быстрее (~18 символов в секунду), поэтому для
    английских частей норму передаём отдельно — иначе проверка обрыва
    приняла бы нормальный кусок за укороченный и дробила бы его зря.
    """
    return max(0.5, len(text) / max(1.0, chars_per_second))


def clean_for_speech(text: str, normalize_numbers: bool = False) -> str:
    """Убираем из ответа то, что нельзя прочитать: код, ссылки, разметку.

    normalize_numbers включаем для локальных движков: они плохо читают цифры
    и сокращения, поэтому переводим их в слова заранее.
    """
    text = re.sub(r"```.*?```", " Код пропущен. ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|.*\|\s*$", "", text, flags=re.MULTILINE)  # таблицы
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)    # маркеры списков
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\*\*|__|\*|_|~~)", "", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]", "", text)  # эмодзи
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    cleaned = _expand_acronyms(text.strip())
    if normalize_numbers:
        cleaned = normalize_numbers_ru(cleaned)
    return cleaned


def normalize_numbers_ru(text: str) -> str:
    """Числа и сокращения — только в русских кусках.

    Английский фрагмент уходит английскому голосу, и «100» в нём должно
    остаться «100», а не превратиться в «сто». Пробелы по краям кусков
    сохраняем: `normalize` их срезает, и слова слиплись бы в «Курспайтон».
    """
    if not has_latin(text):
        return normalize_ru(text)

    out: list[str] = []
    for lang, chunk in split_by_script(text):
        if lang == "en" or not chunk.strip():
            out.append(chunk)
            continue
        lead = chunk[: len(chunk) - len(chunk.lstrip())]
        tail = chunk[len(chunk.rstrip()):]
        out.append(lead + normalize_ru(chunk.strip()) + tail)
    return "".join(out)


# Латиница без гласных («MLVH», «HTTP») не синтезируется: сервис возвращает тишину.
# Поэтому такие сокращения читаем по буквам.
_LETTER_NAMES = {
    "A": "эй", "B": "би", "C": "си", "D": "ди", "E": "и", "F": "эф", "G": "джи",
    "H": "эйч", "I": "ай", "J": "джей", "K": "кей", "L": "эл", "M": "эм", "N": "эн",
    "O": "оу", "P": "пи", "Q": "кью", "R": "ар", "S": "эс", "T": "ти", "U": "ю",
    "V": "ви", "W": "дабл-ю", "X": "экс", "Y": "уай", "Z": "зед",
}
_VOWELS = re.compile(r"[aeiouyаеёиоуыэюя]", re.IGNORECASE)


def _expand_acronyms(text: str) -> str:
    """«MLVH» → «эм эл ви эйч», чтобы синтез не подавился словом без гласных."""

    def replace(match: re.Match) -> str:
        word = match.group(0)
        if _VOWELS.search(word):
            return word
        spelled = [_LETTER_NAMES.get(ch.upper(), ch) for ch in word]
        return " ".join(spelled)

    return re.sub(r"\b[A-Za-z]{2,}\b", replace, text)



def split_into_chunks(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Режем текст на фразы, чтобы речь начиналась, не дожидаясь всего ответа."""
    sentences = re.split(r"(?<=[.!?…:;])\s+", text)
    chunks: list[str] = []
    buffer = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            chunks.extend(_hard_split(sentence, max_chars))
            continue
        if len(buffer) + len(sentence) + 1 <= max_chars:
            buffer = f"{buffer} {sentence}".strip()
        else:
            if buffer:
                chunks.append(buffer)
            buffer = sentence
    if buffer:
        chunks.append(buffer)
    return chunks


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Очень длинное предложение режем по запятым, потом по словам."""
    parts: list[str] = []
    current = ""
    for piece in re.split(r"(?<=,)\s+", text):
        if len(current) + len(piece) + 1 <= max_chars:
            current = f"{current} {piece}".strip()
            continue
        if current:
            parts.append(current)
        current = ""
        for word in piece.split():
            if len(current) + len(word) + 1 <= max_chars:
                current = f"{current} {word}".strip()
            else:
                if current:
                    parts.append(current)
                current = word
    if current:
        parts.append(current)
    return parts


def voice_label(voice: dict) -> str:
    """Подпись для выпадающего списка: «Dmitry · ru-RU · муж.»."""
    short_name = voice.get("ShortName") or voice.get("short_name") or ""
    locale = voice.get("Locale") or voice.get("locale") or ""
    gender = voice.get("Gender") or voice.get("gender") or ""
    name = short_name.split("-")[-1].replace("Neural", "")
    gender_ru = {"Male": "муж.", "Female": "жен."}.get(gender, gender)
    return f"{name} · {locale} · {gender_ru}"


class SentenceBuffer:
    """Копит текст ответа и отдаёт готовые фразы по мере поступления.

    Первую фразу режем коротко — чтобы голос зазвучал как можно раньше,
    дальше укрупняем куски: у Edge TTS фиксированные ~1 с на запрос,
    поэтому длинные куски выгоднее по общему времени.
    """

    MIN_FIRST = 20
    MIN_NEXT = 160
    MAX_CHUNK = 420

    def __init__(self, min_first: int | None = None, min_next: int | None = None,
                 max_chunk: int | None = None) -> None:
        self._min_first = min_first or self.MIN_FIRST
        self._min_next = min_next or self.MIN_NEXT
        self._max_chunk = max_chunk or self.MAX_CHUNK
        self._buffer = ""
        self._in_code = False
        self._first = True

    def push(self, text: str) -> list[str]:
        """Добавляем кусок ответа, возвращаем фразы, готовые к синтезу."""
        if not text:
            return []
        self._buffer += text
        return self._extract(final=False)

    def flush(self) -> list[str]:
        """Отдаём остаток буфера (ответ закончился)."""
        return self._extract(final=True)

    def _min_length(self) -> int:
        return self._min_first if self._first else self._min_next

    def _extract(self, final: bool) -> list[str]:
        ready: list[str] = []
        while True:
            self._drop_code()
            chunk = self._cut(final=final)
            if chunk is None:
                break
            ready.append(chunk)
        return ready

    def _drop_code(self) -> None:
        """Код не читаем: выкидываем всё между ``` … ```."""
        if self._in_code:
            end = self._buffer.find("```")
            if end == -1:
                self._buffer = ""
                return
            self._buffer = self._buffer[end + 3:]
            self._in_code = False
            return
        start = self._buffer.find("```")
        if start != -1:
            self._buffer = self._buffer[:start] + " " + self._buffer[start + 3:]
            self._in_code = True

    def _cut(self, final: bool) -> str | None:
        if self._in_code or not self._buffer.strip():
            return None

        limit = self._min_length()
        text = self._buffer
        cut_at = -1

        for index, char in enumerate(text):
            if char in ".!?…" and index + 1 >= limit:
                cut_at = index + 1
                break
            if char == "\n" and index + 1 >= limit:
                cut_at = index + 1
                break

        if cut_at == -1 and len(text) >= self._max_chunk:
            space = text.rfind(" ", limit, self._max_chunk)
            cut_at = space if space > 0 else self._max_chunk

        if cut_at == -1:
            if not final:
                return None
            if len(text) < 8:
                self._buffer = ""
                return None
            cut_at = len(text)

        chunk = text[:cut_at].strip()
        self._buffer = text[cut_at:]
        self._first = False
        return chunk or None



class TTSService:
    """Синтез + проигрывание. Один экземпляр на приложение."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._playing = threading.Event()
        self._pygame_ready = False
        self._play_lock = threading.Lock()

    @property
    def is_playing(self) -> bool:
        """Идёт ли звук прямо сейчас.

        Именно звук, а не «поток озвучки жив»: синтез следующей фразы и
        подвисший поток не должны глушить микрофон.
        """
        return self._playing.is_set()

    # --- голоса ---------------------------------------------------------
    @staticmethod
    def list_voices(locale_prefix: str = "") -> list[dict]:
        """Список голосов Edge. Без интернета бросит TTSError."""
        try:
            voices = asyncio.run(edge_tts.list_voices())
        except Exception as exc:
            raise TTSError(f"Не удалось получить список голосов: {exc}") from exc
        if locale_prefix:
            voices = [v for v in voices if str(v.get("Locale", "")).startswith(locale_prefix)]
        return sorted(voices, key=lambda v: str(v.get("ShortName", "")))

    # --- синтез ---------------------------------------------------------
    def _synthesize_sync(self, text: str, path: Path, settings: TTSSettings) -> Path:
        """Синтез выбранным движком. Возвращает фактический путь к файлу."""
        engine = get_engine(settings.engine)
        info = engine.probe()
        if not info.available:
            raise TTSError(info.reason or info.hint or f"Движок «{info.title}» недоступен")

        # Локальную модель греет отдельный поток: ждём её, а не грузим параллельно.
        if info.id == "chatterbox" and not engine.wait_ready(180):
            raise TTSError("Модель Chatterbox ещё загружается — попробуйте через полминуты.")

        # Облачный Edge глючит и требует повторов, локальный — нет.
        attempts = 3 if info.id == "edge" else 1
        last_error: object = "движок вернул пустой файл"

        for attempt in range(attempts):
            try:
                result = engine.synthesize(text, path, settings)
                if result.exists() and result.stat().st_size > 0:
                    return result
                last_error = "движок вернул пустой файл"
            except Exception as exc:
                last_error = exc
                for candidate in (path, path.with_suffix(".wav")):
                    try:
                        candidate.unlink(missing_ok=True)
                    except OSError:
                        pass
            if attempt < attempts - 1:
                # Пауза короткая: отказ сервиса обычно разовый, ждать смысла нет.
                time.sleep(0.15 * (attempt + 1))
        raise TTSError(f"Ошибка синтеза речи: {last_error}")

    # --- воспроизведение ------------------------------------------------
    def play(self, path: Path, volume: float = 0.9) -> None:
        """Играем mp3 и ждём окончания (или команды «Стоп»)."""
        try:
            import pygame
        except ImportError as exc:
            raise TTSError("Не установлен pygame-ce: pip install pygame-ce") from exc

        with self._play_lock:
            try:
                if not self._pygame_ready:
                    pygame.mixer.init()
                    self._pygame_ready = True
                pygame.mixer.music.load(str(path))
                pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
                self._playing.set()
                pygame.mixer.music.play()
            except Exception as exc:
                self._playing.clear()
                raise TTSError(f"Не удалось воспроизвести звук: {exc}") from exc

            try:
                while pygame.mixer.music.get_busy():
                    if self._stop.is_set():
                        pygame.mixer.music.stop()
                        break
                    time.sleep(0.05)
            finally:
                self._playing.clear()
                # Без unload() файл остаётся занятым и его нельзя удалить.
                try:
                    pygame.mixer.music.unload()
                except Exception:
                    pass

    # --- главный сценарий ----------------------------------------------
    def speak(self, text: str, settings: TTSSettings) -> None:
        """Озвучиваем текст: синтез следующего куска идёт параллельно с речью."""
        self._stop.clear()
        local = settings.engine in ("chatterbox", "silero")
        cleaned = clean_for_speech(text, normalize_numbers=local)[: max(200, settings.max_speak_chars)]
        if not cleaned:
            return

        chunks = split_into_chunks(cleaned)
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        created: list[Path] = []
        futures: dict[int, Future] = {}

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                def submit(index: int) -> None:
                    if index < len(chunks) and index not in futures:
                        path = TMP_DIR / f"chunk_{index:03d}_{uuid4().hex[:8]}.mp3"
                        futures[index] = pool.submit(
                            self._speak_chunk, chunks[index], path, settings
                        )

                submit(0)
                submit(1)
                played = 0
                for index in range(len(chunks)):
                    if self._stop.is_set():
                        break
                    try:
                        paths = futures.pop(index).result()
                    except TTSError as exc:
                        # Один кусок не озвучился — не срываем всю фразу.
                        log.warning("Кусок %s не озвучен: %s", index, exc)
                        submit(index + 2)
                        continue
                    created.extend(paths)
                    submit(index + 2)
                    if self._stop.is_set():
                        break
                    for path in paths:
                        if self._stop.is_set():
                            break
                        self.play(path, settings.playback_volume)
                        played += 1

                if played == 0 and not self._stop.is_set():
                    # Edge не дал звука — читаем голосом Windows, он всегда под рукой.
                    log.warning("Edge TTS не вернул звук, переключаюсь на SAPI")
                    self._speak_sapi(cleaned)
        finally:
            for path in created:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass  # файл ещё держит плеер — удалим при следующем запуске

    def _speak_chunk(self, text: str, path: Path, settings: TTSSettings) -> list[Path]:
        """Кусок целиком: русские части — выбранным движком, английские — по режиму."""
        parts = build_parts(text, settings)
        result: list[Path] = []
        for index, part in enumerate(parts):
            part_path = path
            if len(parts) > 1:
                part_path = path.with_name(f"{path.stem}_{index}{path.suffix}")
            if part.lang == "en":
                paths = synthesize_english(self, part.text, part_path, settings)
                if not paths:
                    fallback = transliterate_english(part.text)
                    if fallback.strip():
                        paths = [self._synthesize_sync(fallback, part_path, settings)]
            else:
                paths = [self._synthesize_sync(part.text, part_path, settings)]
            result.extend(paths)
        return result

    def _speak_sapi(self, text: str) -> None:
        """Запасной голос Windows (SAPI): офлайн, звучит проще, но всегда доступен."""
        if sys.platform != "win32":
            raise TTSError("Запасной голос доступен только в Windows.")

        TMP_DIR.mkdir(parents=True, exist_ok=True)
        text_file = TMP_DIR / f"sapi_{uuid4().hex[:8]}.txt"
        text_file.write_text(text, encoding="utf-8")

        script = (
            "Add-Type -AssemblyName System.Speech; "
            f"$t = Get-Content -LiteralPath '{text_file}' -Raw -Encoding UTF8; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$ru = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'ru-*' } "
            "| Select-Object -First 1; "
            "if ($ru) { $s.SelectVoice($ru.VoiceInfo.Name) }; "
            "$s.Volume = 95; $s.Speak($t)"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                check=True,
                capture_output=True,
                timeout=180,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise TTSError(f"Запасной голос Windows не сработал: {exc}") from exc
        finally:
            text_file.unlink(missing_ok=True)

    def speak_sapi_to_file(self, text: str, path: Path) -> Path:
        """Тот же SAPI, но в wav-файл: чтобы вставить в очередь без потери порядка."""
        if sys.platform != "win32":
            raise TTSError("Запасной голос доступен только в Windows.")

        TMP_DIR.mkdir(parents=True, exist_ok=True)
        text_file = TMP_DIR / f"sapi_{uuid4().hex[:8]}.txt"
        text_file.write_text(text, encoding="utf-8")
        out = path.with_suffix(".wav")

        script = (
            "Add-Type -AssemblyName System.Speech; "
            f"$t = Get-Content -LiteralPath '{text_file}' -Raw -Encoding UTF8; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$ru = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'ru-*' } "
            "| Select-Object -First 1; "
            "if ($ru) { $s.SelectVoice($ru.VoiceInfo.Name) }; "
            "$s.Volume = 95; "
            f"$s.SetOutputToWaveFile('{out}'); "
            "$s.Speak($t); $s.Dispose()"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                check=True,
                capture_output=True,
                timeout=180,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise TTSError(f"Запасной голос Windows не сработал: {exc}") from exc
        finally:
            text_file.unlink(missing_ok=True)

        if not out.exists() or out.stat().st_size == 0:
            raise TTSError("Запасной голос Windows не создал звук.")
        return out

    def stop(self) -> None:
        """Прерываем речь: следующий play() сразу завершится."""
        self._stop.set()
        self._playing.clear()
        try:
            import pygame

            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            pass

    def reset(self) -> None:
        """Снимаем флаг остановки перед новой озвучкой."""
        self._stop.clear()
        self._playing.clear()

    @staticmethod
    def clear_cache() -> None:
        """Чистим временные mp3 (например, при выходе)."""
        shutil.rmtree(TMP_DIR, ignore_errors=True)


def synthesize_english(
    tts: TTSService, text: str, path: Path, settings: TTSSettings
) -> list[Path]:
    """Английский фрагмент: локальная v3_en, иначе Edge английским голосом.

    Пустой список означает «не получилось» — вызывающий решает, читать ли
    фрагмент транслитом.
    """
    if resolve_mode(settings) == "silero_en":
        local = _synthesize_silero_en(text, path, settings)
        if local:
            return local

    english = replace(settings, engine="edge", voice=settings.english_voice)
    try:
        return [tts._synthesize_sync(text, path, english)]
    except TTSError as exc:
        log.warning("Английский фрагмент не озвучен: %s", exc)
        return []


def _synthesize_silero_en(text: str, path: Path, settings: TTSSettings) -> list[Path]:
    """Локальная английская модель; пустой список — режим недоступен."""
    try:
        from app.engines.silero_en_engine import SileroEnEngine
    except Exception:
        return []
    engine = SileroEnEngine.instance()
    if not engine.probe().available:
        return []
    try:
        return [engine.synthesize(text, path.with_suffix(".wav"), settings)]
    except TTSError as exc:
        log.warning("Silero v3_en не справился: %s", exc)
        return []


class SpeechPipeline:
    """Озвучка по мере поступления текста.

    Текст подкидывается кусками прямо из стрима модели: первая готовая фраза
    уходит в синтез сразу, пока ответ ещё дописывается. Синтез и проигрывание
    живут в разных потоках, поэтому между фразами нет паузы на ~1 с.
    """

    def __init__(
        self,
        tts: TTSService,
        settings: TTSSettings,
        on_done=None,
        on_error=None,
        on_note=None,
        model_settings=None,
    ) -> None:
        self._tts = tts
        self._settings = settings
        self._on_done = on_done
        self._on_error = on_error
        self._on_note = on_note

        self._buffer = SentenceBuffer(*self._chunk_limits())
        self._texts: queue.Queue[str | None] = queue.Queue()
        self._ready: queue.Queue[tuple[int, list[Path] | None]] = queue.Queue()
        self._stop = threading.Event()
        self._spoken: list[str] = []
        self._played = 0
        self._chars = 0
        self._closed = False
        self._threads: list[threading.Thread] = []
        self._wait_terms = False       # первый кусок ждёт словарь
        self._learner = self._make_learner(model_settings)

    def _make_learner(self, model_settings):
        """Обучение терминов — только явным флагом и только если есть провайдер."""
        if model_settings is None or not getattr(self._settings, "learn_terms", False):
            return None
        try:
            # Импорт внутри: без обучения тянуть openai и клиент модели незачем.
            from app.speech_learn import TermLearner

            return TermLearner(model_settings, self._settings, on_note=self._on_note)
        except Exception as exc:          # сеть и ключи проверяются позже, при запросе
            log.warning("Обучение терминов не включилось: %s", exc)
            return None

    # --- управление -----------------------------------------------------
    def start(self) -> None:
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        self._tts.reset()
        for target in (self._synthesize_loop, self._play_loop):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)

    def push(self, text: str) -> None:
        """Добавляем очередной кусок ответа модели."""
        if self._closed or not text:
            return
        budget = max(200, self._settings.max_speak_chars)
        room = budget - self._chars
        if room <= 0:
            return
        if len(text) > room:
            text = text[:room]
        self._chars += len(text)
        self._spoken.append(text)
        for chunk in self._buffer.push(text):
            self._request_terms(chunk)
            self._texts.put(chunk)

    def _request_terms(self, chunk: str) -> None:
        """Просим словарь о незнакомых словах — сразу, параллельно с синтезом.

        Запрос идёт в фоне: ждать его будем только перед первым куском и не
        дольше `learn_timeout`. Если слова нет и не выучилось — прочитаем
        по правилам, слово останется на следующий раз.
        """
        if self._learner is None or not self._learner.enabled:
            return
        try:
            words = english_words(chunk)
        except Exception as exc:
            log.warning("Не удалось собрать незнакомые термины: %s", exc)
            return
        if words:
            self._wait_terms = True
            self._learner.request(words)

    def finish(self) -> None:
        """Ответ закончился: отдаём остаток и ждём, пока всё проиграется."""
        if self._closed:
            return
        self._closed = True
        for chunk in self._buffer.flush():
            self._texts.put(chunk)
        self._texts.put(None)

    def stop(self) -> None:
        """Прерываем озвучку и выбрасываем очередь."""
        self._closed = True
        self._stop.set()
        self._drain(self._texts)
        self._drain(self._ready)
        # Будим оба потока, иначе они останутся висеть на get().
        self._texts.put(None)
        self._ready.put((-1, None))
        self._tts.stop()

    def wait(self, timeout: float = 5.0) -> None:
        for thread in self._threads:
            thread.join(timeout)

    @property
    def is_running(self) -> bool:
        return any(thread.is_alive() for thread in self._threads)

    @staticmethod
    def _drain(target: queue.Queue) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                return

    # --- потоки ---------------------------------------------------------
    def _local_engine(self) -> bool:
        """Локальные движки требуют нормализации чисел и сокращений."""
        return self._settings.engine in ("chatterbox", "silero")

    def _chunk_limits(self) -> tuple[int, int, int]:
        """Размеры кусков под движок.

        У Edge цена запроса ~1 с, у Chatterbox — ~4 с на вызов, поэтому ему
        выгодны крупные куски. Первый кусок везде короткий: важен быстрый старт.
        """
        if self._settings.engine == "chatterbox":
            # Модель обрывает русский текст непредсказуемо: чем короче кусок,
            # тем реже обрыв и меньше дроблений (каждое дробление — лишний синтез).
            return 45, 75, 95
        if self._settings.engine == "silero":
            return 40, 220, 400
        return SentenceBuffer.MIN_FIRST, SentenceBuffer.MIN_NEXT, SentenceBuffer.MAX_CHUNK

    def _note(self, message: str) -> None:
        log.warning(message)
        if self._on_note is not None:
            self._on_note(message)

    def _synthesize_loop(self) -> None:
        index = 0
        first = True
        while not self._stop.is_set():
            text = self._texts.get()
            if text is None or self._stop.is_set():
                break

            if first:
                # Словарь спрашиваем один раз за ответ: ждём не дольше лимита,
                # дальше читаем тем, что успело появиться.
                self._await_terms()
                first = False

            path = TMP_DIR / f"stream_{index:03d}_{uuid4().hex[:8]}.mp3"
            cleaned = clean_for_speech(text, normalize_numbers=self._local_engine())
            if not cleaned:
                self._ready.put((index, None))
                index += 1
                continue

            ready: list[Path] = []
            parts = build_parts(cleaned, self._settings)
            for part_index, part in enumerate(parts):
                if self._stop.is_set():
                    break
                part_path = path
                if len(parts) > 1:
                    part_path = path.with_name(f"{path.stem}_{part_index}{path.suffix}")
                ready.extend(self._synthesize_part(part, part_path))
            self._ready.put((index, ready))
            index += 1
        self._ready.put((-1, None))

    def _await_terms(self) -> None:
        """Ждём обучения не дольше `learn_timeout`, потом читаем как есть."""
        if self._learner is None or not self._wait_terms:
            return
        timeout = max(1.0, float(getattr(self._settings, "learn_timeout", 30.0)))
        started = time.monotonic()
        self._learner.wait(timeout)
        spent = time.monotonic() - started
        if spent >= timeout:
            log.warning("Словарь не ответил за %.0f с — читаю по правилам", timeout)
            self._note("Термины не выучены вовремя — прочитаю по правилам")

    def _synthesize_part(self, part: SpeechPart, path: Path) -> list[Path]:
        """Одна часть: английскую отдаём английскому движку, русскую — обычному."""
        if part.lang != "en":
            return self._synthesize_chunk(part.text, path)

        paths = synthesize_english(self._tts, part.text, path, self._settings)
        if paths:
            return paths

        # Английский движок не дал звука — переписываем кириллицей, чтобы
        # слова не пропали совсем (так было до 0.8).
        fallback = transliterate_english(part.text)
        if fallback.strip():
            self._note("Английский фрагмент читаю русскими буквами")
            return self._synthesize_chunk(fallback, path)
        return []

    def _synthesize_chunk(
        self,
        text: str,
        path: Path,
        depth: int = 0,
        settings: TTSSettings | None = None,
    ) -> list[Path]:
        """Синтез куска; при обрыве дробим текст и синтезируем половинки."""
        active = settings or self._settings
        # Английский голос говорит быстрее: своя норма для проверки обрыва.
        want = expected_seconds(text, 18.0 if not has_cyrillic(text) else 14.0)
        result: Path | None = None

        # Повторяем только при ошибке синтеза; обрыв лечится дроблением текста.
        for attempt in (1, 2):
            try:
                result = self._tts._synthesize_sync(text, path, active)
                break
            except TTSError as exc:
                log.warning("Кусок не озвучен (попытка %s): %s", attempt, exc)
                result = None

        if result is not None:
            seconds = audio_seconds(result)
            ratio = seconds / want if want else 1.0
            if not seconds or ratio >= 0.75:
                return [result]

            log.warning(
                "Кусок оборвался: %.1f с вместо ~%.1f с (отношение %.2f)",
                seconds, want, ratio,
            )
            try:
                result.unlink(missing_ok=True)
            except OSError:
                pass

        # Дробим текст пополам: длинные куски Chatterbox обрывает на полуслове.
        halves = self._split_text(text)
        if depth < 2 and halves:
            log.info("Дроблю кусок на %s частей и синтезирую заново", len(halves))
            paths: list[Path] = []
            for index, part in enumerate(halves):
                part_path = path.with_name(f"{path.stem}_p{index}{path.suffix}")
                paths.extend(self._synthesize_chunk(part, part_path, depth + 1, active))
            if paths:
                return paths

        # Совсем не вышло — читаем голосом Windows, чтобы не терять середину.
        self._note("Кусок не озвучился движком — читаю запасным голосом Windows")
        try:
            return [self._tts.speak_sapi_to_file(text, path)]
        except TTSError as exc:
            self._note(f"Кусок пропущен: {exc}")
            return []

    @staticmethod
    def _split_text(text: str) -> list[str]:
        """Делим текст пополам по границе предложения или слова."""
        if len(text) < 40:
            return []

        middle = len(text) // 2
        for separator in (". ", "! ", "? ", ", "):
            left = text.rfind(separator, 0, middle + 20)
            if left > 20:
                return [text[:left + 1].strip(), text[left + 1:].strip()]

        space = text.rfind(" ", 0, middle + 20)
        if space > 20:
            return [text[:space].strip(), text[space:].strip()]
        return [text[:middle].strip(), text[middle:].strip()]

    def _play_loop(self) -> None:
        try:
            while not self._stop.is_set():
                index, paths = self._ready.get()
                if index == -1:
                    break
                if not paths:
                    continue
                for path in paths:
                    if self._stop.is_set():
                        break
                    try:
                        self._tts.play(path, self._settings.playback_volume)
                        self._played += 1
                    except TTSError as exc:
                        log.warning("Не удалось проиграть кусок: %s", exc)
                    finally:
                        try:
                            path.unlink(missing_ok=True)
                        except OSError:
                            pass
        finally:
            # Выученные термины пишем сразу: файл маленький, а при выходе
            # отложенная запись потерялась бы.
            try:
                from app.speech_terms import dictionary

                dictionary().flush(force=True)
            except Exception:
                pass
            if self._played == 0 and not self._stop.is_set() and self._spoken:
                # Edge не дал звука ни разу — читаем голосом Windows.
                log.warning("Edge TTS молчит, переключаюсь на SAPI")
                try:
                    self._tts._speak_sapi("".join(self._spoken))
                except TTSError as exc:
                    if self._on_error is not None:
                        self._on_error(str(exc))
                    return
            if self._on_done is not None and not self._stop.is_set():
                self._on_done()


