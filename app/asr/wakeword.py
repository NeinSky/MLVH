"""Спецслово: чистое сравнение слов — без Qt и звука.

Вынесено отдельно от `wake.py`, чтобы логику можно было проверять в CI,
где PySide6 не устанавливается.

Vosk small-ru не знает имени «Элви» и слышит его как «элвис», «элвин», «элли»
или «элби». Поэтому сравниваем не строки целиком, а слова с допуском на
опечатку распознавания: Левенштейн ≤ 1 для коротких слов.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_tokens(text: str) -> list[str]:
    """Слова в нижнем регистре, без пунктуации и «ё» — для сравнения."""
    cleaned = text.lower().replace("ё", "е")
    return [token for token in _WORD_RE.findall(cleaned)]


def levenshtein(left: str, right: str) -> int:
    """Расстояние редактирования: сколько замен нужно, чтобы получить одно из другого."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _allowed_distance(word: str) -> int:
    """Короткие слова сравниваем строже: иначе «элви» ловит «эльвира»."""
    return 1 if len(word) <= 5 else 2


def match_wake_word(text: str, word: str) -> int | None:
    """Сколько слов в начале фразы занимает спецслово (None — не найдено).

    Допускаем обращение через «эй» и разбиение на два слова («эл ви»).
    """
    tokens = normalize_tokens(text)
    goal = "".join(normalize_tokens(word))
    if not tokens or not goal:
        return None

    allowed = _allowed_distance(goal)

    for index, token in enumerate(tokens[:3]):
        if levenshtein(token, goal) <= allowed:
            return index + 1

    for index in range(min(2, len(tokens) - 1)):
        pair = tokens[index] + tokens[index + 1]
        if levenshtein(pair, goal) <= allowed:
            return index + 2

    return None


def strip_wake_word(text: str, word: str) -> str | None:
    """Убираем спецслово из начала. None — спецслова не было.

    Регистр остальных слов и вопросительный знак сохраняем: текст уходит в чат.
    """
    size = match_wake_word(text, word)
    if size is None:
        return None

    tokens = _WORD_RE.findall(text)
    rest = " ".join(tokens[size:]).strip()
    if rest and text.rstrip().endswith("?") and not rest.endswith("?"):
        rest += "?"
    return rest
