"""Нормализация русского текста перед локальным синтезом.

Chatterbox и Silero хуже Edge справляются с цифрами и сокращениями, поэтому
для них текст готовим заранее: числа — прописью, «т.д.» — «так далее»,
«5 км» — «пять километров».
"""

from __future__ import annotations

import re

UNITS_M = ("ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь",
           "восемь", "девять")
UNITS_F = ("ноль", "одна", "две", "три", "четыре", "пять", "шесть", "семь",
           "восемь", "девять")
TEENS = ("десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
         "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать")
TENS = ("", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят",
        "семьдесят", "восемьдесят", "девяносто")
HUNDREDS = ("", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот",
            "семьсот", "восемьсот", "девятьсот")

# Сокращения и символы: Chatterbox читает их по буквам или пропускает.
REPLACEMENTS = {
    "т.д.": "так далее",
    "т.е.": "то есть",
    "т.к.": "так как",
    "т.п.": "тому подобное",
    "др.": "другое",
    "напр.": "например",
    "см.": "смотри",
    "рис.": "рисунок",
    "%": " процентов",
    "№": " номер ",
    "₽": " рублей",
    "$": " долларов",
    "€": " евро",
    "км": "километров",
    "кг": "килограммов",
    "мг": "миллиграммов",
    "гб": "гигабайт",
    "мб": "мегабайт",
    "кт": "килотонн",
}

UNITS_WORDS = {
    "км": ("километр", "километра", "километров"),
    "кг": ("килограмм", "килограмма", "килограммов"),
    "м": ("метр", "метра", "метров"),
    "см": ("сантиметр", "сантиметра", "сантиметров"),
    "г": ("грамм", "грамма", "граммов"),
    "руб": ("рубль", "рубля", "рублей"),
    "год": ("год", "года", "лет"),
}


def _plural_index(number: int) -> int:
    """0 — один, 1 — два, 2 — пять (для согласования существительных)."""
    if number % 100 in (11, 12, 13, 14):
        return 2
    last = number % 10
    if last == 1:
        return 0
    if last in (2, 3, 4):
        return 1
    return 2


def _triple_to_words(value: int, feminine: bool = False) -> list[str]:
    """Число от 0 до 999 прописью."""
    words: list[str] = []
    hundreds, rest = divmod(value, 100)
    if hundreds:
        words.append(HUNDREDS[hundreds])

    if rest >= 20:
        tens, unit = divmod(rest, 10)
        words.append(TENS[tens])
        if unit:
            words.append((UNITS_F if feminine else UNITS_M)[unit])
    elif rest >= 10:
        words.append(TEENS[rest - 10])
    elif rest > 0:
        words.append((UNITS_F if feminine else UNITS_M)[rest])
    return words


def number_to_words(number: int) -> str:
    """Целое число прописью (до 999 999 999)."""
    if number == 0:
        return UNITS_M[0]
    if number < 0:
        return "минус " + number_to_words(-number)

    parts: list[str] = []
    millions, rest = divmod(number, 1_000_000)
    thousands, rest = divmod(rest, 1_000)

    if millions:
        parts += _triple_to_words(millions)
        parts.append(("миллион", "миллиона", "миллионов")[_plural_index(millions)])
    if thousands:
        parts += _triple_to_words(thousands, feminine=True)
        parts.append(("тысяча", "тысячи", "тысяч")[_plural_index(thousands)])
    if rest:
        parts += _triple_to_words(rest)
    return " ".join(parts)


def _replace_measure(match: re.Match) -> str:
    """«5 км» → «пять километров» с согласованием."""
    number = int(match.group(1))
    unit = match.group(2).lower()
    forms = UNITS_WORDS.get(unit)
    if forms is None:
        return f"{number_to_words(number)} {unit}"
    return f"{number_to_words(number)} {forms[_plural_index(number)]}"


ORD_UNITS_PREP = ("первом", "втором", "третьем", "четвёртом", "пятом", "шестом",
                  "седьмом", "восьмом", "девятом")
ORD_TEENS_PREP = ("десятом", "одиннадцатом", "двенадцатом", "тринадцатом",
                  "четырнадцатом", "пятнадцатом", "шестнадцатом", "семнадцатом",
                  "восемнадцатом", "девятнадцатом")
ORD_TENS_PREP = ("", "", "двадцатом", "тридцатом", "сороковом", "пятидесятом",
                 "шестидесятом", "семидесятом", "восьмидесятом", "девяностом")
ORD_HUNDREDS_PREP = ("", "сотом", "двухсотом", "трёхсотом", "четырёхсотом",
                     "пятисотом", "шестисотом", "семисотом", "восьмисотом",
                     "девятьсотом")


def _ordinal_prep(value: int) -> str:
    """Порядковое числительное в предложном падеже: 24 → «двадцать четвёртом»."""
    if value <= 0:
        return ""

    hundreds, rest = divmod(value, 100)
    if rest == 0:
        return ORD_HUNDREDS_PREP[hundreds]

    prefix = HUNDREDS[hundreds] if hundreds else ""
    if rest < 10:
        ordinal = ORD_UNITS_PREP[rest - 1]
    elif rest < 20:
        ordinal = ORD_TEENS_PREP[rest - 10]
    else:
        tens, unit = divmod(rest, 10)
        if unit:
            ordinal = f"{TENS[tens]} {ORD_UNITS_PREP[unit - 1]}"
        else:
            ordinal = ORD_TENS_PREP[tens]

    return f"{prefix} {ordinal}".strip()


def year_to_words(year: int) -> str:
    """«2024» → «две тысячи двадцать четвёртом» (для «в … году»)."""
    if not 1000 <= year <= 2999:
        return number_to_words(year)

    thousands, rest = divmod(year, 1000)
    if thousands == 1:
        head = "тысяча"
    elif thousands == 2:
        head = "две тысячи"
    else:
        head = f"{number_to_words(thousands)} тысячи"

    if rest == 0:
        if thousands == 1:
            return "тысячном"
        if thousands == 2:
            return "двухтысячном"
        return f"{number_to_words(thousands)}-тысячном"
    return f"{head} {_ordinal_prep(rest)}"


def _replace_thousands(match: re.Match) -> str:
    """«21 тысяча», «3 миллиона» — согласуем числительное с существительным."""
    number = int(match.group(1))
    noun = match.group(2).lower()
    forms = {
        "тысяча": ("тысяча", "тысячи", "тысяч"),
        "миллион": ("миллион", "миллиона", "миллионов"),
        "миллиард": ("миллиард", "миллиарда", "миллиардов"),
    }.get(noun, (noun, noun, noun))
    feminine = noun == "тысяча"
    count = " ".join(_triple_to_words(number, feminine=feminine))
    return f"{count} {forms[_plural_index(number)]}"


def normalize(text: str) -> str:
    """Готовим текст к локальному синтезу."""
    if not text.strip():
        return text

    result = text

    # Годы: «в 2024 году» → «в две тысячи двадцать четвёртом году».
    result = re.sub(
        r"\b(\d{4})\s*(году|год|года|годом)\b",
        lambda m: f"{year_to_words(int(m.group(1)))} {m.group(2)}",
        result,
    )

    # «21 тысяча», «3 миллиона» — до общей замены цифр.
    result = re.sub(
        r"\b(\d+)\s*(тысяча|тысячи|тысяч|миллион|миллиона|миллионов|миллиард|миллиарда|миллиардов)\b",
        _replace_thousands,
        result,
        flags=re.IGNORECASE,
    )

    # «5 км», «12 кг» — до общей замены цифр.
    result = re.sub(
        r"\b(\d+)\s*(км|кг|см|м|г|руб|год)\b",
        _replace_measure,
        result,
        flags=re.IGNORECASE,
    )

    # Проценты и валюты: «15%» → «пятнадцать процентов».
    def percent(match: re.Match) -> str:
        number = int(match.group(1))
        return f"{number_to_words(number)} {('процент', 'процента', 'процентов')[_plural_index(number)]}"

    result = re.sub(r"\b(\d+)\s*%", percent, result)

    # Остальные числа — прописью.
    result = re.sub(r"\b\d+\b", lambda m: number_to_words(int(m.group(0))), result)

    # Сокращения и символы.
    for source, target in REPLACEMENTS.items():
        if source.isalpha() and len(source) <= 3:
            result = re.sub(rf"\b{re.escape(source)}\b", target, result, flags=re.IGNORECASE)
        else:
            result = result.replace(source, target)

    # Лишние знаки, которые ломают просодию.
    result = re.sub(r"[()\[\]{}\"'«»]", " ", result)
    result = re.sub(r"\s{2,}", " ", result)
    return result.strip()
