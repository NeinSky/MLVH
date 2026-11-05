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


# --- английские термины: разбиение по письменности и транслитерация --------
#
# Локальные модели (Silero v5_ru, Chatterbox) знают только кириллицу: латиницу
# они молча вырезают, а если кириллицы в куске нет вовсе — бросают ошибку, и
# кусок уходит в запасной голос Windows. Поэтому английские слова либо читает
# английский движок, либо мы заранее переписываем их русскими буквами.
#
# Порядок поиска произношения: словарь (`app/speech_terms.py`) → правила ниже.
# Правила — последний шанс: на замерах они дают 8 точных совпадений из 20
# IT-терминов, поэтому словарь важнее, а правила только страхуют.

# Как русские читают латинские буквы: «API» → «эй пи ай», «GPT» → «джи пи ти».
LETTER_NAMES = {
    "A": "эй", "B": "би", "C": "си", "D": "ди", "E": "и", "F": "эф", "G": "джи",
    "H": "эйч", "I": "ай", "J": "джей", "K": "кей", "L": "эл", "M": "эм", "N": "эн",
    "O": "оу", "P": "пи", "Q": "кью", "R": "ар", "S": "эс", "T": "ти", "U": "ю",
    "V": "ви", "W": "дабл-ю", "X": "экс", "Y": "уай", "Z": "зед",
}

_LATIN_CHAR = re.compile(r"[A-Za-z]")
_CYRILLIC_CHAR = re.compile(r"[А-Яа-яЁё]")

# Токен целиком: буква, дальше буквы/цифры и внутренние разделители.
# «Node.js», «postgresql-14», «pull_request» остаются одним токеном.
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[._\-+#][A-Za-z0-9]+)*")

_CODE_EXTENSIONS = (
    ".py", ".pyc", ".json", ".md", ".txt", ".exe", ".dll", ".ps1", ".bat", ".cmd",
    ".yml", ".yaml", ".toml", ".cfg", ".ini", ".log", ".csv", ".xml", ".sh",
    ".js", ".ts", ".html", ".css", ".zip", ".rar", ".part", ".tmp",
)


def has_latin(text: str) -> bool:
    return bool(_LATIN_CHAR.search(text))


def has_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_CHAR.search(text))


def split_by_script(text: str) -> list[tuple[str, str]]:
    """Режем текст на куски одного алфавита: [("ru", "Курс "), ("en", "Python")].

    Цифры, пробелы и знаки препинания прилипают к соседнему языку — иначе
    фраза развалится на десяток кусков. Если текст начинается со знаков,
    они уходят в первый языковой кусок.
    """
    if not text:
        return []

    parts: list[list[str]] = []
    current: list[str] = []
    lang = ""

    for char in text:
        if _CYRILLIC_CHAR.match(char):
            char_lang = "ru"
        elif _LATIN_CHAR.match(char):
            char_lang = "en"
        else:
            char_lang = ""

        if char_lang and lang and char_lang != lang:
            parts.append([lang, "".join(current)])
            current = []
        if char_lang:
            lang = char_lang
        current.append(char)

    if current:
        parts.append([lang, "".join(current)])

    # Ведущие пробелы и знаки (языка ещё нет) — к первому языковому куску.
    if len(parts) > 1 and parts[0][0] == "":
        parts[1][1] = parts[0][1] + parts[1][1]
        parts.pop(0)
    if not parts:
        return []

    merged: list[tuple[str, str]] = []
    for part_lang, part_text in parts:
        if merged and merged[-1][0] == part_lang:
            merged[-1] = (part_lang, merged[-1][1] + part_text)
        else:
            merged.append((part_lang or "ru", part_text))
    return merged


def latin_tokens(text: str) -> list[str]:
    """Латинские токены в порядке появления («Node.js» — один токен)."""
    return [match.group(0) for match in TOKEN.finditer(text)]


def is_probable_code(token: str) -> bool:
    """Похоже на имя файла, путь, хеш или версию — такое вслух не читаем."""
    if not token:
        return True
    low = token.lower()
    if any(char in token for char in "/\\@:"):
        return True
    if low.endswith(_CODE_EXTENSIONS):
        return True
    if "_" in token:                                   # snake_case
        return True
    if re.fullmatch(r"[0-9a-f]{6,}", low):             # хеш коммита
        return True
    if re.fullmatch(r"[a-z]{1,4}\d+(?:\.\d+)*", low):  # v1.2.3, sha256
        return True
    digits = sum(char.isdigit() for char in token)
    if digits and digits / len(token) > 0.4:           # 0x1F, mp3-4k
        return True
    return False


# Диграфы: порядок важен — длинные проверяются раньше коротких.
_DIGRAPHS = (
    ("tion", "шн"), ("sion", "жн"), ("ture", "чер"), ("ough", "о"), ("augh", "о"),
    ("eigh", "эй"),
    ("ch", "ч"), ("sh", "ш"), ("zh", "ж"), ("th", "т"), ("ph", "ф"), ("ck", "к"),
    ("dj", "дж"), ("qu", "кв"), ("gh", "г"), ("wh", "у"), ("oo", "у"), ("ee", "и"),
    ("ea", "и"), ("oa", "оу"), ("ai", "эй"), ("ay", "эй"), ("ey", "эй"), ("oi", "ой"),
    ("oy", "ой"), ("ou", "ау"), ("ow", "оу"), ("au", "о"), ("ew", "ью"),
    ("ei", "ей"), ("ie", "и"), ("ue", "ью"), ("ar", "ар"), ("er", "ер"),
    ("ir", "ир"), ("or", "ор"), ("ur", "ур"),
)

# Окончания: применяем только в конце слова, иначе «ing» в «single» сломается.
_SUFFIXES = (
    ("ing", "инг"), ("ed", "ед"), ("ly", "ли"), ("ity", "ити"), ("ment", "мент"),
    ("ness", "нес"), ("able", "ейбл"), ("ible", "ибл"), ("ous", "ес"), ("ist", "ист"),
    ("ism", "изм"), ("ive", "ив"), ("ate", "ейт"), ("ic", "ик"),
)

_SINGLE = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "у", "x": "кс", "y": "и", "z": "з",
}

_LONG_VOWELS = {"a": "эй", "e": "и", "i": "ай", "o": "оу", "u": "ю", "y": "ай"}
_MAGIC_E = re.compile(r"([aeiouy])([bcdfgklmnprstvz])e$")


def _spell(word: str) -> str:
    """«API» → «эй пи ай»: аббревиатуры читаем по буквам."""
    return " ".join(LETTER_NAMES.get(char.upper(), char) for char in word if char.isalpha())


_VOWELS = "aeiouy"


def _walk(low: str) -> str:
    """Посимвольный проход с диграфами и окончаниями."""
    out: list[str] = []
    index = 0
    while index < len(low):
        for suffix, target in _SUFFIXES:
            if low.startswith(suffix, index) and index + len(suffix) == len(low):
                out.append(target)
                index += len(suffix)
                break
        else:
            for source, target in _DIGRAPHS:
                if low.startswith(source, index):
                    out.append(target)
                    index += len(source)
                    break
            else:
                char = low[index]
                # «w» в начале слова перед гласной читается как «в»: Webpack.
                if char == "w" and index == 0 and index + 1 < len(low) and low[index + 1] in _VOWELS:
                    out.append("в")
                # «s» между гласными звонкая: Tensor → тензор.
                elif (
                    char == "s"
                    and 0 < index < len(low) - 1
                    and low[index - 1] in _VOWELS
                    and low[index + 1] in _VOWELS
                ):
                    out.append("з")
                else:
                    out.append(_SINGLE.get(char, char))
                index += 1
    return "".join(out)


def _core(word: str) -> str:
    """Одно слово без CamelCase: «vite» → «вайт», «ansible» → «ансибл»."""
    low = word.lower()
    magic = _MAGIC_E.search(low)
    if magic and len(low) >= 4:
        # Немая «e» в конце делает гласную долгой: file → файл, type → тайп.
        low = low[:magic.start(1)] + _LONG_VOWELS[magic.group(1)] + magic.group(2)
    elif re.fullmatch(r"[^aeiou]+y", low):
        # Согласные + «y» без других гласных: Py → пай, my → май.
        return _walk(low[:-1]) + "ай"
    elif low.endswith("e") and len(low) > 3:
        low = low[:-1]
    return _walk(low)


def _split_camel(word: str) -> list[str]:
    """«TensorFlow» → ["Tensor", "Flow"], «API» остаётся одним куском."""
    return re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+", word)


def transliterate_word(word: str) -> str:
    """Английское слово русскими буквами: «Kubernetes» → «кубернетес».

    Это запасной путь, когда слова нет в словаре: качество ограниченное,
    поэтому незнакомые термины лучше выучить (см. `app/speech_learn.py`).
    """
    if not word:
        return ""
    if not has_latin(word):
        return word

    # Разделители внутри токена («Node.js») читаем как отдельные слова.
    if any(char in word for char in "._-+#"):
        pieces = [piece for piece in re.split(r"[._\-+#]+", word) if piece]
        return " ".join(transliterate_word(piece) for piece in pieces)

    if word.isupper() and len(word) >= 2:
        return _spell(word)

    parts = _split_camel(word)
    if len(parts) > 1:
        out: list[str] = []
        spelled = False
        for part in parts:
            if part.isupper() and len(part) >= 2:
                out.append(_spell(part))
                spelled = True
            elif len(part) == 1:
                # Одиночная буква в CamelCase — это тоже буква: gRPC.
                out.append(LETTER_NAMES.get(part.upper(), part))
                spelled = True
            else:
                out.append(_core(part))
        # Списанные по буквам части отделяем пробелом, иначе «MongoDB»
        # склеится в «монгоди би».
        return (" " if spelled else "").join(out)

    if word.isupper():
        return _spell(word)
    return _core(word)


def transliterate_text(text: str) -> str:
    """Транслитерируем все латинские токены, остальное не трогаем."""

    def replace(match: re.Match) -> str:
        token = match.group(0)
        if is_probable_code(token):
            return " "
        return transliterate_word(token)

    return TOKEN.sub(replace, text)


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
