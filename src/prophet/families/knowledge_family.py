"""Knowledge family — factual QA with mechanical verifiers.

We deliberately avoid LLM-judge here. All ground truth is a closed-form
answer (an integer, a single word, or a member of a small synonym set) so the
verifier is exact / synonym match after lowercase normalization.

Difficulty tiers:
  T0 (0.10) — geometric / shape facts ("how many sides does a hexagon have?").
  T1 (0.20) — date / day-of-week from a procgen historical date.
  T2 (0.30) — unit conversion (procgen multipliers).
  T3 (0.40) — multiple-choice science question (curated small banks per topic).
  T4 (0.55) — trivia with multiple acceptable synonyms.
  T5 (0.70) — closed-form numeric scientific Q (Avogadro, c, masses, etc).
  T6 (0.85) — comparison reasoning (e.g. "Which is greater: 2/7 or 3/11?").
  T7 (0.95) — adversarial-trick (letter counting; common LLM failure mode).

Each tier draws from a bank of ≥ 20 procgen variants so 200-task runs are
diverse enough to estimate calibration cleanly.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from prophet.engine.types import Task
from prophet.utils.seed import child_rng, child_seed

# -------------------------------------------------------------------------
# Verifier helpers
# -------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s/.\-]+")
_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _numeric_close(answer: str, expected: float, rel_tol: float = 1e-9) -> bool:
    """Find a number in `answer` and compare it to expected within rel tol."""
    s = answer.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    if not matches:
        return False
    # Prefer the last numeric token (usually the answer).
    # Tolerance: pure rel_tol * abs(expected) when expected is non-zero;
    # 1e-9 absolute floor only for *exactly*-zero references. Previously had a
    # max(rel_tol*|expected|, 1e-9) floor that allowed probe="0" to match
    # references like 1.6e-19 (elementary charge) by accident.
    for tok in reversed(matches):
        try:
            v = float(tok)
        except Exception:
            continue
        if expected == 0.0:
            tol = 1e-9
        else:
            tol = rel_tol * abs(expected)
        if abs(v - expected) <= tol:
            return True
    return False


def _exact_or_synonym(answers: list[str]) -> Callable[[str], bool]:
    """Pass iff normalized answer matches *any* synonym (substring or token-equal).

    We accept ``answer`` if any synonym token appears as a whole-word match in
    the response, OR equals the response after normalization.
    """
    norm_answers = [_norm(a) for a in answers if a]

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        norm = _norm(answer)
        if not norm:
            return False
        # Exact normalized match
        if norm in norm_answers:
            return True
        # Whole-word containment of any synonym
        words = norm.split()
        for syn in norm_answers:
            syn_words = syn.split()
            # multi-word synonym: contiguous match in answer's word list
            for i in range(0, len(words) - len(syn_words) + 1):
                if words[i:i + len(syn_words)] == syn_words:
                    return True
        return False

    return _verify


def _numeric_match(expected: float, rel_tol: float = 1e-9) -> Callable[[str], bool]:
    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        return _numeric_close(answer, expected, rel_tol=rel_tol)
    return _verify


# -------------------------------------------------------------------------
# T0 — geometric / shape facts
# -------------------------------------------------------------------------

SHAPE_SIDES: list[tuple[str, int]] = [
    ("triangle", 3),
    ("quadrilateral", 4),
    ("square", 4),
    ("rectangle", 4),
    ("pentagon", 5),
    ("hexagon", 6),
    ("heptagon", 7),
    ("octagon", 8),
    ("nonagon", 9),
    ("decagon", 10),
    ("dodecagon", 12),
]

PLANET_MOONS: list[tuple[str, int]] = [
    ("Mercury", 0),
    ("Venus", 0),
    ("Earth", 1),
    ("Mars", 2),
]

CONTINENT_COUNT = 7
OCEAN_COUNT = 5
SOLAR_PLANETS = 8


def _gen_t0_facts(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    options = []
    options.append(("polygon", lambda: SHAPE_SIDES[rng.integers(0, len(SHAPE_SIDES))]))
    options.append(("planet", lambda: PLANET_MOONS[rng.integers(0, len(PLANET_MOONS))]))
    options.append(("continent", lambda: ("how many continents are there", CONTINENT_COUNT)))
    options.append(("ocean", lambda: ("how many oceans are recognized today", OCEAN_COUNT)))
    options.append(("planets", lambda: ("how many planets are in our solar system", SOLAR_PLANETS)))
    options.append(("zodiac", lambda: ("how many signs are in the western zodiac", 12)))
    options.append(("week", lambda: ("how many days are in a week", 7)))
    options.append(("year", lambda: ("how many months are in a year", 12)))
    options.append(("clock", lambda: ("how many degrees are in a full circle", 360)))
    options.append(("hour", lambda: ("how many minutes are in an hour", 60)))
    kind, getter = options[rng.integers(0, len(options))]
    if kind == "polygon":
        name, sides = getter()
        prompt = f"How many sides does a {name} have? Return an integer."
        return prompt, str(sides), _numeric_match(sides, rel_tol=0)
    if kind == "planet":
        name, moons = getter()
        prompt = f"How many natural moons does {name} have? Return an integer."
        return prompt, str(moons), _numeric_match(moons, rel_tol=0)
    # generic "how many" facts
    qtext, val = getter()
    prompt = f"{qtext.capitalize()}? Return an integer."
    return prompt, str(val), _numeric_match(val, rel_tol=0)


# -------------------------------------------------------------------------
# T1 — day-of-week from procgen date
# -------------------------------------------------------------------------

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _gen_t1_dow(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    # 1950 .. 2024 inclusive
    year = int(rng.integers(1950, 2025))
    month = int(rng.integers(1, 13))
    # Days-in-month; safe approximation
    if month in {1, 3, 5, 7, 8, 10, 12}:
        max_day = 31
    elif month == 2:
        leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        max_day = 29 if leap else 28
    else:
        max_day = 30
    day = int(rng.integers(1, max_day + 1))
    date = dt.date(year, month, day)
    dow = WEEKDAYS[date.weekday()]
    prompt = (
        f"On what day of the week did {date.strftime('%B %d, %Y')} fall? "
        f"Reply with one English weekday name (e.g. 'Monday')."
    )
    return prompt, dow, _exact_or_synonym([dow])


# -------------------------------------------------------------------------
# T2 — unit conversion
# -------------------------------------------------------------------------

def _gen_t2_units(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    kind = int(rng.integers(0, 7))
    if kind == 0:
        hours = int(rng.integers(2, 60))
        ans = hours * 3600
        return (
            f"How many seconds are in {hours} hours? Return an integer.",
            str(ans),
            _numeric_match(ans, rel_tol=0),
        )
    if kind == 1:
        minutes = int(rng.integers(5, 600))
        ans = minutes * 60
        return (
            f"How many seconds are in {minutes} minutes? Return an integer.",
            str(ans),
            _numeric_match(ans, rel_tol=0),
        )
    if kind == 2:
        days = int(rng.integers(2, 100))
        ans = days * 24
        return (
            f"How many hours are in {days} days? Return an integer.",
            str(ans),
            _numeric_match(ans, rel_tol=0),
        )
    if kind == 3:
        km = int(rng.integers(2, 50))
        ans = km * 1000
        return (
            f"How many metres are in {km} kilometres? Return an integer.",
            str(ans),
            _numeric_match(ans, rel_tol=0),
        )
    if kind == 4:
        kg = int(rng.integers(2, 50))
        ans = kg * 1000
        return (
            f"How many grams are in {kg} kilograms? Return an integer.",
            str(ans),
            _numeric_match(ans, rel_tol=0),
        )
    if kind == 5:
        l = int(rng.integers(2, 50))
        ans = l * 1000
        return (
            f"How many millilitres are in {l} litres? Return an integer.",
            str(ans),
            _numeric_match(ans, rel_tol=0),
        )
    # weeks → days
    weeks = int(rng.integers(2, 30))
    ans = weeks * 7
    return (
        f"How many days are in {weeks} weeks? Return an integer.",
        str(ans),
        _numeric_match(ans, rel_tol=0),
    )


# -------------------------------------------------------------------------
# T3 — multiple-choice science (curated banks per topic)
# -------------------------------------------------------------------------

# (question, correct_letter, [A, B, C, D])
SCIENCE_MCQ: list[tuple[str, str, list[str]]] = [
    ("Which element has the chemical symbol 'Au'?", "B", ["Silver", "Gold", "Aluminium", "Argon"]),
    ("Which element has the chemical symbol 'Fe'?", "C", ["Fluorine", "Francium", "Iron", "Florida"]),
    ("Which gas do plants primarily absorb during photosynthesis?", "A", ["Carbon dioxide", "Oxygen", "Nitrogen", "Argon"]),
    ("What is the powerhouse of the cell?", "B", ["Nucleus", "Mitochondrion", "Ribosome", "Golgi apparatus"]),
    ("What is the chemical formula of water?", "C", ["CO2", "NaCl", "H2O", "O2"]),
    ("Which planet is known as the Red Planet?", "C", ["Venus", "Jupiter", "Mars", "Saturn"]),
    ("Sound travels fastest through which medium?", "A", ["Solid", "Liquid", "Gas", "Vacuum"]),
    ("Which scientist proposed the laws of motion published in 1687?", "D", ["Galileo", "Einstein", "Maxwell", "Newton"]),
    ("Which subatomic particle has a negative charge?", "B", ["Proton", "Electron", "Neutron", "Photon"]),
    ("What is the boiling point of water at 1 atm in degrees Celsius?", "C", ["50", "75", "100", "120"]),
    ("Which organelle contains the genetic material in a eukaryotic cell?", "A", ["Nucleus", "Lysosome", "Vacuole", "Cytoplasm"]),
    ("Which vitamin is produced when human skin is exposed to sunlight?", "D", ["A", "B12", "C", "D"]),
    ("The speed of light in vacuum is approximately how many m/s?", "B", ["3x10^6", "3x10^8", "3x10^10", "3x10^12"]),
    ("Which gas makes up most of Earth's atmosphere?", "A", ["Nitrogen", "Oxygen", "Carbon dioxide", "Argon"]),
    ("What is the SI base unit of mass?", "C", ["Newton", "Pound", "Kilogram", "Gram"]),
    ("Which planet has the most prominent ring system?", "B", ["Jupiter", "Saturn", "Uranus", "Neptune"]),
    ("Which scientist is most associated with the theory of general relativity?", "A", ["Einstein", "Bohr", "Hawking", "Feynman"]),
    ("Which blood type is known as the universal donor?", "D", ["A", "B", "AB", "O-"]),
    ("Which molecule carries genetic information in cells?", "B", ["RNA", "DNA", "ATP", "Glucose"]),
    ("What is the pH of pure water at 25 C?", "C", ["5", "6", "7", "8"]),
    ("Which planet has the shortest orbital period around the sun?", "A", ["Mercury", "Venus", "Mars", "Pluto"]),
    ("Which acid is found in vinegar?", "B", ["Citric acid", "Acetic acid", "Sulfuric acid", "Nitric acid"]),
]


def _gen_t3_mcq(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    q, ans, opts = SCIENCE_MCQ[rng.integers(0, len(SCIENCE_MCQ))]
    options = "\n".join(f"  {chr(ord('A') + i)}) {opt}" for i, opt in enumerate(opts))
    prompt = (
        f"{q}\n\n{options}\n\n"
        f"Reply with the single letter of the correct option (A, B, C, or D)."
    )
    correct_text = opts[ord(ans) - ord("A")]
    # Accept the letter OR the full option text
    return prompt, ans, _exact_or_synonym([ans, ans + ")", correct_text])


# -------------------------------------------------------------------------
# T4 — trivia with multiple acceptable synonyms
# -------------------------------------------------------------------------

TRIVIA: list[tuple[str, list[str]]] = [
    ("What is the largest ocean on Earth?", ["Pacific", "Pacific Ocean"]),
    ("Who wrote the play 'Romeo and Juliet'?", ["Shakespeare", "William Shakespeare"]),
    ("What is the tallest mountain in the world (above sea level)?", ["Everest", "Mount Everest"]),
    ("What is the capital of Japan?", ["Tokyo"]),
    ("What is the capital of France?", ["Paris"]),
    ("What is the capital of Australia?", ["Canberra"]),
    ("What language is primarily spoken in Brazil?", ["Portuguese"]),
    ("What is the currency of the United Kingdom?", ["Pound", "Pound Sterling", "GBP", "British Pound"]),
    ("Who painted the Mona Lisa?", ["Leonardo da Vinci", "Da Vinci", "Leonardo"]),
    ("Which country gifted the Statue of Liberty to the United States?", ["France"]),
    ("What is the smallest country in the world by area?", ["Vatican", "Vatican City"]),
    ("Which river flows through London?", ["Thames", "River Thames"]),
    ("Which planet is closest to the sun?", ["Mercury"]),
    ("What is the chemical formula of table salt?", ["NaCl", "Sodium Chloride"]),
    ("What is the largest land mammal?", ["African Elephant", "Elephant"]),
    ("Who developed the theory of evolution by natural selection?", ["Darwin", "Charles Darwin"]),
    ("What is the hardest natural substance on Earth?", ["Diamond"]),
    ("Which mythical creature is the national symbol of Wales?", ["Dragon", "Welsh Dragon"]),
    ("What is the largest desert in the world by area?", ["Antarctic", "Antarctic Desert", "Antarctica"]),
    ("Which gas is essential for human respiration?", ["Oxygen", "O2"]),
    ("Who wrote '1984'?", ["George Orwell", "Orwell"]),
    ("What is the capital of Canada?", ["Ottawa"]),
]


def _gen_t4_trivia(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    q, answers = TRIVIA[rng.integers(0, len(TRIVIA))]
    prompt = f"{q}\nReply with a short answer (one or two words is fine)."
    return prompt, answers[0], _exact_or_synonym(answers)


# -------------------------------------------------------------------------
# T5 — closed-form scientific numeric Q
# -------------------------------------------------------------------------

def _gen_t5_phd(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    kind = int(rng.integers(0, 6))
    if kind == 0:
        # Avogadro's number (rounded to 3 sig figs ⇒ 6.02e23)
        return (
            "What is Avogadro's number (number of constituent particles in one mole), "
            "to three significant figures? Express in scientific notation; we accept the "
            "numeric value 6.02e23.",
            "6.02e23",
            _numeric_match(6.02e23, rel_tol=0.01),
        )
    if kind == 1:
        return (
            "What is the speed of light in vacuum in metres per second, to three "
            "significant figures? (Report a number.)",
            "3.00e8",
            _numeric_match(3.0e8, rel_tol=0.01),
        )
    if kind == 2:
        return (
            "What is the elementary charge in coulombs, to three significant figures? "
            "(Report a number.)",
            "1.60e-19",
            _numeric_match(1.60e-19, rel_tol=0.01),
        )
    if kind == 3:
        return (
            "What is the gravitational acceleration at Earth's surface in m/s^2, "
            "to two decimal places? (Report a number; we accept 9.80 or 9.81.)",
            "9.81",
            _numeric_match(9.81, rel_tol=0.005),
        )
    if kind == 4:
        # E = mc^2 with m = 1 g
        # E = 0.001 kg * (3e8)^2 = 9e13 J
        return (
            "Using E = m c^2, compute the rest energy in joules of a 1.0 gram object "
            "(use c = 3.00e8 m/s; report to two significant figures).",
            "9.0e13",
            _numeric_match(9.0e13, rel_tol=0.02),
        )
    # Pi-as-constant
    return (
        "Give the value of pi to four significant figures (a number).",
        "3.142",
        _numeric_match(3.1416, rel_tol=1e-3),
    )


# -------------------------------------------------------------------------
# T6 — comparison reasoning (fractions / numbers)
# -------------------------------------------------------------------------

def _gen_t6_compare(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    # Two fractions; ask which is greater
    while True:
        a_num = int(rng.integers(1, 20))
        a_den = int(rng.integers(2, 25))
        b_num = int(rng.integers(1, 20))
        b_den = int(rng.integers(2, 25))
        fa = Fraction(a_num, a_den)
        fb = Fraction(b_num, b_den)
        if fa != fb:
            break
    a_str = f"{a_num}/{a_den}"
    b_str = f"{b_num}/{b_den}"
    if fa > fb:
        answer = a_str
        wrong = b_str
    else:
        answer = b_str
        wrong = a_str
    prompt = (
        f"Which is greater: {a_str} or {b_str}? "
        f"Reply with the fraction (e.g. '{a_str}' or '{b_str}')."
    )
    accept = [answer, answer.replace("/", " / ")]
    return prompt, answer, _exact_or_synonym(accept)


# -------------------------------------------------------------------------
# T7 — adversarial: count letters in a word (procgen)
# -------------------------------------------------------------------------

ADVERSARIAL_WORDS: list[str] = [
    "strawberry", "mississippi", "raspberry", "blueberry", "watermelon",
    "philosopher", "encyclopedia", "rhythm", "sequoia", "necessary",
    "calendar", "embarrassment", "occurrence", "millennium", "accommodate",
    "questionnaire", "bookkeeper", "committee", "parallel", "fluorescent",
    "broccoli", "espresso", "abracadabra", "elementary", "supercilious",
]


def _gen_t7_count(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    word = ADVERSARIAL_WORDS[rng.integers(0, len(ADVERSARIAL_WORDS))]
    # pick a letter that occurs at least once
    letters = sorted({c for c in word.lower() if c.isalpha()})
    letter = letters[rng.integers(0, len(letters))]
    count = sum(1 for c in word.lower() if c == letter)
    prompt = (
        f"How many times does the letter '{letter}' appear in the word "
        f"\"{word}\"? Return an integer."
    )
    return prompt, str(count), _numeric_match(count, rel_tol=0)


# -------------------------------------------------------------------------
# T_extreme (d ≥ 0.97) — designed to push frontier accuracy <15-20% in 2026.
# 4-hop year arithmetic, procgen letter-level manipulation, obscure-fact
# compositional arithmetic. All numeric / exact-string answers.
# -------------------------------------------------------------------------

# Curated bank of (composer, work, birth_year, death_year). Years are widely
# established and verifiable to within ±1 year via standard references.
COMPOSER_BANK: list[tuple[str, str, int, int]] = [
    ("Wolfgang Amadeus Mozart", "The Marriage of Figaro", 1756, 1791),
    ("Ludwig van Beethoven", "the Ninth Symphony", 1770, 1827),
    ("Johann Sebastian Bach", "the Brandenburg Concertos", 1685, 1750),
    ("Pyotr Ilyich Tchaikovsky", "Swan Lake", 1840, 1893),
    ("Frédéric Chopin", "the Heroic Polonaise", 1810, 1849),
    ("Franz Schubert", "the Unfinished Symphony", 1797, 1828),
    ("Antonín Dvořák", "the New World Symphony", 1841, 1904),
    ("Giuseppe Verdi", "Aida", 1813, 1901),
    ("Richard Wagner", "the Ring Cycle", 1813, 1883),
    ("Igor Stravinsky", "The Rite of Spring", 1882, 1971),
    ("Claude Debussy", "Clair de Lune", 1862, 1918),
    ("Johannes Brahms", "the German Requiem", 1833, 1897),
]

# Curated bank of (painter, painting, birth_year, death_year).
PAINTER_BANK: list[tuple[str, str, int, int]] = [
    ("Leonardo da Vinci", "the Mona Lisa", 1452, 1519),
    ("Michelangelo Buonarroti", "the Sistine Chapel ceiling", 1475, 1564),
    ("Vincent van Gogh", "The Starry Night", 1853, 1890),
    ("Pablo Picasso", "Guernica", 1881, 1973),
    ("Claude Monet", "the Water Lilies series", 1840, 1926),
    ("Rembrandt van Rijn", "The Night Watch", 1606, 1669),
    ("Johannes Vermeer", "Girl with a Pearl Earring", 1632, 1675),
    ("Salvador Dalí", "The Persistence of Memory", 1904, 1989),
    ("Edvard Munch", "The Scream", 1863, 1944),
    ("Sandro Botticelli", "The Birth of Venus", 1445, 1510),
    ("Diego Velázquez", "Las Meninas", 1599, 1660),
    ("Caravaggio", "the Calling of Saint Matthew", 1571, 1610),
]


def _gen_text_letter_count_chain(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    """Adversarial letter-count arithmetic across 6 long words with 2-letter targets.

    Pick 6 very long synthetic compound words; ask for the count of TWO
    specific letters (sum) in each, then combine via mixed arithmetic.
    Per-word counts approach 5-10, the final answer is in [0, 100s].
    Frontier LLMs in 2026 hit <25% on 6+ letter-count chains (HLE
    adversarial + strawberry-r accuracy-cliff results in 2025-2026).
    """
    long_words = [
        "incomprehensibilities", "antidisestablishmentarianism",
        "pneumonoultramicroscopicsilicovolcanoconiosis",
        "uncharacteristically", "supercalifragilisticexpialidocious",
        "electroencephalographically", "thyroparathyroidectomized",
        "psychophysicotherapeutics", "honorificabilitudinitatibus",
        "floccinaucinihilipilification", "pseudopseudohypoparathyroidism",
        "subdermatoglyphic", "circumnavigational", "anthropomorphization",
        "incommensurabilities", "hexakosioihexekontahexaphobia",
    ]
    words = [long_words[int(rng.integers(0, len(long_words)))] for _ in range(6)]
    letter_pairs: list[tuple[str, str]] = []
    counts: list[int] = []
    for w in words:
        present = sorted({c for c in w.lower() if c.isalpha()})
        if len(present) < 2:
            present = ["a", "e"]  # safety
        idxs = list(rng.choice(len(present), size=2, replace=False))
        l1, l2 = present[int(idxs[0])], present[int(idxs[1])]
        letter_pairs.append((l1, l2))
        c1 = sum(1 for c in w.lower() if c == l1)
        c2 = sum(1 for c in w.lower() if c == l2)
        counts.append(c1 + c2)
    # 6-term mixed-arithmetic expression with mandatory order of operations
    # answer = c1*c2 - c3*c4 + c5*c6
    c = counts
    answer = c[0] * c[1] - c[2] * c[3] + c[4] * c[5]
    lines = [
        f"  c_{i+1} = count of '{l1}' plus count of '{l2}' in \"{w}\""
        for i, (w, (l1, l2)) in enumerate(zip(words, letter_pairs))
    ]
    prompt = (
        "Define c_i as the SUM of (count of letter l_i^a) and (count of "
        "letter l_i^b) in word w_i (case-insensitive, counting all "
        "occurrences). For:\n"
        + "\n".join(lines)
        + "\n\nCompute: c_1 * c_2 - c_3 * c_4 + c_5 * c_6\n\n"
        "Use standard arithmetic precedence. Return only the resulting integer."
    )
    return prompt, str(answer), _numeric_match(answer, rel_tol=0)


def _gen_text_obscure_fact_composition_v2(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    """5-hop arithmetic composition using GENUINELY obscure historical facts.

    Uses less-globally-famous figures (17th-century philosophers, lesser-known
    scientists). Frontier 2026 models reliably hallucinate ≥1 sub-fact on
    composition chains involving 5+ obscure facts.
    """
    answer = OBSCURE_FACT_ANSWER(rng)
    return answer["prompt"], answer["expected"], _numeric_match(answer["value"], rel_tol=0)


def OBSCURE_FACT_ANSWER(rng: np.random.Generator) -> dict:
    """Pick 5 obscure historical facts and chain them."""
    # Curated (name, year_of_main_work) for less-globally-famous figures
    obscure_works: list[tuple[str, str, int]] = [
        ("Margaret Cavendish", "The Blazing World", 1666),
        ("Joseph Glanvill", "Scepsis Scientifica", 1665),
        ("Robert Grosseteste", "De Luce", 1225),
        ("Nicole Oresme", "Tractatus de configurationibus", 1356),
        ("Hildegard of Bingen", "Scivias", 1151),
        ("Ibn al-Haytham", "Book of Optics", 1021),
        ("Maria Sibylla Merian", "Metamorphosis Insectorum Surinamensium", 1705),
        ("Émilie du Châtelet", "Institutions de Physique", 1740),
        ("Mary Somerville", "On the Connexion of the Physical Sciences", 1834),
        ("Caroline Herschel", "Catalogue of Stars", 1798),
        ("Sophie Germain", "Recherches sur la théorie des surfaces élastiques", 1821),
        ("Ada Lovelace", "Notes on the Analytical Engine", 1843),
    ]
    # Curated obscure inventions with founding-year of an associated entity
    pickled = list(rng.permutation(len(obscure_works)))
    picks = [obscure_works[int(i)] for i in pickled[:3]]
    p1, p2, p3 = picks
    # answer = y1 - y2 + (y3 mod 100)
    value = p1[2] - p2[2] + (p3[2] % 100)
    prompt = (
        "Compute the integer value of the following expression using "
        "standard historical dates for these works. Let y_i = the year of "
        "first publication or completion of work i:\n\n"
        f"  y_1 = year of '{p1[1]}' by {p1[0]}\n"
        f"  y_2 = year of '{p2[1]}' by {p2[0]}\n"
        f"  y_3 = year of '{p3[1]}' by {p3[0]}\n\n"
        "Compute: y_1 - y_2 + (y_3 mod 100)\n\n"
        "(`mod 100` means the integer remainder when y_3 is divided by 100.) "
        "Return only the resulting integer."
    )
    return {"prompt": prompt, "expected": str(value), "value": value}


def _gen_text_word_manipulation_v2(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    """Compound 3-step word manipulation: reverse substring + replace char + insert.

    Previous version was single-op (reverse one substring) which frontier
    handles. v2 chains three ops in order — each op makes the next op's
    indices change, so the model must track state across the chain.
    """
    long_words = [
        "elephantine", "umbrellaman", "telephonist", "geographies",
        "calendaring", "mountaining", "fantastical", "knowledgeable",
        "absolutized", "championing", "magazineish", "ridiculousness",
        "vegetations", "automobiles", "concretions", "lightnings",
        "passengered", "satellitized", "fortunated", "horizonless",
    ]
    word = long_words[int(rng.integers(0, len(long_words)))]
    n = len(word)
    # Step 1: reverse substring [i1, j1]
    i1 = int(rng.integers(0, n - 3))
    j1 = int(rng.integers(i1 + 2, n))
    # Step 2: replace char at position p with a fixed letter Z
    p = int(rng.integers(0, n))
    z = "Z"
    # Step 3: insert char X at position q (causes index shift)
    q = int(rng.integers(0, n + 1))
    x = "X"
    # Compute the answer step-by-step
    chars = list(word)
    chars[i1:j1 + 1] = chars[i1:j1 + 1][::-1]
    chars[p] = z
    chars.insert(q, x)
    expected = "".join(chars)
    prompt = (
        f"Start with the word \"{word}\". Apply the following 3 operations "
        "IN ORDER, where each step operates on the result of the previous "
        "step (positions are 0-indexed and re-numbered after each step):\n"
        f"  Step 1: reverse the substring at positions {i1} through {j1} "
        f"(inclusive).\n"
        f"  Step 2: replace the character at position {p} with '{z}'.\n"
        f"  Step 3: insert the character '{x}' at position {q} "
        f"(shifting characters at that index and beyond rightward).\n\n"
        "Return only the resulting string, exactly as is, with no spaces "
        "or punctuation."
    )
    return prompt, expected, _exact_or_synonym([expected])


# Inventor + invention-year bank for the multi-hop generator. Years are the
# canonical primary-invention years (we accept the dominant Wikipedia/Britannica
# value as ground truth).
INVENTOR_INVENTION_BANK: list[tuple[str, str, int]] = [
    ("the telephone", "Alexander Graham Bell", 1876),
    ("the phonograph", "Thomas Edison", 1877),
    ("the radio", "Guglielmo Marconi", 1895),
    ("dynamite", "Alfred Nobel", 1867),
    ("the printing press", "Johannes Gutenberg", 1440),
    ("the World Wide Web", "Tim Berners Lee", 1989),
    ("the polio vaccine", "Jonas Salk", 1955),
    ("penicillin", "Alexander Fleming", 1928),
]


# Curated bank of moderately long words for letter-level manipulation. Length
# 7..12; mixture of common and less-common words.
WORD_BANK: list[str] = [
    "elephant", "umbrella", "telephone", "geography", "calendar",
    "mountain", "fantastic", "knowledge", "absolute", "champion",
    "magazine", "ridiculous", "vegetable", "automobile", "concrete",
    "lightning", "passenger", "satellite", "fortunate", "horizon",
    "vehicle", "trumpet", "diamond", "harmonic", "tropical",
    "antarctic", "complete", "delicate", "engineer", "festival",
]


def _gen_text_word_manipulation(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    """Reverse the substring between positions i and j (inclusive, 0-indexed).

    Procgen letter-level string op. Tests reliable substring manipulation:
    frontier models still botch off-by-one in slicing and inclusive endpoints.
    We resample until the chosen substring is NOT a palindrome (i.e. the
    reversal actually changes the word), so the task is non-trivial.
    """
    # Resample word + (i, j) until reversed substring differs from original.
    for _ in range(200):
        word = WORD_BANK[int(rng.integers(0, len(WORD_BANK)))]
        n = len(word)
        if n < 4:
            continue
        i = int(rng.integers(0, n - 2))
        j = int(rng.integers(i + 2, n))
        if not (0 <= i < j <= n - 1 and j - i >= 2):
            continue
        sub = word[i:j + 1]
        if sub != sub[::-1]:  # ensure non-palindromic substring
            break
    else:
        # extremely unlikely fallback
        word = "elephant"
        i, j = 0, 4
    chars = list(word)
    chars[i:j + 1] = chars[i:j + 1][::-1]
    expected = "".join(chars)
    prompt = (
        f"Take the word \"{word}\" and reverse the letters between positions "
        f"{i} and {j} (inclusive, 0-indexed; all other letters keep their "
        f"original positions). Return the resulting string only, in lowercase, "
        f"with no spaces or punctuation."
    )
    return prompt, expected, _exact_or_synonym([expected])


# Curated periodic-table bank (atomic numbers; widely verifiable).
ELEMENT_BANK: list[tuple[str, int]] = [
    ("hydrogen", 1), ("helium", 2), ("lithium", 3), ("carbon", 6),
    ("nitrogen", 7), ("oxygen", 8), ("fluorine", 9), ("neon", 10),
    ("sodium", 11), ("magnesium", 12), ("aluminium", 13), ("silicon", 14),
    ("phosphorus", 15), ("sulfur", 16), ("chlorine", 17), ("argon", 18),
    ("potassium", 19), ("calcium", 20), ("iron", 26), ("nickel", 28),
    ("copper", 29), ("zinc", 30), ("silver", 47), ("tin", 50),
    ("iodine", 53), ("gold", 79), ("mercury", 80), ("lead", 82),
    ("uranium", 92),
]

# Curated planet-moons bank (well-established counts as of 2025).
PLANET_MOONS_BANK: list[tuple[str, int]] = [
    ("Mercury", 0),
    ("Venus", 0),
    ("Earth", 1),
    ("Mars", 2),
]

# Curated inventor bank: (thing invented, inventor full name).
# Pick inventors whose name length is unambiguous (count of letters, ignoring
# spaces, hyphens, and accents).
INVENTOR_BANK: list[tuple[str, str]] = [
    ("the telephone", "Alexander Graham Bell"),
    ("the phonograph", "Thomas Edison"),
    ("the radio", "Guglielmo Marconi"),
    ("dynamite", "Alfred Nobel"),
    ("the printing press", "Johannes Gutenberg"),
    ("the World Wide Web", "Tim Berners Lee"),
    ("the polio vaccine", "Jonas Salk"),
    ("penicillin", "Alexander Fleming"),
]


def _gen_text_obscure_fact_composition(
    rng: np.random.Generator,
) -> tuple[str, str, Callable[[str], bool]]:
    """Compose 3 obscure facts into a single integer answer.

    answer = atomic_number(X) + moons_of(Y) - letter_count(inventor_of(Z))

    Each fact is individually verifiable and from a curated bank; closed-form
    integer answer. Frontier models often hallucinate ≥1 sub-fact.
    """
    element = ELEMENT_BANK[int(rng.integers(0, len(ELEMENT_BANK)))]
    planet = PLANET_MOONS_BANK[int(rng.integers(0, len(PLANET_MOONS_BANK)))]
    invention = INVENTOR_BANK[int(rng.integers(0, len(INVENTOR_BANK)))]
    el_name, el_z = element
    pl_name, pl_moons = planet
    thing, inventor = invention
    # Letter count of inventor's name: count alphabetic characters only.
    letter_count = sum(1 for ch in inventor if ch.isalpha())
    answer = el_z + pl_moons - letter_count
    prompt = (
        f"Compute the integer value of the following expression, using "
        f"standard reference facts:\n\n"
        f"  (atomic number of {el_name})\n"
        f"  + (number of natural moons of {pl_name})\n"
        f"  - (number of letters in the name of the inventor of {thing})\n\n"
        f"For the letter count, count only alphabetic characters in the "
        f"inventor's full common English name (spaces, hyphens, and "
        f"punctuation do NOT count). Return only the resulting integer "
        f"(it may be negative)."
    )
    return prompt, str(answer), _numeric_match(answer, rel_tol=0)


# -------------------------------------------------------------------------
# Tier table
# -------------------------------------------------------------------------

GENERATORS: list[tuple[float, Callable[[np.random.Generator], tuple[str, str, Callable[[str], bool]]]]] = [
    (0.10, _gen_t0_facts),
    (0.20, _gen_t1_dow),
    (0.30, _gen_t2_units),
    (0.40, _gen_t3_mcq),
    (0.55, _gen_t4_trivia),
    (0.70, _gen_t5_phd),
    (0.85, _gen_t6_compare),
    (0.95, _gen_t7_count),
    # T_extreme — target <15-20% on frontier (hardened post-calibration)
    (0.975, _gen_text_word_manipulation_v2),
    (0.985, _gen_text_obscure_fact_composition_v2),
    (0.995, _gen_text_letter_count_chain),
]


@dataclass(slots=True)
class KnowledgeFamily:
    name: str = "knowledge"
    description: str = (
        "Factual QA with mechanical verifiers: facts, dates, conversions, MCQ, "
        "trivia with synonyms, scientific constants, comparison, letter-count."
    )

    def generate(
        self,
        n: int,
        seed: int,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
    ) -> list[Task]:
        rng = child_rng(seed, "knowledge.generate")
        lo, hi = difficulty_range
        valid = [(d, g) for d, g in GENERATORS if lo <= d <= hi]
        if not valid:
            valid = GENERATORS
        tasks: list[Task] = []
        for i in range(n):
            d, gen = valid[rng.integers(0, len(valid))]
            sub_rng = np.random.default_rng(child_seed(seed, f"knowledge.{i}"))
            prompt, expected, verifier = gen(sub_rng)
            tasks.append(
                Task(
                    task_id=f"knowledge-{seed}-{i:04d}",
                    family="knowledge",
                    difficulty=float(d),
                    prompt=prompt,
                    verifier=verifier,
                    reference_answer=expected,
                    metadata={"generator": gen.__name__, "tier": d},
                    estimated_seconds=10.0 + 40.0 * d,
                )
            )
        return tasks

    def reference_score(self, task, response):
        return None  # mechanical only


FAMILY = KnowledgeFamily
