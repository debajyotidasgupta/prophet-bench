"""Multilingual family — language identification + simple cross-lingual reasoning.

Designed so the verifier is *language-agnostic* — most tasks reduce to picking a
short token (letter / digit / language name / number) from the model's output.
Word lists are bundled as ``data/seeds/multilingual_dict.json`` so the family
has no runtime network dependency.

Difficulty tiers:
  T0 (0.10) — language identification of a single word ("gracias" → spanish).
  T1 (0.20) — multiple-choice translation (EN ↔ ES/FR/DE/HI/JP).
  T2 (0.35) — vowel count of a non-English word.
  T3 (0.50) — cognate identification (which option shares a Romance/Latinate root).
  T4 (0.65) — arithmetic stated in a non-English language ("dos + tres = ?").
  T5 (0.80) — word-ordering: scrambled-sentence multiple choice.
  T6 (0.95) — pick the single grammatically correct sentence among 4 options.

Notes:
  * Where multiple-choice is used, options are labelled A / B / C / D; the
    verifier accepts any case of the correct letter, optionally with trailing
    punctuation or wrapped in ``\\boxed{...}``.
  * Where the answer is a language name, the verifier accepts case-insensitive
    aliases from the bundled dictionary (e.g. "ES", "espanol", "spanish").
  * Where the answer is a small integer, the same integer-tail verifier as the
    math family is used.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Callable

import numpy as np

from prophet.engine.types import Task
from prophet.utils.seed import child_rng, child_seed

# ---------------------------------------------------------------------------
# Bundled dictionary loader
# ---------------------------------------------------------------------------

_DICT_CACHE: dict[str, Any] | None = None


def _dict_path() -> Path:
    """Resolve the on-disk path to the bundled multilingual_dict.json.

    Search order (first hit wins):
      1. installed package data via ``prophet.data.seeds``
      2. dev tree at ``<repo>/data/seeds/multilingual_dict.json``
      3. environment override ``PROPHET_MULTILINGUAL_DICT``
    """
    import os

    # Env override always wins for testing.
    env = os.environ.get("PROPHET_MULTILINGUAL_DICT")
    if env:
        return Path(env)
    # Package-data lookup.
    try:
        ref = resources.files("prophet").joinpath("data/seeds/multilingual_dict.json")
        if ref.is_file():  # type: ignore[attr-defined]
            return Path(str(ref))
    except (ModuleNotFoundError, AttributeError, FileNotFoundError):
        pass
    # Dev tree relative to this file: src/prophet/families/multilingual_family.py
    # → ../../../data/seeds/multilingual_dict.json
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    return repo_root / "data" / "seeds" / "multilingual_dict.json"


def _load_dict() -> dict[str, Any]:
    global _DICT_CACHE
    if _DICT_CACHE is None:
        path = _dict_path()
        with open(path, encoding="utf-8") as f:
            _DICT_CACHE = json.load(f)
    return _DICT_CACHE


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------

_VOWELS = set("aeiouAEIOU")


def _normalize(s: str) -> str:
    s = s.strip().lower()
    # Strip common wrappers and trailing punctuation.
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "").replace("**", "")
    s = s.strip().strip(".,;:!?\"'`()[]")
    return s


def _language_verifier(expected_lang: str, aliases: dict[str, list[str]]) -> Callable[[str], bool]:
    """Accepts any alias of the expected language name (case-insensitive)."""
    expected_aliases = {a.lower() for a in aliases.get(expected_lang, [expected_lang])}

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        s = _normalize(answer)
        # Look at the last non-empty line.
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if lines:
            s = lines[-1]
        # Accept either exact alias match OR alias substring match (e.g.
        # "the language is spanish").
        if s in expected_aliases:
            return True
        return any(a in s.split() or a in s for a in expected_aliases)

    return _verify


def _choice_verifier(letter: str) -> Callable[[str], bool]:
    """Accept the choice letter (A/B/C/D), case-insensitively, with optional wrappers."""
    expected = letter.strip().upper()

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        s = _normalize(answer)
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if lines:
            s = lines[-1]
        # Search for the first letter token; the model may say "Answer: B" etc.
        m = re.search(r"\b([a-d])\b", s)
        if m:
            return m.group(1).upper() == expected
        return s.upper() == expected

    return _verify


def _integer_verifier(expected: int) -> Callable[[str], bool]:
    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        s = answer.replace(",", "")
        s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
        nums = re.findall(r"-?\d+", s)
        if not nums:
            return False
        try:
            return int(nums[-1]) == expected
        except ValueError:
            return False

    return _verify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LETTERS = ["A", "B", "C", "D"]


def _build_choices(
    correct: str,
    distractors: list[str],
    rng: np.random.Generator,
) -> tuple[list[str], str]:
    """Place `correct` among up to 3 random distinct distractors. Returns (options, letter)."""
    pool = list(dict.fromkeys(distractors))  # dedupe preserve order
    # Remove any accidental match to correct
    pool = [d for d in pool if d != correct]
    rng.shuffle(pool)
    options = [correct] + pool[:3]
    while len(options) < 4:
        options.append(f"option_{len(options)}")
    rng.shuffle(options)
    letter = _LETTERS[options.index(correct)]
    return options, letter


def _format_choices(options: list[str]) -> str:
    return "\n".join(f"{_LETTERS[i]}) {o}" for i, o in enumerate(options))


# ---------------------------------------------------------------------------
# Tier generators
# ---------------------------------------------------------------------------


def _gen_t0_lang_id(rng: np.random.Generator, data: dict[str, Any]) -> tuple[str, Callable[[str], bool], str, dict]:
    """Single-word language identification."""
    translations = data["translations"]
    entry = translations[int(rng.integers(0, len(translations)))]
    lang_codes = [k for k in entry if k in {"en", "es", "fr", "de", "hi", "jp"}]
    code = lang_codes[int(rng.integers(0, len(lang_codes)))]
    word = entry[code]
    lang_name = data["language_names"][code]
    prompt = (
        f"What language is this word: '{word}'?\n"
        f"Answer with one of: english, spanish, french, german, hindi, japanese."
    )
    verifier = _language_verifier(lang_name, data["language_aliases"])
    return prompt, verifier, lang_name, {"lang_code": code, "word": word}


def _gen_t1_translation_mc(
    rng: np.random.Generator, data: dict[str, Any]
) -> tuple[str, Callable[[str], bool], str, dict]:
    """Multiple-choice translation."""
    translations = data["translations"]
    # Pick a source entry + a target language.
    entry = translations[int(rng.integers(0, len(translations)))]
    target = ["es", "fr", "de", "hi", "jp"][int(rng.integers(0, 5))]
    if target not in entry:
        target = "es"
    correct = entry[target]
    # Build distractors: same-language words from other entries.
    pool = [e[target] for e in translations if target in e and e[target] != correct]
    rng.shuffle(pool)
    distractors = pool[:5]
    options, letter = _build_choices(correct, distractors, rng)
    target_name = data["language_names"][target]
    prompt = (
        f"Which of the following is the {target_name} word for '{entry['en']}'?\n"
        f"{_format_choices(options)}\n"
        f"Reply with a single letter (A, B, C, or D)."
    )
    verifier = _choice_verifier(letter)
    return prompt, verifier, letter, {"target_lang": target, "correct": correct, "options": options}


def _gen_t2_vowel_count(
    rng: np.random.Generator, data: dict[str, Any]
) -> tuple[str, Callable[[str], bool], str, dict]:
    """Vowel count of a non-English word (transliterated, so a/e/i/o/u apply)."""
    translations = data["translations"]
    entry = translations[int(rng.integers(0, len(translations)))]
    non_en = [k for k in entry if k in {"es", "fr", "de", "hi", "jp"}]
    code = non_en[int(rng.integers(0, len(non_en)))]
    word = entry[code]
    count = sum(1 for ch in word if ch in _VOWELS)
    lang_name = data["language_names"][code]
    prompt = (
        f"Count the number of vowels (a, e, i, o, u; case-insensitive) in the "
        f"{lang_name} word '{word}'. Return a single integer."
    )
    verifier = _integer_verifier(count)
    return prompt, verifier, str(count), {"word": word, "lang_code": code}


def _gen_t3_cognate(
    rng: np.random.Generator, data: dict[str, Any]
) -> tuple[str, Callable[[str], bool], str, dict]:
    """Pick the cognate of the prompt word."""
    groups = data["cognate_groups"]
    group = groups[int(rng.integers(0, len(groups)))]
    root = group["root"]
    members = group["members"]
    # Pick one cognate as correct; the others are removed from distractors.
    correct_entry = members[int(rng.integers(0, len(members)))]
    correct = correct_entry["word"]
    distractors = list(group["distractors"])
    options, letter = _build_choices(correct, distractors, rng)
    prompt = (
        f"Which of the following words shares the same Latin/Greek root as the English word '{root}'?\n"
        f"{_format_choices(options)}\n"
        f"Reply with a single letter (A, B, C, or D)."
    )
    verifier = _choice_verifier(letter)
    return prompt, verifier, letter, {"root": root, "correct": correct, "options": options}


def _gen_t4_arithmetic_lang(
    rng: np.random.Generator, data: dict[str, Any]
) -> tuple[str, Callable[[str], bool], str, dict]:
    """Arithmetic stated in a non-English language; answer is integer."""
    numbers = data["numbers"]
    # Pick a non-English language.
    code = ["es", "fr", "de"][int(rng.integers(0, 3))]
    plus_word = data["plus_words"][code]
    a = int(rng.integers(0, 8))
    b = int(rng.integers(0, 8))
    a_word = numbers[a][code]
    b_word = numbers[b][code]
    ans = a + b
    lang_name = data["language_names"][code]
    prompt = (
        f"Solve this arithmetic problem stated in {lang_name}:\n"
        f"'{a_word} {plus_word} {b_word} = ?'\n"
        f"Return the answer as an Arabic-numeral integer."
    )
    verifier = _integer_verifier(ans)
    return prompt, verifier, str(ans), {"a": a, "b": b, "lang_code": code}


def _gen_t5_word_ordering(
    rng: np.random.Generator, data: dict[str, Any]
) -> tuple[str, Callable[[str], bool], str, dict]:
    """Word-ordering multiple-choice. Show 4 orderings; 1 is the correct sentence."""
    sentences = data["ordering_sentences"]
    sent = sentences[int(rng.integers(0, len(sentences)))]
    ordered = sent["ordered"]
    correct = " ".join(ordered)
    # Build 3 distractor orderings by shuffling deterministically.
    distractors = set()
    attempts = 0
    while len(distractors) < 3 and attempts < 30:
        perm = list(ordered)
        rng.shuffle(perm)
        candidate = " ".join(perm)
        if candidate != correct:
            distractors.add(candidate)
        attempts += 1
    while len(distractors) < 3:
        distractors.add(" ".join(reversed(ordered)) + f" #{len(distractors)}")
    options, letter = _build_choices(correct, list(distractors), rng)
    lang_name = data["language_names"].get(sent.get("lang", "en"), "english")
    prompt = (
        f"Below are four orderings of the same set of {lang_name} words. "
        f"Which ordering forms a grammatically correct sentence?\n"
        f"{_format_choices(options)}\n"
        f"Reply with a single letter (A, B, C, or D)."
    )
    verifier = _choice_verifier(letter)
    return prompt, verifier, letter, {"lang": sent.get("lang", "en"), "options": options}


def _gen_t6_grammar(
    rng: np.random.Generator, data: dict[str, Any]
) -> tuple[str, Callable[[str], bool], str, dict]:
    """Pick the grammatically correct sentence among 4 options spanning languages."""
    choices = data["grammar_choices"]
    item = choices[int(rng.integers(0, len(choices)))]
    correct = item["correct"]
    distractors = list(item["wrong"])
    options, letter = _build_choices(correct, distractors, rng)
    lang_name = data["language_names"].get(item.get("lang", "en"), "english")
    prompt = (
        f"Exactly one of these {lang_name} sentences is grammatically correct. "
        f"Identify it:\n"
        f"{_format_choices(options)}\n"
        f"Reply with a single letter (A, B, C, or D)."
    )
    verifier = _choice_verifier(letter)
    return prompt, verifier, letter, {"lang": item.get("lang", "en"), "options": options}


GENERATORS: list[
    tuple[
        float,
        Callable[
            [np.random.Generator, dict[str, Any]],
            tuple[str, Callable[[str], bool], str, dict],
        ],
    ]
] = [
    (0.10, _gen_t0_lang_id),
    (0.20, _gen_t1_translation_mc),
    (0.35, _gen_t2_vowel_count),
    (0.50, _gen_t3_cognate),
    (0.65, _gen_t4_arithmetic_lang),
    (0.80, _gen_t5_word_ordering),
    (0.95, _gen_t6_grammar),
]


@dataclass(slots=True)
class MultilingualFamily:
    name: str = "multilingual"
    description: str = (
        "Language identification and cross-lingual reasoning over a bundled "
        "EN/ES/FR/DE/HI/JP word list. Verifiers are mechanical (letter, integer, "
        "or language-name alias)."
    )

    def generate(
        self,
        n: int,
        seed: int,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
    ) -> list[Task]:
        rng = child_rng(seed, "multilingual.generate")
        lo, hi = difficulty_range
        data = _load_dict()
        valid = [(d, g) for d, g in GENERATORS if lo <= d <= hi]
        if not valid:
            valid = GENERATORS
        tasks: list[Task] = []
        for i in range(n):
            d, gen = valid[int(rng.integers(0, len(valid)))]
            sub_rng = np.random.default_rng(child_seed(seed, f"multilingual.{i}"))
            prompt, verifier, ref_ans, meta = gen(sub_rng, data)
            tasks.append(
                Task(
                    task_id=f"multilingual-{seed}-{i:04d}",
                    family="multilingual",
                    difficulty=float(d),
                    prompt=prompt,
                    verifier=verifier,
                    reference_answer=ref_ans,
                    metadata={"generator": gen.__name__, "tier": d, **meta},
                    estimated_seconds=15.0 + 30.0 * d,
                )
            )
        return tasks

    def reference_score(self, task, response):  # noqa: ANN001
        return None


FAMILY = MultilingualFamily
