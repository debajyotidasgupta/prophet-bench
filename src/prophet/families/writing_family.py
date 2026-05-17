"""Writing family — open-ended writing with *mechanical* constraint checks.

Open-ended quality is hard to score without an LLM judge, so this family
defines tasks as explicit *requirements* (acrostic, word count, must-contain
phrase, lipogram, JSON schema match, rhyme scheme, etc.) and checks the
requirements with regex / parse-based verifiers. Boolean correctness is
mechanical; quality of prose is a side-channel that an optional judge can
score later.

Difficulty tiers:
  T0 (0.10) — must contain four given words in a single sentence.
  T1 (0.20) — paragraph of exactly N sentences, each containing a word.
  T2 (0.30) — lipogram: avoid a single specific letter.
  T3 (0.45) — acrostic: first letters of each line spell a given word.
  T4 (0.55) — structured rephrase preserving a fixed set of named entities.
  T5 (0.70) — write a JSON object matching a given schema.
  T6 (0.85) — rhyme: poem with ABAB scheme; verify suffix-rhyme.
  T7 (0.95) — combined constraints (length + acrostic + entity-preservation).

The agent's response is graded by `verifier(answer) -> bool`. The
``reference_answer`` we attach is a *valid* example that passes the verifier
(so the OracleAgent baseline earns positive payoff).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from prophet.engine.types import Task
from prophet.utils.seed import child_rng, child_seed


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[^.!?]+[.!?]")


def _words(s: str) -> list[str]:
    return _TOKEN_RE.findall(s)


def _sentences(s: str) -> list[str]:
    # Split on .!? while keeping the closing punctuation
    return [seg.strip() for seg in _SENT_RE.findall(s) if seg.strip()]


def _last_word(line: str) -> str:
    toks = _words(line)
    return toks[-1] if toks else ""


def _suffix_rhyme(a: str, b: str, k: int = 3) -> bool:
    a, b = a.lower(), b.lower()
    if not a or not b:
        return False
    if a == b:
        return True
    # Take last k letters of each (vowel+consonant suffix as poor man's rhyme)
    return a[-k:] == b[-k:]


# -------------------------------------------------------------------------
# T0 — sentence containing exact set of given words
# -------------------------------------------------------------------------

_WORD_BANK = [
    "apple", "banana", "cherry", "river", "mountain", "sky", "ocean", "garden",
    "phoenix", "harbor", "lantern", "shadow", "tiger", "raven", "horizon",
    "compass", "echo", "thunder", "willow", "amber", "marble", "violet",
    "owl", "ember", "comet", "anchor",
]


def _gen_t0_must_contain(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    words = list(rng.choice(_WORD_BANK, size=4, replace=False))
    required = [str(w) for w in words]
    prompt = (
        "Write a single English sentence (ending with '.', '!' or '?') that contains "
        f"all of these words: {', '.join(repr(w) for w in required)}. The order doesn't "
        "matter. Reply with only the sentence."
    )

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        sents = _sentences(answer)
        if not sents:
            return False
        # Use the first non-empty sentence
        first = sents[0].lower()
        return all(re.search(r"\b" + re.escape(w.lower()) + r"\b", first) for w in required)

    reference = (
        "The " + " ".join(required) + " filled the empty meadow with quiet wonder."
    )
    return prompt, reference, _verify


# -------------------------------------------------------------------------
# T1 — paragraph of exactly N sentences, each containing a word
# -------------------------------------------------------------------------

def _gen_t1_n_sentences(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    n = int(rng.integers(3, 6))
    target = str(rng.choice(_WORD_BANK))
    prompt = (
        f"Write a paragraph containing exactly {n} sentences. Each sentence must "
        f"contain the word '{target}' (case-insensitive, whole-word match). Sentences "
        "end with '.', '!' or '?'. Reply with only the paragraph."
    )

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        sents = _sentences(answer)
        if len(sents) != n:
            return False
        rx = re.compile(r"\b" + re.escape(target.lower()) + r"\b")
        return all(rx.search(s.lower()) for s in sents)

    reference = " ".join([f"The {target} drifts past the open window." for _ in range(n)])
    return prompt, reference, _verify


# -------------------------------------------------------------------------
# T2 — lipogram: avoid a given letter
# -------------------------------------------------------------------------

def _gen_t2_lipogram(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    # Pick a less-common letter to keep the task feasible
    letter_choices = ["l", "z", "q", "j", "x"]
    letter = letter_choices[int(rng.integers(0, len(letter_choices)))]
    min_words = int(rng.integers(15, 25))
    prompt = (
        f"Write a paragraph of at least {min_words} words that does NOT contain the "
        f"letter '{letter}' (lowercase or uppercase, anywhere). End sentences with "
        "'.', '!' or '?'. Reply with only the paragraph."
    )

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        if letter.lower() in answer.lower():
            return False
        return len(_words(answer)) >= min_words
    reference_words = [
        "a", "soft", "wind", "moves", "across", "the", "wide", "open", "fields",
        "and", "stirs", "every", "tiny", "thing", "into", "motion", "without",
        "a", "sound", "or", "warning", "as", "twin", "birds", "ride", "the", "air",
    ]
    reference_words = [w for w in reference_words if letter.lower() not in w]
    # ensure at least min_words by padding with safe filler tokens
    safe_filler = [w for w in ["dawn", "sun", "wind", "warm", "tree", "bird", "soft", "open"]
                   if letter.lower() not in w]
    while len(reference_words) < min_words and safe_filler:
        reference_words.append(safe_filler[len(reference_words) % len(safe_filler)])
    reference = " ".join(reference_words) + "."
    return prompt, reference, _verify


# -------------------------------------------------------------------------
# T3 — acrostic
# -------------------------------------------------------------------------

_ACROSTIC_WORDS = [
    "HOPE", "STAR", "RIVER", "OCEAN", "MOUNT", "PRIDE", "MUSIC",
    "PEACE", "STORM", "CLOUD", "WATER", "LIGHT",
]


def _gen_t3_acrostic(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    word = _ACROSTIC_WORDS[int(rng.integers(0, len(_ACROSTIC_WORDS)))]
    prompt = (
        f"Write a poem of exactly {len(word)} lines. The first letter of each line, "
        f"read top to bottom, must spell '{word}' (case-insensitive). Each line should "
        "be a short phrase. Reply with one line per row; do not number the lines."
    )

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
        # Strip surrounding quotes/markdown markers
        lines = [re.sub(r"^[\-*\d\.\)\s]+", "", ln) for ln in lines]
        lines = [ln for ln in lines if ln]
        if len(lines) != len(word):
            return False
        for ch, line in zip(word, lines):
            first_char = next((c for c in line if c.isalpha()), "")
            if first_char.lower() != ch.lower():
                return False
        return True

    sample_words = ["Hope", "Over", "Peace", "Earth", "Strong", "Trust", "And", "River",
                    "Mountain", "Under", "Cloud", "Light", "Water"]
    # Build a reference whose lines start with each letter
    pool: dict[str, list[str]] = {}
    for w in sample_words:
        pool.setdefault(w[0].upper(), []).append(w)
    lines = []
    for ch in word:
        cands = pool.get(ch.upper())
        if cands:
            lines.append(f"{cands[0]} rises softly")
        else:
            lines.append(f"{ch.upper()}aliantly rising")
    reference = "\n".join(lines)
    return prompt, reference, _verify


# -------------------------------------------------------------------------
# T4 — structured rephrase preserving named entities
# -------------------------------------------------------------------------

_NAME_BANK = ["Alice", "Bob", "Carol", "Dan", "Eve", "Mia", "Noah", "Zoe"]
_PLACE_BANK = ["Paris", "Tokyo", "Berlin", "Cairo", "Lima", "Oslo", "Dublin"]


def _gen_t4_rephrase(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    person = _NAME_BANK[int(rng.integers(0, len(_NAME_BANK)))]
    place = _PLACE_BANK[int(rng.integers(0, len(_PLACE_BANK)))]
    year = int(rng.integers(2000, 2026))
    source = f"In {year}, {person} traveled to {place} for a workshop on sustainability."
    prompt = (
        "Rewrite the following sentence as a single new sentence in a different style "
        "(e.g. journalistic, narrative, or formal). The rewrite MUST preserve all of "
        f"these tokens exactly: '{person}', '{place}', '{year}'.\n\n"
        f"Source: {source}\n\nReply with only the rewritten sentence."
    )
    required = [person, place, str(year)]

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        for tok in required:
            if tok not in answer:
                return False
        # Must end with sentence punctuation
        return bool(re.search(r"[.!?]\s*$", answer.strip()))

    reference = (
        f"{person} traveled to {place} in {year}, attending a workshop focused on "
        f"sustainability."
    )
    return prompt, reference, _verify


# -------------------------------------------------------------------------
# T5 — JSON matching a given schema
# -------------------------------------------------------------------------

def _gen_t5_json(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    person = _NAME_BANK[int(rng.integers(0, len(_NAME_BANK)))]
    place = _PLACE_BANK[int(rng.integers(0, len(_PLACE_BANK)))]
    age = int(rng.integers(18, 80))
    schema_descr = (
        '{"name": str, "age": int, "city": str, "hobbies": [str, str, str]}'
    )
    prompt = (
        f"Reply with ONLY a JSON object matching this schema:\n  {schema_descr}\n\n"
        f"Constraints:\n"
        f"  - 'name' must equal '{person}' (exact case).\n"
        f"  - 'age' must equal {age} (integer).\n"
        f"  - 'city' must equal '{place}' (exact case).\n"
        f"  - 'hobbies' must be a list of exactly 3 non-empty strings.\n"
        "No commentary, no markdown — only the raw JSON object."
    )

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        s = answer.strip()
        # Strip optional code-fence
        s = re.sub(r"```(?:json)?\s*", "", s)
        s = s.replace("```", "").strip()
        first = s.find("{")
        last = s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return False
        try:
            data = json.loads(s[first:last + 1])
        except Exception:
            return False
        if not isinstance(data, dict):
            return False
        if data.get("name") != person:
            return False
        if data.get("age") != age:
            return False
        if data.get("city") != place:
            return False
        hobbies = data.get("hobbies")
        if not isinstance(hobbies, list) or len(hobbies) != 3:
            return False
        return all(isinstance(h, str) and h.strip() for h in hobbies)

    reference = json.dumps({
        "name": person,
        "age": age,
        "city": place,
        "hobbies": ["reading", "hiking", "cooking"],
    })
    return prompt, reference, _verify


# -------------------------------------------------------------------------
# T6 — rhyme: ABAB poem
# -------------------------------------------------------------------------

# Each group is a set of words that share their last *two* letters, so the
# verifier's k=2 suffix match treats any pair within the group as a rhyme.
_RHYME_PAIRS = [
    ("night", "light", "bright", "sight"),     # -ht
    ("song", "long", "strong", "wrong"),       # -ng
    ("sea", "tree", "free", "knee"),           # -ee  (sea has -ea so we filter at runtime)
    ("rain", "pain", "gain", "main"),          # -in
    ("day", "way", "stay", "play"),            # -ay
    ("hope", "rope", "scope", "slope"),        # -pe
    ("storm", "form", "warm", "norm"),         # -rm
]


def _gen_t6_rhyme(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    # Pick two *distinct* rhyme groups so A and B suffixes are different.
    # Within each group, narrow to the largest subset that shares last-2 letters
    # so the verifier's suffix-rhyme check is satisfied.
    def _suffix_subset(group: tuple[str, ...]) -> list[str]:
        from collections import Counter
        suffix_counts = Counter(w[-2:].lower() for w in group)
        best_suffix, _ = suffix_counts.most_common(1)[0]
        return [w for w in group if w[-2:].lower() == best_suffix]

    a_idx = int(rng.integers(0, len(_RHYME_PAIRS)))
    b_idx = int(rng.integers(0, len(_RHYME_PAIRS)))
    while b_idx == a_idx:
        b_idx = int(rng.integers(0, len(_RHYME_PAIRS)))
    a_pool = _suffix_subset(_RHYME_PAIRS[a_idx])
    b_pool = _suffix_subset(_RHYME_PAIRS[b_idx])
    # Make sure A and B don't accidentally share last-2 letters.
    if a_pool and b_pool and a_pool[0][-2:].lower() == b_pool[0][-2:].lower():
        # fall back to a different b group
        b_idx = (b_idx + 1) % len(_RHYME_PAIRS)
        b_pool = _suffix_subset(_RHYME_PAIRS[b_idx])
    a_words = list(rng.choice(a_pool, size=2, replace=False))
    b_words = list(rng.choice(b_pool, size=2, replace=False))
    prompt = (
        "Write a four-line poem with the rhyme scheme ABAB. Each line must end in a "
        "word; lines 1 and 3 must rhyme together (the 'A' rhyme), and lines 2 and 4 "
        "must rhyme together (the 'B' rhyme). A and B rhymes must be distinct. "
        "Reply with one line per row; do not number the lines."
    )

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
        lines = [re.sub(r"^[\-*\d\.\)\s]+", "", ln) for ln in lines]
        lines = [ln for ln in lines if ln]
        if len(lines) != 4:
            return False
        endings = [_last_word(ln).lower().rstrip(",.!?;:") for ln in lines]
        if any(not e for e in endings):
            return False
        a_rhyme = _suffix_rhyme(endings[0], endings[2], k=2)
        b_rhyme = _suffix_rhyme(endings[1], endings[3], k=2)
        a_not_b = not _suffix_rhyme(endings[0], endings[1], k=2)
        return a_rhyme and b_rhyme and a_not_b

    # Reference must instantiate the ABAB scheme: line1=A, line2=B, line3=A, line4=B.
    reference = (
        f"Gentle is the morning {a_words[0]}\n"
        f"Birds upon the rooftops {b_words[0]}\n"
        f"Brighter than the silver {a_words[1]}\n"
        f"Singing as they wander {b_words[1]}"
    )
    return prompt, reference, _verify


# -------------------------------------------------------------------------
# T7 — combined constraints (length + acrostic + entity-preservation)
# -------------------------------------------------------------------------

def _gen_t7_combined(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    word = "HOPE"
    person = _NAME_BANK[int(rng.integers(0, len(_NAME_BANK)))]
    place = _PLACE_BANK[int(rng.integers(0, len(_PLACE_BANK)))]
    min_words = 20
    prompt = (
        f"Write a poem of exactly {len(word)} lines. Two combined constraints:\n"
        f"  1. The first letter of each line, top to bottom, must spell '{word}' "
        f"(case-insensitive).\n"
        f"  2. The poem MUST contain the words '{person}' and '{place}' somewhere "
        f"(exact case).\n"
        f"  3. The poem MUST contain at least {min_words} total words.\n"
        "Reply with one line per row; do not number lines."
    )

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
        lines = [re.sub(r"^[\-*\d\.\)\s]+", "", ln) for ln in lines]
        lines = [ln for ln in lines if ln]
        if len(lines) != len(word):
            return False
        for ch, line in zip(word, lines):
            first_char = next((c for c in line if c.isalpha()), "")
            if first_char.lower() != ch.lower():
                return False
        full = "\n".join(lines)
        if person not in full or place not in full:
            return False
        return len(_words(full)) >= min_words

    # Construct a reference that satisfies all constraints. Each line begins with the
    # corresponding letter of 'HOPE' and contains enough words to clear min_words.
    refs = [
        f"Here in {place} the morning rises slow and golden bright",
        f"Over the rooftops {person} walks with a quiet steady step",
        "Past the quiet windows where soft songs are humming low",
        "Every dawn is the promise of another gentle beginning",
    ]
    reference = "\n".join(refs)
    return prompt, reference, _verify


# -------------------------------------------------------------------------
# Tier table
# -------------------------------------------------------------------------

GENERATORS: list[tuple[float, Callable[[np.random.Generator], tuple[str, str, Callable[[str], bool]]]]] = [
    (0.10, _gen_t0_must_contain),
    (0.20, _gen_t1_n_sentences),
    (0.30, _gen_t2_lipogram),
    (0.45, _gen_t3_acrostic),
    (0.55, _gen_t4_rephrase),
    (0.70, _gen_t5_json),
    (0.85, _gen_t6_rhyme),
    (0.95, _gen_t7_combined),
]


@dataclass(slots=True)
class WritingFamily:
    name: str = "writing"
    description: str = (
        "Constrained writing tasks (acrostic, lipogram, word-count, JSON schema, "
        "ABAB rhyme, entity preservation). Mechanical pass/fail verifier; an "
        "optional LLM judge can score prose quality as a side-channel."
    )

    def generate(
        self,
        n: int,
        seed: int,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
    ) -> list[Task]:
        rng = child_rng(seed, "writing.generate")
        lo, hi = difficulty_range
        valid = [(d, g) for d, g in GENERATORS if lo <= d <= hi]
        if not valid:
            valid = GENERATORS
        tasks: list[Task] = []
        for i in range(n):
            d, gen = valid[rng.integers(0, len(valid))]
            sub_rng = np.random.default_rng(child_seed(seed, f"writing.{i}"))
            prompt, expected, verifier = gen(sub_rng)
            tasks.append(
                Task(
                    task_id=f"writing-{seed}-{i:04d}",
                    family="writing",
                    difficulty=float(d),
                    prompt=prompt,
                    verifier=verifier,
                    reference_answer=expected,
                    metadata={"generator": gen.__name__, "tier": d},
                    judge_required=False,  # mechanical verifier; quality is side-channel
                    estimated_seconds=30.0 + 90.0 * d,
                )
            )
        return tasks

    def reference_score(self, task, response):  # noqa: ANN001
        # Constraints are mechanical. We could wire an LLM judge here for prose
        # quality, but it does not affect benchmark pass/fail.
        return None


FAMILY = WritingFamily
