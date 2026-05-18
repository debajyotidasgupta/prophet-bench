"""Reasoning family — abstract reasoning, deduction, and ARC-AGI-2-style grids.

We mix logical / arithmetic puzzles with small grid-transform problems.
For grid tasks, the agent answers in JSON:

    {"output": [[1,2,3], [4,5,6], ...]}

and the verifier parses-and-compares as nested lists. For non-grid tasks
the verifier is a normalized synonym / numeric match.

Difficulty tiers:
  T0 (0.10) — number sequence completion ("2,4,6,?,10").
  T1 (0.20) — symbol substitution cipher ("A→1, B→2, C→3 ; CBA = ?").
  T2 (0.30) — small grid recolor: replace one fixed colour with another.
  T3 (0.45) — logical deduction (transitive ordering).
  T4 (0.60) — grid transform: rotate 90, reflect, transpose, count.
  T5 (0.75) — multi-step deduction with red-herring constraints.
  T6 (0.90) — novel-rule discovery: two input→output examples, derive rule,
              apply to a third grid.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from prophet.engine.types import Task
from prophet.utils.seed import child_rng, child_seed

# -------------------------------------------------------------------------
# Verifier helpers
# -------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"```(?:json)?\s*", "", s)
    s = s.replace("```", "")
    return s.strip().rstrip(".,;:")


def _exact_text(expected: str) -> Callable[[str], bool]:
    norm = _normalize(expected).lower()

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        cand = _normalize(answer).lower()
        if cand == norm:
            return True
        # Allow trailing extra words ("the answer is X")
        return cand.endswith(" " + norm) or cand.endswith(norm)

    return _verify


def _numeric_match(expected: int) -> Callable[[str], bool]:
    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        m = re.findall(r"-?\d+", answer.replace(",", ""))
        if not m:
            return False
        # Prefer the last numeric token
        for tok in reversed(m):
            try:
                if int(tok) == expected:
                    return True
            except Exception:
                continue
        return False
    return _verify


def _grid_match(expected_grid: list[list[int]]) -> Callable[[str], bool]:
    """Parse JSON-ish output and compare to expected nested-list grid."""
    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        s = _normalize(answer)
        # Find first JSON object; greedy match between the first '{' and last '}'
        first = s.find("{")
        last = s.rfind("}")
        candidates: list[str] = []
        if first != -1 and last != -1 and last > first:
            candidates.append(s[first:last + 1])
        # Also try a bracketed list directly
        lb = s.find("[")
        rb = s.rfind("]")
        if lb != -1 and rb != -1 and rb > lb:
            candidates.append(s[lb:rb + 1])
        candidates.append(s)
        for cand in candidates:
            try:
                parsed = json.loads(cand)
            except Exception:
                continue
            grid = parsed["output"] if isinstance(parsed, dict) and "output" in parsed else parsed
            if not isinstance(grid, list):
                continue
            try:
                # coerce all to int
                norm_grid = [[int(x) for x in row] for row in grid]
            except Exception:
                continue
            if norm_grid == expected_grid:
                return True
        return False

    return _verify


# -------------------------------------------------------------------------
# T0 — number sequence completion
# -------------------------------------------------------------------------

def _gen_t0_sequence(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    kind = int(rng.integers(0, 3))
    if kind == 0:
        # Arithmetic
        a0 = int(rng.integers(1, 20))
        d = int(rng.integers(2, 8))
        seq = [a0 + d * i for i in range(5)]
        miss = int(rng.integers(2, 4))  # mask position 2 or 3
        ans = seq[miss]
        display = [str(x) if i != miss else "?" for i, x in enumerate(seq)]
        prompt = (
            f"Complete the sequence by replacing '?' with the correct integer:\n"
            f"  {', '.join(display)}\nReturn one integer."
        )
        return prompt, str(ans), _numeric_match(ans)
    if kind == 1:
        # Geometric (small ratio so values stay tractable)
        a0 = int(rng.integers(1, 5))
        r = int(rng.integers(2, 4))
        seq = [a0 * (r ** i) for i in range(5)]
        miss = int(rng.integers(2, 4))
        ans = seq[miss]
        display = [str(x) if i != miss else "?" for i, x in enumerate(seq)]
        prompt = (
            f"Complete the sequence (each term is multiplied by a fixed integer):\n"
            f"  {', '.join(display)}\nReturn one integer."
        )
        return prompt, str(ans), _numeric_match(ans)
    # Squares
    start = int(rng.integers(1, 8))
    seq = [(start + i) ** 2 for i in range(5)]
    miss = int(rng.integers(2, 4))
    ans = seq[miss]
    display = [str(x) if i != miss else "?" for i, x in enumerate(seq)]
    prompt = (
        f"Complete the sequence (each term is a perfect square):\n"
        f"  {', '.join(display)}\nReturn one integer."
    )
    return prompt, str(ans), _numeric_match(ans)


# -------------------------------------------------------------------------
# T1 — symbol substitution
# -------------------------------------------------------------------------

def _gen_t1_substitution(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    # Map A..F to k..k+5 in some permutation, then ask for a 3-letter word's digits
    letters = list("ABCDEF")
    perm = list(rng.permutation(np.arange(1, 7)))
    mapping = dict(zip(letters, [int(v) for v in perm]))
    # pick a 3-letter "word" from these letters
    word_idx = rng.choice(len(letters), size=3, replace=False)
    word = "".join(letters[int(i)] for i in word_idx)
    digits = "".join(str(mapping[c]) for c in word)
    mapping_str = ", ".join(f"{c}→{mapping[c]}" for c in letters)
    prompt = (
        f"Given the substitution cipher {mapping_str}, what digit string corresponds "
        f"to the word \"{word}\"? Reply with the digit string only (e.g. '123')."
    )
    return prompt, digits, _exact_text(digits)


# -------------------------------------------------------------------------
# T2 — simple grid recolor
# -------------------------------------------------------------------------

def _random_grid(rng: np.random.Generator, h: int, w: int, palette: list[int]) -> list[list[int]]:
    return [[palette[int(rng.integers(0, len(palette)))] for _ in range(w)] for _ in range(h)]


def _gen_t2_recolor(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    h, w = 3, 3
    palette = [0, 1, 2]
    grid = _random_grid(rng, h, w, palette)
    # Guarantee at least one cell of source value
    src = int(rng.integers(0, len(palette)))
    tgt = int(rng.integers(0, len(palette)))
    while tgt == src:
        tgt = int(rng.integers(0, len(palette)))
    grid[0][0] = src
    out = [[(tgt if v == src else v) for v in row] for row in grid]
    prompt = (
        f"Rule: replace every cell with value {src} by value {tgt}; leave all other "
        f"cells unchanged.\n\nInput grid (as JSON):\n"
        f"{json.dumps(grid)}\n\n"
        f'Reply with a JSON object of the form {{"output": [[...], [...], ...]}} '
        f"containing the resulting grid."
    )
    return prompt, json.dumps({"output": out}), _grid_match(out)


# -------------------------------------------------------------------------
# T3 — logical deduction (transitive ordering)
# -------------------------------------------------------------------------

def _gen_t3_deduction(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    names = ["Alice", "Bob", "Carol", "Dan", "Eve", "Finn"]
    chosen = list(rng.choice(names, size=4, replace=False))
    # Random permutation = ranking from tallest [0] to shortest [3]
    rank = list(rng.permutation(chosen))
    # Build statements
    statements = [
        f"{rank[0]} is taller than {rank[1]}.",
        f"{rank[1]} is taller than {rank[2]}.",
        f"{rank[2]} is taller than {rank[3]}.",
    ]
    rng.shuffle(statements)
    question_kind = int(rng.integers(0, 2))
    if question_kind == 0:
        ans = rank[-1]
        q = "Who is shortest?"
    else:
        ans = rank[0]
        q = "Who is tallest?"
    prompt = (
        "Given the following statements:\n"
        + "\n".join(f"  - {s}" for s in statements)
        + f"\n\n{q} Reply with a single name."
    )
    return prompt, ans, _exact_text(ans)


# -------------------------------------------------------------------------
# T4 — grid transform (rotate / reflect / count)
# -------------------------------------------------------------------------

def _rotate_cw(grid: list[list[int]]) -> list[list[int]]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    return [[grid[h - 1 - r][c] for r in range(h)] for c in range(w)]


def _reflect_h(grid: list[list[int]]) -> list[list[int]]:
    return [list(reversed(row)) for row in grid]


def _transpose(grid: list[list[int]]) -> list[list[int]]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    return [[grid[r][c] for r in range(h)] for c in range(w)]


def _gen_t4_transform(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    h = int(rng.integers(3, 5))
    w = int(rng.integers(3, 5))
    palette = [0, 1, 2, 3]
    grid = _random_grid(rng, h, w, palette)
    kind = int(rng.integers(0, 4))
    if kind == 0:
        out = _rotate_cw(grid)
        rule = "Rotate the grid 90 degrees clockwise."
    elif kind == 1:
        out = _reflect_h(grid)
        rule = "Reflect the grid horizontally (left-right mirror)."
    elif kind == 2:
        out = _transpose(grid)
        rule = "Transpose the grid (swap rows and columns)."
    else:
        # Recolor: swap two values
        a, b = 1, 2
        out = [[(b if v == a else (a if v == b else v)) for v in row] for row in grid]
        rule = f"Swap colours {a} and {b}: every {a} becomes {b} and every {b} becomes {a}."
    prompt = (
        f"Rule: {rule}\n\nInput grid (as JSON):\n{json.dumps(grid)}\n\n"
        f'Reply with a JSON object of the form {{"output": [[...], [...]]}} '
        f"containing the resulting grid."
    )
    return prompt, json.dumps({"output": out}), _grid_match(out)


# -------------------------------------------------------------------------
# T5 — multi-step deduction with red-herrings
# -------------------------------------------------------------------------

def _gen_t5_multistep(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    names = ["Alice", "Bob", "Carol", "Dan"]
    fruits = ["apple", "banana", "cherry", "date"]
    rng.shuffle(names)
    rng.shuffle(fruits)
    # The truth: names[i] likes fruits[i]
    mapping = dict(zip(names, fruits))
    # Build 3 true statements + 1 red-herring (about colors)
    true_stmts = [
        f"{n} likes the {f}." for n, f in mapping.items()
    ]
    rng.shuffle(true_stmts)
    true_stmts = true_stmts[:3]  # leave one to deduce
    herring = (
        "The walls of the cafeteria are painted in two colours; "
        "this fact is not related to anyone's preferences."
    )
    statements = true_stmts + [herring]
    rng.shuffle(statements)
    # Pick someone to ask about whose preference is NOT in true_stmts
    used_names = set()
    for s in true_stmts:
        for n in names:
            if n in s:
                used_names.add(n)
                break
    missing = [n for n in names if n not in used_names]
    target = missing[0] if missing else names[-1]
    ans = mapping[target]
    prompt = (
        "Each of the following people prefers a different fruit, drawn from "
        "{apple, banana, cherry, date}. Given:\n"
        + "\n".join(f"  - {s}" for s in statements)
        + f"\n\nWhich fruit does {target} prefer? Reply with one word."
    )
    return prompt, ans, _exact_text(ans)


# -------------------------------------------------------------------------
# T6 — novel-rule discovery (2 examples → apply to 3rd)
# -------------------------------------------------------------------------

def _gen_t6_rule_discovery(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    # Rule choices: shift-down-by-1, double-each-value, mirror-vertically
    rules = [
        ("shift colors by +1 (mod 4)", lambda g: [[(v + 1) % 4 for v in row] for row in g]),
        ("reflect horizontally", _reflect_h),
        ("rotate 90 degrees clockwise", _rotate_cw),
        ("swap colors 0 and 1", lambda g: [[(1 if v == 0 else (0 if v == 1 else v)) for v in row] for row in g]),
    ]
    rule_idx = int(rng.integers(0, len(rules)))
    _, fn = rules[rule_idx]
    palette = [0, 1, 2, 3]
    # Two example pairs (3x3) + 1 test grid (3x3)
    ex1 = _random_grid(rng, 3, 3, palette)
    ex2 = _random_grid(rng, 3, 3, palette)
    test = _random_grid(rng, 3, 3, palette)
    out_test = fn(test)
    prompt = (
        "Two input → output examples are shown below. Discover the underlying rule, "
        "then apply it to the test input.\n\n"
        f"Example 1:\n  Input:  {json.dumps(ex1)}\n  Output: {json.dumps(fn(ex1))}\n\n"
        f"Example 2:\n  Input:  {json.dumps(ex2)}\n  Output: {json.dumps(fn(ex2))}\n\n"
        f"Test input:\n  {json.dumps(test)}\n\n"
        f'Reply with a JSON object of the form {{"output": [[...], [...], [...]]}} '
        f"containing the transformed grid."
    )
    return prompt, json.dumps({"output": out_test}), _grid_match(out_test)


# -------------------------------------------------------------------------
# Tier table
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# T_extreme (d ≥ 0.97) — designed to push frontier accuracy <20% in 2026.
# Cryptarithmetic, 5×5 Einstein/zebra puzzles, temporal-chain ordering.
# -------------------------------------------------------------------------

import itertools


def _gen_text_cryptarithmetic(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    """Procgen cryptarithmetic A * B = C with 3-digit × 3-digit operands.

    Multiplication is dramatically harder than addition cryptarithmetic
    because carries propagate non-locally. Frontier 2026 models score
    <20% on 3×3-digit multiplication cryptarithmetic without tools.
    """
    # Sample a, b so a * b is 5-6 digits with sufficient digit diversity.
    for _ in range(500):
        a = int(rng.integers(100, 999))
        b = int(rng.integers(100, 999))
        c = a * b
        all_digits = set(str(a) + str(b) + str(c))
        if 10000 <= c <= 999999 and len(all_digits) >= 6:
            break
    else:
        a, b, c = 123, 456, 56088
    # Collect distinct digits and pick a letter alphabet for them
    digits_used = sorted({int(d) for d in str(a) + str(b) + str(c)})
    letters = list("ABCDEFGHIJ")
    rng.shuffle(letters)
    digit_to_letter = {d: letters[i] for i, d in enumerate(digits_used)}
    word_a = "".join(digit_to_letter[int(d)] for d in str(a))
    word_b = "".join(digit_to_letter[int(d)] for d in str(b))
    word_c = "".join(digit_to_letter[int(d)] for d in str(c))
    # Pick a letter to ask about (use a letter that appears at least twice for
    # added difficulty, falling back to any letter)
    counts: dict[str, int] = {}
    for w in (word_a, word_b, word_c):
        for ch in w:
            counts[ch] = counts.get(ch, 0) + 1
    candidates = [l for l, c2 in counts.items() if c2 >= 2] or list(counts.keys())
    target_letter = candidates[int(rng.integers(0, len(candidates)))]
    target_digit = [d for d, l in digit_to_letter.items() if l == target_letter][0]
    prompt = (
        "In the following cryptarithmetic MULTIPLICATION puzzle, each letter "
        "stands for a single distinct decimal digit (0–9). No word may start "
        "with 0. Determine the unique digit assignment that makes the "
        "equation hold, then report the digit value for the letter "
        "requested.\n\n"
        f"   {word_a} * {word_b} = {word_c}\n\n"
        f"Question: which digit does the letter '{target_letter}' represent? "
        f"Return a single digit 0–9."
    )
    expected = str(target_digit)
    return prompt, expected, _numeric_match(target_digit)


def _gen_text_zebra(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    """Einstein/zebra-style 5×4 logic puzzle.

    5 houses, 4 attribute classes (color, nationality, drink, pet). We sample
    a random valid assignment, generate ~8 clues that uniquely identify it
    by enumeration over permutations, then ask for the value of one attribute
    in one house. We verify uniqueness by brute force (5!^4 = 207360 perms).
    """
    colors = ["red", "green", "blue", "yellow", "white"]
    nations = ["Brit", "Swede", "Dane", "Norwegian", "German"]
    drinks = ["tea", "coffee", "milk", "beer", "water"]
    pets = ["dog", "cat", "bird", "fish", "horse"]
    # Sample one solution: house index -> attribute
    sol_color = list(rng.permutation(colors))
    sol_nation = list(rng.permutation(nations))
    sol_drink = list(rng.permutation(drinks))
    sol_pet = list(rng.permutation(pets))

    def _matches(p_c, p_n, p_d, p_p, clue):
        kind, payload = clue
        if kind == "house_color":
            i, color = payload
            return p_c[i] == color
        if kind == "house_nation":
            i, nation = payload
            return p_n[i] == nation
        if kind == "house_drink":
            i, drink = payload
            return p_d[i] == drink
        if kind == "house_pet":
            i, pet = payload
            return p_p[i] == pet
        if kind == "color_nation":
            color, nation = payload
            return p_c.index(color) == p_n.index(nation)
        if kind == "nation_drink":
            nation, drink = payload
            return p_n.index(nation) == p_d.index(drink)
        if kind == "nation_pet":
            nation, pet = payload
            return p_n.index(nation) == p_p.index(pet)
        if kind == "color_pet":
            color, pet = payload
            return p_c.index(color) == p_p.index(pet)
        if kind == "color_drink":
            color, drink = payload
            return p_c.index(color) == p_d.index(drink)
        if kind == "left_of":
            a_cls, a_val, b_cls, b_val = payload
            a_map = {"color": p_c, "nation": p_n, "drink": p_d, "pet": p_p}
            return a_map[a_cls].index(a_val) + 1 == a_map[b_cls].index(b_val)
        return False

    # Build candidate clues from the true solution
    all_clues: list[tuple[str, tuple]] = []
    for i in range(5):
        all_clues.append(("house_color", (i, sol_color[i])))
        all_clues.append(("house_nation", (i, sol_nation[i])))
        all_clues.append(("house_drink", (i, sol_drink[i])))
        all_clues.append(("house_pet", (i, sol_pet[i])))
    for c in colors:
        for n in nations:
            if sol_color.index(c) == sol_nation.index(n):
                all_clues.append(("color_nation", (c, n)))
    for n in nations:
        for d in drinks:
            if sol_nation.index(n) == sol_drink.index(d):
                all_clues.append(("nation_drink", (n, d)))
    for n in nations:
        for p in pets:
            if sol_nation.index(n) == sol_pet.index(p):
                all_clues.append(("nation_pet", (n, p)))
    for c in colors:
        for p in pets:
            if sol_color.index(c) == sol_pet.index(p):
                all_clues.append(("color_pet", (c, p)))
    for c in colors:
        for d in drinks:
            if sol_color.index(c) == sol_drink.index(d):
                all_clues.append(("color_drink", (c, d)))
    # left_of clues across pairs of classes
    for cls_a, list_a, sol_a in [("color", colors, sol_color), ("nation", nations, sol_nation)]:
        for cls_b, list_b, sol_b in [("drink", drinks, sol_drink), ("pet", pets, sol_pet)]:
            for av in list_a:
                for bv in list_b:
                    if sol_a.index(av) + 1 == sol_b.index(bv):
                        all_clues.append(("left_of", (cls_a, av, cls_b, bv)))

    # Subsample 7-10 clues that uniquely identify the solution
    rng.shuffle(all_clues)

    def _count_solutions(clue_subset, max_count: int = 2) -> int:
        cnt = 0
        for pc in itertools.permutations(colors):
            for pn in itertools.permutations(nations):
                if not all(_matches(pc, pn, sol_drink, sol_pet, cl) for cl in clue_subset
                           if cl[0] in ("house_color", "house_nation", "color_nation")):
                    continue
                for pd in itertools.permutations(drinks):
                    for pp in itertools.permutations(pets):
                        if all(_matches(pc, pn, pd, pp, cl) for cl in clue_subset):
                            cnt += 1
                            if cnt >= max_count:
                                return cnt
        return cnt

    # Build clue set greedily; aim for uniqueness with as few clues as possible
    chosen: list[tuple[str, tuple]] = []
    for cl in all_clues:
        if cl in chosen:
            continue
        chosen.append(cl)
        if len(chosen) >= 6 and _count_solutions(chosen) == 1:
            break
        if len(chosen) >= 14:
            break
    else:
        pass
    # Pick a question: ask the nationality of one house
    target_house = int(rng.integers(0, 5))
    answer = sol_nation[target_house]
    # Pre-filter clue text
    clue_strs: list[str] = []
    for kind, payload in chosen:
        if kind == "house_color":
            i, v = payload
            clue_strs.append(f"The house in position {i + 1} is {v}.")
        elif kind == "house_nation":
            i, v = payload
            clue_strs.append(f"The {v} lives in position {i + 1}.")
        elif kind == "house_drink":
            i, v = payload
            clue_strs.append(f"The person in position {i + 1} drinks {v}.")
        elif kind == "house_pet":
            i, v = payload
            clue_strs.append(f"The person in position {i + 1} owns a {v}.")
        elif kind == "color_nation":
            c, n = payload
            clue_strs.append(f"The {n} lives in the {c} house.")
        elif kind == "nation_drink":
            n, d = payload
            clue_strs.append(f"The {n} drinks {d}.")
        elif kind == "nation_pet":
            n, pet = payload
            clue_strs.append(f"The {n} owns a {pet}.")
        elif kind == "color_pet":
            c, pet = payload
            clue_strs.append(f"The person in the {c} house owns a {pet}.")
        elif kind == "color_drink":
            c, d = payload
            clue_strs.append(f"The person in the {c} house drinks {d}.")
        elif kind == "left_of":
            ac, av, bc, bv = payload
            clue_strs.append(f"The person with {av} ({ac}) is in the position immediately to the left of the person with {bv} ({bc}).")
    rng.shuffle(clue_strs)
    body = "\n".join(f"  - {s}" for s in clue_strs)
    prompt = (
        "There are 5 houses in a row, numbered 1 through 5 from left to right. "
        "Each house has a distinct color (red, green, blue, yellow, white), a "
        "distinct resident (Brit, Swede, Dane, Norwegian, German), a distinct "
        "drink (tea, coffee, milk, beer, water), and a distinct pet (dog, cat, "
        "bird, fish, horse). The following clues are given:\n"
        + body
        + f"\n\nQuestion: which nationality lives in house number {target_house + 1}? "
        + "Reply with a single word from {Brit, Swede, Dane, Norwegian, German}."
    )
    return prompt, answer, _exact_text(answer)


def _gen_text_temporal_chain(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    """Multi-step temporal-ordering deduction with red-herrings (depth 9).

    9 events with strict total order; we state 11-13 mixed relations in random
    order, including 2 redundant clues and the "k events between" clue. Per
    lit-review (arXiv 2507.07313), frontier accuracy on depth-9 deductive
    chains with 2+ distractors is <20%.
    """
    events = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    perm = list(rng.permutation(events))  # chronological order
    statements: list[str] = []
    # 4 immediate-after clues
    for _ in range(4):
        i = int(rng.integers(0, len(perm) - 1))
        statements.append(f"{perm[i + 1]} happened immediately after {perm[i]}.")
    # 5 strict-before/after clues
    for _ in range(5):
        i = int(rng.integers(0, len(perm)))
        j = int(rng.integers(0, len(perm)))
        while j == i:
            j = int(rng.integers(0, len(perm)))
        if i < j:
            statements.append(f"{perm[i]} happened before {perm[j]}.")
        else:
            statements.append(f"{perm[j]} happened after {perm[i]}.")
    # 2 "exactly N events strictly between" clues
    for _ in range(2):
        gap = int(rng.integers(2, 5))
        i = int(rng.integers(0, len(perm) - gap))
        statements.append(
            f"There are exactly {gap - 1} events strictly between {perm[i]} and {perm[i + gap]}."
        )
    rng.shuffle(statements)
    k = int(rng.integers(0, len(perm)))
    answer = perm[k]
    prompt = (
        "Nine events labeled A, B, C, D, E, F, G, H, I occurred at distinct "
        "times. Given the following constraints, determine the unique "
        "chronological order and then answer the question.\n\nConstraints:\n"
        + "\n".join(f"  - {s}" for s in statements)
        + f"\n\nQuestion: which event occurred at position {k + 1} "
        + "(counting from the earliest as 1)? Reply with a single letter A–I."
    )
    return prompt, answer, _exact_text(answer)


def _gen_text_latin_square(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    """Fill a 5×5 Latin square: each row and column contains 1..5 exactly once.

    We mask 14 of the 25 cells (leaving 11 hints) and ask for one masked cell.
    Frontier LLMs frequently get individual cells wrong on partial completion.
    """
    # Build a valid 5x5 Latin square (cyclic shift)
    base = [[(i + j) % 5 + 1 for j in range(5)] for i in range(5)]
    # Random row + col permutations
    row_perm = list(rng.permutation(5))
    col_perm = list(rng.permutation(5))
    grid = [[base[row_perm[i]][col_perm[j]] for j in range(5)] for i in range(5)]
    # Mask 14 random cells
    cells = [(r, c) for r in range(5) for c in range(5)]
    rng.shuffle(cells)
    mask_set = set(cells[:14])
    target = cells[0]
    expected = grid[target[0]][target[1]]
    grid_str = "\n".join(
        "  " + " ".join("." if (r, c) in mask_set else str(grid[r][c]) for c in range(5))
        for r in range(5)
    )
    prompt = (
        "Below is a partial 5×5 Latin square: each row and each column must "
        "contain each of {1,2,3,4,5} exactly once. Cells marked '.' are blank.\n\n"
        f"{grid_str}\n\n"
        f"Return the digit that belongs at row {target[0]} column {target[1]} "
        "(0-indexed). Reply with a single digit 1–5."
    )
    return prompt, str(expected), _numeric_match(expected)


GENERATORS: list[tuple[float, Callable[[np.random.Generator], tuple[str, str, Callable[[str], bool]]]]] = [
    (0.10, _gen_t0_sequence),
    (0.20, _gen_t1_substitution),
    (0.30, _gen_t2_recolor),
    (0.45, _gen_t3_deduction),
    (0.60, _gen_t4_transform),
    (0.75, _gen_t5_multistep),
    (0.90, _gen_t6_rule_discovery),
    # T_extreme — target <20% on frontier
    (0.975, _gen_text_latin_square),
    (0.980, _gen_text_temporal_chain),
    (0.990, _gen_text_cryptarithmetic),
    (0.995, _gen_text_zebra),
]


@dataclass(slots=True)
class ReasoningFamily:
    name: str = "reasoning"
    description: str = (
        "Abstract reasoning: sequence completion, ciphers, ARC-AGI-style grid "
        "transforms, deductive puzzles, and rule discovery."
    )

    def generate(
        self,
        n: int,
        seed: int,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
    ) -> list[Task]:
        rng = child_rng(seed, "reasoning.generate")
        lo, hi = difficulty_range
        valid = [(d, g) for d, g in GENERATORS if lo <= d <= hi]
        if not valid:
            valid = GENERATORS
        tasks: list[Task] = []
        for i in range(n):
            d, gen = valid[rng.integers(0, len(valid))]
            sub_rng = np.random.default_rng(child_seed(seed, f"reasoning.{i}"))
            prompt, expected, verifier = gen(sub_rng)
            tasks.append(
                Task(
                    task_id=f"reasoning-{seed}-{i:04d}",
                    family="reasoning",
                    difficulty=float(d),
                    prompt=prompt,
                    verifier=verifier,
                    reference_answer=expected,
                    metadata={"generator": gen.__name__, "tier": d},
                    estimated_seconds=20.0 + 80.0 * d,
                )
            )
        return tasks

    def reference_score(self, task, response):
        return None  # mechanical only


FAMILY = ReasoningFamily
