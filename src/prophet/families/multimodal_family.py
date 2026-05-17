"""Multimodal family — text-only encodings of visual-style reasoning tasks.

Because PROPHET must score text-only models too, we encode "images" as
ASCII art, structured grids, and tables. Each tier exercises a distinct
visual-reasoning sub-skill: chart reading, counting, rotation, board-state
recognition, pattern symmetry, and Sudoku-style propagation.

Difficulty tiers:
  T0 (0.10) — read an ASCII bar chart, return the label of the tallest bar.
  T1 (0.25) — count objects of a given character in an ASCII grid.
  T2 (0.40) — rotate a 4×4 grid 90° clockwise, return one specific cell.
  T3 (0.55) — chess board: given pieces in algebraic notation, is white's
              king in check? (yes/no).
  T4 (0.70) — tic-tac-toe winner detection on a 3×3 grid.
  T5 (0.85) — dot-pattern symmetry detection (horizontal / vertical / none).
  T6 (0.95) — 4×4 Sudoku — return the digit at the requested cell.

The reference answer is always a single token (letter / int / yes-no /
'horizontal'-'vertical'-'none'). The verifier uses tolerant string match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from prophet.engine.types import Task
from prophet.utils.seed import child_rng, child_seed


def _match_exact(expected: str) -> Callable[[str], bool]:
    """Case-insensitive substring/last-token match for short answers."""
    norm_exp = str(expected).strip().lower()

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        s = answer.strip().lower()
        s = re.sub(r"[*`\"'`]", "", s)
        s = s.rstrip(".,;:!?")
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if lines:
            s = lines[-1]
        if s == norm_exp:
            return True
        # also accept "the answer is X" / boxed
        m = re.search(r"answer[^a-z0-9]*([a-z0-9\-]+)", s)
        if m and m.group(1) == norm_exp:
            return True
        toks = re.findall(r"[a-z0-9\-]+", s)
        return bool(toks) and toks[-1] == norm_exp

    return _verify


# ---------------------------------------------------------------------------
# Tier generators
# ---------------------------------------------------------------------------


def _gen_t0(rng: np.random.Generator) -> tuple[str, str]:
    labels = list(rng.choice(list("ABCDEF"), size=int(rng.integers(3, 6)), replace=False))
    heights = [int(rng.integers(1, 12)) for _ in labels]
    # Ensure a unique max
    while heights.count(max(heights)) > 1:
        i = int(rng.integers(0, len(heights)))
        heights[i] = int(rng.integers(1, 12))
    max_h = max(heights)
    rows = []
    for h in range(max_h, 0, -1):
        rows.append("  ".join("##" if heights[i] >= h else "  " for i in range(len(labels))))
    rows.append("  ".join(labels))
    chart = "\n".join(rows)
    expected = labels[heights.index(max_h)]
    prompt = (
        "Below is an ASCII bar chart where each '##' is one unit of height.\n"
        "Return the LABEL of the tallest bar (single letter).\n\n"
        f"{chart}"
    )
    return prompt, expected


def _gen_t1(rng: np.random.Generator) -> tuple[str, str]:
    h, w = int(rng.integers(4, 8)), int(rng.integers(4, 8))
    target = str(rng.choice(["*", "o", "x"]))
    other = "." if target != "." else ","
    grid = []
    count = 0
    for _ in range(h):
        row = []
        for _ in range(w):
            ch = target if rng.random() < 0.3 else other
            if ch == target:
                count += 1
            row.append(ch)
        grid.append(" ".join(row))
    body = "\n".join(grid)
    return (
        f"Count the occurrences of the character {target!r} in this {h}×{w} grid. "
        "Return only the integer.\n\n" + body,
        str(count),
    )


def _gen_t2(rng: np.random.Generator) -> tuple[str, str]:
    chars = list("ABCDEFGHIJKLMNOP")
    rng.shuffle(chars)
    grid = [chars[i * 4 : (i + 1) * 4] for i in range(4)]
    # Rotate 90° CW: new[i][j] = old[n-1-j][i]; here we compute target cell
    row = int(rng.integers(0, 4))
    col = int(rng.integers(0, 4))
    rotated_cell = grid[3 - col][row]
    grid_str = "\n".join(" ".join(r) for r in grid)
    return (
        "Consider this 4×4 grid of letters. Rotate it 90° CLOCKWISE.\n"
        f"After rotation, what character is at row {row} column {col} (0-indexed)?\n\n"
        f"{grid_str}",
        rotated_cell,
    )


def _gen_t3(rng: np.random.Generator) -> tuple[str, str]:
    # Chess board: place the white king, then either a queen/rook giving check
    # or a knight not giving check. Deterministic procedure with explicit
    # geometry to keep answers verifiable.
    files = "abcdefgh"
    kf, kr = int(rng.integers(0, 8)), int(rng.integers(0, 8))
    king_sq = f"{files[kf]}{kr+1}"
    want_check = bool(int(rng.integers(0, 2)))
    pieces: list[str] = [f"K{king_sq}"]
    if want_check:
        # Place a black rook on same rank or file at a square that has a clear
        # line of attack (since we have no other pieces on the board).
        same_rank = bool(int(rng.integers(0, 2)))
        if same_rank:
            f2 = int(rng.integers(0, 8))
            while f2 == kf:
                f2 = int(rng.integers(0, 8))
            pieces.append(f"r{files[f2]}{kr+1}")
        else:
            r2 = int(rng.integers(0, 8))
            while r2 == kr:
                r2 = int(rng.integers(0, 8))
            pieces.append(f"r{files[kf]}{r2+1}")
        expected = "yes"
    else:
        # Place a black knight far away — but knights attack in an L-shape, so
        # avoid the 8 knight-move squares around the king.
        knight_offsets = {(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)}
        for _ in range(50):
            f2, r2 = int(rng.integers(0, 8)), int(rng.integers(0, 8))
            if (f2, r2) == (kf, kr):
                continue
            if (f2 - kf, r2 - kr) in knight_offsets:
                continue
            # Also avoid same rank / file / diagonal so we don't accidentally
            # give check via the knight's presence.
            if f2 == kf or r2 == kr or abs(f2 - kf) == abs(r2 - kr):
                continue
            pieces.append(f"n{files[f2]}{r2+1}")
            break
        else:
            # safety: degenerate to far corner knight
            pieces.append("nh8" if king_sq != "h8" else "na1")
        expected = "no"
    return (
        "Below is a chess position. White has one king; black has one other "
        "piece (uppercase = white, lowercase = black). Is white's king in "
        "check? Answer 'yes' or 'no'.\n\n"
        f"Pieces: {', '.join(pieces)}",
        expected,
    )


def _gen_t4(rng: np.random.Generator) -> tuple[str, str]:
    # Build a deterministic finished tic-tac-toe game.
    outcome = int(rng.integers(0, 3))  # 0 = X wins, 1 = O wins, 2 = draw
    board = [[" "] * 3 for _ in range(3)]
    if outcome == 0:
        line = int(rng.integers(0, 8))
        if line < 3:
            for c in range(3):
                board[line][c] = "X"
        elif line < 6:
            for r in range(3):
                board[r][line - 3] = "X"
        elif line == 6:
            for k in range(3):
                board[k][k] = "X"
        else:
            for k in range(3):
                board[k][2 - k] = "X"
        # scatter some Os in the remaining cells (not enough to win)
        empty = [(r, c) for r in range(3) for c in range(3) if board[r][c] == " "]
        for r, c in empty[:2]:
            board[r][c] = "O"
        expected = "x"
    elif outcome == 1:
        line = int(rng.integers(0, 8))
        if line < 3:
            for c in range(3):
                board[line][c] = "O"
        elif line < 6:
            for r in range(3):
                board[r][line - 3] = "O"
        elif line == 6:
            for k in range(3):
                board[k][k] = "O"
        else:
            for k in range(3):
                board[k][2 - k] = "O"
        empty = [(r, c) for r in range(3) for c in range(3) if board[r][c] == " "]
        for r, c in empty[:2]:
            board[r][c] = "X"
        expected = "o"
    else:
        # Specific known draw pattern, scrambled slightly with rotation.
        base = [["X", "O", "X"], ["X", "O", "O"], ["O", "X", "X"]]
        rot = int(rng.integers(0, 4))
        for _ in range(rot):
            base = [list(r) for r in zip(*base[::-1])]
        board = base
        expected = "draw"
    grid = "\n".join("|".join(row) for row in board)
    return (
        "Below is a tic-tac-toe board (3 rows, cells separated by '|').\n"
        "Return 'X' if X has three in a row, 'O' if O does, else 'draw'.\n\n"
        f"{grid}",
        expected,
    )


def _gen_t5(rng: np.random.Generator) -> tuple[str, str]:
    n = 5
    grid = [["." for _ in range(n)] for _ in range(n)]
    sym = int(rng.integers(0, 3))  # 0 horiz, 1 vert, 2 none
    if sym == 0:
        # build top half then mirror to bottom
        for r in range(n // 2 + 1):
            for c in range(n):
                if rng.random() < 0.35:
                    grid[r][c] = "*"
        for r in range(n // 2):
            grid[n - 1 - r] = list(grid[r])
        expected = "horizontal"
    elif sym == 1:
        for r in range(n):
            for c in range(n // 2 + 1):
                if rng.random() < 0.35:
                    grid[r][c] = "*"
        for r in range(n):
            for c in range(n // 2):
                grid[r][n - 1 - c] = grid[r][c]
        expected = "vertical"
    else:
        # asymmetric pattern: make one half random and inject an asymmetric dot
        for r in range(n):
            for c in range(n):
                if rng.random() < 0.35:
                    grid[r][c] = "*"
        # force asymmetry
        grid[0][0] = "*"
        grid[0][n - 1] = "."
        grid[n - 1][0] = "."
        expected = "none"
    body = "\n".join(" ".join(row) for row in grid)
    return (
        "Below is a 5×5 dot pattern. Return 'horizontal' if it has horizontal "
        "(top-bottom) mirror symmetry, 'vertical' if it has left-right mirror "
        "symmetry, or 'none' if neither.\n\n" + body,
        expected,
    )


def _solve_sudoku_4x4(board: list[list[int]]) -> list[list[int]] | None:
    def backtrack() -> bool:
        for r in range(4):
            for c in range(4):
                if board[r][c] == 0:
                    used: set[int] = set()
                    for k in range(4):
                        used.add(board[r][k])
                        used.add(board[k][c])
                    br, bc = (r // 2) * 2, (c // 2) * 2
                    for dr in range(2):
                        for dc in range(2):
                            used.add(board[br + dr][bc + dc])
                    for v in (1, 2, 3, 4):
                        if v not in used:
                            board[r][c] = v
                            if backtrack():
                                return True
                            board[r][c] = 0
                    return False
        return True

    bcopy = [row[:] for row in board]
    if backtrack():
        return board
    return bcopy if any(0 in r for r in board) else board


def _gen_t6(rng: np.random.Generator) -> tuple[str, str]:
    # 4×4 Sudoku — values 1..4, each row/col/2×2 block has each digit once.
    # Build a valid grid by starting from a known Latin square and shuffling.
    solution = np.array([[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]])
    # Swap pairs within each band/stack to keep validity.
    for _ in range(3):
        if rng.random() < 0.5:
            i = int(rng.integers(0, 2)) * 2
            solution[[i, i + 1]] = solution[[i + 1, i]]
        if rng.random() < 0.5:
            i = int(rng.integers(0, 2)) * 2
            solution[:, [i, i + 1]] = solution[:, [i + 1, i]]
    # Permute digits (a relabeling — still a valid solution).
    perm = list(range(1, 5))
    rng.shuffle(perm)
    relabeled = np.vectorize(lambda v: perm[v - 1])(solution).tolist()
    # Mask some cells to make a puzzle (procgen: mask exactly 6 cells but keep
    # uniquely solvable enough for verification; we just hide six random ones).
    puzzle = [row[:] for row in relabeled]
    cells = [(r, c) for r in range(4) for c in range(4)]
    rng.shuffle(cells)
    for r, c in cells[:6]:
        puzzle[r][c] = 0
    target_r, target_c = cells[0]
    target_digit = relabeled[target_r][target_c]
    grid_str = "\n".join(
        " ".join("." if v == 0 else str(v) for v in row) for row in puzzle
    )
    return (
        "Below is a 4×4 Sudoku puzzle: each row, each column, and each 2×2 "
        "block must contain each of 1,2,3,4 exactly once. '.' marks a blank.\n"
        f"Return the digit that belongs at row {target_r} column {target_c} "
        "(0-indexed).\n\n"
        f"{grid_str}",
        str(target_digit),
    )


GENERATORS: list[tuple[float, Callable[[np.random.Generator], tuple[str, str]]]] = [
    (0.10, _gen_t0),
    (0.25, _gen_t1),
    (0.40, _gen_t2),
    (0.55, _gen_t3),
    (0.70, _gen_t4),
    (0.85, _gen_t5),
    (0.95, _gen_t6),
]


@dataclass(slots=True)
class MultimodalFamily:
    name: str = "multimodal"
    description: str = (
        "Text-only encodings of visual reasoning tasks: ASCII bar charts, "
        "grids, chess positions, tic-tac-toe boards, symmetry patterns, and "
        "4×4 Sudoku. Mechanical verifier — exact match on a single token."
    )

    def generate(
        self,
        n: int,
        seed: int,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
    ) -> list[Task]:
        rng = child_rng(seed, "multimodal.generate")
        lo, hi = difficulty_range
        valid = [(d, g) for d, g in GENERATORS if lo <= d <= hi]
        if not valid:
            valid = GENERATORS
        tasks: list[Task] = []
        for i in range(n):
            d, gen = valid[rng.integers(0, len(valid))]
            sub = np.random.default_rng(child_seed(seed, f"multimodal.{i}"))
            prompt, expected = gen(sub)
            tasks.append(
                Task(
                    task_id=f"multimodal-{seed}-{i:04d}",
                    family="multimodal",
                    difficulty=float(d),
                    prompt=prompt,
                    verifier=_match_exact(expected),
                    reference_answer=expected,
                    metadata={"generator": gen.__name__, "tier": d},
                    estimated_seconds=20.0 + 60.0 * d,
                )
            )
        return tasks

    def reference_score(self, task, response):  # noqa: ANN001
        return None


FAMILY = MultimodalFamily
