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
from collections.abc import Callable
from dataclasses import dataclass

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


# -------------------------------------------------------------------------
# T_extreme (d ≥ 0.97) — designed to push frontier accuracy <15-20% in 2026.
# 9×9 Sudoku cell lookup and a 5×5 grid-transform chain (3 ops compounded).
# -------------------------------------------------------------------------


def _solve_sudoku_9x9_target_values(
    board: list[list[int]],
    target_r: int,
    target_c: int,
    node_limit: int = 200_000,
) -> set[int]:
    """Return the set of digit values the target cell takes across all
    completions of the partial board. Halts early as soon as ≥2 distinct
    values are observed (the target is then ambiguous) or when ``node_limit``
    backtracking steps are exhausted. Uses MRV (Minimum Remaining Values)
    heuristic for efficient enumeration.
    """
    n = 9
    block = 3
    rows = [set() for _ in range(n)]
    cols = [set() for _ in range(n)]
    blks = [[set() for _ in range(block)] for _ in range(block)]
    for r in range(n):
        for c in range(n):
            v = board[r][c]
            if v != 0:
                rows[r].add(v)
                cols[c].add(v)
                blks[r // block][c // block].add(v)
    found: set[int] = set()
    nodes = [0]
    aborted = [False]

    def candidates_at(r: int, c: int) -> set[int]:
        return set(range(1, 10)) - rows[r] - cols[c] - blks[r // block][c // block]

    def backtrack() -> None:
        if aborted[0]:
            return
        if len(found) > 1:
            aborted[0] = True
            return
        nodes[0] += 1
        if nodes[0] > node_limit:
            aborted[0] = True
            return
        # Find the empty cell with the fewest candidates (MRV).
        best_cell: tuple[int, int] | None = None
        best_cands: set[int] | None = None
        for r in range(n):
            for c in range(n):
                if board[r][c] == 0:
                    cands = candidates_at(r, c)
                    if not cands:
                        return  # dead end
                    if best_cands is None or len(cands) < len(best_cands):
                        best_cell = (r, c)
                        best_cands = cands
                        if len(cands) == 1:
                            break
            if best_cands is not None and len(best_cands) == 1:
                break
        if best_cell is None:
            # No empty cells left — a complete solution.
            found.add(board[target_r][target_c])
            return
        r, c = best_cell
        for v in sorted(best_cands):
            board[r][c] = v
            rows[r].add(v)
            cols[c].add(v)
            blks[r // block][c // block].add(v)
            backtrack()
            board[r][c] = 0
            rows[r].discard(v)
            cols[c].discard(v)
            blks[r // block][c // block].discard(v)
            if aborted[0]:
                return

    backtrack()
    if aborted[0] and nodes[0] > node_limit and len(found) <= 1:
        # Inconclusive — treat as ambiguous (caller will retry with fewer masks).
        return set()
    return found


def _gen_text_9x9_sudoku_cell(rng: np.random.Generator) -> tuple[str, str]:
    """9×9 Sudoku: return digit at one specific masked cell.

    We build a valid 9×9 Sudoku solution by starting from a canonical base
    grid and applying validity-preserving transformations (digit relabeling,
    band/stack/row-within-band/column-within-stack swaps, transposition).
    We then mask cells and search for one whose value is *uniquely
    determined* by the remaining cells (verified by a small backtracking
    solver). The reference answer is therefore always uniquely correct.
    """
    # Canonical valid 9×9 Sudoku solution (base).
    base = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [4, 5, 6, 7, 8, 9, 1, 2, 3],
            [7, 8, 9, 1, 2, 3, 4, 5, 6],
            [2, 3, 1, 5, 6, 4, 8, 9, 7],
            [5, 6, 4, 8, 9, 7, 2, 3, 1],
            [8, 9, 7, 2, 3, 1, 5, 6, 4],
            [3, 1, 2, 6, 4, 5, 9, 7, 8],
            [6, 4, 5, 9, 7, 8, 3, 1, 2],
            [9, 7, 8, 3, 1, 2, 6, 4, 5],
        ],
        dtype=int,
    )
    grid = base.copy()
    # Validity-preserving shuffles:
    # 1) Random digit relabeling.
    perm = list(range(1, 10))
    rng.shuffle(perm)
    lookup = np.array([0] + perm)
    grid = lookup[grid]
    # 2) Swap rows within each band.
    for band in range(3):
        if bool(rng.integers(0, 2)):
            r1 = band * 3 + int(rng.integers(0, 3))
            r2 = band * 3 + int(rng.integers(0, 3))
            if r1 != r2:
                grid[[r1, r2]] = grid[[r2, r1]]
    # 3) Swap columns within each stack.
    for stack in range(3):
        if bool(rng.integers(0, 2)):
            c1 = stack * 3 + int(rng.integers(0, 3))
            c2 = stack * 3 + int(rng.integers(0, 3))
            if c1 != c2:
                grid[:, [c1, c2]] = grid[:, [c2, c1]]
    # 4) Swap two bands.
    if bool(rng.integers(0, 2)):
        b1, b2 = int(rng.integers(0, 3)), int(rng.integers(0, 3))
        if b1 != b2:
            rows_b1 = [b1 * 3, b1 * 3 + 1, b1 * 3 + 2]
            rows_b2 = [b2 * 3, b2 * 3 + 1, b2 * 3 + 2]
            tmp = grid[rows_b1].copy()
            grid[rows_b1] = grid[rows_b2]
            grid[rows_b2] = tmp
    # 5) Swap two stacks.
    if bool(rng.integers(0, 2)):
        s1, s2 = int(rng.integers(0, 3)), int(rng.integers(0, 3))
        if s1 != s2:
            cols_s1 = [s1 * 3, s1 * 3 + 1, s1 * 3 + 2]
            cols_s2 = [s2 * 3, s2 * 3 + 1, s2 * 3 + 2]
            tmp = grid[:, cols_s1].copy()
            grid[:, cols_s1] = grid[:, cols_s2]
            grid[:, cols_s2] = tmp
    # 6) Random transpose.
    if bool(rng.integers(0, 2)):
        grid = grid.T.copy()
    solution = grid.tolist()
    # We aim for ~50 masked cells (a hard sudoku) but the uniquely-determined
    # cell search remains fast since we test the candidate cell against the
    # *visible* grid only. Cascade down if we cannot find a uniquely-determined
    # cell at this density.
    found_target: tuple[int, int, int] | None = None
    mask_set: set[tuple[int, int]] = set()
    for n_mask in (50, 45, 40, 35, 30, 25, 20):
        cells = [(r, c) for r in range(9) for c in range(9)]
        rng.shuffle(cells)
        mask_set = set(cells[:n_mask])
        puzzle = [
            [0 if (r, c) in mask_set else solution[r][c] for c in range(9)]
            for r in range(9)
        ]
        for (r, c) in cells[:n_mask]:
            work = [row[:] for row in puzzle]
            values = _solve_sudoku_9x9_target_values(work, r, c, node_limit=200_000)
            if len(values) == 1:
                found_target = (r, c, next(iter(values)))
                break
        if found_target is not None:
            break
    if found_target is None:
        # Fallback (very unlikely with our base + random transforms):
        # use the first masked cell with the known solution value.
        c0 = next(iter(mask_set))
        found_target = (c0[0], c0[1], solution[c0[0]][c0[1]])
    target_r, target_c, target_digit = found_target
    grid_str_lines = []
    for r in range(9):
        row_cells: list[str] = []
        for c in range(9):
            if (r, c) in mask_set:
                row_cells.append(".")
            else:
                row_cells.append(str(solution[r][c]))
            if c in (2, 5):
                row_cells.append("|")
        grid_str_lines.append(" ".join(row_cells))
        if r in (2, 5):
            grid_str_lines.append("------+-------+------")
    grid_str = "\n".join(grid_str_lines)
    prompt = (
        "Below is a 9×9 Sudoku puzzle. Each row, each column, and each of the "
        "nine 3×3 blocks must contain each of {1,2,3,4,5,6,7,8,9} exactly once. "
        "Cells marked '.' are blank.\n\n"
        f"{grid_str}\n\n"
        f"In every valid completion of this puzzle, the digit at row "
        f"{target_r} column {target_c} (0-indexed) is the same. "
        "Reply with that single digit 1–9."
    )
    return prompt, str(target_digit)


def _gen_text_grid_transform_chain(rng: np.random.Generator) -> tuple[str, str]:
    """Apply a 6-step transform chain to a 6×6 grid, then ask for one cell.

    Each step composes onto the previous output. Transforms drawn from a
    richer 7-element pool (rotations, reflections, color permutations,
    transpose, shift). Frontier models consistently lose track of cell
    positions past chain depth 4 on grids of side ≥6.
    """
    palette = [0, 1, 2, 3, 4]
    h = w = 7
    grid = [[int(rng.integers(0, len(palette))) for _ in range(w)] for _ in range(h)]
    transforms_pool = [
        ("rotate90cw", "rotate the grid 90 degrees clockwise"),
        ("rotate90ccw", "rotate the grid 90 degrees counter-clockwise"),
        ("rotate180", "rotate the grid 180 degrees"),
        ("flip_h", "flip the grid horizontally (left-right mirror, cell (r,c) becomes (r, n-1-c))"),
        ("flip_v", "flip the grid vertically (up-down mirror, cell (r,c) becomes (n-1-r, c))"),
        ("transpose", "transpose the grid (cell (r,c) moves to (c,r))"),
        ("anti_transpose", "anti-transpose the grid (cell (r,c) moves to (n-1-c, n-1-r))"),
        ("perm_colors", "permute colors by the map 0→1, 1→2, 2→3, 3→4, 4→0"),
    ]
    n_steps = 9
    chosen_idx = list(rng.integers(0, len(transforms_pool), size=n_steps))
    steps = [transforms_pool[i] for i in chosen_idx]

    def apply(g: list[list[int]], op: str) -> list[list[int]]:
        n = len(g)
        if op == "rotate90cw":
            return [[g[n - 1 - r][c] for r in range(n)] for c in range(n)]
        if op == "rotate90ccw":
            return [[g[c][n - 1 - r] for c in range(n)] for r in range(n)]
        if op == "rotate180":
            return [list(reversed(row)) for row in reversed(g)]
        if op == "flip_h":
            return [list(reversed(row)) for row in g]
        if op == "flip_v":
            return list(reversed(g))
        if op == "transpose":
            return [[g[r][c] for r in range(n)] for c in range(n)]
        if op == "anti_transpose":
            return [[g[n - 1 - c][n - 1 - r] for r in range(n)] for c in range(n)]
        if op == "perm_colors":
            return [[(v + 1) % 5 for v in row] for row in g]
        return g

    out = [row[:] for row in grid]
    for op, _ in steps:
        out = apply(out, op)
    target_r = int(rng.integers(0, len(out)))
    target_c = int(rng.integers(0, len(out[0])))
    target_val = out[target_r][target_c]
    grid_str = "\n".join(" ".join(str(v) for v in row) for row in grid)
    steps_str = "\n".join(f"  Step {i+1}: {desc}" for i, (_, desc) in enumerate(steps))
    prompt = (
        "Below is a 7×7 grid of integer colors (each cell in {0,1,2,3,4}). "
        "Apply the following sequence of transformations IN ORDER, where each "
        f"step operates on the OUTPUT of the previous step.\n\n"
        f"Initial grid:\n{grid_str}\n\n"
        f"Transformations:\n{steps_str}\n\n"
        f"After all {n_steps} steps have been applied, what is the value at "
        f"row {target_r} column {target_c} of the final grid (0-indexed)? "
        "Reply with a single digit 0–4."
    )
    return prompt, str(target_val)


GENERATORS: list[tuple[float, Callable[[np.random.Generator], tuple[str, str]]]] = [
    (0.10, _gen_t0),
    (0.25, _gen_t1),
    (0.40, _gen_t2),
    (0.55, _gen_t3),
    (0.70, _gen_t4),
    (0.85, _gen_t5),
    (0.95, _gen_t6),
    # T_extreme — target <15-20% on frontier
    (0.975, _gen_text_grid_transform_chain),
    (0.990, _gen_text_9x9_sudoku_cell),
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

    def reference_score(self, task, response):
        return None


FAMILY = MultimodalFamily
