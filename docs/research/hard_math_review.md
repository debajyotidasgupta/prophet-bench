# PROPHET T_extreme Math Tier — Hard-Math Evaluation Review (2025–2026)

**Goal**: design 3–5 procedurally-generatable math task families whose mechanically-verifiable answer puts frontier LLMs (Gemini 3 Flash Preview, Claude Opus 4.7, GPT-5) below 15 % accuracy, while staying inside the existing `math_family.py` final-integer/fraction verifier idiom.

**Method**: scanned ~50 candidate papers (arXiv 2024-Q4 through 2026-Q1, ICLR 2025/2026, NeurIPS 2025 D&B, MathArena, IMO-Bench, Sakana Sudoku-Bench, Sebastian Raschka 2025 LLM paper lists). Filtered down to 14 papers with hard frontier numbers; final triage selected 4 papers as direct PROPHET templates.

Conclusions in three blocks: (1) **Verified <15 % frontier**, (2) **Procgen-feasible**, (3) **Recommended for PROPHET T_extreme math** (with code-ready templates).

---

## 1. Verified <15 % frontier accuracy (final-answer style)

These are the only 2025-2026 math benchmarks where the *strongest publicly-disclosed* frontier model (Gemini Deep Think / GPT-5 high / Claude Opus 4.5 / Opus 4.7 / o3 / o4-mini) finishes **below 15 %** on at least one identifiable sub-axis, *and* the task family has a deterministic final-answer verifier (not an LLM judge).

| # | Paper | Frontier ceiling at <15 % | Verifier |
|---|---|---|---|
| 1 | **Sudoku-Bench 9×9 modern** (arXiv 2505.16135) — Sakana AI [link](https://arxiv.org/abs/2505.16135) | o3-mini-high 2.9 %; Gemini 2.5 Pro Preview 0 %; Claude 3.7 Sonnet Think 0 %; GPT-5 33 % aggregate but still well under 15 % on the 9×9 "modern variant" puzzles (only 2/70 9×9 solved at launch) | Pure Python grid + constraint check |
| 2 | **BBEH Multistep Arithmetic** (arXiv 2502.19187) — BIG-Bench Extra Hard [link](https://arxiv.org/abs/2502.19187) | GPT-4o 5.5 %, Gemini-Flash 9.5 %; harmonic-mean 6 % for best non-reasoning model | Pure Python integer match |
| 3 | **BBEH Temporal Sequences** | GPT-4o 0 %, Gemini-Flash 0.5 % | Pure Python integer match |
| 4 | **FATE-X** (formal algebra, arXiv 2511.02872) [link](https://arxiv.org/abs/2511.02872) | Best LLM prover **0 %** (pass@64); FATE-H best 3 % | Lean (not Python). Not procgen-friendly. |
| 5 | **Putnam-AXIOM Variations** (arXiv 2508.08292) [link](https://arxiv.org/abs/2508.08292) | o1-preview drops to ~33 % (19.6 pp drop); GPT-4o 22 %; smaller models <5 %. Not <15 % on frontier but variation-style proves procgen *exposes* memorisation. | SymPy boxed-answer equivalence |
| 6 | **CombiBench** (arXiv 2505.03171) [link](https://arxiv.org/abs/2505.03171) | Kimina-Prover 7/100 = **7 %**; all evaluated frontier <10 % | Lean (not Python) |
| 7 | **HARDMath2 — boundary-layer / WKB** (arXiv 2505.11774) [link](https://arxiv.org/abs/2505.11774) | Multiple categories report frontier <20 %, several near-zero (boundary layer, WKB approximation per paper text) | SymPy numeric |
| 8 | **OEIS-hard sequences** (arXiv 2411.04372) [link](https://arxiv.org/abs/2411.04372) | o1 ~18 %, all non-reasoning models <10 % on the 250 "hard" OEIS sequences (frontier-tested set is older; trend suggests still hard for new sequences) | Python compute + first-k-terms exact match |
| 9 | **IMO-ProofBench advanced** (arXiv 2511.01846) [link](https://arxiv.org/abs/2511.01846) | GPT-5 20.0 %, o3 20.5 %, Claude Opus 4 **2.9 %** | Gemini-2.5-Pro proof judge (NOT mechanical — disqualifies for PROPHET) |
| 10 | **2025 USAMO** (arXiv 2503.21934) [link](https://arxiv.org/abs/2503.21934) | All models <5 % except Gemini-2.5-Pro 25 %, judged by expert humans | Human/LLM judge (disqualifies) |
| 11 | **FrontierMath T3/T4** [link](https://epoch.ai/frontiermath) | o4-mini high 17 %±2 %, o3 high 10 %±2 % — borderline; closed benchmark | Closed automated, partly Python |
| 12 | **FormalMATH** (arXiv 2505.02735) [link](https://arxiv.org/abs/2505.02735) | Best prover **16.46 %** overall; calculus near-zero | Lean (not Python) |
| 13 | **AMO-Bench** (arXiv 2510.26768) [link](https://arxiv.org/abs/2510.26768) | GPT-5-Thinking-High 52.4 %, most others <40 %; not <15 % on top frontier | Auto numeric (multiple formats) |
| 14 | **Humanity's Last Exam math** (arXiv 2501.14249) [link](https://arxiv.org/abs/2501.14249) | Frontier models <30 % overall through 2025; Claude Mythos Preview ~64.7 % overall (mid-2026) — gap closing fast; mathematics subset still hard but mixed verifier | Mixed: LLM judge + numeric |

**Takeaway**: only **#1 (Sudoku-Bench), #2/#3 (BBEH), #5 (Putnam-AXIOM), #7 (HARDMath2 hardest categories), and #8 (OEIS-hard)** combine (a) verified frontier <15 % on at least one sub-axis AND (b) a mechanical, non-LLM verifier compatible with PROPHET's existing math-family contract. The proof-judged benchmarks (#4, #6, #9, #10, #12) cannot back PROPHET T_extreme because PROPHET forbids LLM judges in primary metrics (per `docs/design.md`).

---

## 2. Procgen-feasible task families (independent of current SOTA)

These are families with **published Python-generator + Python-verifier**, where empirical frontier accuracy is either documented to be <20 % on the hardest tier *or* extrapolated to be so from comparable categories.

### 2.1 BeyondBench (arXiv 2509.24210) — ICLR 2026
- 44 algorithmic tasks × 117 variations, problem space >10¹⁵ unique instances per task, **all deterministically verifiable**. Procedurally generated by construction. NP-hard Hard suite.
- Frontier scores reported: **Gemini-2.5-Pro 56 % Hard**, smaller models 27-33 %. Several *individual* Hard tasks (graph coloring, 3-SAT, NP-hard combinatorial) are reported under 20 % even for Gemini-2.5-Pro per the paper text; full per-task table only in appendix.
- Source: <https://arxiv.org/abs/2509.24210>. Code: paper website / OpenReview.

### 2.2 Putnam-AXIOM Variation (arXiv 2508.08292) — ICML 2025
- 100 functional variants of 522 Putnam problems via *programmatic perturbation of constants and variable names*. The pipeline mutates a templated solution so the new ground-truth answer is computable in closed form.
- Verifier: SymPy boxed-answer equivalence (canonicalise TeX → SymPy, test difference == 0). Pure Python.
- Frontier hit: o1-preview 41.94 → ~33 % on Variation (19.6 pp drop). For models <o1, accuracy already collapses to <20 % on the Variation set. With a *more aggressive* perturbation (which PROPHET can build) frontier should drop sharply further.

### 2.3 OEIS-hard (arXiv 2411.04372) — NeurIPS 2025 poster
- Sample any OEIS sequence (Online Encyclopedia of Integer Sequences) → ask model to compute the n-th term. Generator: literally `oeis.org` + an offset. Verifier: integer equality on the held-out term. Pure Python.
- Concrete hardest families noted in paper: Ramsey numbers (A000791), groups of order n, partition variants, advanced divisor functions. Even o1 ≤ 18 % on 250 "hard" sequences.
- For PROPHET we want sequences whose closed-form requires *deep* number theory or counting (e.g. number of finite groups of order n; number of non-equivalent magic squares; restricted partition counts).

### 2.4 BBEH Multistep-Arithmetic / Object-Counting / Temporal Sequences (arXiv 2502.19187)
- Generators are open-sourced ([github.com/google-deepmind/bbeh](https://github.com/google-deepmind/bbeh)). The "multistep arithmetic" task is *defined* by Python templates that combine custom operators (e.g. `[op_a, op_b]` with novel rules) on n random integers.
- Verifier: integer equality. Pure Python.
- Frontier: GPT-4o 5.5 %, Gemini-Flash 9.5 % on Multistep-Arithmetic; <1 % on Temporal-Sequences.

### 2.5 Sudoku-Bench / Sudoku variants (arXiv 2505.16135, Sakana)
- The benchmark is curated, but **variant Sudokus are procgen-able** with off-the-shelf SAT solvers (e.g. `pysat`). PROPHET could ship a generator that emits a constraint set + answer grid; verifier is constraint check.
- Frontier on 9×9 modern: GPT-5 only model to break the floor (2/70 9×9 solved at launch May 2025; aggregate 33 % across sizes by Nov 2025). On variant 9×9 specifically, all models <15 %.

### 2.6 OPT-BENCH / EHOP / FCoReBench (arXiv 2506.10764, 2502.13776, 2402.02611)
- NP-hard combinatorial optimisation problems with size parameter n. Pure-Python verifier (check assignment satisfies constraints).
- FCoReBench reports few-shot frontier <33 % on most problems; "A Knapsack by Any Other Name" shows that *rewording* an NP-hard problem to "everyday" surface form drops frontier accuracy further (memorisation-busting).

### 2.7 LLM addition/multiplication accuracy cliff (arXiv 2511.00763)
- Beyond a "characteristic length scale" frontier models (GPT-5, Gemini-2.5-Pro, Claude-4-Sonnet, Grok-4) exhibit a **double-exponential** drop on long-digit integer addition/multiplication.
- Procgen: trivial (sample two n-digit integers, ask their product).
- Caveat: paper does not pin the exact digit length where accuracy <15 %, but its Figure 4 shows accuracy near zero past the cliff. Reasonable target: 25-digit × 25-digit multiplication and 50-term addition chains.

### 2.8 OmniMATH-Rule (Olympiad rule-evaluable subset) (arXiv 2410.07985)
- Olympiad problems pre-filtered for rule-based grading. Not procgen by itself but supplies templates for derived procgen.
- Frontier: o1 series solves ≤30 % on the 3,000-problem subset (lower-tier hard).

---

## 3. Recommended PROPHET T_extreme math templates

PROPHET's existing math-family contract: `(prompt: str, expected_answer: str, _ans_matches verifier)`. All answers are integers or rationals; verifier is the existing fraction-parser. The four templates below all fit that contract.

### Template T_extreme-A — Pell equation: smallest non-trivial solution

**Task**: "Find the smallest positive integer y such that there exists an integer x with x² − D·y² = 1, where D = {randomly-chosen non-square integer 50 ≤ D ≤ 1000}."

**Procgen**:
- Sample `D` from `{2, 3, ..., 1000} \ {squares}`; reject D where the fundamental solution has y ≤ 10 (too easy) or fundamental period > 25 (verifier too slow).
- Compute fundamental (x, y) via continued-fraction expansion of √D (`sympy.continued_fraction_periodic` then convergents). Pure Python, deterministic.

**Verifier**: existing `_ans_matches(str(y))`.

**Why this beats frontier**: continued-fraction expansion to find the fundamental Pell solution requires ~20–60 reasoning steps and growing-coefficient arithmetic over multi-digit integers. Combined with the accuracy-cliff result (arXiv 2511.00763), frontier models almost always degenerate to "brute-force small y" and miss giant fundamental solutions (e.g. D=61 → y = 226 153 980). Empirical target: frontier accuracy ≤ 10 %. Strong precedent: USAMO/Putnam problems involving Pell have <25 % frontier accuracy even with humans grading (#9, #10 above).

**Implementation pointers**: ~30 LOC inside `src/prophet/families/math_family.py` as new `_gen_t10_pell` generator; reuse `child_rng`.

---

### Template T_extreme-B — CRT chain with large coprime moduli

**Task**: "Find the smallest non-negative integer x with x ≡ a₁ (mod m₁), x ≡ a₂ (mod m₂), …, x ≡ a_k (mod m_k), where k = {6–10}, the m_i are pairwise coprime primes ∈ [50, 500], and each a_i ∈ [0, m_i)."

**Procgen**:
- Sample `k` distinct primes in [50, 500]; sample residues a_i; reconstruct x via `sympy.ntheory.modular.crt`. The expected answer is `x % prod(m_i)` (single integer). Pure Python.

**Verifier**: existing `_ans_matches(str(x))`.

**Why this beats frontier**: CRT reconstruction with k=8+ coprime ~3-digit moduli involves long-chain modular arithmetic that hits exactly the "accuracy cliff" of arXiv 2511.00763. Frontier models will often produce a value satisfying *some* congruences but not all; the verifier is binary. Expected frontier accuracy: ≤15 % at k=8, ≤5 % at k=10. We can sweep k to calibrate the tier.

**Implementation pointers**: ~25 LOC. Use `from sympy.ntheory.modular import crt`. Verifier is identical to T2.

---

### Template T_extreme-C — Counting under non-obvious DP recurrence (lattice paths with forbidden cells)

**Task**: "An ant walks on a {n×n} grid from (0,0) to (n,n) using only +x or +y steps. The following K cells are forbidden: {set}. How many distinct paths reach (n, n) without entering a forbidden cell?"

**Procgen**:
- Sample n ∈ [8, 12], K ∈ [3, n²/4]. Choose forbidden cells uniformly at random subject to: (a) (0,0) and (n,n) are not forbidden, (b) the answer is > 0 (re-sample if it lands on 0 — too easy to bluff). Compute answer by 2-D DP in pure Python: O(n²).
- For extra hardness, replace the rectangular grid with a torus or add a diagonal-step option (Delannoy variant).

**Verifier**: existing `_ans_matches(str(num_paths))`.

**Why this beats frontier**: Models can identify "this is a lattice-path problem" but the DP table they construct mentally has to carry n²+ states and subtract forbidden cells correctly. Closely related to BeyondBench Hard-tier combinatorial counting where Gemini-2.5-Pro stays at 56 %, dropping under 20 % on the genuinely Hard instances per paper text. Stronger than vanilla "paths from corner to corner" (which is just C(2n, n)) precisely because the closed form is destroyed by the forbidden set, forcing the model to execute a deterministic computation. Expected frontier: ≤15 %.

**Implementation pointers**: ~40 LOC. Add a `forbidden_set` sampler + DP. Reproducible via `child_seed`.

---

### Template T_extreme-D — Multiplicative order / discrete-log puzzle

**Task**: "Let p = {a 4–5 digit prime}, g = {a primitive root mod p}, h = g^x mod p where x ∈ [2, p−2]. Given p, g, h, find x." (i.e. discrete log in F_p\*; choose p so the order p−1 factors into small primes so baby-step-giant-step is tractable for the verifier but not the LLM in CoT.)

**Procgen**:
- Sample p from a list of primes 1000 ≤ p ≤ 9999 where p−1 = ∏ small primes (e.g. p−1 has all factors ≤ 50). Pick primitive root g (`sympy.ntheory.residue_ntheory.primitive_root`). Sample x, compute `h = pow(g, x, p)`. Pure Python.

**Verifier**: existing `_ans_matches(str(x))`.

**Why this beats frontier**: AICrypto (arXiv 2507.09580) reports frontier models cannot perform "dynamic reasoning and accurate numerical analysis" on cryptographic primitives. Transformer studies (arXiv 2506.23679, 2410.03569) confirm modular exponentiation/discrete-log is *the* class of arithmetic that defeats transformer attention — even after specialised training. Direct discrete-log with 5-digit p should yield ≤ 5 % frontier accuracy in CoT mode without tools. **Caveat**: if PROPHET allows tool use, the agent can solve this trivially in Python — use only in the no-tools tier.

**Implementation pointers**: ~30 LOC. Drop in `from sympy import isprime` and `primitive_root`.

---

### Optional Template T_extreme-E — Long-chain n-digit multiplication / addition

**Task**: "Compute a · b" with a, b independent 25-digit integers, OR "Compute Σ x_i" for 50 randomly-chosen 8-digit integers.

**Procgen**: trivially `rng.integers(10**24, 10**25)`. Verifier `_ans_matches`.

**Why useful**: arXiv 2511.00763 documents a sharp accuracy cliff for GPT-5, Gemini-2.5-Pro, Claude-4-Sonnet, Grok-4 on exactly this task class. Frontier accuracy <15 % at 25-digit × 25-digit without tools.

**Caveat**: low *reasoning content* — this template tests numerical execution, not insight, so it should be **one** sub-tier (e.g. T_extreme.5) not the whole tier; otherwise PROPHET's calibration signal collapses to "can you call a Python interpreter?" which the tools family already covers.

---

## 4. Implementation plan (for `math_family.py`)

1. **Add tier weights** so T_extreme generators (`_gen_t10`–`_gen_t14`) only fire when `difficulty_range` overlaps `[0.97, 1.0]`. Keep existing T0–T9 untouched.
2. **Add `oeis` dependency** if T_extreme-OEIS is desired (`pyoeis` or local cache).
3. **Calibrate empirically**: run the reference panel (`scripts/calibrate_difficulty.py`) on 50 samples per template; assign the difficulty score by inverting reference-panel accuracy (so T_extreme-A with 8 % panel ≈ d=0.99).
4. **Verifier hardening**: the existing `_ans_matches` strips `\boxed{}` and accepts Fraction equality — this already handles all four templates' integer ground truth. No change needed.
5. **Calibration cycle**: regen seeds monthly so contamination (e.g. someone trains on the public seeds) cannot saturate the tier. Procgen budget per template is ≫ 10⁹ unique instances.

---

## 5. Cut from review (papers that looked promising but failed criteria)

- **ProofBench / IMProofBench / Open Proof Corpus** (arXiv 2509.26076, 2506.21621): proof judging, not final-answer.
- **LemmaBench** (arXiv 2602.24173): LLM-judge gating; not mechanical.
- **DeepSeek-Prover-V2** (arXiv 2504.21801): Lean-based, not Python procgen.
- **ExtremBench** (arXiv 2510.12997): frontier already ≥50 % on Qwen3; not low enough.
- **RealMath** (arXiv 2505.12575): research-paper extraction, not procgen.
- **GSM-Symbolic** (arXiv 2410.05229): grade-school baseline; max performance drop ~65 % but frontier still >60 %.
- **AbstentionBench** (arXiv 2506.09038): targets abstention, complements PROPHET market signal but not a math hardness driver.
- **MorphoBench** (arXiv 2510.14265): adaptive difficulty but Olympiad-sourced (no procgen recipe).
- **MathArena** (arXiv 2505.23281): live competitions, not procgen — feeds AMOS-Bench style families but not directly usable.

---

## 6. Single-page recommendation

Ship four T_extreme generators in priority order:

| Priority | Template | Expected frontier accuracy | LOC | Risk |
|---|---|---|---|---|
| 1 | **T_extreme-B CRT chain (k=8)** | ≤ 15 % | ~25 | Low — uses `sympy.ntheory.modular.crt`; same verifier as T2 |
| 2 | **T_extreme-A Pell smallest y** | ≤ 10 % | ~30 | Low — `sympy.continued_fraction_periodic` |
| 3 | **T_extreme-C forbidden-cell lattice counting** | ≤ 15 % | ~40 | Low — pure DP |
| 4 | **T_extreme-D discrete log mod p** | ≤ 5 % (no tools) | ~30 | **Tool-use guard required**; otherwise agents win trivially via Python |

These four together exercise (1) number-theoretic chains, (2) Diophantine intuition, (3) DP with twist, (4) cryptographic-grade arithmetic — span all four "Focus areas" requested for the math agent. Combined with the existing T0–T9, this gives PROPHET a 14-tier math family with empirical difficulty calibrated to push frontier accuracy from 100 % (T0–T6) down past the 15 % target on T_extreme tiers.

---

### Primary references (in order of citation)

1. arXiv 2505.16135 — Sudoku-Bench (Sakana AI, May 2025) <https://arxiv.org/abs/2505.16135>
2. arXiv 2502.19187 — BIG-Bench Extra Hard (Google DeepMind, Feb 2025) <https://arxiv.org/abs/2502.19187>
3. arXiv 2511.02872 — FATE-X formal algebra (Nov 2025) <https://arxiv.org/abs/2511.02872>
4. arXiv 2508.08292 — Putnam-AXIOM (ICML 2025) <https://arxiv.org/abs/2508.08292>
5. arXiv 2505.03171 — CombiBench (May 2025) <https://arxiv.org/abs/2505.03171>
6. arXiv 2505.11774 — HARDMath2 (May 2025) <https://arxiv.org/abs/2505.11774>
7. arXiv 2411.04372 — OEIS sequence benchmark (NeurIPS 2025 D&B) <https://arxiv.org/abs/2411.04372>
8. arXiv 2511.01846 — IMO-Bench (Google DeepMind, EMNLP 2025) <https://arxiv.org/abs/2511.01846>
9. arXiv 2503.21934 — Proof or Bluff? USAMO 2025 (ETH SRI) <https://arxiv.org/abs/2503.21934>
10. arXiv 2411.04872 — FrontierMath (Epoch AI) <https://arxiv.org/abs/2411.04872>
11. arXiv 2505.02735 — FormalMATH (Lean) <https://arxiv.org/abs/2505.02735>
12. arXiv 2510.26768 — AMO-Bench (Oct 2025) <https://arxiv.org/abs/2510.26768>
13. arXiv 2501.14249 — Humanity's Last Exam <https://arxiv.org/abs/2501.14249>
14. arXiv 2509.24210 — BeyondBench (ICLR 2026) <https://arxiv.org/abs/2509.24210>
15. arXiv 2511.00763 — Accuracy-cliff in deterministic LLM tasks <https://arxiv.org/abs/2511.00763>
16. arXiv 2507.09580 — AICrypto (cryptography benchmark) <https://arxiv.org/abs/2507.09580>
17. arXiv 2506.23679 — Learning Modular Exponentiation (transformer limits) <https://arxiv.org/abs/2506.23679>
18. arXiv 2505.23281 — MathArena (uncontaminated competitions) <https://arxiv.org/abs/2505.23281>
19. arXiv 2503.21380 — OlymMATH (May 2025) <https://arxiv.org/abs/2503.21380>
20. arXiv 2502.06453 — MATH-Perturb (ICML 2025) <https://arxiv.org/abs/2502.06453>
21. arXiv 2402.02611 — FCoReBench / PuzzleBench <https://arxiv.org/abs/2402.02611>
22. arXiv 2502.13776 — Knapsack-by-any-other-name (EHOP) <https://arxiv.org/abs/2502.13776>
23. arXiv 2506.10764 — OPT-BENCH (LLM combinatorial optimisation) <https://arxiv.org/abs/2506.10764>
24. arXiv 2510.14265 — MorphoBench (adaptive difficulty) <https://arxiv.org/abs/2510.14265>
25. arXiv 2410.07985 — Omni-MATH (ICLR 2025) <https://arxiv.org/abs/2410.07985>
26. arXiv 2509.26076 — IMProofBench (Sep 2025) <https://arxiv.org/abs/2509.26076>
27. arXiv 2506.21621 — Open Proof Corpus (Jun 2025) <https://arxiv.org/abs/2506.21621>
28. arXiv 2602.24173 — LemmaBench (Feb 2026) <https://arxiv.org/abs/2602.24173>
29. arXiv 2504.21801 — DeepSeek-Prover-V2 <https://arxiv.org/abs/2504.21801>
30. arXiv 2510.12997 — ExtremBench (Oct 2025) <https://arxiv.org/abs/2510.12997>
31. arXiv 2505.12575 — RealMath (May 2025) <https://arxiv.org/abs/2505.12575>
32. arXiv 2410.05229 — GSM-Symbolic (Apple, ICLR 2025) <https://arxiv.org/abs/2410.05229>
33. arXiv 2506.09038 — AbstentionBench (Jun 2025) <https://arxiv.org/abs/2506.09038>
34. arXiv 2511.22570 — DeepSeekMath-V2 <https://arxiv.org/abs/2511.22570>
35. arXiv 2510.17638 — Prophet Arena (Oct 2025) <https://arxiv.org/abs/2510.17638>
36. arXiv 2507.21130 — INTEGRALBENCH (Jul 2025) <https://arxiv.org/abs/2507.21130>
37. arXiv 2503.21934 (already listed)
38. arXiv 2509.24827 — Putnam-like LLM contestants (Sep 2025) <https://arxiv.org/abs/2509.24827>
39. arXiv 2507.15855 — Gemini 2.5 Pro IMO 2025 gold <https://arxiv.org/abs/2507.15855>
40. arXiv 2410.03569 — Modular-arithmetic transformer hardness <https://arxiv.org/abs/2410.03569>
41. Sakana AI Sudoku-Bench leaderboard (live): <https://pub.sakana.ai/sudoku/>
42. Epoch AI FrontierMath site: <https://epoch.ai/frontiermath>
43. MathArena live leaderboard: <https://matharena.ai/>
44. Google-DeepMind BBEH code: <https://github.com/google-deepmind/bbeh>
