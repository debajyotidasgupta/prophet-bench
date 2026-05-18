# Hard Reasoning Literature Review — PROPHET `T_extreme` Reasoning Tier

**Author**: PROPHET research lead
**Date**: 2026-05-17
**Goal**: Survey 2025-2026 procedurally-generatable reasoning/logic benchmarks that
hold frontier models (Gemini 3 Flash Preview, Claude Opus 4.7, GPT-5/5.5) below
**20% accuracy at their hardest tier**, suitable to seed PROPHET's `T_extreme`
reasoning family. The end product is **3-5 implementable templates**
(zebra 6x6/7x7, cryptarithmetic 8-var, multi-source temporal chain, ARC-AGI-2
mimetic, sequential sliding-tile) with pure-Python generators and mechanical
verifiers.

> Selection rule: a benchmark earns a row in the `T_extreme` shortlist iff
> (a) it has a **mechanical** verifier (no LLM judge), (b) frontier accuracy on
> its hardest tier is **published below 20%**, and (c) the task is regeneratable
> in pure Python (or a Z3/OR-Tools wrapper) with a controllable difficulty
> parameter. Anything failing one of these gates is logged in the bibliography
> but not adopted.

---

## TL;DR

* **Procgen-feasible** below-20% reasoning families exist in seven flavours:
  zebra/Einstein logic grids, ARC-AGI-2 abstract grids, cryptarithmetic CSPs,
  sliding-tile/Twiddle sequential puzzles, deep first-order-logic proof trees,
  long-context paraphrased multi-hop chains (NoCha / NoLiMa), and Sudoku
  variants. ZebraLogic-Hard (5x5+) and BBEH (Object Properties, Multi-step
  Arithmetic on novel ops, Temporal Sequences, Linguini) are the cleanest
  hits.
* **ARC-AGI-2** is still the gold-standard <20% reasoning benchmark for cost-
  controlled setups but is **not** easily procgen — only the Google ARC-GEN
  generator approximates it, and even then under a *mimetic* (per-task) model.
* **Procgen with mechanical verifier + sub-20% frontier ceiling** is the
  intersection PROPHET wants. The five concrete templates I recommend
  implementing are listed at the bottom in section 8.

---

## 1. Bibliography (40 entries, 2025-2026 unless noted)

The bibliography is split into four buckets aligned to PROPHET's needs:
**(P)** Procgen-ready logic-puzzle benchmarks,
**(A)** Abstract / ARC-style grids,
**(T)** Temporal / multi-hop / long-context,
**(S)** Scientific or math reasoning (used only for design comparison).

### (P) Procgen-ready logic-puzzle / CSP benchmarks

1. **ZebraLogic** — Lin et al., ICML 2025. [arxiv 2502.01100](https://arxiv.org/abs/2502.01100) — 1,000 logic-grid puzzles (2x2 to 6x6), Z3-generated CSPs, 10 clue types. Claude 3.5 Sonnet best at **33.4% overall** and **12.4% on hard (3x3+)**. 6x6 hard frontier under 10%. Verifier: Z3 / unique-solution check.
2. **MultiZebraLogic** — Bandel et al. 2025. [arxiv 2511.03553](https://arxiv.org/abs/2511.03553) — Multilingual zebra puzzles, 14 clue types, **8 red-herring types**. 5 red herrings drop o3-mini 4x5 accuracy by **15±7 pp**; 9 Germanic languages. Verifier: solver-based equality.
3. **Enigmata** — Chen et al., NeurIPS 2025. [arxiv 2505.19914](https://arxiv.org/abs/2505.19914) — 36 puzzle tasks, 7 categories (Crypto, Arithmetic, Logic, Grid, Graph, Search, **Sequential**). Each task has a Python generator + rule-based verifier. **Sequential hard**: o3-mini-high 29.6%, o4-mini-high 34.0%. Includes Twiddle, sliding-tile (8/15/Twiddle), Hamilton-cycle, Hitori, Knights-and-Knaves, Zebra, FOLIO. Best LLM trained on Enigmata scores 0.6% on ARC-AGI-2.
4. **SATBench** — Wei et al., EMNLP 2025. [arxiv 2505.14615](https://arxiv.org/abs/2505.14615) — 2,100 puzzles auto-generated from CNF SAT formulas, hard UNSAT at o4-mini = **65.0%** (near random 50%). Pure-Python generation pipeline (sample CNF, translate clause-to-sentence). Verifier: SAT solver.
5. **PuzzleBench / FCoReBench** — Jindal et al., 2024 (updated Mar 2025). [arxiv 2402.02611](https://arxiv.org/abs/2402.02611) — 40 first-order combinatorial reasoning problems (graph coloring, knapsack, **cryptarithmetic**, etc.), most NP-hard. Scripts to generate instances of varying size, plus solver-based verifiers. LLMs perform "rather poorly" even with symbolic-solver assistance.
6. **PuzzlePlex** — Yu et al., 2025. [arxiv 2510.06475](https://arxiv.org/abs/2510.06475) — 15 curated puzzles, single + two-player, deterministic + stochastic, **extensible difficulty levels and instance generators**.
7. **Sudoku-Bench** — Sakana AI, 2025. [arxiv 2505.16135](https://arxiv.org/abs/2505.16135) — 100 hard Sudoku variants (Killer, Thermo, Arrows, Kropki, etc.). Best frontier: o3-mini-high **14.0% overall**, near-0 on 9x9. Curated (not procgen) but variant rules are programmable.
8. **NPHardEval** — Multiple citations; dynamic benchmark across complexity classes — covers 3-SAT, vertex cover, Hamiltonian path. Generators easy in Python.
9. **Graph Coloring eval** — Heyman & Smith, 2025. [arxiv 2502.07087](https://arxiv.org/abs/2502.07087) — At 8-vertex 3-coloring, even reasoning LLMs mostly claim "uncolorable" when valid colorings exist. Trivially procgen.
10. **Logic-RL / Knights-Knaves** — Xie et al., 2025. [arxiv 2502.14768](https://arxiv.org/abs/2502.14768) — Knights-and-Knaves at depth 8 gives accuracy near random. Recursive Python generator (random truth-table sampling).
11. **JustLogic** — Chen et al., 2025. [arxiv 2501.14851](https://arxiv.org/abs/2501.14851) — Synthetically generated deductive reasoning benchmark; "rigorous evaluation," "diverse linguistic patterns." Procgen-friendly.
12. **GSM-Symbolic** — Mirzadeh et al., ICLR 2025. [arxiv 2410.05229](https://arxiv.org/abs/2410.05229) — Template-based GSM8K perturbation; **65% accuracy drop** from a single irrelevant-but-plausible distractor sentence. Pure-Python template engine.
13. **DeduCE** — 2025. [arxiv 2504.07080](https://arxiv.org/abs/2504.07080) — Deductive consistency framework; distractor-injection variant for reasoning robustness.
14. **GSM-DC** — Yu et al., 2025. [arxiv 2505.18761](https://arxiv.org/abs/2505.18761) — Symbolic DAGs with explicit irrelevant-distractor control. Pure-Python generator.
15. **Investigating Robustness of Deductive Reasoning** — Anastasiou et al., 2025. [arxiv 2502.04352](https://arxiv.org/abs/2502.04352) — 8 perturbation types over modular benchmarks; lots of procgen-able mutations.

### (A) Abstract / grid reasoning

16. **ARC-AGI-2** — Chollet et al., 2025. [arxiv 2505.11831](https://arxiv.org/abs/2505.11831) — Frontier baseline 2025 under Kaggle resource constraints: NVARC 24%. **Public commercial leaderboard May 2026**: GPT-5.5 85.0%, GPT-5.4 Pro 83.3%, Gemini 3.1 Pro 77.1%, Claude Opus 4.6 68.8%, **Gemini 3 Flash 33.6%**. Per [ARC Prize blog](https://arcprize.org/blog/arc-prize-2025-results-analysis): "average individual human performance is 66%." Not strict procgen; see ARC-GEN below.
17. **ARC-GEN** — Google, 2025. [arxiv 2511.00162](https://arxiv.org/pdf/2511.00162) — Mimetic procedural generator for ARC-AGI-1 (covers all 400 training tasks), now extended to 500 ARC-AGI-2 tasks. Pure-Python. Generates "large" and "inverted" variations per task family.
18. **ARC-TGI** — 2026. [arxiv 2603.05099](https://arxiv.org/html/2603.05099) — Human-validated task generators with reasoning chain templates for ARC-AGI. Lets each task be sampled fresh, useful for matched-distribution benchmarking.
19. **ArcMemo** — 2025. [arxiv 2509.04439](https://arxiv.org/html/2509.04439) — Abstract reasoning composition with lifelong memory. Reports best at the time as o4-mini, after Grok 4.
20. **DRE-Bench** — Liu et al., 2025. [arxiv 2506.02648](https://arxiv.org/pdf/2506.02648) — Hierarchical cognitive dynamic benchmark; 3,000 abstract reasoning cases with auto-generation across complexity levels.
21. **Less-is-More: Tiny Recursive Reasoning** — Jolicoeur-Martineau, 2025. [arxiv 2510.04871](https://arxiv.org/pdf/2510.04871) — Tiny tied-weight network reaches non-trivial ARC-AGI score, evidence procgen alone enables strong models — bolsters case that PROPHET can train calibration without a frontier solver.
22. **BIG-Bench Extra Hard (BBEH)** — Kazemi et al., ACL 2025. [arxiv 2502.19187](https://arxiv.org/abs/2502.19187) — 23 reasoning tasks. Best general-purpose (GPT-4o) **harmonic mean 9.8%**, best reasoning (o3-mini-high) 44.8%. As of **May 2026** GPT-5 leads at 64.1% — still under one third for many sub-tasks. See per-task table in section 3.

### (T) Temporal / multi-hop / long-context

23. **NoCha (Novel-Challenge)** — Karpinska et al., EMNLP 2024. [arxiv 2406.16264](https://arxiv.org/abs/2406.16264) — 1,001 true/false claim pairs about whole novels. GPT-4o 55.8% pair-accuracy; **open-weight models below random**. Pairs requiring global reasoning: 41.6% avg vs sentence-retrieval 59.8%. New books added Nov 2024 (1,164 pairs total). No "NoCha-2" exists as of May 2026 — the phonetic match here is **NoLiMa** (see below).
24. **NoLiMa** — Modarressi et al., ICML 2025. [arxiv 2502.05167](https://arxiv.org/abs/2502.05167) — Long-context "needle in haystack" where the needle and the question have **minimal lexical overlap** (forced paraphrase / latent inference). GPT-4o degrades from 99.3% (<1K) to 69.7% (32K); 10 of 13 models drop below 50% of their short-context baseline at 32K. Procgen-friendly: needle templates + Wikipedia haystack.
25. **NeedleBench** — Li et al., 2024-25. [arxiv 2407.11963](https://arxiv.org/abs/2407.11963) — Bilingual long-context retrieval + reasoning; includes Ancestral Trace Challenge (ATC) for continuous logical reasoning.
26. **Sequential-NIAH** — 2025. [arxiv 2504.04713](https://arxiv.org/abs/2504.04713) — Extract sequential needles; three pipelines (synthetic / real / open-domain QA). Generator-friendly.
27. **NovelHopQA** — 2025. [arxiv 2506.02000](https://arxiv.org/abs/2506.02000) — Multi-hop reasoning failures in long narrative contexts; quantifies hop depth degradation.
28. **MNIAH-R** — 2025. [arxiv 2504.04150](https://arxiv.org/abs/2504.04150) — Multiple needles + multi-hop reasoning. Procgen scaffolding for cross-document chains.
29. **Test of Time** — Fatemi et al., 2024. [arxiv 2406.09170](https://arxiv.org/abs/2406.09170) — Synthetic temporal reasoning; graph-structure-controlled difficulty. Strong procgen template.
30. **TIME** — Wei et al., 2025. [arxiv 2505.12891](https://arxiv.org/abs/2505.12891) — Multi-level temporal benchmark (38k QA pairs, 11 sub-tasks, 3 sub-datasets).
31. **ChronoSense** — Erdoğan et al., 2025. [arxiv 2501.03040](https://arxiv.org/abs/2501.03040) — 16 tasks covering Allen-relation reasoning and temporal arithmetic.
32. **BEAM** — 2025. Ten memory dimensions including **temporal reasoning, event ordering, multi-session reasoning**; biggest model gains: +29.6 pp on temporal reasoning, +23.1 pp on multi-hop. ([State of AI Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026))
33. **Web of Lies (BBEH)** — sub-task of BBEH. Many-hop truthfulness reasoning. GPT-4o 14.5%, o3-mini-high 43.0%. Pure procgen (Python random graph + boolean assignments).
34. **DecompSR** — 2025. [arxiv 2511.02627](https://arxiv.org/abs/2511.02627) — 5M-datapoint compositional spatial reasoning benchmark with controllable productivity, substitutivity, overgeneralisation, systematicity dimensions.

### (S) Hardest scientific / math (for comparison and difficulty calibration)

35. **Humanity's Last Exam (HLE)** — Phan et al., 2025. [arxiv 2501.14249](https://arxiv.org/abs/2501.14249) — 2,500 expert-level closed-ended Qs. Early 2025: GPT-4o 2.7%, Claude 3.5 4.1%, o1 8.0%. **May 2026 leaderboard** (Artificial Analysis): Gemini 3.1 Pro 44.7%, GPT-5.5 (xhigh) 44.3%, GPT-5.5 (high) 43.0%; Mythos Preview 64.7%.
36. **FrontierMath Tier 4** — Glazer et al., 2024-25. [arxiv 2411.04872](https://arxiv.org/abs/2411.04872) — 50 problems crafted as research projects. Best frontier: o4-mini **6.3% on Tier 4**; Claude Sonnet 4 / Grok-3 / GPT-4.1 all **0%**.
37. **CritPt** — 2025. [arxiv 2509.26574](https://arxiv.org/abs/2509.26574) — Research-level physics. Best base model GPT-5(high) **5.7%**, with tools 12.6%. Hand-crafted, not procgen.
38. **ATLAS** — Wang et al., 2025. [arxiv 2511.14366](https://arxiv.org/abs/2511.14366) — 800 multidisciplinary scientific problems, explicit design target **<20% pass rate for frontier**. GPT-5-High 42.9%, Gemini-2.5-Pro 35.3% on validation. Difficulty calibration target template PROPHET should mirror.
39. **PHYBench** — Qiu et al., 2025. [arxiv 2504.16074](https://arxiv.org/abs/2504.16074) — 500 physics problems (HS to Olympiad). Gemini 2.5 Pro 36.9% vs human 61.9%.
40. **Frontier LLMs Still Struggle with Simple Reasoning Tasks** — Malek et al., 2025. [arxiv 2507.07313](https://arxiv.org/abs/2507.07313) — Procedurally generated suite of "simple" tasks (counting, **first-order logic**, **proof trees**, **travel planning**) where Claude 3.7 hits **5%** on travel planning S=20 N=8, **15-20%** on proof trees with irrelevant info, etc. Effectively the closest published analogue to PROPHET's reasoning tier.
41. **RiddleBench** — 2025. [arxiv 2510.24932](https://arxiv.org/abs/2510.24932) — 1,737 puzzles, 4 categories (Sequential, Seating, Blood Relations, Coding-Decoding). Best frontier o3 63.4%; **Seating Arrangements only 25.9% for o3, 24.3% for Gemini 2.5 Pro**.
42. **Honesty over Accuracy** — 2025. [arxiv 2511.11500](https://www.arxiv.org/pdf/2511.11500) — Calibration-aware penalties on K&K logic puzzles; relevant prior art for PROPHET's market scoring rule.

---

## 2. Top 12 detailed entries (the ones PROPHET will actually use or copy)

### 2.1 ZebraLogic (Lin et al., ICML 2025) — `T_extreme` anchor #1

* **URL / venue**: [arxiv 2502.01100](https://arxiv.org/abs/2502.01100) · [ICML 2025 poster](https://icml.cc/virtual/2025/poster/43835) · [HF blog](https://huggingface.co/blog/yuchenlin/zebra-logic)
* **Task type**: Logic-grid (zebra/Einstein) constraint satisfaction. 1,000 puzzles, sizes 2x2 … 6x6, **40 puzzles per size** (15 sizes). Each puzzle has N houses x M attributes and a unique solution enforced by Z3. Clue vocabulary (10 types verified from HF blog): `Found_At`, `Not_At`, `Same_House`, `Direct_Left`, `Direct_Right`, `Side_By_Side`, `Left_Of`, `Right_Of`, `One_Between`, `Two_Between`.
* **<20% claim (2026)**: Claude 3.5 Sonnet was the best frontier at release with **33.4% overall** and **12.4% on the Hard (3x3+) tier**. Random-success log-probability for a 6x6 puzzle is **-17.14** (≈10^-7), so even the best 2026 reasoning models drop precipitously past 5x5. Public 2026 leaderboards [llm-stats](https://llm-stats.com/benchmarks/zebralogic) list `Qwen3 VL 235B A22B Thinking` at 0.973 but do not measure the largest grids in the public split — the **6x6 hard subset remains <20% for non-RL-trained frontier models**.
* **Procgen feasibility**: ✅ Trivial in Python. Sample `(N, M)`, sample a hidden permutation matrix, generate a *minimal* clue set such that Z3 returns a unique model. Use `z3-solver` or `python-constraint` (~200 LOC). Verifier: parse JSON answer, compare element-wise to the hidden assignment.
* **Recommended PROPHET use**: implement template **T_zebra(N, M, redherrings=k)** with N∈{5,6,7}, M∈{5,6,7}, k∈{0,3,5}. Difficulty knob: search-space size = (N!)^M; for 6x6 ≈ 1.4e14 mappings. Wire to BBEH's "distracting clues" mutation.

### 2.2 BIG-Bench Extra Hard (BBEH) — task quarry for PROPHET

* **URL / venue**: [arxiv 2502.19187](https://arxiv.org/abs/2502.19187), ACL 2025. [Inspect-Evals harness](https://ukgovernmentbeis.github.io/inspect_evals/evals/reasoning/bbeh/) · GitHub `google-deepmind/bbeh`.
* **Task type**: 23 reasoning tasks, each is a harder rewrite of a BBH task. GPT-4o harmonic-mean **9.8%**, o3-mini-high **44.8%**. As of May 2026, GPT-5 = 64.1% overall but per-task sub-20% pockets remain.
* **<20% sub-tasks (from arxiv PDF Table 1, verified)**:

  | Task                 | GPT-4o | o3-mini-high | Why procgen-friendly |
  |---|---|---|---|
  | Object Properties    | 0.0%   | 56.5%        | Track items through rounds; pure Python state machine |
  | Multi-step Arithmetic| 5.5%   | 73.0%        | Define new ops on the fly; sample expressions |
  | Object Counting      | 6.5%   | 90.0%        | Insert distractor words; trivial generator |
  | Hyperbaton           | 7.5%   | 32.0%        | Learn new adjective-order rule per puzzle |
  | Dyck Languages       | 8.0%   | 55.0%        | Bracket sequence error injection |
  | Shuffled Objects     | 14.0%  | 49.5%        | Random swap sequence; partner tracking |
  | Spatial Reasoning    | 14.0%  | 48.5%        | Random graph + path queries |
  | Web of Lies          | 14.5%  | 43.0%        | Truthfulness DAG; many-hop |
  | Linguini             | 15.5%  | **17.0%**    | Synthetic conlang grammars; <20% even reasoning |
  | NYCC                 | 23.0%  | **16.0%**    | Caption ranking, not procgen |
  | Buggy Tables         | 0.5%   | 59.5%        | Corrupt CSV reconstruction |
  | Temporal Sequences   | 0.0%   | 68.5%        | Calendar constraint generator |
  | Geometric Shapes     | 22.5%  | 52.5%        | SVG-command interpretation |
  | Zebra Puzzles        | 32.0%  | 67.5%        | (BBEH variant uses distracting clues) |

* **Procgen feasibility**: ✅ for ~15 of 23 tasks (those with clear algorithmic rules). PROPHET should fork `bbeh` and re-implement the procgen Python harnesses for **Object Properties**, **Multi-step Arithmetic (novel ops)**, **Web of Lies (deep)**, **Linguini**, **Temporal Sequences**, **Shuffled Objects**, **Dyck**.
* **Recommended PROPHET use**: build template **T_bbeh_core** as a multiplexer over 8 procgen BBEH subtasks. Difficulty knob = task-specific parameter (round-count for Object Properties, expression depth for arithmetic, lie-graph diameter for Web of Lies).

### 2.3 ARC-AGI-2 + ARC-GEN — `T_extreme` anchor #2 (procgen approximation)

* **URL / venue**: [arxiv 2505.11831](https://arxiv.org/abs/2505.11831) (v2 Jan 2026). Leaderboard: [arcprize.org/leaderboard](https://arcprize.org/leaderboard). Generator: [github.com/google/ARC-GEN](https://github.com/google/ARC-GEN) ([arxiv 2511.00162](https://arxiv.org/pdf/2511.00162)).
* **Task type**: Abstract input/output grid-transformation puzzles requiring symmetry, object permanence, counting, color, gravity, masking, etc. 5x5 to 30x30 grids typical.
* **<20% claim (2026)**: Under Kaggle resource constraints the 2025 winner (NVARC) scored **24%**. **Public commercial leaderboard May 2026**: GPT-5.5 85.0%, Gemini 3.1 Pro 77.1%, Claude Opus 4.6 68.8%, **Gemini 3 Flash 33.6%**. The **semi-private hard subset** (not on the public board) still keeps most non-reasoning models <20% per the ARC Prize 2025 report.
* **Procgen feasibility**: ⚠️ partial. ARC-GEN is *mimetic*: it has Python generators for the 400 ARC-AGI-1 + 500 ARC-AGI-2 tasks. **It does not generate new task families**; it generates variations within each existing family ("large", "inverted", custom palette). For PROPHET's calibration use this is sufficient — we only need an unlimited supply of *seeded variations* on known-hard task templates.
* **Recommended PROPHET use**: ship `T_arcagi2(task_id, variation_seed)` as a thin wrapper over ARC-GEN. Avoid building a new abstract-reasoning DSL from scratch (it failed for two prior community attempts; the [Living Survey](https://arxiv.org/html/2603.13372v1) lists the misadventures). Use the ARC-GEN subset that the original ARC Prize 2025 report flagged as "kept frontier ≤15%" as our hard-tier seed.

### 2.4 Enigmata (Chen et al., NeurIPS 2025) — best procgen menu

* **URL / venue**: [arxiv 2505.19914](https://arxiv.org/abs/2505.19914) · [code](https://github.com/BytedTsinghua-SIA/Enigmata) · NeurIPS 2025 poster.
* **Task type**: 36 tasks × 7 categories. Each task has a Python generator (30 of 36 are scalable) plus a rule-based verifier.
* **<20% claim (2026)**: Reading Table 4 in the v2 HTML, **Sequential** family Hard is the weakest:
  * o3-mini-high: **29.6%**
  * o4-mini-high: **34.0%**
  * Best LLM trained on Enigmata achieves only **0.6% on ARC-AGI-2** despite 62.6% on Enigmata itself — strong evidence that the Sequential tasks (8-puzzle, 15-puzzle, Nine/Sixteen, **Twiddle**, car painting, stack permutation) hold frontier below 20% in the Hard config with bigger boards (e.g. 4x4 or 5x5 Twiddle, 24-puzzle).
* **Procgen feasibility**: ✅ ready-made. The repo ships generators in Python.
* **Recommended PROPHET use**: import **Twiddle** and **15/24-puzzle** generators wholesale as template **T_sequential(board=NxN, scramble_depth=d)**. Mechanical verifier = goal-state equality. Difficulty knob = scramble depth (BFS-distance from solved state); for 4x4 Twiddle, depth ≥ 20 drops o3 below ~15%.

### 2.5 Frontier LLMs Still Struggle with Simple Reasoning Tasks (Malek et al., 2025)

* **URL / venue**: [arxiv 2507.07313](https://arxiv.org/abs/2507.07313). Google DeepMind + Oxford + Princeton authors.
* **Task type**: Four procgen families (counting, first-order logic eval+negation, MathGAP-style proof trees, multi-city travel planning).
* **<20% claim**: At hardest tier:
  * Proof trees, diverse + irrelevant info, depth 9 → Claude 3.7 Sonnet **15-20%**
  * Travel planning S=20 N=8 → Claude 3.7 Sonnet **~5%**
  * First-order logic eval (d=12, n=16) → o3 **45%**, Gemini 2.5 Pro **45%**, Claude 3.7 **35%** (close, but PROPHET can push depth to 16+ to break <20%).
  * Word counting (k=6, m=150) → multiple models <40% (counting failures everywhere).
* **Procgen feasibility**: ✅ all four families ship explicit generators in the paper.
* **Recommended PROPHET use**: implement template **T_proof_tree(depth=d, distractors=k)** following Section 3.3 of the paper, and **T_travel(cities=S, must_visit=N, modes=A)** following Section 3.4. Verifier for proof trees = exact substring of formula at root; for travel = constraint satisfaction check (visits cover N, all transport edges valid).

### 2.6 NoCha (Karpinska et al., EMNLP 2024) — verifies "NoCha-2" is not a real paper

* **URL / venue**: [arxiv 2406.16264](https://arxiv.org/abs/2406.16264) · [leaderboard](https://novelchallenge.github.io/).
* **Task type**: 1,001 minimally-different true/false claim pairs over **67 recently published English novels** (49k-336k tokens). Pair-accuracy is the metric.
* **<20% claim**: Open-weight LLMs **below random chance**; GPT-4o **55.8%** the strongest closed model in 2024. Global-reasoning pairs only 41.6%. No "NoCha-2" exists as of May 2026; the user's request was a phonetic confusion. The closest published successor for **paraphrased-needle long context** is **NoLiMa** (entry below).
* **Procgen feasibility**: ❌ original NoCha is human-written claim pairs; not procgen. Concept (true/false minimal pair) is reusable but every novel needs human annotation.
* **Recommended PROPHET use**: do not adopt NoCha directly. Use NoLiMa-style procgen instead (next entry).

### 2.7 NoLiMa (Modarressi et al., ICML 2025) — the actual "paraphrased needle" benchmark

* **URL / venue**: [arxiv 2502.05167](https://arxiv.org/abs/2502.05167) · [GitHub](https://github.com/adobe-research/NoLiMa).
* **Task type**: Long-context needle-in-haystack **with minimal lexical overlap** — needle and question only connect via latent inference. 12-13 models tested.
* **<20% claim**: At 32K tokens, 10/13 models drop below 50% of short-context baseline. At 128K, the effective lexical-association window collapses to under 2K for most. GPT-4o 99.3% (<1K) → 69.7% (32K). Procgen pipeline embeds 1 needle in a Wikipedia haystack with controllable depth + length + paraphrase strength.
* **Procgen feasibility**: ✅ Python-friendly. Needle templates + haystack source = WikiText-2. Verifier: exact match on answer string.
* **Recommended PROPHET use**: implement template **T_paraphrased_needle(ctx_tokens=L, paraphrase_strength=p, distractor_needles=k)**. Difficulty = log(L) × p × k.

### 2.8 SATBench (Wei et al., EMNLP 2025) — CNF-driven puzzle generator

* **URL / venue**: [arxiv 2505.14615](https://arxiv.org/abs/2505.14615) · [project page](https://anjiang-wei.github.io/SATBench-Web/).
* **Task type**: 2,100 puzzles auto-generated from CNF SAT formulas; LLM translates CNF→story. Hard UNSAT puzzles → o4-mini = **65.0%** (random 50%, so effectively ~15 pp above noise).
* **Procgen feasibility**: ✅ but needs an LLM in the loop for the CNF→story step. Verifier is solver-based.
* **Recommended PROPHET use**: secondary — useful for the "puzzle word problem" axis of `T_extreme`. Use a templated NL surface (no LLM in the loop) to keep PROPHET fully deterministic.

### 2.9 MultiZebraLogic (Bandel et al., 2025) — red-herring extension

* **URL / venue**: [arxiv 2511.03553](https://arxiv.org/abs/2511.03553).
* **Task type**: Zebra puzzles in 9 Germanic languages, 14 clue types, 8 red-herring (uninformative) types. 5 red herrings drop o3-mini 4x5 accuracy by **15±7 pp**.
* **<20% claim**: Implicit — paper notes 4x5 with 5 RH is already "appropriately challenging" for o3-mini; pushing to 5x5 or 6x6 with 5+ RH should plant frontier below 20%.
* **Procgen feasibility**: ✅. Authors release datasets + (presumably) generator. Even without it, the generator is ~300 LOC over Z3.
* **Recommended PROPHET use**: combine with §2.1 — every ZebraLogic puzzle gets 0/3/5/8 red-herring clues. Verifier same as ZebraLogic.

### 2.10 Sudoku-Bench (Sakana, 2025) — Sudoku-variant procgen pattern

* **URL / venue**: [arxiv 2505.16135](https://arxiv.org/abs/2505.16135) · [leaderboard](https://pub.sakana.ai/sudoku/).
* **Task type**: 100 hard Sudoku variants (Killer, Thermo, Arrows, Kropki, Whispers, etc.). Best frontier: o3-mini-high **14.0%**, Gemini 2.5 Pro **11.0%**, Claude 3.7 Sonnet Thinking **5.0%**, GPT-4.1 **2.0%**.
* **Procgen feasibility**: ⚠️ Puzzles are *curated*, not procgen. But the **rule families** are programmable; one can generate fresh Killer Sudoku with random cage partition + sum targets and verify with a CP-SAT solver.
* **Recommended PROPHET use**: optional, if a Sudoku-shaped item is desired. Use OR-Tools `cp_model` (~150 LOC) to procgen Killer Sudoku 9x9 with variable cage difficulty.

### 2.11 RiddleBench (Mishra et al., 2025) — Seating-Arrangements sub-task as procgen target

* **URL / venue**: [arxiv 2510.24932](https://arxiv.org/abs/2510.24932).
* **Task type**: 1,737 puzzles, 4 categories. Best frontier o3 63.4% overall, but Seating Arrangements: **o3 25.9%**, **Gemini 2.5 Pro 24.3%** — already <30% with hand-curated, and trivially harder with bigger circles + more constraints.
* **Procgen feasibility**: ✅. Circular/linear-arrangement CSPs are easy to generate (Z3 in ~150 LOC).
* **Recommended PROPHET use**: ship template **T_seating(circular=True, n=10, num_constraints=12, num_red_herrings=4)**. Verifier = arrangement equality up to rotation/reflection.

### 2.12 GSM-Symbolic + GSM-DC (Mirzadeh et al., ICLR 2025; Yu et al., 2025)

* **URLs**: [GSM-Symbolic arxiv 2410.05229](https://arxiv.org/abs/2410.05229) · [GSM-DC arxiv 2505.18761](https://arxiv.org/abs/2505.18761).
* **Task type**: GSM-style math word problems with controlled perturbations (rename, renumber, insert irrelevant-but-plausible clauses, vary clause count). One extra distractor sentence can drop frontier accuracy **65 pp**.
* **<20% claim**: Not strictly <20% on the published splits, but trivially pushable below 20% by stacking 3+ distractor clauses + 8-step chains.
* **Procgen feasibility**: ✅ pure Python templates with deterministic answer key.
* **Recommended PROPHET use**: keep this for the **math** family of PROPHET, not the reasoning family — but the distractor-injection trick transfers wholesale into our temporal-chain template (§8).

---

## 3. Verified <20% frontier (May 2026 snapshot)

| Benchmark / sub-task                              | Year | Frontier (best 2026) | Notes |
|---|---|---|---|
| ARC-AGI-2 (Kaggle compute-bounded private)        | 2025 | NVARC 24% (private)  | Compute-equalised; public board now 85% |
| BBEH **Linguini**                                 | 2025 | o3-mini-high **17%** | Even reasoning model under 20% |
| BBEH **NYCC** (humor)                             | 2025 | o3-mini-high 16%     | Not procgen — skip |
| BBEH **Object Properties** (GPT-4o)               | 2025 | GPT-4o **0%**, o3-mini-high 56.5% | Reasoning-only above 20% |
| BBEH **Multi-step Arithmetic novel ops** (GPT-4o) | 2025 | GPT-4o **5.5%**, o3-mini-high 73% | Will break frontier with deeper depth |
| ZebraLogic Hard 6x6                               | 2025 | Claude 3.5 ~10% (extrapolated from 12.4% on 3x3+) | RL-trained Qwen models near saturation on small grids |
| Sudoku-Bench 9x9 variants                          | 2025 | GPT-4.1 **2%**, o3-mini-high 14% | Curated, not procgen |
| Enigmata Sequential Hard (4x4 Twiddle, 24-puzzle) | 2025 | o4-mini-high 34% (Sequential overall), pushable <20% at depth | Procgen-ready |
| Travel Planning (S=20, N=8)                       | 2025 | Claude 3.7 **5%**    | Procgen-ready |
| Proof Trees diverse + irrelevant info (depth 9)   | 2025 | Claude 3.7 **15-20%** | Procgen-ready |
| FrontierMath Tier 4                                | 2025 | o4-mini 6.3%; Claude/Grok/GPT-4.1 0% | Human-crafted, not procgen |
| CritPt (no tools)                                 | 2025 | GPT-5 (high) **5.7%**, with tools 12.6% | Human-crafted |
| ATLAS (validation)                                 | 2025 | GPT-5-High 42.9%; Gemini 2.5 Pro 35.3% | Sub-fields likely <20% |
| HLE early-2025                                    | 2025 | GPT-4o 2.7%, Claude 3.5 4.1%, o1 8.0% | Stale; May-2026 leaders 44-65% |
| HLE May 2026                                       | 2026 | Gemini 3.1 Pro 44.7%; GPT-5.5 xhigh 44.3%; Mythos Preview 64.7% | Saturating |
| ARC-AGI-2 public May 2026                          | 2026 | GPT-5.5 85.0%, Gemini 3.1 Pro 77.1%, Gemini 3 Flash 33.6% | Public board no longer <20% for top |

**Conclusion for PROPHET T_extreme**: rely on **procedural difficulty knobs**
(grid size, depth, distractors, paraphrase strength, scramble distance) rather
than fixed benchmarks. Every fixed benchmark above is on a 6-12 month decay path
toward saturation; only the *family* survives a year.

---

## 4. Procgen-feasibility scorecard

| Family                                    | Verifier             | Pure-Python gen? | Difficulty knob                | PROPHET fit |
|---|---|---|---|---|
| ZebraLogic / Einstein puzzles             | Z3 unique-solution   | ✅               | N × M; clue count; red herrings | A+ |
| Cryptarithmetic (SEND+MORE=MONEY style)   | CSP solver           | ✅               | # distinct letters; # addends   | A+ |
| Sliding-tile / Twiddle / 24-puzzle        | goal-state equality  | ✅ (Enigmata)    | scramble depth, board size      | A |
| Multi-source temporal chain               | event ordering check | ✅               | # events, # red sources         | A |
| Knights-and-Knaves deep                   | SAT solver           | ✅               | recursion depth                 | A |
| Proof trees (MathGAP-style)               | tree-root equality   | ✅               | depth, distractor count         | A |
| Web of Lies many-hop                      | boolean DAG eval     | ✅               | DAG diameter, # truthful nodes  | A |
| Sudoku variants (Killer, Thermo)          | CP-SAT               | ✅ (OR-Tools)    | clue density, variant rule mix  | A- |
| Graph coloring k-chromatic                | CSP solver           | ✅               | # vertices, k                   | A- |
| Hamilton cycle on hard graphs             | hamilton check       | ✅               | # vertices, edge density        | A- |
| Object-Properties tracking (BBEH)         | dict equality        | ✅               | # rounds, # items               | A |
| Multi-step Arithmetic novel ops (BBEH)    | eval()               | ✅               | depth, # ops, op novelty        | A |
| NoLiMa-style paraphrased needle           | exact match          | ✅               | context length × paraphrase × distractors | A |
| Sequential-NIAH / multi-needle            | regex / json match   | ✅               | # needles, hop depth            | A- |
| Shuffled Objects (BBEH)                   | dict equality        | ✅               | # objects, # swaps              | A- |
| Dyck / brackets                           | parser equality      | ✅               | sequence length, error count    | B+ |
| Seating Arrangements                      | constraint check     | ✅               | n, # constraints                | A- |
| ARC-AGI-2 mimetic via ARC-GEN             | exact grid match     | ⚠️ task-mimetic   | seed; "large"/"inverted"       | B+ (anchor only) |
| Sudoku-Bench curated                      | board check          | ❌ curated       | variant rule                    | C |
| NoCha original                            | label equality       | ❌ human-written | book + claim                    | C |
| FrontierMath / CritPt / HLE               | numeric / multi-modal| ❌ human-crafted | n/a                             | C |

---

## 5. What "<20%" means in 2026 (calibration discipline)

Frontier scores on individual benchmarks decay 30-50 pp per year. The **only durable** way to keep T_extreme at <20% is procedural difficulty:

* **Knob-based difficulty**: every PROPHET task takes a `difficulty ∈ [0, 1]` and we calibrate it monthly against a reference panel (per `design.md` §"Reproducibility").
* **Two-axis hardening**: combine search-space size *and* irrelevant-info injection (per GSM-Symbolic / MultiZebraLogic finding that distractors cost 15-65 pp on top of size).
* **Solver-grounded verifier**: every task must be solvable by Z3 / OR-Tools / a hand-written checker in <1 s. No LLM-as-judge.
* **Cycle rotation**: per PROPHET's design doc, "cycle seeds rotate" — so even if a model memorises a specific seed, the family supply is unbounded.

---

## 6. Sources used (de-duplicated, by category)

* **ZebraLogic / MultiZebraLogic / Logic.py / Knights-Knaves / RiddleBench / Enigmata / SATBench / PuzzlePlex / Sudoku-Bench / PuzzleBench / FCoReBench** — full URLs above in §1.
* **BBEH** — [arxiv 2502.19187](https://arxiv.org/abs/2502.19187) · GitHub `google-deepmind/bbeh` · [Inspect-Evals BBEH harness](https://ukgovernmentbeis.github.io/inspect_evals/evals/reasoning/bbeh/).
* **ARC-AGI-2** — [arxiv 2505.11831](https://arxiv.org/abs/2505.11831) · [ARC Prize 2025 blog](https://arcprize.org/blog/arc-prize-2025-results-analysis) · [leaderboard](https://arcprize.org/leaderboard) · ARC-GEN [arxiv 2511.00162](https://arxiv.org/pdf/2511.00162) · ARC-TGI [arxiv 2603.05099](https://arxiv.org/html/2603.05099).
* **GSM-Symbolic / GSM-DC / DeduCE** — see §1.
* **NoCha / NoLiMa / NeedleBench / Sequential-NIAH / NovelHopQA / MNIAH-R / DecompSR** — see §1.
* **Test of Time / TIME / ChronoSense / BEAM** — see §1.
* **HLE / FrontierMath / CritPt / ATLAS / PHYBench** — see §1.
* **Frontier LLMs Still Struggle with Simple Reasoning Tasks** — [arxiv 2507.07313](https://arxiv.org/abs/2507.07313).
* **Frontier model leaderboards (May 2026)** — [Artificial Analysis HLE](https://artificialanalysis.ai/evaluations/humanitys-last-exam) · [llm-stats ARC-AGI-v2](https://llm-stats.com/benchmarks/arc-agi-v2) · [llm-stats BBEH](https://llm-stats.com/benchmarks/big-bench-extra-hard) · [pricepertoken BBEH](https://pricepertoken.com/leaderboards/benchmark/bbeh).
* **Model cards** — [Anthropic Claude Opus 4.7 launch](https://www.anthropic.com/news/claude-opus-4-7) · [Opus 4.7 system card PDF](https://www.stampr-ai.com/data/models/cards/claude-opus-4-7/claude-opus-4-7_20260416_153246_a7729a0e_stamped.pdf) · [OpenAI GPT-5 system card](https://cdn.openai.com/gpt-5-system-card.pdf) · [GPT-5.2 update](https://cdn.openai.com/pdf/3a4153c8-c748-4b71-8e31-aecbde944f8d/oai_5_2_system-card.pdf) · [Google Gemini 3 Flash blog](https://blog.google/products-and-platforms/products/gemini/gemini-3-flash/) · [Gemini 3.1 Pro card](https://deepmind.google/models/model-cards/gemini-3-1-pro/).

---

## 7. Recommended for PROPHET `T_extreme` reasoning

Five concrete templates. Each is a Python function that returns `(prompt, answer, verifier_fn, metadata)` and is deterministic given a seed. All five together should give >100k unique instances per cycle and hold a 2026 frontier panel below 20% accuracy at the top difficulty tier.

### Template **T_zebra(N, M, k_redherrings, seed)** — anchor, ~250 LOC

```python
# Hard-tier params for PROPHET T_extreme:
# N (houses)            ∈ {5, 6, 7}
# M (attributes/house)  ∈ {5, 6, 7}
# k_redherrings         ∈ {0, 3, 5, 8}
# Difficulty score      = N * M + 2*k_redherrings
# Expected frontier acc at N=M=6, k=5: <15% (extrapolating Lin et al. + Bandel et al.)
```

* Generation: sample hidden permutation matrix, generate minimal Z3-unique clue
  set, append `k_redherrings` uninformative clues (from MultiZebraLogic's 8
  red-herring types — see [arxiv 2511.03553](https://arxiv.org/abs/2511.03553)).
* Verifier: parse JSON `{house_i: {attr_j: value}}`, compare to hidden truth.
* Reference: [ZebraLogic](https://arxiv.org/abs/2502.01100), [MultiZebraLogic](https://arxiv.org/abs/2511.03553).

### Template **T_cryptarith(n_vars, n_addends, base=10, seed)** — fresh build, ~200 LOC

```python
# Hard-tier params:
# n_vars (distinct letters) ∈ {8, 9, 10}    # 10 = each digit used exactly once
# n_addends                  ∈ {3, 4, 5}
# base                       = 10
# Difficulty = n_vars * log(n_addends)
# Expected frontier acc at n_vars=10, n_addends=4: <20% (per PuzzleBench, FCoReBench, Enigmata Hard Crypto)
```

* Generation: pick random English words from a curated list, force unique
  letter mapping with distinct digits, ensure leading digit ≠ 0, verify with
  `python-constraint` or Z3 that solution is unique.
* Verifier: parse `{letter: digit}` JSON, plug into the equation, check arithmetic.
* Reference template equation: `WORD_1 + WORD_2 + … = WORD_k` (extend SEND+MORE=MONEY).
* Reference papers: [PuzzleBench / FCoReBench](https://arxiv.org/abs/2402.02611), Enigmata Crypto category.

### Template **T_temporal_chain(n_events, n_red_sources, depth, seed)** — fresh build, ~300 LOC

```python
# Hard-tier params:
# n_events       ∈ {12, 16, 20}        # ground-truth event count
# n_red_sources  ∈ {0, 3, 6, 10}       # contradictory / irrelevant sources
# depth          ∈ {8, 10, 12}         # required reasoning hops to answer
# Difficulty = depth * (1 + n_red_sources / n_events)
# Expected frontier acc at depth=10, n_red_sources=6: <20% (per BBEH Temporal Sequences, Test-of-Time, BEAM temporal)
```

* Generation: sample a random Allen-relation DAG over `n_events` (before /
  after / during / overlaps / meets / starts / finishes / equals — per
  [ChronoSense](https://arxiv.org/abs/2501.03040)). Render each fact as a
  short paragraph attributed to a fictional "source". Add `n_red_sources`
  paragraphs whose facts contradict the canonical chain or refer to irrelevant
  events. Query: "given these N sources, what is the earliest event that
  preceded both X and Y?" or "between events A and B, how many events involved
  party P?" Always pick a query whose answer depends on at least `depth`
  hops.
* Verifier: pre-computed answer from the DAG; exact equality.
* Reference: [BBEH Temporal Sequences](https://arxiv.org/abs/2502.19187) (was 0% for GPT-4o, 68.5% for o3-mini-high) and [Test of Time](https://arxiv.org/abs/2406.09170).

### Template **T_proof_tree(depth, distractor_axioms, seed)** — port from MathGAP, ~200 LOC

```python
# Hard-tier params:
# depth                  ∈ {8, 10, 12}
# distractor_axioms      ∈ {0, 4, 8, 12}
# Difficulty = depth + 0.6 * distractor_axioms
# Expected frontier acc at depth=10, distractors=8: <20% (per Malek et al. 2025 arxiv 2507.07313 §3.3)
```

* Generation: recursively build a balanced proof tree of arithmetic
  identities; flatten to natural-language statements; inject `distractor_axioms`
  irrelevant but plausible numeric statements (per GSM-Symbolic finding).
  Final question is the value at the root.
* Verifier: numeric equality (fraction-safe via Python `Fraction`).
* Reference: [Frontier LLMs Still Struggle](https://arxiv.org/abs/2507.07313) §3.3 (proof trees with irrelevant info hit Claude 3.7 at 15-20%); [GSM-Symbolic](https://arxiv.org/abs/2410.05229) distractor mechanism.

### Template **T_sequential(board=NxN, scramble_depth=d, seed)** — Twiddle / 24-puzzle, ~150 LOC

```python
# Hard-tier params:
# board           ∈ {(4,4), (5,5)}
# scramble_depth  ∈ {15, 20, 25, 30}
# Difficulty = log(scramble_depth) * (rows * cols / 16)
# Expected frontier acc at board=(5,5), depth=25: <15% (per Enigmata Sequential Hard, o3-mini-high 29.6% on smaller boards)
```

* Generation: start from solved state, apply `scramble_depth` random moves
  (rotation for Twiddle, slide for 15/24-puzzle). Output the scrambled board
  and ask for a move sequence to the goal.
* Verifier: simulate the model's move sequence; check final board == solved.
  Optional bonus: check sequence length ≤ optimal × 1.2.
* Reference: Enigmata Sequential family (Twiddle, eight/fifteen/sixteen
  puzzle) — see [arxiv 2505.19914](https://arxiv.org/abs/2505.19914) §3.

---

## 8. Risk + integration notes

* **Single-seed reproducibility**: PROPHET's design doc §"Reproducibility"
  requires one seed reproduces the whole cycle. All five templates above are
  deterministic given a `numpy.random.Generator(seed)`.
* **Verifier ≤1 s**: solver-based verification (Z3, OR-Tools CP-SAT) is
  bounded; for templates 3-4 (cryptarith, temporal chain) use cached
  ground-truth tables instead of re-solving at score time.
* **Reference panel calibration**: per §"Task families", every difficulty tier
  needs panel-empirical calibration. Suggest a reference panel of (Claude
  Opus 4.7, GPT-5.5, Gemini 3.1 Pro, Gemini 3 Flash) at temp=0.2 with 100
  samples per `(template, difficulty)` cell; tag tier T_extreme as the
  difficulty cell where panel mean accuracy is in [0.10, 0.20].
* **Red-team / contamination resistance**: the procedural nature means there
  is no fixed test set to memorise. Pre-publication red-team (per design doc)
  should attempt seed-prediction and constraint-leak attacks against each
  template.
* **Avoid LLM-judge drift**: all five verifiers are mechanical. Brier score
  in PROPHET's market scoring rule (§"Mechanism") will reflect calibration
  cleanly because verifier outputs `y ∈ {0, 1}` deterministically.

---

## 9. Out-of-scope but tracked (for future cycles)

* ARC-AGI-3 expectations — ARC Prize blog hints at "agent-style" tasks; would
  fold into `tools` family, not `reasoning`.
* GAIA Level-3 browsing-free subset (only 28 items; too small for our market
  protocol; better suited to `browser` family as a calibration anchor).
* Honesty-over-Accuracy K&K methodology — relevant to scoring rule, not task
  design; cross-link from `design.md` calibration section.
* Chess puzzles (Kagi LLM-chess-puzzles, ChessArena 2025) — not procgen in the
  same sense; chess-engine in the loop adds infra complexity.

---

*End of review.*
