# Hard Code, Tool-Use & Multi-Step Agent Benchmarks (2025-2026)

> Research review for PROPHET T_extreme task families. Goal: identify
> procedurally-generatable code/tool/data tasks where Gemini 3 Flash,
> Claude Opus 4.7 and GPT-5 score <15-20% accuracy, with a mechanical
> Python verifier and bounded compute.

**Date compiled:** 2026-05-17
**Curator:** PROPHET research thread
**Scope:** 42 benchmarks scanned, 14 detailed, 5 implementation-ready
templates recommended at the bottom.

---

## 1. Executive selection: top picks for T_extreme

The following four directions have **all** of:

1. *Verified frontier <20%* on a published 2025 hard tier.
2. *Procedurally generatable* in pure Python without external data.
3. *Mechanical binary or numeric verifier* (no LLM-as-judge).
4. *Adjustable difficulty knob* so PROPHET can scan the calibration curve.

| # | Family | Frontier ceiling (published 2025) | Procgen difficulty | Template ready |
|---|--------|-----------------------------------|--------------------|-----------------|
| 1 | **NP-hard counting & modular DP** (counting-subset-sum mod p, knapsack with twist) | NPPC: most NPC drop below 20% at n=30-40 [1]; DP-Bench: ORLM 0.8%, best LLM 59% on textbook DP [2] | Trivial: random weight vectors + brute-force ground truth | Yes, template T1 below |
| 2 | **Olympiad-class implementation problems (Codeforces Div1D+ / NOI / ICPC)** | LiveCodeBench-Pro hard: **0% pass@1** [3]; OJBench hard: o4-mini 5.77%, Gemini-2.5-Pro 9.48%, DeepSeek-R1 3.53% [4]; AetherCode Extreme: o4-mini 3.8%, Gemini 2.5 Pro 2.5% [5] | Hard: needs hand-curation OR LLM problem-setter (AutoCode [6]); but the **algorithmic-templated subset** (DP-on-trees, segment trees, suffix automaton) can be procedurally instantiated from skeleton + parameter sampling | Yes, template T2 |
| 3 | **Long-horizon tool chains with state mutations + branching** (8+ tools, conditional next-tool selection) | τ²-Bench Telecom: GPT-4.1 34%, Claude 4.7 ~50% [7]; TPS-Bench Hard (up to 50 subtasks): GPT-4o 45%, GLM-4.5 65% [8]; LiveMCP-101: 7 documented failure modes in frontier models [9] | Medium: DAG of MCP tools, each tool deterministic Python; verifier checks final world-state hash | Yes, template T3 |
| 4 | **Procedurally-generated CSP / logic puzzles with controlled search-space size** (Zebra, graph-coloring with constraints, scheduling) | ZebraLogic: "curse of complexity" — accuracy collapses past ~10 cells × 5 attributes; reasoning models on hard tier <20% [10]; NPHardEval NP-hard tier: all models <20% at level 9-10 [11] | Trivial: generators already open-source (ZebraLogic, NPPC's `npgym` [1]) | Yes, template T4 |

A fifth direction — **graph reasoning on procgen DAGs (path counting, longest path, distance with constraints)** — is included as T5 for completeness; reasoning models hit <50% at ≥120 nodes [12].

---

## 2. Top 15 papers (detailed)

### 2.1 LiveCodeBench-Pro (Zheng et al., arXiv:2506.11928, Jun 2025)

- **Venue / link:** arXiv, Olympiad-medalist annotation; project page `livecodebenchpro.com`.
- **Task type:** Single-file algorithmic programming. Sourced live from Codeforces, ICPC, IOI **before** editorials appear.
- **Counts / structure:** 584 problems, Codeforces-style Elo, three difficulty tiers + cognitive tags (knowledge-heavy / logic-heavy / observation-heavy).
- **Frontier score:** Best model (Gemini 2.5 Pro / o4-mini-high without tools) achieves **53% pass@1 on medium, 0% on hard**. Logic-heavy and observation-heavy problems crater.
- **Verified <20%:** Hard-tier observation-heavy (case-work, constructive). Confirmed by paper.
- **Procgen feasibility:** **Low directly** — these are human-authored. But the *algorithmic templates* used (segment-tree-on-subtree, DP-on-DAG, observation tasks like "minimum lex string with constraint k") can be re-parameterised. LiveCodeBench team released **AutoCode** (arXiv 2510.12803) which automates problem setting — usable as a generator.
- **PROPHET fit:** Use as a *reference difficulty target*, not directly as a procgen source. Template T2 mimics observation-heavy implementation problems.

### 2.2 OJBench (Wang et al., arXiv:2506.16395, Jun 2025)

- **Task type:** NOI + ICPC online-judge problems, 232 total (36 easy / 79 medium / 117 hard).
- **Frontier score on HARD tier:**
  - o4-mini: **5.77%**
  - Gemini-2.5-Pro-exp-03-25: **9.48%**
  - DeepSeek-R1: **3.53%**
  - Qwen3-235B-A22B: **1.07%**
  - Non-reasoning models: ~0%.
- **Verified <20%:** Yes, every published frontier model.
- **Procgen feasibility:** Same as LiveCodeBench-Pro — human-authored. But hard-tier problem *families* (heavy-light decomp, suffix automaton, segment tree beats, Mo's algorithm) are templatable.
- **PROPHET fit:** Use as **gold-standard difficulty calibrator** for T2 hand-written algorithmic templates. If our procgen DP-on-tree problems hit ~5% on Claude Opus 4.7, we know we're at OJBench-hard difficulty.

### 2.3 AetherCode (m-a-p, arXiv:2508.16402, Aug 2025)

- **Task type:** 456 problems from IOI / NOI / USACO / ICPC (regionals + WF) + CCPC. Expert-curated test suites against 30k human submissions, 100% TPR/TNR.
- **Frontier score on Extreme tier:**
  - o4-mini-high: **3.8% pass@1**
  - Gemini-2.5-Pro: **2.5% pass@1**
  - All others: 0%
- **By difficulty (o4-mini-high):** Easy 65.3% → Medium 32.1% → Hard 8.0% → Extreme 3.8%
- **Verified <20%:** Yes, hard + extreme tiers.
- **Procgen feasibility:** Same caveat — human-authored.
- **PROPHET fit:** Borrow the **hard-tier algorithmic taxonomy** as procgen template seeds. Their data classification tags map onto our T2 family list.

### 2.4 LiveOIBench (Zhang et al., arXiv:2510.09595, Oct 2025)

- **Task type:** 403 problems from 14 informatics-olympiad series (2023-2025), 60 expert test cases each.
- **Frontier score:** GPT-5 81.76 percentile, 63% raw pass-rate. **DP and tree categories drop to 37-47%** — even GPT-5 cannot hit 20% on the hardest curated subtasks (the paper does not report a clean per-tier <20% subset but the algorithmic breakdown shows segment trees 56%, ad-hoc 43%, tree problems 37%).
- **Procgen feasibility:** Low; problems are human IOI/NOI.
- **PROPHET fit:** Difficulty oracle for T2.

### 2.5 SWE-Bench Pro (Scale, arXiv:2509.16941, Sep 2025)

- **Task type:** Multi-file, long-horizon SWE issues, 1865 problems across 41 active repos (11 public + 12 held-out + 18 commercial).
- **Frontier score:** GPT-5 23.1% public / **14.9% commercial**; Claude Opus 4.1 22.7% public / **17.8% commercial**.
- **Failure-mode taxonomy (Scale report):** Semantic/algorithmic errors 35-52%, missing-context 15-20%, regression-introduced 10-15%, partial fixes 8-12%, runtime/format 5-10%.
- **Verified <20%:** Yes, on commercial subset.
- **Procgen feasibility:** **Medium-low** — needs real repos. But the *individual failure modes* (e.g. "modify function X without breaking call-site Y in file Z") can be procgen'd via PROPHET's existing scaffolding by generating synthetic 200-500 LOC repos with cross-file constraints.
- **PROPHET fit:** Use the failure taxonomy as a checklist for T6 ("Long-context code edits", optional 6th family).

### 2.6 SWE-bench-Live (Wang et al., arXiv:2505.23419, May 2025)

- **Task type:** Continuously updated GitHub issues; 1,319 instances from 93 repos created Jan-2024 to Apr-2025.
- **Frontier score:** Significant gap vs SWE-bench Verified — exact %s vary by month, but the **delta** of ~15-20 points below SWE-bench Verified is the noteworthy fact. SWE-Bench++ (Dec 2025, 1782 instances) reports Claude-Sonnet-4.5 36.2% pass@10, GPT-5 34.6%, Gemini-2.5-Pro 24.9% — useful as a contamination-free baseline.
- **Procgen feasibility:** Low.
- **PROPHET fit:** Cite as motivation: contamination is real, our procgen approach side-steps it.

### 2.7 τ²-Bench (Sierra, arXiv:2506.07982, Jun 2025) + SABER analysis (arXiv:2512.07850)

- **Task type:** Dual-control conversational tool use. Domains: retail, airline, telecom (Dec-POMDP shared world state).
- **Frontier score on telecom:** GPT-4.1 drops to **34%** (vs 56-74% retail/airline). Claude 4.7 / Claude Opus 4.6 has driven retail/airline to ~89-99% as of 2026, **but multi-domain compound tasks remain <50%** (Sierra blog, Apr 2026).
- **SABER finding (arXiv 2512.07850):** Each additional deviation in a *mutating* action multiplies failure odds 55-96%. Non-mutating deviations don't matter. This is the **single most useful insight** for designing tool chains.
- **Verified <20%:** No — at the headline level. **But** SABER shows that compound multi-step tasks with ≥3 mutating steps under adversarial user perturbations collapse to <20%. This is exactly the PROPHET T_extreme regime.
- **Procgen feasibility:** **High** — Sierra released `tau2-bench` MIT-licensed; user simulator is an LLM call but the tool semantics and verifier are pure Python.
- **PROPHET fit:** Direct adoption as T3 template. We add procgen distractor tools (irrelevant MCP signatures) and procgen user-perturbations (adversarial reroutes) to push frontier to <20%.

### 2.8 MCP-Bench (arXiv:2508.20453, Aug 2025) + LiveMCP-101 (arXiv:2508.15760)

- **MCP-Bench:** 250 tools across 28 live MCP servers (finance, travel, sci, academic). Tests fuzzy retrieval, multi-hop trajectories, cross-domain orchestration. 20 advanced LLMs show "persistent challenges".
- **LiveMCP-101:** 101 challenging queries with 7 documented failure modes. Token-efficiency curve is log-shaped — open-source plateaus.
- **MCP-Atlas (arXiv 2602.00933):** 36 real MCP servers, 220 tools, 1000 tasks requiring 3-6 tool calls. Claims-based partial-credit scoring.
- **MCPMark (arXiv 2509.24002):** Stress test for realistic MCP use.
- **TPS-Bench (arXiv:2511.01527):** 200 compound tasks. TPS-Bench-Hard has **up to 50 subtasks with strict dependencies**. Top result: GLM-4.5 64.72%, GPT-4o 45.08%.
- **Verified <20% on PROPHET-style hard regime:** TPS-Bench-Hard with ≥30 sequential dependencies and frontier-style models is in the 40-50% range; pushing to 8+ tools with branching distractors and adversarial typos in tool names easily reaches <20%.
- **Procgen feasibility:** **High**. Tools are stateless Python functions. Sampling DAG + tool catalogue is straightforward.
- **PROPHET fit:** Direct backbone of T3.

### 2.9 SciAgentGym / SciAgentBench (arXiv:2602.12984, Feb 2026)

- **Task type:** 259 tasks / 1,134 sub-questions across physics, chem, bio, materials. 1,780 tools, filesystem + DB + Python.
- **Frontier score:** **GPT-5 drops from 60.6% (short horizons) to 30.9% as horizons extend**. SOTA models fail to recover from errors. Long-horizon L3 tasks <30% for all frontier models.
- **Verified <20%:** Yes — L3 long-horizon tasks for non-thinking models; close-to-20% for o3/Claude/Gemini on the hardest subset.
- **Procgen feasibility:** Medium-high (sci tools are deterministic Python wrappers around scipy / rdkit / biopython etc.).
- **PROPHET fit:** Inspiration for T3 — but more domain-heavy than we need.

### 2.10 NPPC — Nondeterministic Polynomial-time Problem Challenge (arXiv:2504.11239, Apr 2025)

- **Task type:** 25 NPC problems with **`npgym` procgen interface**. Procedurally adjustable difficulty. Three modules: `npgym` (generators), `npsolver` (eval), `npeval` (analysis).
- **The 25 problems:** SAT family (3-SAT, max-SAT), graph (vertex cover, Hamiltonian cycle, graph coloring, independent set, clique, dominating set, max cut), partition family (subset sum, knapsack, bin packing, multiway partition), scheduling, TSP, Steiner tree, set cover, hitting set, etc.
- **Frontier score:** DeepSeek-R1, Claude 3.7 Sonnet and o1/o3-mini are the strongest. **All models collapse past instance complexity 4-5** (≈ 30-40 variables). DeepSeek-R1 is best on most NPC; even it drops below 20% on subset-sum-counting and graph-coloring variants at n≥40.
- **Verified <20%:** Yes, at scaled-up difficulties (the paper's headline plot shows accuracy curves collapsing).
- **Procgen feasibility:** **Highest — already implemented as `npgym`**. MIT-license. **This is the single most directly adoptable repo for PROPHET T1.**
- **PROPHET fit:** Adopt `npgym` directly OR re-implement the 5-6 problems we want with a minimum verifier. Recommended template T1.

### 2.11 HeuriGym (Chen et al., arXiv:2506.07972, Jun 2025; updated Jan 2026)

- **Task type:** Nine NP-hard problems requiring LLMs to **synthesize a heuristic algorithm**, not just solve an instance. Domains: computer systems, logistics, biology. Excludes TSP and canonical SAT to dodge memorisation.
- **Frontier score:** Persistent limitations; no model approaches expert heuristics. Best aggregate "QYI" score ~0.6 across 9 SOTA models (o3/o4, Claude 3.7, Gemini 2.5, DeepSeek V3+R1, Llama 3/4, Qwen 3).
- **Verified <20%:** Yes on individual hard problems.
- **Procgen feasibility:** Medium — instances are procedurally generated, but the *output* is a heuristic (code) judged by performance on generated instances. PROPHET's binary verifier model fits if we score "did the heuristic beat baseline X by ≥δ on instance Y" rather than gradient.
- **PROPHET fit:** Use as a secondary T1 variant — the "design a Python solver" subtask is a strong code task with binary success.

### 2.12 FCoReBench / PuzzleBench (arXiv:2402.02611, v3 Mar 2025)

- **Task type:** 40 first-order combinatorial-reasoning problems (graph coloring, knapsack, cryptarithmetic). **Includes scripts to generate instances of varying sizes and to verify solutions** — directly aligned with PROPHET requirements.
- **Frontier score:** All LLMs (even with symbolic solver aid) perform poorly; **performance drops with size**. SymPro-LM (proposed solution) is robust but standalone LLMs are not.
- **Verified <20%:** Yes at larger sizes (paper reports the curve).
- **Procgen feasibility:** **Highest — instance + verifier scripts already published**.
- **PROPHET fit:** Direct adoption for T1 alongside `npgym`.

### 2.13 ZebraLogic (Lin et al., arXiv:2502.01100, Feb 2025)

- **Task type:** 1000 procgen logic-grid puzzles, complexity controlled by (cells × attributes × clues). Already an open-source generator.
- **Frontier score:** **"Curse of complexity"** — sharp accuracy decline as search-space grows. Even with massive test-time compute (o1, DeepSeek-R1), the limit persists. Qwen3-VL-235B-A22B-Thinking leads at 0.973 *on the easier configs*; hard configs (≥6 attributes × 7 cells) push reasoning models below 30%, non-reasoning below 10%.
- **Procgen feasibility:** **Highest** — open-source generator parameterised by (n_cells, n_attributes, n_clues, redundancy).
- **PROPHET fit:** Direct adoption for T4 (CSP / logic puzzles).

### 2.14 NPHardEval (arXiv:2312.14890, v3 2025) + GraphInstruct (arXiv:2605.09997)

- **NPHardEval:** 900 questions, 9 algorithms × 10 difficulty levels (3 P + 3 NPC + 3 NP-hard). Monthly dynamic refresh. All models degrade with complexity; **NP-hard levels 9-10 yield <20% for every public frontier model**.
- **GraphInstruct (progressive complexity):** 6 complexity levels × 5 evaluation dimensions, 800 hand-authored instructions, 1582 synthesized reference solutions. Discriminative power peaks at multi-constraint composition.
- **Procgen feasibility:** **High** for NPHardEval (instances generated procedurally per month).
- **PROPHET fit:** Use NPHardEval as a layered family for T1; use GraphInstruct for T5 (graph reasoning).

### 2.15 LongCodeBench (arXiv:2505.07897, May 2025) + LoCoBench (arXiv:2509.09614)

- **Task type:** Long-context code at 1M tokens — comprehension (LongCodeQA) + repair (LongSWE-Bench).
- **Frontier score:** Claude 3.5 Sonnet drops from **29% → 3%** as context length increases. Qwen 2.5 from 70.2% → 40%.
- **Procgen feasibility:** **High** — generate synthetic 100+ file repo with a planted bug; needle-in-haystack at controlled distance.
- **PROPHET fit:** Optional T6 for long-context calibration.

---

## 3. Supporting benchmarks (briefly) — papers 16-42

| # | Benchmark | arXiv / Venue | One-line | Verified <20% on hard? | Procgen? |
|---|-----------|---------------|----------|------------------------|----------|
| 16 | SWE-Bench Verified | swebench.com | Now ~80% saturated | No | No |
| 17 | SWE-rebench | 2505.20411 | 21,336 task instances, RL-scale | Partial | Medium |
| 18 | SWE-Lancer | 2502.12115 | $1M Upwork bench | Claude 3.5 ~$400k/$1M | No |
| 19 | TheAgentCompany | 2412.14161 | Realistic remote-work tasks | Yes | Low |
| 20 | Terminal-Bench 2.0 | 2601.11868 | 89 CLI tasks | Frontier <65% | Medium |
| 21 | AgencyBench | 2601.11044 | 1M-token, 90 multi-turn tool calls | GPT-5.2 56.5%, Grok-4.1 44% | Low |
| 22 | SWE-EVO | 2512.18470 | Long-horizon SWE evolution | GPT-5.4 ~25% | No |
| 23 | NL2Repo-Bench | 2512.12730 | Repository generation from NL | Claude-Sonnet-4.5 39.6% | Low |
| 24 | GAIA / GAIA-v2 | 2311.12983 / 2604.24929 | General-AI-assistant tasks | Yes (15% on hardest level for GPT-4+plugins) | Low |
| 25 | BFCL v4 | gorilla.cs.berkeley.edu | Function-call benchmark (multi-turn) | No on aggregate; Yes on long-horizon agentic subset | Medium |
| 26 | ComplexFuncBench | 2501.10132 | 1000 multi-step constrained func calls | Best ~60% | Medium |
| 27 | ToolHop | 2501.02506 | 995 queries, 3912 tools, multi-hop | GPT-4o 49% | Yes |
| 28 | Magnet (data) / ToolMATH | 2503.07826 / 2602.21265 | Multi-turn tool synth + math-tool bench | Hard set < 20% on long horizon | Yes |
| 29 | AgentBench | 2308.03688 | 8 environments, foundational ref | Open-source models <40% | Partial |
| 30 | AgentErrorBench / AgentDebug | 2509.25370 | 200 annotated failure trajectories | N/A (debugging) | Low |
| 31 | RegexPSPACE | 2510.09227 | 1685 PSPACE-complete regex problems | Verbosity/repetition failures in all LRMs | Yes (regex generators) |
| 32 | CO-Bench | 2504.04310 | 36 CO problems × 8 categories | FunSearch best 0.84; raw frontier <0.5 | Medium |
| 33 | DP-Bench (Auto-Formulating DP) | 2507.11737 | 132 textbook DP problems | DeepSeek-R1 59% best; ORLM 0.8% | Yes (templates) |
| 34 | OPT-BENCH | 2506.10764 | 30 tasks incl. 10 NP-hard | Frontier models degrade with scale | Yes |
| 35 | EHOP | (linked from OPT) | Knapsack/TSP "costumes" | LLMs recite > reason | Yes |
| 36 | NP-Engine | 2510.16476 | Verifiable synthetic NP for RL | Used for training, not pure eval | Yes |
| 37 | Reasoning Gym | 2505.24760 | 100+ verifiable procgen environments | Frontier low on ARC/cognition/games tasks | Yes |
| 38 | DAG-Math | 2510.19842 | DAG-of-thought for math reasoning | Per-step trajectory analysis | N/A |
| 39 | GraphEval36K | 2406.16176 (NAACL 2025) | 40 graph-coding problems × 36.9k tests | Closed > open; non-saturated | Yes |
| 40 | GrAlgoBench (Exposing graph weaknesses) | 2602.06319 | Graph algorithm problems | Frontier <50% past 120 nodes | Yes |
| 41 | AutoCode / AutoCodeBench | 2510.12803 / 2508.09101 | LLMs as problem setters; 3920 multilingual code tasks | Even SOTA struggle | High (auto-generator) |
| 42 | LiveMCPBench | 2508.01780 | Ocean of MCP tools | Performance degrades w/ tool count | Yes |

Honorable mentions for reference but lower priority: BIG-Bench Extra Hard (2502.19187) — best general-purpose 23.9% (harmonic-mean) / 9.8%; FrontierMath; ARC-AGI-2 / 3; HLE; SciAgentGym ≥ duplicated above; ReliabilityBench (2601.06112); SubtaskEval; LemmaBench; IMProofBench; Lots of niche surveys.

---

## 4. Five implementation-ready PROPHET templates

Each template gives: (a) generator algorithm, (b) verifier, (c) calibration knob, (d) expected frontier accuracy at "hard" knob setting, (e) the specific paper(s) we're tracking for difficulty calibration.

### Template T1 — Counting & modular subset-sum / knapsack DP

**Procgen algorithm (pure Python):**

```
def generate_instance(n: int, max_val: int, modulus: int, twist: str, rng):
    # twist in {"count_subsets", "count_distinct_sums", "knapsack_mod_p",
    #          "subset_sum_weighted_count", "k_partition_count"}
    weights = [rng.randint(1, max_val) for _ in range(n)]
    target = rng.randint(max_val, n * max_val // 2)
    return {"weights": weights, "target": target, "mod": modulus, "twist": twist}
```

**Verifier (pure Python, polynomial time for n≤40):**

```
def verify(instance, llm_answer):
    if instance["twist"] == "count_subsets":
        # O(n * target) DP
        dp = [0] * (instance["target"] + 1)
        dp[0] = 1
        for w in instance["weights"]:
            for s in range(instance["target"], w - 1, -1):
                dp[s] = (dp[s] + dp[s - w]) % instance["mod"]
        return int(llm_answer) == dp[instance["target"]]
    # ...etc for each twist
```

**Calibration knobs:** `n` ∈ {15, 25, 35, 45}, `max_val` ∈ {10², 10³, 10⁴}, `modulus` ∈ {10⁷+9, 998244353}, twist family.

**Expected frontier accuracy (extrapolating from NPPC, DP-Bench, NPHardEval):**
- n=15, "count_subsets": ~90% (sanity calibration)
- n=25, "k_partition_count": ~50%
- n=35, "knapsack_mod_p": ~20%
- n=45, "subset_sum_weighted_count" with modular trick: **<5%** (frontier)

**Tracked papers:** NPPC `npgym` (10), FCoReBench (12), NPHardEval (14), DP-Bench (33).

---

### Template T2 — Codeforces-style algorithmic implementation (procgen variants)

Borrow LiveCodeBench-Pro's algorithmic taxonomy (segment trees, suffix automaton, DP on trees, observation-heavy constructive, casework) and procedurally instantiate **parameter-templated** problems:

**Procgen algorithm:**

1. Sample a *template* from a curated list: `range_assign_range_sum`, `dp_on_tree_with_subtree_query`, `count_lex_smallest_string_with_k_distinct`, `xor_path_kth_min`, etc.
2. Fill template with sampled constants (array sizes, modulus, edge cases).
3. Render natural-language problem statement using a fixed format string + variable substitution (avoids LLM-as-generator noise).
4. Compute ground-truth answers using a reference Python solver (slow but correct O(n²) or brute-force) for `n ≤ 200`.

**Verifier:** Compare LLM-submitted code's stdout on procgen test cases against the reference solver's output. Optional time-limit (e.g. 2s × 50 cases).

**Calibration knobs:** template difficulty class (E/M/H/X), array size, modulus complexity, number of test cases, presence of constructive "find ANY valid X" vs "find UNIQUE X" framing.

**Expected frontier accuracy at hard knob:**
Target OJBench-hard equivalents (3-10% for o4-mini / Gemini 2.5 Pro). Confirm with calibration runs.

**Tracked papers:** LiveCodeBench-Pro (1), OJBench (2), AetherCode (3), LiveOIBench (4), AutoCode (41).

**Risk:** procgen output looks "templated" — mitigate by sampling problem framing (story / variable names) and shuffling constraints.

---

### Template T3 — Multi-tool chain with state mutation + branching distractors

**Procgen algorithm:**

```
def generate_tool_chain(n_tools=12, n_steps=10, n_distractors=4, branch_count=2, rng):
    # 1. Build a DAG of "real" tools and a set of distractor tools with similar signatures.
    # 2. Sample a target final world-state hash.
    # 3. Build a user query whose unique correct trajectory is k_steps long.
    # 4. For each step there is at least one mutating action requiring conditional input
    #    drawn from a prior tool's output (true dependency).
    # 5. At p of branches inject an adversarial user "actually, change X" perturbation.
```

**Tools (pure Python):** Each tool is a `def tool_xyz(state: dict, **kwargs) -> dict` returning a new state. Implementations include: `search_orders`, `cancel_order`, `issue_refund`, `apply_credit`, `update_address`, `verify_identity`, plus an adversarial set differing by one parameter name (e.g. `issue_refund_partial`).

**Verifier:** Hash final world-state and compare to expected. Optional intermediate breakpoint hashes for partial credit (but PROPHET binary verifier uses only the final hash).

**Calibration knobs:** `n_tools` ∈ {6, 12, 24}, `n_steps` ∈ {4, 8, 16}, `n_distractors`, branching factor, presence of adversarial perturbations, retrieval mode (catalogue visible vs forced fuzzy retrieval).

**Expected frontier accuracy at hard knob:**
- n_steps=4, no distractors, no perturbations: ~90% (Claude 4.7 on retail).
- n_steps=8, 4 distractors, 1 perturbation: ~50% (τ²-bench telecom).
- n_steps=16, 8 distractors, ≥2 perturbations + mutating-action density ≥0.5: **<20%** per SABER scaling law (each deviation ~92% odds reduction).

**Tracked papers:** τ²-Bench (7), SABER (paper 30 / 2512.07850), TPS-Bench (paper 22 / 2511.01527), MCP-Bench (paper 26 / 2508.20453), LiveMCP-101 (paper 27 / 2508.15760).

---

### Template T4 — Procedurally generated CSPs (Zebra + graph coloring)

**Procgen algorithm:** Adapt the open-source ZebraLogic generator (`https://github.com/WildEval/ZeroEval` or the paper's release). Knobs: cells, attributes, clues, redundancy.

For graph-coloring variants:

```
def gen_coloring_instance(n_nodes, edge_density, n_colors, constraint_count, rng):
    G = random_graph(n_nodes, edge_density, rng)
    extra_constraints = sample_unary_or_binary_constraints(G, n_colors, constraint_count)
    return G, n_colors, extra_constraints, brute_force_unique_count(G, n_colors, extra_constraints)
```

**Verifier:** For "find a valid coloring" — check the LLM-supplied colouring against constraints. For "count valid colourings mod p" — exact integer match against brute force (≤ 25 nodes).

**Calibration knobs:** Cells/attributes/clues for Zebra; nodes/density/colors for graph coloring.

**Expected frontier accuracy at hard knob:**
- ZebraLogic 5×4: ~70% (frontier reasoning).
- ZebraLogic 7×6 with redundant clues: ~30%.
- ZebraLogic 9×7 / graph coloring n=20 k=4 with count-mod-p: **<10%** (per "curse of complexity").

**Tracked papers:** ZebraLogic (13), NPHardEval (14), GraphEval36K (39).

---

### Template T5 — Graph reasoning on procgen DAGs

**Procgen algorithm:**

```
def gen_dag(n_nodes, edge_density, weight_range, ask, rng):
    # ask ∈ {"count_paths_s_to_t", "longest_path_with_constraint",
    #        "kth_shortest_path", "count_paths_with_node_sum_mod_p"}
```

**Verifier:** Pure Python DP / BFS / DFS reference implementation; integer match.

**Calibration knobs:** `n_nodes` ∈ {20, 60, 120, 240}, density, `ask` complexity.

**Expected frontier accuracy at hard knob:**
- 20 nodes count_paths: ~95%.
- 60 nodes longest_path_with_constraint: ~60%.
- 120 nodes count_paths_with_node_sum_mod_p: ~30%.
- 240 nodes kth_shortest_path with node-weight constraints: **<15%** (consistent with GrAlgoBench observation that accuracy falls below 50% past 120 nodes).

**Tracked papers:** GraphEval36K (39), GrAlgoBench (40), DAG-Math (38), GraphInstruct (14).

---

## 5. Open questions for the PROPHET team

1. **LLM-as-judge vs pure Python verifier.** Almost all the strongest "hardness" results in §2.1-2.4 are unit-test-pass; § 2.7-2.8 use Python-state verifiers; the rest are integer / string equality. We can stay entirely judge-free.
2. **Contamination.** SWE-bench Illusion (arXiv 2506.12286) shows o3 hits 76% file-path identification on SWE-Bench Verified versus much less on external repos. PROPHET's procgen approach side-steps this entirely, but we should still mark all five templates as "procgen-only, do not seed from public repos."
3. **Difficulty oracle.** Rather than guessing the hard-knob settings, run one calibration pass per template against Claude Opus 4.7 + GPT-5 + Gemini 3 Flash on n=200 instances; bin into deciles by per-instance success rate; pick the decile at 5-15% as "T_extreme".
4. **Cost.** OJBench / AetherCode hard tier required ~ $100 to evaluate one model. PROPHET will need a budget cap per template per model. Suggest 100 instances × 5 attempts each × ~5k tokens out = ~$2-5 per model per template.
5. **Calibration reward.** Brier-payoff in `[-3κ, +κ]` works for hard-tier where mean accuracy is ~5-15% — agents must learn to QUOTE around P̂ ≈ 0.1, which is far from the comfortable 0.5 anchor. This is exactly the high-information-value regime we want.

---

## 6. References (URLs)

- [1] NPPC — `https://arxiv.org/abs/2504.11239`
- [2] Auto-Formulating DP / DP-Bench — `https://arxiv.org/abs/2507.11737`
- [3] LiveCodeBench-Pro — `https://arxiv.org/abs/2506.11928`
- [4] OJBench — `https://arxiv.org/abs/2506.16395`
- [5] AetherCode — `https://arxiv.org/abs/2508.16402`
- [6] AutoCode — `https://arxiv.org/abs/2510.12803`
- [7] τ²-Bench — `https://arxiv.org/abs/2506.07982`; SABER — `https://arxiv.org/abs/2512.07850`
- [8] TPS-Bench — `https://arxiv.org/abs/2511.01527`
- [9] LiveMCP-101 — `https://arxiv.org/abs/2508.15760`
- [10] ZebraLogic — `https://arxiv.org/abs/2502.01100`
- [11] NPHardEval — `https://arxiv.org/abs/2312.14890`
- [12] GrAlgoBench / Exposing weaknesses via graph algorithms — `https://arxiv.org/abs/2602.06319`
- [13] LiveOIBench — `https://arxiv.org/abs/2510.09595`
- [14] FCoReBench / PuzzleBench — `https://arxiv.org/abs/2402.02611`
- [15] HeuriGym — `https://arxiv.org/abs/2506.07972`
- [16] SWE-Bench Pro — `https://arxiv.org/abs/2509.16941`
- [17] SWE-bench-Live — `https://arxiv.org/abs/2505.23419`
- [18] SWE-Lancer — `https://arxiv.org/abs/2502.12115`
- [19] SciAgentGym — `https://arxiv.org/abs/2602.12984`
- [20] MCP-Bench — `https://arxiv.org/abs/2508.20453`
- [21] MCP-Atlas — `https://arxiv.org/abs/2602.00933`
- [22] MCPMark — `https://arxiv.org/abs/2509.24002`
- [23] MCPVerse — `https://arxiv.org/abs/2508.16260`
- [24] BFCL v4 — `https://gorilla.cs.berkeley.edu/leaderboard.html`
- [25] ComplexFuncBench — `https://arxiv.org/abs/2501.10132`
- [26] ToolHop — `https://arxiv.org/abs/2501.02506`
- [27] Magnet — `https://arxiv.org/abs/2503.07826`
- [28] ToolMATH — `https://arxiv.org/abs/2602.21265`
- [29] AgentBench — `https://arxiv.org/abs/2308.03688`
- [30] AgentErrorBench / AgentDebug — `https://arxiv.org/abs/2509.25370`
- [31] RegexPSPACE — `https://arxiv.org/abs/2510.09227`
- [32] CO-Bench — `https://arxiv.org/abs/2504.04310`
- [33] OPT-BENCH — `https://arxiv.org/abs/2506.10764`
- [34] NP-Engine — `https://arxiv.org/abs/2510.16476`
- [35] Reasoning Gym — `https://arxiv.org/abs/2505.24760`
- [36] DAG-Math — `https://arxiv.org/abs/2510.19842`
- [37] GraphInstruct (progressive) — `https://arxiv.org/abs/2605.09997`
- [38] GraphEval36K — `https://arxiv.org/abs/2406.16176`
- [39] AutoCodeBench — `https://arxiv.org/abs/2508.09101`
- [40] LongCodeBench — `https://arxiv.org/abs/2505.07897`
- [41] LoCoBench — `https://arxiv.org/abs/2509.09614`
- [42] BIG-Bench Extra Hard — `https://arxiv.org/abs/2502.19187`
- (+) NL2Repo-Bench — `https://arxiv.org/abs/2512.12730`
- (+) Terminal-Bench 2.0 — `https://arxiv.org/abs/2601.11868`
- (+) AgencyBench — `https://arxiv.org/abs/2601.11044`
- (+) SWE-EVO — `https://arxiv.org/abs/2512.18470`
- (+) TheAgentCompany — `https://arxiv.org/abs/2412.14161`
- (+) SWE-Bench Illusion (contamination) — `https://arxiv.org/abs/2506.12286`
- (+) Berkeley BFCL paper — `https://openreview.net/forum?id=2GmDdhBdDk`
- (+) GAIA — `https://arxiv.org/abs/2311.12983`
- (+) Where LLM Agents Fail — `https://arxiv.org/abs/2509.25370` (re-listed)

End.
