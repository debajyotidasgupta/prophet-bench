# PROPHET — Related-Work Review: Hard Tasks & Calibration Markets

**Author:** Research agent, 2026-05-17
**Scope:** Survey of recent (2025-2026) benchmarks across hard knowledge / multi-hop QA, hard multimodal / spatial-symbolic, low-resource multilingual, and **calibration / proper-scoring / market** evaluation.
**Headline TL;DR:**

1. The **calibration-market concept is no longer novel** in 2026. At least **8 papers between 2025-10 and 2026-04** evaluate LLMs with prediction-market or proper-scoring-rule structures (Prophet Arena, KalshiBench, PolyBench, Prediction Arena, PrediBench, Forecaster Arena, AA-Omniscience, BAS, ForecastBench).
2. **What is still novel in PROPHET** is the *combination*: TAKE/QUOTE/PASS action menu + Brier-calibrated proper scoring + asymmetric `[-3κ, +κ]` overconfidence band + procgen tasks (not live markets) + Pareto-frontier leaderboard + shadow re-attempt for abstention precision. Two papers are very close (BAS, Going All-In). See "Differentiation" section for the wedge.
3. **Hard procgen tasks where frontier still scores <20-30%:** ARC-AGI-2/3, BBEH, FrontierMath Tier 4, ZeroBench (multimodal), Sudoku-Bench (variants), MMMU-Pro vision-only, NoLiMa-Hard, PolyMath at level 4, AfroBench (Yoruba/Swahili gap of 30+ pts).

---

## Section A — Hard Knowledge / Multi-hop QA

### A1. **Humanity's Last Exam (HLE)** — Phan et al., 2501.14249, arXiv Jan 2025; updated 2025-2026.
- **URL:** https://arxiv.org/abs/2501.14249 ; leaderboard https://agi.safe.ai/
- **Task:** 2,500 PhD-level / expert questions across mathematics, humanities, sciences. Multimodal (text + image).
- **<20% Score in 2026:** Early 2025 baseline at 2.7-8% (GPT-4o, o1). By early 2026, top no-tools scores are 31-46% (Gemini 3 Pro Preview, GPT-5.4, Claude Opus 4.6 Thinking Max). Lab "preview" models reportedly past 56-64% (Claude Mythos, Muse Spark). **Knowledge-section subsets still <20% for non-reasoning models.**
- **Procgen feasibility:** Low — it is hand-curated, *not* procgen. Useful as inspiration for difficulty ceiling, not template.
- **Relevance to PROPHET:** Inspiration for `knowledge` family difficulty tiers, but no procgen reuse.

### A2. **FrontierMath** — Glazer et al., 2411.04872; expanded with Tier 4 in 2025-2026.
- **URL:** https://arxiv.org/abs/2411.04872 ; https://epoch.ai/frontiermath
- **Task:** Hundreds of original advanced mathematics problems, all-or-nothing scoring, models submit Python.
- **<20% Score:** Tier 4 (50 hardest) still <20% in 2026. GPT-5.4 leads at 47.6% on full FrontierMath but Tier 4 alone is much harder. Initial o3 score was 25.2%.
- **Procgen feasibility:** Low — expert-authored. Hard to procgen mathematics at that depth.
- **Relevance:** Reference for `math` and `scientific` family ceiling; not a procgen source.

### A3. **GPQA Diamond** — Rein et al., 2023; saturating in 2026.
- **URL:** https://artificialanalysis.ai/evaluations/gpqa-diamond
- **Task:** 198 PhD-level science MCQs.
- **<20% Score:** **Saturated.** GPT-5.4 = 97%, Gemini 3.1 Pro = 94.1%, Claude Opus 4.7 = 94.2% in 2026. *Not* useful as a hard floor anymore.
- **Procgen feasibility:** None (closed set).
- **Relevance:** Avoid — saturated.

### A4. **FRAMES** — Krishna et al., 2409.12941, Google Research + Harvard.
- **URL:** https://arxiv.org/abs/2409.12941
- **Task:** 824 multi-hop questions requiring 2-15 Wikipedia articles per answer. ~36% multi-constraint, 20% numerical, 16% temporal.
- **<20% Score:** SOTA at ~40% no-retrieval, 66% with multi-step retrieval. **No-retrieval frontier on 4+ hop subset is in <20% range.**
- **Procgen feasibility:** **High.** Build a knowledge graph over Wikidata + temporal facts, sample 4-hop paths, generate template questions. See template P1 below.
- **Relevance:** Direct inspiration for PROPHET `knowledge` family procgen.

### A5. **MuSiQue** — Trivedi et al., 2108.00573; still cited as the hard multi-hop reference 2025.
- **Task:** Multihop QA via single-hop composition with DAG-enforced reasoning.
- **<20% Score:** Single-paragraph baseline ~32 F1 on MuSiQue-Ans (vs ~65 on HotpotQA). Hard subsets remain stubbornly low even with reasoning models.
- **Procgen feasibility:** **High** — DAG-based composition pipeline is documented and reusable. Reasoning Gym has a derivative.

### A6. **MultiHop-RAG** — Tang & Yang, 2401.15391, COLM 2024.
- **URL:** https://arxiv.org/abs/2401.15391
- **Task:** Multi-hop queries over news article knowledge base with ground-truth evidence.
- **<20% Score:** Existing RAG systems perform "unsatisfactorily." Multi-hop OOD subset reported scores below 25% for GPT-4 era.
- **Procgen feasibility:** Medium — needs a fresh news/document corpus per cycle.

### A7. **BIG-Bench Extra Hard (BBEH)** — Kazemi et al., 2502.19187, ACL 2025.
- **URL:** https://arxiv.org/abs/2502.19187
- **Task:** Replaces each BBH task with a harder variant; 23 tasks; 6x longer context, deeper reasoning. Semi-adversarial protocol selects until top LLMs <70%.
- **<20% Score:** General-purpose models: 9.8% harmonic mean / 23.9% micro-avg. Reasoning-specialized: up to 54.2%. **Subsets where general-purpose models <20% are documented.** As of April 2026, Gemma 4 31B leads at 74.4% and GPT-5 at 64.1%.
- **Procgen feasibility:** **High** — BBH-style templates are mostly synthetic; BBEH amplification recipe is reusable.

### A8. **GroundCocoa** — NAACL 2025 (aclanthology.org/2025.naacl-long.420).
- **Task:** Compositional + conditional reasoning, 5-stage pipeline w/ symbolic logic.
- **Relevance:** Procgen template for `reasoning` family combining constraints.

### A9. **AA-Omniscience** — Jackson et al., 2511.13029, Artificial Analysis, Nov 2025.
- **URL:** https://arxiv.org/abs/2511.13029
- **Task:** 6,000 questions across 42 topics / 6 domains. Penalizes wrong answers symmetrically (+1/-1/0).
- **<20% Score:** Only 3 of ~24 frontier models scored above 0 on Omniscience Index. Top is Claude 4.1 Opus at 4.8/100 (36% raw accuracy, lowest hallucination). **Most frontier models are net-negative.**
- **Procgen feasibility:** Medium — knowledge questions can be templated from Wikidata, but factoid difficulty calibration is hard.
- **Relevance to PROPHET (D):** Closest existing benchmark to PROPHET's spirit on the *knowledge* family. Crucially **NOT a proper scoring rule** (it is symmetric ±1, not Brier). See Section D.

### A10. **SimpleQA / SimpleQA Verified** — Wei et al., 2411.04368; Verified variant 2509.07968, Sep 2025.
- **URL:** https://arxiv.org/abs/2509.07968
- **Task:** Short factual questions adversarially collected; Verified is a higher-quality re-graded subset.
- **<20% Score:** Frontier hallucination rate still 30-47% (GPT-4.5: 37%; GPT-5-main: 47%). Gemini 2.5 Pro leads at F1=55.6 on Verified.
- **Procgen feasibility:** High — Wikidata triple sampling + adversarial filter for niche entities.

### A11. **BrowseComp** — Wei et al., 2504.12516, OpenAI Apr 2025.
- **URL:** https://openai.com/index/browsecomp/
- **Task:** 1,266 hard problems requiring locating obscure web information.
- **<20% Score:** GPT-4o without browsing ~0.9%; with browsing 1.9%. By 2026 with deep-research agents, GPT-5.5 Pro 90.1%, Claude Mythos 86.9%. **Saturating fast for agentic models** but still <20% for plain-LLM no-browse.
- **Relevance:** Inspiration for PROPHET `browser` family.

### A12. **NoLiMa / NoLiMa-Hard** — Modarressi et al., 2502.05167, ICML 2025.
- **URL:** https://arxiv.org/abs/2502.05167
- **Task:** Needle-in-haystack with *minimal* lexical overlap between query and needle; requires latent association.
- **<20% Score:** At 32K context, 11/13 frontier models drop below 50% of baseline; GPT-4o drops 99.3% → 69.7%. **NoLiMa-Hard subset is the 10 most difficult question-needle pairs — frontier models commonly score <20% there.**
- **Procgen feasibility:** **High** — synthetic needles with controlled lexical overlap and distractor injection.

### A13. **Chemistry Multi-hop Reasoning** — 2504.16414, 2025.
- **Task:** Curated chemistry-domain compositional reasoning benchmark.
- **<20% Score:** SOTA struggles significantly; exact <20% subset bins available.
- **Procgen feasibility:** Medium — needs a small reaction-graph + properties KB.

---

## Section B — Hard Multimodal / Symbolic-Spatial / Text-Encoded Vision

### B1. **ARC-AGI-2** — Chollet et al., 2505.11831, March 2025.
- **URL:** https://arxiv.org/abs/2505.11831 ; https://arcprize.org/arc-agi/2
- **Task:** Few-shot grid-based abstraction puzzles, procedurally generated semi-private and private splits.
- **<20% Score:** Under Kaggle compute caps, top systems 16-24% on private split. Public leaderboard reached 84.6% with Gemini 3 Deep Think at $13.62/task; commercial Opus 4.5 Thinking 37.6% at $2.20/task. **<20% holds for any cost-bounded eval.**
- **Procgen feasibility:** **High** — ARC-AGI-2 uses procgen by design.
- **Relevance:** Direct inspiration for `reasoning` family grids.

### B2. **ARC-AGI-3** — Chollet et al., 2603.24621, March 2026.
- **URL:** https://arxiv.org/abs/2603.24621
- **Task:** Interactive turn-based environments, no instructions / rules / goals; agents must learn world model + acquire goals.
- **<20% Score:** Humans 100%, **frontier AI 0.51%** as of March 2026. Preview comp top score 12.58%.
- **Procgen feasibility:** **High** — game environments are procgen.
- **Relevance:** Sets the floor for interactive agency.

### B3. **ZeroBench** — Roberts et al., 2502.09696, Feb 2025.
- **URL:** https://zerobench.github.io/
- **Task:** 100 hand-curated visual reasoning problems, 334 sub-questions. Multi-step visual reasoning with exact-match answers.
- **<20% Score:** **Frontier SOTA = 0.0% on main task.** Claude Sonnet 3.5 v2 best at 24.3% pass@1 on sub-questions.
- **Procgen feasibility:** Low — hand-curated. But sub-question templates inspirable.

### B4. **MMMU-Pro** — Yue et al., 2409.02813, ACL 2025.
- **URL:** https://arxiv.org/abs/2409.02813
- **Task:** Vision-only embedding of multi-discipline questions; filters out text-only-solvable items.
- **<20% Score:** Models score 16.8-26.9% on MMMU-Pro vs 60-75% on MMMU. 15-20 pt drop for frontier.
- **Relevance:** Strong inspiration for PROPHET `multimodal` (text-encoded) family — the vision-only setting transfers conceptually to "image inside text" challenges.

### B5. **Sudoku-Bench** — Sakana AI, 2505.16135, May 2025.
- **URL:** https://arxiv.org/abs/2505.16135
- **Task:** 100 Sudoku variant puzzles (15 4x4, 15 6x6, 70 9x9) including "modern variants" requiring breakthroughs.
- **<20% Score:** **SOTA <15% unaided on variant Sudoku.** GPT-5 is the first LLM to solve any 9x9 modern Sudoku (theta variant).
- **Procgen feasibility:** **Very high** — Sudoku4LLM (DolbyUUU/Sudoku4LLM) provides procgen generator with 11 serialization formats. Variant rules (anti-knight, killer cages, thermo) are codeable.
- **Relevance:** Backbone of PROPHET `multimodal` family. See template P3 below.

### B6. **ChartQAPro** — Masry et al., 2504.05506, ACL 2025.
- **URL:** https://aclanthology.org/2025.findings-acl.978/
- **Task:** 1,948 questions over 1,341 charts from 99 sources, MCQ + conversational + hypothetical + unanswerable.
- **<20% Score:** Claude Sonnet 3.5 drops from 90.5% (ChartQA) to 55.81% (ChartQAPro). Frontier ceiling 55-60%; *unanswerable* subset known to drag down further.
- **Procgen feasibility:** Medium — chart generators exist; unanswerable variants need careful design.

### B7. **ASCIIBench / ASCIIEval / ViTC** — Multiple groups, 2410.01733 + 2512.04125.
- **URL:** https://arxiv.org/abs/2512.04125 ; https://github.com/JiaQiSJTU/VisionInText
- **Task:** Recognize / classify ASCII-art encoded glyphs and shapes; 5,315 class-labeled samples (ASCIIBench), 3K+ in ASCIIEval categorization tree.
- **<20% Score:** **Text-only LLMs lag dramatically** — even GPT-4o text-only struggles, only vision-mode hits 82% accuracy. Open-source text-only is single-digit.
- **Procgen feasibility:** **Very high** — ASCII glyph rendering is mechanically generated (text-of-text). Direct template P4.
- **Relevance:** Direct match for PROPHET `multimodal` text-encoded family.

### B8. **Reasoning Gym** — Stojanovski et al., 2505.24760, NeurIPS 2025 Spotlight.
- **URL:** https://github.com/open-thought/reasoning-gym
- **Task:** Library of 100+ procgen reasoning environments with verifiable rewards (algebra, arithmetic, computation, cognition, geometry, graph, logic, games).
- **<20% Score:** "Hard" RG tasks: o3-mini and DeepSeek-R1 outperform generic LLMs by >20 pts absolute; non-reasoning models can be <20% on hard tasks.
- **Procgen feasibility:** **Built-in.** **Re-use as a dependency for PROPHET reasoning family.**

---

## Section C — Multilingual & Low-Resource

### C1. **PolyMath** — Wang et al., 2504.18428, NeurIPS 2025 D&B Track.
- **URL:** https://qwen-polymath.github.io/
- **Task:** Multilingual math reasoning, 18 languages × 4 difficulty levels × 9,000 problems.
- **<20% Score:** Top model (Qwen-3-235B-A22B-Thinking) only **54.6% overall, ~40% at hardest level**. Low-resource languages at hardest level commonly <20%.
- **Procgen feasibility:** Medium — math is templated, but translation/cultural-naturalness needs native speakers.

### C2. **MMLU-ProX** — Xuan et al., 2503.10497, EMNLP 2025.
- **URL:** https://mmluprox.github.io/
- **Task:** MMLU-Pro translated to 29 typologically diverse languages, 11,829 items each.
- **<20% Score:** Best models drop from 70.3% English → **40.1% Swahili (Qwen2.5-72B); 25-point Yoruba gap** vs English. African-language subsets straddle 20-40% even for frontier.
- **Procgen feasibility:** Low — translation cost.

### C3. **AfroBench** — Adelani et al., 2311.07978 / Findings of ACL 2025.
- **URL:** https://mcgill-nlp.github.io/AfroBench/
- **Task:** 64 African languages × 15 tasks × 22 datasets (9 NLU, 5 generation, 5 QA/knowledge, 1 math).
- **<20% Score:** Knowledge-intensive + reasoning tasks show the largest gap. Mathematical reasoning on Yoruba/Hausa/Wolof commonly <20% for frontier.
- **Procgen feasibility:** Low — language data is human-collected.
- **Relevance:** Inspiration for PROPHET `multilingual` family.

### C4. **Multilingual Reasoning Gym** — Apple ML Research, 2603.10793, March 2026.
- **URL:** https://machinelearning.apple.com/research/multilingual-reasoning-gym
- **Task:** Procgen reasoning across **14 languages** (en, zh, de, es, fr, ja, pt-br, ru, ko, it, th, bn, te, sw), 94 task templates, native-speaker validated.
- **<20% Score:** Low-resource subsets (Telugu, Swahili, Bengali, Thai) at hardest difficulty land in <20% range for non-frontier models.
- **Procgen feasibility:** **Built-in.** **Direct dependency candidate for PROPHET multilingual family.** Cross-lingually parallel data enables controlled head-to-head difficulty calibration.

### C5. **Cross-lingual benchmark for Cantonese/Japanese/Turkish** — 2511.10664, Nov 2025.
- **Task:** Open-domain QA, summarization, EN-to-X translation, culturally grounded dialogue across three morphologically rich languages.
- **<20% Score:** Turkish agglutinative morphology and Cantonese colloquialisms degrade frontier models on culturally grounded subsets to <20% on certain compositional sub-tasks.
- **Procgen feasibility:** Medium — morphology generator possible (FST tools exist).

### C6. **CRUXEval-X** — Xu et al., 2408.13001, ACL 2025.
- **URL:** https://aclanthology.org/2025.acl-long.1158/
- **Task:** Code reasoning across **19 languages**, 12,660 functions / 19K test cases.
- **<20% Score:** Models trained only on Python achieve at most 34.4% Pass@1 on other languages; obscure languages (Racket) <20%.
- **Relevance:** Optional addition to PROPHET `code` family.

### C7. **IrokoBench** — Adelani et al., NAACL 2025.
- **Task:** African language NLU benchmark, multiple task families.
- **<20% Score:** Frontier degrades sharply on low-resource African languages.

### C8. **mAceReason-Math** — Apple ML Research, 2025-2026.
- **URL:** https://machinelearning.apple.com/research/macereason-math
- **Task:** High-quality multilingual math problems for RLVR.
- **Relevance:** Procgen multilingual data source.

---

## Section D — **Calibration / Confidence / Market / Proper-Scoring** (Critical for PROPHET positioning)

This is the section that most affects PROPHET's positioning. **There is a rapidly growing 2025-2026 literature on calibration and market-style LLM evaluation.** I categorize each by how close it is to PROPHET's specific design.

### D1. **Prophet Arena (LLM-as-a-Prophet)** — Yang et al., 2510.17638, UChicago, October 2025.
- **URL:** https://arxiv.org/abs/2510.17638 ; https://www.prophetarena.co/leaderboard
- **Task:** Probabilistic forecasting on 1,367 Kalshi events, 72,136 binary markets. Three metrics: Brier, ECE, Market Return.
- **Verifier:** Real-world resolution (mechanical via Kalshi settlement).
- **Frontier scores:** GPT-5 (Reasoning) Brier 0.184, ECE 0.042, Market Return 0.943. Market baseline Brier 0.187. Models cluster Brier in [0.18, 0.22].
- **Mechanism:** Direct probability elicitation; no action menu. Risk-neutral allocation rule decides Yes/No contracts.
- **PROPHET overlap:** **CRITICAL — name collision; same Brier-and-market spirit.** They publish a leaderboard called "Prophet Arena."
- **PROPHET differentiation:**
  - Theirs is live-event forecasting; ours is procgen task evaluation across 12 capability families (math, code, knowledge, reasoning, multimodal, multilingual, etc.) with deterministic verifiers.
  - Theirs has no abstention or TAKE/QUOTE/PASS; ours has all three actions with a δ_pass cost.
  - Theirs has no asymmetric overconfidence penalty; ours has `[-3κ, +κ]`.
  - Theirs is contamination-resistant via "future events"; ours is contamination-resistant via "fresh procgen seeds per cycle."
- **Action item:** **Rename benchmark.** Prophet Arena is well-known. Suggest PROPHET → PROPHET-Bench, or rebrand entirely (CALL-Bench, OPTION, MARKETMIND, etc.).

### D2. **KalshiBench** — Nel, 2512.16030, December 2025.
- **URL:** https://arxiv.org/abs/2512.16030
- **Task:** 300 Kalshi market questions resolved post-cutoff; evaluates Brier Skill Score (BSS), ECE.
- **Frontier scores:** Only **1 of 5** frontier models achieved positive BSS (+0.057). Best-calibrated Claude Opus 4.5 ECE=0.120; reasoning-enhanced GPT-5.2-XHigh **worse** at ECE=0.395 despite comparable accuracy.
- **Verifier:** Real market settlement.
- **PROPHET differentiation:** Live-event market, no procgen, no action menu, no Brier-payoff structure.

### D3. **PolyBench** — Cheng et al., 2604.14199, q-fin.CP, February 2026.
- **URL:** https://arxiv.org/abs/2604.14199
- **Task:** 38,666 binary Polymarket prediction markets, 4,997 events; CLOB-realistic execution simulation.
- **Frontier scores:** Only 2 of 7 models positive returns: MiMo-V2-Flash 17.6% CWR, Gemini-3-Flash 6.2% CWR. Open-source MiMo-V2-Flash beats proprietary on calibration.
- **Action structure:** BUY (if conf >0.6) / SKIP / implicit-reject. Explicitly **NOT Brier** ("we do not adopt traditional proper scoring rules such as the Brier Score" due to lack of unbiased probability reference in volatile markets).
- **PROPHET overlap:** Action menu is similar (3 actions) but skin-in-the-game is dollars, not Brier-payoff.
- **PROPHET differentiation:** Procgen tasks + Brier-rewarded QUOTE + asymmetric overconfidence band.

### D4. **Prediction Arena** — Zhang et al., 2604.07355, Arcada Labs + Harvard, March 2026.
- **URL:** https://arxiv.org/abs/2604.07355
- **Task:** Autonomous trading on Kalshi + Polymarket with real settlement.
- **Frontier scores:** **All 6 tested models post negative returns** over 57 days. glm-4.7 –16%, grok-4 –20%, gpt-5.2 –20.5%, claude-opus-4-5 –25.9%, gemini-3-pro –30.5%.
- **PROPHET differentiation:** Continuous buy/sell on live markets vs. discrete TAKE/QUOTE/PASS on synthetic tasks; no Brier, no abstention precision metric.

### D5. **PrediBench** — Presage Labs, late 2025.
- **URL:** https://huggingface.co/blog/charles-azam/predibench ; https://predibench.com/
- **Task:** Every day, models get $1 to allocate across top-10 Polymarket events; tracks Brier, returns, Sharpe.
- **Frontier scores:** Half of tested models beat baseline; Grok-4 +6%. Most recent ranking aligns with LMSys Arena (strong correlation Brier↔general capability).
- **Skin in the game:** Yes — actual $1 allocation across 10 markets, 3x/week.
- **PROPHET differentiation:** Live news, no abstention metric, no procgen, no proper-scoring-rule QUOTE action.

### D6. **Going All-In on LLM Accuracy** — Fake Prediction Markets, 2512.05998, December 2025.
- **URL:** https://www.arxiv.org/abs/2512.05998
- **Task:** 100 math/logic questions, 5,400 predictions/condition. Betting game using fictional LLM currency.
- **Findings:** Incentive run **+2.4 pts accuracy** (81.5% vs 79.1%); **4x faster learning** (12.0 vs 2.9 pts gain across rounds); "whale" bets ~99% correct, small bets ~74% correct. **Betting mechanic creates legible confidence signal absent in binary outputs.**
- **PROPHET overlap:** **Most conceptually similar to PROPHET's market thesis.** It validates the central PROPHET hypothesis (markets surface calibration). However, it uses fake currency, doesn't deploy proper scoring rule, and is a pilot on 100 questions.
- **PROPHET differentiation:** PROPHET is a *full procgen benchmark* across 12 capability families with mechanical verifiers, explicit Brier proper-scoring on QUOTE, asymmetric overconfidence band, and Pareto leaderboard. Going All-In is a pilot study showing the *principle* works.
- **Citation strategy:** **Cite this as motivating evidence in PROPHET's intro.** It is the strongest empirical justification for the marketplace mechanism.

### D7. **AbstentionBench** — Kirichenko et al., 2506.09038, FAIR Meta, June 2025.
- **URL:** https://github.com/facebookresearch/AbstentionBench
- **Task:** Holistic abstention benchmark, 20 datasets / 6 scenarios / 35K unanswerable queries (unknown answer, false premise, stale, subjective, underspec context, underspec intent).
- **Frontier scores:** GPT-4o and Qwen 2.5 best on average; no model consistently outranks. **Reasoning fine-tuning *degrades* abstention by 24%.**
- **Verifier:** Llama 3.1 8B as judge (88% accuracy vs human-annotated sample).
- **Mechanism:** **No cost-of-pass, no skin-in-the-game.** Pure F1 of "did the model correctly abstain."
- **PROPHET overlap:** Abstention precision metric in PROPHET is conceptually similar but PROPHET has explicit δ_pass cost and shadow re-attempt validation. AbstentionBench is the standard reference for the abstention concept; PROPHET should cite + position as "abstention with economic skin in the game."

### D8. **Behavioral Alignment Score (BAS)** — Wu et al., 2604.03216, University of Oxford, April 2026.
- **URL:** https://arxiv.org/abs/2604.03216
- **Task:** Decision-theoretic LLM confidence evaluation; explicit utility model "answer-or-abstain."
- **Mechanism:** **Asymmetric penalty for overconfidence: incorrect predictions incur penalty `ln(1-s)` → −∞ as confidence → 1; correct predictions get only linear reward `s`.** Aggregates utility across all risk thresholds rather than fixing payoffs. Provably truthful confidence maximizes expected BAS (Theorem 2.1).
- **Frontier scores:** GPT-4o –5.06 BAS on AIME (11.7% accuracy); GPT-oss 0.57 BAS (75% accuracy). Even strong models show "severe overconfidence" on open-ended tasks like SimpleQA.
- **PROPHET overlap:** **MOST DESIGN-LEVEL OVERLAP IN THE LITERATURE.** Asymmetric overconfidence penalty + proper-scoring-rule + abstention as first-class action.
- **PROPHET differentiation:**
  - BAS aggregates across **all** risk thresholds (an integral); PROPHET uses **one fixed** market price (`V, C, κ, δ_pass`) per family with **deterministic noise** — keeps the leaderboard auditable per task.
  - BAS uses log-scoring (which goes to –∞); PROPHET uses bounded Brier (worst-case loss `1/4`, factor 4 rescales to 1.0, asymmetric band `[-3κ, +κ]`).
  - BAS is mainly a metric; PROPHET is a procgen benchmark with 12 task families and a Pareto leaderboard.
  - BAS has no action menu (TAKE/QUOTE/PASS); PROPHET does.
- **Citation strategy:** **Must cite. Position PROPHET as: bounded proper-scoring TAKE/QUOTE/PASS deployment of BAS-style asymmetry on procgen multi-family benchmark.**

### D9. **ForecastBench** — Karger et al., ICLR 2025.
- **URL:** https://www.forecastbench.org/
- **Task:** Dynamic biweekly forecasting questions; nightly resolution updates.
- **Frontier scores:** GPT-4.5 Brier 0.101 vs **superforecaster 0.081** (difficulty-adjusted). LLMs improving ~0.016 Brier/year; parity projection November 2026.
- **PROPHET differentiation:** Live events vs procgen; difficulty-adjusted Brier is a useful technique to import.

### D10. **AA-Omniscience** — See A9. **Critical: not Brier, but explicitly penalizes wrong answers.** Closest in spirit to PROPHET on a single capability family (knowledge), but lacks the QUOTE proper-scoring action and asymmetric band.

### D11. **Beyond Binary Rewards** — Damani et al., 2507.16806, MIT.
- **Task:** Training method (not benchmark). Proposes proper scoring rules (log + Brier) as RL rewards instead of binary correctness.
- **Relevance:** PROPHET's QUOTE action is the *evaluation analog* of this *training* idea. Cite as foundational.

### D12. **Behaviorally Calibrated RL** — Wu et al., 2512.19920, ByteDance + CMU + Fudan, December 2025.
- **Task:** Training method. Uses Brier (uniform-prior) or cross-entropy (Beta-prior, asymmetric overconfidence penalty) as proper-scoring RL reward.
- **Relevance:** Demonstrates that asymmetric overconfidence penalty *during training* improves calibration. **PROPHET measures the same property at eval time.**

### D13. **FermiEval** — 2510.26995, October 2025.
- **URL:** https://arxiv.org/abs/2510.26995
- **Task:** Fermi-style estimation with strict confidence-interval scoring (Winkler interval score, coverage).
- **Findings:** Nominal 99% intervals cover only 65% on average → severe overconfidence. Conformal prediction restores 99% coverage, Winkler score –54%.
- **PROPHET overlap:** Different — interval calibration vs binary outcome. Cite as adjacent UQ benchmark.

### D14. **Trust or Escalate** — ICLR 2025 (proceedings 08dabd5345b37fffcbe335bd578b15a0).
- **URL:** https://iclr.cc/virtual/2025/oral/31838
- **Task:** LLM-judge calibration with selective evaluation; Simulated Annotators for confidence estimation. Cascaded selective evaluation w/ provable human-agreement guarantee.
- **Relevance:** Different setting (judge), but **proves the value of selective evaluation with confidence guarantees** — supportive reference.

### D15. **SelectLLM** — OpenReview JJPAy8mvrQ, 2025.
- **Task:** Selective prediction fine-tuning. Balances coverage and risk.
- **Relevance:** Method paper. PROPHET evaluates what SelectLLM trains for.

### D16. **Are LLM Decisions Faithful to Verbal Confidence?** — Wang et al., 2601.07767, USC, Jan 2026.
- **URL:** https://arxiv.org/abs/2601.07767
- **Task:** Tests whether models translate stated confidence into risk-aware decisions. Utility model: +1/−λ/0 with optimal threshold τ(λ) = λ/(1+λ).
- **Findings:** Even at λ=100 penalty, models maintain "answer-heavy" policies, producing **"utility collapse" (normalized utility −0.85 to −0.52 on HLE)**. Post-hoc optimal policy adds +0.31 to +0.69 utility. **Models can express calibrated probabilities but cannot strategically act on them.**
- **PROPHET overlap:** Strongly validates PROPHET's central insight (need explicit action menu to test agency, not just probabilities). **CITE PROMINENTLY IN INTRO.**
- **PROPHET differentiation:** PROPHET formalizes the action menu as a benchmark with 12 capability families and procgen, not just a 4-dataset analysis.

### D17. **HalluLens** — ACL 2025 (aclanthology.org/2025.acl-long.1176).
- **Task:** Hallucination + refusal jointly measured.
- **Relevance:** Demonstrates need for joint hallucination/refusal eval; PROPHET extends with cost weights.

### D18. **Composite Reliability Score (CRS)** — 2512.24058, December 2025.
- **Task:** Single composite metric combining calibration, robustness, uncertainty.
- **Relevance:** **Anti-pattern PROPHET avoids.** PROPHET deliberately reports Pareto frontier instead of scalar; cite as the "single scalar" approach we don't take.

### D19. **CalibrationAcrossLayers (BaseCal, DoublyCal, CritiCal)** — Multiple, 2025-2026.
- **Relevance:** Method papers improving ECE; PROPHET is the eval benchmark these methods would optimize against.

### D20. **Forecaster Arena** — forecasterarena.com, 2025-2026.
- **Task:** Weekly Polymarket competition with $10K virtual budget.
- **Relevance:** Similar to PrediBench. Live event, no abstention metric, no proper-scoring QUOTE.

### D21. **Future Is Unevenly Distributed** — 2511.18394, November 2025.
- **Task:** LLM forecasting variability across question type/domain.
- **Findings:** LLM calibration depends sharply on prompt framing and context.
- **Relevance:** Methodological caveat — PROPHET should report calibration *per family* not aggregate (we already do).

### D22. **MarketBench** — 2512.12264, December 2025.
- **Task:** Introductory quantitative trading evaluation.
- **Relevance:** Adjacent; not direct calibration-market work.

### D23. **PolySwarm** — 2604.03888, March 2026.
- **Task:** Multi-agent LLM framework for prediction-market trading + latency arbitrage.
- **Relevance:** Multi-agent extension; orthogonal to PROPHET's single-agent eval focus.

### D24. **Cost-Saving LLM Cascades with Early Abstention** — 2502.09054, Feb 2025.
- **Findings:** Early abstention trades +4.1% abstention for −13% cost and −5% error.
- **Relevance:** Validates that economic cost-aware abstention works.

### D25. **Know Your Limits (Survey)** — TACL 2025.
- **Relevance:** Reference survey on abstention.

---

## Section D summary table: Who Already Does What

| Property | Prophet Arena | KalshiBench | PolyBench | PrediBench | AA-Omn. | BAS | AbstentionBench | Going All-In | **PROPHET (ours)** |
|---|---|---|---|---|---|---|---|---|---|
| Brier proper score | yes | yes | no | yes | no | log | no | implicit | **yes** |
| Asymmetric overconfidence | no | no | no | no | no | **yes** | no | no | **yes** |
| Action menu (TAKE/QUOTE/PASS) | no | no | partial (BUY/SKIP) | no | partial | partial | no | partial | **yes (explicit 3 actions)** |
| Procgen tasks | no | no | no | no | partial | no | no | no | **yes** |
| 12 capability families | no | no | no | no | knowledge only | no | abstention only | math/logic | **yes** |
| Mechanical verifier | settlement | settlement | settlement | settlement | LLM-graded | various | LLM-judge | grader | **yes (regex/exec/equality)** |
| Skin-in-the-game payoff | implicit | implicit | yes ($) | yes ($) | symmetric ±1 | log utility | no | fake coin | **yes (V, C, κ, δ_pass)** |
| Pareto-frontier leaderboard | no | no | no | no | no | no | no | no | **yes** |
| Shadow re-attempt for abstention precision | no | no | no | no | no | no | no | no | **yes** |

**The wedge for PROPHET:** the *full bundle* — TAKE/QUOTE/PASS × bounded asymmetric Brier × procgen × 12 capability families × mechanical verifier × Pareto leaderboard. **No single existing benchmark has all six.** The closest are Prophet Arena (overlap in name + Brier + market spirit) and BAS (overlap in asymmetric proper-scoring decision theory).

**Action items for PROPHET positioning:**
1. **Rename.** "Prophet Arena" already published Oct 2025 with leaderboard at prophetarena.co. Suggested rebrand: **OPTION-Bench** (Option-pricing-themed) or **VICKREY** (truth-revealing auction) or **PROPHET-Bench** (keep but disambiguate).
2. **Reposition.** In Related Work, frame as: "*the procedural-generation, multi-capability counterpart to live-market benchmarks (Prophet Arena, KalshiBench, PrediBench) and the bundled-action evaluation analog to training-time proper-scoring methods (Beyond Binary Rewards, Behaviorally Calibrated RL).*"
3. **Cite prominently:** D1 (Prophet Arena - name collision must address), D6 (Going All-In - motivation), D8 (BAS - theoretical kin), D16 (Verbal Confidence Decisions - empirical motivation for action menu), D11/D12 (Beyond Binary / Beh. Calib. RL - training analogs), D7 (AbstentionBench - abstention baseline), D9 (ForecastBench - market eval precedent).
4. **Hammer the wedge:** Single sentence — *"PROPHET is the first benchmark to combine **(a)** procedurally generated tasks across 12 capability families with mechanical verifiers, **(b)** an explicit TAKE/QUOTE/PASS action menu, **(c)** bounded Brier-based proper scoring with an asymmetric `[-3κ, +κ]` overconfidence band, and **(d)** a Pareto-frontier rather than scalar leaderboard."*

---

## Procgen Templates (3-5 ready-to-implement)

### P1 — Hard Knowledge: 4-Hop Wikidata Composition (`knowledge` family)

**Idea.** Sample a random 4-hop path through Wikidata properties; phrase as a single natural-language question; mechanical verifier checks against the SPARQL answer.

**Mechanical recipe:**
```python
# Pseudo-template
from rdflib import Graph
import random

PROPS_HOP1 = ["P19_birthplace", "P27_country_of_citizenship"]
PROPS_HOP2 = ["P36_capital", "P30_continent"]
PROPS_HOP3 = ["P38_currency", "P31_instance_of"]
PROPS_HOP4 = ["P2138_exchange_rate"]  # or similar terminal numeric prop

def sample_4hop(seed):
    random.seed(seed)
    start_entity = random.choice(notable_persons_post_1980)
    p1 = random.choice(PROPS_HOP1)
    p2 = random.choice(PROPS_HOP2)
    p3 = random.choice(PROPS_HOP3)
    p4 = random.choice(PROPS_HOP4)
    answer = sparql_chain(start_entity, [p1, p2, p3, p4])
    if not answer: return None  # retry seed
    question = phrase_template(start_entity, [p1, p2, p3, p4])
    return Task(question=question, answer=answer, seed=seed,
                difficulty_features={"hops": 4, "rarest_entity_pageviews": ...})

def verify(model_answer, gold_answer):
    # numeric tolerance ±1% for numeric; canonical entity ID for entities
    return canonical_match(model_answer, gold_answer)
```

**Difficulty knobs:**
- Number of hops (3–6 tier).
- Entity rarity (pageview percentile).
- Property obscurity (e.g., `P361_part_of` vs `P3829_PubMed`).
- Temporal disambiguation (was, as of {year}).
- Numeric vs categorical terminal.

**Expected frontier scores at 4-hop, low-pageview entities:** <20% based on FRAMES no-retrieval baselines (~40% at typical multi-hop) and BBEH-style difficulty amplification.

### P2 — Hard Multilingual: Low-Resource Morphology Parse (`multilingual` family)

**Idea.** Procedurally generate inflected word forms in a polysynthetic / agglutinative low-resource language (e.g., Quechua, Inuktitut, Turkish, Swahili) using a hand-written FST + morpheme inventory; model must segment + gloss.

**Mechanical recipe:**
```python
# Pseudo-template
from hfst import HfstTransducer

MORPHEMES = {
    "ku": {"root": "see", "tag": "V"},
    "ma": {"tag": "PROG"},
    "na": {"tag": "1SG"},
    ...
}

def sample_word(seed, lang="quz", min_morphemes=4, max_morphemes=8):
    random.seed(seed)
    n = random.randint(min_morphemes, max_morphemes)
    morpheme_chain = sample_grammatical_chain(n, lang_grammar)
    surface_form = fst_apply(morpheme_chain)
    gold_gloss = format_gloss(morpheme_chain)
    return Task(prompt=f"Parse '{surface_form}' into morphemes and gloss.",
                gold=gold_gloss, seed=seed)

def verify(model_output, gold):
    parsed = parse_gloss_string(model_output)  # e.g., "see-PROG-1SG"
    return parsed == gold  # exact match, with synonyms set
```

**Difficulty knobs:**
- Morpheme count (4–8).
- Allomorphy (vowel harmony, consonant gradation, sandhi).
- Stem rarity.
- Choice of language (Swahili → Quechua → Inuktitut on resource scale).

**Expected frontier scores:** <20% for >5 morphemes in Quechua/Inuktitut, supported by AfroBench and MMLU-ProX Yoruba/Swahili gaps and the Apple Multilingual Reasoning Gym data.

### P3 — Hard Multimodal: Killer / Anti-Knight 9×9 Sudoku Variant (`multimodal` family)

**Idea.** Use Sudoku4LLM + a variant-rule generator; ASCII-encode the puzzle; mechanical verifier checks constraint satisfaction.

**Mechanical recipe:**
```python
from sudoku_gen import generate_killer_sudoku, generate_anti_knight, ascii_render

def sample_sudoku(seed, variant="anti_knight", min_clues=22, max_clues=28):
    random.seed(seed)
    puzzle, solution = generate_with_unique_solution(variant, seed,
                                                     clue_count=random.randint(min_clues, max_clues))
    ascii_grid = ascii_render(puzzle)  # 9-line ASCII with constraint markers
    return Task(prompt=ascii_grid + variant_rules_text(variant),
                gold=solution, seed=seed)

def verify(model_output, gold_solution):
    extracted = parse_9x9_grid(model_output)
    return (extracted == gold_solution and
            check_variant_constraints(extracted, variant))
```

**Difficulty knobs:**
- Variant rule complexity (vanilla < anti-knight < killer < thermo < arrow).
- Clue count.
- Symmetry of clue placement.
- Combined variants (anti-knight + killer).

**Expected frontier scores:** Sudoku-Bench data shows <15% unaided on 9×9 variants; GPT-5 only just solved its first modern 9×9 variant.

### P4 — Hard Multimodal: ASCII-Art Spatial Transformation (`multimodal` family)

**Idea.** Render a simple geometric shape (arrow, rectangle, letter) in ASCII; ask the model to apply a transformation (rotate 90°, mirror, scale by 2x) and reproduce the ASCII result. Mechanical verifier compares character-by-character on a canonical grid.

**Mechanical recipe:**
```python
def sample_ascii_xform(seed, shape_set=("L", "T", "F", "arrow_right", "rectangle"),
                       xforms=("rotate_90", "rotate_180", "mirror_v", "mirror_h",
                                "scale_2x", "negate")):
    random.seed(seed)
    shape = random.choice(shape_set)
    xform = random.choice(xforms)
    grid = render_ascii(shape, size=random.randint(5, 9))
    gold_grid = apply_xform(grid, xform)
    prompt = f"Below is an ASCII grid. Apply transformation '{xform}'.\n\n{grid}\n"
    return Task(prompt=prompt, gold=gold_grid, seed=seed)

def verify(model_output, gold_grid):
    grid = extract_code_block_or_grid(model_output)
    return canonicalise(grid) == canonicalise(gold_grid)
```

**Difficulty knobs:**
- Shape complexity (block letter < arrow < arbitrary polygon).
- Grid size (5×5 to 15×15).
- Transformation chain length (1, 2, or 3 stacked).

**Expected frontier scores:** ASCIIBench / ASCIIEval data shows text-only models drop dramatically on visual perception in text; even GPT-4o text-only is far below GPT-4o vision-mode (82%). Chained transformations push <20%.

### P5 — Hard Knowledge: Adversarial Temporal Disambiguation (`knowledge` family)

**Idea.** Pick a fact that has changed multiple times across years (e.g., "CEO of X Corp", "currency of Y country", "tallest building in Z"). Ask "as of date D". Mechanical verifier checks against a year-keyed Wikidata snapshot.

**Mechanical recipe:**
```python
def sample_temporal(seed, fact_pool=temporal_facts_wikidata):
    random.seed(seed)
    entity, prop, year_to_value = random.choice(fact_pool)
    date = random_date_with_min_2_changes_before(entity, prop)
    gold = year_to_value[date.year]
    prompt = f"Who/what was {entity}'s {prop} as of {date.isoformat()}?"
    return Task(prompt=prompt, gold=gold, seed=seed,
                difficulty_features={"changes_before_date": ..., "value_rarity": ...})
```

**Difficulty knobs:**
- Number of changes before query date.
- Proximity to a change date (within 30 days).
- Pre- vs post-cutoff dates.

**Expected frontier scores:** FRAMES temporal-disambiguation subset (~16% of FRAMES) drags frontier accuracy down significantly; adversarial near-change dates push <20%.

---

## Recommendations for PROPHET

1. **Rename or disambiguate** to avoid Prophet Arena name collision. **High priority.**
2. **Cite D1, D6, D8, D16 prominently** in introduction; D7, D9, D11, D12 in related-work.
3. **In Related Work, frame the wedge as a 2x2 matrix**: [live-market vs procgen] × [probability-elicitation vs action-menu], with PROPHET as the unique [procgen × action-menu] cell.
4. **Re-use Reasoning Gym (D-spotlight at NeurIPS 2025)** as a procgen dependency for the `reasoning` family — cite, don't reimplement.
5. **Re-use Apple's Multilingual Reasoning Gym** for `multilingual` family — gives a parallel multi-language template inventory.
6. **Re-use Sudoku4LLM** for the Sudoku component of `multimodal`.
7. **Adopt FormalEval / FermiEval-style strict interval scoring** if PROPHET ever measures continuous predictions.
8. **Stress the asymmetric `[-3κ, +κ]` band** as an empirical safety choice (BAS proves theoretical asymmetry; we operationalize a bounded version).
9. **Shadow re-attempt for abstention precision** appears genuinely novel — no surveyed benchmark uses this. Lean on it.
10. **The Pareto frontier with hypervolume scalar** is shared with COM-BOM (2510.01178) for accuracy-calibration; we extend to (net payoff, ECE, cost, abstention precision) — emphasize the multi-objective extension.

---

## Sources (consolidated)

Knowledge / multi-hop:
- HLE: https://arxiv.org/abs/2501.14249
- FrontierMath: https://arxiv.org/abs/2411.04872 ; https://epoch.ai/frontiermath
- GPQA Diamond: https://artificialanalysis.ai/evaluations/gpqa-diamond
- FRAMES: https://arxiv.org/abs/2409.12941
- MuSiQue: https://arxiv.org/pdf/2108.00573
- MultiHop-RAG: https://arxiv.org/abs/2401.15391
- BBEH: https://arxiv.org/abs/2502.19187
- GroundCocoa: https://aclanthology.org/2025.naacl-long.420.pdf
- AA-Omniscience: https://arxiv.org/abs/2511.13029
- SimpleQA Verified: https://arxiv.org/abs/2509.07968
- BrowseComp: https://openai.com/index/browsecomp/
- NoLiMa: https://arxiv.org/abs/2502.05167
- Chemistry multi-hop: https://arxiv.org/abs/2504.16414

Multimodal / spatial / symbolic:
- ARC-AGI-2: https://arxiv.org/abs/2505.11831
- ARC-AGI-3: https://arxiv.org/abs/2603.24621
- ZeroBench: https://arxiv.org/abs/2502.09696
- MMMU-Pro: https://arxiv.org/abs/2409.02813
- Sudoku-Bench: https://arxiv.org/abs/2505.16135
- Sudoku4LLM: https://github.com/DolbyUUU/Sudoku4LLM
- ChartQAPro: https://aclanthology.org/2025.findings-acl.978/
- ASCIIBench: https://arxiv.org/abs/2512.04125
- ASCIIEval: https://arxiv.org/html/2410.01733v2
- Reasoning Gym: https://github.com/open-thought/reasoning-gym

Multilingual:
- PolyMath: https://arxiv.org/abs/2504.18428
- MMLU-ProX: https://arxiv.org/abs/2503.10497
- AfroBench: https://aclanthology.org/2025.findings-acl.976.pdf
- Multilingual Reasoning Gym: https://machinelearning.apple.com/research/multilingual-reasoning-gym
- Cross-lingual Cantonese/Japanese/Turkish: https://arxiv.org/abs/2511.10664
- CRUXEval-X: https://aclanthology.org/2025.acl-long.1158/
- IrokoBench: https://aclanthology.org/2025.naacl-long.139.pdf

Calibration / market / proper scoring:
- Prophet Arena: https://arxiv.org/abs/2510.17638 ; https://www.prophetarena.co/leaderboard
- KalshiBench: https://arxiv.org/abs/2512.16030
- PolyBench: https://arxiv.org/abs/2604.14199
- Prediction Arena: https://arxiv.org/abs/2604.07355
- PrediBench: https://huggingface.co/blog/charles-azam/predibench ; https://predibench.com/
- Going All-In: https://www.arxiv.org/abs/2512.05998
- AbstentionBench: https://arxiv.org/abs/2506.09038
- BAS: https://arxiv.org/abs/2604.03216
- ForecastBench: https://www.forecastbench.org/
- Beyond Binary Rewards: https://arxiv.org/abs/2507.16806
- Behaviorally Calibrated RL: https://arxiv.org/abs/2512.19920
- FermiEval: https://arxiv.org/abs/2510.26995
- Trust or Escalate (ICLR 2025): https://iclr.cc/virtual/2025/oral/31838
- SelectLLM (OpenReview): https://openreview.net/forum?id=JJPAy8mvrQ
- Are LLM Decisions Faithful to Verbal Confidence?: https://arxiv.org/abs/2601.07767
- HalluLens: https://aclanthology.org/2025.acl-long.1176.pdf
- Composite Reliability Score: https://arxiv.org/abs/2512.24058
- Forecaster Arena: https://forecasterarena.com/
- Future Is Unevenly Distributed: https://arxiv.org/abs/2511.18394
- MarketBench: https://arxiv.org/abs/2512.12264
- PolySwarm: https://arxiv.org/abs/2604.03888
- Cost-Saving LLM Cascades: https://arxiv.org/html/2502.09054v1
- Know Your Limits (TACL): https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00754/131566/
- COM-BOM (Pareto): https://arxiv.org/abs/2510.01178

**Total papers covered: ~45.** Top 16 detailed: HLE, FrontierMath, FRAMES, BBEH, AA-Omn, NoLiMa, ARC-AGI-2/3, ZeroBench, MMMU-Pro, Sudoku-Bench, ASCIIBench, PolyMath, MMLU-ProX, AfroBench, Multilingual Reasoning Gym, Prophet Arena, KalshiBench, PolyBench, PrediBench, Going All-In, AbstentionBench, BAS, ForecastBench, FermiEval, Verbal-Confidence-Decisions.
