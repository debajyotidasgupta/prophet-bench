# PROPHET — Design Document

## Goal

A benchmark whose primary measurement is **calibrated, persistent agency**:
how well an LLM agent knows what it knows and prices its commitments before
knowing the outcome.

## Why a market?

A scalar leaderboard incentivises agents to maximise accuracy at any cost.
A market instead reveals whether the agent is *useful* to a buyer with a
budget — the deployment context that matters in production. The marketplace
also induces a **proper scoring rule**: agents maximise expected payoff iff
they report their true probability of success.

## Mechanism

For each task `t` from family `f`, the agent observes the prompt and a
deterministic market offer `(V, C, κ, δ_pass)_f`. The agent commits to one
of three actions:

| Action | Payoff |
|---|---|
| TAKE  | `V · y - C · (1 - y)` where `y ∈ {0,1}` is the mechanical verifier outcome |
| QUOTE | TAKE-payoff plus `κ · (1 - 4·(P̂ - y)²)` (Brier-based proper scoring) |
| PASS  | `-δ_pass` |

**Why Brier and why factor 4?** The Brier loss `(P̂ - y)²` is in `[0, 1/4]`
when `y ∈ {0,1}` and `P̂ ∈ [0,1]`. The factor `4` rescales the maximum loss
to 1.0, so the calibration term lies in `[-3κ, +κ]`. A `P̂ = 0.5` against a
binary outcome receives a calibration payoff of exactly 0, removing the
"always-uniform" gaming strategy.

**Why the asymmetric `[-3κ, +κ]` band?** Over-confidence is more harmful in
production than under-confidence (false confidence drives real bad decisions
while caution merely costs research effort). The asymmetric band reflects
this without breaking properness.

## Headline metrics

* **Net payoff** — sum of payoffs across the run.
* **ECE** — expected calibration error (10 equal-width or 15 adaptive bins).
* **Brier** — `(P̂ - y)²` mean.
* **Log-loss** — clipped negative log-likelihood.
* **MOP** — Model Overreach Point, the difficulty bin where `P̂` exceeds
  empirical accuracy by ≥2σ.
* **Abstention precision** — among PASSed tasks, fraction that would have
  failed if attempted (estimated via shadow re-attempt).

## Leaderboard

We never publish a single scalar ranking. Instead the headline is a
**Pareto frontier** of `(net payoff, ECE)` with secondary frontiers on
`(cost USD, accuracy)` and `(Brier, abstention precision)`. The hypervolume
of the Pareto frontier is reported as a summary scalar for comparison
across cycles.

## Task families (12)

| Family | Capability | Verifier |
|---|---|---|
| math | competition-style problems | Fraction equality |
| code | predict / write / fix programs | sandboxed exec |
| knowledge | closed-form factual QA | exact / synonym set |
| reasoning | ARC-like grids + logic | JSON-equality |
| writing | constrained text (regex/acrostic) | regex / parse |
| tools | tool-use chains in pure-fn sandbox | replay equality |
| browser | text-only browser simulator | replay equality |
| multimodal | ASCII-encoded multimodal tasks | exact match |
| data | data-analysis on small CSVs | numeric tolerance |
| scientific | physics/chem/bio closed-form | numeric tolerance |
| multilingual | language-id + cross-lingual ops | exact / lookup |
| safety | should-refuse-or-comply routing | refusal-token / answer |

Each family ships 6-8 procedurally generated difficulty tiers, calibrated
empirically by a reference panel (see `scripts/calibrate_difficulty.py`).

## Gaming resistance

1. **Procedural generation** — no fixed test set; cycle seeds rotate.
2. **Deterministic market noise** — agents cannot pre-compute prices.
3. **Mechanical verifiers** — primary metrics never use an LLM judge.
4. **Shadow re-attempt** — pass-only strategies are not protected.
5. **Pareto frontier** — single-axis gaming is captured.
6. **Pre-publication red-team** — Berkeley RDI attack family is run
   against every release; any successful exploit retires the offending
   submission category.

## Statistical protocol

* Bootstrap CIs (B=10⁴, percentile) on every aggregate metric.
* Paired tests for two-system comparisons: paired t (n ≥ 30, symmetric),
  paired Wilcoxon, McNemar exact (binary), permutation test on ECE.
* Multiple-comparison correction: Holm-Bonferroni (FWER) by default; BH-FDR
  available for large pairwise matrices.
* Effect sizes alongside every p-value: Cohen's d / rank-biserial /
  paired odds-ratio / |Δ ECE|.
* Post-hoc power analysis: minimum-detectable effect and required-n
  tables in appendix.

## Reproducibility

* Single seed reproduces the entire benchmark cycle.
* Public Docker images for the engine and per-model evaluation containers.
* HF dataset of seeds + reference panel scores.
* `scripts/run_full_eval.sh` produces all paper-ready tables / figures.
* GitHub repo is private until paper submission; flips public on release.
