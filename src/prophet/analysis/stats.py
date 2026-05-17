"""Statistical-rigor module for PROPHET.

Every headline number is reported with a confidence interval and, where
two agents (or two configurations) are compared, with a paired significance
test. Multiple-comparison correction is applied when comparing >2 systems.

Design choices justified:

* **Bootstrap CIs** on ECE / Brier / payoff. Closed-form CIs for ECE are
  intractable; bootstrap is the standard.
* **Paired tests** wherever possible. Two agents see the same task instances;
  treating outcomes as unpaired throws away variance reduction and inflates
  p-values.
* **Test selection**:
    - Continuous metric per task (e.g. payoff, log-loss): paired *t* if
      n ≥ 30 and roughly symmetric, else **Wilcoxon signed-rank** (rank-based,
      no normality assumption).
    - Binary outcome per task (correct/incorrect): **McNemar's exact test**.
    - ECE / Brier difference (group-level metric): **permutation test**
      with B=10,000.
* **Multiple comparisons**: Holm-Bonferroni by default (controls
  family-wise error). BH-FDR available when running many pairwise tests on
  per-family slices. We *always* report unadjusted p in parallel; reviewers
  hate hidden adjustments.
* **Effect sizes**: Cohen's d for continuous; odds-ratio for binary;
  Δ-ECE / Δ-Brier in absolute units. We never report a p without an effect.
* **Power analysis**: post-hoc minimum-detectable-effect (MDE) for the
  realized n and chosen α / power.

Anchor papers and tests adopted directly:
  - Demsar (2006) on comparing classifiers: prefer non-parametric paired tests.
  - Dietterich (1998) on McNemar for paired binary.
  - Efron & Tibshirani (1993) on bootstrap.
  - Holm (1979) for sequential Bonferroni.
  - Benjamini & Hochberg (1995) for FDR.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from prophet.engine.scoring import brier_score, ece, log_score

log = logging.getLogger("prophet.stats")


# -------------------------------------------------------------------------
# Bootstrap CIs
# -------------------------------------------------------------------------

def bootstrap_ci(
    fn,
    data: Sequence,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    method: str = "percentile",
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a scalar statistic.

    Args:
      fn: callable(subsample) -> float
      data: list of items (will be sampled with replacement)
      n_boot: number of bootstrap replicates
      ci: e.g. 0.95
      method: 'percentile' (default) or 'bca' (bias-corrected accelerated)

    Returns (point_estimate, lower, upper).
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(data, dtype=object)
    point = float(fn(list(arr)))
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = float(fn(arr[idx].tolist()))
    alpha = (1.0 - ci) / 2.0
    if method == "bca":
        # Bias correction
        z0 = sps.norm.ppf(np.mean(boots < point))
        # Jackknife acceleration
        jacks = np.empty(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            jacks[i] = float(fn(arr[mask].tolist()))
        jbar = jacks.mean()
        num = ((jbar - jacks) ** 3).sum()
        den = 6.0 * ((jbar - jacks) ** 2).sum() ** 1.5
        a = 0.0 if den < 1e-12 else num / den
        zlo = sps.norm.ppf(alpha)
        zhi = sps.norm.ppf(1 - alpha)
        plo = sps.norm.cdf(z0 + (z0 + zlo) / (1 - a * (z0 + zlo)))
        phi = sps.norm.cdf(z0 + (z0 + zhi) / (1 - a * (z0 + zhi)))
        lo = float(np.quantile(boots, plo))
        hi = float(np.quantile(boots, phi))
    else:
        lo = float(np.quantile(boots, alpha))
        hi = float(np.quantile(boots, 1 - alpha))
    return point, lo, hi


def bootstrap_ci_ece(
    confidences: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
    **kw,
) -> tuple[float, float, float]:
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(outcomes, dtype=int)
    pairs = list(zip(c.tolist(), o.tolist()))
    return bootstrap_ci(
        lambda p: ece([x[0] for x in p], [x[1] for x in p], n_bins=n_bins),
        pairs,
        **kw,
    )


def bootstrap_ci_brier(
    confidences: Sequence[float], outcomes: Sequence[int], **kw
) -> tuple[float, float, float]:
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(outcomes, dtype=int)
    pairs = list(zip(c.tolist(), o.tolist()))
    return bootstrap_ci(
        lambda p: brier_score([x[0] for x in p], [x[1] for x in p]),
        pairs,
        **kw,
    )


def bootstrap_ci_logloss(
    confidences: Sequence[float], outcomes: Sequence[int], **kw
) -> tuple[float, float, float]:
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(outcomes, dtype=int)
    pairs = list(zip(c.tolist(), o.tolist()))
    return bootstrap_ci(
        lambda p: log_score([x[0] for x in p], [x[1] for x in p]),
        pairs,
        **kw,
    )


def bootstrap_ci_mean(
    values: Sequence[float], **kw
) -> tuple[float, float, float]:
    v = list(values)
    return bootstrap_ci(lambda x: float(np.mean(x)), v, **kw)


# -------------------------------------------------------------------------
# Paired significance tests
# -------------------------------------------------------------------------

@dataclass(slots=True)
class TestResult:
    test: str
    statistic: float
    pvalue: float
    n: int
    effect: float
    effect_name: str
    note: str = ""


def paired_t(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Paired Student's t-test. Use when n ≥ 30 and approx symmetric."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b), "paired_t requires equal-length samples"
    n = len(a)
    res = sps.ttest_rel(a, b, nan_policy="omit")
    d = a - b
    # Cohen's d for paired samples
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    cohen = float(np.mean(d)) / sd if sd > 0 else 0.0
    return TestResult(
        test="paired_t",
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        n=n,
        effect=cohen,
        effect_name="cohen_d",
    )


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Paired non-parametric. Use for small samples or skewed differences."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b)
    n = len(a)
    res = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", correction=False)
    d = a - b
    # r (rank-biserial) effect size
    pos = float((d > 0).sum())
    neg = float((d < 0).sum())
    eff = (pos - neg) / max(pos + neg, 1.0)
    return TestResult(
        test="wilcoxon",
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        n=n,
        effect=eff,
        effect_name="rank_biserial_r",
    )


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> TestResult:
    """McNemar's exact test for paired binary outcomes (0/1).

    a, b are correctness (1/0) of system A and B on the same tasks.
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = len(a)
    b_ab = int(((a == 1) & (b == 0)).sum())  # A right, B wrong
    c_ab = int(((a == 0) & (b == 1)).sum())  # A wrong, B right
    # Use exact binomial under H0: b_ab ~ Binom(b+c, 0.5)
    nn = b_ab + c_ab
    if nn == 0:
        return TestResult("mcnemar_exact", 0.0, 1.0, n, 0.0, "odds_ratio_paired")
    # Two-sided
    k = min(b_ab, c_ab)
    pval = float(2.0 * sps.binom.cdf(k, nn, 0.5))
    pval = min(1.0, pval)
    # Effect: odds ratio of disagreement
    or_ = (b_ab + 0.5) / (c_ab + 0.5)
    return TestResult(
        test="mcnemar_exact",
        statistic=float(b_ab - c_ab),
        pvalue=pval,
        n=n,
        effect=float(or_),
        effect_name="paired_odds_ratio",
        note=f"b={b_ab} c={c_ab}",
    )


def permutation_test(
    a_metric_fn,
    b_metric_fn,
    confidences_a: Sequence[float],
    outcomes_a: Sequence[int],
    confidences_b: Sequence[float],
    outcomes_b: Sequence[int],
    n_perm: int = 10000,
    seed: int = 0,
) -> TestResult:
    """Permutation test for difference in group-level metrics (e.g. ECE).

    The two systems' (confidence, outcome) pairs are shuffled between
    populations; we test how often a randomly permuted assignment yields a
    larger absolute Δ in the chosen metric than the observed one.
    """
    rng = np.random.default_rng(seed)
    ca = np.asarray(confidences_a, dtype=float)
    oa = np.asarray(outcomes_a, dtype=int)
    cb = np.asarray(confidences_b, dtype=float)
    ob = np.asarray(outcomes_b, dtype=int)
    if len(ca) == 0 or len(cb) == 0:
        return TestResult("permutation", 0.0, 1.0, 0, 0.0, "delta")
    obs = abs(a_metric_fn(ca, oa) - b_metric_fn(cb, ob))
    pooled_c = np.concatenate([ca, cb])
    pooled_o = np.concatenate([oa, ob])
    n_a = len(ca)
    n_total = len(pooled_c)
    count_ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(n_total)
        ca_p = pooled_c[idx[:n_a]]
        oa_p = pooled_o[idx[:n_a]]
        cb_p = pooled_c[idx[n_a:]]
        ob_p = pooled_o[idx[n_a:]]
        d = abs(a_metric_fn(ca_p, oa_p) - b_metric_fn(cb_p, ob_p))
        if d >= obs:
            count_ge += 1
    p = (count_ge + 1) / (n_perm + 1)
    return TestResult(
        test="permutation",
        statistic=float(obs),
        pvalue=float(p),
        n=int(n_total),
        effect=float(obs),
        effect_name="abs_delta",
    )


def compare_two_agents(
    payoffs_a: Sequence[float],
    payoffs_b: Sequence[float],
    correct_a: Sequence[int] | None = None,
    correct_b: Sequence[int] | None = None,
) -> dict[str, TestResult]:
    """Run the standard paired test battery for two agents.

    Returns a dict keyed by metric name.
    """
    results: dict[str, TestResult] = {}
    pa = np.asarray(payoffs_a, dtype=float)
    pb = np.asarray(payoffs_b, dtype=float)
    if len(pa) == len(pb) and len(pa) > 0:
        # Continuous tests on payoff
        results["payoff_paired_t"] = paired_t(pa, pb)
        results["payoff_wilcoxon"] = wilcoxon_signed_rank(pa, pb)
    if correct_a is not None and correct_b is not None:
        results["accuracy_mcnemar"] = mcnemar_exact(correct_a, correct_b)
    return results


# -------------------------------------------------------------------------
# Multiple-comparison correction
# -------------------------------------------------------------------------

def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Return list of `reject H0` flags using Holm-Bonferroni sequential correction."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    decisions = [False] * m
    for rank, idx in enumerate(order):
        crit = alpha / (m - rank)
        if p[idx] <= crit:
            decisions[idx] = True
        else:
            break  # Holm stops at first failure
    return decisions


def benjamini_hochberg(pvalues: Sequence[float], q: float = 0.05) -> list[bool]:
    """BH-FDR. Returns list of `reject H0`."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1) / m)
    decisions = [False] * m
    cutoff = -1
    for rank, idx in enumerate(order):
        if p[idx] <= thresh[rank]:
            cutoff = rank
    for rank, idx in enumerate(order):
        if rank <= cutoff:
            decisions[idx] = True
    return decisions


# -------------------------------------------------------------------------
# Power analysis
# -------------------------------------------------------------------------

def mde_paired_t(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum-detectable Cohen's d for a paired t-test at this n."""
    if n < 3:
        return float("inf")
    df = n - 1
    t_alpha = sps.t.ppf(1 - alpha / 2, df)
    t_power = sps.t.ppf(power, df)
    return float((t_alpha + t_power) / math.sqrt(n))


def required_n_paired_t(
    cohen_d: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Required sample size for given target effect / α / power (Lehr's rule sanity-check)."""
    z_alpha = sps.norm.ppf(1 - alpha / 2)
    z_power = sps.norm.ppf(power)
    if cohen_d <= 0:
        return 10_000_000
    return int(math.ceil(((z_alpha + z_power) / cohen_d) ** 2))


def post_hoc_power_paired(
    a: Sequence[float], b: Sequence[float], alpha: float = 0.05
) -> tuple[float, float]:
    """Return (observed_cohen_d, post-hoc-power)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    d = a - b
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    cohen = float(np.mean(d)) / sd if sd > 0 else 0.0
    if n < 3 or sd == 0:
        return cohen, 0.0
    nc = abs(cohen) * math.sqrt(n)
    df = n - 1
    crit = sps.t.ppf(1 - alpha / 2, df)
    power = float(1 - sps.nct.cdf(crit, df, nc) + sps.nct.cdf(-crit, df, nc))
    return cohen, max(0.0, min(1.0, power))


# -------------------------------------------------------------------------
# Composite summary (the table the paper publishes)
# -------------------------------------------------------------------------

@dataclass(slots=True)
class AgentSummary:
    name: str
    n: int
    accuracy: float
    accuracy_ci: tuple[float, float]
    ece: float
    ece_ci: tuple[float, float]
    brier: float
    brier_ci: tuple[float, float]
    logloss: float
    logloss_ci: tuple[float, float]
    net_payoff: float
    net_payoff_ci: tuple[float, float]


def summarize_agent(
    confidences: Sequence[float],
    outcomes: Sequence[int],
    payoffs: Sequence[float],
    name: str,
    n_boot: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> AgentSummary:
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(outcomes, dtype=int)
    pay = np.asarray(payoffs, dtype=float)
    n = len(c)
    if n == 0:
        nan = float("nan")
        return AgentSummary(name, 0, nan, (nan, nan), nan, (nan, nan), nan, (nan, nan), nan, (nan, nan), nan, (nan, nan))
    acc_pt, acc_lo, acc_hi = bootstrap_ci_mean(o.tolist(), n_boot=n_boot, ci=ci, seed=seed)
    ece_pt, ece_lo, ece_hi = bootstrap_ci_ece(c, o, n_boot=n_boot, ci=ci, seed=seed)
    brier_pt, brier_lo, brier_hi = bootstrap_ci_brier(c, o, n_boot=n_boot, ci=ci, seed=seed)
    log_pt, log_lo, log_hi = bootstrap_ci_logloss(c, o, n_boot=n_boot, ci=ci, seed=seed)
    pay_pt, pay_lo, pay_hi = bootstrap_ci_mean(pay.tolist(), n_boot=n_boot, ci=ci, seed=seed)
    return AgentSummary(
        name=name,
        n=n,
        accuracy=acc_pt,
        accuracy_ci=(acc_lo, acc_hi),
        ece=ece_pt,
        ece_ci=(ece_lo, ece_hi),
        brier=brier_pt,
        brier_ci=(brier_lo, brier_hi),
        logloss=log_pt,
        logloss_ci=(log_lo, log_hi),
        net_payoff=pay_pt * n,  # report total payoff
        net_payoff_ci=(pay_lo * n, pay_hi * n),
    )
