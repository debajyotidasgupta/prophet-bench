"""Statistical / scoring tests — guard against regressions in core math."""

from __future__ import annotations

import math

import numpy as np

from prophet.analysis.stats import (
    benjamini_hochberg,
    bootstrap_ci_brier,
    bootstrap_ci_ece,
    bootstrap_ci_mean,
    holm_bonferroni,
    mcnemar_exact,
    mde_paired_t,
    paired_t,
    permutation_test,
    required_n_paired_t,
    wilcoxon_signed_rank,
)
from prophet.engine.scoring import (
    adaptive_ece,
    brier_score,
    compute_payoff,
    ece,
    log_score,
    model_overreach_point,
)
from prophet.engine.types import AgentResponse, DecisionMode, MarketOffer


# ---------- Core payoff math --------------------------------------------

def _resp(mode: DecisionMode, p: float) -> AgentResponse:
    return AgentResponse(task_id="t", mode=mode, confidence=p, answer="x" if mode != DecisionMode.PASS else None)


def _offer() -> MarketOffer:
    return MarketOffer(task_id="t", family="f", V_success=100, C_failure=100, kappa_calib=50, delta_pass=2)


def test_take_success_yields_V():
    a, c, t = compute_payoff(_resp(DecisionMode.TAKE, 0.9), True, _offer())
    assert a == 100 and c == 0 and t == 100


def test_take_failure_yields_neg_C():
    a, c, t = compute_payoff(_resp(DecisionMode.TAKE, 0.9), False, _offer())
    assert a == -100 and c == 0 and t == -100


def test_quote_perfect_p1_success_gives_V_plus_kappa():
    a, c, t = compute_payoff(_resp(DecisionMode.QUOTE, 1.0), True, _offer())
    assert math.isclose(c, 50.0)
    assert math.isclose(t, 150.0)


def test_quote_uniform_p05_payoff_calib_zero():
    a, c, t = compute_payoff(_resp(DecisionMode.QUOTE, 0.5), True, _offer())
    assert math.isclose(c, 0.0)


def test_pass_yields_neg_delta():
    _, _, t = compute_payoff(_resp(DecisionMode.PASS, 0.5), None, _offer())
    assert math.isclose(t, -2.0)


# ---------- Calibration metrics -----------------------------------------

def test_perfect_calibration_zero_ece():
    n = 1000
    rng = np.random.default_rng(0)
    p = rng.random(n)
    y = (rng.random(n) < p).astype(int)
    e = ece(p.tolist(), y.tolist(), n_bins=20)
    assert e < 0.05, e


def test_always_one_overconfident_high_ece():
    p = [1.0] * 100
    y = [0] * 50 + [1] * 50
    assert ece(p, y) > 0.4


def test_brier_zero_at_perfect():
    assert brier_score([0.0, 1.0], [0, 1]) == 0.0


def test_logloss_finite_at_extremes():
    # We clip; should not be inf
    v = log_score([0.0, 1.0], [0, 1])
    assert math.isfinite(v)


def test_adaptive_ece_handles_skew():
    n = 200
    rng = np.random.default_rng(1)
    p = np.concatenate([rng.uniform(0.0, 0.05, n - 5), rng.uniform(0.95, 1.0, 5)])
    y = (rng.random(len(p)) < p).astype(int)
    v = adaptive_ece(p.tolist(), y.tolist(), n_bins=10)
    assert 0.0 <= v <= 1.0


def test_mop_detects_overreach():
    rng = np.random.default_rng(2)
    confs = rng.uniform(0.85, 0.99, 200).tolist()
    outs = [0] * 200  # confidently wrong everywhere
    diffs = rng.uniform(0.5, 1.0, 200).tolist()
    res = model_overreach_point(confs, outs, diffs)
    assert res.max_overreach > 0.8


# ---------- Bootstrap CIs -----------------------------------------------

def test_bootstrap_ci_mean_brackets_truth():
    rng = np.random.default_rng(3)
    vals = rng.normal(0.5, 0.1, 500).tolist()
    pt, lo, hi = bootstrap_ci_mean(vals, n_boot=2000, seed=3)
    assert lo < 0.5 < hi


def test_bootstrap_ci_ece_consistent_with_ece():
    rng = np.random.default_rng(4)
    p = rng.random(200)
    y = (rng.random(200) < p).astype(int).tolist()
    e_pt, lo, hi = bootstrap_ci_ece(p.tolist(), y, n_boot=1000, seed=4)
    e_direct = ece(p.tolist(), y)
    assert abs(e_pt - e_direct) < 1e-9
    assert lo <= e_pt <= hi


# ---------- Paired significance tests ------------------------------------

def test_paired_t_detects_real_diff():
    rng = np.random.default_rng(5)
    a = rng.normal(0.5, 0.1, 200)
    b = a + rng.normal(0.2, 0.1, 200)  # b shifted higher
    res = paired_t(a, b)
    assert res.pvalue < 1e-10
    assert res.effect < 0


def test_paired_t_no_diff_when_same_distribution():
    rng = np.random.default_rng(6)
    a = rng.normal(0.5, 0.1, 100)
    b = a + rng.normal(0.0, 0.1, 100)
    res = paired_t(a, b)
    assert res.pvalue > 0.05


def test_wilcoxon_robust_to_outliers():
    rng = np.random.default_rng(7)
    a = rng.normal(0.5, 0.1, 100)
    b = a + 0.2
    # Inject 5 huge outliers
    b[:5] -= 5
    res = wilcoxon_signed_rank(a, b)
    assert res.pvalue < 1e-5


def test_mcnemar_detects_paired_binary_diff():
    # A right on 90 tasks, B right on 60 of the same; b=30, c=0 → highly significant
    n = 100
    a = [1] * 90 + [0] * 10
    b = [1] * 60 + [0] * 40
    res = mcnemar_exact(a, b)
    assert res.pvalue < 1e-3


def test_permutation_test_finds_difference():
    rng = np.random.default_rng(8)
    confs_a = rng.uniform(0, 1, 200)
    outs_a = (rng.random(200) < confs_a).astype(int)  # well-calibrated
    confs_b = np.ones(200) * 0.99
    outs_b = (rng.random(200) < 0.5).astype(int)  # over-confident
    res = permutation_test(
        a_metric_fn=lambda c, o: ece(c.tolist(), o.tolist()),
        b_metric_fn=lambda c, o: ece(c.tolist(), o.tolist()),
        confidences_a=confs_a,
        outcomes_a=outs_a,
        confidences_b=confs_b,
        outcomes_b=outs_b,
        n_perm=1000,
        seed=8,
    )
    assert res.pvalue < 0.05


# ---------- Multiple-comparison correction ------------------------------

def test_holm_bonferroni_reject_smallest_only():
    decisions = holm_bonferroni([0.001, 0.04, 0.5], alpha=0.05)
    assert decisions[0] is True
    assert decisions[1] is False  # 0.04 > 0.05/2 = 0.025
    assert decisions[2] is False


def test_holm_bonferroni_all_significant():
    decisions = holm_bonferroni([1e-6, 1e-5, 1e-4], alpha=0.05)
    assert all(decisions)


def test_benjamini_hochberg_basic():
    p = [0.001, 0.008, 0.03, 0.05, 0.5]
    decisions = benjamini_hochberg(p, q=0.05)
    assert decisions[0] is True
    assert decisions[-1] is False


# ---------- Power analysis ----------------------------------------------

def test_mde_decreases_with_n():
    assert mde_paired_t(20) > mde_paired_t(200) > mde_paired_t(1000)


def test_required_n_increases_for_smaller_effects():
    assert required_n_paired_t(0.8) < required_n_paired_t(0.2) < required_n_paired_t(0.05)
