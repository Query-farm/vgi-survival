"""Unit tests for the pure survival logic, validated against lifelines.

Uses lifelines' bundled datasets (``load_rossi``, ``load_waltons``) and a small
constructed cohort. These test the framework-free ``vgi_survival.survival``
functions directly: correct curves/coefficients plus the error edges.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.datasets import load_rossi, load_waltons
from lifelines.statistics import logrank_test as ll_logrank

from vgi_survival import survival
from vgi_survival.survival import SurvivalError

# --------------------------------------------------------------------------
# Kaplan-Meier
# --------------------------------------------------------------------------


def test_km_curve_starts_at_one_and_is_monotone() -> None:
    df = load_waltons()  # columns: T (duration), E (event), group
    out = survival.kaplan_meier(df, duration="T", event="E")
    surv = out["survival"]
    assert surv[0] == pytest.approx(1.0)
    # Monotonically non-increasing.
    assert all(b <= a + 1e-12 for a, b in zip(surv, surv[1:], strict=False))
    # CI brackets the estimate; at_risk is non-increasing and starts at N.
    for lo, s, hi in zip(out["ci_lower"], surv, out["ci_upper"], strict=False):
        assert lo <= s + 1e-9 <= hi + 2e-9
    assert out["at_risk"][0] == len(df)


def test_km_matches_lifelines_survival_function() -> None:
    df = load_waltons()
    out = survival.kaplan_meier(df, duration="T", event="E")
    kmf = KaplanMeierFitter().fit(df["T"], event_observed=df["E"])
    expected = kmf.survival_function_.iloc[:, 0].to_numpy()
    np.testing.assert_allclose(out["survival"], expected, rtol=1e-9)


# --------------------------------------------------------------------------
# Cox proportional hazards — known covariate sign/significance
# --------------------------------------------------------------------------


def test_cox_recovers_known_prio_effect_on_rossi() -> None:
    # Rossi recidivism data. 'prio' (# prior convictions) is a well-known
    # significant risk factor: more priors -> higher hazard of re-arrest
    # (hazard_ratio > 1, small p). Validate sign, significance, and that we
    # match lifelines' own fit exactly.
    df = load_rossi()  # week (duration), arrest (event=1), + covariates
    out = survival.cox_hazard_ratios(df, duration="week", event="arrest")
    by_cov = dict(zip(out["covariate"], range(len(out["covariate"])), strict=False))
    i = by_cov["prio"]

    assert out["coef"][i] > 0  # positive log-hazard
    assert out["hazard_ratio"][i] > 1.0  # raises hazard
    assert out["p_value"][i] < 0.01  # significant

    # Exact agreement with lifelines' direct fit.
    cph = CoxPHFitter().fit(df, duration_col="week", event_col="arrest")
    assert out["hazard_ratio"][i] == pytest.approx(float(cph.summary.loc["prio", "exp(coef)"]), rel=1e-9)
    assert out["p_value"][i] == pytest.approx(float(cph.summary.loc["prio", "p"]), rel=1e-9)
    # One row per covariate (all columns besides week/arrest).
    assert len(out["covariate"]) == df.shape[1] - 2


# --------------------------------------------------------------------------
# Log-rank
# --------------------------------------------------------------------------


def test_logrank_small_p_for_different_groups() -> None:
    df = load_waltons()  # two clearly different groups (miR-137 vs control)
    out = survival.logrank_test(df, duration="T", event="E", group="group")
    assert out["degrees_freedom"][0] == 1
    assert out["p_value"][0] < 0.001  # groups clearly differ

    # Matches lifelines' two-group logrank on the same split.
    g = df["group"].unique()
    a = df[df["group"] == g[0]]
    b = df[df["group"] == g[1]]
    expected = ll_logrank(a["T"], b["T"], a["E"], b["E"])
    assert out["test_statistic"][0] == pytest.approx(float(expected.test_statistic), rel=1e-9)


def test_logrank_identical_groups_large_p() -> None:
    base = pd.DataFrame({"t": [1, 2, 3, 4, 5, 6], "e": [1, 1, 0, 1, 0, 1]})
    df = pd.concat(
        [base.assign(arm="x"), base.assign(arm="y")], ignore_index=True
    )
    out = survival.logrank_test(df, duration="t", event="e", group="arm")
    assert out["p_value"][0] > 0.5  # no difference -> large p


# --------------------------------------------------------------------------
# Median survival
# --------------------------------------------------------------------------


def test_median_matches_lifelines() -> None:
    df = load_waltons()
    out = survival.median_survival(df, duration="T", event="E")
    kmf = KaplanMeierFitter().fit(df["T"], event_observed=df["E"])
    assert out["median_survival"][0] == pytest.approx(float(kmf.median_survival_time_))


def test_median_all_censored_is_inf() -> None:
    df = pd.DataFrame({"t": [1.0, 2.0, 3.0, 4.0], "e": [0, 0, 0, 0]})
    out = survival.median_survival(df, duration="t", event="e")
    assert math.isinf(out["median_survival"][0])


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------


def test_km_all_censored_stays_at_one() -> None:
    df = pd.DataFrame({"t": [1.0, 2.0, 3.0], "e": [0, 0, 0]})
    out = survival.kaplan_meier(df, duration="t", event="e")
    assert all(s == pytest.approx(1.0) for s in out["survival"])


def test_missing_column_errors() -> None:
    df = pd.DataFrame({"t": [1.0, 2.0], "e": [1, 0]})
    with pytest.raises(SurvivalError, match="missing required column"):
        survival.kaplan_meier(df, duration="duration", event="e")


def test_empty_relation_errors() -> None:
    df = pd.DataFrame({"t": pd.Series([], dtype=float), "e": pd.Series([], dtype=int)})
    with pytest.raises(SurvivalError, match="non-empty"):
        survival.kaplan_meier(df, duration="t", event="e")


def test_cox_requires_a_covariate() -> None:
    df = pd.DataFrame({"t": [1.0, 2.0, 3.0], "e": [1, 0, 1]})
    with pytest.raises(SurvivalError, match="at least one covariate"):
        survival.cox_hazard_ratios(df, duration="t", event="e")


def test_logrank_needs_two_groups() -> None:
    df = pd.DataFrame({"t": [1.0, 2.0, 3.0], "e": [1, 0, 1], "arm": ["x", "x", "x"]})
    with pytest.raises(SurvivalError, match="two distinct groups"):
        survival.logrank_test(df, duration="t", event="e", group="arm")


def test_non_numeric_duration_errors() -> None:
    df = pd.DataFrame({"t": ["a", "b", "c"], "e": [1, 0, 1]})
    with pytest.raises(SurvivalError, match="must be numeric"):
        survival.kaplan_meier(df, duration="t", event="e")
