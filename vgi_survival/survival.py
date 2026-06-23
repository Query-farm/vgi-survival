"""Pure survival / time-to-event analysis logic over lifelines.

This module is the framework-free core: it takes a ``pandas.DataFrame`` (the
buffered input relation) plus the column roles, runs the lifelines estimator,
and returns plain ``dict[str, list]`` column blocks ready to hand to pyarrow.
No VGI, no Arrow, no DuckDB here -- so every function is directly unit-testable.

Event-coding convention
------------------------
The ``event`` column is the *event indicator*: ``1`` / ``true`` means the event
of interest occurred (death / failure / churn) at ``duration``; ``0`` / ``false``
means the observation was right-censored (still alive / still in study) at
``duration``. This matches lifelines' ``event_observed`` convention.

Covariates for Cox
------------------
``cox_hazard_ratios`` treats *every column other than ``duration`` and
``event``* in the input relation as a covariate -- one fitted hazard ratio per
column. Select exactly the covariate columns you want in the ``(SELECT ...)``.

lifelines is MIT-licensed; numpy/pandas are BSD-licensed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Importing lifelines is expensive (pulls in scipy/autograd); do it once at
# module import so the per-call path is cheap. The worker imports this module
# at startup, so the cost is paid before the first SQL call.
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

__all__ = [
    "cox_hazard_ratios",
    "kaplan_meier",
    "logrank_test",
    "median_survival",
]


class SurvivalError(ValueError):
    """Raised for user-facing input problems (missing/empty/non-numeric columns).

    A plain, explicit error so the worker surfaces a clear message to SQL
    instead of crashing with an opaque pandas/lifelines traceback.
    """


def _require_columns(df: pd.DataFrame, required: dict[str, str]) -> None:
    """Validate that each required role maps to a present column.

    Args:
        df: The input relation.
        required: Mapping of role name (e.g. ``"duration"``) to the column
            name the caller passed for that role.

    Raises:
        SurvivalError: If any named column is absent from the relation.
    """
    have = set(df.columns)
    missing = {role: col for role, col in required.items() if col not in have}
    if missing:
        detail = ", ".join(f"{role} := '{col}'" for role, col in missing.items())
        raise SurvivalError(
            f"missing required column(s): {detail}; "
            f"input relation has columns: {', '.join(map(str, df.columns))}"
        )


def _numeric(df: pd.DataFrame, column: str, *, role: str) -> np.ndarray:
    """Coerce a column to a float64 numpy array or raise a clear error.

    Args:
        df: The input relation.
        column: Column name to coerce.
        role: Human-readable role label for the error message.

    Returns:
        The column as a contiguous float64 array.

    Raises:
        SurvivalError: If the column is not numeric / not coercible to float.
    """
    series = df[column]
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.isna().any() and not series.isna().any():
        raise SurvivalError(
            f"{role} column '{column}' must be numeric, but contains "
            f"non-numeric values (dtype {series.dtype})"
        )
    return np.asarray(coerced, dtype=float)


def _event_indicator(df: pd.DataFrame, column: str) -> np.ndarray:
    """Coerce the event column to a 0/1 integer indicator.

    Accepts numeric (0/1), boolean, or anything truthy-coercible. ``1``/true
    means the event occurred; ``0``/false means right-censored.

    Args:
        df: The input relation.
        column: Event column name.

    Returns:
        Integer array of 0/1 event indicators.

    Raises:
        SurvivalError: If the column cannot be interpreted as 0/1.
    """
    series = df[column]
    if series.dtype == bool:
        return series.to_numpy().astype(int)
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.isna().any():
        raise SurvivalError(
            f"event column '{column}' must be 0/1 or boolean, but contains "
            f"values that are neither (dtype {series.dtype})"
        )
    return (np.asarray(coerced, dtype=float) != 0).astype(int)


def kaplan_meier(
    df: pd.DataFrame,
    *,
    duration: str,
    event: str,
    alpha: float = 0.05,
) -> dict[str, list]:
    """Kaplan-Meier survival curve with confidence bands and at-risk counts.

    Args:
        df: Input relation; must contain ``duration`` and ``event`` columns.
        duration: Name of the time-to-event / follow-up column.
        event: Name of the 0/1 event-indicator column.
        alpha: Significance level for the confidence interval (default 0.05 →
            95% CI).

    Returns:
        Column block with keys ``time``, ``survival``, ``ci_lower``,
        ``ci_upper`` (all float), and ``at_risk`` (int), one entry per distinct
        observed time, ordered by time ascending. The curve starts at survival
        1.0 and is monotonically non-increasing.

    Raises:
        SurvivalError: On missing columns or empty input.
    """
    _require_columns(df, {"duration": duration, "event": event})
    if len(df) == 0:
        raise SurvivalError("kaplan_meier requires a non-empty input relation")

    durations = _numeric(df, duration, role="duration")
    events = _event_indicator(df, event)

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events, alpha=alpha)

    timeline = np.asarray(kmf.survival_function_.index, dtype=float)
    survival = np.asarray(kmf.survival_function_.iloc[:, 0], dtype=float)
    ci = kmf.confidence_interval_survival_function_
    ci_lower = np.asarray(ci.iloc[:, 0], dtype=float)
    ci_upper = np.asarray(ci.iloc[:, 1], dtype=float)
    # event_table is indexed by timeline; "at_risk" is the number entering each
    # time. Reindex onto the survival-function timeline (which includes t=0).
    at_risk = kmf.event_table["at_risk"].reindex(timeline).fillna(0.0).to_numpy().astype(np.int64)

    return {
        "time": timeline.tolist(),
        "survival": survival.tolist(),
        "ci_lower": ci_lower.tolist(),
        "ci_upper": ci_upper.tolist(),
        "at_risk": [int(x) for x in at_risk],
    }


def cox_hazard_ratios(
    df: pd.DataFrame,
    *,
    duration: str,
    event: str,
    alpha: float = 0.05,
) -> dict[str, list]:
    """Fit a Cox proportional-hazards model; one row per covariate.

    Every column other than ``duration`` and ``event`` is used as a covariate.

    Args:
        df: Input relation with ``duration``, ``event`` and one or more
            numeric covariate columns.
        duration: Name of the time-to-event column.
        event: Name of the 0/1 event-indicator column.
        alpha: Significance level for the hazard-ratio confidence interval.

    Returns:
        Column block with keys ``covariate`` (str), ``coef``,
        ``hazard_ratio``, ``ci_lower``, ``ci_upper``, ``p_value`` (all float),
        one entry per covariate. ``hazard_ratio`` is ``exp(coef)``; the CI is
        the exponentiated coefficient CI.

    Raises:
        SurvivalError: On missing columns, empty input, or no covariates.
    """
    _require_columns(df, {"duration": duration, "event": event})
    if len(df) == 0:
        raise SurvivalError("cox_hazard_ratios requires a non-empty input relation")

    covariates = [c for c in df.columns if c not in (duration, event)]
    if not covariates:
        raise SurvivalError(
            "cox_hazard_ratios needs at least one covariate column besides "
            f"duration ('{duration}') and event ('{event}'); the relation has none"
        )

    fit_df = pd.DataFrame({duration: _numeric(df, duration, role="duration")})
    fit_df[event] = _event_indicator(df, event)
    for cov in covariates:
        fit_df[cov] = _numeric(df, cov, role="covariate")

    cph = CoxPHFitter(alpha=alpha)
    cph.fit(fit_df, duration_col=duration, event_col=event)

    summary = cph.summary
    coef = summary["coef"]
    return {
        "covariate": [str(c) for c in summary.index],
        "coef": [float(x) for x in coef],
        "hazard_ratio": [float(x) for x in summary["exp(coef)"]],
        "ci_lower": [float(x) for x in summary["exp(coef) lower 95%"]],
        "ci_upper": [float(x) for x in summary["exp(coef) upper 95%"]],
        "p_value": [float(x) for x in summary["p"]],
    }


def logrank_test(
    df: pd.DataFrame,
    *,
    duration: str,
    event: str,
    group: str,
) -> dict[str, list]:
    """Multivariate log-rank test comparing survival across ``group``.

    Args:
        df: Input relation with ``duration``, ``event`` and a ``group`` column.
        duration: Name of the time-to-event column.
        event: Name of the 0/1 event-indicator column.
        group: Name of the grouping column (≥2 distinct values).

    Returns:
        Single-row column block with keys ``test_statistic`` (float),
        ``p_value`` (float), ``degrees_freedom`` (int).

    Raises:
        SurvivalError: On missing columns, empty input, or fewer than two groups.
    """
    _require_columns(df, {"duration": duration, "event": event, "group": group})
    if len(df) == 0:
        raise SurvivalError("logrank_test requires a non-empty input relation")

    groups = df[group].to_numpy()
    n_groups = len(pd.unique(groups[pd.notna(groups)]))
    if n_groups < 2:
        raise SurvivalError(f"logrank_test needs at least two distinct groups in '{group}', found {n_groups}")

    durations = _numeric(df, duration, role="duration")
    events = _event_indicator(df, event)

    result = multivariate_logrank_test(durations, groups, events)
    return {
        "test_statistic": [float(result.test_statistic)],
        "p_value": [float(result.p_value)],
        "degrees_freedom": [int(result.degrees_of_freedom)],
    }


def median_survival(
    df: pd.DataFrame,
    *,
    duration: str,
    event: str,
) -> dict[str, list]:
    """Median survival time (Kaplan-Meier): time at which survival drops to 0.5.

    Args:
        df: Input relation with ``duration`` and ``event`` columns.
        duration: Name of the time-to-event column.
        event: Name of the 0/1 event-indicator column.

    Returns:
        Single-row column block with key ``median_survival`` (float). The value
        is ``inf`` when the survival curve never reaches 0.5 (median undefined).

    Raises:
        SurvivalError: On missing columns or empty input.
    """
    _require_columns(df, {"duration": duration, "event": event})
    if len(df) == 0:
        raise SurvivalError("median_survival requires a non-empty input relation")

    durations = _numeric(df, duration, role="duration")
    events = _event_indicator(df, event)

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events)
    return {"median_survival": [float(kmf.median_survival_time_)]}
