"""Composed example queries that combine several survival functions in one query.

Each example is defined once here and referenced from both the schema-level
``vgi.example_queries`` list (``worker.py``) and the example list of the
function that drives the query (``tables.py``), so the SQL never drifts between
the two places.

All of them run over the same inline two-arm trial cohort (``_TRIAL_COHORT``):
16 subjects, follow-up time ``t``, event indicator ``e`` (1 = event, 0 =
censored), and ``arm`` ('control' / 'treated'). The arms overlap in time, so the
Cox fit converges (a perfectly separated cohort drives the hazard ratio to 0).
"""

from __future__ import annotations

from vgi.metadata import FunctionExample

_TRIAL_COHORT = (
    "WITH cohort AS (SELECT * FROM (VALUES "
    "(2,1,'control'),(3,1,'control'),(4,1,'control'),(5,0,'control'),"
    "(6,1,'control'),(8,1,'control'),(9,0,'control'),(11,1,'control'),"
    "(4,1,'treated'),(7,0,'treated'),(9,1,'treated'),(12,1,'treated'),"
    "(14,0,'treated'),(15,1,'treated'),(18,1,'treated'),(20,0,'treated')"
    ") AS c(t, e, arm))"
)

# log-rank test + Cox hazard ratio: the two numbers a trial report leads with.
# Result: logrank_p_value 0.0199, hazard_ratio 0.175 (CI 0.034-0.905), cox_p_value 0.0377.
TRIAL_SUMMARY_EXAMPLE: dict[str, str] = {
    "description": (
        "Treatment-effect headline for a two-arm trial: the log-rank p-value comparing the arms "
        "next to the Cox hazard ratio (with 95% CI and Wald p-value) of treated vs control, "
        "one row from the same cohort"
    ),
    "sql": (
        f"{_TRIAL_COHORT} "
        "SELECT round(lr.p_value, 4) AS logrank_p_value, round(cx.hazard_ratio, 3) AS hazard_ratio, "
        "round(cx.ci_lower, 3) AS hr_ci_lower, round(cx.ci_upper, 3) AS hr_ci_upper, "
        "round(cx.p_value, 4) AS cox_p_value "
        "FROM survival.main.logrank_test((SELECT t, e, arm FROM cohort), "
        "duration := 't', event := 'e', \"group\" := 'arm') AS lr "
        "CROSS JOIN survival.main.cox_hazard_ratios("
        "(SELECT t, e, (arm = 'treated')::INTEGER AS treated FROM cohort), "
        "duration := 't', event := 'e') AS cx "
        "ORDER BY cx.covariate"
    ),
}

# median_survival per arm + log-rank p-value. Result: 6.0 | 15.0 | 0.0199.
MEDIAN_BY_ARM_EXAMPLE: dict[str, str] = {
    "description": (
        "Median survival time of each trial arm side by side with the log-rank p-value testing "
        "whether the arms differ (control 6.0, treated 15.0, p = 0.0199)"
    ),
    "sql": (
        f"{_TRIAL_COHORT} "
        "SELECT mc.median_survival AS median_control, mt.median_survival AS median_treated, "
        "round(lr.p_value, 4) AS logrank_p_value "
        "FROM survival.main.median_survival((SELECT t, e FROM cohort WHERE arm = 'control'), "
        "duration := 't', event := 'e') AS mc "
        "CROSS JOIN survival.main.median_survival((SELECT t, e FROM cohort WHERE arm = 'treated'), "
        "duration := 't', event := 'e') AS mt "
        "CROSS JOIN survival.main.logrank_test((SELECT t, e, arm FROM cohort), "
        "duration := 't', event := 'e', \"group\" := 'arm') AS lr "
        "ORDER BY median_control"
    ),
}
MEDIAN_BY_ARM_EXPECTED: list[dict[str, float]] = [
    {"median_control": 6.0, "median_treated": 15.0, "logrank_p_value": 0.0199}
]

# One Kaplan-Meier curve per arm, read at landmark times with an ASOF join (the
# curve is a step function, so S(landmark) is the last step at or before it).
# Result: t=6 -> 0.469 | 0.875; t=12 -> 0.0 | 0.583.
LANDMARK_SURVIVAL_EXAMPLE: dict[str, str] = {
    "description": (
        "Landmark survival table: each arm's Kaplan-Meier survival probability at t = 6 and "
        "t = 12, read off the step curves with an ASOF join"
    ),
    "sql": (
        f"{_TRIAL_COHORT}, "
        "landmarks AS (SELECT * FROM (VALUES (6.0), (12.0)) AS l(landmark)) "
        "SELECT l.landmark, round(kc.survival, 3) AS control_survival, "
        "round(kt.survival, 3) AS treated_survival "
        "FROM landmarks AS l "
        "ASOF JOIN survival.main.kaplan_meier((SELECT t, e FROM cohort WHERE arm = 'control'), "
        "duration := 't', event := 'e') AS kc ON kc.time <= l.landmark "
        "ASOF JOIN survival.main.kaplan_meier((SELECT t, e FROM cohort WHERE arm = 'treated'), "
        "duration := 't', event := 'e') AS kt ON kt.time <= l.landmark "
        "ORDER BY l.landmark"
    ),
}
LANDMARK_SURVIVAL_EXPECTED: list[dict[str, float]] = [
    {"landmark": 6.0, "control_survival": 0.469, "treated_survival": 0.875},
    {"landmark": 12.0, "control_survival": 0.0, "treated_survival": 0.583},
]

COMPOSED_EXAMPLES: list[dict[str, str]] = [
    TRIAL_SUMMARY_EXAMPLE,
    MEDIAN_BY_ARM_EXAMPLE,
    LANDMARK_SURVIVAL_EXAMPLE,
]


def function_examples(examples: list[dict[str, str]]) -> list[FunctionExample]:
    """Build ``Meta.examples`` entries from ``{description, sql}`` dicts.

    Args:
        examples: Example dicts, the same ones serialized into ``vgi.example_queries``.

    Returns:
        One ``FunctionExample`` per dict, in order.
    """
    return [FunctionExample(sql=ex["sql"], description=ex["description"]) for ex in examples]
