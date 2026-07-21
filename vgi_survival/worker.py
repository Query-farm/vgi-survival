"""VGI worker exposing survival / time-to-event analysis to DuckDB/SQL.

Assembles the survival table functions in ``vgi_survival`` into a single
``survival`` catalog and provides the process entry point. The repo-root
``survival_worker.py`` is a thin shim over this module for ``uv run``; installed
users get the ``vgi-survival`` console script, which calls ``main`` here.

    ATTACH 'survival' (TYPE vgi, LOCATION 'uv run survival_worker.py');
    SELECT * FROM survival.kaplan_meier((SELECT t, e FROM cohort), duration := 't', event := 'e');
"""

from __future__ import annotations

import json
import sys

from vgi import Worker
from vgi.catalog import Catalog, Schema, View

from vgi_survival.tables import TABLE_FUNCTIONS

_FUNCTIONS: list[type] = [*TABLE_FUNCTIONS]

_CATALOG_DOC_LLM = (
    "Survival / time-to-event analysis for SQL cohorts: estimate Kaplan-Meier survival curves, fit "
    "Cox proportional-hazards models to recover per-covariate hazard ratios, compare survival across "
    "groups with the log-rank test, and compute median survival time. Each function takes a whole "
    "relation as a `(SELECT ...)` subquery plus the names of its duration/event (and group) columns; "
    "`event=1` means the event occurred, `event=0` means right-censored. Use for clinical trials, churn, "
    "reliability, and any time-until-event question answered from a duration + censoring-indicator cohort."
)

_CATALOG_DOC_MD = (
    "# Survival & Time-to-Event Analysis in SQL\n\n"
    "![lifelines logo](https://i.imgur.com/EOowdSD.png)\n\n"
    "**Run Kaplan-Meier survival curves, Cox proportional-hazards regression, and the log-rank test "
    "directly in DuckDB SQL** — no Python notebook, no data export, no glue code. The `survival` catalog "
    "brings rigorous time-to-event statistics to your data warehouse over Apache Arrow.\n\n"
    "## What it does\n\n"
    "Survival analysis answers *time-until-something-happens* questions: how long until a patient relapses, "
    "a customer churns, a machine fails, or a subscription lapses. This extension takes a cohort of subjects "
    "— each with a follow-up **duration** and a 0/1 **event** indicator (`1` = the event of interest "
    "occurred, `0` = the subject was right-censored, i.e. still event-free when observation ended) — and "
    "computes the standard estimators of survival analysis, all from a SQL query. It is built for analysts, "
    "data scientists, and engineers working on clinical trials, customer churn and retention, hardware "
    "reliability, and any cohort whose outcome is *time until an event*.\n\n"
    "## How it works\n\n"
    "Every estimator is powered by [lifelines](https://github.com/CamDavidsonPilon/lifelines), the widely "
    "used, MIT-licensed Python survival-analysis library (see the "
    "[lifelines documentation](https://lifelines.readthedocs.io/)). The worker buffers the entire input "
    "relation, hands it to the appropriate lifelines estimator once, and streams the result back as Arrow "
    "rows — so the numbers match lifelines exactly. Every function is a **table function**: it consumes a "
    "whole `(SELECT ...)` relation as its first argument, and you name which columns play which role using "
    "named string arguments such as `duration := 't'` and `event := 'e'`.\n\n"
    "## When to reach for it\n\n"
    "Reach for the `survival` catalog whenever your question is *time until an event* and your data has a "
    "follow-up duration plus a 0/1 censoring indicator: charting how an event-free population decays over "
    "time, quantifying how covariates raise or lower risk, testing whether two arms differ, or reducing a "
    "cohort to a single headline survival number. It fits [Kaplan-Meier]"
    "(https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator) curves, [Cox proportional-hazards]"
    "(https://en.wikipedia.org/wiki/Proportional_hazards_model) regression, and the [log-rank test]"
    "(https://en.wikipedia.org/wiki/Logrank_test).\n\n"
    "## Notes\n\n"
    "Event coding is `1`/true = event occurred, `0`/false = right-censored — get this backwards and every "
    "curve and hazard ratio inverts. Cox treats every column besides duration/event as a covariate, so "
    "select exactly the covariate columns you want. Because `group` is a SQL keyword, double-quote it at "
    "the `logrank_test` call site: `\"group\" := 'arm'`."
)

_SCHEMA_DOC_LLM = (
    "Survival / time-to-event table functions over Apache Arrow, powered by lifelines. Each takes a "
    "cohort relation — passed as a `(SELECT ...)` subquery — plus the names of its duration and event "
    "(and, for group comparisons, group) columns, where `event=1` means the event occurred and "
    "`event=0` means right-censored. Reach for this schema to estimate and chart survival curves, model "
    "how covariates raise or lower the hazard, test whether survival differs across groups, or summarize "
    "a cohort's survival as a single number."
)

_SCHEMA_DOC_MD = (
    "## main\n\n"
    "The single schema of the `survival` catalog. It holds the survival / time-to-event table functions "
    "over Apache Arrow, powered by lifelines — grouped into estimation, regression, and comparison "
    "categories.\n\n"
    "### Usage\n\n"
    "Pass your cohort as a parenthesised subquery and name the role columns — the follow-up "
    "`duration` and the 0/1 `event` indicator, plus a grouping column for the comparison test. "
    "Each estimator ships a ready-to-run example query; browse the schema's objects to copy one.\n\n"
    "### Notes\n\n"
    "Event coding is `1` = event occurred, `0` = right-censored. The grouping argument of the "
    "group-comparison test collides with the SQL `GROUP` keyword, so quote it at the call site: "
    "`\"group\" := 'arm'`."
)

_SCHEMA_EXAMPLE_QUERIES = json.dumps(
    [
        {
            "description": "Kaplan-Meier survival curve for a small cohort, projected and ordered by time",
            "sql": (
                "SELECT time, survival, at_risk FROM survival.main.kaplan_meier("
                "(SELECT * FROM (VALUES (5,1),(8,0),(12,1),(3,1),(9,0)) AS c(t, e)), "
                "duration := 't', event := 'e') ORDER BY time"
            ),
        },
        {
            "description": "Cox hazard ratio and Wald p-value for a single covariate",
            "sql": (
                "SELECT covariate, hazard_ratio, p_value FROM survival.main.cox_hazard_ratios("
                "(SELECT * FROM (VALUES (5,1,1.0),(8,0,0.0),(12,1,2.0),(3,1,3.0),(9,0,0.0),(6,1,1.0)) "
                "AS c(t, e, prio)), duration := 't', event := 'e') ORDER BY covariate"
            ),
        },
        {
            "description": "Log-rank test comparing survival between two treatment arms",
            "sql": (
                "SELECT test_statistic, p_value, degrees_freedom FROM survival.main.logrank_test("
                "(SELECT * FROM (VALUES (5,1,'a'),(8,0,'a'),(12,1,'b'),(3,1,'b'),(9,0,'a'),(6,1,'b')) "
                "AS c(t, e, arm)), duration := 't', event := 'e', \"group\" := 'arm')"
            ),
        },
        {
            "description": "Median survival time as a single headline number",
            "sql": (
                "SELECT median_survival FROM survival.main.median_survival("
                "(SELECT * FROM (VALUES (5,1),(8,0),(12,1),(3,1),(9,0)) AS c(t, e)), "
                "duration := 't', event := 'e')"
            ),
        },
    ]
)

_CATALOG_KEYWORDS = json.dumps(
    [
        "survival analysis",
        "time-to-event",
        "kaplan-meier",
        "cox",
        "proportional hazards",
        "hazard ratio",
        "log-rank",
        "median survival",
        "censoring",
        "lifelines",
        "churn",
        "reliability",
        "clinical trial",
    ]
)

_SCHEMA_KEYWORDS = json.dumps(
    [
        "survival",
        "time-to-event",
        "kaplan_meier",
        "cox_hazard_ratios",
        "logrank_test",
        "median_survival",
        "hazard ratio",
        "censoring",
        "lifelines",
        "duration",
        "event",
    ]
)

_SCHEMA_CATEGORIES = json.dumps(
    [
        {
            "name": "estimation",
            "description": (
                "Non-parametric survival-curve estimation and single-number summaries "
                "(Kaplan-Meier S(t) and median survival time)."
            ),
        },
        {
            "name": "regression",
            "description": (
                "Covariate models that quantify how each factor raises or lowers the hazard (Cox proportional hazards)."
            ),
        },
        {
            "name": "comparison",
            "description": "Hypothesis tests that compare survival between groups (the log-rank test).",
        },
        {
            "name": "reference",
            "description": (
                "Browsable reference relations describing the catalog itself — the survival methods it "
                "exposes, which question each answers, and what each returns."
            ),
        },
    ]
)

# A VALUES-backed, credential-free browsable relation so an agent can list the
# available survival methods (and pick the right one) before it has to guess a
# table-function's arguments. This clears VGI146 (a worker with table functions
# but no browsable table/view) and scans instantly with no network or secret.
_METHODS_VIEW = View(
    name="methods",
    definition=(
        "SELECT * FROM (VALUES "
        "('kaplan_meier', 'estimation', "
        "'How does an event-free population decline over time?', "
        "'survival.main.kaplan_meier(relation, duration, event)', "
        "'One row per distinct follow-up time: survival S(t), a confidence band, and the at-risk count.'), "
        "('median_survival', 'estimation', "
        "'What single headline time summarizes a cohort''s survival?', "
        "'survival.main.median_survival(relation, duration, event)', "
        "'One row: the median survival time where the Kaplan-Meier curve first reaches S(t)=0.5.'), "
        "('cox_hazard_ratios', 'regression', "
        "'How does each covariate raise or lower the hazard?', "
        "'survival.main.cox_hazard_ratios(relation, duration, event)', "
        "'One row per covariate: coefficient, hazard ratio, a confidence band, and a Wald p-value.'), "
        "('logrank_test', 'comparison', "
        "'Do two or more groups have different survival?', "
        "'survival.main.logrank_test(relation, duration, event, group)', "
        "'One row: the chi-squared statistic, the p-value, and the degrees of freedom.')"
        ") AS m(method, category, question, signature, returns)"
    ),
    comment="Registry of the survival table functions this catalog exposes (method, category, question, returns).",
    column_comments={
        "method": "Table-function name in the survival.main schema.",
        "category": "Navigation category: estimation, regression, or comparison.",
        "question": "The time-to-event question this method answers.",
        "signature": "Fully-qualified call signature showing the relation and role arguments.",
        "returns": "One-line description of the rows the method returns.",
    },
    tags={
        "vgi.title": "Survival Methods Registry",
        "vgi.category": "reference",
        # VGI123 classifying tags use BARE keys (reused schema vocabulary).
        "domain": "statistics",
        "topic": "time-to-event",
        "vgi.keywords": json.dumps(
            [
                "methods",
                "registry",
                "catalog",
                "survival",
                "table functions",
                "discovery",
                "reference",
            ]
        ),
        "vgi.doc_llm": (
            "A browsable registry of the survival / time-to-event methods this catalog exposes. One row "
            "per table function, giving its `method` name, its navigation `category` (estimation, "
            "regression, or comparison), the `question` it answers, its call `signature`, and a "
            "one-line description of what it `returns`. Scan or filter this relation to discover which "
            "function fits a task before calling it — e.g. filter by category to list every estimator, "
            "or read a method's signature to learn which role arguments it needs. It reads instantly "
            "with no arguments, network access, or credentials."
        ),
        "vgi.doc_md": (
            "# methods\n\n"
            "A browsable registry of every survival table function in this catalog — a discovery "
            "aid so an agent can pick the right method before guessing a function's arguments.\n\n"
            "## Columns\n\n"
            "`method` (the function name), `category` (estimation / regression / comparison), "
            "`question` (the time-to-event question it answers), `signature` (its fully-qualified "
            "call shape), and `returns` (a one-line result summary).\n\n"
            "## Usage\n\n"
            "Filter by category to list the estimators, or read a row's signature to learn a "
            "method's role arguments. The relation is VALUES-backed, so it returns instantly with "
            "no network access or credentials.\n\n"
            "## Notes\n\n"
            "This registry only names the methods; call the listed function itself to run the "
            "analysis. Event coding across every method is `1` = event occurred, `0` = right-censored."
        ),
        "vgi.example_queries": json.dumps(
            [
                {
                    "description": "List every regression method with the question it answers",
                    "sql": (
                        "SELECT method, question, returns FROM survival.main.methods "
                        "WHERE category = 'regression' ORDER BY method"
                    ),
                },
                {
                    "description": "Count the available survival methods per category",
                    "sql": (
                        "SELECT category, count(*) AS n_methods FROM survival.main.methods "
                        "GROUP BY category ORDER BY category"
                    ),
                },
            ]
        ),
    },
)

# Deterministic analyst tasks for `vgi-lint simulate` (VGI152/VGI920). Each names
# exactly the columns it asks for; grading compares by VALUE (ignore_column_names)
# so an aliased/reordered projection still matches the canonical reference_sql, and
# both queries run the same lifelines estimator on the same inline cohort — so the
# floats are byte-identical. Together they cover all four functions.
_AGENT_TEST_TASKS = json.dumps(
    [
        {
            "name": "kaplan_meier_curve",
            "prompt": (
                "A cohort has a follow-up time column `t` and an event indicator `e` "
                "(1 = the event occurred, 0 = right-censored), with rows "
                "(5,1), (8,0), (12,1), (3,1), (9,0). Return the Kaplan-Meier survival "
                "curve as two columns — the observed time and the survival probability "
                "S(t) — ordered by time ascending."
            ),
            "reference_sql": (
                "SELECT time, survival FROM survival.main.kaplan_meier("
                "(SELECT * FROM (VALUES (5,1),(8,0),(12,1),(3,1),(9,0)) AS c(t, e)), "
                "duration := 't', event := 'e') ORDER BY time"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "cox_hazard_ratio_for_covariate",
            "prompt": (
                "Fit a Cox proportional-hazards model to a cohort with follow-up time `t`, "
                "event indicator `e` (1 = event, 0 = censored), and a single covariate "
                "`prio`, with rows (5,1,1.0), (8,0,0.0), (12,1,2.0), (3,1,3.0), (9,0,0.0), "
                "(6,1,1.0). Return the covariate name and its hazard ratio."
            ),
            "reference_sql": (
                "SELECT covariate, hazard_ratio FROM survival.main.cox_hazard_ratios("
                "(SELECT * FROM (VALUES (5,1,1.0),(8,0,0.0),(12,1,2.0),(3,1,3.0),(9,0,0.0),"
                "(6,1,1.0)) AS c(t, e, prio)), duration := 't', event := 'e')"
            ),
            "ignore_column_names": True,
            "unordered": True,
        },
        {
            "name": "logrank_compare_arms",
            "prompt": (
                "Two treatment arms are labelled in column `arm`; each subject has a "
                "follow-up time `t` and an event indicator `e` (1 = event, 0 = censored), "
                "with rows (5,1,'a'), (8,0,'a'), (12,1,'b'), (3,1,'b'), (9,0,'a'), "
                "(6,1,'b'). Use the log-rank test to compare survival between the arms and "
                "return the test statistic, the p-value, and the degrees of freedom."
            ),
            "reference_sql": (
                "SELECT test_statistic, p_value, degrees_freedom FROM survival.main.logrank_test("
                "(SELECT * FROM (VALUES (5,1,'a'),(8,0,'a'),(12,1,'b'),(3,1,'b'),(9,0,'a'),"
                "(6,1,'b')) AS c(t, e, arm)), duration := 't', event := 'e', \"group\" := 'arm')"
            ),
            "ignore_column_names": True,
            "unordered": True,
        },
        {
            "name": "median_survival_time",
            "prompt": (
                "A cohort has a follow-up time column `t` and an event indicator `e` "
                "(1 = the event occurred, 0 = right-censored), with rows "
                "(5,1), (8,0), (12,1), (3,1), (9,0). Return the median survival time."
            ),
            "reference_sql": (
                "SELECT median_survival FROM survival.main.median_survival("
                "(SELECT * FROM (VALUES (5,1),(8,0),(12,1),(3,1),(9,0)) AS c(t, e)), "
                "duration := 't', event := 'e')"
            ),
            "ignore_column_names": True,
            "unordered": True,
        },
        {
            "name": "discover_regression_methods",
            "prompt": (
                "This catalog exposes several survival / time-to-event methods and also provides a "
                "browsable relation that lists them. Using that registry of methods, return the name "
                "of every method whose category is regression."
            ),
            "reference_sql": ("SELECT method FROM survival.main.methods WHERE category = 'regression' ORDER BY method"),
            "ignore_column_names": True,
            "unordered": True,
        },
    ]
)

_SURVIVAL_CATALOG = Catalog(
    name="survival",
    default_schema="main",
    comment="Survival / time-to-event analysis (Kaplan-Meier, Cox PH, log-rank, median survival) for SQL.",
    source_url="https://github.com/Query-farm/vgi-survival",
    tags={
        "vgi.title": "Survival & Time-to-Event Analysis",
        "vgi.keywords": _CATALOG_KEYWORDS,
        "vgi.doc_llm": _CATALOG_DOC_LLM,
        "vgi.doc_md": _CATALOG_DOC_MD,
        "vgi.author": "Query.Farm",
        "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
        "vgi.license": "MIT",
        "vgi.support_contact": "https://github.com/Query-farm/vgi-survival/issues",
        "vgi.support_policy_url": "https://github.com/Query-farm/vgi-survival/blob/main/README.md",
        "vgi.agent_test_tasks": _AGENT_TEST_TASKS,
    },
    schemas=[
        Schema(
            name="main",
            comment="Survival / time-to-event analysis (Kaplan-Meier, Cox PH, log-rank) for SQL",
            tags={
                "vgi.title": "Survival Analysis Functions",
                "vgi.keywords": _SCHEMA_KEYWORDS,
                # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
                "domain": "statistics",
                "category": "survival-analysis",
                "topic": "time-to-event",
                "vgi.doc_llm": _SCHEMA_DOC_LLM,
                "vgi.doc_md": _SCHEMA_DOC_MD,
                "vgi.example_queries": _SCHEMA_EXAMPLE_QUERIES,
                "vgi.categories": _SCHEMA_CATEGORIES,
            },
            views=[_METHODS_VIEW],
            functions=list(_FUNCTIONS),
        ),
    ],
)


class SurvivalWorker(Worker):
    """Worker process hosting the ``survival`` catalog."""

    catalog = _SURVIVAL_CATALOG


def main() -> None:
    """Run the worker (stdio by default; pass ``--http`` for the HTTP server)."""
    SurvivalWorker.main()


def main_http() -> None:
    """Run the worker over HTTP (injects ``--http`` into the worker CLI)."""
    argv = sys.argv[1:]
    if "--http" not in argv:
        argv = ["--http", *argv]
    sys.argv = [sys.argv[0], *argv]
    SurvivalWorker.main()
