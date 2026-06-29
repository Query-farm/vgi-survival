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
from vgi.catalog import Catalog, Schema

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
    "Every function is powered by [lifelines](https://github.com/CamDavidsonPilon/lifelines), the widely "
    "used, MIT-licensed Python survival-analysis library (see the "
    "[lifelines documentation](https://lifelines.readthedocs.io/)). The worker buffers the entire input "
    "relation, hands it to the appropriate lifelines estimator once, and streams the result back as Arrow "
    "rows — so the numbers match lifelines' own `KaplanMeierFitter`, `CoxPHFitter`, and "
    "`logrank_test` exactly. Each function is a **table function**: it consumes a whole `(SELECT ...)` "
    "relation as its first argument, and you name which columns play which role using named string "
    "arguments such as `duration := 't'` and `event := 'e'`.\n\n"
    "## SQL use cases & functions\n\n"
    "- `kaplan_meier` — non-parametric [Kaplan-Meier](https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator) "
    "survival curve S(t) with confidence interval and at-risk count; chart how the event-free fraction "
    "decays over time.\n"
    "- `cox_hazard_ratios` — [Cox proportional-hazards](https://en.wikipedia.org/wiki/Proportional_hazards_model) "
    "fit returning one hazard ratio per covariate column; rank which covariates raise or lower risk.\n"
    "- `logrank_test` — the [log-rank test](https://en.wikipedia.org/wiki/Logrank_test) comparing survival "
    "across a grouping column (test statistic, p-value, degrees of freedom); ask whether two arms differ.\n"
    "- `median_survival` — the median survival time (where S(t) = 0.5; returns `inf` when the curve never "
    "reaches 0.5 under heavy censoring).\n\n"
    "Example: "
    "`SELECT * FROM survival.kaplan_meier((SELECT t, e FROM cohort), duration := 't', event := 'e');`\n\n"
    "## Notes\n\n"
    "Event coding is `1`/true = event occurred, `0`/false = right-censored — get this backwards and every "
    "curve and hazard ratio inverts. Cox treats every column besides duration/event as a covariate, so "
    "select exactly the covariate columns you want. Because `group` is a SQL keyword, double-quote it at "
    "the `logrank_test` call site: `\"group\" := 'arm'`."
)

_SCHEMA_DOC_LLM = (
    "Survival / time-to-event table functions: Kaplan-Meier survival curves, Cox proportional-hazards "
    "hazard ratios, the log-rank test across groups, and median survival time. Each takes a cohort "
    "relation plus the duration/event (and group) column names; `event=1` = event occurred, 0 = censored. "
    "Reach for `kaplan_meier` to chart S(t), `cox_hazard_ratios` to rank covariate risk, `logrank_test` "
    "to ask whether two arms differ, and `median_survival` for a single headline number."
)

_SCHEMA_DOC_MD = (
    "## main\n\n"
    "The single schema of the `survival` catalog. It holds the four survival table functions "
    "(`kaplan_meier`, `cox_hazard_ratios`, `logrank_test`, `median_survival`) over Apache Arrow, "
    "powered by lifelines.\n\n"
    "### Usage\n\n"
    "Pass your cohort as a subquery and name the role columns, e.g. "
    "`SELECT * FROM survival.main.kaplan_meier((SELECT t, e FROM cohort), duration := 't', event := 'e')`.\n\n"
    "### Notes\n\n"
    "Event coding is `1` = event occurred, `0` = right-censored. The grouping arg of `logrank_test` "
    "collides with the SQL `GROUP` keyword, so quote it at the call site: `\"group\" := 'arm'`."
)

_SCHEMA_EXAMPLE_QUERIES = (
    "SELECT * FROM survival.main.kaplan_meier("
    "(SELECT * FROM (VALUES (5,1),(8,0),(12,1),(3,1),(9,0)) AS c(t, e)), "
    "duration := 't', event := 'e') ORDER BY time;\n"
    "SELECT * FROM survival.main.cox_hazard_ratios("
    "(SELECT * FROM (VALUES (5,1,1.0),(8,0,0.0),(12,1,2.0),(3,1,3.0),(9,0,0.0),(6,1,1.0)) "
    "AS c(t, e, prio)), duration := 't', event := 'e');\n"
    "SELECT * FROM survival.main.logrank_test("
    "(SELECT * FROM (VALUES (5,1,'a'),(8,0,'a'),(12,1,'b'),(3,1,'b'),(9,0,'a'),(6,1,'b')) "
    "AS c(t, e, arm)), duration := 't', event := 'e', \"group\" := 'arm');\n"
    "SELECT * FROM survival.main.median_survival("
    "(SELECT * FROM (VALUES (5,1),(8,0),(12,1),(3,1),(9,0)) AS c(t, e)), "
    "duration := 't', event := 'e');"
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
            },
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
