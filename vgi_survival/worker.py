"""VGI worker exposing survival / time-to-event analysis to DuckDB/SQL.

Assembles the survival table functions in ``vgi_survival`` into a single
``survival`` catalog and provides the process entry point. The repo-root
``survival_worker.py`` is a thin shim over this module for ``uv run``; installed
users get the ``vgi-survival`` console script, which calls ``main`` here.

    ATTACH 'survival' (TYPE vgi, LOCATION 'uv run survival_worker.py');
    SELECT * FROM survival.kaplan_meier((SELECT t, e FROM cohort), duration := 't', event := 'e');
"""

from __future__ import annotations

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
    "# survival\n\n"
    "Survival / time-to-event analysis over Apache Arrow, powered by "
    "[lifelines](https://lifelines.readthedocs.io/).\n\n"
    "## Overview\n\n"
    "This catalog turns a cohort of subjects — each with a follow-up *duration* and a 0/1 *event* "
    "indicator (1 = the event of interest happened, 0 = the subject was right-censored / still "
    "event-free at the end of observation) — into standard survival statistics, all from SQL.\n\n"
    "## Functions\n\n"
    "All functions consume a `(SELECT ...)` relation; the column roles are passed as named string args:\n\n"
    "- `kaplan_meier` — non-parametric survival curve S(t) with confidence interval and at-risk count.\n"
    "- `cox_hazard_ratios` — Cox proportional-hazards fit; one hazard ratio per covariate column.\n"
    "- `logrank_test` — compare survival across a grouping column (test statistic, p-value, df).\n"
    "- `median_survival` — median survival time (S(t)=0.5; `inf` if never reached).\n\n"
    "## Notes\n\n"
    "Event coding is `1`/true = event occurred, `0`/false = right-censored. Cox treats every column "
    "besides duration/event as a covariate, so select exactly the covariate columns you want."
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

_SCHEMA_SOURCE_URL = "https://github.com/Query-farm/vgi-survival/blob/main/vgi_survival/tables.py"

_SURVIVAL_CATALOG = Catalog(
    name="survival",
    default_schema="main",
    comment="Survival / time-to-event analysis (Kaplan-Meier, Cox PH, log-rank, median survival) for SQL.",
    source_url="https://github.com/Query-farm/vgi-survival",
    tags={
        "vgi.title": "Survival & Time-to-Event Analysis",
        "vgi.keywords": (
            "survival analysis, time-to-event, kaplan-meier, cox, proportional hazards, hazard ratio, "
            "log-rank, median survival, censoring, lifelines, churn, reliability, clinical trial"
        ),
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
                "vgi.keywords": (
                    "survival, time-to-event, kaplan_meier, cox_hazard_ratios, logrank_test, "
                    "median_survival, hazard ratio, censoring, lifelines, duration, event"
                ),
                # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
                "domain": "statistics",
                "category": "survival-analysis",
                "topic": "time-to-event",
                "vgi.source_url": _SCHEMA_SOURCE_URL,
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
