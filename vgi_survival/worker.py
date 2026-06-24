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

_CATALOG_DESCRIPTION_LLM = (
    "Survival / time-to-event analysis for SQL cohorts: estimate Kaplan-Meier survival curves, fit "
    "Cox proportional-hazards models to recover per-covariate hazard ratios, compare survival across "
    "groups with the log-rank test, and compute median survival time. Each function takes a whole "
    "relation as a (SELECT ...) subquery plus the names of its duration/event (and group) columns; "
    "event=1 means the event occurred, event=0 means right-censored. Use for clinical trials, churn, "
    "reliability, and any time-until-event question answered from a duration + censoring-indicator cohort."
)

_CATALOG_DESCRIPTION_MD = (
    "# survival\n\n"
    "Survival / time-to-event analysis over Apache Arrow, powered by "
    "[lifelines](https://lifelines.readthedocs.io/).\n\n"
    "Table functions (all consume a `(SELECT ...)` relation; column roles are named string args):\n\n"
    "- `kaplan_meier` — non-parametric survival curve S(t) with confidence interval and at-risk count.\n"
    "- `cox_hazard_ratios` — Cox proportional-hazards fit; one hazard ratio per covariate column.\n"
    "- `logrank_test` — compare survival across a grouping column (test statistic, p-value, df).\n"
    "- `median_survival` — median survival time (S(t)=0.5; `inf` if never reached).\n\n"
    "Event coding: `1`/true = event occurred, `0`/false = right-censored."
)

_SCHEMA_DESCRIPTION_LLM = (
    "Survival / time-to-event table functions: Kaplan-Meier survival curves, Cox proportional-hazards "
    "hazard ratios, the log-rank test across groups, and median survival time. Each takes a cohort "
    "relation plus the duration/event (and group) column names; event=1 = event occurred, 0 = censored."
)

_SCHEMA_DESCRIPTION_MD = (
    "Survival / time-to-event analysis functions (Kaplan-Meier, Cox PH, log-rank, median survival) "
    "over Apache Arrow, powered by lifelines."
)

_SURVIVAL_CATALOG = Catalog(
    name="survival",
    default_schema="main",
    comment="Survival / time-to-event analysis (Kaplan-Meier, Cox PH, log-rank, median survival) for SQL.",
    source_url="https://github.com/Query-farm/vgi-survival",
    tags={
        "vgi.description_llm": _CATALOG_DESCRIPTION_LLM,
        "vgi.description_md": _CATALOG_DESCRIPTION_MD,
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
                "vgi.description_llm": _SCHEMA_DESCRIPTION_LLM,
                "vgi.description_md": _SCHEMA_DESCRIPTION_MD,
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
