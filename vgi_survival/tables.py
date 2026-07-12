"""Survival / time-to-event table functions for DuckDB via VGI.

Each function consumes a *whole* input relation -- passed as a ``(SELECT ...)``
subquery (positional ``Arg(0)``) -- and the column roles as NAMED string args
(``duration := 'time'``, ``event := 'died'``, ``group := 'arm'``). Because a
survival estimate needs every row, these are buffering (Sink+Source) functions:
they sink all input batches, then run the lifelines estimator once in finalize.

    SELECT * FROM survival.kaplan_meier((SELECT t, e FROM cohort), duration := 't', event := 'e');
    SELECT * FROM survival.cox_hazard_ratios((SELECT * FROM cohort), duration := 't', event := 'e');
    SELECT * FROM survival.logrank_test((SELECT * FROM c), duration := 't', event := 'e', group := 'arm');
    SELECT * FROM survival.median_survival((SELECT t, e FROM cohort), duration := 't', event := 'e');

Event coding: ``event`` = 1/true means the event occurred (death/failure);
0/false means right-censored. Cox uses every column besides duration/event as a
covariate. See ``vgi_survival.survival`` for the math and full conventions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, ClassVar

import pyarrow as pa
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from . import survival
from .buffering import DrainState, SinkBuffer
from .schema_utils import field as sfield


def _json_keywords(*keywords: str) -> str:
    """Serialize discovery keywords as a JSON array string for ``vgi.keywords``.

    Args:
        *keywords: The individual keyword/phrase strings.

    Returns:
        A JSON array (e.g. ``["a","b"]``) as required by the VGI metadata schema.
    """
    return json.dumps(list(keywords))


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

_KM_SCHEMA = pa.schema(
    [
        sfield("time", pa.float64(), "Distinct observed follow-up time.", nullable=False),
        sfield("survival", pa.float64(), "Kaplan-Meier survival probability S(t) at this time."),
        sfield("ci_lower", pa.float64(), "Lower bound of the survival confidence interval."),
        sfield("ci_upper", pa.float64(), "Upper bound of the survival confidence interval."),
        sfield("at_risk", pa.int64(), "Number of subjects still at risk entering this time."),
    ]
)

_COX_SCHEMA = pa.schema(
    [
        sfield("covariate", pa.string(), "Covariate (input column) name.", nullable=False),
        sfield("coef", pa.float64(), "Fitted log-hazard coefficient (beta)."),
        sfield("hazard_ratio", pa.float64(), "Hazard ratio exp(beta); >1 raises hazard, <1 lowers it."),
        sfield("ci_lower", pa.float64(), "Lower bound of the hazard-ratio confidence interval."),
        sfield("ci_upper", pa.float64(), "Upper bound of the hazard-ratio confidence interval."),
        sfield("p_value", pa.float64(), "Wald-test p-value for the coefficient."),
    ]
)

_LOGRANK_SCHEMA = pa.schema(
    [
        sfield("test_statistic", pa.float64(), "Log-rank chi-squared test statistic."),
        sfield("p_value", pa.float64(), "p-value of the log-rank test."),
        sfield("degrees_freedom", pa.int32(), "Degrees of freedom (number of groups minus one)."),
    ]
)

_MEDIAN_SCHEMA = pa.schema(
    [
        sfield(
            "median_survival",
            pa.float64(),
            "Median survival time (where S(t)=0.5); inf if never reached.",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Argument dataclasses -- (SELECT ...) relation as Arg(0), roles as named args
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class KaplanMeierArgs:
    """Arguments for ``kaplan_meier``: the relation plus duration/event roles."""

    data: Annotated[TableInput, Arg(0, doc="Relation containing the duration and event columns.")]
    duration: Annotated[str, Arg("duration", default="duration", doc="Time-to-event / follow-up column.")]
    event: Annotated[str, Arg("event", default="event", doc="0/1 event indicator (1 = event occurred).")]


@dataclass(slots=True, frozen=True)
class CoxArgs:
    """Arguments for ``cox_hazard_ratios``: the relation plus duration/event roles."""

    data: Annotated[
        TableInput,
        Arg(
            0,
            doc=(
                "The cohort relation to fit; any column that is neither the duration nor the event "
                "is fitted as a numeric covariate."
            ),
        ),
    ]
    duration: Annotated[str, Arg("duration", default="duration", doc="Time-to-event / follow-up column.")]
    event: Annotated[str, Arg("event", default="event", doc="0/1 event indicator (1 = event occurred).")]


@dataclass(slots=True, frozen=True)
class LogRankArgs:
    """Arguments for ``logrank_test``: the relation plus duration/event/group roles."""

    data: Annotated[TableInput, Arg(0, doc="Relation containing duration, event, and group columns.")]
    duration: Annotated[str, Arg("duration", default="duration", doc="Time-to-event / follow-up column.")]
    event: Annotated[str, Arg("event", default="event", doc="0/1 event indicator (1 = event occurred).")]
    group: Annotated[str, Arg("group", default="group", doc="Grouping column to compare survival across.")]


@dataclass(slots=True, frozen=True)
class MedianArgs:
    """Arguments for ``median_survival``: the relation plus duration/event roles."""

    data: Annotated[TableInput, Arg(0, doc="Relation containing the duration and event columns.")]
    duration: Annotated[str, Arg("duration", default="duration", doc="Time-to-event / follow-up column.")]
    event: Annotated[str, Arg("event", default="event", doc="0/1 event indicator (1 = event occurred).")]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


class KaplanMeier(SinkBuffer[KaplanMeierArgs, DrainState]):
    """Kaplan-Meier survival curve over a buffered cohort relation."""

    FunctionArguments: ClassVar[type] = KaplanMeierArgs

    class Meta:
        """Catalog metadata for the ``kaplan_meier`` function."""

        name = "kaplan_meier"
        description = (
            "Kaplan-Meier survival curve: (time, survival, ci_lower, ci_upper, at_risk). "
            "event=1/true means the event occurred; 0/false means censored."
        )
        categories = ["survival", "estimator"]
        tags = {
            "vgi.category": "estimation",
            "vgi.title": "Kaplan-Meier Survival Curve",
            "vgi.keywords": _json_keywords(
                "kaplan-meier",
                "kaplan meier",
                "survival curve",
                "survival function",
                "S(t)",
                "confidence interval",
                "at risk",
                "censoring",
                "non-parametric",
                "time-to-event",
            ),
            "vgi.doc_llm": (
                "Estimate the non-parametric Kaplan-Meier survival function S(t) from a cohort "
                "relation. Pass the cohort as `(SELECT ...)` and name the `duration` (follow-up time) "
                "and `event` (0/1 indicator; 1 = event occurred, 0 = right-censored) columns. Returns "
                "one row per distinct observed time, ordered by time, with the survival probability, a "
                "confidence interval, and the number of subjects still at risk. Use it to chart how a "
                "population's survival declines over time — clinical follow-up, customer churn, or "
                "hardware reliability. The curve starts at 1.0 and is monotonically non-increasing."
            ),
            "vgi.doc_md": (
                "# kaplan_meier\n\n"
                "Non-parametric **Kaplan-Meier** estimate of the survival function S(t) over a "
                "buffered cohort relation.\n\n"
                "## Usage\n\n"
                "Pass your cohort as the first argument — a parenthesised subquery projecting the "
                "follow-up time and 0/1 event columns — and name those roles with `duration := 't'` "
                "and `event := 'e'`. Order the result by `time` to read the curve top to bottom. A "
                "ready-to-run example is attached to this function.\n\n"
                "## Returns\n\n"
                "One row per distinct observed follow-up time (ordered by `time`): the survival "
                "probability `survival`, its `ci_lower`/`ci_upper` 95% confidence band, and the "
                "`at_risk` count.\n\n"
                "## Notes\n\n"
                "Event coding is `1` = event occurred, `0` = right-censored. The curve begins at 1.0 "
                "and never increases. The result can be long (one row per distinct time)."
            ),
            "vgi.result_columns_schema": json.dumps(
                [
                    {"name": "time", "type": "DOUBLE", "description": "Distinct observed follow-up time."},
                    {
                        "name": "survival",
                        "type": "DOUBLE",
                        "description": "Kaplan-Meier survival probability S(t) at this time.",
                    },
                    {
                        "name": "ci_lower",
                        "type": "DOUBLE",
                        "description": "Lower bound of the survival confidence interval.",
                    },
                    {
                        "name": "ci_upper",
                        "type": "DOUBLE",
                        "description": "Upper bound of the survival confidence interval.",
                    },
                    {
                        "name": "at_risk",
                        "type": "BIGINT",
                        "description": "Number of subjects still at risk entering this time.",
                    },
                ]
            ),
            "vgi.executable_examples": (
                '[{"description": "Kaplan-Meier survival curve for a small cohort", '
                '"sql": "SELECT time, survival, at_risk FROM survival.main.kaplan_meier('
                "(SELECT * FROM (VALUES (5,1),(8,0),(12,1),(3,1),(9,0)) AS c(t, e)), "
                "duration := 't', event := 'e') ORDER BY time\"}]"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT time, survival, at_risk FROM survival.main.kaplan_meier("
                    "(SELECT * FROM (VALUES (5,1),(8,0),(12,1),(3,1),(9,0)) AS c(t, e)), "
                    "duration := 't', event := 'e') ORDER BY time"
                ),
                description="Kaplan-Meier survival curve projected to the survival probability and at-risk count",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[KaplanMeierArgs]) -> BindResponse:
        """Declare the output schema at bind time.

        Args:
            params: The bind invocation parameters.

        Returns:
            The bind response carrying the output schema.
        """
        return BindResponse(output_schema=_KM_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[KaplanMeierArgs]
    ) -> DrainState:
        """Create the per-finalize-stream cursor.

        Args:
            finalize_state_id: The finalize stream's state id.
            params: The table-buffering invocation parameters.

        Returns:
            A fresh ``DrainState`` cursor (result IPC bytes + offset) for the finalize stream.
        """
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[KaplanMeierArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Compute the estimate once, then stream it in bounded ROWS_PER_TICK slices.

        Args:
            params: The table-buffering invocation parameters.
            finalize_state_id: The finalize stream's state id.
            state: The per-stream cursor (result IPC bytes + offset).
            out: The output collector for result batches.
        """
        a = params.args
        cls.drain_finalize(
            params,
            state,
            out,
            lambda df: survival.kaplan_meier(df, duration=a.duration, event=a.event),
        )


class CoxHazardRatios(SinkBuffer[CoxArgs, DrainState]):
    """Cox proportional-hazards fit: one hazard ratio per covariate column."""

    FunctionArguments: ClassVar[type] = CoxArgs

    class Meta:
        """Catalog metadata for the ``cox_hazard_ratios`` function."""

        name = "cox_hazard_ratios"
        description = (
            "Cox proportional-hazards model. Every column besides duration/event is a "
            "covariate; emits (covariate, coef, hazard_ratio, ci_lower, ci_upper, p_value)."
        )
        categories = ["survival", "regression"]
        tags = {
            "vgi.category": "regression",
            "vgi.title": "Cox Proportional-Hazards Ratios",
            "vgi.keywords": _json_keywords(
                "cox",
                "cox regression",
                "proportional hazards",
                "hazard ratio",
                "coxph",
                "coefficient",
                "covariate",
                "risk factor",
                "multivariate",
                "p-value",
                "regression",
                "time-to-event",
            ),
            "vgi.doc_llm": (
                "Fit a Cox proportional-hazards regression to a cohort relation and report one hazard "
                "ratio per covariate. Pass the cohort as `(SELECT ...)` containing `duration`, `event` "
                "(0/1; 1 = event occurred), and one or more numeric covariate columns; every column "
                "that is not the duration or event is treated as a covariate. Returns one row per "
                "covariate with the fitted log-hazard coefficient, the hazard ratio exp(coef) (>1 "
                "raises risk, <1 lowers it), a 95% confidence interval, and the Wald p-value. Use it to "
                "quantify and rank how each factor influences the rate of the event. Select exactly the "
                "covariate columns you want — do not feed id/string columns."
            ),
            "vgi.doc_md": (
                "# cox_hazard_ratios\n\n"
                "Fit a **Cox proportional-hazards** model and return per-covariate hazard ratios.\n\n"
                "## Usage\n\n"
                "Pass your cohort as the first argument — a parenthesised subquery projecting the "
                "follow-up time, the 0/1 event column, and one or more numeric covariate columns — "
                "and name the `duration := 't'` and `event := 'e'` roles. Project only the covariate "
                "columns you want fitted. A ready-to-run example is attached to this function.\n\n"
                "## Returns\n\n"
                "One row per covariate: the `coef`, the `hazard_ratio` (`exp(coef)`), its "
                "`ci_lower`/`ci_upper` band, and the Wald `p_value`.\n\n"
                "## Notes\n\n"
                "Every column besides `duration`/`event` becomes a covariate, so `SELECT` only the "
                "columns you want fitted (non-numeric covariates raise an error). A relation with no "
                "covariate column raises a clear error."
            ),
            "vgi.result_columns_schema": json.dumps(
                [
                    {"name": "covariate", "type": "VARCHAR", "description": "Covariate (input column) name."},
                    {"name": "coef", "type": "DOUBLE", "description": "Fitted log-hazard coefficient (beta)."},
                    {
                        "name": "hazard_ratio",
                        "type": "DOUBLE",
                        "description": "Hazard ratio exp(beta); >1 raises hazard, <1 lowers it.",
                    },
                    {
                        "name": "ci_lower",
                        "type": "DOUBLE",
                        "description": "Lower bound of the hazard-ratio confidence interval.",
                    },
                    {
                        "name": "ci_upper",
                        "type": "DOUBLE",
                        "description": "Upper bound of the hazard-ratio confidence interval.",
                    },
                    {"name": "p_value", "type": "DOUBLE", "description": "Wald-test p-value for the coefficient."},
                ]
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT covariate, hazard_ratio, p_value FROM survival.main.cox_hazard_ratios("
                    "(SELECT * FROM (VALUES (5,1,1.0),(8,0,0.0),(12,1,2.0),(3,1,3.0),(9,0,0.0),"
                    "(6,1,1.0)) AS c(t, e, prio)), duration := 't', event := 'e') ORDER BY covariate"
                ),
                description="Cox hazard ratio and Wald p-value for each covariate",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[CoxArgs]) -> BindResponse:
        """Declare the output schema at bind time.

        Args:
            params: The bind invocation parameters.

        Returns:
            The bind response carrying the output schema.
        """
        return BindResponse(output_schema=_COX_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[CoxArgs]) -> DrainState:
        """Create the per-finalize-stream cursor.

        Args:
            finalize_state_id: The finalize stream's state id.
            params: The table-buffering invocation parameters.

        Returns:
            A fresh ``DrainState`` cursor (result IPC bytes + offset) for the finalize stream.
        """
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[CoxArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Compute the estimate once, then stream it in bounded ROWS_PER_TICK slices.

        Args:
            params: The table-buffering invocation parameters.
            finalize_state_id: The finalize stream's state id.
            state: The per-stream cursor (result IPC bytes + offset).
            out: The output collector for result batches.
        """
        a = params.args
        cls.drain_finalize(
            params,
            state,
            out,
            lambda df: survival.cox_hazard_ratios(df, duration=a.duration, event=a.event),
        )


class LogRankTest(SinkBuffer[LogRankArgs, DrainState]):
    """Multivariate log-rank test comparing survival across a group column."""

    FunctionArguments: ClassVar[type] = LogRankArgs

    class Meta:
        """Catalog metadata for the ``logrank_test`` function."""

        name = "logrank_test"
        description = (
            "Log-rank test comparing survival across the group column; emits one row "
            "(test_statistic, p_value, degrees_freedom)."
        )
        categories = ["survival", "test"]
        tags = {
            "vgi.category": "comparison",
            "vgi.title": "Log-Rank Test Across Groups",
            "vgi.keywords": _json_keywords(
                "log-rank",
                "logrank",
                "log rank test",
                "survival comparison",
                "chi-squared",
                "p-value",
                "treatment arms",
                "groups",
                "hypothesis test",
                "multivariate",
                "time-to-event",
            ),
            "vgi.doc_llm": (
                "Run the multivariate log-rank test to ask whether survival differs across two or more "
                "groups. Pass the cohort as `(SELECT ...)` with `duration`, `event` (0/1), and a "
                "`group` column, and name those roles. Returns a single row: the chi-squared test "
                "statistic, the p-value, and the degrees of freedom (number of groups minus one). A "
                "small p-value (e.g. < 0.05) indicates the survival curves of the groups differ. Use "
                "it to compare treatment arms, segments, or cohorts. Note the `group` argument collides "
                "with the SQL `GROUP` keyword, so quote it at the call site as `\"group\" := 'arm'`."
            ),
            "vgi.doc_md": (
                "# logrank_test\n\n"
                "Multivariate **log-rank test** comparing survival across a grouping column.\n\n"
                "## Usage\n\n"
                "Pass your cohort as the first argument — a parenthesised subquery projecting the "
                "follow-up time, the 0/1 event column, and the grouping column — and name the "
                "`duration := 't'`, `event := 'e'`, and `\"group\" := 'arm'` roles. Because `group` "
                "is a SQL keyword it must be double-quoted at the call site. A ready-to-run example "
                "is attached to this function.\n\n"
                "## Returns\n\n"
                "Exactly one row: `test_statistic`, `p_value`, and `degrees_freedom`. A small "
                "`p_value` means the groups' survival curves differ.\n\n"
                "## Notes\n\n"
                "Needs at least two distinct groups. `group` is a SQL keyword, so double-quote it at "
                "the call site (`\"group\" := 'arm'`). Event coding is `1` = event, `0` = censored."
            ),
            "vgi.result_columns_schema": json.dumps(
                [
                    {
                        "name": "test_statistic",
                        "type": "DOUBLE",
                        "description": "Log-rank chi-squared test statistic.",
                    },
                    {"name": "p_value", "type": "DOUBLE", "description": "p-value of the log-rank test."},
                    {
                        "name": "degrees_freedom",
                        "type": "INTEGER",
                        "description": "Degrees of freedom (number of groups minus one).",
                    },
                ]
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT test_statistic, p_value, degrees_freedom FROM survival.main.logrank_test("
                    "(SELECT * FROM (VALUES (5,1,'a'),(8,0,'a'),(12,1,'b'),(3,1,'b'),(9,0,'a'),"
                    "(6,1,'b')) AS c(t, e, arm)), "
                    "duration := 't', event := 'e', \"group\" := 'arm')"
                ),
                description="Log-rank statistic and p-value comparing two treatment arms",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[LogRankArgs]) -> BindResponse:
        """Declare the output schema at bind time.

        Args:
            params: The bind invocation parameters.

        Returns:
            The bind response carrying the output schema.
        """
        return BindResponse(output_schema=_LOGRANK_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[LogRankArgs]) -> DrainState:
        """Create the per-finalize-stream cursor.

        Args:
            finalize_state_id: The finalize stream's state id.
            params: The table-buffering invocation parameters.

        Returns:
            A fresh ``DrainState`` cursor (result IPC bytes + offset) for the finalize stream.
        """
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[LogRankArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Compute the estimate once, then stream it in bounded ROWS_PER_TICK slices.

        Args:
            params: The table-buffering invocation parameters.
            finalize_state_id: The finalize stream's state id.
            state: The per-stream cursor (result IPC bytes + offset).
            out: The output collector for result batches.
        """
        a = params.args
        cls.drain_finalize(
            params,
            state,
            out,
            lambda df: survival.logrank_test(df, duration=a.duration, event=a.event, group=a.group),
        )


class MedianSurvival(SinkBuffer[MedianArgs, DrainState]):
    """Median Kaplan-Meier survival time as a one-row table function."""

    FunctionArguments: ClassVar[type] = MedianArgs

    class Meta:
        """Catalog metadata for the ``median_survival`` function."""

        name = "median_survival"
        description = (
            "Median survival time (Kaplan-Meier S(t)=0.5) as a single row; inf if the curve never reaches 0.5."
        )
        categories = ["survival", "estimator"]
        tags = {
            "vgi.category": "estimation",
            "vgi.title": "Median Survival Time",
            "vgi.keywords": _json_keywords(
                "median survival",
                "median survival time",
                "median",
                "kaplan-meier",
                "S(t)=0.5",
                "half life",
                "headline metric",
                "time-to-event",
                "censoring",
            ),
            "vgi.doc_llm": (
                "Compute the median survival time from a cohort relation: the time at which the "
                "Kaplan-Meier survival function S(t) first drops to 0.5. Pass the cohort as "
                "`(SELECT ...)` and name the `duration` and `event` (0/1; 1 = event occurred) columns. "
                "Returns a single row with one `median_survival` value. When the survival curve never "
                "reaches 0.5 (e.g. under heavy censoring) the result is `inf`, not NULL. Use it when you "
                "want a single headline number summarizing a cohort's survival rather than the whole "
                "curve."
            ),
            "vgi.doc_md": (
                "# median_survival\n\n"
                "The **median survival time**: where the Kaplan-Meier curve S(t) first reaches 0.5.\n\n"
                "## Usage\n\n"
                "Pass your cohort as the first argument — a parenthesised subquery projecting the "
                "follow-up time and 0/1 event columns — and name the `duration := 't'` and "
                "`event := 'e'` roles. The result is a single row. A ready-to-run example is "
                "attached to this function.\n\n"
                "## Returns\n\n"
                "Exactly one row with the single `median_survival` value.\n\n"
                "## Notes\n\n"
                "If S(t) never reaches 0.5 the value is `inf` (median undefined), passed through as a "
                "float rather than NULL. Event coding is `1` = event, `0` = right-censored."
            ),
            "vgi.result_columns_schema": json.dumps(
                [
                    {
                        "name": "median_survival",
                        "type": "DOUBLE",
                        "description": "Median survival time (where S(t)=0.5); inf if never reached.",
                    },
                ]
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT median_survival FROM survival.main.median_survival("
                    "(SELECT * FROM (VALUES (5,1),(8,0),(12,1),(3,1),(9,0)) AS c(t, e)), "
                    "duration := 't', event := 'e')"
                ),
                description="Median survival time for a small cohort",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[MedianArgs]) -> BindResponse:
        """Declare the output schema at bind time.

        Args:
            params: The bind invocation parameters.

        Returns:
            The bind response carrying the output schema.
        """
        return BindResponse(output_schema=_MEDIAN_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[MedianArgs]) -> DrainState:
        """Create the per-finalize-stream cursor.

        Args:
            finalize_state_id: The finalize stream's state id.
            params: The table-buffering invocation parameters.

        Returns:
            A fresh ``DrainState`` cursor (result IPC bytes + offset) for the finalize stream.
        """
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[MedianArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Compute the estimate once, then stream it in bounded ROWS_PER_TICK slices.

        Args:
            params: The table-buffering invocation parameters.
            finalize_state_id: The finalize stream's state id.
            state: The per-stream cursor (result IPC bytes + offset).
            out: The output collector for result batches.
        """
        a = params.args
        cls.drain_finalize(
            params,
            state,
            out,
            lambda df: survival.median_survival(df, duration=a.duration, event=a.event),
        )


TABLE_FUNCTIONS: list[type] = [KaplanMeier, CoxHazardRatios, LogRankTest, MedianSurvival]
