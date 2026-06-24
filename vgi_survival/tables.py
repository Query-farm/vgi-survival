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

    data: Annotated[TableInput, Arg(0, doc="Relation: duration, event, and one+ covariate columns.")]
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
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM survival.kaplan_meier((SELECT t, e FROM cohort), "
                    "duration := 't', event := 'e') ORDER BY time"
                ),
                description="Kaplan-Meier survival curve",
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
        examples = [
            FunctionExample(
                sql=("SELECT * FROM survival.cox_hazard_ratios((SELECT * FROM cohort), duration := 't', event := 'e')"),
                description="Cox hazard ratios for every covariate column",
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
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM survival.logrank_test((SELECT * FROM cohort), "
                    "duration := 't', event := 'e', group := 'arm')"
                ),
                description="Log-rank test across treatment arms",
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
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM survival.median_survival((SELECT t, e FROM cohort), duration := 't', event := 'e')"
                ),
                description="Median survival time",
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
