"""Table-function tests via the in-process buffering harness.

Drive each survival function through the real bind -> process(sink) -> combine
-> finalize lifecycle (no subprocess), checking the emitted Arrow result and
that the named column-role args resolve columns in the input relation.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest
from lifelines.datasets import load_rossi, load_waltons

import vgi_survival.buffering as buffering
from vgi_survival.tables import CoxHazardRatios, KaplanMeier, LogRankTest, MedianSurvival

from .harness import run_buffering


def _arrow(df) -> pa.Table:
    return pa.Table.from_pandas(df, preserve_index=False)


def test_kaplan_meier_function() -> None:
    df = load_waltons()
    out = run_buffering(KaplanMeier, _arrow(df[["T", "E"]]), named={"duration": "T", "event": "E"})
    d = out.to_pydict()
    assert d["survival"][0] == pytest.approx(1.0)
    assert all(b <= a + 1e-12 for a, b in zip(d["survival"], d["survival"][1:], strict=False))
    assert out.schema.names == ["time", "survival", "ci_lower", "ci_upper", "at_risk"]
    assert pa.types.is_int64(out.schema.field("at_risk").type)


def test_cox_function_recovers_prio() -> None:
    df = load_rossi()
    out = run_buffering(CoxHazardRatios, _arrow(df), named={"duration": "week", "event": "arrest"})
    d = out.to_pydict()
    i = d["covariate"].index("prio")
    assert d["hazard_ratio"][i] > 1.0
    assert d["p_value"][i] < 0.01


def test_logrank_function() -> None:
    df = load_waltons()
    out = run_buffering(LogRankTest, _arrow(df), named={"duration": "T", "event": "E", "group": "group"})
    d = out.to_pydict()
    assert d["degrees_freedom"][0] == 1
    assert d["p_value"][0] < 0.001


def test_median_function() -> None:
    df = load_waltons()
    out = run_buffering(MedianSurvival, _arrow(df[["T", "E"]]), named={"duration": "T", "event": "E"})
    assert out.num_rows == 1
    assert out.to_pydict()["median_survival"][0] > 0


def test_missing_column_raises() -> None:
    tbl = pa.table({"t": [1.0, 2.0], "e": [1, 0]})
    with pytest.raises(Exception, match="missing required column"):
        run_buffering(KaplanMeier, tbl, named={"duration": "nope", "event": "e"})


def test_median_all_censored_inf() -> None:
    tbl = pa.table({"t": [1.0, 2.0, 3.0], "e": [0, 0, 0]})
    out = run_buffering(MedianSurvival, tbl, named={"duration": "t", "event": "e"})
    assert math.isinf(out.to_pydict()["median_survival"][0])


class TestCursorSurvivesContinuation:
    """Regression guard for the HTTP-continuation cursor.

    KaplanMeier emits one row per distinct follow-up time, so its result is
    genuinely unbounded. Over the stateless HTTP transport the finalize cursor is
    wire-serialized between ticks; the old emit-all + ``done`` finalize restarted
    from row 0 on every resume and looped forever. The offset cursor advances
    across the continuation boundary, so the serialize-between-ticks run produces
    identical, duplicate-free rows and terminates.

    These tests build a cohort whose KM curve has many more than ``ROWS_PER_TICK``
    distinct times (and shrink ``ROWS_PER_TICK`` so the cursor crosses several
    continuation boundaries). On the OLD code the serialized run overruns the
    harness tick guard (``AssertionError``); on the cursor code it passes.
    """

    @staticmethod
    def _big_cohort() -> pa.Table:
        # 500 distinct event times -> a KM curve of ~500 rows, far above any small
        # ROWS_PER_TICK, so the cursor must page across many continuation ticks.
        n = 500
        times = [float(i + 1) for i in range(n)]
        events = [1] * n
        return pa.table({"t": times, "e": events})

    def test_pages_match_single_shot_and_terminate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(buffering, "ROWS_PER_TICK", 8)
        tbl = self._big_cohort()

        single = run_buffering(KaplanMeier, tbl, named={"duration": "t", "event": "e"})
        paged = run_buffering(
            KaplanMeier,
            tbl,
            named={"duration": "t", "event": "e"},
            serialize_state=True,
        )

        # KM over 500 distinct event times yields well over ROWS_PER_TICK rows, so
        # the http leg genuinely pages.
        assert single.num_rows > 8

        # (1) identical content and order across the continuation boundary.
        assert paged.to_pydict() == single.to_pydict()

        # (2) no duplicate rows: every distinct time appears exactly once.
        times = paged.to_pydict()["time"]
        assert len(times) == len(set(times))

        # (3) termination is implied: run_buffering raises if it overruns the guard.

    def test_old_emit_all_would_overrun_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate the OLD position-less finalize (emit-all, never advance) and
        # confirm the serialize-between-ticks harness flags it as non-terminating.
        monkeypatch.setattr(buffering, "ROWS_PER_TICK", 8)

        class _StuckKaplanMeier(KaplanMeier):
            @classmethod
            def drain_finalize(cls, params, state, out, compute):  # type: ignore[override]
                # Old anti-pattern: emit ALL rows every tick, never advancing a
                # cursor, so a resumed (re-serialized) tick restarts from row 0.
                df = cls.buffered_frame(params)
                result = compute(df)
                out.emit(pa.RecordBatch.from_pydict(result, schema=params.output_schema))

        with pytest.raises(AssertionError, match="did not terminate"):
            run_buffering(
                _StuckKaplanMeier,
                self._big_cohort(),
                named={"duration": "t", "event": "e"},
                serialize_state=True,
                tick_guard=200,
            )
