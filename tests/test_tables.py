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
