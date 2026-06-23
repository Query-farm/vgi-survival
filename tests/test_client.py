"""End-to-end tests driving survival_worker.py as a real subprocess.

These spawn the worker via ``vgi.client.Client`` and invoke each function
through the real ``table_buffering_function`` RPC path -- exactly how DuckDB
drives a buffering function after ``ATTACH`` -- exercising bind, the sink
process RPC per batch, combine, and the finalize source stream over the wire.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pytest
from lifelines.datasets import load_rossi, load_waltons
from vgi import Arguments
from vgi.client import Client, ClientError

_WORKER = str(Path(__file__).resolve().parent.parent / "survival_worker.py")


@pytest.fixture(scope="module")
def client() -> Iterator[Client]:
    with Client(f"{sys.executable} {_WORKER}", worker_limit=1) as c:
        yield c


def _run(client: Client, name: str, table: pa.Table, **named: str) -> pa.Table:
    batches = list(
        client.table_buffering_function(
            function_name=name,
            input=iter(table.to_batches()),
            arguments=Arguments(named={k: pa.scalar(v) for k, v in named.items()}),
        )
    )
    return pa.Table.from_batches(batches)


def test_kaplan_meier_e2e(client: Client) -> None:
    df = load_waltons()
    tbl = pa.Table.from_pandas(df[["T", "E"]], preserve_index=False)
    out = _run(client, "kaplan_meier", tbl, duration="T", event="E")
    d = out.to_pydict()
    assert d["survival"][0] == pytest.approx(1.0)
    assert all(b <= a + 1e-9 for a, b in zip(d["survival"], d["survival"][1:], strict=False))


def test_cox_e2e(client: Client) -> None:
    df = load_rossi()
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    out = _run(client, "cox_hazard_ratios", tbl, duration="week", event="arrest")
    d = out.to_pydict()
    i = d["covariate"].index("prio")
    assert d["hazard_ratio"][i] > 1.0
    assert d["p_value"][i] < 0.01


def test_logrank_e2e(client: Client) -> None:
    df = load_waltons()
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    out = _run(client, "logrank_test", tbl, duration="T", event="E", group="group")
    assert out.to_pydict()["p_value"][0] < 0.001


def test_median_e2e(client: Client) -> None:
    df = load_waltons()
    tbl = pa.Table.from_pandas(df[["T", "E"]], preserve_index=False)
    out = _run(client, "median_survival", tbl, duration="T", event="E")
    assert out.to_pydict()["median_survival"][0] > 0


def test_missing_column_errors_e2e(client: Client) -> None:
    tbl = pa.table({"t": [1.0, 2.0, 3.0], "e": [1, 0, 1]})
    with pytest.raises(ClientError):
        _run(client, "kaplan_meier", tbl, duration="nope", event="e")
