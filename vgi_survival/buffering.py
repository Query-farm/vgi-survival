"""Shared plumbing for the table-buffering survival functions.

Every survival function (kaplan_meier, cox_hazard_ratios, logrank_test,
median_survival) must see the *whole* input relation before it can produce any
output: a survival curve, a fitted model, a test statistic. They are therefore
``TableBufferingFunction`` (Sink+Source) functions. The sink phase serializes
each input batch to execution-scoped storage; finalize reassembles the full
table and runs the estimator once.

This module holds the single-bucket sink/combine implementation (``SinkBuffer``)
plus the Arrow (de)serialization and a ``pandas`` assembly helper, so each
function only writes its ``finalize`` logic.

The Source phase is resumable: ``finalize`` carries a ``DrainState`` cursor
(computed result IPC bytes + an integer ``offset``) that wire-serializes through
the HTTP continuation token, so each tick emits a bounded ``ROWS_PER_TICK`` slice
and a resumed tick advances rather than restarting from row 0. See
``DrainState`` for the full HTTP-continuation rationale.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyarrow as pa
from vgi.table_buffering_function import TableBufferingFunction, TableBufferingParams
from vgi_rpc import ArrowSerializableDataclass
from vgi_rpc.rpc import OutputCollector

_DATA_KEY = b"input_batches"

# Rows emitted per finalize tick. Bounded so the cursor (offset) is observable
# across the HTTP limit-1 continuation boundary; correctness no longer depends on
# the whole result fitting in one producer batch.
ROWS_PER_TICK = 64


@dataclass(kw_only=True)
class DrainState(ArrowSerializableDataclass):
    """Externalized finalize cursor: result batch IPC bytes + next-row offset.

    WHY A CURSOR: over the stateless HTTP transport the framework round-trips
    this finalize state through a continuation token -- after each ``finalize``
    tick it wire-serializes the state, the client returns it, and the worker
    resumes by deserializing it, emitting at most the producer batch limit (1)
    per response. A position-less ``done: bool`` flag that emits ALL result rows
    in one ``out.emit()`` restarts from row 0 on every HTTP resume and loops
    forever once the result exceeds one producer batch (KaplanMeier emits one row
    per distinct follow-up time -- genuinely unbounded). subprocess/unix keep the
    live state in-process and so hide the bug; only http (and the
    serialize-between-ticks unit test) exposes it.

    Both fields wire-serialize through the continuation token, so a resumed tick
    sees the advanced ``offset`` and emits the next bounded slice (or finishes) --
    never re-runs the estimator and never re-emits from row 0.

    ``result_ipc`` is empty until the first tick computes the estimate; ``started``
    distinguishes "not yet computed" from "computed an empty result".
    """

    started: bool = False
    offset: int = 0
    result_ipc: bytes = b""


def result_to_ipc(batch: pa.RecordBatch) -> bytes:
    """Serialize a computed result batch to Arrow IPC stream bytes for the cursor.

    Args:
        batch: The materialized result record batch.

    Returns:
        The Arrow IPC stream bytes carried in ``DrainState.result_ipc``.
    """
    sink = pa.BufferOutputStream()
    # pyarrow.ipc.* is untyped (ships py.typed but no ipc stub).
    with pa.ipc.new_stream(sink, batch.schema) as writer:  # type: ignore[no-untyped-call]
        writer.write_batch(batch)
    result: bytes = sink.getvalue().to_pybytes()
    return result


def ipc_to_table(value: bytes) -> pa.Table:
    """Reassemble the cursor's result table from its IPC stream bytes.

    Args:
        value: Bytes produced by :func:`result_to_ipc`.

    Returns:
        The materialized result as an Arrow table.
    """
    # pyarrow.ipc.* is untyped (ships py.typed but no ipc stub).
    reader = pa.ipc.open_stream(pa.BufferReader(value))  # type: ignore[no-untyped-call]
    table: pa.Table = reader.read_all()
    return table


def serialize_batch(batch: pa.RecordBatch) -> bytes:
    """Serialize one RecordBatch to a self-describing Arrow IPC stream.

    Args:
        batch: The record batch to serialize.

    Returns:
        The Arrow IPC stream bytes for the batch.
    """
    sink = pa.BufferOutputStream()
    # pyarrow.ipc.* is untyped (ships py.typed but no ipc stub).
    with pa.ipc.new_stream(sink, batch.schema) as writer:  # type: ignore[no-untyped-call]
        writer.write_batch(batch)
    result: bytes = sink.getvalue().to_pybytes()
    return result


def deserialize_batches(value: bytes) -> list[pa.RecordBatch]:
    """Reassemble the record batches from one serialized blob.

    Args:
        value: Bytes produced by :func:`serialize_batch`.

    Returns:
        The record batches contained in the stream.
    """
    # pyarrow.ipc.* is untyped (ships py.typed but no ipc stub).
    reader = pa.ipc.open_stream(pa.BufferReader(value))  # type: ignore[no-untyped-call]
    batches: list[pa.RecordBatch] = reader.read_all().to_batches()
    return batches


def input_schema_of(params: Any) -> pa.Schema:
    """Input schema from a process/finalize params object."""
    schema = params.init_call.bind_call.input_schema
    assert schema is not None
    return schema


class SinkBuffer[TArgs, TState](TableBufferingFunction[TArgs, TState]):
    """Single-bucket sink/combine: buffer every input batch under one key.

    Subclasses implement ``on_bind``, ``initial_finalize_state``, and
    ``finalize`` (calling ``buffered_frame(params)`` to get the full input as a
    ``pandas.DataFrame``).
    """

    @classmethod
    def process(cls, batch: pa.RecordBatch, params: TableBufferingParams[TArgs]) -> bytes:
        """Sink one input batch under the single shared key.

        Args:
            batch: The input record batch to buffer.
            params: The table-buffering invocation parameters.

        Returns:
            The execution id used as this sink's state key.
        """
        if batch.num_rows:
            params.storage.state_append(_DATA_KEY, b"", serialize_batch(batch))
        return params.execution_id

    @classmethod
    def combine(cls, state_ids: list[bytes], params: TableBufferingParams[TArgs]) -> list[bytes]:
        """Collapse every sink state into the single finalize bucket.

        Args:
            state_ids: The per-sink state ids produced by :meth:`process`.
            params: The table-buffering invocation parameters.

        Returns:
            A single-element list with the one finalize key.
        """
        return [params.execution_id]

    @classmethod
    def drain_finalize(
        cls,
        params: TableBufferingParams[TArgs],
        state: DrainState,
        out: OutputCollector,
        compute: Callable[[pd.DataFrame], dict[str, list[Any]]],
    ) -> None:
        """Compute the result once, then stream it in bounded ``ROWS_PER_TICK`` slices.

        On the first tick (``not state.started``) this runs the estimator over the
        buffered cohort, materializes the result batch into ``state.result_ipc``,
        and resets ``state.offset`` to 0. Every tick then emits at most
        ``ROWS_PER_TICK`` rows from ``state.offset`` and advances it, calling
        ``out.finish()`` once the result is drained. Because ``state`` (result IPC
        bytes + offset) wire-serializes through the HTTP continuation token, a
        resumed tick sees the advanced offset and continues rather than restarting.

        Args:
            params: The table-buffering invocation parameters.
            state: The per-finalize-stream cursor (result IPC bytes + offset).
            out: The output collector for result batches.
            compute: Estimator producing a ``dict[str, list]`` from the cohort frame.
        """
        if not state.started:
            df = cls.buffered_frame(params)
            result = compute(df)
            batch = pa.RecordBatch.from_pydict(result, schema=params.output_schema)
            state.result_ipc = result_to_ipc(batch)
            state.started = True
            state.offset = 0

        table = ipc_to_table(state.result_ipc)
        total = table.num_rows
        if state.offset >= total:
            out.finish()
            return
        end = min(state.offset + ROWS_PER_TICK, total)
        chunk = table.slice(state.offset, end - state.offset)
        out.emit(chunk.combine_chunks().to_batches()[0])
        state.offset = end

    @classmethod
    def buffered_frame(cls, params: TableBufferingParams[TArgs]) -> pd.DataFrame:
        """Reassemble all sunk batches into a single pandas DataFrame.

        Returns an empty (zero-row) frame -- with the right column names -- when
        no rows were sunk, so finalize can apply uniform empty-input handling.
        """
        input_schema = input_schema_of(params)
        batches: list[pa.RecordBatch] = []
        for _sid, value in params.storage.state_log_scan(_DATA_KEY, b""):
            batches.extend(deserialize_batches(value))
        table = pa.Table.from_batches(batches, schema=input_schema)
        frame: pd.DataFrame = table.to_pandas()
        return frame
