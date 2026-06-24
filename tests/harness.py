"""In-process driver for the survival buffering (Sink+Source) functions.

Runs a ``TableBufferingFunction`` through its real bind -> init -> process(sink)
-> combine -> finalize lifecycle without spawning a worker process, so unit
tests stay fast and debuggable while still exercising the framework's argument
parsing, storage round-trip, and output schema.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from vgi.arguments import Arguments
from vgi.function_storage import BoundStorage, FunctionStorageSqlite
from vgi.invocation import FunctionType
from vgi.protocol import BindRequest, InitRequest
from vgi.table_buffering_function import TableBufferingParams


class _Collector:
    """Captures emitted batches from a finalize stream."""

    def __init__(self) -> None:
        self.batches: list[pa.RecordBatch] = []
        self.finished = False

    def emit(self, batch: pa.RecordBatch, *_a: Any, **_kw: Any) -> None:
        self.batches.append(batch)

    def finish(self) -> None:
        self.finished = True

    def client_log(self, *_a: Any, **_kw: Any) -> None:
        pass


def run_buffering(
    func_cls: type,
    table: pa.Table,
    *,
    named: dict[str, str] | None = None,
    serialize_state: bool = False,
    tick_guard: int = 10_000,
) -> pa.Table:
    """Drive a survival buffering function over a whole input ``table``.

    Args:
        func_cls: The ``TableBufferingFunction`` subclass to run.
        table: The input relation (the ``(SELECT ...)`` data) as an Arrow table.
        named: Named string column-role args (e.g. ``{"duration": "t"}``).
        serialize_state: When ``True``, re-serialize the finalize cursor between
            every ``finalize`` tick via
            ``type(state).deserialize_from_bytes(state.serialize_to_bytes())`` --
            emulating the HTTP continuation round-trip. A position-less
            (emit-all + ``done``) finalize re-emits from row 0 forever under this
            mode and overruns ``tick_guard``; the offset cursor advances and
            terminates.
        tick_guard: Maximum number of ``finalize`` ticks before raising
            ``AssertionError`` (guards against an infinite continuation loop).

    Returns:
        The emitted result as a single Arrow table (the function's output).

    Raises:
        AssertionError: If finalize does not terminate within ``tick_guard`` ticks.
    """
    input_schema = table.schema
    args = Arguments(
        positional=(),
        named={k: pa.scalar(v) for k, v in (named or {}).items()},
    )

    bind_req = BindRequest(
        function_name=func_cls.Meta.name,
        arguments=args,
        function_type=FunctionType.TABLE_BUFFERING,
        input_schema=input_schema,
    )
    bind_resp = func_cls.bind(bind_req)

    init_req = InitRequest(bind_call=bind_req, output_schema=bind_resp.output_schema)
    init_resp = func_cls.global_init(init_req)
    execution_id = init_resp.execution_id

    storage = BoundStorage(FunctionStorageSqlite(":memory:"), execution_id)
    parsed_args = func_cls._parse_arguments(func_cls.FunctionArguments, args)

    def make_params() -> TableBufferingParams:
        return TableBufferingParams(
            args=parsed_args,
            init_call=init_req,
            init_response=init_resp,
            output_schema=bind_resp.output_schema,
            settings={},
            secrets={},
            storage=storage,
            execution_id=execution_id,
            attach_id=b"",
            transaction_id=None,
            function_name=func_cls.Meta.name,
        )

    # Sink phase: one process() call per input batch.
    state_ids: list[bytes] = []
    for batch in table.to_batches():
        state_ids.append(func_cls.process(batch, make_params()))

    # Combine phase.
    finalize_ids = func_cls.combine(state_ids, make_params())

    # Source phase: drain each finalize stream.
    out = _Collector()
    for fid in finalize_ids:
        params = make_params()
        state = func_cls.initial_finalize_state(fid, params)
        ticks = 0
        while not out.finished:
            if serialize_state:
                # Emulate the stateless HTTP transport: the cursor is wire-
                # serialized between ticks and the worker resumes from the bytes.
                state = type(state).deserialize_from_bytes(state.serialize_to_bytes())
            func_cls.finalize(params, fid, state, out)
            ticks += 1
            assert ticks <= tick_guard, (
                f"finalize did not terminate within {tick_guard} ticks "
                f"(serialize_state={serialize_state}); the cursor is not advancing "
                f"across the continuation boundary"
            )

    return pa.Table.from_batches(out.batches, schema=bind_resp.output_schema)
