# CLAUDE.md — vgi-survival

Contributor/agent notes. User-facing docs live in `README.md`; this is the
"how it's built and where the sharp edges are" companion. Sibling style/tooling
to `vgi-conform` / `vgi-calendar` (structure) and `vgi-scikit-learn` (the
whole-relation buffering data-flow).

## What this is

A [VGI](https://query.farm) worker exposing **survival / time-to-event
analysis** to DuckDB/SQL via [lifelines](https://lifelines.readthedocs.io/)
(MIT): Kaplan-Meier curves, Cox proportional-hazards, log-rank tests, median
survival. `survival_worker.py` assembles every function into one `survival`
catalog (single `main` schema) over stdio.

## Layout

```
survival_worker.py     repo-root stdio entry shim; PEP 723 inline deps; main()
vgi_survival/
  survival.py          pure lifelines logic over pandas frames; no Arrow/VGI; unit-testable
  buffering.py         SinkBuffer (single-bucket sink/combine) + Arrow<->pandas plumbing
  tables.py            the four TableBufferingFunction wrappers + output schemas + arg classes
  schema_utils.py      pa.Field comment / column-doc helper
  worker.py            assembles the catalog; main() / main_http()
tests/                 pytest: test_survival (pure), test_tables (in-proc harness), test_client (Client RPC)
test/sql/*.test        haybarn-unittest sqllogictest — authoritative E2E
Makefile               test / test-unit / test-sql / lint
```

To add a function: implement the math in `survival.py` (pure, takes a pandas
frame + role kwargs, returns a `dict[str, list]`, raises `SurvivalError` on bad
input), add a `pa.schema` + `@dataclass` args class + a `SinkBuffer` subclass in
`tables.py`, append it to `TABLE_FUNCTIONS`.

## THE core convention (read first): one relation in, named role args

These are **table functions**, not scalars. Each takes the whole input relation
as a single `(SELECT ...)` subquery — `Arg(0)`, typed `TableInput` — and the
column **roles** as NAMED string args (`duration := 't'`, `event := 'e'`,
`"group" := 'arm'`). The relation's columns *are* the data; the named args just
name which column plays which role. This mirrors `vgi-sklearn`'s
`confusion_matrix(..., actual := 'y', predicted := 'yhat')`.

**`group` is a SQL keyword** — in SQL the log-rank arg name must be
double-quoted: `"group" := 'arm'`. The framework arg key itself is plain
`group` (matched to the Python attribute); only the SQL call site needs the
quoting. The in-proc / Client tests pass the bare name `group` (no SQL parser
involved); the `.test` file double-quotes it.

Survival estimates need **every row** before any output, so every function is a
`TableBufferingFunction` (Sink+Source), routed through the C++
`PhysicalVgiTableBuffering` operator:

- `process(batch)` — sink each input batch to execution-scoped `BoundStorage`.
- `combine(state_ids)` — collapse to a single finalize key (one bucket).
- `finalize(...)` — reassemble the full table (`buffered_frame()` → pandas),
  run the lifelines estimator once, emit one result batch, then `out.finish()`.

`SinkBuffer` in `buffering.py` implements `process`/`combine`/`buffered_frame`;
each function only writes `on_bind` (its output schema) + `finalize`. A
`DrainState(done: bool)` cursor makes finalize emit exactly once.

## Sharp edges (learned the hard way)

1. **`haybarn-unittest` silently SKIPS `require vgi`.** Under haybarn the
   extension isn't autoloaded for `require`, so a `.test` using `require vgi` is
   SKIPPED, not run. Use an explicit `statement ok` / `LOAD vgi;` (the `.test`
   here does). `LOAD vgi` also works under a locally-built vgi unittest.
2. **Buffering needs the input schema at bind.** The `(SELECT ...)` relation's
   schema arrives via `bind_call.input_schema`; `buffered_frame()` uses it to
   reassemble even when zero batches were sunk (so empty-input handling is
   uniform). `Client.table_buffering_function` peeks the first batch to learn
   that schema — an entirely empty input iterator yields a `None` schema, so the
   E2E tests always feed at least the typed columns.
3. **Event coding is `1`=event, `0`=censored.** `survival._event_indicator`
   accepts numeric or boolean and maps truthy→1. Get this backwards and every
   curve/HR inverts. Documented in `survival.py`, README, and each `Meta`.
4. **Cox uses ALL non-duration/event columns as covariates.** Don't `SELECT *`
   a relation with an id/string column into Cox — it'll try to fit it (or raise
   on non-numeric). Select exactly the covariate columns. A relation with *no*
   covariate column raises a clear `SurvivalError`.
5. **`median_survival` can be `inf`.** When the KM curve never reaches 0.5 (e.g.
   heavy censoring) lifelines returns `inf`; we pass it through as a float `inf`,
   not NULL. Tested with an all-censored cohort.
6. **The unit suite can pass while the RPC path is broken.** `test_survival.py`
   calls pure functions; only `test_tables.py` (in-proc bind→process→finalize),
   `test_client.py` (real `vgi.client.Client` subprocess), and `test/sql/*.test`
   exercise the framework/wire. **Run the SQL suite** — it's authoritative.

## Known-dataset validation

Tests validate against lifelines' bundled datasets:

- **`load_rossi`** (recidivism): Cox recovers `prio` (# prior convictions) as a
  significant positive risk factor — `hazard_ratio > 1`, `p < 0.01` — and the
  fit matches lifelines' own `CoxPHFitter` exactly (`exp(coef)` and `p`).
- **`load_waltons`** (two groups): KM curve starts at 1.0 and is monotone and
  matches `survival_function_`; the log-rank `p < 0.001` (groups clearly
  differ); median matches `median_survival_time_`.

## Licensing

lifelines is **MIT**; numpy and pandas are **BSD** — all permissive, no
copyleft. The worker's own code is MIT. No vendoring, no patched deps.

## Testing

```sh
uv sync --extra dev
uv run --no-sync pytest -q     # pure logic + in-proc tables + Client RPC E2E
make test-sql                  # haybarn-unittest over test/sql/*  (authoritative)
uv run --no-sync ruff check . && uv run --no-sync mypy vgi_survival/
```

`make test-sql` sets `VGI_SURVIVAL_WORKER="uv run --python 3.13
survival_worker.py"`, puts `~/.local/bin` on PATH, and runs `haybarn-unittest
--test-dir . "test/sql/*"`. Install the runner once with
`uv tool install haybarn-unittest`. Everything is offline/hermetic (lifelines
datasets ship with the package).
