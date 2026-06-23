# vgi-survival

A [VGI](https://query.farm) worker that brings **survival / time-to-event
analysis** to DuckDB/SQL: Kaplan-Meier survival curves, Cox
proportional-hazards regression, the log-rank test, and median survival —
backed by [lifelines](https://lifelines.readthedocs.io/) (MIT).

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'survival' (TYPE vgi, LOCATION 'uv run survival_worker.py');

-- Kaplan-Meier survival curve
SELECT * FROM survival.kaplan_meier((SELECT t, e FROM cohort),
                                    duration := 't', event := 'e')
ORDER BY time;

-- Cox proportional-hazards: one hazard ratio per covariate column
SELECT * FROM survival.cox_hazard_ratios((SELECT * FROM cohort),
                                         duration := 't', event := 'e');

-- Log-rank test comparing arms ("group" is double-quoted: it's a SQL keyword)
SELECT * FROM survival.logrank_test((SELECT * FROM cohort),
                                    duration := 't', event := 'e', "group" := 'arm');

-- Median survival time
SELECT * FROM survival.median_survival((SELECT t, e FROM cohort),
                                       duration := 't', event := 'e');
```

## Data flow: one relation in, a result set out

Every function is a **table function** that consumes a *whole input relation* —
passed as a single `(SELECT ...)` subquery (the positional argument) — and emits
a result set. The roles of the columns inside that relation are passed as
**named string arguments**:

| named arg | meaning |
|-----------|---------|
| `duration := 'col'` | the time-to-event / follow-up column |
| `event := 'col'`    | the 0/1 event indicator |
| `"group" := 'col'`  | (log-rank only) the grouping column — `group` is a SQL keyword, so **double-quote the arg name** |

This mirrors how `vgi-scikit-learn` names `target` / `id`: the relation *is* the
data, and the named args just say which column plays which role. Because a
survival estimate needs **every row** before it can produce output, these are
buffering (Sink+Source) functions — they buffer all input batches, then run the
lifelines estimator once.

## Event-coding convention

The `event` column is the **event indicator**:

- `1` / `true` → the event of interest **occurred** (death / failure / churn) at
  `duration`.
- `0` / `false` → the observation was **right-censored** (still alive / still in
  study) at `duration`.

This matches lifelines' `event_observed` convention.

## Cox covariates

`cox_hazard_ratios` uses **every column other than `duration` and `event`** in
the input relation as a covariate, emitting one row per covariate. Select
exactly the covariate columns you want in the `(SELECT ...)`:

```sql
SELECT * FROM survival.cox_hazard_ratios(
  (SELECT week, arrest, age, prio, fin FROM rossi),  -- age/prio/fin are covariates
  duration := 'week', event := 'arrest');
```

`hazard_ratio` is `exp(coef)`: `> 1` raises the hazard, `< 1` lowers it.

## Functions

| function | returns |
|----------|---------|
| `kaplan_meier(rel, duration, event)` | `(time, survival, ci_lower, ci_upper, at_risk)` |
| `cox_hazard_ratios(rel, duration, event)` | `(covariate, coef, hazard_ratio, ci_lower, ci_upper, p_value)` |
| `logrank_test(rel, duration, event, group)` | `(test_statistic, p_value, degrees_freedom)` |
| `median_survival(rel, duration, event)` | `(median_survival)` — one row |

## Robustness

Missing columns, non-numeric durations, empty input, fewer than two log-rank
groups, or a Cox relation with no covariate columns all surface a **clear
error** rather than crashing the worker.

## Development

```sh
uv sync --extra dev
uv run --no-sync pytest -q                 # unit + in-proc tables + Client RPC E2E
make test-sql                              # haybarn-unittest SQL E2E (authoritative)
uv run --no-sync ruff check . && uv run --no-sync mypy vgi_survival/
```

## Licensing

This worker is MIT. lifelines is **MIT**; numpy and pandas are **BSD** — all
permissive, no copyleft obligations.
