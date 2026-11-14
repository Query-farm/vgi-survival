<p align="center">
  <img src="https://query.farm/media-kit/products/vgi/vgi-logo.png" alt="Vector Gateway Interface (VGI)" width="320">
</p>

<p align="center"><em>A <a href="https://query.farm">Query.Farm</a> VGI worker for DuckDB.</em></p>

# Survival Analysis — Kaplan-Meier, Cox PH & Log-Rank in DuckDB

> **vgi-survival** · a [Query.Farm](https://query.farm) VGI worker · powered by lifelines

A [VGI](https://query.farm) worker that brings **survival / time-to-event
analysis** to DuckDB/SQL: Kaplan-Meier survival curves, Cox
proportional-hazards regression, the log-rank test, and median survival —
backed by [lifelines](https://lifelines.readthedocs.io/) (MIT).

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'survival' (TYPE vgi, LOCATION 'uvx --from git+https://github.com/Query-farm/vgi-survival@v0.4.0 vgi-survival-worker');

-- cohort(t, e, arm, age): follow-up time, event (1 = event, 0 = censored),
-- treatment arm ('control' / 'treated'), and a numeric covariate

-- Kaplan-Meier survival curve
SELECT * FROM survival.kaplan_meier((SELECT t, e FROM cohort),
                                    duration := 't', event := 'e')
ORDER BY time;

-- Cox proportional-hazards: one hazard ratio per covariate column
-- (every column besides duration/event is a covariate, so select only numeric ones)
SELECT * FROM survival.cox_hazard_ratios((SELECT t, e, age FROM cohort),
                                         duration := 't', event := 'e');

-- Log-rank test comparing arms ("group" is double-quoted: it's a SQL keyword)
SELECT * FROM survival.logrank_test((SELECT t, e, arm FROM cohort),
                                    duration := 't', event := 'e', "group" := 'arm');

-- Median survival time
SELECT * FROM survival.median_survival((SELECT t, e FROM cohort),
                                       duration := 't', event := 'e');

-- Combine them: each arm's median survival next to the log-rank p-value
SELECT mc.median_survival AS median_control, mt.median_survival AS median_treated,
       lr.p_value AS logrank_p_value
FROM survival.median_survival((SELECT t, e FROM cohort WHERE arm = 'control'),
                              duration := 't', event := 'e') AS mc
CROSS JOIN survival.median_survival((SELECT t, e FROM cohort WHERE arm = 'treated'),
                                    duration := 't', event := 'e') AS mt
CROSS JOIN survival.logrank_test((SELECT t, e, arm FROM cohort),
                                 duration := 't', event := 'e', "group" := 'arm') AS lr;
```

## Quick start

The worker isn't on PyPI; [uv](https://docs.astral.sh/uv/) runs it straight from a tagged release on GitHub. `uvx` builds an isolated environment on the first ATTACH and reuses it afterwards:

```sql
INSTALL vgi FROM community;
LOAD vgi;

ATTACH 'survival' AS survival (TYPE vgi, LOCATION 'uvx --from git+https://github.com/Query-farm/vgi-survival@v0.4.0 vgi-survival-worker');

-- Kaplan-Meier survival curve for a small cohort, projected and ordered by time
SELECT time, survival, at_risk FROM survival.main.kaplan_meier((SELECT * FROM (VALUES (5,1),(8,0),(12,1),(3,1),(9,0)) AS c(t, e)), duration := 't', event := 'e') ORDER BY time;

-- Cox hazard ratio and Wald p-value for a single covariate
SELECT covariate, hazard_ratio, p_value FROM survival.main.cox_hazard_ratios((SELECT * FROM (VALUES (5,1,1.0),(8,0,0.0),(12,1,2.0),(3,1,3.0),(9,0,0.0),(6,1,1.0)) AS c(t, e, prio)), duration := 't', event := 'e') ORDER BY covariate;
```

Change `@v0.4.0` to move to a newer release.

### Other ways to run it

- **Docker** (the image bundles the worker and its dependencies):

  ```sql
  ATTACH 'survival' AS survival (TYPE vgi, LOCATION 'docker://ghcr.io/query-farm/vgi-survival:0.2.0');
  ```

- **From a local checkout** (`git clone https://github.com/Query-farm/vgi-survival`), point `LOCATION` at `uv run --project /path/to/vgi-survival vgi-survival-worker`.

### Build from source

To build and test the worker yourself from a checkout, see [Development](#development).

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

---

## Authorship & License

Written by [Query.Farm](https://query.farm).

Copyright 2026 Query Farm LLC - https://query.farm


## Executed notebook

[Censoring, risk sets and the median](notebooks/README.md) — an executable SQL investigation using synthetic follow-up records, Kaplan–Meier estimates and a shorter observation-window experiment.
