# Survival analysis SQL notebook

[Censoring, risk sets and the median](01_censoring_risk_sets_and_median.ipynb) uses invented device follow-up records to reconcile a Kaplan–Meier curve with SQL product-limit arithmetic. Shorter follow-up changes the observed events and can leave the median unreached.

## Run from this directory

Install [uv](https://docs.astral.sh/uv/) and Git. The notebook environment uses Python 3.13 or 3.14; the worker uses Python 3.13 explicitly. First use needs network access for dependencies, the published Haybarn/VGI extension and the pinned public Git worker. Analysis then uses only bundled synthetic inputs. A copy of this directory works without the worker checkout or a local build.

```bash
uv sync --frozen
uv run --frozen python execute.py
```

The visible SQL launches public revision `97c18e976e2d4783cb846c41269a42e8febdb869` using `uvx`, with VGI SDK 0.51.0 / RPC 0.53.1, lifelines 0.30.3, NumPy 2.5.4, pandas 2.3.3 and SciPy 1.18.1. These match the tested source lock. The separate notebook environment pins Haybarn 2.0.0rc11. Rerun default and experiment after changing any pin.

For interactive use, register this environment and select it in Jupyter:

```bash
uv run --frozen python -m ipykernel install --user --name vgi-survival-notebooks --display-name 'Python (survival notebooks)'
```

Restart and run all after editing `FOLLOWUP_DAYS`. Parameters, connection and analytical SQL remain visible; Python imports, plots, interpretation source and cleanup are folded. The runner creates a temporary kernel with this directory as cwd and saves only successful outputs. Overrides require a separate output directory; unknown parameters and source-directory destinations are rejected.

## Focused experiment

```bash
uv run --frozen python execute.py --parameters '{"FOLLOWUP_DAYS":8}' --output-dir validation/shorter-followup
```

The default cutoff is 16 days, with accepted integer values 1–22. SQL applies administrative censoring: observed duration is the lesser of recorded duration and cutoff; event is 1 only for a recorded failure at or before the cutoff. Thus a later failure is unobserved at the shorter cutoff, not counted early. Earlier right-censoring remains right-censoring. Compare the two runs only through day 8; there is no estimated short-run curve after its last observation. The median is `inf` if the curve never reaches 0.5; this means unreached within follow-up, not infinite device lifetime. Do not substitute zero or the cutoff.

## Synthetic input and estimator boundary

All 24 identifiers, failure times and censoring times are invented. [Provenance](fixtures/provenance.json) records the exact lists, event coding, units and CSV hash. No devices, patients or calendar timestamps were observed. The fixture deliberately contains ties and is not a random population sample. Regenerate its exact bytes with:

```bash
uv run --frozen python generate_fixture.py
```

All devices enter at synthetic time zero. Event coding is **1 = observed failure**, **0 = right-censored**. The whole projected duration/event relation goes to the real worker through VGI; identifiers and recorded outcomes beyond the selected cutoff never enter the estimator. Worker calls are materialized before local display queries. No Python survival fitting occurs.

Visible SQL independently constructs risk sets, tied event/censor counts and the product of `(1 - failures/at_risk)`. Events and censoring at the same time belong to that starting risk set; censoring affects later denominators. Zero factors are handled explicitly as absorbing zero survival. The dashed original-denominator fraction is an intentionally inadequate comparator under censoring, not a substitute estimator.

The two figures show the worker survival estimate with its default 95% pointwise confidence bounds, then the at-risk counts and exits supporting each step. Generalizing Kaplan–Meier requires assumptions such as independent censoring; this hand-authored fixture cannot establish them. Confidence bounds are illustrated estimator output, not calibrated evidence about actual devices. There is no left truncation, competing-risk model, causal comparison or clinical guidance.

Twelve SQL checks cover identities, event coding, cutoff behavior, exit accounting, risk sets, product-limit agreement, bounds, monotonicity and finite/unreached median semantics, plus fixture hash and cohort-size checks. The assertions do not require a preferred estimate or a fixed median. Validation also executes the 22-day boundary, where the last risk set is exhausted and the product-limit curve reaches zero without a logarithm-of-zero error. Reference: [lifelines KaplanMeierFitter](https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html).

Saved default outputs are embedded. Logs, chart images and experiments are ignored under `validation/`; experiment copies include relative fixtures, branding, helpers and environment files. [STANDARD.md](STANDARD.md) records presentation conventions.

Source checks passed: 27 pytest tests, 39 stdio SQL assertions, 41 HTTP assertions and 39 Unix assertions; Ruff/format, mypy and isolated pydoclint. Metadata is 100/100 L2 with zero findings. The SQL suite used the published Haybarn unittest runner; no engine rebuild, worker source change or root dependency-lock edit was needed.
