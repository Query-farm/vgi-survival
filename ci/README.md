# CI: the vgi-survival worker integration suite

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs the unit tests
and this repo's sqllogictest suite (`test/sql/*.test`) against the vgi-survival
VGI worker through the **real DuckDB `vgi` extension** on every push / PR.

## How it works (no C++ build)

CI drives a **prebuilt** standalone `haybarn-unittest` and installs the
**signed** `vgi` extension from the Haybarn community channel:

1. **Install the worker** — `uv sync --frozen` installs the package and its
   `vgi-survival` console script into the venv; the .test files ATTACH it via
   `.venv/bin/vgi-survival`.
2. **Download the runner** — the matching `haybarn_unittest-*` asset per platform.
3. **Preprocess** — [`preprocess-require.awk`](preprocess-require.awk) injects a
   signed `INSTALL vgi FROM community;` before each bare `LOAD vgi;`.
4. **Run** — [`run-integration.sh`](run-integration.sh) stages the preprocessed
   tree, points `VGI_SURVIVAL_WORKER` at the console script, warms the extension
   cache, then runs the suite.

## Run it locally

```bash
uv sync --python 3.13
HAYBARN_UNITTEST=/path/to/haybarn-unittest \
VGI_SURVIVAL_WORKER="$PWD/.venv/bin/vgi-survival" \
  ci/run-integration.sh
```
