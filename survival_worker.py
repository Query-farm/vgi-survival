# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.8.4",
#     "lifelines>=0.27",
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
# ///
"""Stdio entry shim for the survival VGI worker.

Lets the worker run straight from a source checkout (``uv run
survival_worker.py``) and keeps ``import survival_worker`` working for tests.
The implementation lives in ``vgi_survival.worker``; installed users invoke the
``vgi-survival`` console script (which points at ``vgi_survival.worker:main``).

    ATTACH 'survival' (TYPE vgi, LOCATION 'uv run survival_worker.py');
    SELECT * FROM survival.kaplan_meier((SELECT t, e FROM cohort), duration := 't', event := 'e');
"""

from vgi_survival.worker import SurvivalWorker, main

__all__ = ["SurvivalWorker", "main"]

if __name__ == "__main__":
    main()
