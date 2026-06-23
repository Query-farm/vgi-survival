"""Survival / time-to-event analysis as a VGI worker for DuckDB/SQL.

The implementation is split so each concern stays focused:

- ``survival``    -- pure lifelines logic (Kaplan-Meier, Cox PH, log-rank,
  median survival) over ``pandas`` frames; no Arrow or VGI dependency, directly
  unit-testable.
- ``buffering``   -- the single-bucket Sink+Source plumbing every function
  shares (buffer all input batches, then estimate once).
- ``tables``      -- the VGI ``TableBufferingFunction`` wrappers: relation in
  via ``(SELECT ...)`` (``Arg(0)``), column roles as named string args.

``survival_worker.py`` at the repo root assembles these into the ``survival``
catalog and runs the worker over stdio (or HTTP).
"""

from __future__ import annotations

__version__ = "0.1.0"
