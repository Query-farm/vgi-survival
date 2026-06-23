"""Put the repo root on ``sys.path`` so tests can import the worker modules.

The mere presence of this file makes pytest add the repo root to ``sys.path``,
which lets the test suite ``import survival_worker`` and ``import vgi_survival``.
"""
